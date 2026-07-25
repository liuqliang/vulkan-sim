#include "rtcore_v04_stack_timing_driver.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace stack_timing {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  if (bytes == NULL) return false;
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool config_valid(const config_v0 &config) {
  return config.stack_unit_count != 0 &&
         config.stack_unit_count <= kMaxStackUnits &&
         config.stack_latency != 0 &&
         config.stack_initiation_interval != 0 &&
         config.stack_issue_width != 0 &&
         config.stack_issue_width <= config.stack_unit_count &&
         bytes_are_zero(config.reserved_zero,
                        sizeof(config.reserved_zero));
}

bool make_request_owner(
    const private_frontier::owner_binding_v0 &private_owner,
    request_owner::lane_binding_v0 *request_binding) {
  if (request_binding == NULL || private_owner.request_identity == 0 ||
      private_owner.generation == 0 ||
      private_owner.generation > request_owner::kRequestGenerationMax ||
      private_owner.resident_warp_id >=
          request_owner::kResidentWarpCapacity ||
      private_owner.private_slot_id >
          std::numeric_limits<uint16_t>::max() ||
      private_owner.lane_id >= request_owner::kLaneCapacity ||
      !bytes_are_zero(private_owner.reserved_zero,
                      sizeof(private_owner.reserved_zero))) {
    return false;
  }
  request_owner::internal_request_key_fields_v0 fields = {};
  if (request_owner::unpack_internal_request_key(
          private_owner.request_identity, &fields) !=
          request_owner::kStatusOk ||
      fields.resident_warp_slot != private_owner.resident_warp_id ||
      fields.lane_id != private_owner.lane_id ||
      fields.request_generation != private_owner.generation) {
    return false;
  }
  *request_binding = request_owner::lane_binding_v0();
  request_binding->packed_request_key = private_owner.request_identity;
  request_binding->owner_hw_sid = private_owner.owner_hw_sid;
  request_binding->request_control_slot = fields.request_control_slot;
  request_binding->request_generation = fields.request_generation;
  request_binding->private_slot_id =
      static_cast<uint16_t>(private_owner.private_slot_id);
  request_binding->resident_warp_slot = fields.resident_warp_slot;
  request_binding->lane_id = fields.lane_id;
  return true;
}

bool live_target_matches(
    const timing_driver::state_v0 &timing_state,
    const stack_operation::operation_packet_v0 &packet,
    request_owner::lane_binding_v0 *request_binding) {
  if (!make_request_owner(packet.owner, request_binding)) return false;
  const timing_driver::lane_control_state_v0 *control =
      timing_driver::find_live_lane_control(timing_state,
                                            *request_binding);
  return control != NULL &&
         control->live_target_operation_seq ==
             packet.target_operation_seq &&
         control->live_commit_producer_operation_seq == 0 &&
         control->live_commit_epoch == 0 &&
         control->pending_recovery_operation_seq == 0 &&
         control->live_memory_transaction_count == 0 &&
         control->live_commit_memory_transaction_count == 0;
}

bool operation_packet_valid(
    const stack_operation::operation_packet_v0 &packet) {
  const bool push_valid =
      packet.operation_kind ==
          typed_stack::kPushRemainderAndForwardSelected &&
      packet.input.operation_kind ==
          typed_stack::kPushRemainderAndForwardSelected;
  const bool pop_valid =
      packet.operation_kind == typed_stack::kPopNext &&
      packet.pop_input.operation_kind == typed_stack::kPopNext &&
      packet.pop_input.has_top_entry == 1;
  const bool empty_valid =
      packet.operation_kind == typed_stack::kPopNext &&
      packet.empty_input.operation_kind == typed_stack::kPopNext &&
      packet.empty_input.parent_frame_available <= 1;
  return packet.valid == 1 && packet.reservation_id != 0 &&
         packet.reservation_age != 0 &&
         packet.target_operation_seq != 0 &&
         packet.producer_operation_seq != 0 &&
         packet.target_operation_seq != packet.producer_operation_seq &&
         packet.slot_generation != 0 &&
         (push_valid || pop_valid || empty_valid) &&
         bytes_are_zero(packet.reserved_zero,
                        sizeof(packet.reserved_zero)) &&
         bytes_are_zero(packet.ray_policy.reserved_zero,
                        sizeof(packet.ray_policy.reserved_zero));
}

int find_free_pipeline(const state_v0 &state) {
  for (unsigned index = 0; index < kMaxPipelineEntries; ++index) {
    if (state.pipeline[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_oldest_matured_pipeline(const state_v0 &state,
                                 uint64_t service_cycle) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < kMaxPipelineEntries; ++index) {
    const pipeline_entry_v0 &entry = state.pipeline[index];
    if (entry.valid == 0 || entry.result_ready_cycle > service_cycle) {
      continue;
    }
    if (entry.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = entry.issue_age;
    }
  }
  return selected;
}

int find_available_unit(const state_v0 &state, uint64_t service_cycle,
                        uint8_t issued_unit_mask) {
  for (unsigned index = 0; index < state.config.stack_unit_count; ++index) {
    const uint8_t bit = static_cast<uint8_t>(1u << index);
    if ((issued_unit_mask & bit) == 0 &&
        state.units[index].next_issue_cycle <= service_cycle) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

status_kind capture_matured_result(
    state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t service_cycle, const result_sink_v0 *result_sink,
    cycle_result_v0 *result) {
  const int pipeline_index =
      find_oldest_matured_pipeline(*state, service_cycle);
  if (pipeline_index < 0) return kStatusOk;
  if (result_sink == NULL || result_sink->accept == NULL) {
    result->stall_mask = static_cast<uint8_t>(
        result->stall_mask | kStallResultSinkBackpressure);
    return kStatusOk;
  }

  pipeline_entry_v0 &pipeline = state->pipeline[pipeline_index];
  const bool push =
      pipeline.operation_packet.operation_kind ==
      typed_stack::kPushRemainderAndForwardSelected;
  const bool empty =
      !push &&
      pipeline.operation_packet.empty_input.operation_kind ==
          typed_stack::kPopNext;
  if (!operation_packet_valid(pipeline.operation_packet) ||
      (push
           ? !typed_stack::validate_push_result(
                 pipeline.typed_result)
           : empty
                 ? !typed_stack::validate_empty_result(
                       pipeline.operation_packet.empty_input,
                       pipeline.typed_empty_result)
                 : !typed_stack::validate_pop_result(
                 pipeline.operation_packet.pop_input,
                 pipeline.typed_pop_result)) ||
      pipeline.operator_invocation_count != 1) {
    return kStatusInvalidOperationPacket;
  }
  request_owner::lane_binding_v0 request_binding = {};
  if (!live_target_matches(*timing_state, pipeline.operation_packet,
                           &request_binding)) {
    return kStatusOwnerMismatch;
  }

  timing_driver::state_v0 staged_timing = *timing_state;
  uint32_t commit_epoch = 0;
  uint32_t target_operation_seq = 0;
  const bool terminal =
      empty &&
      (pipeline.typed_empty_result.result_kind ==
           typed_stack::kStackFinalHit ||
       pipeline.typed_empty_result.result_kind ==
           typed_stack::kStackFinalMiss);
  if (timing_driver::begin_result_commit(
          &staged_timing, request_binding,
          pipeline.operation_packet.target_operation_seq,
          &commit_epoch) != timing_driver::kStatusOk ||
      (!terminal &&
       timing_driver::allocate_commit_successor_operation(
           &staged_timing, request_binding,
           pipeline.operation_packet.target_operation_seq, commit_epoch,
           &target_operation_seq) != timing_driver::kStatusOk)) {
    return kStatusTimingControlRejected;
  }

  completed_push_receipt_v0 receipt = {};
  receipt.operation_packet = pipeline.operation_packet;
  receipt.typed_result = pipeline.typed_result;
  receipt.typed_pop_result = pipeline.typed_pop_result;
  receipt.typed_empty_result = pipeline.typed_empty_result;
  receipt.issue_age = pipeline.issue_age;
  receipt.issue_cycle = pipeline.issue_cycle;
  receipt.result_ready_cycle = pipeline.result_ready_cycle;
  receipt.capture_cycle = service_cycle;
  receipt.producer_operation_seq =
      pipeline.operation_packet.target_operation_seq;
  receipt.commit_epoch = commit_epoch;
  receipt.target_operation_seq = target_operation_seq;
  receipt.valid = 1;
  receipt.operation_kind =
      pipeline.operation_packet.operation_kind;
  receipt.operator_invocation_count =
      pipeline.operator_invocation_count;
  receipt.terminal_boundary = terminal ? 1 : 0;

  const result_sink_kind sink_status =
      result_sink->accept(&receipt, &staged_timing,
                          result_sink->context);
  if (sink_status == kResultSinkBackpressure) {
    result->stall_mask = static_cast<uint8_t>(
        result->stall_mask | kStallResultSinkBackpressure);
    return kStatusOk;
  }
  if (sink_status != kResultSinkAccepted) {
    return kStatusResultSinkRejected;
  }

  *timing_state = staged_timing;
  result->completed_push = receipt;
  result->captured_result_count = 1;
  std::memset(&pipeline, 0, sizeof(pipeline));
  ++state->total_results_captured;
  return kStatusOk;
}

status_kind issue_operations(
    state_v0 *state, stack_operation::engine_state_v0 *operation_state,
    const timing_driver::state_v0 &timing_state, uint64_t service_cycle,
    cycle_result_v0 *result) {
  uint8_t issued_unit_mask = 0;
  for (unsigned issued = 0; issued < state->config.stack_issue_width;
       ++issued) {
    const int pipeline_index = find_free_pipeline(*state);
    if (pipeline_index < 0) {
      if (stack_operation::ready_slot_count(*operation_state) != 0) {
        result->stall_mask = static_cast<uint8_t>(
            result->stall_mask | kStallPipelineFull);
      }
      return kStatusOk;
    }
    const int unit_index =
        find_available_unit(*state, service_cycle, issued_unit_mask);
    if (unit_index < 0) {
      if (stack_operation::ready_slot_count(*operation_state) != 0) {
        result->stall_mask = static_cast<uint8_t>(
            result->stall_mask | kStallUnitUnavailable);
      }
      return kStatusOk;
    }

    stack_operation::operation_packet_v0 candidate = {};
    const stack_operation::status_kind peek_status =
        stack_operation::peek_ready_operation(*operation_state,
                                              &candidate);
    if (peek_status == stack_operation::kStatusNoReadyOperation) {
      return kStatusOk;
    }
    if (peek_status != stack_operation::kStatusOk ||
        !operation_packet_valid(candidate)) {
      return kStatusQueueInvariant;
    }
    request_owner::lane_binding_v0 request_binding = {};
    if (!live_target_matches(timing_state, candidate,
                             &request_binding)) {
      return kStatusOwnerMismatch;
    }

    typed_stack::push_result_v0 typed_result = {};
    typed_stack::pop_result_v0 typed_pop_result = {};
    typed_stack::empty_result_v0 typed_empty_result = {};
    if (candidate.operation_kind ==
        typed_stack::kPushRemainderAndForwardSelected) {
      typed_result = typed_stack::execute_push(candidate.input);
      if (!typed_stack::validate_push_result(typed_result)) {
        return kStatusOperatorFailed;
      }
    } else if (candidate.empty_input.operation_kind ==
               typed_stack::kPopNext) {
      typed_empty_result =
          typed_stack::execute_empty(candidate.empty_input);
      if (!typed_stack::validate_empty_result(
              candidate.empty_input, typed_empty_result)) {
        return kStatusOperatorFailed;
      }
    } else {
      typed_pop_result =
          typed_stack::execute_pop(candidate.pop_input);
      if (!typed_stack::validate_pop_result(
              candidate.pop_input, typed_pop_result)) {
        return kStatusOperatorFailed;
      }
    }
    stack_operation::operation_packet_v0 packet = {};
    if (stack_operation::pop_ready_operation(
            operation_state, true, &packet) !=
            stack_operation::kStatusOk ||
        std::memcmp(&packet, &candidate, sizeof(packet)) != 0) {
      return kStatusQueueInvariant;
    }
    pipeline_entry_v0 &pipeline = state->pipeline[pipeline_index];
    std::memset(&pipeline, 0, sizeof(pipeline));
    pipeline.operation_packet = packet;
    pipeline.typed_result = typed_result;
    pipeline.typed_pop_result = typed_pop_result;
    pipeline.typed_empty_result = typed_empty_result;
    pipeline.issue_age = state->next_issue_age++;
    pipeline.issue_cycle = service_cycle;
    pipeline.result_ready_cycle =
        service_cycle + state->config.stack_latency;
    pipeline.valid = 1;
    pipeline.unit_index = static_cast<uint8_t>(unit_index);
    pipeline.operator_invocation_count = 1;
    state->units[unit_index].next_issue_cycle =
        service_cycle + state->config.stack_initiation_interval;
    issued_unit_mask = static_cast<uint8_t>(
        issued_unit_mask | (1u << unit_index));
    ++state->total_operator_invocations;
    ++result->issued_count;
  }
  return kStatusOk;
}

}  // namespace

config_v0 candidate_profile_config() {
  config_v0 config = {};
  config.stack_unit_count = 1;
  config.stack_latency = 2;
  config.stack_initiation_interval = 1;
  config.stack_issue_width = 1;
  return config;
}

status_kind initialize(state_v0 *state, const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!config_valid(config)) return kStatusInvalidConfiguration;
  *state = state_v0();
  state->config = config;
  state->next_issue_age = 1;
  state->initialized = 1;
  return kStatusOk;
}

status_kind service_cycle(
    state_v0 *state, stack_operation::engine_state_v0 *operation_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    const result_sink_v0 *result_sink, cycle_result_v0 *result) {
  if (state == NULL || operation_state == NULL || timing_state == NULL ||
      result == NULL || state->initialized != 1 ||
      operation_state->initialized != 1 || !timing_state->initialized) {
    return kStatusInvalidArgument;
  }
  *result = cycle_result_v0();
  if (state->last_service_cycle_valid != 0 &&
      service_cycle <= state->last_service_cycle) {
    return kStatusCycleRegression;
  }
  state->last_service_cycle = service_cycle;
  state->last_service_cycle_valid = 1;

  status_kind status =
      capture_matured_result(state, timing_state, service_cycle,
                             result_sink, result);
  if (status != kStatusOk) return status;
  status = issue_operations(state, operation_state, *timing_state,
                            service_cycle, result);
  if (status != kStatusOk) return status;
  result->active_pipeline_entries = active_pipeline_count(*state);
  result->ready_stack_entries =
      stack_operation::ready_slot_count(*operation_state);
  return kStatusOk;
}

uint8_t active_pipeline_count(const state_v0 &state) {
  if (state.initialized != 1) return 0;
  uint8_t count = 0;
  for (unsigned index = 0; index < kMaxPipelineEntries; ++index) {
    count += state.pipeline[index].valid != 0 ? 1 : 0;
  }
  return count;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfiguration:
      return "invalid_configuration";
    case kStatusCycleRegression:
      return "cycle_regression";
    case kStatusInvalidOperationPacket:
      return "invalid_operation_packet";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusOperatorFailed:
      return "operator_failed";
    case kStatusTimingControlRejected:
      return "timing_control_rejected";
    case kStatusQueueInvariant:
      return "queue_invariant";
    case kStatusResultSinkRejected:
      return "result_sink_rejected";
  }
  return "unknown";
}

}  // namespace stack_timing
}  // namespace v04
}  // namespace rtcore
