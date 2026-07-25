#include "rtcore_v04_instance_timing_driver.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace instance_timing {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool valid_config(const config_v0 &config) {
  return config.instance_unit_count != 0 &&
         config.instance_unit_count <= kMaxInstanceUnits &&
         config.instance_latency != 0 &&
         config.instance_initiation_interval != 0 &&
         config.instance_issue_width != 0 &&
         config.instance_issue_width <= config.instance_unit_count &&
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
  request_binding->request_control_slot =
      fields.request_control_slot;
  request_binding->request_generation =
      fields.request_generation;
  request_binding->private_slot_id =
      static_cast<uint16_t>(private_owner.private_slot_id);
  request_binding->resident_warp_slot =
      fields.resident_warp_slot;
  request_binding->lane_id = fields.lane_id;
  return true;
}

bool live_target_matches(
    const timing_driver::state_v0 &timing_state,
    const fetch_target::operation_packet_v0 &packet,
    request_owner::lane_binding_v0 *request_binding) {
  if (!make_request_owner(packet.owner, request_binding)) return false;
  const timing_driver::lane_control_state_v0 *control =
      timing_driver::find_live_lane_control(
          timing_state, *request_binding);
  return control != NULL &&
         control->live_target_operation_seq ==
             packet.target_operation_seq &&
         control->live_commit_producer_operation_seq == 0 &&
         control->live_commit_epoch == 0 &&
         control->pending_recovery_operation_seq == 0 &&
         control->live_memory_transaction_count == 0 &&
         control->live_commit_memory_transaction_count == 0;
}

bool restore_packet_valid(
    const fetch_target::operation_packet_v0 &packet) {
  return packet.valid == 1 &&
         packet.target_kind == fetch_target::kTargetInstance &&
         packet.operation_kind ==
             fetch_target::kOperationInstanceRestoreParent &&
         packet.reservation_id != 0 && packet.reservation_age != 0 &&
         packet.target_operation_seq != 0 &&
         packet.producer_operation_seq == 0 &&
         packet.producer_commit_epoch == 0 &&
         packet.slot_generation != 0 &&
         packet.raw_payload_base_address == 0 &&
         packet.raw_payload_bytes == 0 &&
         bytes_are_zero(packet.reserved_zero,
                        sizeof(packet.reserved_zero));
}

int find_free_pipeline(const state_v0 &state) {
  for (unsigned index = 0; index < kMaxPipelineEntries; ++index) {
    if (state.pipeline[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_oldest_matured(const state_v0 &state,
                        uint64_t service_cycle) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < kMaxPipelineEntries; ++index) {
    const pipeline_entry_v0 &entry = state.pipeline[index];
    if (entry.valid != 0 &&
        entry.result_ready_cycle <= service_cycle &&
        entry.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = entry.issue_age;
    }
  }
  return selected;
}

int find_available_unit(const state_v0 &state, uint64_t cycle,
                        uint8_t issued_unit_mask) {
  for (unsigned index = 0;
       index < state.config.instance_unit_count; ++index) {
    const uint8_t bit = static_cast<uint8_t>(1u << index);
    if ((issued_unit_mask & bit) == 0 &&
        state.units[index].next_issue_cycle <= cycle) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

status_kind capture_matured(
    state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t service_cycle, const result_sink_v0 *result_sink,
    cycle_result_v0 *result) {
  for (unsigned captured = 0;
       captured < state->config.instance_issue_width; ++captured) {
    const int index = find_oldest_matured(*state, service_cycle);
    if (index < 0) return kStatusOk;
    if (result_sink == NULL || result_sink->accept == NULL) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallResultSinkBackpressure);
      return kStatusOk;
    }
    pipeline_entry_v0 &pipeline = state->pipeline[index];
    typed_instance::restore_parent_input_v0 input = {};
    input.profile_id = typed_instance::kGenRtDerivedProfileId;
    input.operation_kind = typed_instance::kRestoreParent;
    input.parent_frame = pipeline.operation_packet.parent_frame;
    if (!restore_packet_valid(pipeline.operation_packet) ||
        !typed_instance::validate_restore_parent_result(
            input, pipeline.typed_result) ||
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
    if (timing_driver::begin_result_commit(
            &staged_timing, request_binding,
            pipeline.operation_packet.target_operation_seq,
            &commit_epoch) != timing_driver::kStatusOk ||
        timing_driver::allocate_commit_successor_operation(
            &staged_timing, request_binding,
            pipeline.operation_packet.target_operation_seq, commit_epoch,
            &target_operation_seq) != timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }

    completed_restore_receipt_v0 receipt = {};
    receipt.operation_packet = pipeline.operation_packet;
    receipt.typed_result = pipeline.typed_result;
    receipt.issue_age = pipeline.issue_age;
    receipt.issue_cycle = pipeline.issue_cycle;
    receipt.result_ready_cycle = pipeline.result_ready_cycle;
    receipt.capture_cycle = service_cycle;
    receipt.producer_operation_seq =
        pipeline.operation_packet.target_operation_seq;
    receipt.commit_epoch = commit_epoch;
    receipt.target_operation_seq = target_operation_seq;
    receipt.valid = 1;
    receipt.operator_invocation_count =
        pipeline.operator_invocation_count;
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
    result->completed_restores[result->captured_result_count] = receipt;
    ++result->captured_result_count;
    pipeline = pipeline_entry_v0();
    ++state->total_results_captured;
  }
  return kStatusOk;
}

status_kind issue_operations(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    const timing_driver::state_v0 &timing_state, uint64_t service_cycle,
    cycle_result_v0 *result) {
  uint8_t issued_unit_mask = 0;
  for (unsigned issued = 0;
       issued < state->config.instance_issue_width; ++issued) {
    fetch_target::operation_packet_v0 candidate = {};
    const fetch_target::status_kind peek_status =
        fetch_target::peek_ready_operation_kind(
            *target_state, fetch_target::kTargetInstance,
            fetch_target::kOperationInstanceRestoreParent, &candidate);
    if (peek_status == fetch_target::kStatusNoReadyOperation) {
      return kStatusOk;
    }
    if (peek_status != fetch_target::kStatusOk) {
      return kStatusQueueInvariant;
    }
    if (!restore_packet_valid(candidate)) {
      return kStatusInvalidOperationPacket;
    }
    const int pipeline_index = find_free_pipeline(*state);
    if (pipeline_index < 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallPipelineFull);
      return kStatusOk;
    }
    const int unit_index =
        find_available_unit(*state, service_cycle, issued_unit_mask);
    if (unit_index < 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallUnitUnavailable);
      return kStatusOk;
    }
    request_owner::lane_binding_v0 request_binding = {};
    if (!live_target_matches(timing_state, candidate,
                             &request_binding)) {
      return kStatusOwnerMismatch;
    }

    typed_instance::restore_parent_input_v0 input = {};
    input.profile_id = typed_instance::kGenRtDerivedProfileId;
    input.operation_kind = typed_instance::kRestoreParent;
    input.parent_frame = candidate.parent_frame;
    const typed_instance::restore_parent_result_v0 typed_result =
        typed_instance::execute_restore_parent(input);
    if (!typed_instance::validate_restore_parent_result(
            input, typed_result)) {
      return kStatusOperatorFailed;
    }
    fetch_target::operation_packet_v0 popped = {};
    if (fetch_target::pop_ready_operation_kind(
            target_state, fetch_target::kTargetInstance,
            fetch_target::kOperationInstanceRestoreParent, true,
            &popped) != fetch_target::kStatusOk ||
        std::memcmp(&candidate, &popped, sizeof(candidate)) != 0) {
      return kStatusQueueInvariant;
    }

    pipeline_entry_v0 &pipeline = state->pipeline[pipeline_index];
    pipeline = pipeline_entry_v0();
    pipeline.operation_packet = popped;
    pipeline.typed_result = typed_result;
    pipeline.issue_age = state->next_issue_age++;
    pipeline.issue_cycle = service_cycle;
    pipeline.result_ready_cycle =
        service_cycle + state->config.instance_latency;
    pipeline.valid = 1;
    pipeline.unit_index = static_cast<uint8_t>(unit_index);
    pipeline.operator_invocation_count = 1;
    state->units[unit_index].next_issue_cycle =
        service_cycle + state->config.instance_initiation_interval;
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
  config.instance_unit_count = 2;
  config.instance_latency = 4;
  config.instance_initiation_interval = 1;
  config.instance_issue_width = 2;
  return config;
}

status_kind initialize(state_v0 *state, const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!valid_config(config)) return kStatusInvalidConfiguration;
  *state = state_v0();
  state->config = config;
  state->next_issue_age = 1;
  state->initialized = 1;
  return kStatusOk;
}

status_kind service_cycle(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    const result_sink_v0 *result_sink, cycle_result_v0 *result) {
  if (state == NULL || target_state == NULL || timing_state == NULL ||
      result == NULL || state->initialized != 1 ||
      target_state->initialized != 1 || !timing_state->initialized) {
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
      capture_matured(state, timing_state, service_cycle,
                      result_sink, result);
  if (status != kStatusOk) return status;
  status = issue_operations(state, target_state, *timing_state,
                            service_cycle, result);
  if (status != kStatusOk) return status;
  result->active_pipeline_entries = active_pipeline_count(*state);
  result->ready_instance_entries = fetch_target::ready_slot_count(
      *target_state, fetch_target::kTargetInstance);
  return kStatusOk;
}

uint8_t active_pipeline_count(const state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return count;
  for (unsigned index = 0; index < kMaxPipelineEntries; ++index) {
    count += state.pipeline[index].valid != 0;
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

}  // namespace instance_timing
}  // namespace v04
}  // namespace rtcore
