#include "rtcore_v04_primitive_timing_driver.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace primitive_timing {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool valid_config(const config_v0 &config) {
  return config.primitive_unit_count != 0 &&
         config.primitive_unit_count <= kMaxPrimitiveUnits &&
         config.primitive_first_batch_latency != 0 &&
         config.primitive_batch_width != 0 &&
         config.primitive_batch_interval != 0 &&
         config.primitive_issue_width != 0 &&
         config.primitive_issue_width <= config.primitive_unit_count &&
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

unsigned population_count(uint64_t mask) {
  unsigned count = 0;
  while (mask != 0) {
    mask &= mask - 1;
    ++count;
  }
  return count;
}

uint64_t compute_latency(const config_v0 &config,
                         uint64_t input_slot_mask) {
  const unsigned slot_count = population_count(input_slot_mask);
  if (slot_count == 0) return 0;
  const unsigned batch_count =
      (slot_count + config.primitive_batch_width - 1) /
      config.primitive_batch_width;
  return static_cast<uint64_t>(
      config.primitive_first_batch_latency) +
      static_cast<uint64_t>(batch_count - 1) *
          config.primitive_batch_interval;
}

bool result_requires_internal_successor(
    const typed_primitive::route_result_v0 &result) {
  return result.result_kind ==
             typed_primitive::kRouteResultNoCandidate ||
         result.result_kind ==
             typed_primitive::kRouteResultCommitOpaque;
}

int find_idle_unit(const state_v0 &state) {
  for (unsigned index = 0;
       index < state.config.primitive_unit_count; ++index) {
    if (state.units[index].phase == kUnitIdle) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_oldest_matured(state_v0 *state, uint64_t service_cycle) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0;
       index < state->config.primitive_unit_count; ++index) {
    unit_state_v0 &unit = state->units[index];
    if (unit.phase == kUnitExecutingLeaf &&
        unit.result_ready_cycle <= service_cycle) {
      unit.phase = kUnitOutputPending;
    }
    if (unit.phase == kUnitOutputPending &&
        (selected < 0 || unit.issue_age < selected_age)) {
      selected = static_cast<int>(index);
      selected_age = unit.issue_age;
    }
  }
  return selected;
}

status_kind capture_matured(
    state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t service_cycle, const result_sink_v0 *result_sink,
    cycle_result_v0 *result) {
  for (unsigned captured = 0;
       captured < state->config.primitive_issue_width; ++captured) {
    const int index = find_oldest_matured(state, service_cycle);
    if (index < 0) return kStatusOk;
    unit_state_v0 &unit = state->units[index];
    if (result_sink == NULL || result_sink->accept == NULL) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallResultSinkBackpressure);
      return kStatusOk;
    }
    if (unit.operator_invocation_count != 1 ||
        unit.typed_result.status != typed_primitive::kStatusOk) {
      return kStatusInvalidOperationPacket;
    }
    request_owner::lane_binding_v0 request_binding = {};
    if (!live_target_matches(*timing_state, unit.operation_packet,
                             &request_binding)) {
      return kStatusOwnerMismatch;
    }

    timing_driver::state_v0 staged_timing = *timing_state;
    uint32_t commit_epoch = 0;
    uint32_t target_operation_seq = 0;
    if (timing_driver::begin_result_commit(
            &staged_timing, request_binding,
            unit.operation_packet.target_operation_seq,
            &commit_epoch) != timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
    if (result_requires_internal_successor(unit.typed_result) &&
        timing_driver::allocate_commit_successor_operation(
            &staged_timing, request_binding,
            unit.operation_packet.target_operation_seq, commit_epoch,
            &target_operation_seq) != timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }

    completed_receipt_v0 receipt = {};
    receipt.operation_packet = unit.operation_packet;
    receipt.typed_input = unit.typed_input;
    receipt.typed_result = unit.typed_result;
    receipt.issue_age = unit.issue_age;
    receipt.issue_cycle = unit.issue_cycle;
    receipt.result_ready_cycle = unit.result_ready_cycle;
    receipt.capture_cycle = service_cycle;
    receipt.producer_operation_seq =
        unit.operation_packet.target_operation_seq;
    receipt.commit_epoch = commit_epoch;
    receipt.target_operation_seq = target_operation_seq;
    receipt.valid = 1;
    receipt.operator_invocation_count =
        unit.operator_invocation_count;
    receipt.unit_index = static_cast<uint8_t>(index);
    const result_sink_kind sink_status =
        result_sink->accept(
            &receipt, &staged_timing, result_sink->context);
    if (sink_status == kResultSinkBackpressure) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallResultSinkBackpressure);
      return kStatusOk;
    }
    if (sink_status != kResultSinkAccepted) {
      return kStatusResultSinkRejected;
    }

    *timing_state = staged_timing;
    result->completed[result->captured_result_count++] = receipt;
    unit = unit_state_v0();
    ++state->total_results_captured;
  }
  return kStatusOk;
}

status_kind issue_operations(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    const timing_driver::state_v0 &timing_state,
    uint64_t service_cycle, cycle_result_v0 *result) {
  for (unsigned issued = 0;
       issued < state->config.primitive_issue_width; ++issued) {
    fetch_target::operation_packet_v0 packet = {};
    const fetch_target::status_kind peek_status =
        fetch_target::peek_ready_operation(
            *target_state, fetch_target::kTargetPrimitive, &packet);
    if (peek_status == fetch_target::kStatusNoReadyOperation) {
      return kStatusOk;
    }
    if (peek_status != fetch_target::kStatusOk) {
      return kStatusQueueInvariant;
    }
    const int unit_index = find_idle_unit(*state);
    if (unit_index < 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallUnitUnavailable);
      return kStatusOk;
    }
    request_owner::lane_binding_v0 request_binding = {};
    if (!live_target_matches(timing_state, packet,
                             &request_binding)) {
      return kStatusOwnerMismatch;
    }
    typed_primitive::route_input_v0 input = {};
    if (functional_driver::prepare_primitive_operator_input(
            packet, &input) != functional_driver::kStatusOk) {
      return kStatusInvalidOperationPacket;
    }
    const uint64_t latency =
        compute_latency(state->config, input.input_slot_mask);
    if (state->next_issue_age ==
            std::numeric_limits<uint64_t>::max() ||
        latency >
            std::numeric_limits<uint64_t>::max() - service_cycle) {
      return kStatusCounterExhausted;
    }
    const typed_primitive::route_result_v0 typed_result =
        typed_primitive::execute_route(input);
    if (typed_result.status != typed_primitive::kStatusOk ||
        typed_result.typed_operator_invocation_count != 1) {
      result->typed_operator_status = typed_result.status;
      return kStatusOperatorFailed;
    }
    fetch_target::operation_packet_v0 popped = {};
    if (fetch_target::pop_ready_operation(
            target_state, fetch_target::kTargetPrimitive, true,
            &popped) != fetch_target::kStatusOk ||
        std::memcmp(&packet, &popped, sizeof(packet)) != 0) {
      return kStatusQueueInvariant;
    }

    unit_state_v0 &unit = state->units[unit_index];
    unit = unit_state_v0();
    unit.operation_packet = popped;
    unit.typed_input = input;
    unit.typed_result = typed_result;
    unit.issue_age = state->next_issue_age++;
    unit.issue_cycle = service_cycle;
    unit.result_ready_cycle = service_cycle + latency;
    unit.phase = kUnitExecutingLeaf;
    unit.operator_invocation_count = 1;
    ++state->total_operator_invocations;
    ++result->issued_count;
  }
  return kStatusOk;
}

}  // namespace

config_v0 candidate_profile_config() {
  config_v0 config = {};
  config.primitive_unit_count = 4;
  config.primitive_first_batch_latency = 4;
  config.primitive_batch_width = 2;
  config.primitive_batch_interval = 1;
  config.primitive_issue_width = 4;
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
  status = issue_operations(
      state, target_state, *timing_state, service_cycle, result);
  if (status != kStatusOk) return status;
  result->active_unit_count = active_unit_count(*state);
  result->executing_unit_count = executing_unit_count(*state);
  result->output_pending_count = output_pending_count(*state);
  result->ready_primitive_entries = fetch_target::ready_slot_count(
      *target_state, fetch_target::kTargetPrimitive);
  return kStatusOk;
}

uint8_t active_unit_count(const state_v0 &state) {
  return static_cast<uint8_t>(
      executing_unit_count(state) + output_pending_count(state));
}

uint8_t executing_unit_count(const state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return count;
  for (unsigned index = 0;
       index < state.config.primitive_unit_count; ++index) {
    count += state.units[index].phase == kUnitExecutingLeaf;
  }
  return count;
}

uint8_t output_pending_count(const state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return count;
  for (unsigned index = 0;
       index < state.config.primitive_unit_count; ++index) {
    count += state.units[index].phase == kUnitOutputPending;
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
    case kStatusCounterExhausted:
      return "counter_exhausted";
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

}  // namespace primitive_timing
}  // namespace v04
}  // namespace rtcore
