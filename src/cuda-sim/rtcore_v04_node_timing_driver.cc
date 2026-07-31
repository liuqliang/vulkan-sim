#include "rtcore_v04_node_timing_driver.h"

#include <cstring>
#include <limits>

#include "rtcore_v04_stall_attribution.h"
#include "rtcore_v04_typed_diagnostic_collector.h"

namespace rtcore {
namespace v04 {
namespace node_timing {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  if (bytes == NULL) return false;
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool config_valid(const config_v0 &config) {
  return config.node_unit_count != 0 &&
         config.node_unit_count <= kMaxNodeUnits &&
         config.node_latency != 0 &&
         config.node_initiation_interval != 0 &&
         config.node_issue_width != 0 &&
         config.node_issue_width <= config.node_unit_count &&
         config.result_commit_capacity != 0 &&
         config.result_commit_capacity <= kMaxResultCommitEntries &&
         config.result_commit_width != 0 &&
         config.result_commit_width <= config.result_commit_capacity &&
         bytes_are_zero(config.reserved_zero,
                        sizeof(config.reserved_zero));
}

result_identity_envelope_v0 make_result_identity(
    const fetch_target::operation_packet_v0 &packet) {
  result_identity_envelope_v0 identity = {};
  identity.owner = packet.owner;
  identity.reservation_id = packet.reservation_id;
  identity.target_operation_seq = packet.target_operation_seq;
  identity.slot_generation = packet.slot_generation;
  identity.private_storage_profile =
      packet.private_storage_profile;
  return identity;
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
  request_binding->packed_request_key =
      private_owner.request_identity;
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
    const result_identity_envelope_v0 &identity,
    request_owner::lane_binding_v0 *request_binding) {
  if (!make_request_owner(identity.owner, request_binding)) {
    return false;
  }
  const timing_driver::lane_control_state_v0 *control =
      timing_driver::find_live_lane_control(
          timing_state, *request_binding);
  return control != NULL &&
         control->live_target_operation_seq ==
             identity.target_operation_seq &&
         control->live_commit_producer_operation_seq == 0 &&
         control->live_commit_epoch == 0 &&
         control->pending_recovery_operation_seq == 0 &&
         control->live_memory_transaction_count == 0 &&
         control->live_commit_memory_transaction_count == 0;
}

void emit_attempt(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, uint64_t service_cycle,
    uint16_t arbitration_slot,
    stall_attribution::stage_kind stage,
    stall_attribution::outcome_kind outcome,
    stall_attribution::action_kind action,
    stall_attribution::reason_kind reason) {
  if (!stall_attribution::enabled()) return;
  stall_attribution::attempt_record_v0 record = {};
  record.service_cycle = service_cycle;
  record.owner_hw_sid = owner.owner_hw_sid;
  record.request_identity = owner.request_identity;
  record.request_generation = owner.generation;
  record.operation_seq = operation_seq;
  record.chunk_id = 0;
  record.chunk_count = 1;
  record.arbitration_slot = arbitration_slot;
  record.lane_id = owner.lane_id;
  record.unit = stall_attribution::kUnitNode;
  record.stage = stage;
  record.outcome = outcome;
  record.action = action;
  record.reason = reason;
  stall_attribution::emit_attempt(record);
}

void emit_ready_issue_stall(
    const fetch_target::engine_state_v0 &target_state,
    uint64_t service_cycle, uint16_t arbitration_slot,
    stall_attribution::reason_kind reason) {
  if (!stall_attribution::enabled()) return;
  fetch_target::operation_packet_v0 packet = {};
  if (fetch_target::peek_ready_operation(
          target_state, fetch_target::kTargetNode, &packet) !=
      fetch_target::kStatusOk) {
    return;
  }
  emit_attempt(
      packet.owner, packet.target_operation_seq, service_cycle,
      arbitration_slot, stall_attribution::kStageIssue,
      stall_attribution::kOutcomeStall, stall_attribution::kActionNone,
      reason);
}

int find_free_pipeline(const state_v0 &state) {
  for (unsigned index = 0; index < kMaxNodePipelineEntries; ++index) {
    if (state.node_pipeline[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_free_result(const state_v0 &state) {
  for (unsigned index = 0; index < state.config.result_commit_capacity;
       ++index) {
    if (state.result_commits[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_available_unit(const state_v0 &state, uint64_t service_cycle,
                        uint8_t issued_unit_mask) {
  for (unsigned index = 0; index < state.config.node_unit_count; ++index) {
    const uint8_t unit_bit = static_cast<uint8_t>(1u << index);
    if ((issued_unit_mask & unit_bit) == 0 &&
        state.node_units[index].next_issue_cycle <= service_cycle) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_oldest_matured_pipeline(const state_v0 &state,
                                 uint64_t service_cycle) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < kMaxNodePipelineEntries; ++index) {
    const node_pipeline_entry_v0 &entry = state.node_pipeline[index];
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

int find_oldest_result(const state_v0 &state) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.result_commit_capacity;
       ++index) {
    const result_commit_entry_v0 &entry = state.result_commits[index];
    if (entry.valid != 0 && entry.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = entry.issue_age;
    }
  }
  return selected;
}

status_kind commit_results(
    state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t service_cycle, bool result_commit_accepts,
    const route_sink_v0 *route_sink,
    cycle_result_v0 *result) {
  if (!result_commit_accepts) {
    if (active_result_count(*state) != 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallResultCommitNotAccepted);
      const int result_index = find_oldest_result(*state);
      if (result_index >= 0) {
        const result_commit_entry_v0 &entry =
            state->result_commits[result_index];
        emit_attempt(
            entry.result_identity.owner,
            entry.result_identity.target_operation_seq, service_cycle, 0,
            stall_attribution::kStageCommit,
            stall_attribution::kOutcomeStall,
            stall_attribution::kActionNone,
            stall_attribution::kReasonResultSinkCapacity);
      }
    }
    return kStatusOk;
  }
  for (unsigned committed = 0;
       committed < state->config.result_commit_width; ++committed) {
    const int result_index = find_oldest_result(*state);
    if (result_index < 0) break;
    result_commit_entry_v0 &entry =
        state->result_commits[result_index];
    request_owner::lane_binding_v0 request_binding = {};
    if (!make_request_owner(entry.result_identity.owner,
                            &request_binding)) {
      return kStatusOwnerMismatch;
    }
    timing_driver::state_v0 staged_timing = *timing_state;
    const timing_driver::status_kind timing_status =
        timing_driver::complete_result_commit(
            &staged_timing, request_binding,
            entry.result_identity.target_operation_seq,
            entry.commit_epoch);
    if (timing_status != timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
    committed_route_receipt_v0 receipt = {};
    receipt.result_identity = entry.result_identity;
    if (result_semantic::materialize_node_result(
            entry.semantic_plan, &receipt.typed_result) !=
        result_semantic::kStatusOk) {
      return kStatusSemanticApplyFailed;
    }
    receipt.semantic_plan = entry.semantic_plan;
    receipt.ray_policy = entry.ray_policy;
    receipt.current_target_reference =
        entry.current_target_reference;
    receipt.current_decode_context = entry.current_decode_context;
    receipt.pending_parent_resume =
        entry.pending_parent_resume;
    receipt.pending_parent_resume_valid =
        entry.pending_parent_resume_valid;
    receipt.issue_cycle = entry.issue_cycle;
    receipt.result_ready_cycle = entry.result_ready_cycle;
    receipt.capture_cycle = entry.capture_cycle;
    receipt.commit_cycle = service_cycle;
    receipt.commit_epoch = entry.commit_epoch;
    receipt.current_traversal_bound_bits =
        entry.current_traversal_bound_bits;
    receipt.valid = 1;
    receipt.operator_invocation_count =
        entry.operator_invocation_count;
    if (route_sink != NULL) {
      const route_sink_result_kind sink_result =
          route_sink->accept(&receipt, &staged_timing,
                             route_sink->context);
      if (sink_result == kRouteSinkBackpressure) {
        result->stall_mask = static_cast<uint8_t>(
            result->stall_mask | kStallRouteSinkBackpressure);
        emit_attempt(
            entry.result_identity.owner,
            entry.result_identity.target_operation_seq, service_cycle,
            static_cast<uint16_t>(committed),
            stall_attribution::kStageCommit,
            stall_attribution::kOutcomeStall,
            stall_attribution::kActionNone,
            stall_attribution::kReasonResultSinkCapacity);
        return kStatusOk;
      }
      if (sink_result != kRouteSinkAccepted) {
        return kStatusRouteSinkRejected;
      }
    }
    *timing_state = staged_timing;
    result->committed_routes[result->committed_route_count++] =
        receipt;
    emit_attempt(
        entry.result_identity.owner,
        entry.result_identity.target_operation_seq, service_cycle,
        static_cast<uint16_t>(committed),
        stall_attribution::kStageCommit,
        stall_attribution::kOutcomeProgress,
        stall_attribution::kActionCommit,
        stall_attribution::kReasonNone);
    std::memset(&entry, 0, sizeof(entry));
    ++state->total_routes_committed;
  }
  return kStatusOk;
}

status_kind capture_matured_results(
    state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t service_cycle, cycle_result_v0 *result) {
  while (true) {
    const int pipeline_index =
        find_oldest_matured_pipeline(*state, service_cycle);
    if (pipeline_index < 0) return kStatusOk;
    const int result_index = find_free_result(*state);
    if (result_index < 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallResultCommitFull);
      const node_pipeline_entry_v0 &pipeline =
          state->node_pipeline[pipeline_index];
      emit_attempt(
          pipeline.operation_packet.owner,
          pipeline.operation_packet.target_operation_seq, service_cycle,
          result->captured_result_count,
          stall_attribution::kStageCapture,
          stall_attribution::kOutcomeStall,
          stall_attribution::kActionNone,
          stall_attribution::kReasonResultCommitCapacity);
      return kStatusOk;
    }

    node_pipeline_entry_v0 &pipeline =
        state->node_pipeline[pipeline_index];
    if (!validate_result_identity(
            pipeline.result_identity, pipeline.operation_packet)) {
      return kStatusResultIdentityMismatch;
    }
    request_owner::lane_binding_v0 request_binding = {};
    if (!live_target_matches(
            *timing_state, pipeline.result_identity,
            &request_binding)) {
      return kStatusOwnerMismatch;
    }
    result_semantic::node_commit_plan_v0 semantic_plan = {};
    if (result_semantic::prepare_node_result(
            pipeline.operation_packet, pipeline.typed_result,
            &semantic_plan) != result_semantic::kStatusOk) {
      return kStatusSemanticApplyFailed;
    }
    typed_node::route_input_v0 typed_input = {};
    if (functional_driver::prepare_node_operator_input(
            pipeline.operation_packet, &typed_input) !=
            functional_driver::kStatusOk) {
      return kStatusInvalidOperationPacket;
    }
    uint32_t commit_epoch = 0;
    if (timing_driver::begin_result_commit(
            timing_state, request_binding,
            pipeline.result_identity.target_operation_seq,
            &commit_epoch) != timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }

    result_commit_entry_v0 &entry =
        state->result_commits[result_index];
    std::memset(&entry, 0, sizeof(entry));
    entry.result_identity = pipeline.result_identity;
    entry.semantic_plan = semantic_plan;
    entry.ray_policy = pipeline.operation_packet.ray_policy;
    entry.current_target_reference =
        pipeline.operation_packet.target_reference;
    entry.current_decode_context =
        pipeline.operation_packet.private_operands.decode_context;
    entry.pending_parent_resume =
        pipeline.operation_packet.pending_parent_resume;
    entry.pending_parent_resume_valid =
        pipeline.operation_packet.pending_parent_resume_valid;
    entry.issue_age = pipeline.issue_age;
    entry.issue_cycle = pipeline.issue_cycle;
    entry.result_ready_cycle = pipeline.result_ready_cycle;
    entry.capture_cycle = service_cycle;
    entry.commit_epoch = commit_epoch;
    entry.current_traversal_bound_bits = fp32_bits(
        pipeline.operation_packet.private_operands.committed_hit.valid != 0
            ? pipeline.operation_packet.private_operands.committed_hit.hit_t
            : pipeline.operation_packet.private_operands.mutable_ray.t_max);
    entry.operator_invocation_count =
        pipeline.operator_invocation_count;
    entry.valid = 1;
    typed_diagnostic::record_v0 diagnostic = {};
    diagnostic.owner = pipeline.operation_packet.owner;
    diagnostic.operation_seq =
        pipeline.operation_packet.target_operation_seq;
    diagnostic.driver = typed_diagnostic::kDriverTiming;
    diagnostic.unit = typed_diagnostic::kUnitNode;
    diagnostic.operation_kind =
        pipeline.operation_packet.operation_kind;
    diagnostic.semantic_plan_kind =
        functional_driver::kSemanticPlanNode;
    diagnostic.route_kind = semantic_plan.route_kind;
    diagnostic.typed_input = &typed_input;
    diagnostic.typed_input_bytes = sizeof(typed_input);
    diagnostic.typed_result = &pipeline.typed_result;
    diagnostic.typed_result_bytes = sizeof(pipeline.typed_result);
    diagnostic.semantic_plan = &semantic_plan;
    diagnostic.semantic_plan_bytes = sizeof(semantic_plan);
    if (!typed_diagnostic::emit_record(diagnostic)) {
      return kStatusSemanticApplyFailed;
    }
    emit_attempt(
        pipeline.operation_packet.owner,
        pipeline.operation_packet.target_operation_seq, service_cycle,
        result->captured_result_count,
        stall_attribution::kStageCapture,
        stall_attribution::kOutcomeProgress,
        stall_attribution::kActionCapture,
        stall_attribution::kReasonNone);
    std::memset(&pipeline, 0, sizeof(pipeline));
    ++result->captured_result_count;
  }
}

status_kind issue_node_operations(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    const timing_driver::state_v0 &timing_state,
    uint64_t service_cycle, cycle_result_v0 *result) {
  uint8_t issued_unit_mask = 0;
  for (unsigned issued = 0; issued < state->config.node_issue_width;
       ++issued) {
    const int pipeline_index = find_free_pipeline(*state);
    if (pipeline_index < 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallNodePipelineFull);
      emit_ready_issue_stall(
          *target_state, service_cycle, static_cast<uint16_t>(issued),
          stall_attribution::kReasonPipelineCapacity);
      return kStatusOk;
    }
    const int unit_index =
        find_available_unit(*state, service_cycle, issued_unit_mask);
    if (unit_index < 0) {
      if (fetch_target::ready_slot_count(
              *target_state, fetch_target::kTargetNode) != 0) {
        result->stall_mask = static_cast<uint8_t>(
            result->stall_mask | kStallNodeUnitUnavailable);
        emit_ready_issue_stall(
            *target_state, service_cycle, static_cast<uint16_t>(issued),
            stall_attribution::kReasonUnitBusy);
      }
      return kStatusOk;
    }

    fetch_target::operation_packet_v0 packet = {};
    const fetch_target::status_kind peek_status =
        fetch_target::peek_ready_operation(
            *target_state, fetch_target::kTargetNode, &packet);
    if (peek_status == fetch_target::kStatusNoReadyOperation) {
      return kStatusOk;
    }
    if (peek_status != fetch_target::kStatusOk ||
        !result_semantic::validate_node_operation_packet(packet)) {
      return kStatusInvalidOperationPacket;
    }
    const result_identity_envelope_v0 identity =
        make_result_identity(packet);
    request_owner::lane_binding_v0 request_binding = {};
    if (!live_target_matches(
            timing_state, identity, &request_binding)) {
      return kStatusOwnerMismatch;
    }

    fetch_target::operation_packet_v0 popped = {};
    if (fetch_target::pop_ready_operation(
            target_state, fetch_target::kTargetNode, true,
            &popped) != fetch_target::kStatusOk ||
        std::memcmp(&packet, &popped, sizeof(packet)) != 0) {
      return kStatusQueueInvariant;
    }
    typed_node::route_result_v0 typed_result = {};
    uint8_t invocation_count = 0;
    if (functional_driver::execute_node_operator_once(
            popped, &typed_result, &invocation_count) !=
            functional_driver::kStatusOk ||
        invocation_count != 1) {
      return kStatusOperatorFailed;
    }

    node_pipeline_entry_v0 &pipeline =
        state->node_pipeline[pipeline_index];
    std::memset(&pipeline, 0, sizeof(pipeline));
    pipeline.operation_packet = popped;
    pipeline.typed_result = typed_result;
    pipeline.result_identity = identity;
    pipeline.issue_age = state->next_issue_age++;
    pipeline.issue_cycle = service_cycle;
    pipeline.result_ready_cycle =
        service_cycle + state->config.node_latency;
    pipeline.valid = 1;
    pipeline.unit_index = static_cast<uint8_t>(unit_index);
    pipeline.operator_invocation_count = invocation_count;
    state->node_units[unit_index].next_issue_cycle =
        service_cycle + state->config.node_initiation_interval;
    issued_unit_mask = static_cast<uint8_t>(
        issued_unit_mask | (1u << unit_index));
    ++state->total_operator_invocations;
    ++result->issued_count;
    emit_attempt(
        popped.owner, popped.target_operation_seq, service_cycle,
        static_cast<uint16_t>(issued),
        stall_attribution::kStageIssue,
        stall_attribution::kOutcomeProgress,
        stall_attribution::kActionIssue,
        stall_attribution::kReasonNone);
  }
  return kStatusOk;
}

}  // namespace

config_v0 candidate_profile_config() {
  config_v0 config = {};
  config.node_unit_count = 8;
  config.node_latency = 2;
  config.node_initiation_interval = 1;
  config.node_issue_width = 8;
  config.result_commit_capacity = 16;
  config.result_commit_width = 16;
  return config;
}

status_kind initialize(state_v0 *state, const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!config_valid(config)) return kStatusInvalidConfiguration;
  std::memset(state, 0, sizeof(*state));
  state->config = config;
  state->next_issue_age = 1;
  state->initialized = 1;
  return kStatusOk;
}

bool validate_result_identity(
    const result_identity_envelope_v0 &identity,
    const fetch_target::operation_packet_v0 &packet) {
  return packet.valid == 1 &&
         private_frontier::owners_equal(identity.owner, packet.owner) &&
         identity.reservation_id != 0 &&
         identity.reservation_id == packet.reservation_id &&
         identity.target_operation_seq != 0 &&
         identity.target_operation_seq ==
             packet.target_operation_seq &&
         identity.slot_generation != 0 &&
         identity.slot_generation == packet.slot_generation &&
         identity.private_storage_profile ==
             packet.private_storage_profile &&
         bytes_are_zero(identity.reserved_zero,
                        sizeof(identity.reserved_zero));
}

status_kind service_cycle(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    bool result_commit_accepts, const route_sink_v0 *route_sink,
    cycle_result_v0 *result) {
  if (state == NULL || target_state == NULL || timing_state == NULL ||
      result == NULL || state->initialized != 1 ||
      target_state->initialized != 1 || !timing_state->initialized) {
    return kStatusInvalidArgument;
  }
  if (route_sink != NULL && route_sink->accept == NULL) {
    return kStatusInvalidArgument;
  }
  *result = cycle_result_v0();
  if (state->last_service_cycle_valid != 0 &&
      service_cycle <= state->last_service_cycle) {
    return kStatusCycleRegression;
  }

  status_kind status =
      commit_results(state, timing_state, service_cycle,
                     result_commit_accepts, route_sink, result);
  if (status == kStatusOk) {
    status = capture_matured_results(
        state, timing_state, service_cycle, result);
  }
  if (status == kStatusOk) {
    status = issue_node_operations(
        state, target_state, *timing_state, service_cycle, result);
  }
  if (status != kStatusOk) return status;

  state->last_service_cycle = service_cycle;
  state->last_service_cycle_valid = 1;
  result->active_pipeline_entries = active_pipeline_count(*state);
  result->active_result_entries = active_result_count(*state);
  result->ready_node_entries = fetch_target::ready_slot_count(
      *target_state, fetch_target::kTargetNode);
  return kStatusOk;
}

status_kind service_cycle(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state, uint64_t cycle,
    bool result_commit_accepts, cycle_result_v0 *result) {
  return service_cycle(state, target_state, timing_state, cycle,
                       result_commit_accepts, NULL, result);
}

uint8_t active_pipeline_count(const state_v0 &state) {
  uint8_t count = 0;
  for (unsigned index = 0; index < kMaxNodePipelineEntries; ++index) {
    count = static_cast<uint8_t>(
        count + (state.node_pipeline[index].valid != 0));
  }
  return count;
}

uint8_t active_result_count(const state_v0 &state) {
  uint8_t count = 0;
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    count = static_cast<uint8_t>(
        count + (state.result_commits[index].valid != 0));
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
    case kStatusResultIdentityMismatch:
      return "result_identity_mismatch";
    case kStatusSemanticApplyFailed:
      return "semantic_apply_failed";
    case kStatusTimingControlRejected:
      return "timing_control_rejected";
    case kStatusQueueInvariant:
      return "queue_invariant";
    case kStatusRouteSinkRejected:
      return "route_sink_rejected";
  }
  return "unknown";
}

}  // namespace node_timing
}  // namespace v04
}  // namespace rtcore
