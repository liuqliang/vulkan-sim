#include "rtcore_v04_instance_timing_driver.h"

#include <cstring>
#include <limits>

#include "rtcore_v04_stall_attribution.h"

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
  record.unit = stall_attribution::kUnitInstance;
  record.stage = stage;
  record.outcome = outcome;
  record.action = action;
  record.reason = reason;
  stall_attribution::emit_attempt(record);
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

bool enter_packet_valid(
    const fetch_target::operation_packet_v0 &packet) {
  const bool producer_identity_valid =
      (packet.producer_operation_seq == 0 &&
       packet.producer_commit_epoch == 0) ||
      (packet.producer_operation_seq != 0 &&
       packet.producer_commit_epoch != 0 &&
       packet.producer_operation_seq !=
           packet.target_operation_seq);
  return packet.valid == 1 &&
         packet.target_kind == fetch_target::kTargetInstance &&
         packet.operation_kind == fetch_target::kOperationFetchTarget &&
         packet.reservation_id != 0 && packet.reservation_age != 0 &&
         packet.target_operation_seq != 0 &&
         producer_identity_valid &&
         packet.slot_generation != 0 &&
         packet.raw_payload_base_address != 0 &&
         packet.raw_payload_bytes ==
             fetch_target::kInstanceRawPayloadBytes &&
         packet.target_reference.payload_kind ==
             typed_node::kInstancePayloadKind &&
         packet.target_reference.payload_byte_count ==
             fetch_target::kInstanceRawPayloadBytes &&
         packet.target_reference.level == typed_node::kLevelTlas &&
         packet.target_reference.source_kind ==
             fetch_target::
                 kTargetReferenceSelectedFetchCompatibilityAdapter &&
         packet.target_reference.proxy_delegated == 1 &&
         bytes_are_zero(packet.reserved_zero,
                        sizeof(packet.reserved_zero));
}

void record_packet_failure(
    cycle_result_v0 *result, failure_point_kind failure_point,
    const fetch_target::operation_packet_v0 &packet,
    uint8_t operator_invocation_count, uint8_t typed_status,
    uint8_t typed_result_kind) {
  if (result == NULL) return;
  result->failure_point = failure_point;
  result->failure_packet_valid = packet.valid;
  result->failure_target_kind = packet.target_kind;
  result->failure_operation_kind = packet.operation_kind;
  result->failure_operator_invocation_count =
      operator_invocation_count;
  result->failure_typed_status = typed_status;
  result->failure_typed_result_kind = typed_result_kind;
  result->failure_operation_seq = packet.target_operation_seq;
  result->failure_producer_operation_seq =
      packet.producer_operation_seq;
  result->failure_producer_commit_epoch =
      packet.producer_commit_epoch;
}

status_kind select_oldest_candidate(
    const fetch_target::engine_state_v0 &target_state,
    const result_sink_v0 *result_sink,
    fetch_target::operation_packet_v0 *candidate) {
  if (candidate == NULL) return kStatusInvalidArgument;
  *candidate = fetch_target::operation_packet_v0();
  fetch_target::operation_packet_v0 restore = {};
  fetch_target::operation_packet_v0 enter = {};
  const bool restore_enabled =
      result_sink != NULL && result_sink->accept != NULL;
  const bool enter_enabled =
      result_sink != NULL && result_sink->prepare_enter != NULL &&
      result_sink->accept_enter != NULL;
  const fetch_target::status_kind restore_status =
      restore_enabled
          ? fetch_target::peek_ready_operation_kind(
                target_state, fetch_target::kTargetInstance,
                fetch_target::kOperationInstanceRestoreParent,
                &restore)
          : fetch_target::kStatusNoReadyOperation;
  const fetch_target::status_kind enter_status =
      enter_enabled
          ? fetch_target::peek_ready_operation_kind(
                target_state, fetch_target::kTargetInstance,
                fetch_target::kOperationFetchTarget, &enter)
          : fetch_target::kStatusNoReadyOperation;
  if ((restore_status != fetch_target::kStatusOk &&
       restore_status != fetch_target::kStatusNoReadyOperation) ||
      (enter_status != fetch_target::kStatusOk &&
       enter_status != fetch_target::kStatusNoReadyOperation)) {
    return kStatusQueueInvariant;
  }
  if (restore_status == fetch_target::kStatusNoReadyOperation &&
      enter_status == fetch_target::kStatusNoReadyOperation) {
    return kStatusOk;
  }
  if (restore_status == fetch_target::kStatusOk &&
      (enter_status != fetch_target::kStatusOk ||
       restore.reservation_age < enter.reservation_age)) {
    *candidate = restore;
  } else {
    *candidate = enter;
  }
  return kStatusOk;
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
    pipeline_entry_v0 &pipeline = state->pipeline[index];
    const bool restore_operation =
        pipeline.operation_packet.operation_kind ==
        fetch_target::kOperationInstanceRestoreParent;
    const bool enter_operation =
        pipeline.operation_packet.operation_kind ==
        fetch_target::kOperationFetchTarget;
    if (result_sink == NULL ||
        (restore_operation && result_sink->accept == NULL) ||
        (enter_operation && result_sink->accept_enter == NULL)) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallResultSinkBackpressure);
      emit_attempt(
          pipeline.operation_packet.owner,
          pipeline.operation_packet.target_operation_seq, service_cycle,
          static_cast<uint16_t>(captured),
          stall_attribution::kStageCapture,
          stall_attribution::kOutcomeStall,
          stall_attribution::kActionNone,
          stall_attribution::kReasonResultSinkCapacity);
      return kStatusOk;
    }
    if ((!restore_operation && !enter_operation) ||
        pipeline.operator_invocation_count != 1) {
      record_packet_failure(
          result, kFailurePointCaptureKind,
          pipeline.operation_packet,
          pipeline.operator_invocation_count, 0, 0);
      return kStatusInvalidOperationPacket;
    }
    if (restore_operation) {
      typed_instance::restore_parent_input_v0 input = {};
      input.profile_id = typed_instance::kGenRtDerivedProfileId;
      input.operation_kind = typed_instance::kRestoreParent;
      input.parent_frame = pipeline.operation_packet.parent_frame;
      if (!restore_packet_valid(pipeline.operation_packet) ||
          !typed_instance::validate_restore_parent_result(
              input, pipeline.typed_result)) {
        record_packet_failure(
            result, kFailurePointCaptureRestore,
            pipeline.operation_packet,
            pipeline.operator_invocation_count,
            pipeline.typed_result.status,
            pipeline.typed_result.result_kind);
        return kStatusInvalidOperationPacket;
      }
    } else if (!enter_packet_valid(pipeline.operation_packet) ||
               pipeline.enter_result.status !=
                   typed_instance::kStatusOk ||
               (pipeline.enter_result.result_kind !=
                    typed_instance::kEnterResultCulled &&
                pipeline.enter_result.result_kind !=
                    typed_instance::kEnterResultBlasRoot)) {
      record_packet_failure(
          result, kFailurePointCaptureEnter,
          pipeline.operation_packet,
          pipeline.operator_invocation_count,
          pipeline.enter_result.status,
          pipeline.enter_result.result_kind);
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

    result_sink_kind sink_status = kResultSinkRejected;
    completed_restore_receipt_v0 restore_receipt = {};
    completed_enter_receipt_v0 enter_receipt = {};
    if (restore_operation) {
      restore_receipt.operation_packet = pipeline.operation_packet;
      restore_receipt.typed_result = pipeline.typed_result;
      restore_receipt.issue_age = pipeline.issue_age;
      restore_receipt.issue_cycle = pipeline.issue_cycle;
      restore_receipt.result_ready_cycle = pipeline.result_ready_cycle;
      restore_receipt.capture_cycle = service_cycle;
      restore_receipt.producer_operation_seq =
          pipeline.operation_packet.target_operation_seq;
      restore_receipt.commit_epoch = commit_epoch;
      restore_receipt.target_operation_seq = target_operation_seq;
      restore_receipt.valid = 1;
      restore_receipt.operator_invocation_count =
          pipeline.operator_invocation_count;
      sink_status =
          result_sink->accept(&restore_receipt, &staged_timing,
                              result_sink->context);
    } else {
      enter_receipt.operation_packet = pipeline.operation_packet;
      enter_receipt.typed_input = pipeline.enter_input;
      enter_receipt.typed_result = pipeline.enter_result;
      enter_receipt.issue_age = pipeline.issue_age;
      enter_receipt.issue_cycle = pipeline.issue_cycle;
      enter_receipt.result_ready_cycle = pipeline.result_ready_cycle;
      enter_receipt.capture_cycle = service_cycle;
      enter_receipt.producer_operation_seq =
          pipeline.operation_packet.target_operation_seq;
      enter_receipt.commit_epoch = commit_epoch;
      enter_receipt.target_operation_seq = target_operation_seq;
      enter_receipt.valid = 1;
      enter_receipt.operator_invocation_count =
          pipeline.operator_invocation_count;
      sink_status =
          result_sink->accept_enter(&enter_receipt, &staged_timing,
                                    result_sink->context);
    }
    if (sink_status == kResultSinkBackpressure) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallResultSinkBackpressure);
      emit_attempt(
          pipeline.operation_packet.owner,
          pipeline.operation_packet.target_operation_seq, service_cycle,
          static_cast<uint16_t>(captured),
          stall_attribution::kStageCapture,
          stall_attribution::kOutcomeStall,
          stall_attribution::kActionNone,
          stall_attribution::kReasonResultSinkCapacity);
      return kStatusOk;
    }
    if (sink_status != kResultSinkAccepted) {
      record_packet_failure(
          result, kFailurePointCaptureKind,
          pipeline.operation_packet,
          pipeline.operator_invocation_count,
          restore_operation ? pipeline.typed_result.status
                            : pipeline.enter_result.status,
          restore_operation ? pipeline.typed_result.result_kind
                            : pipeline.enter_result.result_kind);
      return kStatusResultSinkRejected;
    }

    *timing_state = staged_timing;
    if (restore_operation) {
      result->completed_restores[result->captured_restore_count] =
          restore_receipt;
      ++result->captured_restore_count;
    } else {
      result->completed_enters[result->captured_enter_count] =
          enter_receipt;
      ++result->captured_enter_count;
    }
    ++result->captured_result_count;
    emit_attempt(
        pipeline.operation_packet.owner,
        pipeline.operation_packet.target_operation_seq, service_cycle,
        static_cast<uint16_t>(captured),
        stall_attribution::kStageCapture,
        stall_attribution::kOutcomeProgress,
        stall_attribution::kActionCapture,
        stall_attribution::kReasonNone);
    pipeline = pipeline_entry_v0();
    ++state->total_results_captured;
  }
  return kStatusOk;
}

status_kind issue_operations(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    const timing_driver::state_v0 &timing_state, uint64_t service_cycle,
    const result_sink_v0 *result_sink,
    cycle_result_v0 *result) {
  uint8_t issued_unit_mask = 0;
  for (unsigned issued = 0;
       issued < state->config.instance_issue_width; ++issued) {
    fetch_target::operation_packet_v0 candidate = {};
    const status_kind select_status =
        select_oldest_candidate(*target_state, result_sink, &candidate);
    if (select_status != kStatusOk) return select_status;
    if (candidate.valid == 0) return kStatusOk;
    const bool restore_operation =
        candidate.operation_kind ==
        fetch_target::kOperationInstanceRestoreParent;
    const bool enter_operation =
        candidate.operation_kind ==
        fetch_target::kOperationFetchTarget;
    if ((!restore_operation && !enter_operation) ||
        (restore_operation && !restore_packet_valid(candidate)) ||
        (enter_operation && !enter_packet_valid(candidate))) {
      record_packet_failure(
          result, kFailurePointIssuePacket, candidate, 0, 0, 0);
      return kStatusInvalidOperationPacket;
    }
    const int pipeline_index = find_free_pipeline(*state);
    if (pipeline_index < 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallPipelineFull);
      emit_attempt(
          candidate.owner, candidate.target_operation_seq, service_cycle,
          static_cast<uint16_t>(issued),
          stall_attribution::kStageIssue,
          stall_attribution::kOutcomeStall,
          stall_attribution::kActionNone,
          stall_attribution::kReasonPipelineCapacity);
      return kStatusOk;
    }
    const int unit_index =
        find_available_unit(*state, service_cycle, issued_unit_mask);
    if (unit_index < 0) {
      result->stall_mask = static_cast<uint8_t>(
          result->stall_mask | kStallUnitUnavailable);
      emit_attempt(
          candidate.owner, candidate.target_operation_seq, service_cycle,
          static_cast<uint16_t>(issued),
          stall_attribution::kStageIssue,
          stall_attribution::kOutcomeStall,
          stall_attribution::kActionNone,
          stall_attribution::kReasonUnitBusy);
      return kStatusOk;
    }
    request_owner::lane_binding_v0 request_binding = {};
    if (!live_target_matches(timing_state, candidate,
                             &request_binding)) {
      return kStatusOwnerMismatch;
    }

    typed_instance::restore_parent_result_v0 restore_result = {};
    typed_instance::enter_input_v0 enter_input = {};
    typed_instance::enter_result_v0 enter_result = {};
    if (restore_operation) {
      typed_instance::restore_parent_input_v0 input = {};
      input.profile_id = typed_instance::kGenRtDerivedProfileId;
      input.operation_kind = typed_instance::kRestoreParent;
      input.parent_frame = candidate.parent_frame;
      restore_result = typed_instance::execute_restore_parent(input);
      if (!typed_instance::validate_restore_parent_result(
              input, restore_result)) {
        return kStatusOperatorFailed;
      }
    } else {
      const result_sink_kind provider_status =
          result_sink->prepare_enter(
              &candidate, &enter_input, result_sink->context);
      if (provider_status == kResultSinkBackpressure) {
        result->stall_mask = static_cast<uint8_t>(
            result->stall_mask | kStallResultSinkBackpressure);
        emit_attempt(
            candidate.owner, candidate.target_operation_seq,
            service_cycle, static_cast<uint16_t>(issued),
            stall_attribution::kStageInputPrepare,
            stall_attribution::kOutcomeStall,
            stall_attribution::kActionNone,
            stall_attribution::kReasonInputProvider);
        return kStatusOk;
      }
      if (provider_status != kResultSinkAccepted) {
        return kStatusResultSinkRejected;
      }
      emit_attempt(
          candidate.owner, candidate.target_operation_seq, service_cycle,
          static_cast<uint16_t>(issued),
          stall_attribution::kStageInputPrepare,
          stall_attribution::kOutcomeProgress,
          stall_attribution::kActionInputPrepareAccept,
          stall_attribution::kReasonNone);
      enter_result = typed_instance::execute_enter(enter_input);
      if (enter_result.status != typed_instance::kStatusOk) {
        return kStatusOperatorFailed;
      }
    }
    fetch_target::operation_packet_v0 popped = {};
    if (fetch_target::pop_ready_operation_kind(
            target_state, fetch_target::kTargetInstance,
            static_cast<fetch_target::operation_kind>(
                candidate.operation_kind),
            true,
            &popped) != fetch_target::kStatusOk ||
        std::memcmp(&candidate, &popped, sizeof(candidate)) != 0) {
      return kStatusQueueInvariant;
    }

    pipeline_entry_v0 &pipeline = state->pipeline[pipeline_index];
    pipeline = pipeline_entry_v0();
    pipeline.operation_packet = popped;
    pipeline.enter_input = enter_input;
    pipeline.enter_result = enter_result;
    pipeline.typed_result = restore_result;
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
                            service_cycle, result_sink, result);
  if (status != kStatusOk) return status;
  result->active_pipeline_entries = active_pipeline_count(*state);
  result->ready_instance_entries = fetch_target::ready_slot_count(
      *target_state, fetch_target::kTargetInstance);
  ++state->total_service_cycles;
  state->total_issued += result->issued_count;
  state->total_stall_unit_unavailable +=
      (result->stall_mask & kStallUnitUnavailable) != 0;
  state->total_stall_pipeline_full +=
      (result->stall_mask & kStallPipelineFull) != 0;
  state->total_stall_result_sink_backpressure +=
      (result->stall_mask & kStallResultSinkBackpressure) != 0;
  state->total_issue_width_limited +=
      result->issued_count == state->config.instance_issue_width &&
      result->ready_instance_entries != 0 &&
      (result->stall_mask & kStallUnitUnavailable) == 0;
  if (result->active_pipeline_entries >
      state->max_active_pipeline_entries) {
    state->max_active_pipeline_entries =
        result->active_pipeline_entries;
  }
  if (result->ready_instance_entries >
      state->max_ready_instance_entries) {
    state->max_ready_instance_entries =
        result->ready_instance_entries;
  }
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

const char *failure_point_name(failure_point_kind failure_point) {
  switch (failure_point) {
    case kFailurePointCaptureKind: return "capture_kind";
    case kFailurePointCaptureRestore: return "capture_restore";
    case kFailurePointCaptureEnter: return "capture_enter";
    case kFailurePointIssuePacket: return "issue_packet";
    case kFailurePointNone: break;
  }
  return "none";
}

}  // namespace instance_timing
}  // namespace v04
}  // namespace rtcore
