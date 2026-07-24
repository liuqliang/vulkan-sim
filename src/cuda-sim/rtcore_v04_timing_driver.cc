#include "rtcore_v04_timing_driver.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace timing_driver {
namespace {

uint32_t lane_bit(uint32_t lane) {
  return lane < request_owner::kLaneCapacity ? uint32_t{1} << lane : 0;
}

bool bindings_equal(const request_owner::lane_binding_v0 &lhs,
                    const request_owner::lane_binding_v0 &rhs) {
  return lhs.packed_request_key == rhs.packed_request_key &&
         lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.request_control_slot == rhs.request_control_slot &&
         lhs.request_generation == rhs.request_generation &&
         lhs.private_slot_id == rhs.private_slot_id &&
         lhs.resident_warp_slot == rhs.resident_warp_slot &&
         lhs.lane_id == rhs.lane_id;
}

bool validate_owner_binding(const state_v0 &state,
                            const request_owner::lane_binding_v0 &owner) {
  if (!state.initialized ||
      owner.request_control_slot >= request_owner::kRequestControlCapacity ||
      owner.resident_warp_slot >= request_owner::kResidentWarpCapacity) {
    return false;
  }
  const request_owner::resident_slot_state_v0 &resident =
      state.request_control.resident_slots[owner.resident_warp_slot];
  return resident.live &&
         request_owner::validate_live_binding(
             state.request_control, owner, resident.identity.warp_uid,
             resident.identity.active_mask);
}

lane_control_state_v0 *find_lane_control(
    state_v0 *state, const request_owner::lane_binding_v0 &owner) {
  if (state == NULL ||
      owner.request_control_slot >= request_owner::kRequestControlCapacity) {
    return NULL;
  }
  lane_control_state_v0 &lane =
      state->lane_controls[owner.request_control_slot];
  return lane.live && bindings_equal(lane.owner, owner) ? &lane : NULL;
}

bool lane_is_quiescent(const lane_control_state_v0 &lane) {
  return lane.live && lane.live_target_operation_seq == 0 &&
         lane.live_commit_producer_operation_seq == 0 &&
         lane.live_commit_epoch == 0 &&
         lane.pending_recovery_operation_seq == 0 &&
         lane.live_memory_transaction_count == 0 &&
         lane.live_commit_memory_transaction_count == 0;
}

status_kind validate_plan_epoch(const state_v0 &state, bool plan_valid,
                                uint64_t expected_mutation_epoch) {
  if (!state.initialized || !plan_valid) return kStatusInvalidArgument;
  return state.mutation_epoch == expected_mutation_epoch
             ? kStatusOk
             : kStatusStalePlan;
}

status_kind map_request_owner_status(request_owner::status_kind status) {
  switch (status) {
    case request_owner::kStatusOk: return kStatusOk;
    case request_owner::kStatusInvalidArgument:
      return kStatusInvalidArgument;
    case request_owner::kStatusInvalidActiveMask:
      return kStatusInvalidActiveMask;
    case request_owner::kStatusDuplicateWarp:
      return kStatusDuplicateWarp;
    case request_owner::kStatusResidentCapacityExceeded:
      return kStatusResidentCapacityExceeded;
    case request_owner::kStatusRequestCapacityExceeded:
      return kStatusRequestCapacityExceeded;
    case request_owner::kStatusGenerationExhausted:
      return kStatusRequestGenerationExhausted;
    case request_owner::kStatusStalePlan:
      return kStatusStalePlan;
    case request_owner::kStatusOwnerMismatch:
      return kStatusOwnerMismatch;
    case request_owner::kStatusInvalidKey:
      return kStatusInvalidRequestKey;
    case request_owner::kStatusInvalidMaskShrink:
      return kStatusInvalidMaskShrink;
  }
  return kStatusOwnerMismatch;
}

}  // namespace

void initialize(state_v0 *state) {
  if (state == NULL) return;
  std::memset(state, 0, sizeof(*state));
  request_owner::initialize_allocator(&state->request_control);
  state->result_commit_control.next_commit_epoch = 1;
  state->mutation_epoch = 1;
  state->initialized = true;
}

request_owner::allocator_state_v0 *request_control(state_v0 *state) {
  return state != NULL && state->initialized ? &state->request_control : NULL;
}

const request_owner::allocator_state_v0 *request_control(
    const state_v0 &state) {
  return state.initialized ? &state.request_control : NULL;
}

status_kind prepare_new_submit(
    const state_v0 &state, const request_owner::warp_identity_v0 &identity,
    new_submit_plan_v0 *plan) {
  if (!state.initialized || plan == NULL) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  const request_owner::status_kind owner_status =
      request_owner::prepare_new_warp(state.request_control, identity,
                                      &plan->owner_plan);
  if (owner_status != request_owner::kStatusOk)
    return map_request_owner_status(owner_status);
  plan->valid = true;
  plan->expected_mutation_epoch = state.mutation_epoch;
  return kStatusOk;
}

status_kind commit_new_submit(state_v0 *state,
                              const new_submit_plan_v0 &plan) {
  if (state == NULL) return kStatusInvalidArgument;
  const status_kind epoch_status =
      validate_plan_epoch(*state, plan.valid, plan.expected_mutation_epoch);
  if (epoch_status != kStatusOk) return epoch_status;

  state_v0 staged = *state;
  const request_owner::status_kind owner_status =
      request_owner::commit_new_warp(&staged.request_control,
                                     plan.owner_plan);
  if (owner_status != request_owner::kStatusOk)
    return map_request_owner_status(owner_status);
  for (uint32_t lane = 0; lane < request_owner::kLaneCapacity; ++lane) {
    if ((plan.owner_plan.identity.active_mask & lane_bit(lane)) == 0) continue;
    const request_owner::lane_binding_v0 &owner =
        plan.owner_plan.lane_bindings[lane];
    lane_control_state_v0 &control =
        staged.lane_controls[owner.request_control_slot];
    if (control.live) return kStatusOwnerMismatch;
    std::memset(&control, 0, sizeof(control));
    control.live = true;
    control.owner = owner;
    control.next_target_operation_seq = 1;
  }
  ++staged.mutation_epoch;
  *state = staged;
  return kStatusOk;
}

status_kind prepare_resubmit(
    const state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t previous_warp_uid,
    uint32_t next_warp_uid, uint32_t warp_id, uint32_t next_active_mask,
    resubmit_plan_v0 *plan) {
  if (!state.initialized || plan == NULL) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  request_owner::mask_shrink_plan_v0 owner_plan = {};
  const request_owner::status_kind owner_status =
      request_owner::prepare_mask_shrink(
          state.request_control, resident_warp_slot, owner_hw_sid,
          previous_warp_uid, next_warp_uid, warp_id, next_active_mask,
          &owner_plan);
  if (owner_status != request_owner::kStatusOk)
    return map_request_owner_status(owner_status);
  for (uint32_t lane = 0; lane < request_owner::kLaneCapacity; ++lane) {
    if ((owner_plan.previous_active_mask & lane_bit(lane)) == 0) continue;
    const lane_control_state_v0 *control =
        find_live_lane_control(state, owner_plan.lane_bindings[lane]);
    if (control == NULL) return kStatusOwnerMismatch;
    if (!lane_is_quiescent(*control)) return kStatusOperationInFlight;
  }
  plan->valid = true;
  plan->expected_mutation_epoch = state.mutation_epoch;
  plan->owner_plan = owner_plan;
  return kStatusOk;
}

status_kind commit_resubmit(state_v0 *state,
                            const resubmit_plan_v0 &plan) {
  if (state == NULL) return kStatusInvalidArgument;
  const status_kind epoch_status =
      validate_plan_epoch(*state, plan.valid, plan.expected_mutation_epoch);
  if (epoch_status != kStatusOk) return epoch_status;

  state_v0 staged = *state;
  const request_owner::status_kind owner_status =
      request_owner::commit_mask_shrink(&staged.request_control,
                                        plan.owner_plan);
  if (owner_status != request_owner::kStatusOk)
    return map_request_owner_status(owner_status);
  for (uint32_t lane = 0; lane < request_owner::kLaneCapacity; ++lane) {
    if ((plan.owner_plan.release_mask & lane_bit(lane)) == 0) continue;
    const request_owner::lane_binding_v0 &owner =
        plan.owner_plan.lane_bindings[lane];
    lane_control_state_v0 *control = find_lane_control(&staged, owner);
    if (control == NULL || !lane_is_quiescent(*control)) {
      return kStatusOwnerMismatch;
    }
    std::memset(control, 0, sizeof(*control));
  }
  ++staged.mutation_epoch;
  *state = staged;
  return kStatusOk;
}

status_kind prepare_retire(
    const state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t current_warp_uid, uint32_t warp_id,
    retire_plan_v0 *plan) {
  if (!state.initialized || plan == NULL) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  request_owner::release_warp_plan_v0 owner_plan = {};
  const request_owner::status_kind owner_status =
      request_owner::prepare_release_warp(
          state.request_control, resident_warp_slot, owner_hw_sid,
          current_warp_uid, warp_id, &owner_plan);
  if (owner_status != request_owner::kStatusOk)
    return map_request_owner_status(owner_status);
  for (uint32_t lane = 0; lane < request_owner::kLaneCapacity; ++lane) {
    if ((owner_plan.identity.active_mask & lane_bit(lane)) == 0) continue;
    const lane_control_state_v0 *control =
        find_live_lane_control(state, owner_plan.lane_bindings[lane]);
    if (control == NULL) return kStatusOwnerMismatch;
    if (!lane_is_quiescent(*control)) return kStatusRetireNotQuiescent;
  }
  plan->valid = true;
  plan->expected_mutation_epoch = state.mutation_epoch;
  plan->owner_plan = owner_plan;
  return kStatusOk;
}

status_kind commit_retire(state_v0 *state, const retire_plan_v0 &plan) {
  if (state == NULL) return kStatusInvalidArgument;
  const status_kind epoch_status =
      validate_plan_epoch(*state, plan.valid, plan.expected_mutation_epoch);
  if (epoch_status != kStatusOk) return epoch_status;

  state_v0 staged = *state;
  const request_owner::status_kind owner_status =
      request_owner::commit_release_warp(&staged.request_control,
                                         plan.owner_plan);
  if (owner_status != request_owner::kStatusOk)
    return map_request_owner_status(owner_status);
  for (uint32_t lane = 0; lane < request_owner::kLaneCapacity; ++lane) {
    if ((plan.owner_plan.identity.active_mask & lane_bit(lane)) == 0) continue;
    lane_control_state_v0 *control =
        find_lane_control(&staged, plan.owner_plan.lane_bindings[lane]);
    if (control == NULL || !lane_is_quiescent(*control)) {
      return kStatusOwnerMismatch;
    }
    std::memset(control, 0, sizeof(*control));
  }
  ++staged.mutation_epoch;
  *state = staged;
  return kStatusOk;
}

status_kind allocate_target_operation(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t *target_operation_seq) {
  if (state == NULL || target_operation_seq == NULL ||
      !state->initialized) {
    return kStatusInvalidArgument;
  }
  lane_control_state_v0 *control = find_lane_control(state, owner);
  if (control == NULL || !validate_owner_binding(*state, owner)) {
    return kStatusOwnerMismatch;
  }
  if (control->live_target_operation_seq != 0 ||
      control->live_commit_producer_operation_seq != 0 ||
      control->live_commit_epoch != 0 ||
      control->pending_recovery_operation_seq != 0 ||
      control->live_memory_transaction_count != 0 ||
      control->live_commit_memory_transaction_count != 0) {
    return kStatusOperationInFlight;
  }
  if (control->next_target_operation_seq == 0) {
    return kStatusOperationSequenceExhausted;
  }
  *target_operation_seq = control->next_target_operation_seq;
  control->live_target_operation_seq = *target_operation_seq;
  ++control->next_target_operation_seq;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind begin_result_commit(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t *commit_epoch) {
  if (state == NULL || commit_epoch == NULL || producer_operation_seq == 0 ||
      !state->initialized) {
    return kStatusInvalidArgument;
  }
  lane_control_state_v0 *control = find_lane_control(state, owner);
  if (control == NULL || !validate_owner_binding(*state, owner)) {
    return kStatusOwnerMismatch;
  }
  if (control->live_target_operation_seq != producer_operation_seq ||
      control->live_commit_epoch != 0 ||
      control->live_commit_producer_operation_seq != 0 ||
      control->pending_recovery_operation_seq != 0) {
    return kStatusCommitMismatch;
  }
  if (control->live_memory_transaction_count != 0) {
    return kStatusOperationInFlight;
  }
  if (control->live_commit_memory_transaction_count != 0) {
    return kStatusCommitMismatch;
  }
  if (state->result_commit_control.next_commit_epoch == 0)
    return kStatusCommitEpochExhausted;
  *commit_epoch = state->result_commit_control.next_commit_epoch;
  ++state->result_commit_control.next_commit_epoch;
  control->live_target_operation_seq = 0;
  control->live_commit_producer_operation_seq = producer_operation_seq;
  control->live_commit_epoch = *commit_epoch;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind allocate_commit_successor_operation(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t *target_operation_seq) {
  if (state == NULL || target_operation_seq == NULL ||
      producer_operation_seq == 0 || commit_epoch == 0 ||
      !state->initialized) {
    return kStatusInvalidArgument;
  }
  lane_control_state_v0 *control = find_lane_control(state, owner);
  if (control == NULL || !validate_owner_binding(*state, owner)) {
    return kStatusOwnerMismatch;
  }
  if (control->live_commit_producer_operation_seq !=
          producer_operation_seq ||
      control->live_commit_epoch != commit_epoch) {
    return kStatusCommitMismatch;
  }
  if (control->live_target_operation_seq != 0 ||
      control->pending_recovery_operation_seq != 0 ||
      control->live_memory_transaction_count != 0) {
    return kStatusOperationInFlight;
  }
  if (control->next_target_operation_seq == 0) {
    return kStatusOperationSequenceExhausted;
  }
  if (control->next_target_operation_seq == producer_operation_seq) {
    return kStatusCommitMismatch;
  }
  *target_operation_seq = control->next_target_operation_seq;
  control->live_target_operation_seq = *target_operation_seq;
  ++control->next_target_operation_seq;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind mark_commit_successor_pending_recovery(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq) {
  if (state == NULL || producer_operation_seq == 0 || commit_epoch == 0 ||
      target_operation_seq == 0 || !state->initialized) {
    return kStatusInvalidArgument;
  }
  lane_control_state_v0 *control = find_lane_control(state, owner);
  if (control == NULL || !validate_owner_binding(*state, owner)) {
    return kStatusOwnerMismatch;
  }
  if (control->live_commit_producer_operation_seq !=
          producer_operation_seq ||
      control->live_commit_epoch != commit_epoch ||
      control->live_target_operation_seq != target_operation_seq ||
      control->pending_recovery_operation_seq != 0 ||
      control->live_memory_transaction_count != 0) {
    return kStatusCommitMismatch;
  }
  control->pending_recovery_operation_seq = target_operation_seq;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind begin_memory_transaction(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t operation_seq) {
  if (state == NULL || operation_seq == 0 || !state->initialized) {
    return kStatusInvalidArgument;
  }
  lane_control_state_v0 *control = find_lane_control(state, owner);
  if (control == NULL || !validate_owner_binding(*state, owner)) {
    return kStatusOwnerMismatch;
  }
  const bool target_transaction =
      control->live_target_operation_seq == operation_seq;
  const bool commit_transaction =
      control->live_commit_producer_operation_seq == operation_seq;
  if (target_transaction == commit_transaction) {
    return kStatusMemoryTransactionMismatch;
  }
  if (target_transaction &&
      control->pending_recovery_operation_seq == operation_seq) {
    return kStatusOperationInFlight;
  }
  uint16_t &transaction_count =
      target_transaction ? control->live_memory_transaction_count
                         : control->live_commit_memory_transaction_count;
  if (transaction_count == 0xffffu) {
    return kStatusMemoryTransactionOverflow;
  }
  ++transaction_count;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind complete_memory_transaction(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t operation_seq) {
  if (state == NULL || operation_seq == 0 || !state->initialized) {
    return kStatusInvalidArgument;
  }
  lane_control_state_v0 *control = find_lane_control(state, owner);
  if (control == NULL || !validate_owner_binding(*state, owner)) {
    return kStatusOwnerMismatch;
  }
  const bool target_transaction =
      control->live_target_operation_seq == operation_seq;
  const bool commit_transaction =
      control->live_commit_producer_operation_seq == operation_seq;
  if (target_transaction == commit_transaction) {
    return kStatusMemoryTransactionMismatch;
  }
  uint16_t &transaction_count =
      target_transaction ? control->live_memory_transaction_count
                         : control->live_commit_memory_transaction_count;
  if (transaction_count == 0) {
    return kStatusMemoryTransactionMismatch;
  }
  --transaction_count;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind complete_result_commit(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch) {
  if (state == NULL || producer_operation_seq == 0 || commit_epoch == 0 ||
      !state->initialized) {
    return kStatusInvalidArgument;
  }
  lane_control_state_v0 *control = find_lane_control(state, owner);
  if (control == NULL || !validate_owner_binding(*state, owner)) {
    return kStatusOwnerMismatch;
  }
  if (control->live_commit_producer_operation_seq != producer_operation_seq ||
      control->live_commit_epoch != commit_epoch ||
      control->live_commit_memory_transaction_count != 0) {
    return kStatusCommitMismatch;
  }
  control->live_commit_producer_operation_seq = 0;
  control->live_commit_epoch = 0;
  ++state->mutation_epoch;
  return kStatusOk;
}

const lane_control_state_v0 *find_live_lane_control(
    const state_v0 &state, const request_owner::lane_binding_v0 &owner) {
  if (!validate_owner_binding(state, owner)) return NULL;
  const lane_control_state_v0 &control =
      state.lane_controls[owner.request_control_slot];
  return control.live && bindings_equal(control.owner, owner) ? &control
                                                              : NULL;
}

uint32_t active_resident_count(const state_v0 &state) {
  if (!state.initialized) return 0;
  uint32_t count = 0;
  for (uint32_t slot = 0; slot < request_owner::kResidentWarpCapacity;
       ++slot) {
    count += state.request_control.resident_slots[slot].live ? 1u : 0u;
  }
  return count;
}

uint32_t active_lane_count(const state_v0 &state) {
  if (!state.initialized) return 0;
  uint32_t count = 0;
  for (uint32_t slot = 0; slot < request_owner::kRequestControlCapacity;
       ++slot) {
    count += state.lane_controls[slot].live ? 1u : 0u;
  }
  return count;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk: return "ok";
    case kStatusInvalidArgument: return "invalid_argument";
    case kStatusStalePlan: return "stale_plan";
    case kStatusOwnerMismatch: return "owner_mismatch";
    case kStatusInvalidActiveMask: return "invalid_active_mask";
    case kStatusDuplicateWarp: return "duplicate_warp";
    case kStatusResidentCapacityExceeded:
      return "resident_capacity_exceeded";
    case kStatusRequestCapacityExceeded:
      return "request_capacity_exceeded";
    case kStatusRequestGenerationExhausted:
      return "request_generation_exhausted";
    case kStatusInvalidRequestKey: return "invalid_request_key";
    case kStatusInvalidMaskShrink: return "invalid_mask_shrink";
    case kStatusOperationInFlight: return "operation_in_flight";
    case kStatusOperationSequenceExhausted:
      return "operation_sequence_exhausted";
    case kStatusCommitEpochExhausted: return "commit_epoch_exhausted";
    case kStatusCommitMismatch: return "commit_mismatch";
    case kStatusMemoryTransactionOverflow:
      return "memory_transaction_overflow";
    case kStatusMemoryTransactionMismatch:
      return "memory_transaction_mismatch";
    case kStatusRetireNotQuiescent: return "retire_not_quiescent";
  }
  return "unknown";
}

}  // namespace timing_driver
}  // namespace v04
}  // namespace rtcore
