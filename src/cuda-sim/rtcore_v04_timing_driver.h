#ifndef RTCORE_V04_TIMING_DRIVER_H
#define RTCORE_V04_TIMING_DRIVER_H

#include <cstdint>

#include "rtcore_v04_request_owner_binding.h"

namespace rtcore {
namespace v04 {
namespace timing_driver {

static const uint32_t kOperationSequenceMax = 0xffffffffu;
static const uint32_t kCommitEpochMax = 0xffffffffu;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusStalePlan,
  kStatusOwnerMismatch,
  kStatusInvalidActiveMask,
  kStatusDuplicateWarp,
  kStatusResidentCapacityExceeded,
  kStatusRequestCapacityExceeded,
  kStatusRequestGenerationExhausted,
  kStatusInvalidRequestKey,
  kStatusInvalidMaskShrink,
  kStatusOperationInFlight,
  kStatusOperationSequenceExhausted,
  kStatusCommitEpochExhausted,
  kStatusCommitMismatch,
  kStatusMemoryTransactionOverflow,
  kStatusMemoryTransactionMismatch,
  kStatusRetireNotQuiescent,
};

struct result_commit_control_state_v0 {
  uint32_t next_commit_epoch;
};

struct lane_control_state_v0 {
  bool live;
  uint8_t reserved_zero[3];
  request_owner::lane_binding_v0 owner;
  uint32_t next_target_operation_seq;
  uint32_t live_target_operation_seq;
  uint32_t live_commit_producer_operation_seq;
  uint32_t live_commit_epoch;
  uint16_t live_memory_transaction_count;
  uint16_t reserved_one;
};

struct state_v0 {
  bool initialized;
  uint8_t reserved_zero[3];
  result_commit_control_state_v0 result_commit_control;
  uint64_t mutation_epoch;
  request_owner::allocator_state_v0 request_control;
  lane_control_state_v0 lane_controls[
      request_owner::kRequestControlCapacity];
};

struct new_submit_plan_v0 {
  bool valid;
  uint8_t reserved_zero[7];
  uint64_t expected_mutation_epoch;
  request_owner::new_warp_plan_v0 owner_plan;
};

struct resubmit_plan_v0 {
  bool valid;
  uint8_t reserved_zero[7];
  uint64_t expected_mutation_epoch;
  request_owner::mask_shrink_plan_v0 owner_plan;
};

struct retire_plan_v0 {
  bool valid;
  uint8_t reserved_zero[7];
  uint64_t expected_mutation_epoch;
  request_owner::release_warp_plan_v0 owner_plan;
};

void initialize(state_v0 *state);

request_owner::allocator_state_v0 *request_control(state_v0 *state);
const request_owner::allocator_state_v0 *request_control(
    const state_v0 &state);

status_kind prepare_new_submit(
    const state_v0 &state, const request_owner::warp_identity_v0 &identity,
    new_submit_plan_v0 *plan);
status_kind commit_new_submit(state_v0 *state,
                              const new_submit_plan_v0 &plan);

status_kind prepare_resubmit(
    const state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t previous_warp_uid,
    uint32_t next_warp_uid, uint32_t warp_id, uint32_t next_active_mask,
    resubmit_plan_v0 *plan);
status_kind commit_resubmit(state_v0 *state,
                            const resubmit_plan_v0 &plan);

status_kind prepare_retire(
    const state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t current_warp_uid, uint32_t warp_id,
    retire_plan_v0 *plan);
status_kind commit_retire(state_v0 *state, const retire_plan_v0 &plan);

status_kind allocate_target_operation(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t *target_operation_seq);
status_kind begin_result_commit(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t *commit_epoch);
status_kind begin_memory_transaction(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t operation_seq);
status_kind complete_memory_transaction(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t operation_seq);
status_kind complete_result_commit(
    state_v0 *state, const request_owner::lane_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch);

const lane_control_state_v0 *find_live_lane_control(
    const state_v0 &state, const request_owner::lane_binding_v0 &owner);
uint32_t active_resident_count(const state_v0 &state);
uint32_t active_lane_count(const state_v0 &state);

const char *status_name(status_kind status);

}  // namespace timing_driver
}  // namespace v04
}  // namespace rtcore

#endif
