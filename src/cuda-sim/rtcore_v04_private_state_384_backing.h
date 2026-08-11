#ifndef RTCORE_V04_PRIVATE_STATE_384_BACKING_H
#define RTCORE_V04_PRIVATE_STATE_384_BACKING_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_state_384_operand_materializer.h"
#include "rtcore_v04_private_storage_profile.h"

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace backing {

static const uint16_t kPrivateSlotCapacity = 256;
static const uint8_t kResidentWarpCapacity = 8;
static const uint8_t kLaneCapacity = 32;
static const uint32_t kFullChunkByteMask = 0xffffffffu;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidState,
  kStatusInvalidAdmission,
  kStatusCapacityExceeded,
  kStatusDuplicateWarp,
  kStatusOwnerAlias,
  kStatusStalePlan,
  kStatusOwnerMismatch,
  kStatusInvalidReadPlan,
  kStatusUninitializedChunk,
  kStatusInvalidWritePlan,
  kStatusOperationSequenceStale,
  kStatusIncompleteFirstWrite,
  kStatusInvalidMaskShrink,
  kStatusInvalidRelease,
};

struct lane_slot_v1 {
  uint8_t live;
  uint8_t reserved_zero[3];
  private_frontier::owner_binding_v0 owner;
  uint32_t valid_byte_masks[kChunkCount];
  uint32_t last_committed_operation_sequence;
  uint32_t last_transition_spill_operation_sequence;
  image_v1 image;
};

struct resident_warp_v1 {
  uint8_t live;
  uint8_t resident_warp_slot;
  uint16_t reserved_zero;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
};

struct state_v1 {
  uint8_t initialized;
  uint8_t reserved_zero[3];
  uint32_t owner_hw_sid;
  uint64_t mutation_epoch;
  resident_warp_v1 resident_warps[kResidentWarpCapacity];
  lane_slot_v1 slots[kPrivateSlotCapacity];
};

struct admission_lane_v1 {
  uint8_t valid;
  uint8_t lane_id;
  uint16_t private_slot_id;
  private_frontier::owner_binding_v0 owner;
  uint32_t valid_byte_masks[kChunkCount];
  image_v1 image;
};

struct new_warp_plan_v1 {
  uint8_t valid;
  uint8_t resident_warp_slot;
  uint8_t active_lane_count;
  uint8_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
  uint64_t expected_mutation_epoch;
  admission_lane_v1 lanes[kLaneCapacity];
};

struct mask_shrink_plan_v1 {
  uint8_t valid;
  uint8_t resident_warp_slot;
  uint16_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t previous_warp_uid;
  uint32_t next_warp_uid;
  uint32_t warp_id;
  uint32_t previous_active_mask;
  uint32_t next_active_mask;
  uint32_t release_mask;
  uint64_t expected_mutation_epoch;
  private_frontier::owner_binding_v0 released_owners[kLaneCapacity];
};

struct release_warp_plan_v1 {
  uint8_t valid;
  uint8_t resident_warp_slot;
  uint16_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
  uint64_t expected_mutation_epoch;
  private_frontier::owner_binding_v0 released_owners[kLaneCapacity];
};

void initialize(state_v1 *state, uint32_t owner_hw_sid);

status_kind prepare_new_warp(
    const state_v1 &state,
    const private_storage::admission_candidate_plan_v0 &candidate,
    uint32_t warp_uid, uint32_t warp_id, new_warp_plan_v1 *plan);

status_kind commit_new_warp(state_v1 *state,
                            const new_warp_plan_v1 &plan);

status_kind prepare_read_responses(
    const state_v1 &state,
    const operand_plan::read_plan_v1 &read_plan,
    const operand_materializer::operation_identity_v1 &identity,
    operand_materializer::chunk_response_v1
        responses[operand_materializer::kMaxOperationReadChunks],
    uint8_t *response_count);

status_kind prepare_read_response(
    const state_v1 &state,
    const operand_plan::read_plan_v1 &read_plan,
    const operand_materializer::operation_identity_v1 &identity,
    uint8_t response_index,
    operand_materializer::chunk_response_v1 *response);

status_kind apply_sparse_deltas(
    state_v1 *state,
    const operand_materializer::operation_identity_v1 &identity,
    const operand_plan::write_request_v1 &request,
    const operand_plan::chunk_delta_v1 *deltas, size_t delta_count);

status_kind prepare_mask_shrink(
    const state_v1 &state, uint8_t resident_warp_slot,
    uint32_t previous_warp_uid, uint32_t next_warp_uid,
    uint32_t warp_id, uint32_t next_active_mask,
    mask_shrink_plan_v1 *plan);

status_kind commit_mask_shrink(state_v1 *state,
                               const mask_shrink_plan_v1 &plan);

status_kind prepare_release_warp(
    const state_v1 &state, uint8_t resident_warp_slot,
    uint32_t warp_uid, uint32_t warp_id,
    release_warp_plan_v1 *plan);

status_kind commit_release_warp(state_v1 *state,
                                const release_warp_plan_v1 &plan);

const lane_slot_v1 *find_live_lane(
    const state_v1 &state,
    const private_frontier::owner_binding_v0 &owner);

const char *status_name(status_kind status);

static_assert(kPrivateSlotCapacity == 8 * 32,
              "384B backing must cover eight fully divergent warps");
static_assert(sizeof(image_v1) == 384,
              "384B backing image size changed");

}  // namespace backing
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore

#endif
