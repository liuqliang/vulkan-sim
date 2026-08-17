#ifndef RTCORE_V04_REQUEST_OWNER_BINDING_H
#define RTCORE_V04_REQUEST_OWNER_BINDING_H

#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"

namespace rtcore {
namespace v04 {
namespace request_owner {

static const uint32_t kResidentWarpCapacity = 8;
static const uint32_t kRequestControlCapacity = 256;
static const uint32_t kLaneCapacity = 32;
static const uint32_t kRequestGenerationMax = 0xffff;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidActiveMask,
  kStatusDuplicateWarp,
  kStatusResidentCapacityExceeded,
  kStatusRequestCapacityExceeded,
  kStatusGenerationExhausted,
  kStatusStalePlan,
  kStatusOwnerMismatch,
  kStatusInvalidKey,
  kStatusInvalidMaskShrink,
};

struct internal_request_key_fields_v0 {
  uint8_t resident_warp_slot;
  uint8_t lane_id;
  uint16_t request_control_slot;
  uint16_t request_generation;
  uint16_t reserved_zero;
};

struct warp_identity_v0 {
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
};

struct lane_binding_v0 {
  uint32_t packed_request_key;
  uint32_t owner_hw_sid;
  uint16_t request_control_slot;
  uint16_t request_generation;
  uint16_t private_slot_id;
  uint8_t resident_warp_slot;
  uint8_t lane_id;
};

struct resident_slot_state_v0 {
  bool live;
  warp_identity_v0 identity;
  uint16_t request_control_slots[kLaneCapacity];
  uint16_t request_generations[kLaneCapacity];
};

struct request_slot_state_v0 {
  bool live;
  bool generation_exhausted;
  uint8_t resident_warp_slot;
  uint8_t lane_id;
  uint16_t last_generation;
};

struct allocator_state_v0 {
  bool initialized;
  uint8_t next_resident_slot;
  uint16_t next_request_slot;
  uint64_t mutation_epoch;
  resident_slot_state_v0 resident_slots[kResidentWarpCapacity];
  request_slot_state_v0 request_slots[kRequestControlCapacity];
};

struct new_warp_plan_v0 {
  bool valid;
  uint64_t expected_mutation_epoch;
  warp_identity_v0 identity;
  uint8_t resident_warp_slot;
  uint8_t reserved_zero[3];
  lane_binding_v0 lane_bindings[kLaneCapacity];
};

struct mask_shrink_plan_v0 {
  bool valid;
  uint64_t expected_mutation_epoch;
  uint32_t owner_hw_sid;
  uint32_t previous_warp_uid;
  uint32_t next_warp_uid;
  uint32_t warp_id;
  uint32_t previous_active_mask;
  uint32_t next_active_mask;
  uint32_t release_mask;
  uint8_t resident_warp_slot;
  uint8_t reserved_zero[3];
  lane_binding_v0 lane_bindings[kLaneCapacity];
};

struct release_warp_plan_v0 {
  bool valid;
  uint64_t expected_mutation_epoch;
  warp_identity_v0 identity;
  uint8_t resident_warp_slot;
  uint8_t reserved_zero[3];
  lane_binding_v0 lane_bindings[kLaneCapacity];
};

void initialize_allocator(allocator_state_v0 *state);

status_kind pack_internal_request_key(
    const internal_request_key_fields_v0 &fields, uint32_t *packed_key);

status_kind unpack_internal_request_key(
    uint32_t packed_key, internal_request_key_fields_v0 *fields);

bool validate_private_frontier_owner_identity(
    const private_frontier::owner_binding_v0 &owner);

status_kind prepare_new_warp(const allocator_state_v0 &state,
                             const warp_identity_v0 &identity,
                             new_warp_plan_v0 *plan);

status_kind commit_new_warp(allocator_state_v0 *state,
                            const new_warp_plan_v0 &plan);

// A recursive TraceRay keeps its caller's logical RT owner live while a
// distinct child traversal uses another RT-private slot.  The physical shader
// warp is intentionally the same, so this explicit API is the only admission
// path that permits an additional live owner with the same (SM, warp_id).
status_kind prepare_nested_warp(const allocator_state_v0 &state,
                                const warp_identity_v0 &identity,
                                new_warp_plan_v0 *plan);

status_kind commit_nested_warp(allocator_state_v0 *state,
                               const new_warp_plan_v0 &plan);

status_kind prepare_mask_shrink(
    const allocator_state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t previous_warp_uid,
    uint32_t next_warp_uid, uint32_t warp_id, uint32_t next_active_mask,
    mask_shrink_plan_v0 *plan);

status_kind commit_mask_shrink(allocator_state_v0 *state,
                               const mask_shrink_plan_v0 &plan);

status_kind prepare_release_warp(
    const allocator_state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t current_warp_uid, uint32_t warp_id,
    release_warp_plan_v0 *plan);

status_kind commit_release_warp(allocator_state_v0 *state,
                                const release_warp_plan_v0 &plan);

bool validate_live_binding(const allocator_state_v0 &state,
                           const lane_binding_v0 &binding,
                           uint32_t current_warp_uid,
                           uint32_t current_active_mask);

private_frontier::owner_binding_v0 make_private_frontier_owner(
    const lane_binding_v0 &binding);

const char *status_name(status_kind status);

}  // namespace request_owner
}  // namespace v04
}  // namespace rtcore

#endif
