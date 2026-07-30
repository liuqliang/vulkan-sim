#ifndef RTCORE_V04_SHORT_STACK_SHARED_CODEC_H
#define RTCORE_V04_SHORT_STACK_SHARED_CODEC_H

#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_short_stack_replay.h"

namespace rtcore {
namespace v04 {
namespace short_stack_shared {

static const uint32_t kMetadataTag = 0x31525353u;  // "SSR1"
static const uint32_t kEncodedEntryCount = short_stack::kLogicalCapacity;
static const uint32_t kEncodedEntriesBytes =
    kEncodedEntryCount * private_frontier::kFrontierEntryBytes;
static const uint8_t kStateAccessChunkCount = 6;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOwner,
  kStatusOwnerMismatch,
  kStatusInvalidRegion,
  kStatusAddressOverflow,
  kStatusInvalidMetadata,
  kStatusInvalidState,
  kStatusPlanCapacityExceeded,
};

struct metadata_image_v0 {
  uint32_t format_tag;
  uint32_t tlas_build_generation;
  uint32_t blas_build_generation;
  uint8_t stack_count;
  uint8_t stack_top_ptr;
  uint8_t cross_as;
  uint8_t lost;
  uint8_t active_domain;
  uint8_t recovery_target_inflight;
  uint8_t reserved_zero[6];
};

struct persistent_state_v0 {
  uint32_t tlas_build_generation;
  uint32_t blas_build_generation;
  short_stack::state_v0 stack;
  uint8_t recovery_target_inflight;
  uint8_t reserved_zero[7];
};

static_assert(sizeof(metadata_image_v0) ==
                  private_frontier::kFrontierMetadataBytes,
              "short-stack metadata must reuse the 24-byte metadata field");
static_assert(sizeof(short_stack::entry_v0) ==
                  private_frontier::kFrontierEntryBytes,
              "short-stack entry must reuse the 16-byte entry envelope");
static_assert(kEncodedEntriesBytes == 96,
              "six short-stack entries must occupy 96 bytes");
static_assert(sizeof(persistent_state_v0) == 120,
              "short-stack persistent state layout changed");

bool validate_persistent_state(const persistent_state_v0 &state);

status_kind apply_persistent_state(
    private_frontier::shadow_slot_v0 *slot,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::region_binding_v0 &region,
    const persistent_state_v0 &state,
    private_frontier::access_plan_v0 *write_plan);

status_kind decode_persistent_state(
    const private_frontier::shadow_slot_v0 &slot,
    const private_frontier::owner_binding_v0 &owner,
    persistent_state_v0 *state);

status_kind build_persistent_state_read_plan(
    const private_frontier::shadow_slot_v0 &slot,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::region_binding_v0 &region,
    private_frontier::access_plan_v0 *read_plan);

const char *status_name(status_kind status);

}  // namespace short_stack_shared
}  // namespace v04
}  // namespace rtcore

#endif
