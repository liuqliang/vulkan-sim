#ifndef RTCORE_V04_PRIVATE_STORAGE_PROFILE_H
#define RTCORE_V04_PRIVATE_STORAGE_PROFILE_H

#include <cstdint>

#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_private_state_384_codec.h"
#include "rtcore_v04_typed_node_kernel.h"

namespace rtcore {
namespace v04 {
namespace private_storage {

static const char kSelectorEnvironmentName[] =
    "VULKAN_SIM_RTCORE_V04_PRIVATE_STORAGE_PROFILE";

enum profile_kind : uint8_t {
  kProfileLegacyShared832 = 0,
  kProfileCompressedShared384 = 1,
  kProfileGlobal384 = 2,
};

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidSelector,
  kStatusUnsupportedProfile,
  kStatusLegacyAdmissionRejected,
  kStatusCodecRejected,
};

struct lane_launch_candidate_v0 {
  bool valid;
  uint8_t valid_chunk_mask;
  uint8_t reserved_zero[6];
  private_frontier::owner_binding_v0 owner;
  private_state_384::sparse_write_plan_v1 sparse_writes;
};

struct admission_candidate_plan_v0 {
  bool valid;
  bool compressed_candidate_valid;
  uint8_t profile;
  uint8_t active_lane_count;
  uint32_t active_mask;
  private_shared::new_warp_plan_v0 legacy_live_plan;
  lane_launch_candidate_v0 lanes[private_shared::kLaneCapacity];
};

status_kind parse_profile(const char *value, profile_kind *profile);

const char *profile_name(profile_kind profile);

status_kind prepare_new_warp_from_selector(
    const private_shared::backing_state_v0 &state,
    uint32_t warp_uid, uint32_t warp_id, uint32_t active_mask,
    const private_frontier::owner_binding_v0
        owners[private_shared::kLaneCapacity],
    const private_frontier::root_private_operands_v0
        root_operands[private_shared::kLaneCapacity],
    const typed_node::ray_policy_v0
        ray_policies[private_shared::kLaneCapacity],
    const uint32_t
        root_build_generations[private_shared::kLaneCapacity],
    const char *selector_value, bool short_stack_enabled,
    private_shared::status_kind *legacy_failure_status,
    admission_candidate_plan_v0 *plan);

const char *status_name(status_kind status);

}  // namespace private_storage
}  // namespace v04
}  // namespace rtcore

#endif
