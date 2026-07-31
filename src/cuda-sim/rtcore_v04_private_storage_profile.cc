#include "rtcore_v04_private_storage_profile.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace private_storage {
namespace {

static uint8_t count_lanes(uint32_t mask) {
  uint8_t count = 0;
  while (mask != 0) {
    count = static_cast<uint8_t>(count + (mask & 1u));
    mask >>= 1;
  }
  return count;
}

static bool bytes_are_zero(const void *bytes, size_t count) {
  const uint8_t *cursor = static_cast<const uint8_t *>(bytes);
  for (size_t index = 0; index < count; ++index) {
    if (cursor[index] != 0) return false;
  }
  return true;
}

static bool launch_policy_is_valid(
    const typed_node::ray_policy_v0 &policy) {
  const uint32_t opacity_group =
      policy.ray_flags &
      (typed_primitive::kRayFlagOpaque |
       typed_primitive::kRayFlagNoOpaque |
       typed_primitive::kRayFlagCullOpaque |
       typed_primitive::kRayFlagCullNoOpaque);
  const uint32_t triangle_group =
      policy.ray_flags &
      (typed_primitive::kRayFlagSkipTriangles |
       typed_primitive::kRayFlagCullFrontFacingTriangles |
       typed_primitive::kRayFlagCullBackFacingTriangles);
  const bool opacity_conflict =
      opacity_group != 0 &&
      (opacity_group & (opacity_group - 1u)) != 0;
  const bool triangle_conflict =
      triangle_group != 0 &&
      (triangle_group & (triangle_group - 1u)) != 0;
  const bool skip_all_geometry =
      (policy.ray_flags &
       typed_primitive::kRayFlagSkipTriangles) != 0 &&
      (policy.ray_flags &
       typed_primitive::kRayFlagSkipAabbs) != 0;
  return (policy.ray_flags &
          ~typed_primitive::kSupportedRayFlagMask) == 0 &&
         !opacity_conflict && !triangle_conflict &&
         !skip_all_geometry &&
         bytes_are_zero(policy.reserved_zero,
                        sizeof(policy.reserved_zero));
}

static private_state_384::launch_input_v1 make_launch_input(
    const private_frontier::root_private_operands_v0 &operands,
    const typed_node::ray_policy_v0 &policy,
    uint32_t root_build_generation) {
  private_state_384::launch_input_v1 input = {};
  for (unsigned component = 0; component < 3; ++component) {
    input.ray.origin[component] =
        operands.mutable_ray.origin[component];
    input.ray.direction[component] =
        operands.mutable_ray.direction[component];
  }
  input.ray.t_min = operands.mutable_ray.t_min;
  input.ray.t_max = operands.mutable_ray.t_max;
  input.tlas_context.as_object_id =
      operands.decode_context.as_object.object_id;
  input.tlas_context.device_base =
      operands.decode_context.device_base;
  input.tlas_context.device_range_bytes =
      operands.decode_context.device_range_bytes;
  input.tlas_context.as_object_generation =
      operands.decode_context.as_object.generation;
  input.tlas_context.as_type =
      operands.decode_context.as_object.as_type;
  input.tlas_context.cull_mask = policy.cull_mask;
  input.ray_flags = policy.ray_flags;
  input.tlas_build_generation = root_build_generation;
  return input;
}

}  // namespace

status_kind parse_profile(const char *value, profile_kind *profile) {
  if (profile == NULL) return kStatusInvalidArgument;
  if (value == NULL) {
    *profile = kProfileLegacyShared832;
    return kStatusOk;
  }
  if (std::strcmp(value, "legacy_shared832") == 0) {
    *profile = kProfileLegacyShared832;
    return kStatusOk;
  }
  if (std::strcmp(value, "compressed_shared384") == 0) {
    *profile = kProfileCompressedShared384;
    return kStatusOk;
  }
  if (std::strcmp(value, "global384") == 0) {
    *profile = kProfileGlobal384;
    return kStatusOk;
  }
  return kStatusInvalidSelector;
}

const char *profile_name(profile_kind profile) {
  switch (profile) {
    case kProfileLegacyShared832:
      return "legacy_shared832";
    case kProfileCompressedShared384:
      return "compressed_shared384";
    case kProfileGlobal384:
      return "global384";
  }
  return "invalid";
}

status_kind profile_resident_charge_bytes_per_lane(
    profile_kind profile, uint32_t *charge_bytes_per_lane) {
  if (charge_bytes_per_lane == NULL) return kStatusInvalidArgument;
  switch (profile) {
    case kProfileLegacyShared832:
      *charge_bytes_per_lane =
          private_shared::kLegacyResidentChargeBytesPerLane;
      return kStatusOk;
    case kProfileCompressedShared384:
      *charge_bytes_per_lane =
          private_shared::kCompressedResidentChargeBytesPerLane;
      return kStatusOk;
    case kProfileGlobal384:
      *charge_bytes_per_lane = 0;
      return kStatusOk;
  }
  return kStatusInvalidSelector;
}

static private_shared::resident_charge_profile_kind
profile_resident_charge_profile(profile_kind profile) {
  switch (profile) {
    case kProfileLegacyShared832:
      return private_shared::kResidentChargeProfileLegacyShared832;
    case kProfileCompressedShared384:
      return private_shared::kResidentChargeProfileCompressedShared384;
    case kProfileGlobal384:
      break;
  }
  return private_shared::kResidentChargeProfileInvalid;
}

static status_kind prepare_new_warp_with_profile(
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
    profile_kind profile, bool short_stack_enabled,
    private_shared::status_kind *legacy_failure_status,
    admission_candidate_plan_v0 *plan) {
  if (plan == NULL || legacy_failure_status == NULL) {
    return kStatusInvalidArgument;
  }
  std::memset(plan, 0, sizeof(*plan));
  *legacy_failure_status = private_shared::kStatusOk;
  if (owners == NULL || root_operands == NULL || active_mask == 0 ||
      ((short_stack_enabled ||
        profile == kProfileCompressedShared384) &&
       root_build_generations == NULL) ||
      (profile == kProfileCompressedShared384 &&
       ray_policies == NULL)) {
    return kStatusInvalidArgument;
  }
  if (profile == kProfileGlobal384) {
    return kStatusUnsupportedProfile;
  }
  if (profile != kProfileLegacyShared832 &&
      profile != kProfileCompressedShared384) {
    return kStatusInvalidSelector;
  }

  uint32_t resident_charge_bytes_per_lane = 0;
  const private_shared::resident_charge_profile_kind
      resident_charge_profile =
          profile_resident_charge_profile(profile);
  const status_kind charge_status =
      profile_resident_charge_bytes_per_lane(
          profile, &resident_charge_bytes_per_lane);
  if (charge_status != kStatusOk ||
      resident_charge_bytes_per_lane == 0 ||
      resident_charge_profile ==
          private_shared::kResidentChargeProfileInvalid) {
    return charge_status == kStatusOk ? kStatusUnsupportedProfile
                                     : charge_status;
  }

  admission_candidate_plan_v0 prepared = {};
  if (profile == kProfileCompressedShared384 &&
      !short_stack_enabled) {
    return kStatusUnsupportedProfile;
  }
  prepared.profile = static_cast<uint8_t>(profile);
  prepared.active_mask = active_mask;
  prepared.active_lane_count = count_lanes(active_mask);
  prepared.resident_charge_profile =
      static_cast<uint8_t>(resident_charge_profile);
  prepared.resident_charge_bytes_per_lane =
      resident_charge_bytes_per_lane;
  if (profile == kProfileCompressedShared384) {
    private_state_384::sparse_write_plan_v1
        launch_plans[private_shared::kLaneCapacity] = {};
    for (uint32_t lane = 0; lane < private_shared::kLaneCapacity;
         ++lane) {
      if ((active_mask & (uint32_t{1} << lane)) == 0) continue;
      lane_launch_candidate_v0 &candidate = prepared.lanes[lane];
      const private_frontier::committed_hit_projection_v0 empty_hit = {};
      if (!launch_policy_is_valid(ray_policies[lane]) ||
          std::memcmp(&root_operands[lane].committed_hit, &empty_hit,
                      sizeof(empty_hit)) != 0) {
        return kStatusCodecRejected;
      }
      const private_state_384::launch_input_v1 input =
          make_launch_input(root_operands[lane], ray_policies[lane],
                            root_build_generations[lane]);
      // The full image is transient. Only its valid C0-C4 write set may leave
      // admission, so stale C5-C11 bytes never become candidate authority.
      private_state_384::image_v1 launch_image = {};
      const private_state_384::status_kind codec_status =
          private_state_384::initialize_new_launch_image(
              private_state_384::kPrivateLayoutProfileId,
              private_state_384::kGenRtBvhFormatProfileId, input,
              &launch_image, &candidate.sparse_writes);
      if (codec_status != private_state_384::kStatusOk ||
          candidate.sparse_writes.write_count !=
              private_state_384::kLaunchWriteCount) {
        return kStatusCodecRejected;
      }
      candidate.valid = true;
      candidate.valid_chunk_mask =
          static_cast<uint8_t>(
              (uint8_t{1} << private_state_384::kLaunchWriteCount) - 1u);
      candidate.owner = owners[lane];
      launch_plans[lane] = candidate.sparse_writes;
    }
    const private_shared::status_kind compressed_status =
        private_shared::prepare_new_warp_with_compressed_launch(
            state, warp_uid, warp_id, active_mask, owners,
            launch_plans, &prepared.legacy_live_plan);
    if (compressed_status != private_shared::kStatusOk) {
      *legacy_failure_status = compressed_status;
      return kStatusLegacyAdmissionRejected;
    }
    prepared.compressed_candidate_valid = true;
  } else {
    const private_shared::status_kind legacy_status =
        short_stack_enabled
            ? private_shared::
                  prepare_new_warp_with_root_operands_and_short_stack(
                      state, warp_uid, warp_id, active_mask, owners,
                      root_operands, root_build_generations,
                      &prepared.legacy_live_plan)
            : private_shared::prepare_new_warp_with_root_operands(
                  state, warp_uid, warp_id, active_mask, owners,
                  root_operands, &prepared.legacy_live_plan);
    if (legacy_status != private_shared::kStatusOk) {
      *legacy_failure_status = legacy_status;
      return kStatusLegacyAdmissionRejected;
    }
  }
  prepared.valid = true;
  *plan = prepared;
  return kStatusOk;
}

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
    admission_candidate_plan_v0 *plan) {
  if (plan == NULL || legacy_failure_status == NULL) {
    return kStatusInvalidArgument;
  }
  std::memset(plan, 0, sizeof(*plan));
  *legacy_failure_status = private_shared::kStatusOk;
  profile_kind profile = kProfileLegacyShared832;
  const status_kind selector_status =
      parse_profile(selector_value, &profile);
  if (selector_status != kStatusOk) return selector_status;
  return prepare_new_warp_with_profile(
      state, warp_uid, warp_id, active_mask, owners, root_operands,
      ray_policies, root_build_generations, profile,
      short_stack_enabled, legacy_failure_status, plan);
}

status_kind prepare_global384_launch_candidate_from_selector(
    const char *selector_value,
    const private_global_region::region_config_v0 &config,
    const private_global_region::address_range_v0
        *application_visible_ranges,
    size_t application_visible_range_count,
    uint32_t owner_hw_sid, uint32_t warp_uid, uint32_t warp_id,
    uint8_t resident_warp_slot, uint32_t active_mask,
    const private_frontier::owner_binding_v0
        owners[private_global_region::kLaneCapacity],
    const private_state_384::sparse_write_plan_v1
        launch_plans[private_global_region::kLaneCapacity],
    private_global_region::status_kind *global_failure_status,
    private_global_region::whole_mask_launch_plan_v0 *plan) {
  if (global_failure_status != NULL) {
    *global_failure_status =
        private_global_region::kStatusInvalidArgument;
  }
  if (plan != NULL) {
    *plan = private_global_region::whole_mask_launch_plan_v0();
  }
  if (global_failure_status == NULL || plan == NULL) {
    return kStatusInvalidArgument;
  }
  *global_failure_status =
      private_global_region::kStatusOk;
  profile_kind profile = kProfileLegacyShared832;
  const status_kind profile_status =
      parse_profile(selector_value, &profile);
  if (profile_status != kStatusOk) return profile_status;
  if (profile != kProfileGlobal384) {
    return kStatusUnsupportedProfile;
  }
  const private_global_region::status_kind global_status =
      private_global_region::prepare_whole_mask_launch_plan(
          config, application_visible_ranges,
          application_visible_range_count, owner_hw_sid,
          warp_uid, warp_id, resident_warp_slot, active_mask,
          owners, launch_plans, plan);
  if (global_status != private_global_region::kStatusOk) {
    *global_failure_status = global_status;
    return kStatusGlobalPlanRejected;
  }
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidSelector:
      return "invalid_selector";
    case kStatusUnsupportedProfile:
      return "unsupported_profile";
    case kStatusLegacyAdmissionRejected:
      return "legacy_admission_rejected";
    case kStatusCodecRejected:
      return "codec_rejected";
    case kStatusGlobalPlanRejected:
      return "global_plan_rejected";
  }
  return "unknown";
}

}  // namespace private_storage
}  // namespace v04
}  // namespace rtcore
