#include "rtcore_v04_private_global_region.h"

#include <cstring>
#include <limits>

#include "rtcore_v04_request_owner_binding.h"

namespace rtcore {
namespace v04 {
namespace private_global_region {
namespace {

bool bytes_are_zero(const void *value, size_t byte_count) {
  const uint8_t *bytes = static_cast<const uint8_t *>(value);
  for (size_t index = 0; index < byte_count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (result == NULL ||
      rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (result == NULL ||
      (lhs != 0 &&
       rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

uint8_t count_lanes(uint32_t mask) {
  uint8_t count = 0;
  while (mask != 0) {
    count = static_cast<uint8_t>(count + (mask & 1u));
    mask >>= 1;
  }
  return count;
}

bool supported_base_color(uint8_t base_color) {
  return base_color == 0 || base_color == 1 || base_color == 7;
}

bool ranges_overlap(uint64_t lhs_base, uint64_t lhs_end,
                    uint64_t rhs_base, uint64_t rhs_end) {
  return lhs_base < rhs_end && rhs_base < lhs_end;
}

uint8_t launch_field_kind(uint8_t chunk) {
  switch (chunk) {
    case 0:
      return private_frontier::kFieldMutableRayState;
    case 1:
      return private_frontier::kFieldAsDecodeContext;
    case 2:
    case 3:
      return private_frontier::kFieldCommittedHit;
    case 4:
      return private_frontier::kFieldCurrentInstance;
  }
  return private_frontier::kFieldInvalid;
}

bool canonical_layout_shape(const region_layout_v0 &layout) {
  if (layout.valid != 1 || layout.sm_count == 0 ||
      layout.region_bytes != kRegionBytes ||
      layout.sm_region_stride < kRegionBytes ||
      layout.configured_l1_line_bytes < kAddressAlignmentBytes ||
      layout.configured_l1_line_bytes % kAddressAlignmentBytes != 0 ||
      layout.hidden_base % kAddressAlignmentBytes != 0 ||
      layout.sm_region_stride % kAddressAlignmentBytes != 0 ||
      !supported_base_color(layout.base_color) ||
      !bytes_are_zero(layout.reserved_zero,
                      sizeof(layout.reserved_zero))) {
    return false;
  }
  uint64_t color_delta = 0;
  uint64_t colored_base = 0;
  uint64_t last_sm_delta = 0;
  uint64_t last_sm_base = 0;
  uint64_t span_end = 0;
  return checked_mul_u64(
             layout.base_color,
             layout.configured_l1_line_bytes,
             &color_delta) &&
         checked_add_u64(layout.hidden_base, color_delta,
                         &colored_base) &&
         colored_base == layout.colored_hidden_base &&
         checked_mul_u64(layout.sm_count - 1,
                         layout.sm_region_stride,
                         &last_sm_delta) &&
         checked_add_u64(colored_base, last_sm_delta,
                         &last_sm_base) &&
         checked_add_u64(last_sm_base, kRegionBytes,
                         &span_end) &&
         span_end == layout.address_span_end;
}

bool owner_valid(const private_frontier::owner_binding_v0 &owner,
                 uint32_t owner_hw_sid,
                 uint8_t resident_warp_slot, uint8_t lane_id) {
  return request_owner::validate_private_frontier_owner_identity(owner) &&
         owner.owner_hw_sid == owner_hw_sid &&
         owner.resident_warp_id == resident_warp_slot &&
         owner.private_slot_id < kSlotCountPerSm &&
         owner.lane_id == lane_id;
}

bool launch_plan_is_canonical(
    const private_state_384::sparse_write_plan_v1 &plan) {
  if (plan.write_count != kLaunchWriteCount ||
      !bytes_are_zero(plan.reserved_zero,
                      sizeof(plan.reserved_zero))) {
    return false;
  }
  for (uint8_t chunk = 0; chunk < kLaunchWriteCount; ++chunk) {
    const private_state_384::chunk_write_v1 &write =
        plan.writes[chunk];
    if (write.slot_byte_offset !=
            chunk * private_state_384::kChunkBytes ||
        write.byte_count != private_state_384::kChunkBytes ||
        write.byte_mask != std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }

  private_state_384::image_v1 supplied_image = {};
  for (uint8_t chunk = 0; chunk < kLaunchWriteCount; ++chunk) {
    std::memcpy(
        supplied_image.bytes + plan.writes[chunk].slot_byte_offset,
        plan.writes[chunk].payload, private_state_384::kChunkBytes);
  }
  const private_state_384::control_tags_v1 control = {};
  private_state_384::state_v1 decoded = {};
  if (private_state_384::decode_image(
          private_state_384::kPrivateLayoutProfileId,
          private_state_384::kGenRtBvhFormatProfileId,
          supplied_image, control,
          &decoded) != private_state_384::kStatusOk) {
    return false;
  }

  private_state_384::launch_input_v1 launch = {};
  launch.ray = decoded.ray;
  launch.tlas_context = decoded.active_as;
  launch.ray_flags = decoded.ray_flags;
  launch.tlas_build_generation =
      decoded.tlas_build_generation;
  private_state_384::image_v1 canonical_image = {};
  private_state_384::sparse_write_plan_v1 canonical_plan = {};
  return private_state_384::initialize_new_launch_image(
             private_state_384::kPrivateLayoutProfileId,
             private_state_384::kGenRtBvhFormatProfileId,
             launch, &canonical_image,
             &canonical_plan) == private_state_384::kStatusOk &&
         std::memcmp(&plan, &canonical_plan, sizeof(plan)) == 0;
}

}  // namespace

status_kind prepare_region_layout(
    const region_config_v0 &config,
    const address_range_v0 *application_visible_ranges,
    size_t application_visible_range_count,
    region_layout_v0 *layout) {
  if (layout == NULL) return kStatusInvalidArgument;
  *layout = region_layout_v0();
  if ((application_visible_range_count != 0 &&
       application_visible_ranges == NULL) ||
      !bytes_are_zero(config.reserved_zero,
                      sizeof(config.reserved_zero)) ||
      config.sm_count == 0 ||
      config.sm_region_stride < kRegionBytes ||
      config.configured_l1_line_bytes < kAddressAlignmentBytes ||
      config.hidden_base % kAddressAlignmentBytes != 0 ||
      config.sm_region_stride % kAddressAlignmentBytes != 0 ||
      config.configured_l1_line_bytes %
              kAddressAlignmentBytes !=
          0) {
    return kStatusInvalidConfiguration;
  }
  if (!supported_base_color(config.base_color)) {
    return kStatusInvalidBaseColor;
  }

  uint64_t color_delta = 0;
  uint64_t colored_base = 0;
  uint64_t last_sm_delta = 0;
  uint64_t last_sm_base = 0;
  uint64_t span_end = 0;
  if (!checked_mul_u64(config.base_color,
                       config.configured_l1_line_bytes,
                       &color_delta) ||
      !checked_add_u64(config.hidden_base, color_delta,
                       &colored_base) ||
      !checked_mul_u64(config.sm_count - 1,
                       config.sm_region_stride,
                       &last_sm_delta) ||
      !checked_add_u64(colored_base, last_sm_delta,
                       &last_sm_base) ||
      !checked_add_u64(last_sm_base, kRegionBytes,
                       &span_end)) {
    return kStatusAddressOverflow;
  }

  for (size_t range_index = 0;
       range_index < application_visible_range_count;
       ++range_index) {
    const address_range_v0 &visible =
        application_visible_ranges[range_index];
    uint64_t visible_end = 0;
    if (visible.byte_count == 0 ||
        !checked_add_u64(visible.base, visible.byte_count,
                         &visible_end)) {
      return kStatusInvalidConfiguration;
    }
    for (uint32_t sm_id = 0; sm_id < config.sm_count; ++sm_id) {
      uint64_t sm_delta = 0;
      uint64_t sm_base = 0;
      uint64_t sm_end = 0;
      if (!checked_mul_u64(sm_id, config.sm_region_stride,
                           &sm_delta) ||
          !checked_add_u64(colored_base, sm_delta, &sm_base) ||
          !checked_add_u64(sm_base, kRegionBytes, &sm_end)) {
        return kStatusAddressOverflow;
      }
      if (ranges_overlap(sm_base, sm_end,
                         visible.base, visible_end)) {
        return kStatusVisibleRangeCollision;
      }
    }
  }

  region_layout_v0 prepared = {};
  prepared.valid = 1;
  prepared.base_color = config.base_color;
  prepared.sm_count = config.sm_count;
  prepared.configured_l1_line_bytes =
      config.configured_l1_line_bytes;
  prepared.hidden_base = config.hidden_base;
  prepared.colored_hidden_base = colored_base;
  prepared.sm_region_stride = config.sm_region_stride;
  prepared.region_bytes = kRegionBytes;
  prepared.address_span_end = span_end;
  if (!canonical_layout_shape(prepared)) {
    return kStatusInvalidConfiguration;
  }
  *layout = prepared;
  return kStatusOk;
}

status_kind slot_address(const region_layout_v0 &layout,
                         uint32_t owner_hw_sid,
                         uint32_t private_slot_id,
                         uint64_t *address) {
  if (address == NULL) return kStatusInvalidArgument;
  *address = 0;
  if (!canonical_layout_shape(layout) ||
      owner_hw_sid >= layout.sm_count ||
      private_slot_id >= kSlotCountPerSm) {
    return kStatusInvalidConfiguration;
  }
  uint64_t sm_delta = 0;
  uint64_t sm_base = 0;
  uint64_t slot_delta = 0;
  uint64_t result = 0;
  if (!checked_mul_u64(owner_hw_sid, layout.sm_region_stride,
                       &sm_delta) ||
      !checked_add_u64(layout.colored_hidden_base, sm_delta,
                       &sm_base) ||
      !checked_mul_u64(private_slot_id, kSlotStrideBytes,
                       &slot_delta) ||
      !checked_add_u64(sm_base, slot_delta, &result) ||
      result % kAddressAlignmentBytes != 0) {
    return kStatusAddressOverflow;
  }
  *address = result;
  return kStatusOk;
}

status_kind prepare_whole_mask_launch_plan(
    const region_config_v0 &config,
    const address_range_v0 *application_visible_ranges,
    size_t application_visible_range_count,
    uint32_t owner_hw_sid, uint32_t warp_uid, uint32_t warp_id,
    uint8_t resident_warp_slot, uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    const private_state_384::sparse_write_plan_v1
        launch_plans[kLaneCapacity],
    whole_mask_launch_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = whole_mask_launch_plan_v0();
  if (owners == NULL || launch_plans == NULL ||
      active_mask == 0 ||
      resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidArgument;
  }
  region_layout_v0 layout = {};
  const status_kind layout_status =
      prepare_region_layout(config, application_visible_ranges,
                            application_visible_range_count,
                            &layout);
  if (layout_status != kStatusOk) return layout_status;
  if (owner_hw_sid >= layout.sm_count) {
    return kStatusInvalidOwner;
  }

  whole_mask_launch_plan_v0 prepared = {};
  prepared.valid = 1;
  prepared.resident_warp_slot = resident_warp_slot;
  prepared.active_lane_count = count_lanes(active_mask);
  prepared.owner_hw_sid = owner_hw_sid;
  prepared.warp_uid = warp_uid;
  prepared.warp_id = warp_id;
  prepared.active_mask = active_mask;
  prepared.layout = layout;
  bool planned_slots[kSlotCountPerSm] = {};
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    const bool active =
        (active_mask & (uint32_t{1} << lane)) != 0;
    if (!active) {
      if (!bytes_are_zero(&owners[lane], sizeof(owners[lane])) ||
          !bytes_are_zero(&launch_plans[lane],
                          sizeof(launch_plans[lane]))) {
        return kStatusInvalidOwner;
      }
      continue;
    }
    if (!owner_valid(owners[lane], owner_hw_sid,
                     resident_warp_slot, lane)) {
      return kStatusInvalidOwner;
    }
    for (uint8_t previous_lane = 0; previous_lane < lane;
         ++previous_lane) {
      if ((active_mask & (uint32_t{1} << previous_lane)) != 0 &&
          owners[previous_lane].request_identity ==
              owners[lane].request_identity) {
        return kStatusOwnerAlias;
      }
    }
    const uint32_t private_slot_id =
        owners[lane].private_slot_id;
    if (planned_slots[private_slot_id]) {
      return kStatusOwnerAlias;
    }
    planned_slots[private_slot_id] = true;
    if (!launch_plan_is_canonical(launch_plans[lane])) {
      return kStatusMalformedLaunchPlan;
    }
    uint64_t slot_base = 0;
    const status_kind address_status =
        slot_address(layout, owner_hw_sid, private_slot_id,
                     &slot_base);
    if (address_status != kStatusOk) return address_status;

    lane_launch_plan_v0 &lane_plan = prepared.lanes[lane];
    lane_plan.valid = 1;
    lane_plan.lane_id = lane;
    lane_plan.write_count = kLaunchWriteCount;
    lane_plan.owner = owners[lane];
    for (uint8_t chunk = 0; chunk < kLaunchWriteCount; ++chunk) {
      const private_state_384::chunk_write_v1 &source =
          launch_plans[lane].writes[chunk];
      global_chunk_write_v0 &destination =
          lane_plan.writes[chunk];
      uint64_t address = 0;
      if (!checked_add_u64(slot_base, source.slot_byte_offset,
                           &address)) {
        return kStatusAddressOverflow;
      }
      destination.aligned_32b_address = address;
      destination.byte_mask = source.byte_mask;
      destination.slot_byte_offset = source.slot_byte_offset;
      destination.byte_count = source.byte_count;
      destination.field_kind = launch_field_kind(chunk);
      destination.address_space = kAddressSpaceGlobal;
      std::memcpy(destination.payload, source.payload,
                  private_state_384::kChunkBytes);
    }
  }
  *plan = prepared;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfiguration:
      return "invalid_configuration";
    case kStatusInvalidBaseColor:
      return "invalid_base_color";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusVisibleRangeCollision:
      return "visible_range_collision";
    case kStatusInvalidOwner:
      return "invalid_owner";
    case kStatusOwnerAlias:
      return "owner_alias";
    case kStatusMalformedLaunchPlan:
      return "malformed_launch_plan";
  }
  return "unknown";
}

}  // namespace private_global_region
}  // namespace v04
}  // namespace rtcore
