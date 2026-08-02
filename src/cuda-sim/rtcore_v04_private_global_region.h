#ifndef RTCORE_V04_PRIVATE_GLOBAL_REGION_H
#define RTCORE_V04_PRIVATE_GLOBAL_REGION_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_private_state_384_codec.h"

namespace rtcore {
namespace v04 {
namespace private_global_region {

static const uint32_t kSlotCountPerSm = 256;
static const uint32_t kSlotStrideBytes =
    private_state_384::kSlotBytes;
static const uint32_t kRegionBytes =
    kSlotCountPerSm * kSlotStrideBytes;
static const uint32_t kAddressAlignmentBytes =
    private_state_384::kChunkBytes;
static const uint8_t kLaneCapacity = 32;
static const uint8_t kResidentWarpCapacity = 8;
static const uint8_t kLaunchWriteCount =
    private_state_384::kLaunchWriteCount;
static const char kBaseColorEnvironmentName[] =
    "VULKAN_SIM_RTCORE_REPLAY_V04_GLOBAL384_BASE_COLOR";

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusInvalidBaseColor,
  kStatusAddressOverflow,
  kStatusVisibleRangeCollision,
  kStatusInvalidOwner,
  kStatusOwnerAlias,
  kStatusMalformedLaunchPlan,
};

enum address_space_kind : uint8_t {
  kAddressSpaceInvalid = 0,
  kAddressSpaceGlobal = 1,
};

struct address_range_v0 {
  uint64_t base;
  uint64_t byte_count;
};

struct region_config_v0 {
  uint64_t hidden_base;
  uint64_t sm_region_stride;
  uint32_t sm_count;
  uint32_t configured_l1_line_bytes;
  uint8_t base_color;
  uint8_t reserved_zero[7];
};

struct region_layout_v0 {
  uint8_t valid;
  uint8_t base_color;
  uint8_t reserved_zero[6];
  uint32_t sm_count;
  uint32_t configured_l1_line_bytes;
  uint64_t hidden_base;
  uint64_t colored_hidden_base;
  uint64_t sm_region_stride;
  uint64_t region_bytes;
  uint64_t address_span_end;
};

struct global_chunk_write_v0 {
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
  uint16_t slot_byte_offset;
  uint16_t byte_count;
  uint8_t field_kind;
  uint8_t address_space;
  uint8_t reserved_zero[6];
  uint8_t payload[private_state_384::kChunkBytes];
};

struct lane_launch_plan_v0 {
  uint8_t valid;
  uint8_t lane_id;
  uint8_t write_count;
  uint8_t reserved_zero[5];
  private_frontier::owner_binding_v0 owner;
  global_chunk_write_v0 writes[kLaunchWriteCount];
};

struct whole_mask_launch_plan_v0 {
  uint8_t valid;
  uint8_t resident_warp_slot;
  uint8_t active_lane_count;
  uint8_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
  region_layout_v0 layout;
  lane_launch_plan_v0 lanes[kLaneCapacity];
};

status_kind parse_base_color(const char *value, uint8_t *base_color);

status_kind prepare_region_layout(
    const region_config_v0 &config,
    const address_range_v0 *application_visible_ranges,
    size_t application_visible_range_count,
    region_layout_v0 *layout);

status_kind slot_address(const region_layout_v0 &layout,
                         uint32_t owner_hw_sid,
                         uint32_t private_slot_id,
                         uint64_t *address);

status_kind prepare_whole_mask_launch_plan(
    const region_config_v0 &config,
    const address_range_v0 *application_visible_ranges,
    size_t application_visible_range_count,
    uint32_t owner_hw_sid, uint32_t warp_uid, uint32_t warp_id,
    uint8_t resident_warp_slot, uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    const private_state_384::sparse_write_plan_v1
        launch_plans[kLaneCapacity],
    whole_mask_launch_plan_v0 *plan);

const char *status_name(status_kind status);

}  // namespace private_global_region
}  // namespace v04
}  // namespace rtcore

#endif
