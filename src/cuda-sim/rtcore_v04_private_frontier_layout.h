#ifndef RTCORE_V04_PRIVATE_FRONTIER_LAYOUT_H
#define RTCORE_V04_PRIVATE_FRONTIER_LAYOUT_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_typed_stack_kernel.h"

namespace rtcore {
namespace v04 {
namespace private_frontier {

static const uint32_t kLayoutProfileId =
    typed_stack::kGenRtDerivedProfileId;
static const uint32_t kPrivateDataSlotBytes = 0x340;
static const uint32_t kPrivateDataSlotAlignment = 32;
static const uint32_t kSharedAccessChunkBytes = 32;
static const uint32_t kMutableRayStateOffset = 0x000;
static const uint32_t kMutableRayStateBytes = 44;
static const uint32_t kFrontierMetadataOffset = 0x02c;
static const uint32_t kFrontierMetadataBytes = 24;
static const uint32_t kAsDecodeContextOffset = 0x048;
static const uint32_t kAsDecodeContextBytes = 40;
static const uint32_t kCommittedHitOffset = 0x088;
static const uint32_t kCommittedHitBytes = 64;
static const uint32_t kFrontierEntriesOffset = 0x108;
static const uint32_t kFrontierEntryBytes = 16;
static const uint32_t kFrontierEntryCapacity = 16;
static const uint32_t kFrontierEntriesEnd = 0x208;
static const uint32_t kTransitionSpillOffset = 0x288;
static const uint32_t kTransitionSpillBytes = 184;
static const uint32_t kStackTransitionSpillBytes = 128;
static const uint32_t kStackSelectedFetchBytes =
    sizeof(typed_node::selected_child_fetch_work_item_v0);
static const uint32_t kTransitionSpillEnd =
    kTransitionSpillOffset + kTransitionSpillBytes;
static const uint32_t kMaxAccessChunks = 12;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedProfile,
  kStatusInvalidOwner,
  kStatusOwnerMismatch,
  kStatusInvalidRegion,
  kStatusAddressOverflow,
  kStatusInvalidMetadata,
  kStatusInvalidEntryIndex,
  kStatusInvalidDelta,
  kStatusPlanCapacityExceeded,
  kStatusInvalidSpillPayload,
};

enum access_kind : uint8_t {
  kAccessInvalid = 0,
  kAccessRead = 1,
  kAccessWrite = 2,
};

enum field_kind : uint8_t {
  kFieldInvalid = 0,
  kFieldFrontierMetadata = 1,
  kFieldFrontierEntry = 2,
  kFieldTransitionSpill = 3,
  kFieldMutableRayState = 4,
  kFieldAsDecodeContext = 5,
  kFieldCommittedHit = 6,
};

struct owner_binding_v0 {
  uint32_t owner_hw_sid;
  uint32_t resident_warp_id;
  uint32_t request_identity;
  uint32_t generation;
  uint32_t private_slot_id;
  uint8_t lane_id;
  uint8_t reserved_zero[3];
};

struct region_binding_v0 {
  uint32_t profile_id;
  uint32_t slot_count;
  uint64_t private_region_base;
};

struct frontier_metadata_image_v0 {
  uint32_t frontier_top;
  uint32_t frontier_count;
  uint32_t frontier_capacity;
  uint32_t current_level;
  uint32_t level_frame_depth;
  uint32_t max_level_depth;
};

struct mutable_ray_state_v0 {
  float origin[3];
  float direction[3];
  float inverse_direction[3];
  float t_min;
  float t_max;
};

struct committed_hit_projection_v0 {
  uint8_t valid;
  uint8_t geometry_type;
  uint8_t hit_kind;
  uint8_t attribute_word_count;
  uint8_t attribute_location;
  uint8_t attribute_format;
  uint8_t reserved_zero0[2];
  float hit_t;
  uint32_t policy_flags;
  uint64_t instance_metadata_ref;
  uint32_t primitive_index;
  uint32_t geometry_index;
  uint32_t instance_index;
  uint32_t instance_custom_index;
  uint32_t instance_sbt_contribution;
  uint32_t reserved_zero1;
  uint32_t inline_attributes[4];
};

struct root_private_operands_v0 {
  mutable_ray_state_v0 mutable_ray;
  typed_blas::as_decode_context_v0 decode_context;
  committed_hit_projection_v0 committed_hit;
};

struct shared_chunk_access_v0 {
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
  uint16_t slot_byte_offset;
  uint8_t byte_count;
  uint8_t field_kind;
  uint8_t access_kind;
  uint8_t reserved_zero[7];
};

struct access_plan_v0 {
  owner_binding_v0 owner;
  uint8_t access_count;
  uint8_t reserved_zero[7];
  shared_chunk_access_v0 accesses[kMaxAccessChunks];
};

struct shadow_slot_v0 {
  owner_binding_v0 owner;
  uint8_t bytes[kPrivateDataSlotBytes];
};

static_assert(sizeof(owner_binding_v0) == 24,
              "private frontier owner binding must remain 24 bytes");
static_assert(sizeof(region_binding_v0) == 16,
              "private frontier region binding must remain 16 bytes");
static_assert(sizeof(frontier_metadata_image_v0) == kFrontierMetadataBytes,
              "private frontier metadata must remain 24 bytes");
static_assert(sizeof(mutable_ray_state_v0) == kMutableRayStateBytes,
              "mutable ray state must remain 44 bytes");
static_assert(sizeof(typed_blas::as_decode_context_v0) ==
                  kAsDecodeContextBytes,
              "AS decode context must remain 40 bytes");
static_assert(sizeof(committed_hit_projection_v0) == kCommittedHitBytes,
              "committed hit projection must remain 64 bytes");
static_assert(sizeof(typed_node::compact_child_work_item_v0) ==
                  kFrontierEntryBytes,
              "private frontier entry must remain 16 bytes");
static_assert(kFrontierEntriesOffset +
                      kFrontierEntryCapacity * kFrontierEntryBytes ==
                  kFrontierEntriesEnd,
              "private frontier entry range changed");
static_assert(kTransitionSpillEnd == kPrivateDataSlotBytes,
              "transition spill must end at private-slot boundary");
static_assert(kStackSelectedFetchBytes == 56,
              "Stack selected-fetch spill payload changed");
static_assert(kStackSelectedFetchBytes <= kStackTransitionSpillBytes,
              "Stack selected fetch exceeds its transition-spill arm");
static_assert(kPrivateDataSlotBytes % kPrivateDataSlotAlignment == 0,
              "private data slot stride must remain 32-byte aligned");
static_assert(sizeof(shared_chunk_access_v0) == 24,
              "shared chunk descriptor must remain 24 bytes");

status_kind initialize_shadow_slot(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const frontier_metadata_image_v0 &metadata,
    access_plan_v0 *metadata_write_plan);

status_kind initialize_root_shadow_slot(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const frontier_metadata_image_v0 &metadata,
    const root_private_operands_v0 &root_operands,
    access_plan_v0 *initial_write_plan);

status_kind build_root_operand_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan);

status_kind decode_root_private_operands(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    root_private_operands_v0 *operands);

status_kind decode_metadata(const shadow_slot_v0 &slot,
                            const owner_binding_v0 &owner,
                            frontier_metadata_image_v0 *metadata);

status_kind decode_entry(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    uint32_t entry_index,
    typed_node::compact_child_work_item_v0 *entry);

status_kind apply_append_delta(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::frontier_append_delta_v0 &delta,
    access_plan_v0 *write_plan);

status_kind read_top_entry(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    typed_node::compact_child_work_item_v0 *entry,
    uint32_t *entry_index, access_plan_v0 *read_plan);

status_kind apply_pop_delta(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::frontier_pop_delta_v0 &delta,
    access_plan_v0 *write_plan);

status_kind apply_stack_selected_fetch_spill(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::push_result_v0 &result,
    access_plan_v0 *write_plan);

status_kind decode_stack_selected_fetch_spill(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    typed_node::selected_child_fetch_work_item_v0 *selected_fetch);

bool owners_equal(const owner_binding_v0 &lhs,
                  const owner_binding_v0 &rhs);

const char *status_name(status_kind status);

}  // namespace private_frontier
}  // namespace v04
}  // namespace rtcore

#endif
