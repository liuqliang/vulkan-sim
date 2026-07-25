#ifndef RTCORE_V04_PRIVATE_FRONTIER_LAYOUT_H
#define RTCORE_V04_PRIVATE_FRONTIER_LAYOUT_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_typed_primitive_kernel.h"
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
static const uint32_t kCurrentInstanceOffset = 0x070;
static const uint32_t kCurrentInstanceBytes = 24;
static const uint32_t kCommittedHitOffset = 0x088;
static const uint32_t kCommittedHitBytes = 64;
static const uint32_t kRetainedCandidateOffset = 0x0c8;
static const uint32_t kRetainedCandidateBytes = 48;
static const uint32_t kPrimitiveResumeOffset = 0x0f8;
static const uint32_t kPrimitiveResumeBytes = 16;
static const uint32_t kFrontierEntriesOffset = 0x108;
static const uint32_t kFrontierEntryBytes = 16;
static const uint32_t kFrontierEntryCapacity = 16;
static const uint32_t kFrontierEntriesEnd = 0x208;
static const uint32_t kParentFrameOffset = 0x208;
static const uint32_t kParentFrameBytes = 128;
static const uint32_t kParentFrameEnd = 0x288;
static const uint32_t kTransitionSpillOffset = 0x288;
static const uint32_t kTransitionSpillBytes = 184;
static const uint32_t kStackTransitionSpillBytes = 128;
static const uint32_t kStackSelectedFetchBytes =
    sizeof(typed_node::selected_child_fetch_work_item_v0);
static const uint32_t kTransitionSpillEnd =
    kTransitionSpillOffset + kTransitionSpillBytes;
static const uint32_t kMaxAccessChunks = 12;
static const uint32_t kMaxNonemptyPopOperandChunks = 9;
static const uint32_t kMaxEmptyPopOperandChunks = 5;

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
  kFieldParentFrame = 7,
  kFieldCurrentInstance = 8,
  kFieldRetainedCandidate = 9,
  kFieldPrimitiveResume = 10,
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

typedef typed_stack::mutable_ray_state_v0 mutable_ray_state_v0;
typedef typed_stack::committed_hit_projection_v0
    committed_hit_projection_v0;
typedef typed_stack::traversal_frame_projection_v0
    traversal_frame_projection_v0;
typedef typed_stack::instance_shader_projection_v0
    instance_shader_projection_v0;

struct root_private_operands_v0 {
  mutable_ray_state_v0 mutable_ray;
  typed_blas::as_decode_context_v0 decode_context;
  committed_hit_projection_v0 committed_hit;
};

struct retained_candidate_projection_v0 {
  typed_primitive::primitive_identity_policy_facts_v0
      identity_and_policy;
  typed_primitive::triangle_hit_facts_v0 triangle_hit;
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
static_assert(sizeof(instance_shader_projection_v0) ==
                  kCurrentInstanceBytes,
              "current Instance projection must remain 24 bytes");
static_assert(kAsDecodeContextOffset + kAsDecodeContextBytes ==
                  kCurrentInstanceOffset,
              "current Instance projection must follow AS context");
static_assert(kCurrentInstanceOffset + kCurrentInstanceBytes ==
                  kCommittedHitOffset,
              "committed hit must follow current Instance projection");
static_assert(sizeof(committed_hit_projection_v0) == kCommittedHitBytes,
              "committed hit projection must remain 64 bytes");
static_assert(sizeof(retained_candidate_projection_v0) ==
                  kRetainedCandidateBytes,
              "retained candidate projection must remain 48 bytes");
static_assert(kCommittedHitOffset + kCommittedHitBytes ==
                  kRetainedCandidateOffset,
              "retained candidate must follow committed hit");
static_assert(kRetainedCandidateOffset + kRetainedCandidateBytes ==
                  kPrimitiveResumeOffset,
              "Primitive resume must follow retained candidate");
static_assert(sizeof(typed_primitive::primitive_resume_data_v0) ==
                  kPrimitiveResumeBytes,
              "Primitive resume data must remain 16 bytes");
static_assert(kPrimitiveResumeOffset + kPrimitiveResumeBytes ==
                  kFrontierEntriesOffset,
              "frontier entries must follow Primitive resume");
static_assert(sizeof(typed_node::compact_child_work_item_v0) ==
                  kFrontierEntryBytes,
              "private frontier entry must remain 16 bytes");
static_assert(kFrontierEntriesOffset +
                      kFrontierEntryCapacity * kFrontierEntryBytes ==
                  kFrontierEntriesEnd,
              "private frontier entry range changed");
static_assert(kParentFrameOffset == kFrontierEntriesEnd,
              "parent frame must follow frontier entries");
static_assert(kParentFrameEnd == kTransitionSpillOffset,
              "parent frame must end at transition spill");
static_assert(sizeof(traversal_frame_projection_v0) == kParentFrameBytes,
              "private parent frame must remain 128 bytes");
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

status_kind build_primitive_operand_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan);

status_kind build_frontier_metadata_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan);

status_kind build_nonempty_pop_operand_read_plan(
    const owner_binding_v0 &owner, const region_binding_v0 &region,
    const frontier_metadata_image_v0 &returned_metadata,
    access_plan_v0 *read_plan);

status_kind build_empty_pop_operand_read_plan(
    const owner_binding_v0 &owner, const region_binding_v0 &region,
    const frontier_metadata_image_v0 &returned_metadata,
    access_plan_v0 *read_plan);

status_kind build_parent_frame_read_plan(
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

status_kind decode_parent_frame(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    traversal_frame_projection_v0 *parent_frame);

status_kind decode_current_instance(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    instance_shader_projection_v0 *current_instance);

status_kind decode_committed_hit(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    committed_hit_projection_v0 *committed_hit);

status_kind decode_retained_candidate(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    retained_candidate_projection_v0 *retained_candidate);

status_kind decode_primitive_resume(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    typed_primitive::primitive_resume_data_v0 *primitive_resume);

status_kind apply_primitive_result_state(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const committed_hit_projection_v0 *committed_hit,
    const retained_candidate_projection_v0 *retained_candidate,
    const typed_primitive::primitive_resume_data_v0 *primitive_resume,
    access_plan_v0 *write_plan);

status_kind capture_parent_frame(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    traversal_frame_projection_v0 *parent_frame);

status_kind apply_parent_frame_push(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const traversal_frame_projection_v0 &parent_frame,
    access_plan_v0 *write_plan);

status_kind apply_parent_restore_delta(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::frontier_level_delta_v0 &delta,
    access_plan_v0 *write_plan);

status_kind apply_parent_state_restore(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const traversal_frame_projection_v0 &parent_frame,
    access_plan_v0 *write_plan);

status_kind apply_instance_enter_state(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const mutable_ray_state_v0 &object_ray,
    const typed_blas::as_decode_context_v0 &blas_decode_context,
    const instance_shader_projection_v0 &current_instance,
    access_plan_v0 *write_plan);

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

status_kind apply_stack_selected_fetch_spill_payload(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_node::selected_child_fetch_work_item_v0 &selected_fetch,
    access_plan_v0 *write_plan);

status_kind build_stack_selected_fetch_spill_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan);

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
