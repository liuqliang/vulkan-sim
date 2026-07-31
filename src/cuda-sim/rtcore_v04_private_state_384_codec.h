#ifndef RTCORE_V04_PRIVATE_STATE_384_CODEC_H
#define RTCORE_V04_PRIVATE_STATE_384_CODEC_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_short_stack_replay.h"
#include "rtcore_v04_typed_primitive_kernel.h"

namespace rtcore {
namespace v04 {
namespace private_state_384 {

static const uint32_t kPrivateLayoutProfileId = 0x00040001u;
static const uint32_t kGenRtBvhFormatProfileId =
    typed_blas::kGenRtDerivedProfileId;
static const uint32_t kSlotBytes = 384;
static const uint32_t kSlotAlignmentBytes = 32;
static const uint32_t kChunkBytes = 32;
static const uint8_t kChunkCount = 12;
static const uint8_t kLaunchWriteCount = 5;
static const uint64_t kSharedPlacementBase = UINT64_C(0xfe00000000000000);
static const uint64_t kSharedPlacementOwnerStride = UINT64_C(0x1000000);
static const uint8_t kCommittedHitProjectionBytes = 56;
static const uint8_t kBoundaryProjectionBytes = 64;

static const uint16_t kChunk0Offset = 0x000;
static const uint16_t kChunk1Offset = 0x020;
static const uint16_t kChunk2Offset = 0x040;
static const uint16_t kChunk3Offset = 0x060;
static const uint16_t kChunk4Offset = 0x080;
static const uint16_t kChunk5Offset = 0x0a0;
static const uint16_t kChunk6Offset = 0x0c0;
static const uint16_t kChunk7Offset = 0x0e0;
static const uint16_t kChunk8Offset = 0x100;
static const uint16_t kChunk9Offset = 0x120;
static const uint16_t kChunk10Offset = 0x140;
static const uint16_t kChunk11Offset = 0x160;

static const uint16_t kMutableRayOffset = kChunk0Offset;
static const uint16_t kActiveAsContextOffset = kChunk1Offset;
static const uint16_t kCommittedHitOffset = kChunk2Offset;
static const uint16_t kCurrentInstanceRefOffset = 0x078;
static const uint16_t kCurrentInstanceOffset = kChunk4Offset;
static const uint16_t kStackMetadataOffset = 0x098;
static const uint16_t kStackEntriesOffset = kChunk5Offset;
static const uint16_t kBoundaryTransitionOffset = kChunk8Offset;
static const uint16_t kParentRestoreOffset = kChunk10Offset;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedPrivateLayout,
  kStatusUnsupportedBvhProfile,
  kStatusInvalidControl,
  kStatusInvalidRay,
  kStatusInvalidAsContext,
  kStatusAddressOverflow,
  kStatusInvalidCommittedHit,
  kStatusInvalidCurrentInstance,
  kStatusInvalidStack,
  kStatusInvalidBoundary,
  kStatusInvalidTransition,
  kStatusInvalidParentRestore,
  kStatusInvalidReservedBits,
};

enum union_arm_kind : uint8_t {
  kUnionArmNone = 0,
  kUnionArmBoundary = 1,
  kUnionArmTransition = 2,
};

enum boundary_reason_kind : uint8_t {
  kBoundaryReasonNone = 0,
  kBoundaryReasonAnyHit = 1,
  kBoundaryReasonProceduralIntersection = 2,
};

struct alignas(kSlotAlignmentBytes) image_v1 {
  uint8_t bytes[kSlotBytes];
};

struct ray_v1 {
  float origin[3];
  float direction[3];
  float t_min;
  float t_max;
};

struct as_context_v1 {
  uint64_t as_object_id;
  uint64_t device_base;
  uint64_t device_range_bytes;
  uint32_t as_object_generation;
  uint8_t as_type;
  uint8_t cull_mask;
  uint8_t reserved_zero[2];
};

struct current_instance_v1 {
  uint64_t instance_metadata_ref;
  uint32_t instance_index;
  uint32_t instance_custom_index;
  uint32_t instance_sbt_contribution;
  uint8_t instance_policy_flags;
  uint8_t reserved_zero[3];
};

struct boundary_state_v1 {
  typed_primitive::primitive_identity_policy_facts_v0 identity_and_policy;
  typed_primitive::triangle_hit_facts_v0 triangle_hit;
  typed_primitive::intersection_boundary_facts_v0 intersection;
  typed_primitive::primitive_resume_data_v0 primitive_resume;
};

struct transition_state_v1 {
  short_stack::entry_v0 selected;
  as_context_v1 decode_context;
  short_stack::entry_v0 pending_parent_resume;
};

struct parent_restore_state_v1 {
  ray_v1 ray;
  as_context_v1 tlas_context;
};

// These tags remain in the request-control record and are not duplicated in
// the 384-byte image.
struct control_tags_v1 {
  uint8_t union_arm;
  uint8_t boundary_reason;
  uint8_t pending_parent_resume_valid;
  uint8_t parent_restore_valid;
  uint8_t recovery_target_inflight;
  uint8_t reserved_zero[3];
};

struct state_v1 {
  ray_v1 ray;
  as_context_v1 active_as;
  typed_stack::committed_hit_projection_v0 committed_hit;
  current_instance_v1 current_instance;
  uint32_t ray_flags;
  uint32_t tlas_build_generation;
  uint32_t blas_build_generation;
  short_stack::state_v0 stack;
  boundary_state_v1 boundary;
  transition_state_v1 transition;
  parent_restore_state_v1 parent_restore;
};

struct launch_input_v1 {
  ray_v1 ray;
  as_context_v1 tlas_context;
  uint32_t ray_flags;
  uint32_t tlas_build_generation;
};

struct chunk_write_v1 {
  uint16_t slot_byte_offset;
  uint16_t byte_count;
  uint32_t byte_mask;
  uint8_t payload[kChunkBytes];
};

struct sparse_write_plan_v1 {
  uint8_t write_count;
  uint8_t reserved_zero[7];
  chunk_write_v1 writes[kLaunchWriteCount];
};

struct stack_sparse_projection_v1 {
  uint8_t metadata[4];
  uint8_t entries[short_stack::kLogicalCapacity *
                  sizeof(short_stack::entry_v0)];
};

status_kind encode_image(uint32_t private_layout_profile_id,
                         uint32_t bvh_format_profile_id,
                         const state_v1 &state,
                         const control_tags_v1 &control,
                         image_v1 *image);

status_kind decode_image(uint32_t private_layout_profile_id,
                         uint32_t bvh_format_profile_id,
                         const image_v1 &image,
                         const control_tags_v1 &control,
                         state_v1 *state);

status_kind initialize_new_launch_image(
    uint32_t private_layout_profile_id,
    uint32_t bvh_format_profile_id,
    const launch_input_v1 &input,
    image_v1 *image,
    sparse_write_plan_v1 *plan);

status_kind encode_stack_sparse_projection(
    const short_stack::state_v0 &stack,
    stack_sparse_projection_v1 *projection,
    uint8_t recovery_target_inflight = 0);

status_kind encode_committed_hit_sparse_projection(
    const typed_stack::committed_hit_projection_v0 &hit,
    uint8_t payload[kCommittedHitProjectionBytes]);

status_kind encode_boundary_sparse_projection(
    const boundary_state_v1 &boundary, uint8_t reason,
    const as_context_v1 &active_as,
    uint8_t payload[kBoundaryProjectionBytes]);

const char *status_name(status_kind status);

static_assert(sizeof(image_v1) == kSlotBytes,
              "384B private-state image size changed");
static_assert(alignof(image_v1) == kSlotAlignmentBytes,
              "384B private-state image alignment changed");
static_assert(sizeof(ray_v1) == 32,
              "384B private-state ray projection must remain 32 bytes");
static_assert(sizeof(as_context_v1) == 32,
              "384B private-state AS projection must remain 32 bytes");
static_assert(sizeof(current_instance_v1) == 24,
              "384B current Instance projection must remain 24 bytes");
static_assert(sizeof(short_stack::entry_v0) == 16,
              "384B short-stack entry must remain 16 bytes");
static_assert(kStackEntriesOffset +
                      short_stack::kLogicalCapacity *
                          sizeof(short_stack::entry_v0) ==
                  kBoundaryTransitionOffset,
              "384B short-stack range changed");
static_assert(kParentRestoreOffset + 2 * kChunkBytes == kSlotBytes,
              "384B parent-restore range changed");
static_assert(kChunk0Offset == 0 * kChunkBytes &&
                  kChunk1Offset == 1 * kChunkBytes &&
                  kChunk2Offset == 2 * kChunkBytes &&
                  kChunk3Offset == 3 * kChunkBytes &&
                  kChunk4Offset == 4 * kChunkBytes &&
                  kChunk5Offset == 5 * kChunkBytes &&
                  kChunk6Offset == 6 * kChunkBytes &&
                  kChunk7Offset == 7 * kChunkBytes &&
                  kChunk8Offset == 8 * kChunkBytes &&
                  kChunk9Offset == 9 * kChunkBytes &&
                  kChunk10Offset == 10 * kChunkBytes &&
                  kChunk11Offset == 11 * kChunkBytes,
              "384B C0-C11 chunk offsets changed");

}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore

#endif
