#ifndef RTCORE_V04_TYPED_INSTANCE_KERNEL_H
#define RTCORE_V04_TYPED_INSTANCE_KERNEL_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_typed_blas_decode_context.h"

namespace rtcore {
namespace v04 {
namespace typed_instance {

static const uint32_t kGenRtDerivedProfileId = 0x00010001u;
static const uint8_t kInstancePayloadKind = 1u;
static const uint8_t kAsTypeTlas = 1u;
static const uint8_t kAsTypeBlas = 2u;
static const unsigned kMatrixElementCount = 12;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedProfile,
  kStatusMalformedEnvelope,
  kStatusMalformedLeaf,
  kStatusUnsupportedLeafEncoding,
  kStatusInvalidNumericInput,
  kStatusInvalidRay,
  kStatusDegenerateTransform,
  kStatusInvalidTransitionBinding,
};

enum level_kind : uint8_t {
  kLevelInvalid = 0,
  kLevelTlas = 1,
  kLevelBlas = 2,
};

enum enter_result_kind : uint8_t {
  kEnterResultInvalid = 0,
  kEnterResultCulled = 1,
  kEnterResultBlasRoot = 2,
};

enum enter_output_valid_bit : uint8_t {
  kObjectRayValid = 1u << 0,
  kInstanceProjectionValid = 1u << 1,
  kRootFetchValid = 1u << 2,
};

struct raw_payload_header_v0 {
  uint8_t expected_payload_kind;
  uint8_t expected_chunk_count;
  uint16_t payload_byte_count;
  uint8_t received_chunk_mask;
  uint8_t reserved_zero[3];
};

struct raw_instance_payload_v0 {
  raw_payload_header_v0 header;
  uint8_t raw_bytes[128];
};

struct alignas(16) boundary_input_v0 {
  uint32_t profile_id;
  uint32_t reserved_zero[3];
  raw_instance_payload_v0 raw_instance;
};

struct alignas(16) boundary_result_v0 {
  uint8_t status;
  uint8_t geometry_ray_mask;
  uint8_t leaf_type;
  uint8_t geometry_flags;
  uint8_t instance_flags;
  uint8_t reserved_zero[3];
  uint32_t shader_index;
  uint32_t instance_sbt_contribution;
  uint64_t start_node_address;
  uint64_t bvh_address;
  uint32_t instance_custom_index;
  uint32_t instance_index;
  uint32_t world_to_object_bits[kMatrixElementCount];
  uint32_t object_to_world_bits[kMatrixElementCount];
  uint32_t reserved_tail[2];
};

struct mutable_ray_state_v0 {
  float origin[3];
  float direction[3];
  float inverse_direction[3];
  float t_min;
  float t_max;
};

struct ray_policy_v0 {
  uint32_t ray_flags;
  uint8_t cull_mask;
  uint8_t reserved_zero[3];
};

struct instance_blas_reference_v0 {
  uint64_t tlas_object_id;
  uint64_t instance_metadata_reference;
  uint64_t blas_object_id;
  uint32_t tlas_generation;
  uint32_t tlas_build_generation;
  uint32_t blas_generation;
  uint8_t valid;
  uint8_t reserved_zero[3];
};

struct instance_shader_projection_v0 {
  uint64_t instance_metadata_reference;
  uint32_t shader_index;
  uint32_t instance_sbt_contribution;
  uint32_t instance_custom_index;
  uint32_t instance_index;
  uint8_t instance_flags;
  uint8_t geometry_flags;
  uint8_t geometry_ray_mask;
  uint8_t reserved_zero;
};

struct root_fetch_work_item_v0 {
  uint64_t encoded_reference;
  uint32_t build_generation;
  uint8_t expected_payload_kind;
  uint8_t reserved_zero[3];
  typed_blas::as_decode_context_v0 decode_context;
};

struct alignas(16) enter_input_v0 {
  uint32_t profile_id;
  uint8_t current_level;
  uint8_t reserved_zero0[11];
  mutable_ray_state_v0 world_ray;
  ray_policy_v0 policy;
  uint32_t reserved_zero1;
  raw_instance_payload_v0 raw_instance;
  instance_blas_reference_v0 instance_blas_reference;
  typed_blas::as_decode_context_v0 tlas_decode_context;
  typed_blas::as_decode_context_v0 blas_decode_context;
  typed_blas::root_descriptor_v0 blas_root_descriptor;
};

struct alignas(16) enter_result_v0 {
  uint8_t status;
  uint8_t result_kind;
  uint8_t mask_visible;
  uint8_t output_valid_mask;
  uint8_t reserved_zero[12];
  mutable_ray_state_v0 object_ray;
  instance_shader_projection_v0 instance_projection;
  root_fetch_work_item_v0 root_fetch;
};

static_assert(sizeof(raw_payload_header_v0) == 8,
              "raw payload header must remain 8 bytes");
static_assert(sizeof(raw_instance_payload_v0) == 136,
              "GEN_RT raw Instance envelope must remain 136 bytes");
static_assert(sizeof(boundary_input_v0) == 160,
              "typed Instance boundary input must remain 160 bytes");
static_assert(alignof(boundary_input_v0) == 16,
              "typed Instance boundary input must remain 16-byte aligned");
static_assert(offsetof(boundary_input_v0, raw_instance) == 16,
              "typed Instance raw payload offset changed");
static_assert(sizeof(boundary_result_v0) == 144,
              "typed Instance boundary result must remain 144 bytes");
static_assert(offsetof(boundary_result_v0, shader_index) == 8,
              "typed Instance shader index offset changed");
static_assert(offsetof(boundary_result_v0, start_node_address) == 16,
              "typed Instance start-node offset changed");
static_assert(offsetof(boundary_result_v0, bvh_address) == 24,
              "typed Instance BVH address offset changed");
static_assert(offsetof(boundary_result_v0, world_to_object_bits) == 40,
              "typed Instance W2O projection offset changed");
static_assert(offsetof(boundary_result_v0, object_to_world_bits) == 88,
              "typed Instance O2W projection offset changed");
static_assert(sizeof(mutable_ray_state_v0) == 44,
              "mutable ray state must remain 44 bytes");
static_assert(sizeof(ray_policy_v0) == 8,
              "ray policy must remain 8 bytes");
static_assert(sizeof(instance_blas_reference_v0) == 40,
              "Instance-to-BLAS relation must remain 40 bytes");
static_assert(sizeof(instance_shader_projection_v0) == 32,
              "Instance shader projection must remain 32 bytes");
static_assert(sizeof(root_fetch_work_item_v0) == 56,
              "BLAS root fetch work item must remain 56 bytes");
static_assert(sizeof(enter_input_v0) == 368,
              "typed Instance enter input must remain 368 bytes");
static_assert(alignof(enter_input_v0) == 16,
              "typed Instance enter input must remain aligned");
static_assert(offsetof(enter_input_v0, world_ray) == 16,
              "typed Instance world ray offset changed");
static_assert(offsetof(enter_input_v0, raw_instance) == 72,
              "typed Instance raw envelope offset changed");
static_assert(offsetof(enter_input_v0, instance_blas_reference) == 208,
              "typed Instance relation offset changed");
static_assert(offsetof(enter_input_v0, tlas_decode_context) == 248,
              "typed Instance TLAS context offset changed");
static_assert(offsetof(enter_input_v0, blas_decode_context) == 288,
              "typed Instance BLAS context offset changed");
static_assert(offsetof(enter_input_v0, blas_root_descriptor) == 328,
              "typed Instance BLAS root descriptor offset changed");
static_assert(sizeof(enter_result_v0) == 160,
              "typed Instance enter result must remain 160 bytes");
static_assert(offsetof(enter_result_v0, object_ray) == 16,
              "typed Instance object ray offset changed");
static_assert(offsetof(enter_result_v0, instance_projection) == 64,
              "typed Instance shader projection offset changed");
static_assert(offsetof(enter_result_v0, root_fetch) == 96,
              "typed Instance root fetch offset changed");

bool make_raw_instance_payload(const void *raw_instance_bytes,
                               raw_instance_payload_v0 *payload);

boundary_result_v0 execute(const boundary_input_v0 &input);

bool make_mutable_ray_state(const float origin[3], const float direction[3],
                            float t_min, float t_max,
                            mutable_ray_state_v0 *ray);

enter_result_v0 execute_enter(const enter_input_v0 &input);

const char *status_name(status_kind status);

}  // namespace typed_instance
}  // namespace v04
}  // namespace rtcore

#endif
