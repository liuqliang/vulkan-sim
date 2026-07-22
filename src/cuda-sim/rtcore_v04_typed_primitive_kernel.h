#ifndef RTCORE_V04_TYPED_PRIMITIVE_KERNEL_H
#define RTCORE_V04_TYPED_PRIMITIVE_KERNEL_H

#include <cstddef>
#include <cstdint>

namespace rtcore {
namespace v04 {
namespace typed_primitive {

static const uint32_t kGenRtDerivedProfileId = 0x00010001u;
static const uint8_t kQuadPayloadKind = 4u;
static const uint32_t kTriangleFrontCounterclockwise = 0x2u;
static const uint8_t kHitKindFrontFacing = 0xfeu;
static const uint8_t kHitKindBackFacing = 0xffu;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedProfile,
  kStatusMalformedEnvelope,
  kStatusMalformedLeaf,
  kStatusUnsupportedLeafEncoding,
  kStatusInvalidNumericInput,
};

struct raw_payload_header_v0 {
  uint8_t expected_payload_kind;
  uint8_t expected_chunk_count;
  uint16_t payload_byte_count;
  uint8_t received_chunk_mask;
  uint8_t reserved_zero[3];
};

struct raw_primitive_payload_v0 {
  raw_payload_header_v0 header;
  uint8_t raw_bytes[64];
};

struct ray_state_v0 {
  float origin[3];
  float direction[3];
  float t_min;
  float t_max;
};

struct alignas(16) candidate_input_v0 {
  uint32_t profile_id;
  uint32_t instance_flags;
  ray_state_v0 object_ray;
  float world_to_object_t_multiplier;
  float committed_world_t;
  float world_t_min;
  float world_t_max;
  raw_primitive_payload_v0 raw_primitive;
};

struct candidate_result_v0 {
  uint8_t status;
  uint8_t geometric_hit;
  uint8_t candidate_hit;
  uint8_t counter_clockwise_facing;
  uint8_t front_facing;
  uint8_t hit_kind;
  uint8_t geometry_ray_mask;
  uint8_t geometry_flags;
  uint32_t shader_index;
  uint32_t geometry_index;
  uint32_t primitive_index;
  uint32_t raw_quad_control;
  uint32_t object_t_bits;
  uint32_t world_t_bits;
  uint32_t bary_vertex1_bits;
  uint32_t bary_vertex2_bits;
  uint32_t determinant_bits;
  uint32_t reserved_zero;
};

static_assert(sizeof(raw_payload_header_v0) == 8,
              "raw payload header must remain 8 bytes");
static_assert(sizeof(raw_primitive_payload_v0) == 72,
              "GEN_RT raw primitive envelope must remain 72 bytes");
static_assert(sizeof(ray_state_v0) == 32,
              "local typed object ray input must remain 32 bytes");
static_assert(sizeof(candidate_input_v0) == 128,
              "typed primitive candidate input must remain 128 bytes");
static_assert(sizeof(candidate_result_v0) == 48,
              "typed primitive candidate result must remain 48 bytes");

bool make_raw_primitive_payload(const void *raw_primitive_bytes,
                                raw_primitive_payload_v0 *payload);

candidate_result_v0 execute(const candidate_input_v0 &input);

const char *status_name(status_kind status);

}  // namespace typed_primitive
}  // namespace v04
}  // namespace rtcore

#endif
