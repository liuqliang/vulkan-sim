#ifndef RTCORE_V04_TYPED_INSTANCE_KERNEL_H
#define RTCORE_V04_TYPED_INSTANCE_KERNEL_H

#include <cstddef>
#include <cstdint>

namespace rtcore {
namespace v04 {
namespace typed_instance {

static const uint32_t kGenRtDerivedProfileId = 0x00010001u;
static const uint8_t kInstancePayloadKind = 1u;
static const unsigned kMatrixElementCount = 12;

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

bool make_raw_instance_payload(const void *raw_instance_bytes,
                               raw_instance_payload_v0 *payload);

boundary_result_v0 execute(const boundary_input_v0 &input);

const char *status_name(status_kind status);

}  // namespace typed_instance
}  // namespace v04
}  // namespace rtcore

#endif
