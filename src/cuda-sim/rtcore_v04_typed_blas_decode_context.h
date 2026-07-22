#ifndef RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT_H
#define RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT_H

#include <cstddef>
#include <cstdint>

namespace rtcore {
namespace v04 {
namespace typed_blas {

static const uint32_t kGenRtDerivedProfileId = 0x00010001u;
static const uint32_t kGenRtPayloadFormatId = 0x00000001u;
static const uint8_t kAsTypeBlas = 2u;
static const uint8_t kInternalPayloadKind = 0u;
static const uint8_t kProceduralPayloadKind = 3u;
static const uint8_t kQuadPayloadKind = 4u;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedProfile,
  kStatusInvalidBinding,
  kStatusMalformedEnvelope,
  kStatusMalformedHeader,
  kStatusInvalidNumericInput,
  kStatusInvalidRootDescriptor,
};

struct as_object_identity_v0 {
  uint64_t object_id;
  uint32_t generation;
  uint8_t as_type;
  uint8_t reserved_zero[3];
};

struct as_decode_context_v0 {
  uint32_t bvh_format_profile_id;
  uint32_t reserved_zero;
  as_object_identity_v0 as_object;
  uint64_t device_base;
  uint64_t device_range_bytes;
};

struct binding_input_v0 {
  uint64_t object_id;
  uint32_t generation;
  uint8_t as_type;
  uint8_t reserved_zero[3];
  uint64_t device_base;
  uint64_t device_range_bytes;
};

struct raw_header_envelope_v0 {
  uint8_t expected_chunk_count;
  uint8_t received_chunk_mask;
  uint16_t payload_byte_count;
  uint32_t reserved_zero;
};

struct raw_bvh_header_v0 {
  raw_header_envelope_v0 envelope;
  uint8_t raw_bytes[64];
};

struct alignas(16) boundary_input_v0 {
  uint32_t profile_id;
  uint32_t reserved_zero[3];
  binding_input_v0 binding;
  raw_bvh_header_v0 raw_header;
};

struct alignas(16) boundary_result_v0 {
  uint8_t status;
  uint8_t root_payload_kind_valid;
  uint8_t reserved_zero[6];
  as_decode_context_v0 decode_context;
  uint64_t root_payload_offset;
  uint32_t bounds_min_bits[3];
  uint32_t bounds_max_bits[3];
};

struct root_descriptor_v0 {
  uint64_t object_id;
  uint64_t root_payload_offset;
  uint32_t object_generation;
  uint32_t build_generation;
  uint32_t bvh_format_profile_id;
  uint32_t payload_format_id;
  uint8_t as_type;
  uint8_t root_payload_kind;
  uint8_t valid;
  uint8_t reserved_zero[5];
};

struct alignas(16) root_binding_input_v0 {
  as_decode_context_v0 decode_context;
  root_descriptor_v0 root_descriptor;
};

struct alignas(16) root_binding_result_v0 {
  uint8_t status;
  uint8_t root_payload_kind_valid;
  uint8_t root_payload_kind;
  uint8_t reserved_zero0[5];
  uint64_t root_device_address;
  uint64_t root_payload_offset;
  uint32_t build_generation;
  uint32_t reserved_zero1;
};

static_assert(sizeof(as_object_identity_v0) == 16,
              "AS object identity must remain 16 bytes");
static_assert(sizeof(as_decode_context_v0) == 40,
              "AS decode context must remain 40 bytes");
static_assert(sizeof(binding_input_v0) == 32,
              "BLAS binding input must remain 32 bytes");
static_assert(sizeof(raw_header_envelope_v0) == 8,
              "raw header envelope must remain 8 bytes");
static_assert(sizeof(raw_bvh_header_v0) == 72,
              "raw GEN_RT BVH header envelope must remain 72 bytes");
static_assert(sizeof(boundary_input_v0) == 128,
              "typed BLAS context input must remain 128 bytes");
static_assert(alignof(boundary_input_v0) == 16,
              "typed BLAS context input must remain 16-byte aligned");
static_assert(offsetof(boundary_input_v0, binding) == 16,
              "typed BLAS binding offset changed");
static_assert(offsetof(boundary_input_v0, raw_header) == 48,
              "typed BLAS raw header offset changed");
static_assert(sizeof(boundary_result_v0) == 80,
              "typed BLAS context result must remain 80 bytes");
static_assert(offsetof(boundary_result_v0, decode_context) == 8,
              "typed BLAS decode context result offset changed");
static_assert(offsetof(boundary_result_v0, root_payload_offset) == 48,
              "typed BLAS root offset result changed");
static_assert(sizeof(root_descriptor_v0) == 40,
              "typed BLAS root descriptor must remain 40 bytes");
static_assert(offsetof(root_descriptor_v0, root_payload_offset) == 8,
              "typed BLAS root offset descriptor changed");
static_assert(offsetof(root_descriptor_v0, object_generation) == 16,
              "typed BLAS object generation descriptor changed");
static_assert(offsetof(root_descriptor_v0, as_type) == 32,
              "typed BLAS AS type descriptor changed");
static_assert(sizeof(root_binding_input_v0) == 80,
              "typed BLAS root binding input must remain 80 bytes");
static_assert(alignof(root_binding_input_v0) == 16,
              "typed BLAS root binding input must remain aligned");
static_assert(offsetof(root_binding_input_v0, root_descriptor) == 40,
              "typed BLAS root descriptor input offset changed");
static_assert(sizeof(root_binding_result_v0) == 32,
              "typed BLAS root binding result must remain 32 bytes");

bool make_raw_bvh_header(const void *raw_header_bytes,
                         uint64_t available_bytes,
                         raw_bvh_header_v0 *header);

boundary_result_v0 execute(const boundary_input_v0 &input);

root_binding_result_v0 execute_root_binding(
    const root_binding_input_v0 &input);

const char *status_name(status_kind status);

}  // namespace typed_blas
}  // namespace v04
}  // namespace rtcore

#endif
