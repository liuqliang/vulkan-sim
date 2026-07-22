#include "rtcore_v04_typed_instance_kernel.h"

#include <cmath>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace typed_instance {
namespace {

static bool bytes_are_zero(const void *data, size_t size) {
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  for (size_t index = 0; index < size; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static uint32_t read_le_u32(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

static uint64_t read_le_u64(const uint8_t *bytes) {
  return static_cast<uint64_t>(read_le_u32(bytes)) |
         (static_cast<uint64_t>(read_le_u32(bytes + 4)) << 32);
}

static bool finite_fp32_bits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return std::isfinite(value);
}

}  // namespace

bool make_raw_instance_payload(const void *raw_instance_bytes,
                               raw_instance_payload_v0 *payload) {
  if (raw_instance_bytes == NULL || payload == NULL) return false;
  std::memset(payload, 0, sizeof(*payload));
  payload->header.expected_payload_kind = kInstancePayloadKind;
  payload->header.expected_chunk_count = 4;
  payload->header.payload_byte_count = 128;
  payload->header.received_chunk_mask = 0x0f;
  std::memcpy(payload->raw_bytes, raw_instance_bytes,
              sizeof(payload->raw_bytes));
  return true;
}

boundary_result_v0 execute(const boundary_input_v0 &input) {
  boundary_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (!bytes_are_zero(input.reserved_zero, sizeof(input.reserved_zero))) {
    return result;
  }

  const raw_payload_header_v0 &header = input.raw_instance.header;
  if (header.expected_payload_kind != kInstancePayloadKind ||
      header.expected_chunk_count != 4 || header.payload_byte_count != 128 ||
      header.received_chunk_mask != 0x0f ||
      !bytes_are_zero(header.reserved_zero, sizeof(header.reserved_zero))) {
    result.status = kStatusMalformedEnvelope;
    return result;
  }

  const uint8_t *raw = input.raw_instance.raw_bytes;
  const uint32_t descriptor0 = read_le_u32(raw + 0);
  const uint32_t descriptor1 = read_le_u32(raw + 4);
  result.shader_index = descriptor0 & 0x00ffffffu;
  result.geometry_ray_mask = static_cast<uint8_t>(descriptor0 >> 24);
  result.instance_sbt_contribution = descriptor1 & 0x00ffffffu;
  result.leaf_type = static_cast<uint8_t>((descriptor1 >> 29) & 0x1u);
  result.geometry_flags = static_cast<uint8_t>((descriptor1 >> 30) & 0x3u);
  if ((descriptor1 & 0x1f000000u) != 0 || result.leaf_type != 0u ||
      result.geometry_flags > 1u) {
    result.status = kStatusMalformedLeaf;
    return result;
  }

  const uint64_t start_control = read_le_u64(raw + 8);
  result.start_node_address = start_control & 0x0000ffffffffffffULL;
  result.instance_flags = static_cast<uint8_t>((start_control >> 48) & 0xffu);
  if ((start_control >> 56) != 0 || (result.instance_flags & ~0x0fu) != 0) {
    result.status = kStatusUnsupportedLeafEncoding;
    return result;
  }

  result.bvh_address = read_le_u64(raw + 64);
  result.instance_custom_index = read_le_u32(raw + 72);
  result.instance_index = read_le_u32(raw + 76);

  for (unsigned element = 0; element < 9; ++element) {
    result.world_to_object_bits[element] = read_le_u32(raw + 16 + element * 4);
    result.object_to_world_bits[element] = read_le_u32(raw + 80 + element * 4);
  }
  for (unsigned element = 0; element < 3; ++element) {
    result.world_to_object_bits[9 + element] =
        read_le_u32(raw + 116 + element * 4);
    result.object_to_world_bits[9 + element] =
        read_le_u32(raw + 52 + element * 4);
  }
  for (unsigned element = 0; element < kMatrixElementCount; ++element) {
    if (!finite_fp32_bits(result.world_to_object_bits[element]) ||
        !finite_fp32_bits(result.object_to_world_bits[element])) {
      result.status = kStatusInvalidNumericInput;
      return result;
    }
  }

  result.status = kStatusOk;
  return result;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusUnsupportedProfile:
      return "unsupported_profile";
    case kStatusMalformedEnvelope:
      return "malformed_envelope";
    case kStatusMalformedLeaf:
      return "malformed_leaf";
    case kStatusUnsupportedLeafEncoding:
      return "unsupported_leaf_encoding";
    case kStatusInvalidNumericInput:
      return "invalid_numeric_input";
  }
  return "unknown";
}

}  // namespace typed_instance
}  // namespace v04
}  // namespace rtcore
