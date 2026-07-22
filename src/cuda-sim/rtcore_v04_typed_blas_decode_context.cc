#include "rtcore_v04_typed_blas_decode_context.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace typed_blas {
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

static float fp32_from_bits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

bool make_raw_bvh_header(const void *raw_header_bytes,
                         uint64_t available_bytes,
                         raw_bvh_header_v0 *header) {
  if (raw_header_bytes == NULL || header == NULL ||
      available_bytes < sizeof(header->raw_bytes)) {
    return false;
  }
  std::memset(header, 0, sizeof(*header));
  header->envelope.expected_chunk_count = 2;
  header->envelope.received_chunk_mask = 0x03;
  header->envelope.payload_byte_count = 64;
  std::memcpy(header->raw_bytes, raw_header_bytes,
              sizeof(header->raw_bytes));
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

  const binding_input_v0 &binding = input.binding;
  if (binding.object_id == 0 || binding.generation == 0 ||
      binding.as_type != kAsTypeBlas ||
      !bytes_are_zero(binding.reserved_zero,
                      sizeof(binding.reserved_zero)) ||
      binding.device_base == 0 || binding.device_range_bytes < 64 ||
      binding.device_range_bytes >
          std::numeric_limits<uint64_t>::max() - binding.device_base) {
    result.status = kStatusInvalidBinding;
    return result;
  }

  const raw_header_envelope_v0 &envelope = input.raw_header.envelope;
  if (envelope.expected_chunk_count != 2 ||
      envelope.received_chunk_mask != 0x03 ||
      envelope.payload_byte_count != 64 || envelope.reserved_zero != 0) {
    result.status = kStatusMalformedEnvelope;
    return result;
  }

  const uint8_t *raw = input.raw_header.raw_bytes;
  result.root_payload_offset = read_le_u64(raw);
  if (result.root_payload_offset < 64 ||
      (result.root_payload_offset & uint64_t{0x3f}) != 0 ||
      result.root_payload_offset > binding.device_range_bytes ||
      64 > binding.device_range_bytes - result.root_payload_offset) {
    result.status = kStatusMalformedHeader;
    return result;
  }

  for (unsigned component = 0; component < 3; ++component) {
    result.bounds_min_bits[component] = read_le_u32(raw + 8 + component * 4);
    result.bounds_max_bits[component] = read_le_u32(raw + 20 + component * 4);
    if (!finite_fp32_bits(result.bounds_min_bits[component]) ||
        !finite_fp32_bits(result.bounds_max_bits[component])) {
      result.status = kStatusInvalidNumericInput;
      return result;
    }
    if (fp32_from_bits(result.bounds_min_bits[component]) >
        fp32_from_bits(result.bounds_max_bits[component])) {
      result.status = kStatusMalformedHeader;
      return result;
    }
  }

  result.decode_context.bvh_format_profile_id = input.profile_id;
  result.decode_context.as_object.object_id = binding.object_id;
  result.decode_context.as_object.generation = binding.generation;
  result.decode_context.as_object.as_type = binding.as_type;
  result.decode_context.device_base = binding.device_base;
  result.decode_context.device_range_bytes = binding.device_range_bytes;
  result.root_payload_kind_valid = 0;
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
    case kStatusInvalidBinding:
      return "invalid_binding";
    case kStatusMalformedEnvelope:
      return "malformed_envelope";
    case kStatusMalformedHeader:
      return "malformed_header";
    case kStatusInvalidNumericInput:
      return "invalid_numeric_input";
  }
  return "unknown";
}

}  // namespace typed_blas
}  // namespace v04
}  // namespace rtcore
