#include "rtcore_v04_typed_instance_kernel.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "rtcore_v04_canonical_ray.h"

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

static float fp32_from_bits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static float fp32_add(float lhs, float rhs) { return lhs + rhs; }
static float fp32_sub(float lhs, float rhs) { return lhs - rhs; }
static float fp32_mul(float lhs, float rhs) { return lhs * rhs; }

static bool valid_inverse_component(float direction, float inverse) {
  return canonical_ray::inverse_direction_matches(direction, inverse);
}

static bool valid_ray(const mutable_ray_state_v0 &ray) {
  bool any_direction = false;
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(ray.origin[component]) ||
        !std::isfinite(ray.direction[component]) ||
        !valid_inverse_component(ray.direction[component],
                                 ray.inverse_direction[component])) {
      return false;
    }
    any_direction = any_direction || ray.direction[component] != 0.0f;
  }
  return any_direction && std::isfinite(ray.t_min) &&
         std::isfinite(ray.t_max) && ray.t_min <= ray.t_max;
}

static bool valid_decode_context(
    const typed_blas::as_decode_context_v0 &context, uint8_t as_type,
    uint64_t minimum_range_bytes) {
  return context.bvh_format_profile_id == kGenRtDerivedProfileId &&
         context.reserved_zero == 0 &&
         context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         context.as_object.as_type == as_type &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         context.device_base != 0 &&
         context.device_range_bytes >= minimum_range_bytes &&
         context.device_range_bytes <=
             std::numeric_limits<uint64_t>::max() - context.device_base;
}

static bool valid_instance_reference(
    const instance_blas_reference_v0 &reference,
    const typed_blas::as_decode_context_v0 &tlas_context,
    const typed_blas::as_decode_context_v0 &blas_context) {
  if (reference.valid != 1 ||
      !bytes_are_zero(reference.reserved_zero,
                      sizeof(reference.reserved_zero)) ||
      reference.tlas_object_id != tlas_context.as_object.object_id ||
      reference.tlas_generation != tlas_context.as_object.generation ||
      reference.tlas_build_generation == 0 ||
      reference.blas_object_id != blas_context.as_object.object_id ||
      reference.blas_generation != blas_context.as_object.generation ||
      reference.instance_metadata_reference == 0 ||
      (reference.instance_metadata_reference & uint64_t{0x3f}) != 0 ||
      reference.instance_metadata_reference < tlas_context.device_base) {
    return false;
  }
  const uint64_t offset =
      reference.instance_metadata_reference - tlas_context.device_base;
  return offset <= tlas_context.device_range_bytes &&
         uint64_t{128} <= tlas_context.device_range_bytes - offset;
}

static bool valid_parent_frame(
    const typed_stack::traversal_frame_projection_v0 &frame) {
  bool any_direction = false;
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(frame.ray.origin[component]) ||
        !std::isfinite(frame.ray.direction[component]) ||
        !valid_inverse_component(frame.ray.direction[component],
                                 frame.ray.inverse_direction[component])) {
      return false;
    }
    any_direction =
        any_direction || frame.ray.direction[component] != 0.0f;
  }
  const typed_blas::as_decode_context_v0 &context =
      frame.current_decode_context;
  return any_direction && std::isfinite(frame.ray.t_min) &&
         std::isfinite(frame.ray.t_max) &&
         frame.ray.t_min <= frame.ray.t_max &&
         frame.traversal_level == 0 &&
         frame.frontier_marker.frontier_top ==
             frame.frontier_marker.frontier_count &&
         frame.frontier_marker.frontier_top <= 16 &&
         frame.frontier_marker.level_frame_depth == 0 &&
         frame.frontier_marker.reserved_zero == 0 &&
         context.bvh_format_profile_id == kGenRtDerivedProfileId &&
         context.reserved_zero == 0 &&
         context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         context.as_object.as_type == kAsTypeTlas &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         context.device_base != 0 &&
         context.device_range_bytes >= 64 &&
         bytes_are_zero(frame.current_instance.reserved_zero,
                        sizeof(frame.current_instance.reserved_zero));
}

static bool nondegenerate_matrix(const uint32_t bits[kMatrixElementCount]) {
  const float m00 = fp32_from_bits(bits[0]);
  const float m01 = fp32_from_bits(bits[1]);
  const float m02 = fp32_from_bits(bits[2]);
  const float m10 = fp32_from_bits(bits[3]);
  const float m11 = fp32_from_bits(bits[4]);
  const float m12 = fp32_from_bits(bits[5]);
  const float m20 = fp32_from_bits(bits[6]);
  const float m21 = fp32_from_bits(bits[7]);
  const float m22 = fp32_from_bits(bits[8]);

  const float minor0 = fp32_sub(fp32_mul(m11, m22), fp32_mul(m12, m21));
  const float minor1 = fp32_sub(fp32_mul(m01, m22), fp32_mul(m02, m21));
  const float minor2 = fp32_sub(fp32_mul(m01, m12), fp32_mul(m02, m11));
  const float term0 = fp32_mul(m00, minor0);
  const float term1 = fp32_mul(m10, minor1);
  const float term2 = fp32_mul(m20, minor2);
  const float determinant =
      fp32_add(fp32_sub(term0, term1), term2);
  return std::isfinite(determinant) && determinant != 0.0f;
}

static bool transform_ray(
    const mutable_ray_state_v0 &world_ray,
    const uint32_t matrix_bits[kMatrixElementCount],
    mutable_ray_state_v0 *object_ray) {
  if (object_ray == NULL) return false;
  float matrix[kMatrixElementCount] = {};
  for (unsigned element = 0; element < kMatrixElementCount; ++element) {
    matrix[element] = fp32_from_bits(matrix_bits[element]);
  }

  float origin[3] = {};
  float direction[3] = {};
  for (unsigned component = 0; component < 3; ++component) {
    const float origin_product0 =
        fp32_mul(matrix[component], world_ray.origin[0]);
    float origin_accumulator =
        fp32_add(matrix[9 + component], origin_product0);
    const float origin_product1 =
        fp32_mul(matrix[3 + component], world_ray.origin[1]);
    origin_accumulator = fp32_add(origin_accumulator, origin_product1);
    const float origin_product2 =
        fp32_mul(matrix[6 + component], world_ray.origin[2]);
    origin[component] = fp32_add(origin_accumulator, origin_product2);

    float direction_accumulator =
        fp32_mul(matrix[component], world_ray.direction[0]);
    const float direction_product1 =
        fp32_mul(matrix[3 + component], world_ray.direction[1]);
    direction_accumulator =
        fp32_add(direction_accumulator, direction_product1);
    const float direction_product2 =
        fp32_mul(matrix[6 + component], world_ray.direction[2]);
    direction[component] =
        fp32_add(direction_accumulator, direction_product2);
  }
  return make_mutable_ray_state(origin, direction, world_ray.t_min,
                                world_ray.t_max, object_ray);
}

}  // namespace

bool validate_enter_transition_binding(const enter_input_v0 &input) {
  if (!valid_decode_context(
          input.tlas_decode_context, kAsTypeTlas, 128) ||
      !valid_decode_context(
          input.blas_decode_context, kAsTypeBlas, 64) ||
      !valid_instance_reference(
          input.instance_blas_reference, input.tlas_decode_context,
          input.blas_decode_context)) {
    return false;
  }
  typed_blas::root_binding_input_v0 root_input = {};
  root_input.decode_context = input.blas_decode_context;
  root_input.root_descriptor = input.blas_root_descriptor;
  const typed_blas::root_binding_result_v0 root =
      typed_blas::execute_root_binding(root_input);
  return root.status == typed_blas::kStatusOk &&
         root.root_payload_kind_valid == 1;
}

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

bool make_mutable_ray_state(const float origin[3], const float direction[3],
                            float t_min, float t_max,
                            mutable_ray_state_v0 *ray) {
  if (origin == NULL || direction == NULL || ray == NULL) return false;
  std::memset(ray, 0, sizeof(*ray));
  bool any_direction = false;
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(origin[component]) ||
        !std::isfinite(direction[component])) {
      return false;
    }
    ray->origin[component] = origin[component];
    ray->direction[component] = direction[component];
    any_direction = any_direction || direction[component] != 0.0f;
    ray->inverse_direction[component] =
        canonical_ray::inverse_direction(direction[component]);
  }
  if (!any_direction || !std::isfinite(t_min) || !std::isfinite(t_max) ||
      t_min > t_max) {
    return false;
  }
  ray->t_min = t_min;
  ray->t_max = t_max;
  return true;
}

enter_result_v0 execute_enter(const enter_input_v0 &input) {
  enter_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (input.current_level != kLevelTlas ||
      !bytes_are_zero(input.reserved_zero0,
                      sizeof(input.reserved_zero0)) ||
      input.reserved_zero1 != 0 ||
      !bytes_are_zero(input.policy.reserved_zero,
                      sizeof(input.policy.reserved_zero))) {
    return result;
  }
  if (!valid_ray(input.world_ray)) {
    result.status = kStatusInvalidRay;
    return result;
  }

  boundary_input_v0 decode_input = {};
  decode_input.profile_id = input.profile_id;
  decode_input.raw_instance = input.raw_instance;
  const boundary_result_v0 decoded = execute(decode_input);
  if (decoded.status != kStatusOk) {
    result.status = decoded.status;
    return result;
  }

  result.mask_visible =
      (decoded.geometry_ray_mask & input.policy.cull_mask) != 0 ? 1 : 0;
  if (result.mask_visible == 0) {
    result.status = kStatusOk;
    result.result_kind = kEnterResultCulled;
    return result;
  }

  if (!nondegenerate_matrix(decoded.world_to_object_bits) ||
      !nondegenerate_matrix(decoded.object_to_world_bits)) {
    result.status = kStatusDegenerateTransform;
    return result;
  }
  if (!transform_ray(input.world_ray, decoded.world_to_object_bits,
                     &result.object_ray)) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }

  if (!valid_decode_context(input.tlas_decode_context, kAsTypeTlas, 128) ||
      !valid_decode_context(input.blas_decode_context, kAsTypeBlas, 64) ||
      !valid_instance_reference(input.instance_blas_reference,
                                input.tlas_decode_context,
                                input.blas_decode_context)) {
    result.status = kStatusInvalidTransitionBinding;
    return result;
  }

  typed_blas::root_binding_input_v0 root_input = {};
  root_input.decode_context = input.blas_decode_context;
  root_input.root_descriptor = input.blas_root_descriptor;
  const typed_blas::root_binding_result_v0 root =
      typed_blas::execute_root_binding(root_input);
  if (root.status != typed_blas::kStatusOk ||
      root.root_payload_kind_valid != 1) {
    result.status = kStatusInvalidTransitionBinding;
    return result;
  }

  result.instance_projection.instance_metadata_reference =
      input.instance_blas_reference.instance_metadata_reference;
  result.instance_projection.shader_index = decoded.shader_index;
  result.instance_projection.instance_sbt_contribution =
      decoded.instance_sbt_contribution;
  result.instance_projection.instance_custom_index =
      decoded.instance_custom_index;
  result.instance_projection.instance_index = decoded.instance_index;
  result.instance_projection.instance_flags = decoded.instance_flags;
  result.instance_projection.geometry_flags = decoded.geometry_flags;
  result.instance_projection.geometry_ray_mask = decoded.geometry_ray_mask;

  result.root_fetch.encoded_reference = root.root_payload_offset;
  result.root_fetch.build_generation = root.build_generation;
  result.root_fetch.expected_payload_kind = root.root_payload_kind;
  result.root_fetch.decode_context = input.blas_decode_context;

  result.status = kStatusOk;
  result.result_kind = kEnterResultBlasRoot;
  result.output_valid_mask =
      kObjectRayValid | kInstanceProjectionValid | kRootFetchValid;
  return result;
}

restore_parent_result_v0 execute_restore_parent(
    const restore_parent_input_v0 &input) {
  restore_parent_result_v0 result = {};
  result.status = kStatusInvalidArgument;
  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (input.operation_kind != kRestoreParent ||
      !bytes_are_zero(input.reserved_zero, sizeof(input.reserved_zero))) {
    return result;
  }
  if (!valid_parent_frame(input.parent_frame)) {
    result.status = kStatusInvalidParentFrame;
    return result;
  }
  result.status = kStatusOk;
  result.result_kind = kInstanceParentRestored;
  result.output_valid_mask = kParentStateRestoredValid;
  result.restored_parent = input.parent_frame;
  return result;
}

bool validate_restore_parent_result(
    const restore_parent_input_v0 &input,
    const restore_parent_result_v0 &result) {
  return input.profile_id == kGenRtDerivedProfileId &&
         input.operation_kind == kRestoreParent &&
         bytes_are_zero(input.reserved_zero, sizeof(input.reserved_zero)) &&
         valid_parent_frame(input.parent_frame) &&
         result.status == kStatusOk &&
         result.result_kind == kInstanceParentRestored &&
         result.output_valid_mask == kParentStateRestoredValid &&
         bytes_are_zero(result.reserved_zero,
                        sizeof(result.reserved_zero)) &&
         std::memcmp(&result.restored_parent, &input.parent_frame,
                     sizeof(input.parent_frame)) == 0;
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
    case kStatusInvalidRay:
      return "invalid_ray";
    case kStatusDegenerateTransform:
      return "degenerate_transform";
    case kStatusInvalidTransitionBinding:
      return "invalid_transition_binding";
    case kStatusInvalidParentFrame:
      return "invalid_parent_frame";
  }
  return "unknown";
}

}  // namespace typed_instance
}  // namespace v04
}  // namespace rtcore
