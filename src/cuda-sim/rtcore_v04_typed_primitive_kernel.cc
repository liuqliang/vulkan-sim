#include "rtcore_v04_typed_primitive_kernel.h"

#include <cmath>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace typed_primitive {
namespace {

static const uint32_t kSingleTriangleQuadControl = 1u << 22;

struct fp32_vec3 {
  float x;
  float y;
  float z;
};

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

static float read_le_fp32(const uint8_t *bytes) {
  const uint32_t bits = read_le_u32(bytes);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float fp32_add(float lhs, float rhs) { return lhs + rhs; }
static float fp32_sub(float lhs, float rhs) { return lhs - rhs; }
static float fp32_mul(float lhs, float rhs) { return lhs * rhs; }
static float fp32_div(float lhs, float rhs) { return lhs / rhs; }

static fp32_vec3 subtract(const fp32_vec3 &lhs, const fp32_vec3 &rhs) {
  const fp32_vec3 result = {fp32_sub(lhs.x, rhs.x),
                            fp32_sub(lhs.y, rhs.y),
                            fp32_sub(lhs.z, rhs.z)};
  return result;
}

static fp32_vec3 cross(const fp32_vec3 &lhs, const fp32_vec3 &rhs) {
  const fp32_vec3 result = {
      fp32_sub(fp32_mul(lhs.y, rhs.z), fp32_mul(lhs.z, rhs.y)),
      fp32_sub(fp32_mul(lhs.z, rhs.x), fp32_mul(lhs.x, rhs.z)),
      fp32_sub(fp32_mul(lhs.x, rhs.y), fp32_mul(lhs.y, rhs.x))};
  return result;
}

static float dot(const fp32_vec3 &lhs, const fp32_vec3 &rhs) {
  const float xy = fp32_add(fp32_mul(lhs.x, rhs.x),
                            fp32_mul(lhs.y, rhs.y));
  return fp32_add(xy, fp32_mul(lhs.z, rhs.z));
}

static bool finite_vec3(const fp32_vec3 &value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

static bool finite_ray(const ray_state_v0 &ray) {
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(ray.origin[axis]) ||
        !std::isfinite(ray.direction[axis])) {
      return false;
    }
  }
  return std::isfinite(ray.t_min) && std::isfinite(ray.t_max) &&
         ray.t_min <= ray.t_max;
}

}  // namespace

bool make_raw_primitive_payload(const void *raw_primitive_bytes,
                                raw_primitive_payload_v0 *payload) {
  if (raw_primitive_bytes == NULL || payload == NULL) return false;
  std::memset(payload, 0, sizeof(*payload));
  payload->header.expected_payload_kind = kQuadPayloadKind;
  payload->header.expected_chunk_count = 2;
  payload->header.payload_byte_count = 64;
  payload->header.received_chunk_mask = 0x03;
  std::memcpy(payload->raw_bytes, raw_primitive_bytes,
              sizeof(payload->raw_bytes));
  return true;
}

candidate_result_v0 execute(const candidate_input_v0 &input) {
  candidate_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }

  const raw_payload_header_v0 &header = input.raw_primitive.header;
  if (header.expected_payload_kind != kQuadPayloadKind ||
      header.expected_chunk_count != 2 || header.payload_byte_count != 64 ||
      header.received_chunk_mask != 0x03 ||
      !bytes_are_zero(header.reserved_zero, sizeof(header.reserved_zero))) {
    result.status = kStatusMalformedEnvelope;
    return result;
  }
  if (!finite_ray(input.object_ray) ||
      !std::isfinite(input.world_to_object_t_multiplier) ||
      input.world_to_object_t_multiplier <= 0.0f ||
      !std::isfinite(input.committed_world_t) ||
      !std::isfinite(input.world_t_min) ||
      !std::isfinite(input.world_t_max) ||
      input.world_t_min > input.world_t_max) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }

  const uint8_t *raw = input.raw_primitive.raw_bytes;
  const uint32_t descriptor0 = read_le_u32(raw + 0);
  const uint32_t descriptor1 = read_le_u32(raw + 4);
  result.shader_index = descriptor0 & 0x00ffffffu;
  result.geometry_ray_mask = static_cast<uint8_t>(descriptor0 >> 24);
  result.geometry_index = descriptor1 & 0x1fffffffu;
  const uint8_t leaf_type = static_cast<uint8_t>((descriptor1 >> 29) & 0x1u);
  result.geometry_flags = static_cast<uint8_t>((descriptor1 >> 30) & 0x3u);
  result.primitive_index = read_le_u32(raw + 8);
  result.raw_quad_control = read_le_u32(raw + 12);

  if (leaf_type != 0 || result.geometry_flags > 1u) {
    result.status = kStatusMalformedLeaf;
    return result;
  }
  if (result.raw_quad_control != kSingleTriangleQuadControl) {
    result.status = kStatusUnsupportedLeafEncoding;
    return result;
  }

  fp32_vec3 vertices[4] = {};
  for (unsigned vertex = 0; vertex < 4; ++vertex) {
    const uint8_t *vertex_bytes = raw + 16 + vertex * 12;
    vertices[vertex].x = read_le_fp32(vertex_bytes + 0);
    vertices[vertex].y = read_le_fp32(vertex_bytes + 4);
    vertices[vertex].z = read_le_fp32(vertex_bytes + 8);
    if (!finite_vec3(vertices[vertex])) {
      result.status = kStatusInvalidNumericInput;
      return result;
    }
  }

  const fp32_vec3 ray_origin = {input.object_ray.origin[0],
                                input.object_ray.origin[1],
                                input.object_ray.origin[2]};
  const fp32_vec3 ray_direction = {input.object_ray.direction[0],
                                   input.object_ray.direction[1],
                                   input.object_ray.direction[2]};
  const fp32_vec3 edge1 = subtract(vertices[1], vertices[0]);
  const fp32_vec3 edge2 = subtract(vertices[2], vertices[0]);
  const fp32_vec3 pvec = cross(ray_direction, edge2);
  const float determinant = dot(edge1, pvec);
  if (!finite_vec3(edge1) || !finite_vec3(edge2) || !finite_vec3(pvec) ||
      !std::isfinite(determinant)) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }
  result.determinant_bits = fp32_bits(determinant);
  if (determinant == 0.0f) {
    result.status = kStatusOk;
    return result;
  }

  const float inverse_determinant = fp32_div(1.0f, determinant);
  if (!std::isfinite(inverse_determinant)) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }
  const fp32_vec3 tvec = subtract(ray_origin, vertices[0]);
  const float bary_vertex1 = fp32_mul(dot(tvec, pvec), inverse_determinant);
  if (!finite_vec3(tvec) || !std::isfinite(bary_vertex1)) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }
  if (bary_vertex1 < 0.0f || bary_vertex1 > 1.0f) {
    result.status = kStatusOk;
    return result;
  }

  const fp32_vec3 qvec = cross(tvec, edge1);
  const float bary_vertex2 =
      fp32_mul(dot(ray_direction, qvec), inverse_determinant);
  const float bary_sum = fp32_add(bary_vertex1, bary_vertex2);
  if (!finite_vec3(qvec) || !std::isfinite(bary_vertex2) ||
      !std::isfinite(bary_sum)) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }
  if (bary_vertex2 < 0.0f || bary_sum > 1.0f) {
    result.status = kStatusOk;
    return result;
  }

  const float object_t = fp32_mul(dot(edge2, qvec), inverse_determinant);
  const float world_t =
      fp32_div(object_t, input.world_to_object_t_multiplier);
  if (!std::isfinite(object_t) || !std::isfinite(world_t)) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }

  const bool counter_clockwise = determinant > 0.0f;
  const bool front_facing =
      (input.instance_flags & kTriangleFrontCounterclockwise) != 0
          ? counter_clockwise
          : !counter_clockwise;
  result.geometric_hit = 1;
  result.counter_clockwise_facing = counter_clockwise ? 1 : 0;
  result.front_facing = front_facing ? 1 : 0;
  result.hit_kind =
      front_facing ? kHitKindFrontFacing : kHitKindBackFacing;
  result.object_t_bits = fp32_bits(object_t);
  result.world_t_bits = fp32_bits(world_t);
  result.bary_vertex1_bits = fp32_bits(bary_vertex1);
  result.bary_vertex2_bits = fp32_bits(bary_vertex2);
  result.candidate_hit =
      world_t >= input.world_t_min && world_t <= input.world_t_max &&
              world_t < input.committed_world_t
          ? 1
          : 0;
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

}  // namespace typed_primitive
}  // namespace v04
}  // namespace rtcore
