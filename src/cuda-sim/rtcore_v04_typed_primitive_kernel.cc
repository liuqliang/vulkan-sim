#include "rtcore_v04_typed_primitive_kernel.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "rtcore_v04_canonical_ray.h"

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

static bool valid_route_ray(const typed_stack::mutable_ray_state_v0 &ray) {
  bool any_direction = false;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(ray.origin[axis]) ||
        !canonical_ray::inverse_direction_matches(
            ray.direction[axis], ray.inverse_direction[axis])) {
      return false;
    }
    any_direction = any_direction || ray.direction[axis] != 0.0f;
  }
  return any_direction && std::isfinite(ray.t_min) &&
         std::isfinite(ray.t_max) && ray.t_min <= ray.t_max;
}

static bool valid_decode_context(
    const typed_blas::as_decode_context_v0 &context,
    uint64_t leaf_fetch_address) {
  if (context.bvh_format_profile_id != kGenRtDerivedProfileId ||
      context.reserved_zero != 0 || context.as_object.object_id == 0 ||
      context.as_object.generation == 0 ||
      context.as_object.as_type != typed_blas::kAsTypeBlas ||
      !bytes_are_zero(context.as_object.reserved_zero,
                      sizeof(context.as_object.reserved_zero)) ||
      context.device_base == 0 || context.device_range_bytes < 64 ||
      context.device_range_bytes >
          std::numeric_limits<uint64_t>::max() - context.device_base ||
      leaf_fetch_address < context.device_base) {
    return false;
  }
  const uint64_t offset = leaf_fetch_address - context.device_base;
  return offset <= context.device_range_bytes &&
         uint64_t{64} <= context.device_range_bytes - offset;
}

static bool valid_committed_hit(
    const typed_stack::committed_hit_projection_v0 &hit,
    const typed_stack::mutable_ray_state_v0 &ray) {
  if (hit.valid > 1 || hit.attribute_word_count > 4 ||
      !bytes_are_zero(hit.reserved_zero0,
                      sizeof(hit.reserved_zero0)) ||
      hit.reserved_zero1 != 0) {
    return false;
  }
  if (hit.valid == 0) {
    const typed_stack::committed_hit_projection_v0 empty = {};
    return std::memcmp(&hit, &empty, sizeof(empty)) == 0;
  }
  return (hit.geometry_type == kGeometryTypeTriangle ||
          hit.geometry_type == kGeometryTypeProcedural) &&
         std::isfinite(hit.hit_t) && hit.hit_t >= ray.t_min &&
         hit.hit_t <= ray.t_max && hit.instance_metadata_ref != 0;
}

static bool ray_policy_valid(const typed_node::ray_policy_v0 &policy) {
  const uint32_t opacity_group =
      policy.ray_flags &
      (kRayFlagOpaque | kRayFlagNoOpaque | kRayFlagCullOpaque |
       kRayFlagCullNoOpaque);
  const uint32_t triangle_group =
      policy.ray_flags &
      (kRayFlagSkipTriangles |
       kRayFlagCullFrontFacingTriangles |
       kRayFlagCullBackFacingTriangles);
  const bool opacity_conflict =
      opacity_group != 0 &&
      (opacity_group & (opacity_group - 1u)) != 0;
  const bool triangle_conflict =
      triangle_group != 0 &&
      (triangle_group & (triangle_group - 1u)) != 0;
  const bool skip_all_geometry =
      (policy.ray_flags & kRayFlagSkipTriangles) != 0 &&
      (policy.ray_flags & kRayFlagSkipAabbs) != 0;
  return (policy.ray_flags & ~kSupportedRayFlagMask) == 0 &&
         !opacity_conflict && !triangle_conflict &&
         !skip_all_geometry &&
         bytes_are_zero(policy.reserved_zero,
                        sizeof(policy.reserved_zero));
}

static bool current_instance_valid(
    const typed_stack::instance_shader_projection_v0 &instance) {
  const bool force_conflict =
      (instance.instance_policy_flags & kInstanceForceOpaque) != 0 &&
      (instance.instance_policy_flags & kInstanceForceNoOpaque) != 0;
  return instance.instance_metadata_ref != 0 &&
         (instance.instance_metadata_ref & uint64_t{0x3f}) == 0 &&
         (instance.instance_policy_flags &
          ~kSupportedInstancePolicyMask) == 0 &&
         !force_conflict &&
         bytes_are_zero(instance.reserved_zero,
                        sizeof(instance.reserved_zero));
}

static bool route_input_valid(const route_input_v0 &input) {
  return input.operation_kind == kOperationTestLeaf &&
         bytes_are_zero(input.reserved_zero,
                        sizeof(input.reserved_zero)) &&
         input.leaf_fetch_address != 0 &&
         (input.leaf_fetch_address & uint64_t{0x1f}) == 0 &&
         input.input_slot_mask == uint64_t{1} &&
         valid_route_ray(input.ray) &&
         valid_decode_context(input.decode_context,
                              input.leaf_fetch_address) &&
         valid_committed_hit(input.current_committed_hit, input.ray);
}

static bool policy_input_valid(const route_input_v0 &input) {
  return ray_policy_valid(input.ray_policy) &&
         current_instance_valid(input.current_instance) &&
         input.geometry_policy.geometry_flags <= kGeometryOpaque &&
         input.geometry_policy.pipeline_policy_bits == 0;
}

static bool effective_opaque(const route_input_v0 &input) {
  if ((input.ray_policy.ray_flags & kRayFlagOpaque) != 0) return true;
  if ((input.ray_policy.ray_flags & kRayFlagNoOpaque) != 0) return false;
  if ((input.current_instance.instance_policy_flags &
       kInstanceForceOpaque) != 0) {
    return true;
  }
  if ((input.current_instance.instance_policy_flags &
       kInstanceForceNoOpaque) != 0) {
    return false;
  }
  return (input.geometry_policy.geometry_flags & kGeometryOpaque) != 0;
}

static bool opacity_culled(const route_input_v0 &input, bool opaque) {
  return opaque
             ? (input.ray_policy.ray_flags & kRayFlagCullOpaque) != 0
             : (input.ray_policy.ray_flags & kRayFlagCullNoOpaque) != 0;
}

static primitive_identity_policy_facts_v0 make_identity_and_policy(
    const route_input_v0 &input, uint32_t primitive_index,
    uint32_t geometry_index, geometry_type_kind geometry_type,
    bool opaque) {
  primitive_identity_policy_facts_v0 facts = {};
  facts.instance_metadata_ref =
      input.current_instance.instance_metadata_ref;
  facts.primitive_index = primitive_index;
  facts.geometry_index = geometry_index;
  facts.instance_index = input.current_instance.instance_index;
  facts.instance_custom_index =
      input.current_instance.instance_custom_index;
  facts.instance_sbt_contribution =
      input.current_instance.instance_sbt_contribution;
  facts.geometry_type = geometry_type;
  facts.geometry_policy_flags =
      static_cast<uint8_t>(input.geometry_policy.geometry_flags);
  facts.instance_policy_flags =
      input.current_instance.instance_policy_flags;
  if (opaque) facts.effective_policy_flags |= kPolicyEffectiveOpaque;
  if (geometry_type == kGeometryTypeProcedural && !opaque) {
    facts.effective_policy_flags |=
        kPolicyProceduralAnyHitEligible;
  }
  return facts;
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

bool make_raw_procedural_payload(const void *raw_primitive_bytes,
                                 raw_primitive_payload_v0 *payload) {
  if (raw_primitive_bytes == NULL || payload == NULL) return false;
  std::memset(payload, 0, sizeof(*payload));
  payload->header.expected_payload_kind = kProceduralPayloadKind;
  payload->header.expected_chunk_count = 2;
  payload->header.payload_byte_count = 64;
  payload->header.received_chunk_mask = 0x03;
  std::memcpy(payload->raw_bytes, raw_primitive_bytes,
              sizeof(payload->raw_bytes));
  return true;
}

bool extract_geometry_policy(
    const raw_primitive_payload_v0 &raw_primitive,
    geometry_policy_projection_v0 *geometry_policy) {
  if (geometry_policy == NULL ||
      raw_primitive.header.expected_chunk_count != 2 ||
      raw_primitive.header.payload_byte_count != 64 ||
      raw_primitive.header.received_chunk_mask != 0x03 ||
      !bytes_are_zero(raw_primitive.header.reserved_zero,
                      sizeof(raw_primitive.header.reserved_zero)) ||
      (raw_primitive.header.expected_payload_kind !=
           kQuadPayloadKind &&
       raw_primitive.header.expected_payload_kind !=
           kProceduralPayloadKind)) {
    return false;
  }
  const uint32_t descriptor1 =
      read_le_u32(raw_primitive.raw_bytes + 4);
  const uint8_t leaf_type =
      static_cast<uint8_t>((descriptor1 >> 29) & 0x1u);
  const uint8_t expected_leaf_type =
      raw_primitive.header.expected_payload_kind ==
              kProceduralPayloadKind
          ? 1
          : 0;
  const uint8_t geometry_flags =
      static_cast<uint8_t>((descriptor1 >> 30) & 0x3u);
  if (leaf_type != expected_leaf_type ||
      geometry_flags > kGeometryOpaque) {
    return false;
  }
  *geometry_policy = geometry_policy_projection_v0();
  geometry_policy->geometry_flags = geometry_flags;
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

procedural_result_v0 execute_procedural(const procedural_input_v0 &input) {
  procedural_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (input.cull_mask > 0xffu ||
      !bytes_are_zero(input.reserved_zero, sizeof(input.reserved_zero))) {
    return result;
  }

  const raw_payload_header_v0 &header = input.raw_primitive.header;
  if (header.expected_payload_kind != kProceduralPayloadKind ||
      header.expected_chunk_count != 2 || header.payload_byte_count != 64 ||
      header.received_chunk_mask != 0x03 ||
      !bytes_are_zero(header.reserved_zero, sizeof(header.reserved_zero))) {
    result.status = kStatusMalformedEnvelope;
    return result;
  }

  const uint8_t *raw = input.raw_primitive.raw_bytes;
  const uint32_t descriptor0 = read_le_u32(raw + 0);
  const uint32_t descriptor1 = read_le_u32(raw + 4);
  result.shader_index = descriptor0 & 0x00ffffffu;
  result.geometry_ray_mask = static_cast<uint8_t>(descriptor0 >> 24);
  result.geometry_index = descriptor1 & 0x1fffffffu;
  result.leaf_type = static_cast<uint8_t>((descriptor1 >> 29) & 0x1u);
  result.geometry_flags = static_cast<uint8_t>((descriptor1 >> 30) & 0x3u);
  result.raw_control = read_le_u32(raw + 8);
  result.primitive_count = static_cast<uint8_t>(result.raw_control & 0x0fu);
  result.last_primitive =
      static_cast<uint16_t>((result.raw_control >> 19) & 0x1fffu);
  result.primitive_index = read_le_u32(raw + 12);

  if (result.leaf_type != 1u || result.geometry_flags > 1u) {
    result.status = kStatusMalformedLeaf;
    return result;
  }
  if (result.raw_control != 1u ||
      !bytes_are_zero(raw + 16, sizeof(input.raw_primitive.raw_bytes) - 16)) {
    result.status = kStatusUnsupportedLeafEncoding;
    return result;
  }

  result.mask_visible =
      (result.geometry_ray_mask & static_cast<uint8_t>(input.cull_mask)) != 0
          ? 1
          : 0;
  result.status = kStatusOk;
  return result;
}

route_result_v0 execute_route(const route_input_v0 &input) {
  route_result_v0 result = {};
  result.status = kStatusInvalidRouteInput;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (!route_input_valid(input)) return result;
  if (!policy_input_valid(input)) {
    result.status = kStatusInvalidPolicy;
    return result;
  }

  const uint8_t payload_kind =
      input.raw_primitive.header.expected_payload_kind;
  const bool opaque = effective_opaque(input);
  if (payload_kind == kQuadPayloadKind) {
    candidate_input_v0 candidate_input = {};
    candidate_input.profile_id = input.profile_id;
    candidate_input.instance_flags =
        input.current_instance.instance_policy_flags;
    std::memcpy(candidate_input.object_ray.origin, input.ray.origin,
                sizeof(candidate_input.object_ray.origin));
    std::memcpy(candidate_input.object_ray.direction,
                input.ray.direction,
                sizeof(candidate_input.object_ray.direction));
    candidate_input.object_ray.t_min = input.ray.t_min;
    candidate_input.object_ray.t_max = input.ray.t_max;
    candidate_input.world_to_object_t_multiplier = 1.0f;
    candidate_input.committed_world_t =
        input.current_committed_hit.valid != 0
            ? input.current_committed_hit.hit_t
            : input.ray.t_max;
    candidate_input.world_t_min = input.ray.t_min;
    candidate_input.world_t_max = input.ray.t_max;
    candidate_input.raw_primitive = input.raw_primitive;

    const candidate_result_v0 candidate = execute(candidate_input);
    result.typed_operator_invocation_count = 1;
    if (candidate.status != kStatusOk) {
      result.status = candidate.status;
      return result;
    }
    if (candidate.geometry_flags !=
        input.geometry_policy.geometry_flags) {
      result.status = kStatusInvalidPolicy;
      return result;
    }

    const bool visible =
        (candidate.geometry_ray_mask & input.ray_policy.cull_mask) != 0;
    const bool facing_culled =
        (input.current_instance.instance_policy_flags &
         kInstanceTriangleFacingCullDisable) == 0 &&
        ((candidate.front_facing != 0 &&
          (input.ray_policy.ray_flags &
           kRayFlagCullFrontFacingTriangles) != 0) ||
         (candidate.front_facing == 0 &&
          (input.ray_policy.ray_flags &
           kRayFlagCullBackFacingTriangles) != 0));
    const bool culled =
        !visible ||
        (input.ray_policy.ray_flags & kRayFlagSkipTriangles) != 0 ||
        facing_culled || opacity_culled(input, opaque);
    result.status = kStatusOk;
    if (candidate.candidate_hit == 0 || culled) {
      result.result_kind = kRouteResultNoCandidate;
      return result;
    }

    result.identity_and_policy = make_identity_and_policy(
        input, candidate.primitive_index, candidate.geometry_index,
        kGeometryTypeTriangle, opaque);
    result.triangle_hit.hit_t_bits = candidate.world_t_bits;
    result.triangle_hit.bary_vertex1_bits =
        candidate.bary_vertex1_bits;
    result.triangle_hit.bary_vertex2_bits =
        candidate.bary_vertex2_bits;
    result.triangle_hit.hit_kind = candidate.hit_kind;
    result.output_valid_mask = static_cast<uint8_t>(
        kIdentityAndPolicyValid | kTriangleHitValid);
    if (!opaque) {
      result.result_kind = kRouteResultAnyHitBoundary;
    } else if ((input.ray_policy.ray_flags &
                kRayFlagTerminateOnFirstHit) != 0) {
      result.result_kind = kRouteResultFinalHit;
    } else {
      result.result_kind = kRouteResultCommitOpaque;
    }
    return result;
  }

  if (payload_kind == kProceduralPayloadKind) {
    procedural_input_v0 procedural_input = {};
    procedural_input.profile_id = input.profile_id;
    procedural_input.cull_mask = input.ray_policy.cull_mask;
    procedural_input.raw_primitive = input.raw_primitive;
    const procedural_result_v0 procedural =
        execute_procedural(procedural_input);
    result.typed_operator_invocation_count = 1;
    if (procedural.status != kStatusOk) {
      result.status = procedural.status;
      return result;
    }
    if (procedural.geometry_flags !=
        input.geometry_policy.geometry_flags) {
      result.status = kStatusInvalidPolicy;
      return result;
    }
    result.status = kStatusOk;
    if (procedural.mask_visible == 0 ||
        (input.ray_policy.ray_flags & kRayFlagSkipAabbs) != 0 ||
        opacity_culled(input, opaque)) {
      result.result_kind = kRouteResultNoCandidate;
      return result;
    }
    result.result_kind = kRouteResultIntersectionBoundary;
    result.output_valid_mask = static_cast<uint8_t>(
        kIdentityAndPolicyValid | kIntersectionBoundaryValid);
    result.identity_and_policy = make_identity_and_policy(
        input, procedural.primitive_index, procedural.geometry_index,
        kGeometryTypeProcedural, opaque);
    result.intersection_boundary.boundary_ray_tmax_bits =
        fp32_bits(input.current_committed_hit.valid != 0
                      ? input.current_committed_hit.hit_t
                      : input.ray.t_max);
    return result;
  }

  result.status = kStatusMalformedEnvelope;
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
    case kStatusInvalidRouteInput:
      return "invalid_route_input";
    case kStatusInvalidPolicy:
      return "invalid_policy";
  }
  return "unknown";
}

}  // namespace typed_primitive
}  // namespace v04
}  // namespace rtcore
