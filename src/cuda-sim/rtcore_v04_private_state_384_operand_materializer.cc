#include "rtcore_v04_private_state_384_operand_materializer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "rtcore_v04_canonical_ray.h"

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace operand_materializer {
namespace {

static const uint8_t kAsTypeTlas = 1;
static const uint8_t kAsTypeBlas = 2;
static const uint32_t kCommittedPolicyMask = 0x1u;
static const uint8_t kGeometryPolicyMask = 0x1u;
static const uint8_t kEffectivePolicyMask = 0x1u;

struct committed_hot_v1 {
  uint8_t valid;
  uint8_t reserved_zero[3];
  float hit_t;
};

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static uint16_t load_u16(const uint8_t *bytes) {
  return static_cast<uint16_t>(
      static_cast<uint16_t>(bytes[0]) |
      static_cast<uint16_t>(bytes[1]) << 8);
}

static uint32_t load_u32(const uint8_t *bytes) {
  uint32_t value = 0;
  for (uint8_t index = 0; index < 4; ++index) {
    value |= static_cast<uint32_t>(bytes[index]) << (index * 8);
  }
  return value;
}

static uint64_t load_u64(const uint8_t *bytes) {
  uint64_t value = 0;
  for (uint8_t index = 0; index < 8; ++index) {
    value |= static_cast<uint64_t>(bytes[index]) << (index * 8);
  }
  return value;
}

static float fp32_value(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static bool valid_instance_policy(uint8_t policy) {
  return (policy & ~typed_primitive::kSupportedInstancePolicyMask) == 0 &&
         !((policy & typed_primitive::kInstanceForceOpaque) != 0 &&
           (policy & typed_primitive::kInstanceForceNoOpaque) != 0);
}

static bool valid_operation_identity(
    const operation_identity_v1 &identity) {
  return identity.request_identity != 0 &&
         identity.request_generation != 0 &&
         identity.operation_sequence != 0 &&
         identity.lane_id < 32 &&
         bytes_are_zero(identity.reserved_zero,
                        sizeof(identity.reserved_zero));
}

static bool same_operation_identity(
    const operation_identity_v1 &lhs,
    const operation_identity_v1 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.resident_warp_id == rhs.resident_warp_id &&
         lhs.request_identity == rhs.request_identity &&
         lhs.request_generation == rhs.request_generation &&
         lhs.private_slot_id == rhs.private_slot_id &&
         lhs.operation_sequence == rhs.operation_sequence &&
         lhs.lane_id == rhs.lane_id &&
         bytes_are_zero(lhs.reserved_zero,
                        sizeof(lhs.reserved_zero)) &&
         bytes_are_zero(rhs.reserved_zero,
                        sizeof(rhs.reserved_zero));
}

static bool collector_shape_valid(
    const response_collector_v1 &collector) {
  if (collector.initialized != 1 ||
      collector.failed > 1 ||
      !valid_operation_identity(collector.identity) ||
      collector.private_layout_profile_id !=
          kPrivateLayoutProfileId ||
      collector.required_count == 0 ||
      collector.required_count > kMaxOperationReadChunks ||
      collector.received_count > collector.required_count ||
      (collector.received_chunk_mask &
       ~collector.required_chunk_mask) != 0 ||
      collector.reserved_zero != 0) {
    return false;
  }
  operand_plan::read_request_v1 request = {};
  request.private_layout_profile_id =
      collector.private_layout_profile_id;
  request.consumer = collector.consumer;
  request.operation = collector.operation;
  request.completion_reason = collector.completion_reason;
  operand_plan::read_plan_v1 canonical = {};
  if (operand_plan::make_read_plan(request, &canonical) !=
          operand_plan::kStatusOk ||
      canonical.read_count != collector.required_count) {
    return false;
  }
  uint16_t required_mask = 0;
  for (uint8_t index = 0; index < canonical.read_count; ++index) {
    required_mask |= static_cast<uint16_t>(
        uint16_t{1} << canonical.reads[index].chunk_index);
  }
  if (required_mask != collector.required_chunk_mask) {
    return false;
  }

  uint16_t observed_mask = 0;
  for (uint8_t index = 0; index < kMaxOperationReadChunks; ++index) {
    const collected_chunk_v1 &chunk = collector.chunks[index];
    if (index >= collector.received_count) {
      if (!bytes_are_zero(
              reinterpret_cast<const uint8_t *>(&chunk),
              sizeof(chunk))) {
        return false;
      }
      continue;
    }
    if (chunk.valid != 1 ||
        !bytes_are_zero(chunk.reserved_zero,
                        sizeof(chunk.reserved_zero)) ||
        chunk.chunk_index >= kChunkCount) {
      return false;
    }
    const uint16_t bit = static_cast<uint16_t>(
        uint16_t{1} << chunk.chunk_index);
    if ((required_mask & bit) == 0 ||
        (observed_mask & bit) != 0) {
      return false;
    }
    observed_mask |= bit;
  }
  return observed_mask == collector.received_chunk_mask;
}

static const uint8_t *find_chunk(
    const response_collector_v1 &collector, uint8_t chunk_index) {
  for (uint8_t index = 0; index < collector.received_count; ++index) {
    if (collector.chunks[index].valid != 0 &&
        collector.chunks[index].chunk_index == chunk_index) {
      return collector.chunks[index].payload;
    }
  }
  return NULL;
}

static status_kind require_collector(
    const response_collector_v1 &collector, uint8_t consumer,
    uint8_t operation, uint8_t completion_reason) {
  if (collector.initialized == 0) return kStatusNotInitialized;
  if (collector.failed != 0) return kStatusCollectorFailed;
  if (!collector_shape_valid(collector)) {
    return kStatusInvalidCollector;
  }
  if (collector.private_layout_profile_id != kPrivateLayoutProfileId) {
    return kStatusUnsupportedPrivateLayout;
  }
  if (collector.consumer != consumer ||
      collector.operation != operation ||
      collector.completion_reason != completion_reason) {
    return kStatusInvalidConsumerOperation;
  }
  if (!responses_complete(collector)) {
    return kStatusIncompleteResponses;
  }
  return kStatusOk;
}

static status_kind poison_collector(status_kind status,
                                    response_collector_v1 *collector) {
  collector->failed = 1;
  return status;
}

static status_kind validate_materialize_context(
    const materialize_context_v1 &context) {
  if (context.bvh_format_profile_id != kGenRtBvhFormatProfileId) {
    return kStatusUnsupportedBvhProfile;
  }
  return context.reserved_zero == 0 &&
                 context.recovery_target_inflight <= 1 &&
                 bytes_are_zero(context.reserved_zero1,
                                sizeof(context.reserved_zero1))
             ? kStatusOk
             : kStatusInvalidEncoding;
}

static status_kind decode_ray(const uint8_t *bytes, ray_v1 *ray) {
  if (bytes == NULL || ray == NULL) return kStatusInvalidArgument;
  ray_v1 decoded = {};
  for (uint8_t component = 0; component < 3; ++component) {
    decoded.origin[component] =
        fp32_value(load_u32(bytes + component * 4));
    decoded.direction[component] =
        fp32_value(load_u32(bytes + 12 + component * 4));
    if (!std::isfinite(decoded.origin[component]) ||
        !std::isfinite(decoded.direction[component])) {
      return kStatusInvalidEncoding;
    }
  }
  decoded.t_min = fp32_value(load_u32(bytes + 24));
  decoded.t_max = fp32_value(load_u32(bytes + 28));
  if (!std::isfinite(decoded.t_min) ||
      !std::isfinite(decoded.t_max) ||
      decoded.t_min > decoded.t_max) {
    return kStatusInvalidEncoding;
  }
  *ray = decoded;
  return kStatusOk;
}

template <typename TypedRay>
static void make_typed_ray(const ray_v1 &ray, TypedRay *typed) {
  for (uint8_t component = 0; component < 3; ++component) {
    typed->origin[component] = ray.origin[component];
    typed->direction[component] = ray.direction[component];
    typed->inverse_direction[component] =
        canonical_ray::inverse_direction(ray.direction[component]);
  }
  typed->t_min = ray.t_min;
  typed->t_max = ray.t_max;
}

static status_kind decode_as_context(
    const uint8_t *bytes, uint32_t bvh_format_profile_id,
    as_context_v1 *context,
    typed_blas::as_decode_context_v0 *typed) {
  if (bytes == NULL || context == NULL || typed == NULL) {
    return kStatusInvalidArgument;
  }
  as_context_v1 decoded = {};
  decoded.as_object_id = load_u64(bytes + 0);
  decoded.device_base = load_u64(bytes + 8);
  decoded.device_range_bytes = load_u64(bytes + 16);
  decoded.as_object_generation = load_u32(bytes + 24);
  decoded.as_type = bytes[28];
  decoded.cull_mask = bytes[29];
  decoded.reserved_zero[0] = bytes[30];
  decoded.reserved_zero[1] = bytes[31];
  if (!bytes_are_zero(decoded.reserved_zero,
                      sizeof(decoded.reserved_zero)) ||
      decoded.as_object_id == 0 || decoded.device_base == 0 ||
      decoded.device_range_bytes == 0 ||
      decoded.as_object_generation == 0 ||
      (decoded.as_type != kAsTypeTlas &&
       decoded.as_type != kAsTypeBlas) ||
      decoded.device_range_bytes >
          std::numeric_limits<uint64_t>::max() -
              decoded.device_base) {
    return kStatusInvalidEncoding;
  }

  typed_blas::as_decode_context_v0 typed_result = {};
  typed_result.bvh_format_profile_id = bvh_format_profile_id;
  typed_result.as_object.object_id = decoded.as_object_id;
  typed_result.as_object.generation =
      decoded.as_object_generation;
  typed_result.as_object.as_type = decoded.as_type;
  typed_result.device_base = decoded.device_base;
  typed_result.device_range_bytes = decoded.device_range_bytes;
  *context = decoded;
  *typed = typed_result;
  return kStatusOk;
}

static status_kind decode_hit_control(
    uint32_t packed,
    typed_stack::committed_hit_projection_v0 *hit) {
  static const uint32_t kDefinedMask =
      0x00000001u | 0x00000006u | 0x0000ff00u |
      0x00070000u | 0x03000000u | 0x1c000000u;
  if ((packed & ~kDefinedMask) != 0) {
    return kStatusInvalidEncoding;
  }
  hit->valid = static_cast<uint8_t>(packed & 0x1u);
  hit->geometry_type = static_cast<uint8_t>((packed >> 1) & 0x3u);
  hit->hit_kind = static_cast<uint8_t>((packed >> 8) & 0xffu);
  hit->attribute_word_count =
      static_cast<uint8_t>((packed >> 16) & 0x7u);
  hit->attribute_location =
      static_cast<uint8_t>((packed >> 24) & 0x3u);
  hit->attribute_format =
      static_cast<uint8_t>((packed >> 26) & 0x7u);
  return kStatusOk;
}

static bool valid_hit_shape(
    const typed_stack::committed_hit_projection_v0 &hit) {
  if (hit.geometry_type > typed_primitive::kGeometryTypeProcedural ||
      hit.attribute_word_count > 4 ||
      hit.attribute_location > 2 ||
      hit.attribute_format > 2 ||
      (hit.policy_flags & ~kCommittedPolicyMask) != 0) {
    return false;
  }
  if (hit.geometry_type == typed_primitive::kGeometryTypeTriangle) {
    return (hit.hit_kind == typed_primitive::kHitKindFrontFacing ||
            hit.hit_kind == typed_primitive::kHitKindBackFacing) &&
           hit.attribute_word_count == 2 &&
           hit.attribute_location != 0 &&
           hit.attribute_format == 1;
  }
  if (hit.geometry_type ==
      typed_primitive::kGeometryTypeProcedural) {
    return hit.hit_kind <= 0x7fu &&
           hit.attribute_format ==
               (hit.attribute_word_count == 0 ? 0 : 2) &&
           (hit.attribute_word_count == 0
                ? hit.attribute_location == 0
                : hit.attribute_location != 0);
  }
  return false;
}

static status_kind decode_committed_hot(
    const uint8_t *chunk2, committed_hot_v1 *hot) {
  if (chunk2 == NULL || hot == NULL) return kStatusInvalidArgument;
  typed_stack::committed_hit_projection_v0 hit = {};
  status_kind status =
      decode_hit_control(load_u32(chunk2), &hit);
  if (status != kStatusOk) return status;
  if (hit.valid == 0) {
    if (!bytes_are_zero(chunk2, kChunkBytes)) {
      return kStatusInvalidEncoding;
    }
    *hot = committed_hot_v1();
    return kStatusOk;
  }
  hit.hit_t = fp32_value(load_u32(chunk2 + 4));
  hit.policy_flags = load_u32(chunk2 + 8);
  hit.instance_metadata_ref = load_u64(chunk2 + 12);
  if (!std::isfinite(hit.hit_t) ||
      hit.instance_metadata_ref == 0 ||
      !valid_hit_shape(hit)) {
    return kStatusInvalidEncoding;
  }
  hot->valid = 1;
  hot->hit_t = hit.hit_t;
  return kStatusOk;
}

static status_kind decode_committed_hit(
    const uint8_t *chunk2, const uint8_t *chunk3,
    typed_stack::committed_hit_projection_v0 *hit) {
  if (chunk2 == NULL || chunk3 == NULL || hit == NULL) {
    return kStatusInvalidArgument;
  }
  typed_stack::committed_hit_projection_v0 decoded = {};
  status_kind status =
      decode_hit_control(load_u32(chunk2), &decoded);
  if (status != kStatusOk) return status;
  if (decoded.valid == 0) {
    if (!bytes_are_zero(chunk2, kChunkBytes) ||
        !bytes_are_zero(chunk3, 24)) {
      return kStatusInvalidEncoding;
    }
    *hit = decoded;
    return kStatusOk;
  }
  decoded.hit_t = fp32_value(load_u32(chunk2 + 4));
  decoded.policy_flags = load_u32(chunk2 + 8);
  decoded.instance_metadata_ref = load_u64(chunk2 + 12);
  decoded.primitive_index = load_u32(chunk2 + 20);
  decoded.geometry_index = load_u32(chunk2 + 24);
  decoded.instance_index = load_u32(chunk2 + 28);
  decoded.instance_custom_index = load_u32(chunk3 + 0);
  decoded.instance_sbt_contribution = load_u32(chunk3 + 4);
  for (uint8_t word = 0; word < 4; ++word) {
    decoded.inline_attributes[word] =
        load_u32(chunk3 + 8 + word * 4);
  }
  if (!std::isfinite(decoded.hit_t) ||
      decoded.instance_metadata_ref == 0 ||
      (decoded.instance_sbt_contribution & 0xff000000u) != 0 ||
      !valid_hit_shape(decoded)) {
    return kStatusInvalidEncoding;
  }
  *hit = decoded;
  return kStatusOk;
}

static status_kind decode_current_instance(
    const uint8_t *chunk3, const uint8_t *chunk4, uint8_t active_as_type,
    typed_stack::instance_shader_projection_v0 *instance) {
  if (chunk3 == NULL || chunk4 == NULL || instance == NULL) {
    return kStatusInvalidArgument;
  }
  typed_stack::instance_shader_projection_v0 decoded = {};
  decoded.instance_metadata_ref = load_u64(chunk3 + 24);
  decoded.instance_index = load_u32(chunk4 + 0);
  decoded.instance_custom_index = load_u32(chunk4 + 4);
  const uint32_t packed = load_u32(chunk4 + 8);
  decoded.instance_sbt_contribution = packed & 0x00ffffffu;
  decoded.instance_policy_flags = static_cast<uint8_t>(packed >> 24);
  if (!valid_instance_policy(decoded.instance_policy_flags) ||
      (active_as_type == kAsTypeBlas &&
       decoded.instance_metadata_ref == 0) ||
      (active_as_type == kAsTypeTlas &&
       decoded.instance_metadata_ref == 0 &&
       (decoded.instance_index != 0 ||
        decoded.instance_custom_index != 0 ||
        decoded.instance_sbt_contribution != 0 ||
        decoded.instance_policy_flags != 0))) {
    return kStatusInvalidEncoding;
  }
  *instance = decoded;
  return kStatusOk;
}

static status_kind decode_policy(
    const uint8_t *chunk1, const uint8_t *chunk4,
    typed_node::ray_policy_v0 *policy) {
  if (chunk1 == NULL || chunk4 == NULL || policy == NULL) {
    return kStatusInvalidArgument;
  }
  typed_node::ray_policy_v0 decoded = {};
  decoded.ray_flags = load_u32(chunk4 + 12);
  decoded.cull_mask = chunk1[29];
  if ((decoded.ray_flags &
       ~typed_primitive::kSupportedRayFlagMask) != 0 ||
      !bytes_are_zero(chunk4 + 28, 4)) {
    return kStatusInvalidEncoding;
  }
  *policy = decoded;
  return kStatusOk;
}

static float effective_bound(const ray_v1 &ray,
                             const committed_hot_v1 &hot) {
  return hot.valid != 0 ? std::min(ray.t_max, hot.hit_t)
                        : ray.t_max;
}

static bool relative_range_contains(const as_context_v1 &context,
                                    uint64_t offset,
                                    uint64_t byte_count) {
  return byte_count <= context.device_range_bytes &&
         offset <= context.device_range_bytes - byte_count;
}

static bool absolute_range_contains(
    const typed_blas::as_decode_context_v0 &context,
    uint64_t address, uint64_t byte_count) {
  if (byte_count > context.device_range_bytes ||
      address < context.device_base) {
    return false;
  }
  const uint64_t offset = address - context.device_base;
  return offset <= context.device_range_bytes - byte_count;
}

static void decode_entry(const uint8_t *bytes,
                         short_stack::entry_v0 *entry) {
  entry->payload_offset = load_u64(bytes + 0);
  entry->near_t_bits = load_u32(bytes + 8);
  entry->payload_byte_count = load_u16(bytes + 12);
  entry->payload_kind = bytes[14];
  entry->control = bytes[15];
}

static status_kind decode_stack(
    const uint8_t *chunk4, const uint8_t *chunk5,
    const uint8_t *chunk6, const uint8_t *chunk7,
    uint8_t recovery_target_inflight,
    short_stack::state_v0 *stack) {
  if (chunk4 == NULL || chunk5 == NULL || chunk6 == NULL ||
      chunk7 == NULL || stack == NULL ||
      recovery_target_inflight > 1) {
    return kStatusInvalidArgument;
  }
  short_stack::state_v0 decoded = {};
  decoded.stack_count = chunk4[24];
  decoded.stack_top_ptr = chunk4[25];
  decoded.lost = chunk4[26];
  decoded.active_domain = chunk4[27];
  decoded.cross_as =
      decoded.active_domain == short_stack::kDomainBlas ? 1 : 0;
  if (decoded.stack_count > short_stack::kLogicalCapacity ||
      (decoded.stack_count == 0 && decoded.stack_top_ptr != 0) ||
      decoded.stack_top_ptr >= short_stack::kLogicalCapacity ||
      decoded.lost > 1 ||
      decoded.active_domain > short_stack::kDomainBlas ||
      !bytes_are_zero(chunk4 + 28, 4)) {
    return kStatusInvalidEncoding;
  }
  uint8_t encoded_entries[3 * kChunkBytes] = {};
  std::memcpy(encoded_entries + 0, chunk5, kChunkBytes);
  std::memcpy(encoded_entries + 32, chunk6, kChunkBytes);
  std::memcpy(encoded_entries + 64, chunk7, kChunkBytes);
  for (uint8_t logical = 0; logical < decoded.stack_count; ++logical) {
    const uint8_t physical = static_cast<uint8_t>(
        (decoded.stack_top_ptr + logical) %
        short_stack::kLogicalCapacity);
    decode_entry(encoded_entries + physical * 16,
                 &decoded.entries[physical]);
  }
  const bool stable =
      recovery_target_inflight == 0 &&
      short_stack::validate_state(decoded);
  const bool recovery =
      recovery_target_inflight == 1 &&
      short_stack::validate_drained_recovery_state(decoded);
  if (!stable && !recovery) {
    return kStatusInvalidEncoding;
  }
  *stack = decoded;
  return kStatusOk;
}

static status_kind decode_boundary(
    const response_collector_v1 &collector,
    software_boundary_v1 *boundary) {
  if (boundary == NULL) return kStatusInvalidArgument;
  const uint8_t *chunk8 = find_chunk(collector, 8);
  const uint8_t *chunk9 = find_chunk(collector, 9);
  if (chunk8 == NULL || chunk9 == NULL) {
    return kStatusIncompleteResponses;
  }
  software_boundary_v1 decoded = {};
  decoded.reason = collector.completion_reason;
  typed_primitive::primitive_identity_policy_facts_v0 &identity =
      decoded.identity_and_policy;
  identity.instance_metadata_ref = load_u64(chunk8 + 0);
  identity.primitive_index = load_u32(chunk8 + 8);
  identity.geometry_index = load_u32(chunk8 + 12);
  identity.instance_index = load_u32(chunk8 + 16);
  identity.instance_custom_index = load_u32(chunk8 + 20);
  identity.instance_sbt_contribution = load_u32(chunk8 + 24);
  identity.geometry_type = chunk8[28];
  identity.geometry_policy_flags = chunk8[29];
  identity.instance_policy_flags = chunk8[30];
  identity.effective_policy_flags = chunk8[31];
  if (identity.instance_metadata_ref == 0 ||
      identity.geometry_type == typed_primitive::kGeometryTypeInvalid ||
      identity.geometry_type > typed_primitive::kGeometryTypeProcedural ||
      (identity.geometry_policy_flags & ~kGeometryPolicyMask) != 0 ||
      !valid_instance_policy(identity.instance_policy_flags) ||
      (identity.effective_policy_flags & ~kEffectivePolicyMask) != 0 ||
      (identity.instance_sbt_contribution & 0xff000000u) != 0) {
    return kStatusInvalidEncoding;
  }

  if (collector.completion_reason ==
      operand_plan::kCompletionReasonAnyHitRequired) {
    if (identity.geometry_type !=
        typed_primitive::kGeometryTypeTriangle) {
      return kStatusInvalidEncoding;
    }
    typed_primitive::triangle_hit_facts_v0 &hit =
        decoded.reason_facts.triangle_hit;
    hit.hit_t_bits = load_u32(chunk9 + 0);
    hit.bary_vertex1_bits = load_u32(chunk9 + 4);
    hit.bary_vertex2_bits = load_u32(chunk9 + 8);
    hit.hit_kind = chunk9[12];
    std::memcpy(hit.reserved_zero, chunk9 + 13, 3);
    if (!std::isfinite(fp32_value(hit.hit_t_bits)) ||
        (hit.hit_kind != typed_primitive::kHitKindFrontFacing &&
         hit.hit_kind != typed_primitive::kHitKindBackFacing) ||
        !bytes_are_zero(hit.reserved_zero,
                        sizeof(hit.reserved_zero))) {
      return kStatusInvalidEncoding;
    }
  } else if (collector.completion_reason ==
             operand_plan::kCompletionReasonIntersectionRequired) {
    if (identity.geometry_type !=
        typed_primitive::kGeometryTypeProcedural) {
      return kStatusInvalidEncoding;
    }
    typed_primitive::intersection_boundary_facts_v0 &intersection =
        decoded.reason_facts.intersection;
    intersection.boundary_ray_tmax_bits = load_u32(chunk9 + 0);
    intersection.reserved_zero = load_u32(chunk9 + 4);
    if (!std::isfinite(
            fp32_value(intersection.boundary_ray_tmax_bits)) ||
        intersection.reserved_zero != 0 ||
        !bytes_are_zero(chunk9 + 8, 8)) {
      return kStatusInvalidEncoding;
    }
  } else {
    return kStatusInvalidConsumerOperation;
  }

  decoded.primitive_resume.leaf_fetch_address =
      load_u64(chunk9 + 16);
  decoded.primitive_resume.remaining_slot_mask =
      load_u64(chunk9 + 24);
  if (decoded.primitive_resume.remaining_slot_mask == 0) {
    if (decoded.primitive_resume.leaf_fetch_address != 0) {
      return kStatusInvalidEncoding;
    }
  } else if (decoded.primitive_resume.leaf_fetch_address == 0 ||
             (decoded.primitive_resume.leaf_fetch_address &
              uint64_t{0x3f}) != 0) {
    return kStatusInvalidEncoding;
  }
  *boundary = decoded;
  return kStatusOk;
}

static status_kind materialize_common(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context, ray_v1 *ray,
    as_context_v1 *as_context,
    typed_blas::as_decode_context_v0 *typed_context,
    typed_node::ray_policy_v0 *policy,
    committed_hot_v1 *committed_hot) {
  status_kind status = validate_materialize_context(context);
  if (status != kStatusOk) return status;
  const uint8_t *chunk0 = find_chunk(collector, 0);
  const uint8_t *chunk1 = find_chunk(collector, 1);
  const uint8_t *chunk2 = find_chunk(collector, 2);
  const uint8_t *chunk4 = find_chunk(collector, 4);
  if (chunk0 == NULL || chunk1 == NULL || chunk2 == NULL ||
      chunk4 == NULL) {
    return kStatusIncompleteResponses;
  }
  status = decode_ray(chunk0, ray);
  if (status != kStatusOk) return status;
  status = decode_as_context(
      chunk1, context.bvh_format_profile_id,
      as_context, typed_context);
  if (status != kStatusOk) return status;
  status = decode_policy(chunk1, chunk4, policy);
  if (status != kStatusOk) return status;
  const uint32_t tlas_build_generation = load_u32(chunk4 + 16);
  const uint32_t blas_build_generation = load_u32(chunk4 + 20);
  if (tlas_build_generation == 0 ||
      (as_context->as_type == kAsTypeBlas &&
       blas_build_generation == 0)) {
    return kStatusInvalidEncoding;
  }
  status = decode_committed_hot(chunk2, committed_hot);
  if (status != kStatusOk) return status;
  return committed_hot->valid != 0 &&
                 committed_hot->hit_t < ray->t_min
             ? kStatusInvalidEncoding
             : kStatusOk;
}

static status_kind materialize_primitive_base(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    primitive_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  primitive_operands_v1 decoded = {};
  ray_v1 ray = {};
  as_context_v1 as_context = {};
  committed_hot_v1 hot = {};
  status_kind status = materialize_common(
      collector, context, &ray, &as_context,
      &decoded.decode_context, &decoded.ray_policy, &hot);
  if (status != kStatusOk) return status;
  if (as_context.as_type != kAsTypeBlas) {
    return kStatusInvalidEncoding;
  }
  make_typed_ray(ray, &decoded.ray);
  const uint8_t *chunk2 = find_chunk(collector, 2);
  const uint8_t *chunk3 = find_chunk(collector, 3);
  const uint8_t *chunk4 = find_chunk(collector, 4);
  status = decode_committed_hit(
      chunk2, chunk3, &decoded.committed_hit);
  if (status != kStatusOk) return status;
  status = decode_current_instance(
      chunk3, chunk4, as_context.as_type,
      &decoded.current_instance);
  if (status != kStatusOk) return status;
  decoded.effective_traversal_bound = effective_bound(ray, hot);
  *operands = decoded;
  return kStatusOk;
}

static status_kind materialize_stack_common(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    stack_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  stack_operands_v1 decoded = {};
  ray_v1 ray = {};
  as_context_v1 as_context = {};
  committed_hot_v1 hot = {};
  status_kind status = materialize_common(
      collector, context, &ray, &as_context,
      &decoded.active_decode_context, &decoded.ray_policy, &hot);
  if (status != kStatusOk) return status;
  make_typed_ray(ray, &decoded.ray);
  status = decode_stack(
      find_chunk(collector, 4), find_chunk(collector, 5),
      find_chunk(collector, 6), find_chunk(collector, 7),
      context.recovery_target_inflight,
      &decoded.stack);
  if (status != kStatusOk) return status;
  const uint8_t *chunk4 = find_chunk(collector, 4);
  decoded.tlas_build_generation = load_u32(chunk4 + 16);
  decoded.blas_build_generation = load_u32(chunk4 + 20);
  if (decoded.tlas_build_generation == 0 ||
      (decoded.stack.active_domain == short_stack::kDomainBlas
           ? decoded.blas_build_generation == 0
           : decoded.blas_build_generation != 0)) {
    return kStatusInvalidEncoding;
  }
  if ((decoded.stack.active_domain == short_stack::kDomainTlas
           ? kAsTypeTlas
           : kAsTypeBlas) != as_context.as_type) {
    return kStatusInvalidEncoding;
  }
  for (uint8_t logical = 0; logical < decoded.stack.stack_count;
       ++logical) {
    short_stack::entry_v0 entry = {};
    if (!short_stack::read_logical_entry(
            decoded.stack, logical, &entry)) {
      return kStatusInvalidEncoding;
    }
    if (short_stack::control_kind(entry.control) ==
        short_stack::kEntryCrossAsReturn) {
      continue;
    }
    const short_stack::domain_kind expected_domain =
        as_context.as_type == kAsTypeTlas
            ? short_stack::kDomainTlas
            : short_stack::kDomainBlas;
    if (short_stack::control_domain(entry.control) !=
            expected_domain ||
        !relative_range_contains(
            as_context, entry.payload_offset,
            entry.payload_byte_count)) {
      return kStatusInvalidEncoding;
    }
  }
  decoded.effective_traversal_bound = effective_bound(ray, hot);
  decoded.committed_valid = hot.valid;
  *operands = decoded;
  return kStatusOk;
}

static bool is_final_reason(uint8_t reason) {
  return reason == operand_plan::kCompletionReasonMiss ||
         reason ==
             operand_plan::kCompletionReasonClosestHitReady ||
         reason ==
             operand_plan::kCompletionReasonTraceDoneNoShader;
}

}  // namespace

status_kind initialize_collector(
    const operand_plan::read_plan_v1 &plan,
    const operation_identity_v1 &identity,
    response_collector_v1 *collector) {
  if (collector == NULL) return kStatusInvalidArgument;
  std::memset(collector, 0, sizeof(*collector));
  operand_plan::read_request_v1 request = {};
  request.private_layout_profile_id =
      plan.private_layout_profile_id;
  request.consumer = plan.consumer;
  request.operation = plan.operation;
  request.completion_reason = plan.completion_reason;
  operand_plan::read_plan_v1 canonical = {};
  const operand_plan::status_kind plan_status =
      operand_plan::make_read_plan(request, &canonical);
  if (plan_status ==
      operand_plan::kStatusUnsupportedPrivateLayout) {
    return kStatusUnsupportedPrivateLayout;
  }
  if (plan_status != operand_plan::kStatusOk ||
      std::memcmp(&canonical, &plan, sizeof(plan)) != 0 ||
      plan.read_count == 0 ||
      plan.read_count > kMaxOperationReadChunks) {
    return kStatusInvalidPlan;
  }
  if (!valid_operation_identity(identity)) {
    return kStatusInvalidOperationIdentity;
  }

  response_collector_v1 result = {};
  result.identity = identity;
  result.private_layout_profile_id =
      plan.private_layout_profile_id;
  result.consumer = plan.consumer;
  result.operation = plan.operation;
  result.completion_reason = plan.completion_reason;
  result.required_count = plan.read_count;
  result.initialized = 1;
  for (uint8_t index = 0; index < plan.read_count; ++index) {
    result.required_chunk_mask |= static_cast<uint16_t>(
        uint16_t{1} << plan.reads[index].chunk_index);
  }
  *collector = result;
  return kStatusOk;
}

status_kind accept_response(const chunk_response_v1 &response,
                            response_collector_v1 *collector) {
  if (collector == NULL) return kStatusInvalidArgument;
  if (collector->initialized == 0) return kStatusNotInitialized;
  if (collector->failed != 0) return kStatusCollectorFailed;
  if (!collector_shape_valid(*collector)) {
    return poison_collector(kStatusInvalidCollector, collector);
  }
  if (!same_operation_identity(response.identity,
                               collector->identity) ||
      response.private_layout_profile_id !=
          collector->private_layout_profile_id ||
      response.consumer != collector->consumer ||
      response.operation != collector->operation ||
      response.completion_reason !=
          collector->completion_reason) {
    return poison_collector(
        kStatusResponseIdentityMismatch, collector);
  }
  if (!bytes_are_zero(response.reserved_zero,
                      sizeof(response.reserved_zero)) ||
      response.chunk_index >= kChunkCount ||
      response.slot_byte_offset !=
          response.chunk_index * kChunkBytes ||
      response.byte_count != kChunkBytes) {
    return poison_collector(
        kStatusResponseShapeMismatch, collector);
  }
  const uint16_t chunk_bit = static_cast<uint16_t>(
      uint16_t{1} << response.chunk_index);
  if ((collector->required_chunk_mask & chunk_bit) == 0) {
    return poison_collector(kStatusUnrequestedChunk, collector);
  }
  if ((collector->received_chunk_mask & chunk_bit) != 0) {
    return poison_collector(kStatusDuplicateChunk, collector);
  }
  if (collector->received_count >= collector->required_count ||
      collector->received_count >= kMaxOperationReadChunks) {
    return poison_collector(
        kStatusResponseShapeMismatch, collector);
  }
  collected_chunk_v1 &destination =
      collector->chunks[collector->received_count++];
  destination.chunk_index = response.chunk_index;
  destination.valid = 1;
  std::memcpy(destination.payload, response.payload,
              kChunkBytes);
  collector->received_chunk_mask |= chunk_bit;
  return kStatusOk;
}

bool responses_complete(const response_collector_v1 &collector) {
  return collector.initialized != 0 &&
         collector.failed == 0 &&
         collector_shape_valid(collector) &&
         collector.required_count != 0 &&
         collector.received_count == collector.required_count &&
         collector.received_chunk_mask ==
             collector.required_chunk_mask;
}

status_kind promote_stack_collector(
    const response_collector_v1 &base_collector,
    uint8_t selected_operation,
    response_collector_v1 *selected_collector) {
  if (selected_collector == NULL) return kStatusInvalidArgument;
  *selected_collector = response_collector_v1();
  if (selected_operation !=
          operand_plan::kOperationStackTerminal &&
      selected_operation !=
          operand_plan::kOperationStackCrossAsReturn) {
    return kStatusInvalidConsumerOperation;
  }
  status_kind status = require_collector(
      base_collector, operand_plan::kConsumerStack,
      operand_plan::kOperationDefault,
      operand_plan::kCompletionReasonNone);
  if (status != kStatusOk) return status;

  operand_plan::read_request_v1 request = {};
  request.private_layout_profile_id =
      base_collector.private_layout_profile_id;
  request.consumer = operand_plan::kConsumerStack;
  request.operation = selected_operation;
  request.completion_reason =
      operand_plan::kCompletionReasonNone;
  operand_plan::read_plan_v1 plan = {};
  if (operand_plan::make_read_plan(request, &plan) !=
      operand_plan::kStatusOk) {
    return kStatusInvalidPlan;
  }
  response_collector_v1 promoted = {};
  status = initialize_collector(
      plan, base_collector.identity, &promoted);
  if (status != kStatusOk) return status;

  for (uint8_t index = 0;
       index < base_collector.received_count; ++index) {
    const collected_chunk_v1 &source =
        base_collector.chunks[index];
    chunk_response_v1 response = {};
    response.identity = base_collector.identity;
    response.private_layout_profile_id =
        base_collector.private_layout_profile_id;
    response.consumer = operand_plan::kConsumerStack;
    response.operation = selected_operation;
    response.completion_reason =
        operand_plan::kCompletionReasonNone;
    response.chunk_index = source.chunk_index;
    response.slot_byte_offset = static_cast<uint16_t>(
        source.chunk_index * kChunkBytes);
    response.byte_count = kChunkBytes;
    std::memcpy(response.payload, source.payload, kChunkBytes);
    status = accept_response(response, &promoted);
    if (status != kStatusOk) return status;
  }
  *selected_collector = promoted;
  return kStatusOk;
}

status_kind materialize_node(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    node_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = node_operands_v1();
  status_kind status = require_collector(
      collector, operand_plan::kConsumerNode,
      operand_plan::kOperationDefault,
      operand_plan::kCompletionReasonNone);
  if (status != kStatusOk) return status;
  node_operands_v1 decoded = {};
  ray_v1 ray = {};
  as_context_v1 as_context = {};
  committed_hot_v1 hot = {};
  status = materialize_common(
      collector, context, &ray, &as_context,
      &decoded.decode_context, &decoded.ray_policy, &hot);
  if (status != kStatusOk) return status;
  make_typed_ray(ray, &decoded.ray);
  decoded.effective_traversal_bound = effective_bound(ray, hot);
  decoded.committed_valid = hot.valid;
  *operands = decoded;
  return kStatusOk;
}

status_kind materialize_primitive(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    primitive_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = primitive_operands_v1();
  status_kind status = require_collector(
      collector, operand_plan::kConsumerPrimitive,
      operand_plan::kOperationDefault,
      operand_plan::kCompletionReasonNone);
  return status == kStatusOk
             ? materialize_primitive_base(collector, context, operands)
             : status;
}

status_kind materialize_primitive_resume(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    primitive_resume_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = primitive_resume_operands_v1();
  if (collector.completion_reason !=
          operand_plan::kCompletionReasonAnyHitRequired &&
      collector.completion_reason !=
          operand_plan::kCompletionReasonIntersectionRequired) {
    return kStatusInvalidConsumerOperation;
  }
  status_kind status = require_collector(
      collector, operand_plan::kConsumerPrimitive,
      operand_plan::kOperationPrimitiveResume,
      collector.completion_reason);
  if (status != kStatusOk) return status;
  primitive_resume_operands_v1 decoded = {};
  status = materialize_primitive_base(
      collector, context, &decoded.persistent);
  if (status != kStatusOk) return status;
  status = decode_boundary(collector, &decoded.boundary);
  if (status != kStatusOk) return status;
  if (decoded.boundary.primitive_resume.remaining_slot_mask != 0 &&
      !absolute_range_contains(
          decoded.persistent.decode_context,
          decoded.boundary.primitive_resume.leaf_fetch_address, 64)) {
    return kStatusInvalidEncoding;
  }
  *operands = decoded;
  return kStatusOk;
}

status_kind materialize_instance(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    instance_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = instance_operands_v1();
  status_kind status = require_collector(
      collector, operand_plan::kConsumerInstance,
      operand_plan::kOperationDefault,
      operand_plan::kCompletionReasonNone);
  if (status != kStatusOk) return status;
  status = validate_materialize_context(context);
  if (status != kStatusOk) return status;
  const uint8_t *chunk0 = find_chunk(collector, 0);
  const uint8_t *chunk1 = find_chunk(collector, 1);
  const uint8_t *chunk4 = find_chunk(collector, 4);
  if (chunk0 == NULL || chunk1 == NULL || chunk4 == NULL) {
    return kStatusIncompleteResponses;
  }
  instance_operands_v1 decoded = {};
  ray_v1 ray = {};
  as_context_v1 as_context = {};
  status = decode_ray(chunk0, &ray);
  if (status != kStatusOk) return status;
  make_typed_ray(ray, &decoded.world_ray);
  status = decode_as_context(
      chunk1, context.bvh_format_profile_id,
      &as_context, &decoded.tlas_decode_context);
  if (status != kStatusOk || as_context.as_type != kAsTypeTlas) {
    return kStatusInvalidEncoding;
  }
  typed_node::ray_policy_v0 node_policy = {};
  status = decode_policy(chunk1, chunk4, &node_policy);
  if (status != kStatusOk) return status;
  decoded.ray_policy.ray_flags = node_policy.ray_flags;
  decoded.ray_policy.cull_mask = node_policy.cull_mask;
  decoded.tlas_build_generation = load_u32(chunk4 + 16);
  if (decoded.tlas_build_generation == 0) {
    return kStatusInvalidEncoding;
  }
  *operands = decoded;
  return kStatusOk;
}

status_kind materialize_stack_base(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    stack_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = stack_operands_v1();
  status_kind status = require_collector(
      collector, operand_plan::kConsumerStack,
      operand_plan::kOperationDefault,
      operand_plan::kCompletionReasonNone);
  return status == kStatusOk
             ? materialize_stack_common(collector, context, operands)
             : status;
}

status_kind materialize_stack_terminal(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    stack_terminal_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = stack_terminal_operands_v1();
  status_kind status = require_collector(
      collector, operand_plan::kConsumerStack,
      operand_plan::kOperationStackTerminal,
      operand_plan::kCompletionReasonNone);
  if (status != kStatusOk) return status;
  stack_terminal_operands_v1 decoded = {};
  status = materialize_stack_common(
      collector, context, &decoded.base);
  if (status != kStatusOk) return status;
  status = decode_committed_hit(
      find_chunk(collector, 2), find_chunk(collector, 3),
      &decoded.committed_hit);
  if (status != kStatusOk) return status;
  if (decoded.base.stack.stack_count != 0 ||
      decoded.base.stack.active_domain != short_stack::kDomainTlas) {
    return kStatusInvalidEncoding;
  }
  *operands = decoded;
  return kStatusOk;
}

status_kind materialize_stack_cross_as(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    stack_cross_as_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = stack_cross_as_operands_v1();
  status_kind status = require_collector(
      collector, operand_plan::kConsumerStack,
      operand_plan::kOperationStackCrossAsReturn,
      operand_plan::kCompletionReasonNone);
  if (status != kStatusOk) return status;
  stack_cross_as_operands_v1 decoded = {};
  status = materialize_stack_common(
      collector, context, &decoded.base);
  if (status != kStatusOk) return status;
  const uint8_t *chunk10 = find_chunk(collector, 10);
  const uint8_t *chunk11 = find_chunk(collector, 11);
  if (chunk10 == NULL || chunk11 == NULL) {
    return kStatusIncompleteResponses;
  }
  ray_v1 parent_ray = {};
  status = decode_ray(chunk10, &parent_ray);
  if (status != kStatusOk) return status;
  make_typed_ray(parent_ray, &decoded.parent.ray);
  as_context_v1 parent_context = {};
  status = decode_as_context(
      chunk11, context.bvh_format_profile_id,
      &parent_context, &decoded.parent.tlas_decode_context);
  if (status != kStatusOk ||
      parent_context.as_type != kAsTypeTlas) {
    return kStatusInvalidEncoding;
  }
  for (uint8_t logical = 0;
       logical < decoded.base.stack.stack_count; ++logical) {
    short_stack::entry_v0 entry = {};
    if (!short_stack::read_logical_entry(
            decoded.base.stack, logical, &entry)) {
      return kStatusInvalidEncoding;
    }
    if (short_stack::control_kind(entry.control) !=
        short_stack::kEntryCrossAsReturn) {
      continue;
    }
    if (!relative_range_contains(
            parent_context, entry.payload_offset,
            entry.payload_byte_count)) {
      return kStatusInvalidEncoding;
    }
  }
  decoded.parent.ray_policy.ray_flags =
      decoded.base.ray_policy.ray_flags;
  decoded.parent.ray_policy.cull_mask =
      parent_context.cull_mask;
  *operands = decoded;
  return kStatusOk;
}

status_kind materialize_final_completion(
    const response_collector_v1 &collector,
    final_completion_operands_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = final_completion_operands_v1();
  if (!is_final_reason(collector.completion_reason)) {
    return kStatusInvalidConsumerOperation;
  }
  status_kind status = require_collector(
      collector, operand_plan::kConsumerCompletionPublisher,
      operand_plan::kOperationDefault,
      collector.completion_reason);
  if (status != kStatusOk) return status;
  final_completion_operands_v1 decoded = {};
  decoded.completion_reason = collector.completion_reason;
  status = decode_committed_hit(
      find_chunk(collector, 2), find_chunk(collector, 3),
      &decoded.committed_hit);
  if (status != kStatusOk) return status;
  if (collector.completion_reason ==
          operand_plan::kCompletionReasonClosestHitReady &&
      decoded.committed_hit.valid == 0) {
    return kStatusInvalidEncoding;
  }
  if (collector.completion_reason ==
          operand_plan::kCompletionReasonMiss &&
      decoded.committed_hit.valid != 0) {
    return kStatusInvalidEncoding;
  }
  *operands = decoded;
  return kStatusOk;
}

status_kind materialize_boundary_completion(
    const response_collector_v1 &collector,
    software_boundary_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = software_boundary_v1();
  if (collector.completion_reason !=
          operand_plan::kCompletionReasonAnyHitRequired &&
      collector.completion_reason !=
          operand_plan::kCompletionReasonIntersectionRequired) {
    return kStatusInvalidConsumerOperation;
  }
  status_kind status = require_collector(
      collector, operand_plan::kConsumerCompletionPublisher,
      operand_plan::kOperationDefault,
      collector.completion_reason);
  return status == kStatusOk
             ? decode_boundary(collector, operands)
             : status;
}

status_kind materialize_resubmit(
    const response_collector_v1 &collector,
    software_boundary_v1 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  *operands = software_boundary_v1();
  if (collector.completion_reason !=
          operand_plan::kCompletionReasonAnyHitRequired &&
      collector.completion_reason !=
          operand_plan::kCompletionReasonIntersectionRequired) {
    return kStatusInvalidConsumerOperation;
  }
  status_kind status = require_collector(
      collector, operand_plan::kConsumerResubmitApply,
      operand_plan::kOperationDefault,
      collector.completion_reason);
  return status == kStatusOk
             ? decode_boundary(collector, operands)
             : status;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidPlan:
      return "invalid_plan";
    case kStatusInvalidOperationIdentity:
      return "invalid_operation_identity";
    case kStatusInvalidCollector:
      return "invalid_collector";
    case kStatusUnsupportedPrivateLayout:
      return "unsupported_private_layout";
    case kStatusUnsupportedBvhProfile:
      return "unsupported_bvh_profile";
    case kStatusNotInitialized:
      return "not_initialized";
    case kStatusResponseIdentityMismatch:
      return "response_identity_mismatch";
    case kStatusResponseShapeMismatch:
      return "response_shape_mismatch";
    case kStatusUnrequestedChunk:
      return "unrequested_chunk";
    case kStatusDuplicateChunk:
      return "duplicate_chunk";
    case kStatusCollectorFailed:
      return "collector_failed";
    case kStatusIncompleteResponses:
      return "incomplete_responses";
    case kStatusInvalidConsumerOperation:
      return "invalid_consumer_operation";
    case kStatusInvalidEncoding:
      return "invalid_encoding";
  }
  return "unknown";
}

}  // namespace operand_materializer
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore
