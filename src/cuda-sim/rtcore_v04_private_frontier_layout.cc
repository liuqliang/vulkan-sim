#include "rtcore_v04_private_frontier_layout.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace private_frontier {
namespace {

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static bool checked_add_u64(uint64_t lhs, uint64_t rhs,
                            uint64_t *result) {
  if (result == NULL || lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

static bool checked_mul_u64(uint64_t lhs, uint64_t rhs,
                            uint64_t *result) {
  if (result == NULL ||
      (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

static void encode_u16_le(uint8_t *destination, uint16_t value) {
  destination[0] = static_cast<uint8_t>(value);
  destination[1] = static_cast<uint8_t>(value >> 8);
}

static void encode_u32_le(uint8_t *destination, uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte) {
    destination[byte] = static_cast<uint8_t>(value >> (byte * 8));
  }
}

static void encode_u64_le(uint8_t *destination, uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte) {
    destination[byte] = static_cast<uint8_t>(value >> (byte * 8));
  }
}

static uint16_t decode_u16_le(const uint8_t *source) {
  return static_cast<uint16_t>(source[0]) |
         static_cast<uint16_t>(source[1]) << 8;
}

static uint32_t decode_u32_le(const uint8_t *source) {
  uint32_t value = 0;
  for (unsigned byte = 0; byte < 4; ++byte) {
    value |= static_cast<uint32_t>(source[byte]) << (byte * 8);
  }
  return value;
}

static uint64_t decode_u64_le(const uint8_t *source) {
  uint64_t value = 0;
  for (unsigned byte = 0; byte < 8; ++byte) {
    value |= static_cast<uint64_t>(source[byte]) << (byte * 8);
  }
  return value;
}

static uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float fp32_value(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static bool valid_owner(const owner_binding_v0 &owner) {
  return owner.request_identity != 0 && owner.generation != 0 &&
         owner.lane_id < 32 &&
         bytes_are_zero(owner.reserved_zero,
                        sizeof(owner.reserved_zero));
}

static status_kind slot_base_address(const owner_binding_v0 &owner,
                                     const region_binding_v0 &region,
                                     uint64_t *slot_base) {
  if (region.profile_id != kLayoutProfileId) {
    return kStatusUnsupportedProfile;
  }
  if (region.slot_count == 0 ||
      (region.private_region_base % kPrivateDataSlotAlignment) != 0 ||
      owner.private_slot_id >= region.slot_count) {
    return kStatusInvalidRegion;
  }

  uint64_t region_bytes = 0;
  uint64_t slot_delta = 0;
  uint64_t region_end = 0;
  uint64_t candidate = 0;
  uint64_t slot_end = 0;
  if (!checked_mul_u64(region.slot_count, kPrivateDataSlotBytes,
                       &region_bytes) ||
      !checked_add_u64(region.private_region_base, region_bytes,
                       &region_end) ||
      !checked_mul_u64(owner.private_slot_id, kPrivateDataSlotBytes,
                       &slot_delta) ||
      !checked_add_u64(region.private_region_base, slot_delta,
                       &candidate) ||
      !checked_add_u64(candidate, kPrivateDataSlotBytes, &slot_end) ||
      slot_end > region_end) {
    return kStatusAddressOverflow;
  }
  *slot_base = candidate;
  return kStatusOk;
}

static bool valid_metadata(const frontier_metadata_image_v0 &metadata) {
  const bool valid_level_depth =
      (metadata.current_level == 0 &&
       metadata.level_frame_depth == 0) ||
      (metadata.current_level == 1 &&
       metadata.level_frame_depth == 1);
  return metadata.frontier_top == metadata.frontier_count &&
         metadata.frontier_top <= metadata.frontier_capacity &&
         metadata.frontier_count <= metadata.frontier_capacity &&
         metadata.frontier_capacity == kFrontierEntryCapacity &&
         metadata.max_level_depth == 1 && valid_level_depth;
}

static void encode_metadata_bytes(
    uint8_t *destination,
    const frontier_metadata_image_v0 &metadata) {
  encode_u32_le(destination + 0, metadata.frontier_top);
  encode_u32_le(destination + 4, metadata.frontier_count);
  encode_u32_le(destination + 8, metadata.frontier_capacity);
  encode_u32_le(destination + 12, metadata.current_level);
  encode_u32_le(destination + 16, metadata.level_frame_depth);
  encode_u32_le(destination + 20, metadata.max_level_depth);
}

static frontier_metadata_image_v0 decode_metadata_bytes(
    const uint8_t *source) {
  frontier_metadata_image_v0 metadata = {};
  metadata.frontier_top = decode_u32_le(source + 0);
  metadata.frontier_count = decode_u32_le(source + 4);
  metadata.frontier_capacity = decode_u32_le(source + 8);
  metadata.current_level = decode_u32_le(source + 12);
  metadata.level_frame_depth = decode_u32_le(source + 16);
  metadata.max_level_depth = decode_u32_le(source + 20);
  return metadata;
}

static void encode_entry_bytes(
    uint8_t *destination,
    const typed_node::compact_child_work_item_v0 &entry) {
  encode_u64_le(destination + 0, entry.payload_offset);
  encode_u32_le(destination + 8, entry.near_t_bits);
  encode_u16_le(destination + 12, entry.payload_byte_count);
  destination[14] = entry.payload_kind;
  destination[15] = entry.child_slot;
}

static typed_node::compact_child_work_item_v0 decode_entry_bytes(
    const uint8_t *source) {
  typed_node::compact_child_work_item_v0 entry = {};
  entry.payload_offset = decode_u64_le(source + 0);
  entry.near_t_bits = decode_u32_le(source + 8);
  entry.payload_byte_count = decode_u16_le(source + 12);
  entry.payload_kind = source[14];
  entry.child_slot = source[15];
  return entry;
}

static void encode_selected_fetch_bytes(
    uint8_t *destination,
    const typed_node::selected_child_fetch_work_item_v0 &selected) {
  encode_entry_bytes(destination, selected.child);
  const typed_blas::as_decode_context_v0 &context =
      selected.decode_context;
  encode_u32_le(destination + 16, context.bvh_format_profile_id);
  encode_u32_le(destination + 20, context.reserved_zero);
  encode_u64_le(destination + 24, context.as_object.object_id);
  encode_u32_le(destination + 32, context.as_object.generation);
  destination[36] = context.as_object.as_type;
  std::memcpy(destination + 37, context.as_object.reserved_zero,
              sizeof(context.as_object.reserved_zero));
  encode_u64_le(destination + 40, context.device_base);
  encode_u64_le(destination + 48, context.device_range_bytes);
}

static typed_node::selected_child_fetch_work_item_v0
decode_selected_fetch_bytes(const uint8_t *source) {
  typed_node::selected_child_fetch_work_item_v0 selected = {};
  selected.child = decode_entry_bytes(source);
  typed_blas::as_decode_context_v0 &context =
      selected.decode_context;
  context.bvh_format_profile_id = decode_u32_le(source + 16);
  context.reserved_zero = decode_u32_le(source + 20);
  context.as_object.object_id = decode_u64_le(source + 24);
  context.as_object.generation = decode_u32_le(source + 32);
  context.as_object.as_type = source[36];
  std::memcpy(context.as_object.reserved_zero, source + 37,
              sizeof(context.as_object.reserved_zero));
  context.device_base = decode_u64_le(source + 40);
  context.device_range_bytes = decode_u64_le(source + 48);
  return selected;
}

static void encode_mutable_ray_bytes(
    uint8_t *destination, const mutable_ray_state_v0 &ray) {
  for (unsigned component = 0; component < 3; ++component) {
    encode_u32_le(destination + component * 4,
                  fp32_bits(ray.origin[component]));
    encode_u32_le(destination + 12 + component * 4,
                  fp32_bits(ray.direction[component]));
    encode_u32_le(destination + 24 + component * 4,
                  fp32_bits(ray.inverse_direction[component]));
  }
  encode_u32_le(destination + 36, fp32_bits(ray.t_min));
  encode_u32_le(destination + 40, fp32_bits(ray.t_max));
}

static mutable_ray_state_v0 decode_mutable_ray_bytes(const uint8_t *source) {
  mutable_ray_state_v0 ray = {};
  for (unsigned component = 0; component < 3; ++component) {
    ray.origin[component] = fp32_value(decode_u32_le(source + component * 4));
    ray.direction[component] =
        fp32_value(decode_u32_le(source + 12 + component * 4));
    ray.inverse_direction[component] =
        fp32_value(decode_u32_le(source + 24 + component * 4));
  }
  ray.t_min = fp32_value(decode_u32_le(source + 36));
  ray.t_max = fp32_value(decode_u32_le(source + 40));
  return ray;
}

static void encode_decode_context_bytes(
    uint8_t *destination,
    const typed_blas::as_decode_context_v0 &context) {
  encode_u32_le(destination + 0, context.bvh_format_profile_id);
  encode_u32_le(destination + 4, context.reserved_zero);
  encode_u64_le(destination + 8, context.as_object.object_id);
  encode_u32_le(destination + 16, context.as_object.generation);
  destination[20] = context.as_object.as_type;
  std::memcpy(destination + 21, context.as_object.reserved_zero,
              sizeof(context.as_object.reserved_zero));
  encode_u64_le(destination + 24, context.device_base);
  encode_u64_le(destination + 32, context.device_range_bytes);
}

static typed_blas::as_decode_context_v0 decode_decode_context_bytes(
    const uint8_t *source) {
  typed_blas::as_decode_context_v0 context = {};
  context.bvh_format_profile_id = decode_u32_le(source + 0);
  context.reserved_zero = decode_u32_le(source + 4);
  context.as_object.object_id = decode_u64_le(source + 8);
  context.as_object.generation = decode_u32_le(source + 16);
  context.as_object.as_type = source[20];
  std::memcpy(context.as_object.reserved_zero, source + 21,
              sizeof(context.as_object.reserved_zero));
  context.device_base = decode_u64_le(source + 24);
  context.device_range_bytes = decode_u64_le(source + 32);
  return context;
}

static void encode_committed_hit_bytes(
    uint8_t *destination, const committed_hit_projection_v0 &hit) {
  destination[0] = hit.valid;
  destination[1] = hit.geometry_type;
  destination[2] = hit.hit_kind;
  destination[3] = hit.attribute_word_count;
  destination[4] = hit.attribute_location;
  destination[5] = hit.attribute_format;
  std::memcpy(destination + 6, hit.reserved_zero0,
              sizeof(hit.reserved_zero0));
  encode_u32_le(destination + 8, fp32_bits(hit.hit_t));
  encode_u32_le(destination + 12, hit.policy_flags);
  encode_u64_le(destination + 16, hit.instance_metadata_ref);
  encode_u32_le(destination + 24, hit.primitive_index);
  encode_u32_le(destination + 28, hit.geometry_index);
  encode_u32_le(destination + 32, hit.instance_index);
  encode_u32_le(destination + 36, hit.instance_custom_index);
  encode_u32_le(destination + 40, hit.instance_sbt_contribution);
  encode_u32_le(destination + 44, hit.reserved_zero1);
  for (unsigned word = 0; word < 4; ++word) {
    encode_u32_le(destination + 48 + word * 4, hit.inline_attributes[word]);
  }
}

static committed_hit_projection_v0 decode_committed_hit_bytes(
    const uint8_t *source) {
  committed_hit_projection_v0 hit = {};
  hit.valid = source[0];
  hit.geometry_type = source[1];
  hit.hit_kind = source[2];
  hit.attribute_word_count = source[3];
  hit.attribute_location = source[4];
  hit.attribute_format = source[5];
  std::memcpy(hit.reserved_zero0, source + 6, sizeof(hit.reserved_zero0));
  hit.hit_t = fp32_value(decode_u32_le(source + 8));
  hit.policy_flags = decode_u32_le(source + 12);
  hit.instance_metadata_ref = decode_u64_le(source + 16);
  hit.primitive_index = decode_u32_le(source + 24);
  hit.geometry_index = decode_u32_le(source + 28);
  hit.instance_index = decode_u32_le(source + 32);
  hit.instance_custom_index = decode_u32_le(source + 36);
  hit.instance_sbt_contribution = decode_u32_le(source + 40);
  hit.reserved_zero1 = decode_u32_le(source + 44);
  for (unsigned word = 0; word < 4; ++word) {
    hit.inline_attributes[word] = decode_u32_le(source + 48 + word * 4);
  }
  return hit;
}

static void encode_retained_candidate_bytes(
    uint8_t *destination,
    const retained_candidate_projection_v0 &candidate) {
  const typed_primitive::primitive_identity_policy_facts_v0 &identity =
      candidate.identity_and_policy;
  encode_u64_le(destination + 0, identity.instance_metadata_ref);
  encode_u32_le(destination + 8, identity.primitive_index);
  encode_u32_le(destination + 12, identity.geometry_index);
  encode_u32_le(destination + 16, identity.instance_index);
  encode_u32_le(destination + 20, identity.instance_custom_index);
  encode_u32_le(destination + 24, identity.instance_sbt_contribution);
  destination[28] = identity.geometry_type;
  destination[29] = identity.geometry_policy_flags;
  destination[30] = identity.instance_policy_flags;
  destination[31] = identity.effective_policy_flags;
  const typed_primitive::triangle_hit_facts_v0 &triangle =
      candidate.triangle_hit;
  encode_u32_le(destination + 32, triangle.hit_t_bits);
  encode_u32_le(destination + 36, triangle.bary_vertex1_bits);
  encode_u32_le(destination + 40, triangle.bary_vertex2_bits);
  destination[44] = triangle.hit_kind;
  std::memcpy(destination + 45, triangle.reserved_zero,
              sizeof(triangle.reserved_zero));
}

static retained_candidate_projection_v0
decode_retained_candidate_bytes(const uint8_t *source) {
  retained_candidate_projection_v0 candidate = {};
  typed_primitive::primitive_identity_policy_facts_v0 &identity =
      candidate.identity_and_policy;
  identity.instance_metadata_ref = decode_u64_le(source + 0);
  identity.primitive_index = decode_u32_le(source + 8);
  identity.geometry_index = decode_u32_le(source + 12);
  identity.instance_index = decode_u32_le(source + 16);
  identity.instance_custom_index = decode_u32_le(source + 20);
  identity.instance_sbt_contribution = decode_u32_le(source + 24);
  identity.geometry_type = source[28];
  identity.geometry_policy_flags = source[29];
  identity.instance_policy_flags = source[30];
  identity.effective_policy_flags = source[31];
  typed_primitive::triangle_hit_facts_v0 &triangle =
      candidate.triangle_hit;
  triangle.hit_t_bits = decode_u32_le(source + 32);
  triangle.bary_vertex1_bits = decode_u32_le(source + 36);
  triangle.bary_vertex2_bits = decode_u32_le(source + 40);
  triangle.hit_kind = source[44];
  std::memcpy(triangle.reserved_zero, source + 45,
              sizeof(triangle.reserved_zero));
  return candidate;
}

static void encode_primitive_resume_bytes(
    uint8_t *destination,
    const typed_primitive::primitive_resume_data_v0 &resume) {
  encode_u64_le(destination + 0, resume.leaf_fetch_address);
  encode_u64_le(destination + 8, resume.remaining_slot_mask);
}

static typed_primitive::primitive_resume_data_v0
decode_primitive_resume_bytes(const uint8_t *source) {
  typed_primitive::primitive_resume_data_v0 resume = {};
  resume.leaf_fetch_address = decode_u64_le(source + 0);
  resume.remaining_slot_mask = decode_u64_le(source + 8);
  return resume;
}

static void encode_instance_projection_bytes(
    uint8_t *destination,
    const typed_stack::instance_shader_projection_v0 &instance) {
  encode_u64_le(destination + 0, instance.instance_metadata_ref);
  encode_u32_le(destination + 8, instance.instance_index);
  encode_u32_le(destination + 12, instance.instance_custom_index);
  encode_u32_le(destination + 16,
                instance.instance_sbt_contribution);
  destination[20] = instance.instance_policy_flags;
  std::memcpy(destination + 21, instance.reserved_zero,
              sizeof(instance.reserved_zero));
}

static typed_stack::instance_shader_projection_v0
decode_instance_projection_bytes(const uint8_t *source) {
  typed_stack::instance_shader_projection_v0 instance = {};
  instance.instance_metadata_ref = decode_u64_le(source + 0);
  instance.instance_index = decode_u32_le(source + 8);
  instance.instance_custom_index = decode_u32_le(source + 12);
  instance.instance_sbt_contribution = decode_u32_le(source + 16);
  instance.instance_policy_flags = source[20];
  std::memcpy(instance.reserved_zero, source + 21,
              sizeof(instance.reserved_zero));
  return instance;
}

static void encode_parent_frame_bytes(
    uint8_t *destination,
    const traversal_frame_projection_v0 &frame) {
  encode_mutable_ray_bytes(destination + 0, frame.ray);
  encode_u32_le(destination + 44, frame.traversal_level);
  encode_u32_le(destination + 48,
                frame.frontier_marker.frontier_top);
  encode_u32_le(destination + 52,
                frame.frontier_marker.frontier_count);
  encode_u32_le(destination + 56,
                frame.frontier_marker.level_frame_depth);
  encode_u32_le(destination + 60,
                frame.frontier_marker.reserved_zero);
  encode_decode_context_bytes(destination + 64,
                              frame.current_decode_context);
  encode_instance_projection_bytes(destination + 104,
                                   frame.current_instance);
}

static traversal_frame_projection_v0 decode_parent_frame_bytes(
    const uint8_t *source) {
  traversal_frame_projection_v0 frame = {};
  frame.ray = decode_mutable_ray_bytes(source + 0);
  frame.traversal_level = decode_u32_le(source + 44);
  frame.frontier_marker.frontier_top = decode_u32_le(source + 48);
  frame.frontier_marker.frontier_count = decode_u32_le(source + 52);
  frame.frontier_marker.level_frame_depth = decode_u32_le(source + 56);
  frame.frontier_marker.reserved_zero = decode_u32_le(source + 60);
  frame.current_decode_context = decode_decode_context_bytes(source + 64);
  frame.current_instance = decode_instance_projection_bytes(source + 104);
  return frame;
}

static bool valid_committed_hit(
    const committed_hit_projection_v0 &hit) {
  return hit.valid <= 1 && hit.attribute_word_count <= 4 &&
         bytes_are_zero(hit.reserved_zero0,
                        sizeof(hit.reserved_zero0)) &&
         hit.reserved_zero1 == 0 &&
         (hit.valid == 0 || std::isfinite(hit.hit_t));
}

static bool valid_parent_frame(
    const traversal_frame_projection_v0 &frame) {
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(frame.ray.origin[component]) ||
        !std::isfinite(frame.ray.direction[component])) {
      return false;
    }
  }
  const typed_blas::as_decode_context_v0 &context =
      frame.current_decode_context;
  return std::isfinite(frame.ray.t_min) &&
         std::isfinite(frame.ray.t_max) &&
         frame.ray.t_min <= frame.ray.t_max &&
         frame.traversal_level == 0 &&
         frame.frontier_marker.frontier_top ==
             frame.frontier_marker.frontier_count &&
         frame.frontier_marker.frontier_top <=
             kFrontierEntryCapacity &&
         frame.frontier_marker.level_frame_depth == 0 &&
         frame.frontier_marker.reserved_zero == 0 &&
         context.bvh_format_profile_id == kLayoutProfileId &&
         context.reserved_zero == 0 &&
         context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         context.as_object.as_type == 1 &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         context.device_base != 0 &&
         context.device_range_bytes >= 64 &&
         bytes_are_zero(frame.current_instance.reserved_zero,
                        sizeof(frame.current_instance.reserved_zero));
}

static bool valid_root_operands(const root_private_operands_v0 &operands) {
  const mutable_ray_state_v0 &ray = operands.mutable_ray;
  const typed_blas::as_decode_context_v0 &context =
      operands.decode_context;
  const committed_hit_projection_v0 &hit = operands.committed_hit;
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(ray.origin[component]) ||
        !std::isfinite(ray.direction[component])) {
      return false;
    }
  }
  return std::isfinite(ray.t_min) && std::isfinite(ray.t_max) &&
         ray.t_min <= ray.t_max &&
         context.bvh_format_profile_id == kLayoutProfileId &&
         context.reserved_zero == 0 && context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         (context.as_object.as_type == 1 ||
          context.as_object.as_type == typed_blas::kAsTypeBlas) &&
         context.device_base != 0 && context.device_range_bytes >= 64 &&
         valid_committed_hit(hit);
}

static status_kind validate_slot_owner(const shadow_slot_v0 &slot,
                                       const owner_binding_v0 &owner) {
  if (!valid_owner(owner)) return kStatusInvalidOwner;
  if (!owners_equal(slot.owner, owner)) return kStatusOwnerMismatch;
  return kStatusOk;
}

static void initialize_plan(access_plan_v0 *plan,
                            const owner_binding_v0 &owner) {
  std::memset(plan, 0, sizeof(*plan));
  plan->owner = owner;
}

static status_kind append_range_to_plan(
    access_plan_v0 *plan, const owner_binding_v0 &owner,
    const region_binding_v0 &region, field_kind field,
    access_kind access, uint32_t slot_offset, uint32_t byte_count) {
  if (plan == NULL || byte_count == 0 ||
      (access != kAccessRead && access != kAccessWrite) ||
      (field != kFieldFrontierMetadata &&
       field != kFieldFrontierEntry &&
       field != kFieldTransitionSpill &&
       field != kFieldMutableRayState &&
       field != kFieldAsDecodeContext &&
       field != kFieldCommittedHit &&
       field != kFieldParentFrame &&
       field != kFieldCurrentInstance &&
       field != kFieldRetainedCandidate &&
       field != kFieldPrimitiveResume)) {
    return kStatusInvalidArgument;
  }
  if (slot_offset >= kPrivateDataSlotBytes ||
      byte_count > kPrivateDataSlotBytes - slot_offset) {
    return kStatusInvalidRegion;
  }

  uint64_t slot_base = 0;
  status_kind status = slot_base_address(owner, region, &slot_base);
  if (status != kStatusOk) return status;

  const uint32_t range_end = slot_offset + byte_count;
  uint32_t chunk_offset =
      slot_offset - (slot_offset % kSharedAccessChunkBytes);
  while (chunk_offset < range_end) {
    if (plan->access_count >= kMaxAccessChunks) {
      return kStatusPlanCapacityExceeded;
    }
    const uint32_t first =
        slot_offset > chunk_offset ? slot_offset - chunk_offset : 0;
    const uint32_t chunk_end = chunk_offset + kSharedAccessChunkBytes;
    const uint32_t covered_end =
        range_end < chunk_end ? range_end : chunk_end;
    const uint32_t count = covered_end - (chunk_offset + first);
    const uint64_t low_bits =
        count == 32 ? uint64_t{0xffffffff}
                    : (uint64_t{1} << count) - uint64_t{1};

    uint64_t address = 0;
    if (!checked_add_u64(slot_base, chunk_offset, &address)) {
      return kStatusAddressOverflow;
    }
    shared_chunk_access_v0 &chunk =
        plan->accesses[plan->access_count++];
    chunk.aligned_32b_address = address;
    chunk.byte_mask = static_cast<uint32_t>(low_bits << first);
    chunk.slot_byte_offset =
        static_cast<uint16_t>(chunk_offset + first);
    chunk.byte_count = static_cast<uint8_t>(count);
    chunk.field_kind = static_cast<uint8_t>(field);
    chunk.access_kind = static_cast<uint8_t>(access);
    chunk_offset += kSharedAccessChunkBytes;
  }
  return kStatusOk;
}

static status_kind build_metadata_plan(
    access_plan_v0 *plan, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_kind access) {
  return append_range_to_plan(plan, owner, region,
                              kFieldFrontierMetadata, access,
                              kFrontierMetadataOffset,
                              kFrontierMetadataBytes);
}

static bool child_item_is_zero(
    const typed_node::compact_child_work_item_v0 &item) {
  const typed_node::compact_child_work_item_v0 zero = {};
  return std::memcmp(&item, &zero, sizeof(item)) == 0;
}

}  // namespace

bool owners_equal(const owner_binding_v0 &lhs,
                  const owner_binding_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.resident_warp_id == rhs.resident_warp_id &&
         lhs.request_identity == rhs.request_identity &&
         lhs.generation == rhs.generation &&
         lhs.private_slot_id == rhs.private_slot_id &&
         lhs.lane_id == rhs.lane_id &&
         bytes_are_zero(lhs.reserved_zero, sizeof(lhs.reserved_zero)) &&
         bytes_are_zero(rhs.reserved_zero, sizeof(rhs.reserved_zero));
}

status_kind initialize_shadow_slot(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const frontier_metadata_image_v0 &metadata,
    access_plan_v0 *metadata_write_plan) {
  if (slot == NULL || metadata_write_plan == NULL) {
    return kStatusInvalidArgument;
  }
  if (!valid_owner(owner)) return kStatusInvalidOwner;
  uint64_t ignored_slot_base = 0;
  status_kind status = slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;
  if (!valid_metadata(metadata)) return kStatusInvalidMetadata;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = build_metadata_plan(&plan, owner, region, kAccessWrite);
  if (status != kStatusOk) return status;

  shadow_slot_v0 initialized = {};
  initialized.owner = owner;
  encode_metadata_bytes(initialized.bytes + kFrontierMetadataOffset,
                        metadata);
  *slot = initialized;
  *metadata_write_plan = plan;
  return kStatusOk;
}

status_kind build_frontier_metadata_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan) {
  if (read_plan == NULL) return kStatusInvalidArgument;
  const status_kind owner_status = validate_slot_owner(slot, owner);
  if (owner_status != kStatusOk) return owner_status;
  uint64_t ignored_slot_base = 0;
  const status_kind region_status =
      slot_base_address(owner, region, &ignored_slot_base);
  if (region_status != kStatusOk) return region_status;
  initialize_plan(read_plan, owner);
  return build_metadata_plan(read_plan, owner, region, kAccessRead);
}

status_kind build_nonempty_pop_operand_read_plan(
    const owner_binding_v0 &owner, const region_binding_v0 &region,
    const frontier_metadata_image_v0 &returned_metadata,
    access_plan_v0 *read_plan) {
  if (read_plan == NULL || !valid_metadata(returned_metadata) ||
      returned_metadata.frontier_top == 0 ||
      returned_metadata.frontier_count == 0) {
    return kStatusInvalidMetadata;
  }
  uint64_t ignored_slot_base = 0;
  status_kind status =
      slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  const uint32_t top_index = returned_metadata.frontier_top - 1;
  status = append_range_to_plan(
      &plan, owner, region, kFieldFrontierEntry, kAccessRead,
      kFrontierEntriesOffset + top_index * kFrontierEntryBytes,
      kFrontierEntryBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldMutableRayState, kAccessRead,
      kMutableRayStateOffset, kMutableRayStateBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldAsDecodeContext, kAccessRead,
      kAsDecodeContextOffset, kAsDecodeContextBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldCommittedHit, kAccessRead,
      kCommittedHitOffset, kCommittedHitBytes);
  if (status != kStatusOk) return status;
  if (plan.access_count == 0 ||
      plan.access_count > kMaxNonemptyPopOperandChunks) {
    return kStatusPlanCapacityExceeded;
  }
  *read_plan = plan;
  return kStatusOk;
}

status_kind build_empty_pop_operand_read_plan(
    const owner_binding_v0 &owner, const region_binding_v0 &region,
    const frontier_metadata_image_v0 &returned_metadata,
    access_plan_v0 *read_plan) {
  if (read_plan == NULL || !valid_metadata(returned_metadata) ||
      returned_metadata.frontier_top !=
          returned_metadata.frontier_count) {
    return kStatusInvalidMetadata;
  }
  uint64_t ignored_slot_base = 0;
  status_kind status =
      slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  if (returned_metadata.current_level == 1 &&
      returned_metadata.level_frame_depth == 1) {
    status = append_range_to_plan(
        &plan, owner, region, kFieldParentFrame, kAccessRead,
        kParentFrameOffset, kParentFrameBytes);
  } else if (returned_metadata.current_level == 0 &&
             returned_metadata.level_frame_depth == 0 &&
             returned_metadata.frontier_count == 0) {
    status = append_range_to_plan(
        &plan, owner, region, kFieldCommittedHit, kAccessRead,
        kCommittedHitOffset, kCommittedHitBytes);
  } else {
    return kStatusInvalidMetadata;
  }
  if (status != kStatusOk) return status;
  if (plan.access_count == 0 ||
      plan.access_count > kMaxEmptyPopOperandChunks) {
    return kStatusPlanCapacityExceeded;
  }
  *read_plan = plan;
  return kStatusOk;
}

status_kind build_parent_frame_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan) {
  if (read_plan == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldParentFrame, kAccessRead,
      kParentFrameOffset, kParentFrameBytes);
  if (status != kStatusOk) return status;
  if (plan.access_count != 5) return kStatusPlanCapacityExceeded;
  *read_plan = plan;
  return kStatusOk;
}

status_kind initialize_root_shadow_slot(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const frontier_metadata_image_v0 &metadata,
    const root_private_operands_v0 &root_operands,
    access_plan_v0 *initial_write_plan) {
  if (slot == NULL || initial_write_plan == NULL ||
      !valid_root_operands(root_operands)) {
    return kStatusInvalidArgument;
  }
  if (!valid_owner(owner)) return kStatusInvalidOwner;
  uint64_t ignored_slot_base = 0;
  status_kind status = slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;
  if (!valid_metadata(metadata)) return kStatusInvalidMetadata;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldMutableRayState, kAccessWrite,
      kMutableRayStateOffset, kMutableRayStateBytes);
  if (status != kStatusOk) return status;
  status = build_metadata_plan(&plan, owner, region, kAccessWrite);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldAsDecodeContext, kAccessWrite,
      kAsDecodeContextOffset, kAsDecodeContextBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldCommittedHit, kAccessWrite,
      kCommittedHitOffset, kCommittedHitBytes);
  if (status != kStatusOk) return status;

  shadow_slot_v0 initialized = {};
  initialized.owner = owner;
  encode_mutable_ray_bytes(
      initialized.bytes + kMutableRayStateOffset,
      root_operands.mutable_ray);
  encode_metadata_bytes(initialized.bytes + kFrontierMetadataOffset,
                        metadata);
  encode_decode_context_bytes(
      initialized.bytes + kAsDecodeContextOffset,
      root_operands.decode_context);
  encode_committed_hit_bytes(
      initialized.bytes + kCommittedHitOffset,
      root_operands.committed_hit);
  *slot = initialized;
  *initial_write_plan = plan;
  return kStatusOk;
}

status_kind build_root_operand_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan) {
  if (read_plan == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldMutableRayState, kAccessRead,
      kMutableRayStateOffset, kMutableRayStateBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldAsDecodeContext, kAccessRead,
      kAsDecodeContextOffset, kAsDecodeContextBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldCommittedHit, kAccessRead,
      kCommittedHitOffset, kCommittedHitBytes);
  if (status != kStatusOk) return status;
  *read_plan = plan;
  return kStatusOk;
}

status_kind build_primitive_operand_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan) {
  if (read_plan == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldMutableRayState, kAccessRead,
      kMutableRayStateOffset, kMutableRayStateBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldAsDecodeContext, kAccessRead,
      kAsDecodeContextOffset, kAsDecodeContextBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldCommittedHit, kAccessRead,
      kCommittedHitOffset, kCommittedHitBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldCurrentInstance, kAccessRead,
      kCurrentInstanceOffset, kCurrentInstanceBytes);
  if (status != kStatusOk) return status;
  *read_plan = plan;
  return kStatusOk;
}

status_kind decode_root_private_operands(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    root_private_operands_v0 *operands) {
  if (operands == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  root_private_operands_v0 decoded = {};
  decoded.mutable_ray = decode_mutable_ray_bytes(
      slot.bytes + kMutableRayStateOffset);
  decoded.decode_context = decode_decode_context_bytes(
      slot.bytes + kAsDecodeContextOffset);
  decoded.committed_hit = decode_committed_hit_bytes(
      slot.bytes + kCommittedHitOffset);
  if (!valid_root_operands(decoded)) return kStatusInvalidArgument;
  *operands = decoded;
  return kStatusOk;
}

status_kind decode_metadata(const shadow_slot_v0 &slot,
                            const owner_binding_v0 &owner,
                            frontier_metadata_image_v0 *metadata) {
  if (metadata == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  const frontier_metadata_image_v0 decoded = decode_metadata_bytes(
      slot.bytes + kFrontierMetadataOffset);
  if (!valid_metadata(decoded)) return kStatusInvalidMetadata;
  *metadata = decoded;
  return kStatusOk;
}

status_kind decode_entry(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    uint32_t entry_index,
    typed_node::compact_child_work_item_v0 *entry) {
  if (entry == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  if (entry_index >= kFrontierEntryCapacity) {
    return kStatusInvalidEntryIndex;
  }
  *entry = decode_entry_bytes(
      slot.bytes + kFrontierEntriesOffset +
      entry_index * kFrontierEntryBytes);
  return kStatusOk;
}

status_kind decode_parent_frame(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    traversal_frame_projection_v0 *parent_frame) {
  if (parent_frame == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  const traversal_frame_projection_v0 decoded =
      decode_parent_frame_bytes(slot.bytes + kParentFrameOffset);
  if (!valid_parent_frame(decoded)) return kStatusInvalidArgument;
  *parent_frame = decoded;
  return kStatusOk;
}

status_kind decode_current_instance(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    instance_shader_projection_v0 *current_instance) {
  if (current_instance == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  *current_instance = decode_instance_projection_bytes(
      slot.bytes + kCurrentInstanceOffset);
  return kStatusOk;
}

status_kind decode_committed_hit(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    committed_hit_projection_v0 *committed_hit) {
  if (committed_hit == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  const committed_hit_projection_v0 decoded =
      decode_committed_hit_bytes(slot.bytes + kCommittedHitOffset);
  if (!valid_committed_hit(decoded)) return kStatusInvalidArgument;
  *committed_hit = decoded;
  return kStatusOk;
}

status_kind decode_retained_candidate(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    retained_candidate_projection_v0 *retained_candidate) {
  if (retained_candidate == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  *retained_candidate = decode_retained_candidate_bytes(
      slot.bytes + kRetainedCandidateOffset);
  return kStatusOk;
}

status_kind decode_primitive_resume(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    typed_primitive::primitive_resume_data_v0 *primitive_resume) {
  if (primitive_resume == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  *primitive_resume = decode_primitive_resume_bytes(
      slot.bytes + kPrimitiveResumeOffset);
  return kStatusOk;
}

status_kind apply_primitive_result_state(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const committed_hit_projection_v0 *committed_hit,
    const retained_candidate_projection_v0 *retained_candidate,
    const typed_primitive::primitive_resume_data_v0 *primitive_resume,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL ||
      (committed_hit == NULL && retained_candidate == NULL &&
       primitive_resume == NULL) ||
      (committed_hit != NULL && retained_candidate != NULL)) {
    return kStatusInvalidArgument;
  }
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  if ((committed_hit != NULL && !valid_committed_hit(*committed_hit)) ||
      (retained_candidate != NULL &&
       (retained_candidate->identity_and_policy.instance_metadata_ref == 0 ||
        (retained_candidate->identity_and_policy.geometry_type !=
             typed_primitive::kGeometryTypeTriangle &&
         retained_candidate->identity_and_policy.geometry_type !=
             typed_primitive::kGeometryTypeProcedural) ||
        !bytes_are_zero(
            retained_candidate->triangle_hit.reserved_zero,
            sizeof(retained_candidate->triangle_hit.reserved_zero)))) ||
      (primitive_resume != NULL &&
       (primitive_resume->leaf_fetch_address == 0 ||
        (primitive_resume->leaf_fetch_address &
         uint64_t{kSharedAccessChunkBytes - 1}) != 0 ||
        primitive_resume->remaining_slot_mask == 0))) {
    return kStatusInvalidArgument;
  }

  shadow_slot_v0 updated = *slot;
  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  if (committed_hit != NULL) {
    status = append_range_to_plan(
        &plan, owner, region, kFieldCommittedHit, kAccessWrite,
        kCommittedHitOffset, kCommittedHitBytes);
    if (status != kStatusOk) return status;
    encode_committed_hit_bytes(
        updated.bytes + kCommittedHitOffset, *committed_hit);
  }
  if (retained_candidate != NULL) {
    status = append_range_to_plan(
        &plan, owner, region, kFieldRetainedCandidate, kAccessWrite,
        kRetainedCandidateOffset, kRetainedCandidateBytes);
    if (status != kStatusOk) return status;
    encode_retained_candidate_bytes(
        updated.bytes + kRetainedCandidateOffset, *retained_candidate);
  }
  if (primitive_resume != NULL) {
    status = append_range_to_plan(
        &plan, owner, region, kFieldPrimitiveResume, kAccessWrite,
        kPrimitiveResumeOffset, kPrimitiveResumeBytes);
    if (status != kStatusOk) return status;
    encode_primitive_resume_bytes(
        updated.bytes + kPrimitiveResumeOffset, *primitive_resume);
  }
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind capture_parent_frame(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    traversal_frame_projection_v0 *parent_frame) {
  if (parent_frame == NULL) return kStatusInvalidArgument;
  root_private_operands_v0 operands = {};
  frontier_metadata_image_v0 metadata = {};
  instance_shader_projection_v0 current_instance = {};
  status_kind status =
      decode_root_private_operands(slot, owner, &operands);
  if (status != kStatusOk) return status;
  status = decode_metadata(slot, owner, &metadata);
  if (status != kStatusOk) return status;
  status = decode_current_instance(slot, owner, &current_instance);
  if (status != kStatusOk) return status;
  if (metadata.current_level != 0 ||
      metadata.level_frame_depth != 0 ||
      operands.decode_context.as_object.as_type !=
          1) {
    return kStatusInvalidMetadata;
  }

  traversal_frame_projection_v0 captured = {};
  captured.ray = operands.mutable_ray;
  captured.traversal_level = 0;
  captured.frontier_marker.frontier_top = metadata.frontier_top;
  captured.frontier_marker.frontier_count = metadata.frontier_count;
  captured.frontier_marker.level_frame_depth =
      metadata.level_frame_depth;
  captured.current_decode_context = operands.decode_context;
  captured.current_instance = current_instance;
  if (!valid_parent_frame(captured)) return kStatusInvalidMetadata;
  *parent_frame = captured;
  return kStatusOk;
}

status_kind apply_parent_frame_push(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const traversal_frame_projection_v0 &parent_frame,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL ||
      !valid_parent_frame(parent_frame)) {
    return kStatusInvalidArgument;
  }
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  frontier_metadata_image_v0 metadata = {};
  status = decode_metadata(*slot, owner, &metadata);
  if (status != kStatusOk) return status;
  if (metadata.current_level != 0 ||
      metadata.level_frame_depth != 0 ||
      parent_frame.frontier_marker.frontier_top !=
          metadata.frontier_top ||
      parent_frame.frontier_marker.frontier_count !=
          metadata.frontier_count) {
    return kStatusInvalidDelta;
  }

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldParentFrame, kAccessWrite,
      kParentFrameOffset, kParentFrameBytes);
  if (status != kStatusOk) return status;
  status = build_metadata_plan(&plan, owner, region, kAccessWrite);
  if (status != kStatusOk) return status;

  shadow_slot_v0 updated = *slot;
  encode_parent_frame_bytes(updated.bytes + kParentFrameOffset,
                            parent_frame);
  metadata.current_level = 1;
  metadata.level_frame_depth = 1;
  encode_metadata_bytes(updated.bytes + kFrontierMetadataOffset,
                        metadata);
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind apply_parent_restore_delta(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::frontier_level_delta_v0 &delta,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  frontier_metadata_image_v0 metadata = {};
  status = decode_metadata(*slot, owner, &metadata);
  if (status != kStatusOk) return status;
  traversal_frame_projection_v0 frame = {};
  status = decode_parent_frame(*slot, owner, &frame);
  if (status != kStatusOk) return status;
  if (metadata.current_level != 1 ||
      metadata.level_frame_depth != 1 ||
      metadata.frontier_top != frame.frontier_marker.frontier_top ||
      metadata.frontier_count != frame.frontier_marker.frontier_count ||
      delta.action != typed_stack::kFrontierActionPopFrame ||
      !bytes_are_zero(delta.reserved_zero,
                      sizeof(delta.reserved_zero)) ||
      delta.new_frontier_top != frame.frontier_marker.frontier_top ||
      delta.new_frontier_count !=
          frame.frontier_marker.frontier_count ||
      delta.new_current_level != frame.traversal_level ||
      delta.new_level_frame_depth !=
          frame.frontier_marker.level_frame_depth ||
      delta.max_level_depth != metadata.max_level_depth) {
    return kStatusInvalidDelta;
  }

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = build_metadata_plan(&plan, owner, region, kAccessWrite);
  if (status != kStatusOk) return status;
  shadow_slot_v0 updated = *slot;
  metadata.frontier_top = delta.new_frontier_top;
  metadata.frontier_count = delta.new_frontier_count;
  metadata.current_level = delta.new_current_level;
  metadata.level_frame_depth = delta.new_level_frame_depth;
  encode_metadata_bytes(updated.bytes + kFrontierMetadataOffset,
                        metadata);
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind apply_parent_state_restore(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const traversal_frame_projection_v0 &parent_frame,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL ||
      !valid_parent_frame(parent_frame)) {
    return kStatusInvalidArgument;
  }
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldMutableRayState, kAccessWrite,
      kMutableRayStateOffset, kMutableRayStateBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldAsDecodeContext, kAccessWrite,
      kAsDecodeContextOffset, kAsDecodeContextBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldCurrentInstance, kAccessWrite,
      kCurrentInstanceOffset, kCurrentInstanceBytes);
  if (status != kStatusOk) return status;
  if (plan.access_count != 6) return kStatusPlanCapacityExceeded;

  shadow_slot_v0 updated = *slot;
  encode_mutable_ray_bytes(updated.bytes + kMutableRayStateOffset,
                           parent_frame.ray);
  encode_decode_context_bytes(updated.bytes + kAsDecodeContextOffset,
                              parent_frame.current_decode_context);
  encode_instance_projection_bytes(updated.bytes + kCurrentInstanceOffset,
                                   parent_frame.current_instance);
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind apply_instance_enter_state(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const mutable_ray_state_v0 &object_ray,
    const typed_blas::as_decode_context_v0 &blas_decode_context,
    const instance_shader_projection_v0 &current_instance,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL ||
      !bytes_are_zero(current_instance.reserved_zero,
                      sizeof(current_instance.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(object_ray.origin[component]) ||
        !std::isfinite(object_ray.direction[component]) ||
        !std::isfinite(object_ray.inverse_direction[component])) {
      return kStatusInvalidArgument;
    }
  }
  if (!std::isfinite(object_ray.t_min) ||
      !std::isfinite(object_ray.t_max) ||
      object_ray.t_min > object_ray.t_max ||
      blas_decode_context.bvh_format_profile_id != kLayoutProfileId ||
      blas_decode_context.reserved_zero != 0 ||
      blas_decode_context.as_object.object_id == 0 ||
      blas_decode_context.as_object.generation == 0 ||
      blas_decode_context.as_object.as_type != typed_blas::kAsTypeBlas ||
      !bytes_are_zero(
          blas_decode_context.as_object.reserved_zero,
          sizeof(blas_decode_context.as_object.reserved_zero)) ||
      blas_decode_context.device_base == 0 ||
      blas_decode_context.device_range_bytes < 64) {
    return kStatusInvalidArgument;
  }
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldMutableRayState, kAccessWrite,
      kMutableRayStateOffset, kMutableRayStateBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldAsDecodeContext, kAccessWrite,
      kAsDecodeContextOffset, kAsDecodeContextBytes);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldCurrentInstance, kAccessWrite,
      kCurrentInstanceOffset, kCurrentInstanceBytes);
  if (status != kStatusOk || plan.access_count != 6) {
    return status == kStatusOk ? kStatusPlanCapacityExceeded : status;
  }

  shadow_slot_v0 updated = *slot;
  encode_mutable_ray_bytes(updated.bytes + kMutableRayStateOffset,
                           object_ray);
  encode_decode_context_bytes(updated.bytes + kAsDecodeContextOffset,
                              blas_decode_context);
  encode_instance_projection_bytes(
      updated.bytes + kCurrentInstanceOffset, current_instance);
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind apply_append_delta(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::frontier_append_delta_v0 &delta,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  uint64_t ignored_slot_base = 0;
  status = slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;

  frontier_metadata_image_v0 metadata = {};
  status = decode_metadata(*slot, owner, &metadata);
  if (status != kStatusOk) return status;
  if (delta.action != typed_stack::kFrontierActionAppendChildren ||
      delta.reserved_zero != 0 ||
      delta.write_count > typed_stack::kMaxRemainderChildren ||
      delta.append_base_index != metadata.frontier_top ||
      delta.write_count > metadata.frontier_capacity - metadata.frontier_top ||
      delta.write_count > metadata.frontier_capacity - metadata.frontier_count ||
      delta.new_frontier_top !=
          metadata.frontier_top + delta.write_count ||
      delta.new_frontier_count !=
          metadata.frontier_count + delta.write_count) {
    return kStatusInvalidDelta;
  }
  for (unsigned index = delta.write_count;
       index < typed_stack::kMaxRemainderChildren; ++index) {
    if (!child_item_is_zero(delta.written_items[index])) {
      return kStatusInvalidDelta;
    }
  }

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  if (delta.write_count != 0) {
    status = append_range_to_plan(
        &plan, owner, region, kFieldFrontierEntry, kAccessWrite,
        kFrontierEntriesOffset +
            delta.append_base_index * kFrontierEntryBytes,
        delta.write_count * kFrontierEntryBytes);
    if (status != kStatusOk) return status;
  }
  status = build_metadata_plan(&plan, owner, region, kAccessWrite);
  if (status != kStatusOk) return status;

  shadow_slot_v0 updated = *slot;
  for (unsigned index = 0; index < delta.write_count; ++index) {
    encode_entry_bytes(
        updated.bytes + kFrontierEntriesOffset +
            (delta.append_base_index + index) * kFrontierEntryBytes,
        delta.written_items[index]);
  }
  metadata.frontier_top = delta.new_frontier_top;
  metadata.frontier_count = delta.new_frontier_count;
  encode_metadata_bytes(updated.bytes + kFrontierMetadataOffset,
                        metadata);
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind read_top_entry(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    typed_node::compact_child_work_item_v0 *entry,
    uint32_t *entry_index, access_plan_v0 *read_plan) {
  if (entry == NULL || entry_index == NULL || read_plan == NULL) {
    return kStatusInvalidArgument;
  }
  frontier_metadata_image_v0 metadata = {};
  status_kind status = decode_metadata(slot, owner, &metadata);
  if (status != kStatusOk) return status;
  uint64_t ignored_slot_base = 0;
  status = slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;
  if (metadata.frontier_top == 0 || metadata.frontier_count == 0) {
    return kStatusInvalidEntryIndex;
  }

  const uint32_t top_index = metadata.frontier_top - 1;
  typed_node::compact_child_work_item_v0 decoded = {};
  status = decode_entry(slot, owner, top_index, &decoded);
  if (status != kStatusOk) return status;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = build_metadata_plan(&plan, owner, region, kAccessRead);
  if (status != kStatusOk) return status;
  status = append_range_to_plan(
      &plan, owner, region, kFieldFrontierEntry, kAccessRead,
      kFrontierEntriesOffset + top_index * kFrontierEntryBytes,
      kFrontierEntryBytes);
  if (status != kStatusOk) return status;

  *entry = decoded;
  *entry_index = top_index;
  *read_plan = plan;
  return kStatusOk;
}

status_kind apply_pop_delta(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::frontier_pop_delta_v0 &delta,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  uint64_t ignored_slot_base = 0;
  status = slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;

  frontier_metadata_image_v0 metadata = {};
  status = decode_metadata(*slot, owner, &metadata);
  if (status != kStatusOk) return status;
  if (metadata.frontier_top == 0 || metadata.frontier_count == 0 ||
      delta.action != typed_stack::kFrontierActionPopChild ||
      delta.pop_count != 1 || delta.reserved_zero != 0 ||
      delta.popped_index != metadata.frontier_top - 1 ||
      delta.new_frontier_top != metadata.frontier_top - 1 ||
      delta.new_frontier_count != metadata.frontier_count - 1) {
    return kStatusInvalidDelta;
  }

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = build_metadata_plan(&plan, owner, region, kAccessWrite);
  if (status != kStatusOk) return status;

  shadow_slot_v0 updated = *slot;
  metadata.frontier_top = delta.new_frontier_top;
  metadata.frontier_count = delta.new_frontier_count;
  encode_metadata_bytes(updated.bytes + kFrontierMetadataOffset,
                        metadata);
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind apply_stack_selected_fetch_spill(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_stack::push_result_v0 &result,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL) return kStatusInvalidArgument;
  std::memset(write_plan, 0, sizeof(*write_plan));
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  uint64_t ignored_slot_base = 0;
  status = slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;
  if (!typed_stack::validate_push_result(result) ||
      result.result_kind != typed_stack::kStackPushedAndSelected ||
      result.output_valid_mask !=
          static_cast<uint8_t>(
              typed_stack::kFrontierDeltaValid |
              typed_stack::kSelectedFetchValid)) {
    return kStatusInvalidSpillPayload;
  }
  return apply_stack_selected_fetch_spill_payload(
      slot, owner, region, result.selected_fetch, write_plan);
}

status_kind apply_stack_selected_fetch_spill_payload(
    shadow_slot_v0 *slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region,
    const typed_node::selected_child_fetch_work_item_v0 &selected_fetch,
    access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL) return kStatusInvalidArgument;
  std::memset(write_plan, 0, sizeof(*write_plan));
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  uint64_t ignored_slot_base = 0;
  status = slot_base_address(owner, region, &ignored_slot_base);
  if (status != kStatusOk) return status;

  access_plan_v0 plan = {};
  initialize_plan(&plan, owner);
  status = append_range_to_plan(
      &plan, owner, region, kFieldTransitionSpill, kAccessWrite,
      kTransitionSpillOffset, kStackTransitionSpillBytes);
  if (status != kStatusOk) return status;

  shadow_slot_v0 updated = *slot;
  std::memset(updated.bytes + kTransitionSpillOffset, 0,
              kStackTransitionSpillBytes);
  encode_selected_fetch_bytes(
      updated.bytes + kTransitionSpillOffset,
      selected_fetch);
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind build_stack_selected_fetch_spill_read_plan(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    const region_binding_v0 &region, access_plan_v0 *read_plan) {
  if (read_plan == NULL) return kStatusInvalidArgument;
  std::memset(read_plan, 0, sizeof(*read_plan));
  const status_kind owner_status = validate_slot_owner(slot, owner);
  if (owner_status != kStatusOk) return owner_status;
  uint64_t ignored_slot_base = 0;
  const status_kind address_status =
      slot_base_address(owner, region, &ignored_slot_base);
  if (address_status != kStatusOk) return address_status;
  initialize_plan(read_plan, owner);
  return append_range_to_plan(
      read_plan, owner, region, kFieldTransitionSpill, kAccessRead,
      kTransitionSpillOffset, kStackTransitionSpillBytes);
}

status_kind decode_stack_selected_fetch_spill(
    const shadow_slot_v0 &slot, const owner_binding_v0 &owner,
    typed_node::selected_child_fetch_work_item_v0 *selected_fetch) {
  if (selected_fetch == NULL) return kStatusInvalidArgument;
  const status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;
  *selected_fetch = decode_selected_fetch_bytes(
      slot.bytes + kTransitionSpillOffset);
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusUnsupportedProfile:
      return "unsupported_profile";
    case kStatusInvalidOwner:
      return "invalid_owner";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusInvalidRegion:
      return "invalid_region";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusInvalidMetadata:
      return "invalid_metadata";
    case kStatusInvalidEntryIndex:
      return "invalid_entry_index";
    case kStatusInvalidDelta:
      return "invalid_delta";
    case kStatusPlanCapacityExceeded:
      return "plan_capacity_exceeded";
    case kStatusInvalidSpillPayload:
      return "invalid_spill_payload";
  }
  return "unknown";
}

}  // namespace private_frontier
}  // namespace v04
}  // namespace rtcore
