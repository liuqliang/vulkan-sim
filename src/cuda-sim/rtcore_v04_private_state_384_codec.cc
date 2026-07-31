#include "rtcore_v04_private_state_384_codec.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace {

static const uint8_t kAsTypeTlas = 1;
static const uint8_t kAsTypeBlas = 2;
static const uint32_t kCommittedPolicyMask = 0x1u;
static const uint8_t kGeometryPolicyMask = 0x1u;
static const uint8_t kEffectivePolicyMask = 0x1u;
static const uint32_t kFullChunkMask = 0xffffffffu;

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
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

static void store_u16(uint8_t *bytes, uint16_t value) {
  bytes[0] = static_cast<uint8_t>(value);
  bytes[1] = static_cast<uint8_t>(value >> 8);
}

static void store_u32(uint8_t *bytes, uint32_t value) {
  for (unsigned index = 0; index < 4; ++index) {
    bytes[index] = static_cast<uint8_t>(value >> (index * 8));
  }
}

static void store_u64(uint8_t *bytes, uint64_t value) {
  for (unsigned index = 0; index < 8; ++index) {
    bytes[index] = static_cast<uint8_t>(value >> (index * 8));
  }
}

static uint16_t load_u16(const uint8_t *bytes) {
  return static_cast<uint16_t>(
      static_cast<uint16_t>(bytes[0]) |
      static_cast<uint16_t>(bytes[1]) << 8);
}

static uint32_t load_u32(const uint8_t *bytes) {
  uint32_t value = 0;
  for (unsigned index = 0; index < 4; ++index) {
    value |= static_cast<uint32_t>(bytes[index]) << (index * 8);
  }
  return value;
}

static uint64_t load_u64(const uint8_t *bytes) {
  uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index) {
    value |= static_cast<uint64_t>(bytes[index]) << (index * 8);
  }
  return value;
}

static status_kind validate_profiles(uint32_t private_layout_profile_id,
                                     uint32_t bvh_format_profile_id) {
  if (private_layout_profile_id != kPrivateLayoutProfileId) {
    return kStatusUnsupportedPrivateLayout;
  }
  if (bvh_format_profile_id != kGenRtBvhFormatProfileId) {
    return kStatusUnsupportedBvhProfile;
  }
  return kStatusOk;
}

static status_kind validate_control(const control_tags_v1 &control) {
  if (!bytes_are_zero(control.reserved_zero,
                      sizeof(control.reserved_zero)) ||
      control.union_arm > kUnionArmTransition ||
      control.boundary_reason > kBoundaryReasonProceduralIntersection ||
      control.pending_parent_resume_valid > 1 ||
      control.parent_restore_valid > 1) {
    return kStatusInvalidControl;
  }
  if (control.union_arm == kUnionArmBoundary) {
    if (control.boundary_reason == kBoundaryReasonNone ||
        control.pending_parent_resume_valid != 0) {
      return kStatusInvalidControl;
    }
  } else if (control.union_arm == kUnionArmTransition) {
    if (control.boundary_reason != kBoundaryReasonNone) {
      return kStatusInvalidControl;
    }
  } else if (control.boundary_reason != kBoundaryReasonNone ||
             control.pending_parent_resume_valid != 0) {
    return kStatusInvalidControl;
  }
  return kStatusOk;
}

static status_kind validate_ray(const ray_v1 &ray) {
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(ray.origin[component]) ||
        !std::isfinite(ray.direction[component])) {
      return kStatusInvalidRay;
    }
  }
  if (!std::isfinite(ray.t_min) || !std::isfinite(ray.t_max) ||
      ray.t_min > ray.t_max) {
    return kStatusInvalidRay;
  }
  return kStatusOk;
}

static status_kind validate_as_context(const as_context_v1 &context,
                                       bool cull_mask_present) {
  if (!bytes_are_zero(context.reserved_zero,
                      sizeof(context.reserved_zero))) {
    return kStatusInvalidReservedBits;
  }
  if (context.as_object_id == 0 || context.device_base == 0 ||
      context.device_range_bytes == 0 ||
      context.as_object_generation == 0 ||
      (context.as_type != kAsTypeTlas &&
       context.as_type != kAsTypeBlas) ||
      (!cull_mask_present && context.cull_mask != 0)) {
    return kStatusInvalidAsContext;
  }
  if (context.device_range_bytes >
      std::numeric_limits<uint64_t>::max() - context.device_base) {
    return kStatusAddressOverflow;
  }
  return kStatusOk;
}

static bool range_contains(const as_context_v1 &context,
                           uint64_t address, uint64_t bytes) {
  if (bytes > context.device_range_bytes ||
      address < context.device_base) {
    return false;
  }
  const uint64_t relative = address - context.device_base;
  return relative <= context.device_range_bytes - bytes;
}

static bool relative_range_contains(const as_context_v1 &context,
                                    uint64_t offset, uint64_t bytes) {
  return bytes <= context.device_range_bytes &&
         offset <= context.device_range_bytes - bytes;
}

static bool valid_instance_policy(uint8_t policy) {
  return (policy & ~typed_primitive::kSupportedInstancePolicyMask) == 0 &&
         !((policy & typed_primitive::kInstanceForceOpaque) != 0 &&
           (policy & typed_primitive::kInstanceForceNoOpaque) != 0);
}

static status_kind validate_current_instance(
    const current_instance_v1 &instance, uint8_t active_domain) {
  if (!bytes_are_zero(instance.reserved_zero,
                      sizeof(instance.reserved_zero))) {
    return kStatusInvalidReservedBits;
  }
  if ((instance.instance_sbt_contribution & 0xff000000u) != 0 ||
      !valid_instance_policy(instance.instance_policy_flags)) {
    return kStatusInvalidCurrentInstance;
  }
  if (active_domain == short_stack::kDomainBlas) {
    if (instance.instance_metadata_ref == 0) {
      return kStatusInvalidCurrentInstance;
    }
  } else if (instance.instance_metadata_ref == 0 &&
             (instance.instance_index != 0 ||
              instance.instance_custom_index != 0 ||
              instance.instance_sbt_contribution != 0 ||
              instance.instance_policy_flags != 0)) {
    return kStatusInvalidCurrentInstance;
  }
  return kStatusOk;
}

static status_kind validate_hit(
    const typed_stack::committed_hit_projection_v0 &hit) {
  if (!bytes_are_zero(hit.reserved_zero0,
                      sizeof(hit.reserved_zero0)) ||
      hit.reserved_zero1 != 0) {
    return kStatusInvalidReservedBits;
  }
  if (hit.valid > 1 || hit.geometry_type > 2 ||
      hit.attribute_word_count > 4 ||
      hit.attribute_location > 2 ||
      hit.attribute_format > 2 ||
      (hit.policy_flags & ~kCommittedPolicyMask) != 0 ||
      (hit.instance_sbt_contribution & 0xff000000u) != 0) {
    return kStatusInvalidCommittedHit;
  }
  if (hit.valid == 0) {
    const typed_stack::committed_hit_projection_v0 zero = {};
    return std::memcmp(&hit, &zero, sizeof(hit)) == 0
               ? kStatusOk
               : kStatusInvalidCommittedHit;
  }
  if (!std::isfinite(hit.hit_t) ||
      hit.instance_metadata_ref == 0) {
    return kStatusInvalidCommittedHit;
  }
  if (hit.geometry_type == typed_primitive::kGeometryTypeTriangle) {
    if ((hit.hit_kind != typed_primitive::kHitKindFrontFacing &&
         hit.hit_kind != typed_primitive::kHitKindBackFacing) ||
        hit.attribute_word_count != 2 ||
        hit.attribute_location == 0 ||
        hit.attribute_format != 1) {
      return kStatusInvalidCommittedHit;
    }
  } else if (hit.geometry_type ==
             typed_primitive::kGeometryTypeProcedural) {
    if (hit.hit_kind > 0x7fu ||
        hit.attribute_format !=
            (hit.attribute_word_count == 0 ? 0 : 2) ||
        (hit.attribute_word_count == 0
             ? hit.attribute_location != 0
             : hit.attribute_location == 0)) {
      return kStatusInvalidCommittedHit;
    }
  } else {
    return kStatusInvalidCommittedHit;
  }
  return kStatusOk;
}

static uint32_t pack_hit_control(
    const typed_stack::committed_hit_projection_v0 &hit) {
  return static_cast<uint32_t>(hit.valid) |
         static_cast<uint32_t>(hit.geometry_type) << 1 |
         static_cast<uint32_t>(hit.hit_kind) << 8 |
         static_cast<uint32_t>(hit.attribute_word_count) << 16 |
         static_cast<uint32_t>(hit.attribute_location) << 24 |
         static_cast<uint32_t>(hit.attribute_format) << 26;
}

static status_kind unpack_hit_control(
    uint32_t packed,
    typed_stack::committed_hit_projection_v0 *hit) {
  static const uint32_t kDefinedMask =
      0x00000001u | 0x00000006u | 0x0000ff00u |
      0x00070000u | 0x03000000u | 0x1c000000u;
  if (hit == NULL) return kStatusInvalidArgument;
  if ((packed & ~kDefinedMask) != 0) {
    return kStatusInvalidReservedBits;
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

static void encode_ray(uint8_t *bytes, const ray_v1 &ray) {
  for (unsigned component = 0; component < 3; ++component) {
    store_u32(bytes + component * 4, fp32_bits(ray.origin[component]));
    store_u32(bytes + 12 + component * 4,
              fp32_bits(ray.direction[component]));
  }
  store_u32(bytes + 24, fp32_bits(ray.t_min));
  store_u32(bytes + 28, fp32_bits(ray.t_max));
}

static void decode_ray(const uint8_t *bytes, ray_v1 *ray) {
  for (unsigned component = 0; component < 3; ++component) {
    ray->origin[component] =
        fp32_value(load_u32(bytes + component * 4));
    ray->direction[component] =
        fp32_value(load_u32(bytes + 12 + component * 4));
  }
  ray->t_min = fp32_value(load_u32(bytes + 24));
  ray->t_max = fp32_value(load_u32(bytes + 28));
}

static void encode_as_context(uint8_t *bytes,
                              const as_context_v1 &context,
                              bool include_cull_mask) {
  store_u64(bytes + 0, context.as_object_id);
  store_u64(bytes + 8, context.device_base);
  store_u64(bytes + 16, context.device_range_bytes);
  store_u32(bytes + 24, context.as_object_generation);
  bytes[28] = context.as_type;
  bytes[29] = include_cull_mask ? context.cull_mask : 0;
  bytes[30] = 0;
  bytes[31] = 0;
}

static void decode_as_context(const uint8_t *bytes,
                              bool include_cull_mask,
                              as_context_v1 *context) {
  context->as_object_id = load_u64(bytes + 0);
  context->device_base = load_u64(bytes + 8);
  context->device_range_bytes = load_u64(bytes + 16);
  context->as_object_generation = load_u32(bytes + 24);
  context->as_type = bytes[28];
  context->cull_mask = include_cull_mask ? bytes[29] : 0;
  context->reserved_zero[0] = bytes[30];
  context->reserved_zero[1] = bytes[31];
}

static void encode_entry(uint8_t *bytes,
                         const short_stack::entry_v0 &entry) {
  store_u64(bytes + 0, entry.payload_offset);
  store_u32(bytes + 8, entry.near_t_bits);
  store_u16(bytes + 12, entry.payload_byte_count);
  bytes[14] = entry.payload_kind;
  bytes[15] = entry.control;
}

static void decode_entry(const uint8_t *bytes,
                         short_stack::entry_v0 *entry) {
  entry->payload_offset = load_u64(bytes + 0);
  entry->near_t_bits = load_u32(bytes + 8);
  entry->payload_byte_count = load_u16(bytes + 12);
  entry->payload_kind = bytes[14];
  entry->control = bytes[15];
}

static status_kind validate_stack(const state_v1 &state,
                                  const control_tags_v1 &control) {
  if (!short_stack::validate_state(state.stack)) {
    return kStatusInvalidStack;
  }
  for (uint8_t logical = 0; logical < state.stack.stack_count;
       ++logical) {
    short_stack::entry_v0 entry = {};
    if (!short_stack::read_logical_entry(state.stack, logical, &entry)) {
      return kStatusInvalidStack;
    }
    const as_context_v1 *context = &state.active_as;
    if (short_stack::control_domain(entry.control) ==
            short_stack::kDomainTlas &&
        state.active_as.as_type == kAsTypeBlas) {
      if (control.parent_restore_valid == 0) {
        return kStatusInvalidStack;
      }
      context = &state.parent_restore.tlas_context;
    }
    if (!relative_range_contains(*context, entry.payload_offset,
                                 entry.payload_byte_count)) {
      return kStatusInvalidStack;
    }
  }
  return kStatusOk;
}

static status_kind validate_boundary(const state_v1 &state,
                                     const control_tags_v1 &control) {
  if (control.union_arm != kUnionArmBoundary) return kStatusOk;
  const typed_primitive::primitive_identity_policy_facts_v0 &identity =
      state.boundary.identity_and_policy;
  if (identity.instance_metadata_ref == 0 ||
      identity.geometry_type == typed_primitive::kGeometryTypeInvalid ||
      identity.geometry_type > typed_primitive::kGeometryTypeProcedural ||
      (identity.geometry_policy_flags & ~kGeometryPolicyMask) != 0 ||
      !valid_instance_policy(identity.instance_policy_flags) ||
      (identity.effective_policy_flags & ~kEffectivePolicyMask) != 0 ||
      (identity.instance_sbt_contribution & 0xff000000u) != 0) {
    return kStatusInvalidBoundary;
  }
  if (control.boundary_reason == kBoundaryReasonAnyHit) {
    const typed_primitive::triangle_hit_facts_v0 &hit =
        state.boundary.triangle_hit;
    if (identity.geometry_type !=
            typed_primitive::kGeometryTypeTriangle ||
        !bytes_are_zero(hit.reserved_zero,
                        sizeof(hit.reserved_zero)) ||
        !std::isfinite(fp32_value(hit.hit_t_bits)) ||
        (hit.hit_kind != typed_primitive::kHitKindFrontFacing &&
         hit.hit_kind != typed_primitive::kHitKindBackFacing)) {
      return kStatusInvalidBoundary;
    }
  } else {
    if (identity.geometry_type !=
            typed_primitive::kGeometryTypeProcedural ||
        state.boundary.intersection.reserved_zero != 0 ||
        !std::isfinite(fp32_value(
            state.boundary.intersection.boundary_ray_tmax_bits))) {
      return kStatusInvalidBoundary;
    }
  }
  const typed_primitive::primitive_resume_data_v0 &resume =
      state.boundary.primitive_resume;
  if (resume.remaining_slot_mask == 0) {
    if (resume.leaf_fetch_address != 0) {
      return kStatusInvalidBoundary;
    }
  } else if (resume.leaf_fetch_address == 0 ||
             (resume.leaf_fetch_address & uint64_t{0x3f}) != 0 ||
             !range_contains(state.active_as,
                             resume.leaf_fetch_address, 64)) {
    return kStatusInvalidBoundary;
  }
  return kStatusOk;
}

static status_kind validate_transition(
    const state_v1 &state, const control_tags_v1 &control) {
  if (control.union_arm != kUnionArmTransition) return kStatusOk;
  status_kind status =
      validate_as_context(state.transition.decode_context, false);
  if (status != kStatusOk) return kStatusInvalidTransition;
  const short_stack::entry_kind selected_kind =
      short_stack::control_kind(state.transition.selected.control);
  if (!short_stack::validate_entry(state.transition.selected) ||
      (selected_kind != short_stack::kEntryDirectTarget &&
       selected_kind != short_stack::kEntrySameNodeReplay) ||
      short_stack::control_domain(state.transition.selected.control) !=
          (state.transition.decode_context.as_type == kAsTypeTlas
               ? short_stack::kDomainTlas
               : short_stack::kDomainBlas) ||
      !relative_range_contains(
          state.transition.decode_context,
          state.transition.selected.payload_offset,
          state.transition.selected.payload_byte_count)) {
    return kStatusInvalidTransition;
  }
  if (control.pending_parent_resume_valid != 0) {
    const short_stack::entry_v0 &pending =
        state.transition.pending_parent_resume;
    if (!short_stack::validate_entry(pending) ||
        short_stack::control_kind(pending.control) !=
            short_stack::kEntryParentResume ||
        short_stack::control_domain(pending.control) !=
            (state.active_as.as_type == kAsTypeTlas
                 ? short_stack::kDomainTlas
                 : short_stack::kDomainBlas) ||
        !relative_range_contains(
            state.active_as, pending.payload_offset,
            pending.payload_byte_count)) {
      return kStatusInvalidTransition;
    }
  }
  return kStatusOk;
}

static status_kind validate_parent_restore(
    const state_v1 &state, const control_tags_v1 &control) {
  if (control.parent_restore_valid == 0) return kStatusOk;
  status_kind status = validate_ray(state.parent_restore.ray);
  if (status != kStatusOk) return kStatusInvalidParentRestore;
  status =
      validate_as_context(state.parent_restore.tlas_context, true);
  if (status != kStatusOk ||
      state.parent_restore.tlas_context.as_type != kAsTypeTlas) {
    return kStatusInvalidParentRestore;
  }
  return kStatusOk;
}

static status_kind validate_state(const state_v1 &state,
                                  const control_tags_v1 &control) {
  status_kind status = validate_control(control);
  if (status != kStatusOk) return status;
  status = validate_ray(state.ray);
  if (status != kStatusOk) return status;
  status = validate_as_context(state.active_as, true);
  if (status != kStatusOk) return status;
  if (state.ray_flags & ~typed_primitive::kSupportedRayFlagMask) {
    return kStatusInvalidReservedBits;
  }
  if (state.tlas_build_generation == 0 ||
      state.active_as.as_type !=
          (state.stack.active_domain == short_stack::kDomainTlas
               ? kAsTypeTlas : kAsTypeBlas) ||
      (state.stack.active_domain == short_stack::kDomainBlas &&
       state.blas_build_generation == 0)) {
    return kStatusInvalidStack;
  }
  status = validate_hit(state.committed_hit);
  if (status != kStatusOk) return status;
  status = validate_current_instance(
      state.current_instance, state.stack.active_domain);
  if (status != kStatusOk) return status;
  status = validate_parent_restore(state, control);
  if (status != kStatusOk) return status;
  status = validate_stack(state, control);
  if (status != kStatusOk) return status;
  status = validate_boundary(state, control);
  if (status != kStatusOk) return status;
  return validate_transition(state, control);
}

static void encode_committed_hit(
    uint8_t *bytes,
    const typed_stack::committed_hit_projection_v0 &hit) {
  store_u32(bytes + 0, pack_hit_control(hit));
  store_u32(bytes + 4, fp32_bits(hit.hit_t));
  store_u32(bytes + 8, hit.policy_flags);
  store_u64(bytes + 12, hit.instance_metadata_ref);
  store_u32(bytes + 20, hit.primitive_index);
  store_u32(bytes + 24, hit.geometry_index);
  store_u32(bytes + 28, hit.instance_index);
  store_u32(bytes + 32, hit.instance_custom_index);
  store_u32(bytes + 36, hit.instance_sbt_contribution);
  for (unsigned word = 0; word < 4; ++word) {
    store_u32(bytes + 40 + word * 4, hit.inline_attributes[word]);
  }
}

static status_kind decode_committed_hit(
    const uint8_t *bytes,
    typed_stack::committed_hit_projection_v0 *hit) {
  status_kind status = unpack_hit_control(load_u32(bytes), hit);
  if (status != kStatusOk) return status;
  hit->hit_t = fp32_value(load_u32(bytes + 4));
  hit->policy_flags = load_u32(bytes + 8);
  hit->instance_metadata_ref = load_u64(bytes + 12);
  hit->primitive_index = load_u32(bytes + 20);
  hit->geometry_index = load_u32(bytes + 24);
  hit->instance_index = load_u32(bytes + 28);
  hit->instance_custom_index = load_u32(bytes + 32);
  hit->instance_sbt_contribution = load_u32(bytes + 36);
  for (unsigned word = 0; word < 4; ++word) {
    hit->inline_attributes[word] =
        load_u32(bytes + 40 + word * 4);
  }
  return kStatusOk;
}

static void encode_current_instance(
    uint8_t *bytes, const current_instance_v1 &instance) {
  store_u64(bytes + 0, instance.instance_metadata_ref);
  store_u32(bytes + 8, instance.instance_index);
  store_u32(bytes + 12, instance.instance_custom_index);
  store_u32(bytes + 16,
            instance.instance_sbt_contribution |
            static_cast<uint32_t>(
                instance.instance_policy_flags) << 24);
}

static void decode_current_instance(
    const uint8_t *bytes, current_instance_v1 *instance) {
  instance->instance_metadata_ref = load_u64(bytes + 0);
  instance->instance_index = load_u32(bytes + 8);
  instance->instance_custom_index = load_u32(bytes + 12);
  const uint32_t packed = load_u32(bytes + 16);
  instance->instance_sbt_contribution = packed & 0x00ffffffu;
  instance->instance_policy_flags =
      static_cast<uint8_t>(packed >> 24);
}

static void encode_stack(const short_stack::state_v0 &stack,
                         uint8_t *bytes) {
  bytes[kStackMetadataOffset + 0] = stack.stack_count;
  bytes[kStackMetadataOffset + 1] = stack.stack_top_ptr;
  bytes[kStackMetadataOffset + 2] = stack.lost;
  bytes[kStackMetadataOffset + 3] = stack.active_domain;
  for (uint8_t logical = 0; logical < stack.stack_count; ++logical) {
    const uint8_t physical = static_cast<uint8_t>(
        (stack.stack_top_ptr + logical) %
        short_stack::kLogicalCapacity);
    encode_entry(bytes + kStackEntriesOffset +
                     physical * sizeof(short_stack::entry_v0),
                 stack.entries[physical]);
  }
}

static status_kind decode_stack(const uint8_t *bytes,
                                short_stack::state_v0 *stack) {
  stack->stack_count = bytes[kStackMetadataOffset + 0];
  stack->stack_top_ptr = bytes[kStackMetadataOffset + 1];
  stack->lost = bytes[kStackMetadataOffset + 2];
  stack->active_domain = bytes[kStackMetadataOffset + 3];
  if (!bytes_are_zero(bytes + kStackMetadataOffset + 4, 4) ||
      stack->stack_count > short_stack::kLogicalCapacity ||
      (stack->stack_count == 0 && stack->stack_top_ptr != 0) ||
      stack->stack_top_ptr >= short_stack::kLogicalCapacity ||
      stack->lost > 1 ||
      stack->active_domain > short_stack::kDomainBlas) {
    return kStatusInvalidStack;
  }
  stack->cross_as =
      stack->active_domain == short_stack::kDomainBlas ? 1 : 0;
  for (uint8_t logical = 0; logical < stack->stack_count; ++logical) {
    const uint8_t physical = static_cast<uint8_t>(
        (stack->stack_top_ptr + logical) %
        short_stack::kLogicalCapacity);
    decode_entry(bytes + kStackEntriesOffset +
                     physical * sizeof(short_stack::entry_v0),
                 &stack->entries[physical]);
  }
  return kStatusOk;
}

static void encode_boundary(uint8_t *bytes,
                            const boundary_state_v1 &boundary,
                            uint8_t reason) {
  const typed_primitive::primitive_identity_policy_facts_v0 &identity =
      boundary.identity_and_policy;
  store_u64(bytes + 0, identity.instance_metadata_ref);
  store_u32(bytes + 8, identity.primitive_index);
  store_u32(bytes + 12, identity.geometry_index);
  store_u32(bytes + 16, identity.instance_index);
  store_u32(bytes + 20, identity.instance_custom_index);
  store_u32(bytes + 24, identity.instance_sbt_contribution);
  bytes[28] = identity.geometry_type;
  bytes[29] = identity.geometry_policy_flags;
  bytes[30] = identity.instance_policy_flags;
  bytes[31] = identity.effective_policy_flags;
  if (reason == kBoundaryReasonAnyHit) {
    store_u32(bytes + 32, boundary.triangle_hit.hit_t_bits);
    store_u32(bytes + 36,
              boundary.triangle_hit.bary_vertex1_bits);
    store_u32(bytes + 40,
              boundary.triangle_hit.bary_vertex2_bits);
    bytes[44] = boundary.triangle_hit.hit_kind;
  } else {
    store_u32(bytes + 32,
              boundary.intersection.boundary_ray_tmax_bits);
  }
  store_u64(bytes + 48,
            boundary.primitive_resume.leaf_fetch_address);
  store_u64(bytes + 56,
            boundary.primitive_resume.remaining_slot_mask);
}

static void decode_boundary(const uint8_t *bytes,
                            uint8_t reason,
                            boundary_state_v1 *boundary) {
  typed_primitive::primitive_identity_policy_facts_v0 &identity =
      boundary->identity_and_policy;
  identity.instance_metadata_ref = load_u64(bytes + 0);
  identity.primitive_index = load_u32(bytes + 8);
  identity.geometry_index = load_u32(bytes + 12);
  identity.instance_index = load_u32(bytes + 16);
  identity.instance_custom_index = load_u32(bytes + 20);
  identity.instance_sbt_contribution = load_u32(bytes + 24);
  identity.geometry_type = bytes[28];
  identity.geometry_policy_flags = bytes[29];
  identity.instance_policy_flags = bytes[30];
  identity.effective_policy_flags = bytes[31];
  if (reason == kBoundaryReasonAnyHit) {
    boundary->triangle_hit.hit_t_bits = load_u32(bytes + 32);
    boundary->triangle_hit.bary_vertex1_bits =
        load_u32(bytes + 36);
    boundary->triangle_hit.bary_vertex2_bits =
        load_u32(bytes + 40);
    boundary->triangle_hit.hit_kind = bytes[44];
    std::memcpy(boundary->triangle_hit.reserved_zero,
                bytes + 45, 3);
  } else {
    boundary->intersection.boundary_ray_tmax_bits =
        load_u32(bytes + 32);
    boundary->intersection.reserved_zero = load_u32(bytes + 36);
    if (!bytes_are_zero(bytes + 40, 8)) {
      boundary->intersection.reserved_zero = 1;
    }
  }
  boundary->primitive_resume.leaf_fetch_address =
      load_u64(bytes + 48);
  boundary->primitive_resume.remaining_slot_mask =
      load_u64(bytes + 56);
}

static void encode_transition(uint8_t *bytes,
                              const transition_state_v1 &transition,
                              bool pending_valid) {
  encode_entry(bytes, transition.selected);
  encode_as_context(bytes + 16, transition.decode_context, false);
  if (pending_valid) {
    encode_entry(bytes + 48, transition.pending_parent_resume);
  }
}

static status_kind decode_transition(
    const uint8_t *bytes, bool pending_valid,
    transition_state_v1 *transition) {
  if (!bytes_are_zero(bytes + 45, 3)) {
    return kStatusInvalidReservedBits;
  }
  decode_entry(bytes, &transition->selected);
  decode_as_context(bytes + 16, false,
                    &transition->decode_context);
  if (pending_valid) {
    decode_entry(bytes + 48,
                 &transition->pending_parent_resume);
  }
  return kStatusOk;
}

static void encode_parent_restore(
    uint8_t *bytes, const parent_restore_state_v1 &parent) {
  encode_ray(bytes, parent.ray);
  encode_as_context(bytes + 32, parent.tlas_context, true);
}

static void decode_parent_restore(
    const uint8_t *bytes, parent_restore_state_v1 *parent) {
  decode_ray(bytes, &parent->ray);
  decode_as_context(bytes + 32, true, &parent->tlas_context);
}

}  // namespace

status_kind encode_image(uint32_t private_layout_profile_id,
                         uint32_t bvh_format_profile_id,
                         const state_v1 &state,
                         const control_tags_v1 &control,
                         image_v1 *image) {
  if (image == NULL) return kStatusInvalidArgument;
  status_kind status =
      validate_profiles(private_layout_profile_id,
                        bvh_format_profile_id);
  if (status != kStatusOk) return status;
  status = validate_state(state, control);
  if (status != kStatusOk) return status;

  std::memset(image->bytes, 0, sizeof(image->bytes));
  encode_ray(image->bytes + kMutableRayOffset, state.ray);
  encode_as_context(image->bytes + kActiveAsContextOffset,
                    state.active_as, true);
  encode_committed_hit(image->bytes + kCommittedHitOffset,
                       state.committed_hit);
  encode_current_instance(
      image->bytes + kCurrentInstanceRefOffset,
      state.current_instance);
  store_u32(image->bytes + 0x08c, state.ray_flags);
  store_u32(image->bytes + 0x090,
            state.tlas_build_generation);
  store_u32(image->bytes + 0x094,
            state.blas_build_generation);
  encode_stack(state.stack, image->bytes);
  if (control.union_arm == kUnionArmBoundary) {
    encode_boundary(image->bytes + kBoundaryTransitionOffset,
                    state.boundary, control.boundary_reason);
  } else if (control.union_arm == kUnionArmTransition) {
    encode_transition(image->bytes + kBoundaryTransitionOffset,
                      state.transition,
                      control.pending_parent_resume_valid != 0);
  }
  if (control.parent_restore_valid != 0) {
    encode_parent_restore(image->bytes + kParentRestoreOffset,
                          state.parent_restore);
  }
  return kStatusOk;
}

status_kind decode_image(uint32_t private_layout_profile_id,
                         uint32_t bvh_format_profile_id,
                         const image_v1 &image,
                         const control_tags_v1 &control,
                         state_v1 *state) {
  if (state == NULL) return kStatusInvalidArgument;
  status_kind status =
      validate_profiles(private_layout_profile_id,
                        bvh_format_profile_id);
  if (status != kStatusOk) return status;
  status = validate_control(control);
  if (status != kStatusOk) return status;

  state_v1 decoded = {};
  decode_ray(image.bytes + kMutableRayOffset, &decoded.ray);
  decode_as_context(image.bytes + kActiveAsContextOffset, true,
                    &decoded.active_as);
  status = decode_committed_hit(
      image.bytes + kCommittedHitOffset, &decoded.committed_hit);
  if (status != kStatusOk) return status;
  decode_current_instance(
      image.bytes + kCurrentInstanceRefOffset,
      &decoded.current_instance);
  decoded.ray_flags = load_u32(image.bytes + 0x08c);
  decoded.tlas_build_generation = load_u32(image.bytes + 0x090);
  decoded.blas_build_generation = load_u32(image.bytes + 0x094);
  if (!bytes_are_zero(image.bytes + 0x09c, 4)) {
    return kStatusInvalidReservedBits;
  }
  status = decode_stack(image.bytes, &decoded.stack);
  if (status != kStatusOk) return status;
  if (control.union_arm == kUnionArmBoundary) {
    decode_boundary(image.bytes + kBoundaryTransitionOffset,
                    control.boundary_reason, &decoded.boundary);
  } else if (control.union_arm == kUnionArmTransition) {
    status = decode_transition(
        image.bytes + kBoundaryTransitionOffset,
        control.pending_parent_resume_valid != 0,
        &decoded.transition);
    if (status != kStatusOk) return status;
  }
  if (control.parent_restore_valid != 0) {
    decode_parent_restore(image.bytes + kParentRestoreOffset,
                          &decoded.parent_restore);
  }
  status = validate_state(decoded, control);
  if (status != kStatusOk) return status;
  *state = decoded;
  return kStatusOk;
}

status_kind initialize_new_launch_image(
    uint32_t private_layout_profile_id,
    uint32_t bvh_format_profile_id,
    const launch_input_v1 &input,
    image_v1 *image,
    sparse_write_plan_v1 *plan) {
  if (image == NULL || plan == NULL) return kStatusInvalidArgument;
  state_v1 initial = {};
  initial.ray = input.ray;
  initial.active_as = input.tlas_context;
  initial.ray_flags = input.ray_flags;
  initial.tlas_build_generation = input.tlas_build_generation;
  initial.stack.active_domain = short_stack::kDomainTlas;
  control_tags_v1 control = {};
  image_v1 staging = {};
  status_kind status =
      encode_image(private_layout_profile_id, bvh_format_profile_id,
                   initial, control, &staging);
  if (status != kStatusOk) return status;

  sparse_write_plan_v1 result = {};
  result.write_count = kLaunchWriteCount;
  for (uint8_t chunk = 0; chunk < kLaunchWriteCount; ++chunk) {
    const uint16_t offset =
        static_cast<uint16_t>(chunk * kChunkBytes);
    result.writes[chunk].slot_byte_offset = offset;
    result.writes[chunk].byte_count = kChunkBytes;
    result.writes[chunk].byte_mask = kFullChunkMask;
    std::memcpy(result.writes[chunk].payload,
                staging.bytes + offset, kChunkBytes);
    std::memcpy(image->bytes + offset,
                staging.bytes + offset, kChunkBytes);
  }
  *plan = result;
  return kStatusOk;
}

status_kind encode_stack_sparse_projection(
    const short_stack::state_v0 &stack,
    stack_sparse_projection_v1 *projection) {
  if (projection == NULL) return kStatusInvalidArgument;
  *projection = stack_sparse_projection_v1();
  if (!short_stack::validate_state(stack)) {
    return kStatusInvalidStack;
  }
  projection->metadata[0] = stack.stack_count;
  projection->metadata[1] = stack.stack_top_ptr;
  projection->metadata[2] = stack.lost;
  projection->metadata[3] = stack.active_domain;
  for (uint8_t logical = 0; logical < stack.stack_count; ++logical) {
    const uint8_t physical = static_cast<uint8_t>(
        (stack.stack_top_ptr + logical) %
        short_stack::kLogicalCapacity);
    encode_entry(
        projection->entries +
            physical * sizeof(short_stack::entry_v0),
        stack.entries[physical]);
  }
  return kStatusOk;
}

status_kind encode_committed_hit_sparse_projection(
    const typed_stack::committed_hit_projection_v0 &hit,
    uint8_t payload[kCommittedHitProjectionBytes]) {
  if (payload == NULL) return kStatusInvalidArgument;
  std::memset(payload, 0, kCommittedHitProjectionBytes);
  const status_kind status = validate_hit(hit);
  if (status != kStatusOk) return status;
  encode_committed_hit(payload, hit);
  return kStatusOk;
}

status_kind encode_boundary_sparse_projection(
    const boundary_state_v1 &boundary, uint8_t reason,
    const as_context_v1 &active_as,
    uint8_t payload[kBoundaryProjectionBytes]) {
  if (payload == NULL ||
      (reason != kBoundaryReasonAnyHit &&
       reason != kBoundaryReasonProceduralIntersection)) {
    return kStatusInvalidArgument;
  }
  std::memset(payload, 0, kBoundaryProjectionBytes);
  state_v1 validation = {};
  validation.active_as = active_as;
  validation.boundary = boundary;
  control_tags_v1 control = {};
  control.union_arm = kUnionArmBoundary;
  control.boundary_reason = reason;
  status_kind status = validate_as_context(active_as, true);
  if (status != kStatusOk) return status;
  status = validate_boundary(validation, control);
  if (status != kStatusOk) return status;
  encode_boundary(payload, boundary, reason);
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusUnsupportedPrivateLayout:
      return "unsupported_private_layout";
    case kStatusUnsupportedBvhProfile:
      return "unsupported_bvh_profile";
    case kStatusInvalidControl:
      return "invalid_control";
    case kStatusInvalidRay:
      return "invalid_ray";
    case kStatusInvalidAsContext:
      return "invalid_as_context";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusInvalidCommittedHit:
      return "invalid_committed_hit";
    case kStatusInvalidCurrentInstance:
      return "invalid_current_instance";
    case kStatusInvalidStack:
      return "invalid_stack";
    case kStatusInvalidBoundary:
      return "invalid_boundary";
    case kStatusInvalidTransition:
      return "invalid_transition";
    case kStatusInvalidParentRestore:
      return "invalid_parent_restore";
    case kStatusInvalidReservedBits:
      return "invalid_reserved_bits";
  }
  return "unknown";
}

}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore
