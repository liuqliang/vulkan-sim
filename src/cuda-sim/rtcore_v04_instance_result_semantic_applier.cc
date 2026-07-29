#include "rtcore_v04_instance_result_semantic_applier.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace instance_semantic {
namespace {

unsigned count_mask_bits(uint32_t mask) {
  unsigned count = 0;
  while (mask != 0) {
    count += mask & 1u;
    mask >>= 1;
  }
  return count;
}

bool fragment_is_valid(const private_write_fragment_v0 &fragment) {
  const bool field_valid =
      fragment.field_kind == private_frontier::kFieldMutableRayState ||
      fragment.field_kind == private_frontier::kFieldAsDecodeContext ||
      fragment.field_kind == private_frontier::kFieldCurrentInstance ||
      fragment.field_kind == private_frontier::kFieldFrontierMetadata ||
      fragment.field_kind == private_frontier::kFieldParentFrame;
  if (!field_valid ||
      (fragment.aligned_32b_address %
       private_frontier::kSharedAccessChunkBytes) != 0 ||
      fragment.byte_mask == 0 || fragment.byte_count == 0 ||
      fragment.byte_count > private_frontier::kSharedAccessChunkBytes ||
      count_mask_bits(fragment.byte_mask) != fragment.byte_count) {
    return false;
  }
  for (unsigned byte = 0;
       byte < private_frontier::kSharedAccessChunkBytes; ++byte) {
    if ((fragment.byte_mask & (uint32_t{1} << byte)) == 0 &&
        fragment.payload[byte] != 0) {
      return false;
    }
  }
  return true;
}

status_kind append_fragments(
    const private_frontier::access_plan_v0 &write_plan,
    const private_frontier::shadow_slot_v0 &updated_slot,
    private_write_fragment_v0 *fragments, uint8_t capacity,
    uint8_t *fragment_count) {
  if (fragments == NULL || fragment_count == NULL ||
      write_plan.access_count > capacity - *fragment_count) {
    return kStatusInvalidWriteFragment;
  }
  for (unsigned index = 0; index < write_plan.access_count; ++index) {
    const private_frontier::shared_chunk_access_v0 &access =
        write_plan.accesses[index];
    if (access.access_kind != private_frontier::kAccessWrite ||
        access.slot_byte_offset >=
            private_frontier::kPrivateDataSlotBytes) {
      return kStatusInvalidWriteFragment;
    }
    const uint32_t aligned_slot_offset =
        access.slot_byte_offset -
        access.slot_byte_offset %
            private_frontier::kSharedAccessChunkBytes;
    if (aligned_slot_offset >
        private_frontier::kPrivateDataSlotBytes -
            private_frontier::kSharedAccessChunkBytes) {
      return kStatusInvalidWriteFragment;
    }
    private_write_fragment_v0 &fragment =
        fragments[*fragment_count];
    fragment = private_write_fragment_v0();
    fragment.aligned_32b_address = access.aligned_32b_address;
    fragment.byte_mask = access.byte_mask;
    fragment.slot_byte_offset = access.slot_byte_offset;
    fragment.byte_count = access.byte_count;
    fragment.field_kind = access.field_kind;
    for (unsigned byte = 0;
         byte < private_frontier::kSharedAccessChunkBytes; ++byte) {
      if ((access.byte_mask & (uint32_t{1} << byte)) != 0) {
        fragment.payload[byte] =
            updated_slot.bytes[aligned_slot_offset + byte];
      }
    }
    if (!fragment_is_valid(fragment)) {
      return kStatusInvalidWriteFragment;
    }
    ++*fragment_count;
  }
  return kStatusOk;
}

}  // namespace

bool validate_private_write_fragment(
    const private_write_fragment_v0 &fragment) {
  return fragment_is_valid(fragment);
}

status_kind prepare_restore_parent(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_instance::restore_parent_result_v0 &result,
    restore_commit_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = restore_commit_plan_v0();
  if (operation_seq == 0) return kStatusInvalidOperationIdentity;
  if (result.status != typed_instance::kStatusOk ||
      result.result_kind != typed_instance::kInstanceParentRestored ||
      result.output_valid_mask !=
          typed_instance::kParentStateRestoredValid) {
    return kStatusInvalidTypedResult;
  }

  private_frontier::shadow_slot_v0 updated_slot = canonical_slot;
  private_frontier::access_plan_v0 write_plan = {};
  if (private_frontier::apply_parent_state_restore(
          &updated_slot, owner, region, result.restored_parent,
          &write_plan) != private_frontier::kStatusOk ||
      write_plan.access_count != kRestoreWriteFragmentCount ||
      !private_frontier::owners_equal(write_plan.owner, owner)) {
    return kStatusLayoutRejected;
  }

  restore_commit_plan_v0 prepared = {};
  prepared.owner = owner;
  prepared.operation_seq = operation_seq;
  prepared.route_kind = kRouteStackPopNext;
  prepared.write_fragment_count = write_plan.access_count;
  prepared.required_ack_count = write_plan.access_count;
  uint8_t fragment_count = 0;
  if (append_fragments(write_plan, updated_slot,
                       prepared.write_fragments,
                       kRestoreWriteFragmentCount,
                       &fragment_count) != kStatusOk ||
      fragment_count != kRestoreWriteFragmentCount) {
    return kStatusInvalidWriteFragment;
  }
  prepared.valid = 1;
  *plan = prepared;
  return kStatusOk;
}

status_kind prepare_enter(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_instance::enter_input_v0 &input,
    const typed_instance::enter_result_v0 &result,
    enter_commit_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = enter_commit_plan_v0();
  if (operation_seq == 0) return kStatusInvalidOperationIdentity;
  if (result.status != typed_instance::kStatusOk ||
      result.mask_visible > 1) {
    return kStatusInvalidTypedResult;
  }

  enter_commit_plan_v0 prepared = {};
  prepared.owner = owner;
  prepared.operation_seq = operation_seq;
  if (result.result_kind == typed_instance::kEnterResultCulled) {
    if (result.mask_visible != 0 || result.output_valid_mask != 0) {
      return kStatusInvalidTypedResult;
    }
    prepared.route_kind = kRouteStackPopNext;
    prepared.valid = 1;
    *plan = prepared;
    return kStatusOk;
  }
  if (result.result_kind != typed_instance::kEnterResultBlasRoot ||
      result.mask_visible != 1 ||
      result.output_valid_mask !=
          (typed_instance::kObjectRayValid |
           typed_instance::kInstanceProjectionValid |
           typed_instance::kRootFetchValid) ||
      result.root_fetch.expected_payload_kind !=
          typed_node::kInternalPayloadKind ||
      result.root_fetch.decode_context.as_object.as_type !=
          typed_blas::kAsTypeBlas ||
      result.root_fetch.build_generation == 0 ||
      result.root_fetch.encoded_reference == 0) {
    return kStatusInvalidTypedResult;
  }

  private_frontier::traversal_frame_projection_v0 parent = {};
  if (private_frontier::capture_parent_frame(
          canonical_slot, owner, &parent) !=
          private_frontier::kStatusOk ||
      std::memcmp(&parent.ray, &input.world_ray,
                  sizeof(parent.ray)) != 0 ||
      std::memcmp(&parent.current_decode_context,
                  &input.tlas_decode_context,
                  sizeof(parent.current_decode_context)) != 0) {
    return kStatusLayoutRejected;
  }

  private_frontier::shadow_slot_v0 updated_slot = canonical_slot;
  private_frontier::access_plan_v0 parent_plan = {};
  if (private_frontier::apply_parent_frame_push(
          &updated_slot, owner, region, parent, &parent_plan) !=
      private_frontier::kStatusOk) {
    return kStatusLayoutRejected;
  }
  private_frontier::mutable_ray_state_v0 object_ray = {};
  std::memcpy(&object_ray, &result.object_ray, sizeof(object_ray));
  private_frontier::instance_shader_projection_v0 current_instance = {};
  current_instance.instance_metadata_ref =
      result.instance_projection.instance_metadata_reference;
  current_instance.instance_index =
      result.instance_projection.instance_index;
  current_instance.instance_custom_index =
      result.instance_projection.instance_custom_index;
  current_instance.instance_sbt_contribution =
      result.instance_projection.instance_sbt_contribution;
  current_instance.instance_policy_flags =
      result.instance_projection.instance_flags;
  private_frontier::access_plan_v0 child_plan = {};
  if (private_frontier::apply_instance_enter_state(
          &updated_slot, owner, region, object_ray,
          result.root_fetch.decode_context, current_instance,
          &child_plan) != private_frontier::kStatusOk ||
      parent_plan.access_count + child_plan.access_count !=
          kEnterVisibleWriteFragmentCount) {
    return kStatusLayoutRejected;
  }

  uint8_t fragment_count = 0;
  if (append_fragments(parent_plan, updated_slot,
                       prepared.write_fragments,
                       kMaxWriteFragmentCount,
                       &fragment_count) != kStatusOk ||
      append_fragments(child_plan, updated_slot,
                       prepared.write_fragments,
                       kMaxWriteFragmentCount,
                       &fragment_count) != kStatusOk ||
      fragment_count != kEnterVisibleWriteFragmentCount) {
    return kStatusInvalidWriteFragment;
  }
  prepared.route_kind = kRouteBlasRootNode;
  prepared.root_build_generation =
      result.root_fetch.build_generation;
  prepared.write_fragment_count = fragment_count;
  prepared.required_ack_count = fragment_count;
  prepared.root_fetch.child.payload_offset =
      result.root_fetch.encoded_reference;
  std::memcpy(&prepared.root_fetch.child.near_t_bits,
              &result.object_ray.t_min,
              sizeof(prepared.root_fetch.child.near_t_bits));
  prepared.root_fetch.child.payload_byte_count = 64;
  prepared.root_fetch.child.payload_kind =
      result.root_fetch.expected_payload_kind;
  prepared.root_fetch.child.child_slot = 0;
  prepared.root_fetch.decode_context =
      result.root_fetch.decode_context;
  prepared.ray_policy.ray_flags = input.policy.ray_flags;
  prepared.ray_policy.cull_mask = input.policy.cull_mask;
  prepared.valid = 1;
  *plan = prepared;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidOperationIdentity:
      return "invalid_operation_identity";
    case kStatusInvalidTypedResult:
      return "invalid_typed_result";
    case kStatusLayoutRejected:
      return "layout_rejected";
    case kStatusInvalidWriteFragment:
      return "invalid_write_fragment";
  }
  return "unknown";
}

}  // namespace instance_semantic
}  // namespace v04
}  // namespace rtcore
