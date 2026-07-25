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
      fragment.field_kind == private_frontier::kFieldCurrentInstance;
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
        prepared.write_fragments[index];
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
  }
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
