#include "rtcore_v04_stack_result_semantic_applier.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace stack_semantic {
namespace {

status_kind translate_layout_status(private_frontier::status_kind status) {
  switch (status) {
    case private_frontier::kStatusOk:
      return kStatusOk;
    case private_frontier::kStatusInvalidArgument:
      return kStatusInvalidArgument;
    case private_frontier::kStatusUnsupportedProfile:
      return kStatusUnsupportedProfile;
    case private_frontier::kStatusInvalidOwner:
      return kStatusInvalidOwner;
    case private_frontier::kStatusOwnerMismatch:
      return kStatusOwnerMismatch;
    case private_frontier::kStatusInvalidRegion:
      return kStatusInvalidRegion;
    case private_frontier::kStatusAddressOverflow:
      return kStatusAddressOverflow;
    case private_frontier::kStatusInvalidMetadata:
    case private_frontier::kStatusInvalidEntryIndex:
      return kStatusInvalidFrontierState;
    case private_frontier::kStatusInvalidDelta:
      return kStatusInvalidDelta;
    case private_frontier::kStatusPlanCapacityExceeded:
      return kStatusPlanCapacityExceeded;
  }
  return kStatusInvalidArgument;
}

unsigned count_mask_bits(uint32_t mask) {
  unsigned count = 0;
  while (mask != 0) {
    count += mask & 1u;
    mask >>= 1;
  }
  return count;
}

bool fragment_is_valid(const private_write_fragment_v0 &fragment) {
  if ((fragment.aligned_32b_address %
       private_frontier::kSharedAccessChunkBytes) != 0 ||
      fragment.byte_mask == 0 ||
      fragment.byte_count == 0 ||
      fragment.byte_count > private_frontier::kSharedAccessChunkBytes ||
      count_mask_bits(fragment.byte_mask) != fragment.byte_count ||
      (fragment.field_kind != private_frontier::kFieldFrontierMetadata &&
       fragment.field_kind != private_frontier::kFieldFrontierEntry)) {
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

bool plan_structure_is_valid(
    const append_commit_plan_v0 &plan,
    const private_frontier::owner_binding_v0 &expected_owner,
    uint32_t expected_operation_seq,
    const typed_node::selected_child_fetch_work_item_v0
        &expected_selected_fetch) {
  if (plan.valid != 1 ||
      expected_operation_seq == 0 ||
      plan.operation_seq != expected_operation_seq ||
      !private_frontier::owners_equal(plan.owner, expected_owner) ||
      plan.route_kind != kRouteStackPushedAndSelected ||
      plan.target_selector_kind !=
          kTargetSelectorFetchByExpectedPayloadKind ||
      plan.required_output_mask != kStackPushedRequiredOutputMask ||
      plan.allowed_output_mask != kStackPushedRequiredOutputMask ||
      plan.forward_mask != kStackPushedForwardMask ||
      plan.persist_mask != kStackPushedPersistMask ||
      plan.fallback_spill_mask != kStackPushedForwardMask ||
      plan.fallback_spill_kind !=
          kFallbackSpillStackToMemory ||
      plan.write_fragment_count > kMaxPrivateWriteFragments ||
      plan.required_ack_count != plan.write_fragment_count ||
      std::memcmp(&plan.selected_fetch, &expected_selected_fetch,
                  sizeof(plan.selected_fetch)) != 0) {
    return false;
  }
  for (unsigned byte = 0; byte < sizeof(plan.reserved_zero); ++byte) {
    if (plan.reserved_zero[byte] != 0) return false;
  }
  for (unsigned index = 0; index < plan.write_fragment_count; ++index) {
    if (!fragment_is_valid(plan.write_fragments[index])) return false;
  }
  const private_write_fragment_v0 zero = {};
  for (unsigned index = plan.write_fragment_count;
       index < kMaxPrivateWriteFragments; ++index) {
    if (std::memcmp(&plan.write_fragments[index], &zero,
                    sizeof(zero)) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

status_kind prepare_stack_pushed_and_selected(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_result_v0 &result,
    append_commit_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  if (operation_seq == 0) return kStatusInvalidOperationIdentity;
  if (!typed_stack::validate_push_result(result) ||
      result.result_kind != typed_stack::kStackPushedAndSelected ||
      result.output_valid_mask != kStackPushedRequiredOutputMask) {
    return kStatusInvalidTypedResult;
  }

  private_frontier::shadow_slot_v0 updated_slot = canonical_slot;
  private_frontier::access_plan_v0 access_plan = {};
  const private_frontier::status_kind layout_status =
      private_frontier::apply_append_delta(
          &updated_slot, owner, region, result.frontier_delta,
          &access_plan);
  if (layout_status != private_frontier::kStatusOk) {
    return translate_layout_status(layout_status);
  }
  const bool slot_changed =
      std::memcmp(&updated_slot, &canonical_slot,
                  sizeof(updated_slot)) != 0;
  if (!slot_changed &&
      result.frontier_delta.write_count == 0) {
    access_plan.access_count = 0;
    std::memset(access_plan.accesses, 0,
                sizeof(access_plan.accesses));
  }
  if (!private_frontier::owners_equal(access_plan.owner, owner) ||
      (slot_changed && access_plan.access_count == 0) ||
      access_plan.access_count > kMaxPrivateWriteFragments) {
    return kStatusPlanCapacityExceeded;
  }

  append_commit_plan_v0 prepared = {};
  prepared.owner = owner;
  prepared.operation_seq = operation_seq;
  prepared.route_kind = kRouteStackPushedAndSelected;
  prepared.target_selector_kind =
      kTargetSelectorFetchByExpectedPayloadKind;
  prepared.required_output_mask = kStackPushedRequiredOutputMask;
  prepared.allowed_output_mask = kStackPushedRequiredOutputMask;
  prepared.forward_mask = kStackPushedForwardMask;
  prepared.persist_mask = kStackPushedPersistMask;
  prepared.fallback_spill_mask = kStackPushedForwardMask;
  prepared.fallback_spill_kind =
      kFallbackSpillStackToMemory;
  prepared.write_fragment_count = access_plan.access_count;
  prepared.required_ack_count = access_plan.access_count;
  prepared.selected_fetch = result.selected_fetch;

  for (unsigned index = 0; index < access_plan.access_count; ++index) {
    const private_frontier::shared_chunk_access_v0 &access =
        access_plan.accesses[index];
    if (access.access_kind != private_frontier::kAccessWrite ||
        access.slot_byte_offset >= private_frontier::kPrivateDataSlotBytes) {
      return kStatusInvalidWriteFragment;
    }
    const uint32_t aligned_slot_offset =
        access.slot_byte_offset -
        (access.slot_byte_offset %
         private_frontier::kSharedAccessChunkBytes);
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
  if (!plan_structure_is_valid(
          prepared, owner, operation_seq, result.selected_fetch)) {
    return kStatusInvalidWriteFragment;
  }
  *plan = prepared;
  return kStatusOk;
}

bool validate_append_commit_plan(
    const append_commit_plan_v0 &plan,
    const private_frontier::owner_binding_v0 &expected_owner,
    uint32_t expected_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_result_v0 &result) {
  append_commit_plan_v0 expected = {};
  const status_kind status = prepare_stack_pushed_and_selected(
      expected_owner, expected_operation_seq, region,
      canonical_slot, result, &expected);
  return status == kStatusOk &&
         std::memcmp(&plan, &expected, sizeof(plan)) == 0;
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
    case kStatusInvalidFrontierState:
      return "invalid_frontier_state";
    case kStatusInvalidDelta:
      return "invalid_delta";
    case kStatusPlanCapacityExceeded:
      return "plan_capacity_exceeded";
    case kStatusInvalidWriteFragment:
      return "invalid_write_fragment";
  }
  return "unknown";
}

}  // namespace stack_semantic
}  // namespace v04
}  // namespace rtcore
