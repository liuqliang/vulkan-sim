#ifndef RTCORE_V04_STACK_RESULT_SEMANTIC_APPLIER_H
#define RTCORE_V04_STACK_RESULT_SEMANTIC_APPLIER_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"

namespace rtcore {
namespace v04 {
namespace stack_semantic {

static const uint8_t kStackPushedRequiredOutputMask =
    static_cast<uint8_t>(typed_stack::kFrontierDeltaValid |
                         typed_stack::kSelectedFetchValid);
static const uint8_t kStackPushedForwardMask =
    typed_stack::kSelectedFetchValid;
static const uint8_t kStackPushedPersistMask =
    typed_stack::kFrontierDeltaValid;
static const unsigned kMaxAppendEntryFragments =
    (private_frontier::kSharedAccessChunkBytes - 1 +
     typed_stack::kMaxRemainderChildren *
         private_frontier::kFrontierEntryBytes +
     private_frontier::kSharedAccessChunkBytes - 1) /
    private_frontier::kSharedAccessChunkBytes;
static const unsigned kFrontierMetadataFragments =
    (private_frontier::kFrontierMetadataOffset %
         private_frontier::kSharedAccessChunkBytes +
     private_frontier::kFrontierMetadataBytes +
     private_frontier::kSharedAccessChunkBytes - 1) /
    private_frontier::kSharedAccessChunkBytes;
static const uint8_t kMaxPrivateWriteFragments =
    kMaxAppendEntryFragments + kFrontierMetadataFragments;
static_assert(kMaxPrivateWriteFragments == 6,
              "Stack append fragment bound changed");
static_assert(kMaxPrivateWriteFragments <=
                  private_frontier::kMaxAccessChunks,
              "Stack append fragments exceed layout planner capacity");

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOperationIdentity,
  kStatusInvalidTypedResult,
  kStatusUnsupportedProfile,
  kStatusInvalidOwner,
  kStatusOwnerMismatch,
  kStatusInvalidRegion,
  kStatusAddressOverflow,
  kStatusInvalidFrontierState,
  kStatusInvalidDelta,
  kStatusPlanCapacityExceeded,
  kStatusInvalidWriteFragment,
};

enum route_kind : uint8_t {
  kRouteInvalid = 0,
  kRouteStackPushedAndSelected = 1,
};

enum target_selector_kind : uint8_t {
  kTargetSelectorInvalid = 0,
  kTargetSelectorFetchByExpectedPayloadKind = 1,
};

enum fallback_spill_kind : uint8_t {
  kFallbackSpillInvalid = 0,
  kFallbackSpillStackToMemory = 1,
};

struct private_write_fragment_v0 {
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
  uint16_t slot_byte_offset;
  uint8_t byte_count;
  uint8_t field_kind;
  uint8_t payload[private_frontier::kSharedAccessChunkBytes];
};

struct append_commit_plan_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint8_t valid;
  uint8_t route_kind;
  uint8_t target_selector_kind;
  uint8_t required_output_mask;
  uint8_t allowed_output_mask;
  uint8_t forward_mask;
  uint8_t persist_mask;
  uint8_t fallback_spill_mask;
  uint8_t fallback_spill_kind;
  uint8_t write_fragment_count;
  uint8_t required_ack_count;
  uint8_t reserved_zero[1];
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  private_write_fragment_v0
      write_fragments[kMaxPrivateWriteFragments];
};

static_assert(sizeof(private_write_fragment_v0) == 48,
              "private write fragment must remain 48 bytes");
static_assert(offsetof(private_write_fragment_v0, payload) == 16,
              "private write payload offset changed");

bool validate_private_write_fragment(
    const private_write_fragment_v0 &fragment);

status_kind prepare_stack_pushed_and_selected(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_result_v0 &result,
    append_commit_plan_v0 *plan);

bool validate_append_commit_plan(
    const append_commit_plan_v0 &plan,
    const private_frontier::owner_binding_v0 &expected_owner,
    uint32_t expected_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_result_v0 &result);

const char *status_name(status_kind status);

}  // namespace stack_semantic
}  // namespace v04
}  // namespace rtcore

#endif
