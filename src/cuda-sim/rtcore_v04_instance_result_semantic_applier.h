#ifndef RTCORE_V04_INSTANCE_RESULT_SEMANTIC_APPLIER_H
#define RTCORE_V04_INSTANCE_RESULT_SEMANTIC_APPLIER_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_typed_instance_kernel.h"

namespace rtcore {
namespace v04 {
namespace instance_semantic {

static const uint8_t kRestoreWriteFragmentCount = 6;
static const uint8_t kEnterShortStackWriteFragmentCount = 6;
static const uint8_t kEnterVisibleWriteFragmentCount = 13;
static const uint8_t kMaxWriteFragmentCount =
    kEnterVisibleWriteFragmentCount;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOperationIdentity,
  kStatusInvalidTypedResult,
  kStatusLayoutRejected,
  kStatusInvalidWriteFragment,
};

enum route_kind : uint8_t {
  kRouteInvalid = 0,
  kRouteStackPopNext = 1,
  kRouteBlasRootNode = 2,
};

struct private_write_fragment_v0 {
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
  uint16_t slot_byte_offset;
  uint8_t byte_count;
  uint8_t field_kind;
  uint8_t payload[private_frontier::kSharedAccessChunkBytes];
};

struct restore_commit_plan_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint8_t valid;
  uint8_t route_kind;
  uint8_t write_fragment_count;
  uint8_t required_ack_count;
  uint8_t reserved_zero[4];
  private_write_fragment_v0
      write_fragments[kRestoreWriteFragmentCount];
};

struct enter_commit_plan_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint32_t root_build_generation;
  uint8_t valid;
  uint8_t route_kind;
  uint8_t write_fragment_count;
  uint8_t required_ack_count;
  uint8_t reserved_zero[4];
  typed_node::selected_child_fetch_work_item_v0 root_fetch;
  typed_node::ray_policy_v0 ray_policy;
  private_write_fragment_v0 write_fragments[kMaxWriteFragmentCount];
};

static_assert(sizeof(private_write_fragment_v0) == 48,
              "Instance private write fragment must remain 48 bytes");
static_assert(offsetof(private_write_fragment_v0, payload) == 16,
              "Instance private write payload offset changed");

bool validate_private_write_fragment(
    const private_write_fragment_v0 &fragment);

status_kind prepare_restore_parent(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_instance::restore_parent_result_v0 &result,
    restore_commit_plan_v0 *plan);

status_kind prepare_enter(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_instance::enter_input_v0 &input,
    const typed_instance::enter_result_v0 &result,
    enter_commit_plan_v0 *plan, bool short_stack_mode = false);

const char *status_name(status_kind status);

}  // namespace instance_semantic
}  // namespace v04
}  // namespace rtcore

#endif
