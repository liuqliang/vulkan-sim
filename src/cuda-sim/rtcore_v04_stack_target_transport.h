#ifndef RTCORE_V04_STACK_TARGET_TRANSPORT_H
#define RTCORE_V04_STACK_TARGET_TRANSPORT_H

#include <cstdint>

#include "rtcore_v04_fetch_target_queue.h"

namespace rtcore {
namespace v04 {
namespace stack_target {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusStackCommitBackpressure,
  kStatusStackIssueRejected,
  kStatusTargetReservationRejected,
  kStatusReservationDecisionMismatch,
  kStatusNoReadyEvent,
  kStatusTargetCommitRejected,
};

struct issue_receipt_v0 {
  uint8_t valid;
  uint8_t used_transition_spill;
  uint8_t reservation_status;
  uint8_t reserved_zero[5];
  stack_commit::issue_receipt_v0 stack_issue;
  fetch_target::reservation_receipt_v0 target_reservation;
};

struct ready_route_receipt_v0 {
  uint8_t valid;
  uint8_t reserved_zero[7];
  stack_commit::ready_event_v0 stack_event;
  fetch_target::reservation_receipt_v0 target_reservation;
};

status_kind issue_stack_push(
    stack_commit::engine_state_v0 *stack_state,
    fetch_target::engine_state_v0 *target_state,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_input_v0 &input, uint64_t reservation_cycle,
    issue_receipt_v0 *receipt);

status_kind capture_live_stack_push_result(
    stack_commit::engine_state_v0 *stack_state,
    fetch_target::engine_state_v0 *target_state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::frontier_metadata_image_v0 &frontier_metadata,
    const typed_stack::push_input_v0 &input,
    const typed_stack::push_result_v0 &result,
    const typed_node::ray_policy_v0 &forwarded_ray_policy,
    uint64_t reservation_cycle, issue_receipt_v0 *receipt);

status_kind capture_live_stack_pop_result(
    stack_commit::engine_state_v0 *stack_state,
    fetch_target::engine_state_v0 *target_state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::frontier_metadata_image_v0
        &frontier_metadata,
    const typed_stack::pop_input_v0 &input,
    const typed_stack::pop_result_v0 &result,
    const typed_node::ray_policy_v0 &forwarded_ray_policy,
    uint64_t reservation_cycle, issue_receipt_v0 *receipt);

status_kind route_next_ready_event(
    stack_commit::engine_state_v0 *stack_state,
    fetch_target::engine_state_v0 *target_state,
    ready_route_receipt_v0 *receipt);

const char *status_name(status_kind status);

}  // namespace stack_target
}  // namespace v04
}  // namespace rtcore

#endif
