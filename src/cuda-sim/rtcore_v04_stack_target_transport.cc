#include "rtcore_v04_stack_target_transport.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace stack_target {
namespace {

struct selector_context_v0 {
  fetch_target::engine_state_v0 *target_state;
  uint64_t reservation_cycle;
  fetch_target::status_kind reservation_status;
  fetch_target::reservation_receipt_v0 reservation;
};

stack_commit::forwarding_kind select_target_reservation(
    void *opaque,
    const stack_commit::forwarding_decision_input_v0 &decision) {
  if (opaque == NULL) return stack_commit::kForwardingInvalid;
  selector_context_v0 *context =
      static_cast<selector_context_v0 *>(opaque);
  context->reservation_status = fetch_target::try_reserve_prefill(
      context->target_state, decision, context->reservation_cycle,
      &context->reservation);
  if (context->reservation_status == fetch_target::kStatusOk) {
    return stack_commit::kForwardingRegistered;
  }
  if (context->reservation_status ==
          fetch_target::kStatusCapacityBackpressure ||
      context->reservation_status ==
          fetch_target::kStatusReservationBudgetBackpressure) {
    return stack_commit::kForwardingSpillToMemory;
  }
  return stack_commit::kForwardingInvalid;
}

}  // namespace

status_kind issue_stack_push(
    stack_commit::engine_state_v0 *stack_state,
    fetch_target::engine_state_v0 *target_state,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_input_v0 &input, uint64_t reservation_cycle,
    issue_receipt_v0 *receipt) {
  if (stack_state == NULL || target_state == NULL || receipt == NULL ||
      stack_state->initialized != 1 || target_state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  std::memset(receipt, 0, sizeof(*receipt));

  stack_commit::engine_state_v0 staged_stack = *stack_state;
  fetch_target::engine_state_v0 staged_targets = *target_state;
  selector_context_v0 selector_context = {};
  selector_context.target_state = &staged_targets;
  selector_context.reservation_cycle = reservation_cycle;
  selector_context.reservation_status = fetch_target::kStatusInvalidArgument;

  stack_commit::issue_receipt_v0 stack_receipt = {};
  const stack_commit::status_kind stack_status =
      stack_commit::issue_stack_push_with_selector(
          &staged_stack, owner, operation_seq, region, canonical_slot, input,
          select_target_reservation, &selector_context, &stack_receipt);
  if (stack_status != stack_commit::kStatusOk) {
    return stack_status ==
                   stack_commit::kStatusForwardingDecisionRejected
               ? kStatusTargetReservationRejected
               : kStatusStackIssueRejected;
  }

  const bool registered =
      stack_receipt.forwarding_kind == stack_commit::kForwardingRegistered;
  const bool spilled =
      stack_receipt.forwarding_kind == stack_commit::kForwardingSpillToMemory;
  const bool recoverable_reservation_loss =
      selector_context.reservation_status ==
          fetch_target::kStatusCapacityBackpressure ||
      selector_context.reservation_status ==
          fetch_target::kStatusReservationBudgetBackpressure;
  if ((registered &&
       (selector_context.reservation_status != fetch_target::kStatusOk ||
        selector_context.reservation.valid != 1)) ||
      (spilled &&
       (!recoverable_reservation_loss ||
        selector_context.reservation.valid != 0)) ||
      (!registered && !spilled)) {
    return kStatusReservationDecisionMismatch;
  }

  *stack_state = staged_stack;
  *target_state = staged_targets;
  receipt->valid = 1;
  receipt->used_transition_spill = spilled;
  receipt->reservation_status = selector_context.reservation_status;
  receipt->stack_issue = stack_receipt;
  receipt->target_reservation = selector_context.reservation;
  return kStatusOk;
}

status_kind route_next_ready_event(
    stack_commit::engine_state_v0 *stack_state,
    fetch_target::engine_state_v0 *target_state,
    ready_route_receipt_v0 *receipt) {
  if (stack_state == NULL || target_state == NULL || receipt == NULL ||
      stack_state->initialized != 1 || target_state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  std::memset(receipt, 0, sizeof(*receipt));

  stack_commit::engine_state_v0 staged_stack = *stack_state;
  fetch_target::engine_state_v0 staged_targets = *target_state;
  stack_commit::ready_event_v0 event = {};
  const stack_commit::status_kind pop_status =
      stack_commit::pop_ready_event(&staged_stack, &event);
  if (pop_status == stack_commit::kStatusNoReadyEvent) {
    return kStatusNoReadyEvent;
  }
  if (pop_status != stack_commit::kStatusOk) {
    return kStatusStackIssueRejected;
  }

  fetch_target::reservation_receipt_v0 reservation = {};
  if (event.ready_kind == stack_commit::kReadyForwardedTarget) {
    if (fetch_target::complete_producer_commit(
            &staged_targets, event.owner, event.operation_seq,
            event.commit_epoch, &reservation) != fetch_target::kStatusOk) {
      return kStatusTargetCommitRejected;
    }
  } else if (event.ready_kind != stack_commit::kReadySpillRecovery) {
    return kStatusTargetCommitRejected;
  }

  *stack_state = staged_stack;
  *target_state = staged_targets;
  receipt->valid = 1;
  receipt->stack_event = event;
  receipt->target_reservation = reservation;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusStackIssueRejected:
      return "stack_issue_rejected";
    case kStatusTargetReservationRejected:
      return "target_reservation_rejected";
    case kStatusReservationDecisionMismatch:
      return "reservation_decision_mismatch";
    case kStatusNoReadyEvent:
      return "no_ready_event";
    case kStatusTargetCommitRejected:
      return "target_commit_rejected";
  }
  return "unknown";
}

}  // namespace stack_target
}  // namespace v04
}  // namespace rtcore
