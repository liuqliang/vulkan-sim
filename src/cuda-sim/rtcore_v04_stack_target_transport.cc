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

stack_commit::forwarding_kind select_live_target_reservation(
    void *opaque,
    const stack_commit::forwarding_decision_input_v0 &decision) {
  if (opaque == NULL ||
      decision.producer_operation_seq == 0 ||
      decision.target_operation_seq == 0 ||
      decision.producer_operation_seq ==
          decision.target_operation_seq) {
    return stack_commit::kForwardingInvalid;
  }
  selector_context_v0 *context =
      static_cast<selector_context_v0 *>(opaque);
  fetch_target::selected_fetch_reservation_input_v0 input = {};
  input.owner = decision.owner;
  input.selected_fetch = decision.selected_fetch;
  input.forwarded_ray_policy = decision.forwarded_ray_policy;
  input.target_operation_seq = decision.target_operation_seq;
  input.producer_operation_seq =
      decision.producer_operation_seq;
  input.producer_commit_epoch = decision.commit_epoch;
  input.producer_commit_required = 1;
  input.required_operand_mask = static_cast<uint8_t>(
      fetch_target::kOperandTargetReferenceValid |
      fetch_target::kOperandRawPayloadValid |
      fetch_target::kOperandMutableRayValid |
      fetch_target::kOperandRayPolicyValid |
      fetch_target::kOperandDecodeContextValid |
      fetch_target::kOperandCommittedHitValid);
  input.forwarded_operand_mask =
      fetch_target::kOperandRayPolicyValid;
  context->reservation_status =
      fetch_target::try_reserve_selected_fetch(
          context->target_state, input,
          context->reservation_cycle, &context->reservation);
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

stack_commit::forwarding_kind
select_live_target_reservation_without_ray_policy(
    void *opaque,
    const stack_commit::forwarding_decision_input_v0 &decision) {
  if (opaque == NULL ||
      decision.producer_operation_seq == 0 ||
      decision.target_operation_seq == 0 ||
      decision.producer_operation_seq ==
          decision.target_operation_seq) {
    return stack_commit::kForwardingInvalid;
  }
  selector_context_v0 *context =
      static_cast<selector_context_v0 *>(opaque);
  fetch_target::selected_fetch_reservation_input_v0 input = {};
  input.owner = decision.owner;
  input.selected_fetch = decision.selected_fetch;
  input.target_operation_seq = decision.target_operation_seq;
  input.producer_operation_seq =
      decision.producer_operation_seq;
  input.producer_commit_epoch = decision.commit_epoch;
  input.producer_commit_required = 1;
  input.required_operand_mask = static_cast<uint8_t>(
      fetch_target::kOperandTargetReferenceValid |
      fetch_target::kOperandRawPayloadValid |
      fetch_target::kOperandMutableRayValid |
      fetch_target::kOperandRayPolicyValid |
      fetch_target::kOperandDecodeContextValid |
      fetch_target::kOperandCommittedHitValid);
  input.forwarded_operand_mask = 0;
  context->reservation_status =
      fetch_target::try_reserve_selected_fetch(
          context->target_state, input,
          context->reservation_cycle, &context->reservation);
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
    if (stack_status ==
            stack_commit::kStatusResultCommitBackpressure ||
        stack_status == stack_commit::kStatusTrackerBackpressure) {
      return kStatusStackCommitBackpressure;
    }
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
    uint64_t reservation_cycle, issue_receipt_v0 *receipt) {
  if (stack_state == NULL || target_state == NULL || receipt == NULL ||
      stack_state->initialized != 1 || target_state->initialized != 1 ||
      producer_operation_seq == 0 || commit_epoch == 0 ||
      target_operation_seq == 0 ||
      producer_operation_seq == target_operation_seq) {
    return kStatusInvalidArgument;
  }
  std::memset(receipt, 0, sizeof(*receipt));

  stack_commit::engine_state_v0 staged_stack = *stack_state;
  fetch_target::engine_state_v0 staged_targets = *target_state;
  selector_context_v0 selector_context = {};
  selector_context.target_state = &staged_targets;
  selector_context.reservation_cycle = reservation_cycle;
  selector_context.reservation_status =
      fetch_target::kStatusInvalidArgument;

  stack_commit::issue_receipt_v0 stack_receipt = {};
  const stack_commit::status_kind stack_status =
      stack_commit::
          capture_stack_push_result_from_projection_with_selector(
              &staged_stack, owner, producer_operation_seq,
              commit_epoch, target_operation_seq, region,
              frontier_metadata, input, result,
              forwarded_ray_policy, select_live_target_reservation,
              &selector_context, &stack_receipt);
  if (stack_status != stack_commit::kStatusOk) {
    if (stack_status ==
            stack_commit::kStatusResultCommitBackpressure ||
        stack_status == stack_commit::kStatusTrackerBackpressure) {
      return kStatusStackCommitBackpressure;
    }
    return stack_status ==
                   stack_commit::kStatusForwardingDecisionRejected
               ? kStatusTargetReservationRejected
               : kStatusStackIssueRejected;
  }

  const bool registered =
      stack_receipt.forwarding_kind ==
      stack_commit::kForwardingRegistered;
  const bool spilled =
      stack_receipt.forwarding_kind ==
      stack_commit::kForwardingSpillToMemory;
  const bool recoverable_reservation_loss =
      selector_context.reservation_status ==
          fetch_target::kStatusCapacityBackpressure ||
      selector_context.reservation_status ==
          fetch_target::kStatusReservationBudgetBackpressure;
  if (stack_receipt.producer_operation_seq !=
          producer_operation_seq ||
      stack_receipt.target_operation_seq !=
          target_operation_seq ||
      stack_receipt.commit_epoch != commit_epoch ||
      (registered &&
       (selector_context.reservation_status !=
            fetch_target::kStatusOk ||
        selector_context.reservation.valid != 1 ||
        selector_context.reservation.target_operation_seq !=
            target_operation_seq ||
        selector_context.reservation.producer_operation_seq !=
            producer_operation_seq ||
        selector_context.reservation.producer_commit_epoch !=
            commit_epoch)) ||
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
  receipt->reservation_status =
      selector_context.reservation_status;
  receipt->stack_issue = stack_receipt;
  receipt->target_reservation =
      selector_context.reservation;
  return kStatusOk;
}

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
    uint64_t reservation_cycle, issue_receipt_v0 *receipt) {
  if (stack_state == NULL || target_state == NULL || receipt == NULL ||
      stack_state->initialized != 1 || target_state->initialized != 1 ||
      producer_operation_seq == 0 || commit_epoch == 0 ||
      target_operation_seq == 0 ||
      producer_operation_seq == target_operation_seq) {
    return kStatusInvalidArgument;
  }
  std::memset(receipt, 0, sizeof(*receipt));

  stack_commit::engine_state_v0 staged_stack = *stack_state;
  fetch_target::engine_state_v0 staged_targets = *target_state;
  selector_context_v0 selector_context = {};
  selector_context.target_state = &staged_targets;
  selector_context.reservation_cycle = reservation_cycle;
  selector_context.reservation_status =
      fetch_target::kStatusInvalidArgument;
  stack_commit::issue_receipt_v0 stack_receipt = {};
  const stack_commit::status_kind stack_status =
      stack_commit::
          capture_stack_pop_result_from_projection_with_selector(
              &staged_stack, owner, producer_operation_seq,
              commit_epoch, target_operation_seq, region,
              frontier_metadata, input, result,
              forwarded_ray_policy,
              select_live_target_reservation_without_ray_policy,
              &selector_context, &stack_receipt);
  if (stack_status != stack_commit::kStatusOk) {
    if (stack_status ==
            stack_commit::kStatusResultCommitBackpressure ||
        stack_status == stack_commit::kStatusTrackerBackpressure) {
      return kStatusStackCommitBackpressure;
    }
    return stack_status ==
                   stack_commit::kStatusForwardingDecisionRejected
               ? kStatusTargetReservationRejected
               : kStatusStackIssueRejected;
  }

  const bool retry =
      stack_receipt.forwarding_kind ==
      stack_commit::kForwardingRetryStackPop;
  const bool registered =
      stack_receipt.forwarding_kind ==
      stack_commit::kForwardingRegistered;
  const bool spilled =
      stack_receipt.forwarding_kind ==
      stack_commit::kForwardingSpillToMemory;
  const bool recoverable_reservation_loss =
      selector_context.reservation_status ==
          fetch_target::kStatusCapacityBackpressure ||
      selector_context.reservation_status ==
          fetch_target::kStatusReservationBudgetBackpressure;
  if (stack_receipt.producer_operation_seq !=
          producer_operation_seq ||
      stack_receipt.target_operation_seq !=
          target_operation_seq ||
      stack_receipt.commit_epoch != commit_epoch ||
      (retry &&
       (result.result_kind != typed_stack::kStackPrunedRetryPop ||
        selector_context.reservation.valid != 0)) ||
      (registered &&
       (selector_context.reservation_status !=
            fetch_target::kStatusOk ||
        selector_context.reservation.valid != 1)) ||
      (spilled &&
       (!recoverable_reservation_loss ||
        selector_context.reservation.valid != 0)) ||
      (!retry && !registered && !spilled)) {
    return kStatusReservationDecisionMismatch;
  }

  *stack_state = staged_stack;
  *target_state = staged_targets;
  receipt->valid = 1;
  receipt->used_transition_spill = spilled;
  receipt->reservation_status =
      selector_context.reservation_status;
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
            &staged_targets, event.owner,
            event.producer_operation_seq,
            event.commit_epoch, &reservation) != fetch_target::kStatusOk) {
      return kStatusTargetCommitRejected;
    }
  } else if (event.ready_kind != stack_commit::kReadySpillRecovery &&
             event.ready_kind != stack_commit::kReadyStackPopRetry) {
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
    case kStatusStackCommitBackpressure:
      return "stack_commit_backpressure";
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
