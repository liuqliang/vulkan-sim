#include "rtcore_v04_stack_shared_transport.h"

#include <cstring>

#include "rtcore_v04_private_shared_backing_internal.h"

namespace rtcore {
namespace v04 {
namespace stack_shared {

status_kind prepare_shared_write(const stack_commit::write_offer_v0 &offer,
                                 uint64_t enqueue_cycle,
                                 private_shared::shared_write_v0 *operation) {
  if (operation == NULL || offer.operation_seq == 0 ||
      offer.commit_epoch == 0 || offer.memory_operation_seq == 0 ||
      offer.write_count == 0 ||
      offer.memory_operation_seq > offer.write_count ||
      offer.write_count > stack_commit::kMaxWritesPerTransaction ||
      !stack_semantic::validate_private_write_fragment(offer.fragment)) {
    return kStatusInvalidArgument;
  }
  for (unsigned byte = 0; byte < sizeof(offer.reserved_zero); ++byte) {
    if (offer.reserved_zero[byte] != 0) {
      return kStatusInvalidWriteOffer;
    }
  }

  *operation = private_shared::shared_write_v0();
  operation->valid = true;
  operation->address_space = private_shared::kAddressSpaceShared;
  operation->address_mode = private_shared::kAddressModePrivateField;
  operation->access_operation = private_shared::kAccessOperationWrite;
  operation->destination = private_shared::kDestinationPrivateCommitAck;
  operation->owner = offer.owner;
  operation->operation_seq = offer.operation_seq;
  operation->commit_epoch = offer.commit_epoch;
  operation->memory_op_seq = offer.memory_operation_seq;
  operation->chunk_id = static_cast<uint8_t>(offer.memory_operation_seq - 1);
  operation->chunk_count = static_cast<uint8_t>(offer.write_count);
  operation->field_kind = offer.fragment.field_kind;
  operation->aligned_32b_address = offer.fragment.aligned_32b_address;
  operation->byte_mask = offer.fragment.byte_mask;
  std::memcpy(operation->payload, offer.fragment.payload,
              sizeof(operation->payload));
  operation->enqueue_cycle = enqueue_cycle;
  return kStatusOk;
}

status_kind transfer_next_write(stack_commit::engine_state_v0 *stack_state,
                                private_shared::backing_state_v0 *shared_state,
                                uint64_t enqueue_cycle,
                                transfer_receipt_v0 *receipt) {
  if (stack_state == NULL || shared_state == NULL || receipt == NULL) {
    return kStatusInvalidArgument;
  }
  std::memset(receipt, 0, sizeof(*receipt));

  stack_commit::write_offer_v0 offer = {};
  const stack_commit::status_kind peek_status =
      stack_commit::peek_write_offer(*stack_state, &offer);
  if (peek_status == stack_commit::kStatusNoWriteOffer) {
    return kStatusNoWriteOffer;
  }
  if (peek_status != stack_commit::kStatusOk ||
      stack_commit::validate_write_offer(*stack_state, offer) !=
          stack_commit::kStatusOk) {
    return kStatusInvalidWriteOffer;
  }

  private_shared::shared_write_v0 operation = {};
  const status_kind prepare_status =
      prepare_shared_write(offer, enqueue_cycle, &operation);
  if (prepare_status != kStatusOk) return prepare_status;
  const private_shared::status_kind memory_validation =
      private_shared::validate_runtime_write(*shared_state, operation);
  if (memory_validation == private_shared::kStatusQueueFull) {
    return kStatusSharedQueueBackpressure;
  }
  if (memory_validation != private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }

  stack_commit::engine_state_v0 staged_stack = *stack_state;
  if (stack_commit::accept_write_offer(&staged_stack, offer) !=
      stack_commit::kStatusOk) {
    return kStatusInvalidWriteOffer;
  }
  if (private_shared::enqueue_runtime_write(shared_state, operation) !=
      private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }
  *stack_state = staged_stack;
  receipt->valid = true;
  receipt->stack_offer = offer;
  receipt->shared_write = operation;
  return kStatusOk;
}

status_kind service_next_ack(stack_commit::engine_state_v0 *stack_state,
                             private_shared::backing_state_v0 *shared_state,
                             uint64_t service_cycle, ack_receipt_v0 *receipt) {
  if (stack_state == NULL || shared_state == NULL || receipt == NULL) {
    return kStatusInvalidArgument;
  }
  std::memset(receipt, 0, sizeof(*receipt));

  private_shared::runtime_write_ack_v0 ack = {};
  const private_shared::status_kind peek_status =
      private_shared::detail::peek_runtime_write_ack(*shared_state,
                                                     service_cycle, &ack);
  if (peek_status == private_shared::kStatusNoAckReady) {
    return kStatusNoAckReady;
  }
  if (peek_status != private_shared::kStatusOk) {
    return kStatusSharedAckRejected;
  }

  stack_commit::engine_state_v0 staged_stack = *stack_state;
  if (stack_commit::accept_write_ack(
          &staged_stack, ack.owner, ack.operation_seq, ack.commit_epoch,
          ack.memory_operation_seq) != stack_commit::kStatusOk) {
    return kStatusStackAckRejected;
  }
  if (private_shared::detail::commit_runtime_write_ack(
          shared_state, service_cycle, ack) != private_shared::kStatusOk) {
    return kStatusSharedAckRejected;
  }

  *stack_state = staged_stack;
  receipt->valid = true;
  receipt->shared_ack = ack;
  return kStatusOk;
}

status_kind service_ready_acks(stack_commit::engine_state_v0 *stack_state,
                               private_shared::backing_state_v0 *shared_state,
                               uint64_t service_cycle, uint32_t response_budget,
                               ack_service_summary_v0 *summary) {
  if (stack_state == NULL || shared_state == NULL || summary == NULL ||
      response_budget == 0) {
    return kStatusInvalidArgument;
  }
  *summary = ack_service_summary_v0();

  while (summary->consumed < response_budget &&
         !shared_state->outstanding.empty() &&
         shared_state->outstanding.front().ack_cycle <= service_cycle) {
    const private_shared::shared_write_v0 &front =
        shared_state->outstanding.front();
    const bool init_transaction =
        front.operation_seq == 0 && front.commit_epoch == 0;
    const bool runtime_transaction =
        front.operation_seq != 0 && front.commit_epoch != 0;
    if (init_transaction) {
      if (private_shared::service_write_acks(shared_state, service_cycle, 1) !=
          1) {
        return kStatusSharedAckRejected;
      }
      ++summary->init_consumed;
    } else if (runtime_transaction) {
      ack_receipt_v0 receipt = {};
      const status_kind status =
          service_next_ack(stack_state, shared_state, service_cycle, &receipt);
      if (status != kStatusOk || !receipt.valid) return status;
      ++summary->runtime_consumed;
    } else {
      return kStatusSharedAckRejected;
    }
    ++summary->consumed;
  }
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusNoWriteOffer:
      return "no_write_offer";
    case kStatusInvalidWriteOffer:
      return "invalid_write_offer";
    case kStatusSharedQueueBackpressure:
      return "shared_queue_backpressure";
    case kStatusSharedWriteRejected:
      return "shared_write_rejected";
    case kStatusNoAckReady:
      return "no_ack_ready";
    case kStatusStackAckRejected:
      return "stack_ack_rejected";
    case kStatusSharedAckRejected:
      return "shared_ack_rejected";
  }
  return "unknown";
}

}  // namespace stack_shared
}  // namespace v04
}  // namespace rtcore
