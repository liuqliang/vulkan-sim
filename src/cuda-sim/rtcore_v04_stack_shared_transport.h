#ifndef RTCORE_V04_STACK_SHARED_TRANSPORT_H
#define RTCORE_V04_STACK_SHARED_TRANSPORT_H

#include <cstdint>

#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_stack_result_commit.h"

namespace rtcore {
namespace v04 {
namespace stack_shared {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusNoWriteOffer,
  kStatusInvalidWriteOffer,
  kStatusSharedQueueBackpressure,
  kStatusSharedWriteRejected,
  kStatusNoAckReady,
  kStatusStackAckRejected,
  kStatusSharedAckRejected,
};

struct transfer_receipt_v0 {
  bool valid;
  uint8_t reserved_zero[7];
  stack_commit::write_offer_v0 stack_offer;
  private_shared::shared_write_v0 shared_write;
};

struct ack_receipt_v0 {
  bool valid;
  uint8_t reserved_zero[7];
  private_shared::runtime_write_ack_v0 shared_ack;
};

struct ack_service_summary_v0 {
  uint32_t consumed;
  uint32_t init_consumed;
  uint32_t runtime_consumed;
  uint32_t reserved_zero;
};

status_kind prepare_shared_write(const stack_commit::write_offer_v0 &offer,
                                 uint64_t enqueue_cycle,
                                 private_shared::shared_write_v0 *operation);

status_kind transfer_next_write(stack_commit::engine_state_v0 *stack_state,
                                private_shared::backing_state_v0 *shared_state,
                                uint64_t enqueue_cycle,
                                transfer_receipt_v0 *receipt);

status_kind service_next_ack(stack_commit::engine_state_v0 *stack_state,
                             private_shared::backing_state_v0 *shared_state,
                             uint64_t service_cycle, ack_receipt_v0 *receipt);

status_kind service_ready_acks(stack_commit::engine_state_v0 *stack_state,
                               private_shared::backing_state_v0 *shared_state,
                               uint64_t service_cycle, uint32_t response_budget,
                               ack_service_summary_v0 *summary);

const char *status_name(status_kind status);

}  // namespace stack_shared
}  // namespace v04
}  // namespace rtcore

#endif
