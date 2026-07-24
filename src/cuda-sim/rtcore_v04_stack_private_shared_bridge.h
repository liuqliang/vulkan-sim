#ifndef RTCORE_V04_STACK_PRIVATE_SHARED_BRIDGE_H
#define RTCORE_V04_STACK_PRIVATE_SHARED_BRIDGE_H

#include <cstdint>

#include "rtcore_replay_interface.h"
#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_stack_operation_queue.h"

namespace rtcore {
namespace v04 {
namespace stack_private_shared {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusMalformedPlan,
  kStatusMalformedTransport,
  kStatusPrivateReadRejected,
  kStatusStackFillRejected,
};

struct request_plan_v0 {
  rtcore_memory_unit_request_snapshot
      requests[stack_operation::kFrontierMetadataReadChunks];
  uint8_t request_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

status_kind prepare_request_plan(
    const stack_operation::reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan);

status_kind accept_request_and_fill(
    const private_shared::backing_state_v0 &backing,
    stack_operation::engine_state_v0 *stack_state,
    const rtcore_memory_unit_request_snapshot &request);

const char *status_name(status_kind status);

}  // namespace stack_private_shared
}  // namespace v04
}  // namespace rtcore

#endif
