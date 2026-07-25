#ifndef RTCORE_V04_TARGET_SHARED_MEMORY_BRIDGE_H
#define RTCORE_V04_TARGET_SHARED_MEMORY_BRIDGE_H

#include <cstdint>

#include "rtcore_replay_interface.h"
#include "rtcore_v04_fetch_target_queue.h"
#include "rtcore_v04_private_shared_backing.h"

namespace rtcore {
namespace v04 {
namespace target_shared_memory {

static const uint8_t kRootPrivateReadChunks = 7;
static const uint8_t kPrimitivePrivateReadChunks = 9;
static const uint8_t kMaxPrivateReadChunks = kPrimitivePrivateReadChunks;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusMalformedPlan,
  kStatusMalformedTransport,
  kStatusPrivateReadRejected,
  kStatusTargetFillRejected,
};

struct request_plan_v0 {
  rtcore_memory_unit_request_snapshot requests[kMaxPrivateReadChunks];
  uint8_t request_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

status_kind prepare_request_plan(
    const fetch_target::reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan);

status_kind accept_request_and_fill(
    const private_shared::backing_state_v0 &backing,
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request);

const char *status_name(status_kind status);

}  // namespace target_shared_memory
}  // namespace v04
}  // namespace rtcore

#endif
