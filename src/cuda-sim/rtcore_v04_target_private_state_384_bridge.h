#ifndef RTCORE_V04_TARGET_PRIVATE_STATE_384_BRIDGE_H
#define RTCORE_V04_TARGET_PRIVATE_STATE_384_BRIDGE_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_fetch_target_queue.h"

namespace rtcore {
namespace v04 {
namespace target_private_state_384 {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusPlanMismatch,
  kStatusTargetRejected,
  kStatusResponseRejected,
  kStatusMaterializeRejected,
};

status_kind configure_read(
    fetch_target::engine_state_v0 *state,
    const fetch_target::reservation_receipt_v0 &reservation,
    const private_state_384::live_bridge::read_request_plan_v1 &plan,
    fetch_target::reservation_receipt_v0 *updated_reservation);

status_kind accept_response(
    fetch_target::engine_state_v0 *state,
    const private_state_384::backing::state_v1 &backing,
    const rtcore_memory_unit_request_snapshot &request);

status_kind accept_response_bytes(
    fetch_target::engine_state_v0 *state,
    const rtcore_memory_unit_request_snapshot &request,
    const uint8_t *payload, size_t payload_byte_count);

const char *status_name(status_kind status);

}  // namespace target_private_state_384
}  // namespace v04
}  // namespace rtcore

#endif
