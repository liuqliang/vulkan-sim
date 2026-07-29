#ifndef RTCORE_V04_SELECTED_FETCH_TRANSITION_H
#define RTCORE_V04_SELECTED_FETCH_TRANSITION_H

#include <cstdint>

#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_target_memory_bridge.h"
#include "rtcore_v04_target_shared_memory_bridge.h"
#include "rtcore_v04_timing_driver.h"

namespace rtcore {
namespace v04 {
namespace selected_fetch_transition {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusOwnerMismatch,
  kStatusTimingControlRejected,
  kStatusTargetBackpressure,
  kStatusTargetReservationRejected,
  kStatusMemoryPlanRejected,
};

struct direct_transition_input_v0 {
  private_frontier::owner_binding_v0 owner;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  typed_node::ray_policy_v0 ray_policy;
  uint32_t producer_operation_seq;
  uint32_t build_generation;
  uint64_t reservation_cycle;
  uint8_t reserved_zero[4];
};

struct accepted_transition_v0 {
  fetch_target::reservation_receipt_v0 reservation;
  target_memory::raw_read_plan_v0 raw_read_plan;
  target_shared_memory::request_plan_v0 private_request_plan;
  uint32_t producer_operation_seq;
  uint32_t target_operation_seq;
  uint8_t target_kind;
  uint8_t valid;
  uint8_t reserved_zero[2];
};

status_kind try_accept_direct(
    timing_driver::state_v0 *timing_state,
    fetch_target::engine_state_v0 *target_state,
    const private_shared::backing_state_v0 &private_backing,
    const direct_transition_input_v0 &input,
    accepted_transition_v0 *accepted);

const char *status_name(status_kind status);

}  // namespace selected_fetch_transition
}  // namespace v04
}  // namespace rtcore

#endif
