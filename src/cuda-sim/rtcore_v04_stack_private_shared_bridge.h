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

enum read_phase_kind : uint8_t {
  kReadPhaseInvalid = 0,
  kReadPhaseFrontierMetadata = 1,
  kReadPhasePopOperands = 2,
  kReadPhaseEmptyOperands = 3,
};

struct request_plan_v0 {
  rtcore_memory_unit_request_snapshot
      requests[stack_operation::kMaxPopOperandReadChunks];
  uint8_t request_count;
  uint8_t read_phase;
  uint8_t valid;
  uint8_t reserved_zero[5];
};

struct fill_result_v0 {
  private_frontier::access_plan_v0 followup_read_plan;
  uint8_t followup_required;
  uint8_t empty_frontier_boundary;
  uint8_t ready;
  uint8_t valid;
  uint8_t reserved_zero[4];
};

bool should_activate_live_followup(
    const fill_result_v0 &fill_result,
    bool empty_frontier_gate_enabled);

status_kind prepare_request_plan(
    const stack_operation::reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan);

status_kind prepare_pop_operand_request_plan(
    const stack_operation::reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan);

status_kind prepare_empty_operand_request_plan(
    const stack_operation::reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan);

status_kind accept_request_and_fill(
    const private_shared::backing_state_v0 &backing,
    stack_operation::engine_state_v0 *stack_state,
    const rtcore_memory_unit_request_snapshot &request);

status_kind accept_request_and_fill(
    const private_shared::backing_state_v0 &backing,
    stack_operation::engine_state_v0 *stack_state,
    const rtcore_memory_unit_request_snapshot &request,
    fill_result_v0 *result);

const char *status_name(status_kind status);

}  // namespace stack_private_shared
}  // namespace v04
}  // namespace rtcore

#endif
