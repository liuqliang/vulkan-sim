#ifndef RTCORE_V04_STACK_SPILL_RECOVERY_H
#define RTCORE_V04_STACK_SPILL_RECOVERY_H

#include <cstdint>

#include "rtcore_replay_interface.h"
#include "rtcore_v04_fetch_target_queue.h"
#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_target_private_state_384_bridge.h"
#include "rtcore_v04_target_shared_memory_bridge.h"
#include "rtcore_v04_timing_driver.h"

namespace rtcore {
namespace v04 {
namespace stack_spill_recovery {

static const uint8_t kSpillReadCount =
    fetch_target::kStackSpillRecoveryChunks;
static const uint8_t kHandoffReadCount = 1;
static const uint8_t kPrivateReadCount =
    target_shared_memory::kRootPrivateReadChunks;
static const uint8_t kMaxPrivateReadCount =
    target_shared_memory::kMaxPrivateReadChunks;
static const uint8_t kInitialRequestCount =
    kSpillReadCount + kHandoffReadCount + kPrivateReadCount;
static const uint8_t kMaxInitialRequestCount =
    kSpillReadCount + kHandoffReadCount + kMaxPrivateReadCount;
static const uint8_t kPrivateState384TransitionReadCount =
    fetch_target::kPrivateState384TransitionRecoveryChunks;
static const uint8_t kPrivateState384MaxInitialRequestCount =
    kPrivateState384TransitionReadCount +
    private_state_384::operand_materializer::kMaxOperationReadChunks;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusOwnerMismatch,
  kStatusUnsupportedRoute,
  kStatusTargetBackpressure,
  kStatusTargetReservationRejected,
  kStatusPrivateReadPlanRejected,
  kStatusTransportPlanRejected,
  kStatusTimingRejected,
  kStatusMalformedTransport,
  kStatusPrivateReadRejected,
  kStatusTargetFillRejected,
  kStatusHandoffPayloadRejected,
};

struct initial_request_plan_v0 {
  fetch_target::reservation_receipt_v0 reservation;
  rtcore_memory_unit_request_snapshot requests[kMaxInitialRequestCount];
  uint8_t request_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

struct handoff_authority_binding_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t handoff_window_base;
};

struct private_state_384_initial_request_plan_v1 {
  fetch_target::reservation_receipt_v0 reservation;
  rtcore_memory_unit_request_snapshot
      requests[kPrivateState384MaxInitialRequestCount];
  uint8_t request_count;
  uint8_t transition_read_count;
  uint8_t private_read_count;
  uint8_t valid;
  uint8_t reserved_zero[4];
};

status_kind prepare_handoff_ray_policy_request(
    const fetch_target::reservation_receipt_v0 &reservation,
    uint64_t handoff_lane_address, uint64_t issue_cycle,
    rtcore_memory_unit_request_snapshot *request);

status_kind try_reserve_and_prepare_initial_requests(
    fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &backing,
    const timing_driver::pending_recovery_snapshot_v0 &pending,
    const private_frontier::owner_binding_v0 &private_owner,
    uint64_t handoff_lane_address, uint64_t reservation_cycle,
    initial_request_plan_v0 *plan);

status_kind try_reserve_and_prepare_private_state_384_requests(
    fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state,
    const timing_driver::pending_recovery_snapshot_v0 &pending,
    const private_frontier::owner_binding_v0 &private_owner,
    uint64_t private_slot_base_address, uint8_t private_storage_profile,
    uint64_t reservation_cycle,
    private_state_384_initial_request_plan_v1 *plan);

status_kind accept_spill_read_response(
    const private_shared::backing_state_v0 &backing,
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request,
    fetch_target::reservation_receipt_v0 *updated_reservation,
    typed_node::selected_child_fetch_work_item_v0 *selected_fetch);

status_kind accept_private_state_384_transition_read_response(
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request,
    const uint8_t *response_payload, uint8_t response_bytes,
    fetch_target::reservation_receipt_v0 *updated_reservation,
    typed_node::selected_child_fetch_work_item_v0 *selected_fetch);

status_kind accept_handoff_ray_policy_response(
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request,
    const handoff_authority_binding_v0 &authority,
    const uint8_t *response_payload, uint8_t response_bytes);

status_kind validate_handoff_authority_binding(
    const rtcore_memory_unit_request_snapshot &request,
    const handoff_authority_binding_v0 &authority,
    uint64_t *handoff_lane_address);

bool reconstruct_reservation(
    const rtcore_memory_unit_request_snapshot &request,
    fetch_target::reservation_receipt_v0 *reservation);

const char *status_name(status_kind status);

}  // namespace stack_spill_recovery
}  // namespace v04
}  // namespace rtcore

#endif
