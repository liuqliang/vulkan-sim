#include "rtcore_v04_selected_fetch_transition.h"

#include <cstddef>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace selected_fetch_transition {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  if (bytes == NULL) return false;
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool make_request_owner(
    const private_frontier::owner_binding_v0 &private_owner,
    request_owner::lane_binding_v0 *request_binding) {
  if (request_binding == NULL || private_owner.request_identity == 0 ||
      private_owner.generation == 0 ||
      private_owner.generation > request_owner::kRequestGenerationMax ||
      private_owner.resident_warp_id >=
          request_owner::kResidentWarpCapacity ||
      private_owner.private_slot_id >
          std::numeric_limits<uint16_t>::max() ||
      private_owner.lane_id >= request_owner::kLaneCapacity ||
      !bytes_are_zero(private_owner.reserved_zero,
                      sizeof(private_owner.reserved_zero))) {
    return false;
  }
  request_owner::internal_request_key_fields_v0 fields = {};
  if (request_owner::unpack_internal_request_key(
          private_owner.request_identity, &fields) !=
          request_owner::kStatusOk ||
      fields.resident_warp_slot != private_owner.resident_warp_id ||
      fields.lane_id != private_owner.lane_id ||
      fields.request_generation != private_owner.generation) {
    return false;
  }
  *request_binding = request_owner::lane_binding_v0();
  request_binding->packed_request_key =
      private_owner.request_identity;
  request_binding->owner_hw_sid = private_owner.owner_hw_sid;
  request_binding->request_control_slot =
      fields.request_control_slot;
  request_binding->request_generation =
      fields.request_generation;
  request_binding->private_slot_id =
      static_cast<uint16_t>(private_owner.private_slot_id);
  request_binding->resident_warp_slot =
      fields.resident_warp_slot;
  request_binding->lane_id = fields.lane_id;
  return private_frontier::owners_equal(
      request_owner::make_private_frontier_owner(*request_binding),
      private_owner);
}

bool target_backpressure(fetch_target::status_kind status) {
  return status == fetch_target::kStatusCapacityBackpressure ||
         status ==
             fetch_target::kStatusReservationBudgetBackpressure;
}

}  // namespace

status_kind try_accept_direct(
    timing_driver::state_v0 *timing_state,
    fetch_target::engine_state_v0 *target_state,
    const private_shared::backing_state_v0 &private_backing,
    const direct_transition_input_v0 &input,
    accepted_transition_v0 *accepted) {
  if (timing_state == NULL || target_state == NULL || accepted == NULL ||
      !timing_state->initialized || target_state->initialized != 1 ||
      input.producer_operation_seq == 0 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *accepted = accepted_transition_v0();

  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(input.owner, &request_binding) ||
      timing_driver::find_live_lane_control(
          *timing_state, request_binding) == NULL ||
      private_shared::find_live_lane(
          private_backing, input.owner) == NULL) {
    return kStatusOwnerMismatch;
  }

  timing_driver::state_v0 staged_timing = *timing_state;
  fetch_target::engine_state_v0 staged_target = *target_state;
  uint32_t target_operation_seq = 0;
  if (timing_driver::allocate_target_operation(
          &staged_timing, request_binding,
          &target_operation_seq) != timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  if (target_operation_seq == input.producer_operation_seq) {
    return kStatusTimingControlRejected;
  }

  fetch_target::selected_fetch_reservation_input_v0
      reservation_input = {};
  reservation_input.owner = input.owner;
  reservation_input.selected_fetch = input.selected_fetch;
  reservation_input.forwarded_ray_policy = input.ray_policy;
  reservation_input.target_operation_seq = target_operation_seq;
  reservation_input.producer_commit_required = 0;
  reservation_input.required_operand_mask = static_cast<uint8_t>(
      fetch_target::kOperandTargetReferenceValid |
      fetch_target::kOperandRawPayloadValid |
      fetch_target::kOperandMutableRayValid |
      fetch_target::kOperandRayPolicyValid |
      fetch_target::kOperandDecodeContextValid |
      fetch_target::kOperandCommittedHitValid);
  reservation_input.forwarded_operand_mask =
      fetch_target::kOperandRayPolicyValid;

  fetch_target::reservation_receipt_v0 reservation = {};
  const fetch_target::status_kind reserve_status =
      fetch_target::try_reserve_selected_fetch(
          &staged_target, reservation_input,
          input.reservation_cycle, &reservation);
  if (target_backpressure(reserve_status)) {
    return kStatusTargetBackpressure;
  }
  if (reserve_status != fetch_target::kStatusOk) {
    return kStatusTargetReservationRejected;
  }

  target_memory::raw_read_plan_v0 raw_read_plan = {};
  private_frontier::access_plan_v0 private_read_plan = {};
  target_shared_memory::request_plan_v0 private_request_plan = {};
  if (target_memory::prepare_raw_read_plan(
          reservation, &raw_read_plan) != target_memory::kStatusOk ||
      private_shared::prepare_root_operand_read_plan(
          private_backing, input.owner,
          &private_read_plan) != private_shared::kStatusOk ||
      target_shared_memory::prepare_request_plan(
          reservation, private_read_plan,
          input.reservation_cycle,
          &private_request_plan) !=
          target_shared_memory::kStatusOk) {
    return kStatusMemoryPlanRejected;
  }

  const unsigned transaction_count =
      static_cast<unsigned>(raw_read_plan.chunk_count) +
      static_cast<unsigned>(private_request_plan.request_count);
  for (unsigned index = 0; index < transaction_count; ++index) {
    if (timing_driver::begin_memory_transaction(
            &staged_timing, request_binding,
            target_operation_seq) != timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
  }

  accepted->reservation = reservation;
  accepted->raw_read_plan = raw_read_plan;
  accepted->private_request_plan = private_request_plan;
  accepted->producer_operation_seq =
      input.producer_operation_seq;
  accepted->target_operation_seq = target_operation_seq;
  accepted->target_kind = reservation.target_kind;
  accepted->valid = 1;
  *timing_state = staged_timing;
  *target_state = staged_target;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusTimingControlRejected:
      return "timing_control_rejected";
    case kStatusTargetBackpressure:
      return "target_backpressure";
    case kStatusTargetReservationRejected:
      return "target_reservation_rejected";
    case kStatusMemoryPlanRejected:
      return "memory_plan_rejected";
  }
  return "unknown";
}

}  // namespace selected_fetch_transition
}  // namespace v04
}  // namespace rtcore
