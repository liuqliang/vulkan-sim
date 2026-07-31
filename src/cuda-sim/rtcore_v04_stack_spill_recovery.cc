#include "rtcore_v04_stack_spill_recovery.h"

#include <cstring>

#include "rtcore_abi_v04_generated.h"

namespace rtcore {
namespace v04 {
namespace stack_spill_recovery {
namespace {

static const unsigned kResponseTargetRtcore = 1;
static const uint32_t kRecoveryBaseRequiredOperandMask =
    fetch_target::kOperandTargetReferenceValid |
    fetch_target::kOperandRawPayloadValid |
    fetch_target::kOperandMutableRayValid |
    fetch_target::kOperandRayPolicyValid |
    fetch_target::kOperandDecodeContextValid |
    fetch_target::kOperandCommittedHitValid;
static const uint32_t kHandoffRayPolicyByteMask =
    (uint32_t{0xf} << 16) | (uint32_t{0xf} << 20);
static const uint16_t kHandoffRayPolicySlotOffset =
    abi_v04::kRayFlags.word * sizeof(uint32_t);
static const uint16_t kHandoffRayPolicyChunkOffset =
    (kHandoffRayPolicySlotOffset /
     private_frontier::kSharedAccessChunkBytes) *
    private_frontier::kSharedAccessChunkBytes;
static const uint8_t kRayFlagsOffsetInChunk =
    static_cast<uint8_t>(kHandoffRayPolicySlotOffset -
                         kHandoffRayPolicyChunkOffset);
static const uint8_t kCullMaskOffsetInChunk =
    static_cast<uint8_t>(
        abi_v04::kCullMask.word * sizeof(uint32_t) -
        kHandoffRayPolicyChunkOffset);
static const uint32_t kSpillMemoryOpSeqBase = 0x20;
static const uint32_t kHandoffMemoryOpSeq = 0x30;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool owners_match(
    const request_owner::lane_binding_v0 &request,
    const private_frontier::owner_binding_v0 &private_owner) {
  return request.packed_request_key == private_owner.request_identity &&
         request.owner_hw_sid == private_owner.owner_hw_sid &&
         request.request_generation == private_owner.generation &&
         request.private_slot_id == private_owner.private_slot_id &&
         request.resident_warp_slot ==
             private_owner.resident_warp_id &&
         request.lane_id == private_owner.lane_id &&
         bytes_are_zero(private_owner.reserved_zero,
                        sizeof(private_owner.reserved_zero));
}

void populate_common_request(
    const fetch_target::reservation_receipt_v0 &reservation,
    uint64_t issue_cycle, rtcore_memory_unit_request_snapshot *request) {
  request->valid = true;
  request->operation = RTCORE_MEMORY_OPERATION_READ;
  request->destination =
      RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL;
  request->response_target = kResponseTargetRtcore;
  request->owner_hw_sid = reservation.owner.owner_hw_sid;
  request->rt_request_id = reservation.owner.request_identity;
  request->lane_id = reservation.owner.lane_id;
  request->resident_warp_id = reservation.owner.resident_warp_id;
  request->request_generation = reservation.owner.generation;
  request->private_slot_id = reservation.owner.private_slot_id;
  request->is_write = false;
  request->issue_cycle = issue_cycle;

  rtcore_v04_target_raw_read_transport_snapshot &extension =
      request->v04_target_raw_read;
  extension.reservation_id = reservation.reservation_id;
  extension.reservation_age = reservation.reservation_age;
  extension.raw_payload_base_address =
      reservation.raw_payload_base_address;
  extension.target_operation_seq =
      reservation.target_operation_seq;
  extension.producer_operation_seq =
      reservation.producer_operation_seq;
  extension.producer_commit_epoch =
      reservation.producer_commit_epoch;
  extension.target_slot_generation = reservation.slot_generation;
  extension.private_layout_profile_id =
      reservation.private_layout_profile_id;
  extension.bvh_format_profile_id =
      reservation.bvh_format_profile_id;
  extension.raw_payload_bytes = reservation.raw_payload_bytes;
  extension.target_kind = reservation.target_kind;
  extension.target_slot_index = reservation.slot_index;
  extension.producer_commit_required =
      reservation.producer_commit_required;
  extension.transfer_bytes =
      private_frontier::kSharedAccessChunkBytes;
  extension.private_chunk_count =
      reservation.private_chunk_count;
  extension.operation_kind = reservation.operation_kind;
  extension.private_storage_profile =
      reservation.private_storage_profile;
  extension.valid = 1;
}

bool common_transport_valid(
    const rtcore_memory_unit_request_snapshot &request) {
  const bool producer_valid =
      request.v04_target_raw_read.producer_commit_required == 0
          ? request.v04_target_raw_read.producer_operation_seq == 0 &&
                request.v04_target_raw_read.producer_commit_epoch == 0
          : request.v04_target_raw_read.producer_commit_required == 1 &&
                request.v04_target_raw_read.producer_operation_seq != 0 &&
                request.v04_target_raw_read.producer_commit_epoch != 0;
  return request.valid &&
         request.operation == RTCORE_MEMORY_OPERATION_READ &&
         request.destination ==
             RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL &&
         request.response_target == kResponseTargetRtcore &&
         request.rt_request_id != 0 && request.lane_id < 32 &&
         request.resident_warp_id < request_owner::kResidentWarpCapacity &&
         request.request_generation != 0 &&
         request.private_slot_id <
             request_owner::kRequestControlCapacity &&
         !request.is_write && request.v04_target_raw_read.valid == 1 &&
         request.v04_target_raw_read.reservation_id != 0 &&
         request.v04_target_raw_read.reservation_age != 0 &&
         request.v04_target_raw_read.target_operation_seq != 0 &&
         request.v04_target_raw_read.target_slot_generation != 0 &&
         request.v04_target_raw_read.private_layout_profile_id ==
             private_frontier::kLayoutProfileId &&
         request.v04_target_raw_read.bvh_format_profile_id ==
             typed_node::kGenRtDerivedProfileId &&
         request.v04_target_raw_read.private_storage_profile ==
             private_storage::kProfileLegacyShared832 &&
         producer_valid &&
         request.v04_target_raw_read.private_chunk_count ==
             (request.v04_target_raw_read.target_kind ==
                      fetch_target::kTargetPrimitive
                  ? target_shared_memory::kPrimitivePrivateReadChunks
                  : target_shared_memory::kRootPrivateReadChunks) &&
         request.v04_target_raw_read.transfer_bytes ==
             private_frontier::kSharedAccessChunkBytes &&
         bytes_are_zero(request.v04_target_raw_read.reserved_zero,
                        sizeof(request.v04_target_raw_read.reserved_zero));
}

uint32_t read_le_u32(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

}  // namespace

static_assert(abi_v04::kRayFlags.word == 12,
              "recovery handoff ray-flags word changed");
static_assert(abi_v04::kCullMask.word == 13 &&
                  abi_v04::kCullMask.lsb == 0 &&
                  abi_v04::kCullMask.width == 8,
              "recovery handoff cull-mask field changed");

status_kind prepare_handoff_ray_policy_request(
    const fetch_target::reservation_receipt_v0 &reservation,
    uint64_t handoff_lane_address, uint64_t issue_cycle,
    rtcore_memory_unit_request_snapshot *request) {
  if (request == NULL || reservation.valid != 1 ||
      reservation.target_operation_seq == 0 ||
      handoff_lane_address == 0 ||
      (handoff_lane_address &
       (private_frontier::kSharedAccessChunkBytes - 1)) != 0 ||
      handoff_lane_address >
          UINT64_MAX - kHandoffRayPolicyChunkOffset) {
    return kStatusInvalidArgument;
  }
  *request = rtcore_memory_unit_request_snapshot();
  populate_common_request(reservation, issue_cycle, request);
  request->address_space = RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
  request->memory_op_seq = kHandoffMemoryOpSeq;
  request->chunk_id = 0;
  request->chunk_count = 1;
  request->access_kind =
      RTCORE_MEMORY_ACCESS_HANDOFF_RAY_POLICY_READ;
  request->aligned_32b_addr =
      handoff_lane_address + kHandoffRayPolicyChunkOffset;
  request->byte_mask = kHandoffRayPolicyByteMask;
  request->v04_target_raw_read.slot_chunk_offset =
      kHandoffRayPolicySlotOffset;
  request->v04_target_raw_read.operand_kind =
      RTCORE_MEMORY_TARGET_OPERAND_HANDOFF_RAY_POLICY;
  return common_transport_valid(*request) ? kStatusOk
                                          : kStatusMalformedTransport;
}

bool reconstruct_reservation(
    const rtcore_memory_unit_request_snapshot &request,
    fetch_target::reservation_receipt_v0 *reservation) {
  if (reservation == NULL || !common_transport_valid(request)) {
    return false;
  }
  const rtcore_v04_target_raw_read_transport_snapshot &extension =
      request.v04_target_raw_read;
  *reservation = fetch_target::reservation_receipt_v0();
  reservation->owner.owner_hw_sid = request.owner_hw_sid;
  reservation->owner.resident_warp_id = request.resident_warp_id;
  reservation->owner.request_identity = request.rt_request_id;
  reservation->owner.generation = request.request_generation;
  reservation->owner.private_slot_id = request.private_slot_id;
  reservation->owner.lane_id = request.lane_id;
  reservation->reservation_id = extension.reservation_id;
  reservation->reservation_age = extension.reservation_age;
  reservation->raw_payload_base_address =
      extension.raw_payload_base_address;
  reservation->target_operation_seq =
      extension.target_operation_seq;
  reservation->producer_operation_seq =
      extension.producer_operation_seq;
  reservation->producer_commit_epoch =
      extension.producer_commit_epoch;
  reservation->slot_generation =
      extension.target_slot_generation;
  reservation->private_layout_profile_id =
      extension.private_layout_profile_id;
  reservation->bvh_format_profile_id =
      extension.bvh_format_profile_id;
  reservation->raw_payload_bytes = extension.raw_payload_bytes;
  reservation->target_kind = extension.target_kind;
  reservation->slot_index = extension.target_slot_index;
  reservation->raw_chunk_count = static_cast<uint8_t>(
      extension.raw_payload_bytes /
      private_frontier::kSharedAccessChunkBytes);
  reservation->private_chunk_count =
      extension.private_chunk_count;
  reservation->producer_commit_required =
      extension.producer_commit_required;
  reservation->operation_kind = extension.operation_kind;
  reservation->private_storage_profile =
      extension.private_storage_profile;
  reservation->valid = 1;
  return true;
}

status_kind validate_handoff_authority_binding(
    const rtcore_memory_unit_request_snapshot &request,
    const handoff_authority_binding_v0 &authority,
    uint64_t *handoff_lane_address) {
  if (handoff_lane_address == NULL ||
      authority.handoff_window_base == 0 ||
      (authority.handoff_window_base &
       (abi_v04::kLaneSlotBytes - 1)) != 0 ||
      authority.owner.lane_id >= 32 ||
      !bytes_are_zero(authority.owner.reserved_zero,
                      sizeof(authority.owner.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  fetch_target::reservation_receipt_v0 reservation = {};
  if (!reconstruct_reservation(request, &reservation)) {
    return kStatusMalformedTransport;
  }
  const private_frontier::owner_binding_v0 &request_owner =
      reservation.owner;
  if (request_owner.owner_hw_sid != authority.owner.owner_hw_sid ||
      request_owner.resident_warp_id !=
          authority.owner.resident_warp_id ||
      request_owner.request_identity !=
          authority.owner.request_identity ||
      request_owner.generation != authority.owner.generation ||
      request_owner.private_slot_id !=
          authority.owner.private_slot_id ||
      request_owner.lane_id != authority.owner.lane_id) {
    return kStatusOwnerMismatch;
  }
  const uint64_t lane_offset =
      static_cast<uint64_t>(authority.owner.lane_id) *
      abi_v04::kLaneSlotBytes;
  if (authority.handoff_window_base > UINT64_MAX - lane_offset) {
    return kStatusInvalidArgument;
  }
  const uint64_t lane_address =
      authority.handoff_window_base + lane_offset;
  if (lane_address > UINT64_MAX - kHandoffRayPolicyChunkOffset ||
      request.aligned_32b_addr !=
          lane_address + kHandoffRayPolicyChunkOffset) {
    return kStatusMalformedTransport;
  }
  *handoff_lane_address = lane_address;
  return kStatusOk;
}

status_kind try_reserve_and_prepare_initial_requests(
    fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &backing,
    const timing_driver::pending_recovery_snapshot_v0 &pending,
    const private_frontier::owner_binding_v0 &private_owner,
    uint64_t handoff_lane_address, uint64_t reservation_cycle,
    initial_request_plan_v0 *plan) {
  if (target_state == NULL || timing_state == NULL || plan == NULL ||
      pending.valid != 1 || pending.target_operation_seq == 0 ||
      handoff_lane_address == 0 ||
      (handoff_lane_address &
       (private_frontier::kSharedAccessChunkBytes - 1)) != 0 ||
      handoff_lane_address >
          UINT64_MAX -
              private_frontier::kSharedAccessChunkBytes) {
    return kStatusInvalidArgument;
  }
  *plan = initial_request_plan_v0();
  if (!owners_match(pending.owner, private_owner)) {
    return kStatusOwnerMismatch;
  }
  if (pending.route_kind !=
      timing_driver::kPendingRecoveryRouteStackSelectedFetch) {
    return kStatusUnsupportedRoute;
  }

  fetch_target::engine_state_v0 staged_target = *target_state;
  timing_driver::state_v0 staged_timing = *timing_state;
  fetch_target::recovery_reservation_input_v0 input = {};
  input.owner = private_owner;
  input.target_operation_seq = pending.target_operation_seq;
  input.target_kind = pending.target_kind;
  input.required_operand_mask = static_cast<uint8_t>(
      kRecoveryBaseRequiredOperandMask |
      (pending.target_kind == fetch_target::kTargetPrimitive
           ? fetch_target::kOperandCurrentInstanceValid
           : 0));
  fetch_target::reservation_receipt_v0 reservation = {};
  const fetch_target::status_kind reserve_status =
      fetch_target::try_reserve_recovery(
          &staged_target, input, reservation_cycle, &reservation);
  if (reserve_status == fetch_target::kStatusCapacityBackpressure ||
      reserve_status ==
          fetch_target::kStatusReservationBudgetBackpressure) {
    return kStatusTargetBackpressure;
  }
  if (reserve_status != fetch_target::kStatusOk) {
    return kStatusTargetReservationRejected;
  }

  private_frontier::access_plan_v0 spill_read_plan = {};
  private_frontier::access_plan_v0 private_read_plan = {};
  if (private_shared::prepare_stack_selected_fetch_spill_read_plan(
          backing, private_owner, &spill_read_plan) !=
          private_shared::kStatusOk ||
      spill_read_plan.access_count != kSpillReadCount ||
      (pending.target_kind == fetch_target::kTargetPrimitive
           ? private_shared::prepare_primitive_operand_read_plan(
                 backing, private_owner, &private_read_plan)
           : private_shared::prepare_root_operand_read_plan(
                 backing, private_owner, &private_read_plan)) !=
          private_shared::kStatusOk) {
    return kStatusPrivateReadPlanRejected;
  }
  target_shared_memory::request_plan_v0 private_requests = {};
  if (target_shared_memory::prepare_request_plan(
          reservation, private_read_plan, reservation_cycle,
          &private_requests) != target_shared_memory::kStatusOk ||
      private_requests.request_count !=
          (pending.target_kind == fetch_target::kTargetPrimitive
               ? target_shared_memory::kPrimitivePrivateReadChunks
               : target_shared_memory::kRootPrivateReadChunks)) {
    return kStatusTransportPlanRejected;
  }
  const uint8_t actual_request_count = static_cast<uint8_t>(
      kSpillReadCount + kHandoffReadCount +
      private_requests.request_count);

  for (unsigned index = 0; index < spill_read_plan.access_count; ++index) {
    const private_frontier::shared_chunk_access_v0 &access =
        spill_read_plan.accesses[index];
    rtcore_memory_unit_request_snapshot &request =
        plan->requests[index];
    populate_common_request(reservation, reservation_cycle, &request);
    request.address_space = RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
    request.memory_op_seq = kSpillMemoryOpSeqBase + index;
    request.chunk_id = index;
    request.chunk_count = spill_read_plan.access_count;
    request.access_kind =
        RTCORE_MEMORY_ACCESS_STACK_SPILL_RECOVERY_READ;
    request.aligned_32b_addr = access.aligned_32b_address;
    request.byte_mask = access.byte_mask;
    request.v04_target_raw_read.slot_chunk_offset =
        static_cast<uint16_t>(
            access.aligned_32b_address -
            (UINT64_C(0xff00000000000000) +
             static_cast<uint64_t>(private_owner.owner_hw_sid) *
                 UINT64_C(0x1000000) +
             static_cast<uint64_t>(private_owner.private_slot_id) *
                 private_frontier::kPrivateDataSlotBytes));
    request.v04_target_raw_read.operand_kind =
        RTCORE_MEMORY_TARGET_OPERAND_STACK_SPILL;
    request.v04_target_raw_read.field_kind =
        private_frontier::kFieldTransitionSpill;
  }

  rtcore_memory_unit_request_snapshot &handoff_request =
      plan->requests[kSpillReadCount];
  populate_common_request(
      reservation, reservation_cycle, &handoff_request);
  handoff_request.address_space =
      RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
  handoff_request.memory_op_seq = kHandoffMemoryOpSeq;
  handoff_request.chunk_id = 0;
  handoff_request.chunk_count = 1;
  handoff_request.access_kind =
      RTCORE_MEMORY_ACCESS_HANDOFF_RAY_POLICY_READ;
  handoff_request.aligned_32b_addr =
      handoff_lane_address + kHandoffRayPolicyChunkOffset;
  handoff_request.byte_mask = kHandoffRayPolicyByteMask;
  handoff_request.v04_target_raw_read.slot_chunk_offset =
      kHandoffRayPolicySlotOffset;
  handoff_request.v04_target_raw_read.operand_kind =
      RTCORE_MEMORY_TARGET_OPERAND_HANDOFF_RAY_POLICY;

  std::memcpy(
      plan->requests + kSpillReadCount + kHandoffReadCount,
      private_requests.requests,
      sizeof(private_requests.requests[0]) *
          private_requests.request_count);
  for (unsigned index = 0; index < actual_request_count; ++index) {
    if (timing_driver::begin_memory_transaction(
            &staged_timing, pending.owner,
            pending.target_operation_seq) != timing_driver::kStatusOk) {
      return kStatusTimingRejected;
    }
  }
  if (timing_driver::mark_pending_recovery_reservation_retained(
          &staged_timing, pending.owner,
          pending.target_operation_seq, pending.target_kind,
          pending.route_kind) != timing_driver::kStatusOk) {
    return kStatusTimingRejected;
  }

  plan->reservation = reservation;
  plan->request_count = actual_request_count;
  plan->valid = 1;
  *target_state = staged_target;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind accept_spill_read_response(
    const private_shared::backing_state_v0 &backing,
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request,
    fetch_target::reservation_receipt_v0 *updated_reservation,
    typed_node::selected_child_fetch_work_item_v0 *selected_fetch) {
  if (target_state == NULL || updated_reservation == NULL ||
      selected_fetch == NULL || !common_transport_valid(request) ||
      request.address_space != RTCORE_MEMORY_ADDRESS_SPACE_SHARED ||
      request.access_kind !=
          RTCORE_MEMORY_ACCESS_STACK_SPILL_RECOVERY_READ ||
      request.v04_target_raw_read.operand_kind !=
          RTCORE_MEMORY_TARGET_OPERAND_STACK_SPILL ||
      request.v04_target_raw_read.field_kind !=
          private_frontier::kFieldTransitionSpill ||
      request.v04_target_raw_read.producer_commit_required != 0 ||
      request.chunk_count != kSpillReadCount ||
      request.chunk_id >= request.chunk_count ||
      request.memory_op_seq != kSpillMemoryOpSeqBase + request.chunk_id ||
      request.byte_mask == 0) {
    return kStatusMalformedTransport;
  }
  fetch_target::reservation_receipt_v0 reservation = {};
  if (!reconstruct_reservation(request, &reservation)) {
    return kStatusMalformedTransport;
  }
  private_frontier::access_plan_v0 expected_plan = {};
  if (private_shared::prepare_stack_selected_fetch_spill_read_plan(
          backing, reservation.owner, &expected_plan) !=
          private_shared::kStatusOk ||
      expected_plan.access_count != kSpillReadCount ||
      expected_plan.accesses[request.chunk_id].aligned_32b_address !=
          request.aligned_32b_addr ||
      expected_plan.accesses[request.chunk_id].byte_mask !=
          request.byte_mask) {
    return kStatusMalformedTransport;
  }
  private_frontier::shared_chunk_access_v0 access = {};
  access.aligned_32b_address = request.aligned_32b_addr;
  access.byte_mask = request.byte_mask;
  access.field_kind = private_frontier::kFieldTransitionSpill;
  access.access_kind = private_frontier::kAccessRead;
  uint8_t payload[private_frontier::kSharedAccessChunkBytes] = {};
  if (private_shared::read_canonical_chunk(
          backing, reservation.owner, access, payload) !=
      private_shared::kStatusOk) {
    return kStatusPrivateReadRejected;
  }
  return fetch_target::fill_recovery_descriptor_chunk(
             target_state, reservation,
             static_cast<uint8_t>(request.chunk_id),
             static_cast<uint8_t>(request.chunk_count),
             request.v04_target_raw_read.slot_chunk_offset,
             request.byte_mask, payload, updated_reservation,
             selected_fetch) == fetch_target::kStatusOk
             ? kStatusOk
             : kStatusTargetFillRejected;
}

status_kind accept_handoff_ray_policy_response(
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request,
    const handoff_authority_binding_v0 &authority,
    const uint8_t *response_payload, uint8_t response_bytes) {
  uint64_t expected_handoff_lane_address = 0;
  const status_kind binding_status =
      validate_handoff_authority_binding(
          request, authority, &expected_handoff_lane_address);
  if (binding_status != kStatusOk) {
    return binding_status;
  }
  if (target_state == NULL || response_payload == NULL ||
      response_bytes != private_frontier::kSharedAccessChunkBytes ||
      expected_handoff_lane_address == 0 ||
      !common_transport_valid(request) ||
      request.address_space != RTCORE_MEMORY_ADDRESS_SPACE_SHARED ||
      request.access_kind !=
          RTCORE_MEMORY_ACCESS_HANDOFF_RAY_POLICY_READ ||
      request.v04_target_raw_read.operand_kind !=
          RTCORE_MEMORY_TARGET_OPERAND_HANDOFF_RAY_POLICY ||
      request.chunk_id != 0 || request.chunk_count != 1 ||
      request.memory_op_seq != kHandoffMemoryOpSeq ||
      request.byte_mask != kHandoffRayPolicyByteMask ||
      request.v04_target_raw_read.slot_chunk_offset !=
          kHandoffRayPolicySlotOffset) {
    return kStatusMalformedTransport;
  }
  fetch_target::reservation_receipt_v0 reservation = {};
  if (!reconstruct_reservation(request, &reservation)) {
    return kStatusMalformedTransport;
  }
  typed_node::ray_policy_v0 policy = {};
  policy.ray_flags =
      read_le_u32(response_payload + kRayFlagsOffsetInChunk);
  const uint32_t cull_word =
      read_le_u32(response_payload + kCullMaskOffsetInChunk);
  policy.cull_mask = static_cast<uint8_t>(
      (cull_word & abi_v04::kCullMask.mask) >>
      abi_v04::kCullMask.lsb);
  return fetch_target::fill_recovery_ray_policy(
             target_state, reservation, policy) ==
                 fetch_target::kStatusOk
             ? kStatusOk
             : kStatusTargetFillRejected;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusUnsupportedRoute:
      return "unsupported_route";
    case kStatusTargetBackpressure:
      return "target_backpressure";
    case kStatusTargetReservationRejected:
      return "target_reservation_rejected";
    case kStatusPrivateReadPlanRejected:
      return "private_read_plan_rejected";
    case kStatusTransportPlanRejected:
      return "transport_plan_rejected";
    case kStatusTimingRejected:
      return "timing_rejected";
    case kStatusMalformedTransport:
      return "malformed_transport";
    case kStatusPrivateReadRejected:
      return "private_read_rejected";
    case kStatusTargetFillRejected:
      return "target_fill_rejected";
    case kStatusHandoffPayloadRejected:
      return "handoff_payload_rejected";
  }
  return "unknown";
}

}  // namespace stack_spill_recovery
}  // namespace v04
}  // namespace rtcore
