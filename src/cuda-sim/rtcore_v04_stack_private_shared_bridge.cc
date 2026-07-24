#include "rtcore_v04_stack_private_shared_bridge.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace stack_private_shared {
namespace {

static const unsigned kResponseTargetRtcore = 1;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

uint64_t private_slot_base(
    const private_frontier::owner_binding_v0 &owner) {
  return UINT64_C(0xff00000000000000) +
         static_cast<uint64_t>(owner.owner_hw_sid) * UINT64_C(0x1000000) +
         static_cast<uint64_t>(owner.private_slot_id) *
             private_frontier::kPrivateDataSlotBytes;
}

bool receipt_valid(
    const stack_operation::reservation_receipt_v0 &reservation) {
  return reservation.valid == 1 &&
         reservation.reservation_id != 0 &&
         reservation.reservation_age != 0 &&
         reservation.target_operation_seq != 0 &&
         reservation.source_node_operation_seq != 0 &&
         reservation.target_operation_seq !=
             reservation.source_node_operation_seq &&
         reservation.slot_generation != 0 &&
         reservation.slot_index < stack_operation::kMaxSlots &&
         reservation.metadata_chunk_count ==
             stack_operation::kFrontierMetadataReadChunks &&
         reservation.owner.request_identity != 0 &&
         reservation.owner.generation != 0 &&
         reservation.owner.resident_warp_id < 8 &&
         reservation.owner.private_slot_id < 256 &&
         reservation.owner.lane_id < 32 &&
         bytes_are_zero(reservation.owner.reserved_zero,
                        sizeof(reservation.owner.reserved_zero)) &&
         reservation.reserved_zero == 0;
}

bool request_valid(
    const rtcore_memory_unit_request_snapshot &request) {
  const rtcore_v04_stack_private_read_transport_snapshot &extension =
      request.v04_stack_private_read;
  private_frontier::owner_binding_v0 owner = {};
  owner.owner_hw_sid = request.owner_hw_sid;
  owner.resident_warp_id = request.resident_warp_id;
  owner.request_identity = request.rt_request_id;
  owner.generation = request.request_generation;
  owner.private_slot_id = request.private_slot_id;
  owner.lane_id = request.lane_id;
  const uint64_t expected_address =
      private_slot_base(owner) + extension.slot_chunk_offset;
  return request.valid &&
         request.address_space == RTCORE_MEMORY_ADDRESS_SPACE_SHARED &&
         request.operation == RTCORE_MEMORY_OPERATION_READ &&
         request.destination ==
             RTCORE_MEMORY_DESTINATION_STACK_QUEUE_FILL &&
         request.response_target == kResponseTargetRtcore &&
         request.rt_request_id != 0 && request.lane_id < 32 &&
         request.resident_warp_id < 8 &&
         request.request_generation != 0 &&
         request.private_slot_id < 256 &&
         request.memory_op_seq == request.chunk_id + 1 &&
         request.chunk_count ==
             stack_operation::kFrontierMetadataReadChunks &&
         request.chunk_id < request.chunk_count &&
         request.access_kind ==
             RTCORE_MEMORY_ACCESS_STACK_PRIVATE_READ &&
         (request.aligned_32b_addr &
          (private_frontier::kSharedAccessChunkBytes - 1)) == 0 &&
         request.aligned_32b_addr == expected_address &&
         request.byte_mask != 0 && !request.is_write &&
         extension.valid == 1 &&
         extension.reservation_id != 0 &&
         extension.reservation_age != 0 &&
         extension.target_operation_seq != 0 &&
         extension.source_node_operation_seq != 0 &&
         extension.target_operation_seq !=
             extension.source_node_operation_seq &&
         extension.target_slot_generation != 0 &&
         extension.target_slot_index < stack_operation::kMaxSlots &&
         extension.field_kind ==
             private_frontier::kFieldFrontierMetadata &&
         bytes_are_zero(extension.reserved_zero,
                        sizeof(extension.reserved_zero));
}

stack_operation::reservation_receipt_v0 reconstruct_reservation(
    const rtcore_memory_unit_request_snapshot &request) {
  const rtcore_v04_stack_private_read_transport_snapshot &extension =
      request.v04_stack_private_read;
  stack_operation::reservation_receipt_v0 reservation = {};
  reservation.owner.owner_hw_sid = request.owner_hw_sid;
  reservation.owner.resident_warp_id = request.resident_warp_id;
  reservation.owner.request_identity = request.rt_request_id;
  reservation.owner.generation = request.request_generation;
  reservation.owner.private_slot_id = request.private_slot_id;
  reservation.owner.lane_id = request.lane_id;
  reservation.reservation_id = extension.reservation_id;
  reservation.reservation_age = extension.reservation_age;
  reservation.target_operation_seq =
      extension.target_operation_seq;
  reservation.source_node_operation_seq =
      extension.source_node_operation_seq;
  reservation.slot_generation =
      extension.target_slot_generation;
  reservation.slot_index = extension.target_slot_index;
  reservation.metadata_chunk_count =
      stack_operation::kFrontierMetadataReadChunks;
  reservation.valid = 1;
  return reservation;
}

}  // namespace

status_kind prepare_request_plan(
    const stack_operation::reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan) {
  if (request_plan == NULL) return kStatusInvalidArgument;
  *request_plan = request_plan_v0();
  if (!receipt_valid(reservation) ||
      !private_frontier::owners_equal(reservation.owner,
                                      read_plan.owner) ||
      read_plan.access_count !=
          stack_operation::kFrontierMetadataReadChunks) {
    return kStatusMalformedPlan;
  }
  const uint64_t slot_base = private_slot_base(reservation.owner);
  for (unsigned index = 0; index < read_plan.access_count; ++index) {
    const private_frontier::shared_chunk_access_v0 &access =
        read_plan.accesses[index];
    if (access.access_kind != private_frontier::kAccessRead ||
        access.field_kind !=
            private_frontier::kFieldFrontierMetadata ||
        access.aligned_32b_address < slot_base ||
        access.aligned_32b_address - slot_base > UINT16_MAX ||
        access.byte_mask == 0) {
      return kStatusMalformedPlan;
    }
    rtcore_memory_unit_request_snapshot &request =
        request_plan->requests[index];
    request.valid = true;
    request.address_space = RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
    request.operation = RTCORE_MEMORY_OPERATION_READ;
    request.destination =
        RTCORE_MEMORY_DESTINATION_STACK_QUEUE_FILL;
    request.response_target = kResponseTargetRtcore;
    request.owner_hw_sid = reservation.owner.owner_hw_sid;
    request.rt_request_id = reservation.owner.request_identity;
    request.lane_id = reservation.owner.lane_id;
    request.resident_warp_id = reservation.owner.resident_warp_id;
    request.request_generation = reservation.owner.generation;
    request.private_slot_id = reservation.owner.private_slot_id;
    request.memory_op_seq = index + 1;
    request.chunk_id = index;
    request.chunk_count = read_plan.access_count;
    request.access_kind =
        RTCORE_MEMORY_ACCESS_STACK_PRIVATE_READ;
    request.aligned_32b_addr = access.aligned_32b_address;
    request.byte_mask = access.byte_mask;
    request.is_write = false;
    request.issue_cycle = issue_cycle;

    rtcore_v04_stack_private_read_transport_snapshot &extension =
        request.v04_stack_private_read;
    extension.reservation_id = reservation.reservation_id;
    extension.reservation_age = reservation.reservation_age;
    extension.target_operation_seq =
        reservation.target_operation_seq;
    extension.source_node_operation_seq =
        reservation.source_node_operation_seq;
    extension.target_slot_generation =
        reservation.slot_generation;
    extension.slot_chunk_offset = static_cast<uint16_t>(
        access.aligned_32b_address - slot_base);
    extension.target_slot_index = reservation.slot_index;
    extension.field_kind = access.field_kind;
    extension.valid = 1;
    if (!request_valid(request)) return kStatusMalformedPlan;
  }
  request_plan->request_count = read_plan.access_count;
  request_plan->valid = 1;
  return kStatusOk;
}

status_kind accept_request_and_fill(
    const private_shared::backing_state_v0 &backing,
    stack_operation::engine_state_v0 *stack_state,
    const rtcore_memory_unit_request_snapshot &request) {
  if (stack_state == NULL || !request_valid(request)) {
    return kStatusMalformedTransport;
  }
  private_frontier::shared_chunk_access_v0 access = {};
  access.aligned_32b_address = request.aligned_32b_addr;
  access.byte_mask = request.byte_mask;
  access.slot_byte_offset =
      request.v04_stack_private_read.slot_chunk_offset;
  access.field_kind = request.v04_stack_private_read.field_kind;
  access.access_kind = private_frontier::kAccessRead;
  uint8_t payload[private_frontier::kSharedAccessChunkBytes] = {};
  const stack_operation::reservation_receipt_v0 reservation =
      reconstruct_reservation(request);
  if (!receipt_valid(reservation) ||
      private_shared::read_canonical_chunk(
          backing, reservation.owner, access, payload) !=
          private_shared::kStatusOk) {
    return kStatusPrivateReadRejected;
  }
  return stack_operation::fill_frontier_metadata_chunk(
             stack_state, reservation,
             static_cast<uint8_t>(request.chunk_id),
             static_cast<uint8_t>(request.chunk_count),
             request.v04_stack_private_read.slot_chunk_offset,
             request.byte_mask, payload) ==
             stack_operation::kStatusOk
         ? kStatusOk
         : kStatusStackFillRejected;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusMalformedPlan:
      return "malformed_plan";
    case kStatusMalformedTransport:
      return "malformed_transport";
    case kStatusPrivateReadRejected:
      return "private_read_rejected";
    case kStatusStackFillRejected:
      return "stack_fill_rejected";
  }
  return "unknown";
}

}  // namespace stack_private_shared
}  // namespace v04
}  // namespace rtcore
