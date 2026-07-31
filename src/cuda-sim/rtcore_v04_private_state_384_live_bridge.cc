#include "rtcore_v04_private_state_384_live_bridge.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace live_bridge {
namespace {

static const uint64_t kPrivate384Base = UINT64_C(0xfe00000000000000);
static const uint64_t kOwnerStride = UINT64_C(0x1000000);
static const uint8_t kResponseTargetRtcore = 1;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool owner_valid(const private_frontier::owner_binding_v0 &owner) {
  return owner.request_identity != 0 && owner.generation != 0 &&
         owner.resident_warp_id < backing::kResidentWarpCapacity &&
         owner.private_slot_id < backing::kPrivateSlotCapacity &&
         owner.lane_id < backing::kLaneCapacity &&
         bytes_are_zero(owner.reserved_zero,
                        sizeof(owner.reserved_zero));
}

operand_materializer::operation_identity_v1 make_identity(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_sequence) {
  operand_materializer::operation_identity_v1 identity = {};
  identity.owner_hw_sid = owner.owner_hw_sid;
  identity.resident_warp_id = owner.resident_warp_id;
  identity.request_identity = owner.request_identity;
  identity.request_generation = owner.generation;
  identity.private_slot_id = owner.private_slot_id;
  identity.operation_sequence = operation_sequence;
  identity.lane_id = owner.lane_id;
  return identity;
}

bool destination_valid(uint8_t destination) {
  return destination ==
             RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL ||
         destination ==
             RTCORE_MEMORY_DESTINATION_STACK_QUEUE_FILL ||
         destination ==
             RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL ||
         destination == RTCORE_MEMORY_DESTINATION_LEGACY;
}

bool request_shape_valid(
    const rtcore_memory_unit_request_snapshot &request,
    operand_plan::read_plan_v1 *canonical_plan,
    operand_materializer::operation_identity_v1 *identity) {
  if (canonical_plan == NULL || identity == NULL) return false;
  *canonical_plan = operand_plan::read_plan_v1();
  *identity = operand_materializer::operation_identity_v1();
  const rtcore_v04_private_state_384_read_transport_snapshot &transport =
      request.v04_private_state_384_read;
  operand_plan::read_request_v1 read_request = {};
  read_request.private_layout_profile_id =
      transport.private_layout_profile_id;
  read_request.consumer = transport.consumer;
  read_request.operation = transport.operation_kind;
  read_request.completion_reason = transport.completion_reason;
  if (operand_plan::make_read_plan(read_request, canonical_plan) !=
      operand_plan::kStatusOk) {
    return false;
  }
  if (!request.valid ||
      request.address_space != RTCORE_MEMORY_ADDRESS_SPACE_SHARED ||
      request.operation != RTCORE_MEMORY_OPERATION_READ ||
      !destination_valid(static_cast<uint8_t>(request.destination)) ||
      request.response_target != kResponseTargetRtcore ||
      request.rt_request_id == 0 ||
      request.resident_warp_id >= backing::kResidentWarpCapacity ||
      request.request_generation == 0 ||
      request.private_slot_id >= backing::kPrivateSlotCapacity ||
      request.lane_id >= backing::kLaneCapacity ||
      request.access_kind !=
          RTCORE_MEMORY_ACCESS_PRIVATE_STATE_384_READ ||
      request.is_write || request.byte_mask != backing::kFullChunkByteMask ||
      transport.valid != 1 ||
      !bytes_are_zero(transport.reserved_zero,
                      sizeof(transport.reserved_zero)) ||
      transport.operation_sequence == 0 ||
      transport.read_count != canonical_plan->read_count ||
      transport.read_index >= transport.read_count ||
      request.chunk_id != transport.read_index ||
      request.chunk_count != transport.read_count ||
      request.memory_op_seq !=
          static_cast<unsigned>(transport.read_index) + 1u) {
    return false;
  }
  const operand_plan::chunk_read_v1 &read =
      canonical_plan->reads[transport.read_index];
  private_frontier::owner_binding_v0 owner = {};
  owner.owner_hw_sid = request.owner_hw_sid;
  owner.resident_warp_id =
      static_cast<uint8_t>(request.resident_warp_id);
  owner.request_identity = request.rt_request_id;
  owner.generation = request.request_generation;
  owner.private_slot_id = request.private_slot_id;
  owner.lane_id = static_cast<uint8_t>(request.lane_id);
  if (!owner_valid(owner) ||
      request.aligned_32b_addr != private_slot_base(owner) +
                                         read.slot_byte_offset) {
    return false;
  }
  *identity = make_identity(owner, transport.operation_sequence);
  return true;
}

}  // namespace

uint64_t private_slot_base(
    const private_frontier::owner_binding_v0 &owner) {
  return kPrivate384Base +
         static_cast<uint64_t>(owner.owner_hw_sid) * kOwnerStride +
         static_cast<uint64_t>(owner.private_slot_id) * kSlotBytes;
}

status_kind prepare_read_requests(const read_input_v1 &input,
                                  read_request_plan_v1 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = read_request_plan_v1();
  if (!owner_valid(input.owner)) return kStatusInvalidOwner;
  if (input.operation_sequence == 0) return kStatusInvalidOperation;
  if (!destination_valid(input.destination)) {
    return kStatusInvalidDestination;
  }
  operand_plan::read_request_v1 request = {};
  request.private_layout_profile_id = kPrivateLayoutProfileId;
  request.consumer = input.consumer;
  request.operation = input.operation;
  request.completion_reason = input.completion_reason;
  read_request_plan_v1 prepared = {};
  if (operand_plan::make_read_plan(request, &prepared.operand_plan) !=
      operand_plan::kStatusOk) {
    return kStatusInvalidOperation;
  }
  prepared.identity =
      make_identity(input.owner, input.operation_sequence);
  for (uint8_t index = 0;
       index < prepared.operand_plan.read_count; ++index) {
    const operand_plan::chunk_read_v1 &read =
        prepared.operand_plan.reads[index];
    rtcore_memory_unit_request_snapshot &memory =
        prepared.requests[index];
    memory.valid = true;
    memory.address_space = RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
    memory.operation = RTCORE_MEMORY_OPERATION_READ;
    memory.destination = input.destination;
    memory.response_target = kResponseTargetRtcore;
    memory.owner_hw_sid = input.owner.owner_hw_sid;
    memory.rt_request_id = input.owner.request_identity;
    memory.lane_id = input.owner.lane_id;
    memory.resident_warp_id = input.owner.resident_warp_id;
    memory.request_generation = input.owner.generation;
    memory.private_slot_id = input.owner.private_slot_id;
    memory.memory_op_seq = static_cast<unsigned>(index) + 1u;
    memory.chunk_id = index;
    memory.chunk_count = prepared.operand_plan.read_count;
    memory.access_kind =
        RTCORE_MEMORY_ACCESS_PRIVATE_STATE_384_READ;
    memory.aligned_32b_addr =
        private_slot_base(input.owner) + read.slot_byte_offset;
    memory.byte_mask = backing::kFullChunkByteMask;
    memory.is_write = false;
    memory.issue_cycle = input.issue_cycle;
    rtcore_v04_private_state_384_read_transport_snapshot &transport =
        memory.v04_private_state_384_read;
    transport.operation_sequence = input.operation_sequence;
    transport.private_layout_profile_id = kPrivateLayoutProfileId;
    transport.consumer = input.consumer;
    transport.operation_kind = input.operation;
    transport.completion_reason = input.completion_reason;
    transport.read_index = index;
    transport.read_count = prepared.operand_plan.read_count;
    transport.valid = 1;
    operand_plan::read_plan_v1 checked_plan = {};
    operand_materializer::operation_identity_v1 checked_identity = {};
    if (!request_shape_valid(memory, &checked_plan, &checked_identity) ||
        std::memcmp(&checked_plan, &prepared.operand_plan,
                    sizeof(checked_plan)) != 0 ||
        std::memcmp(&checked_identity, &prepared.identity,
                    sizeof(checked_identity)) != 0) {
      return kStatusMalformedTransport;
    }
  }
  prepared.request_count = prepared.operand_plan.read_count;
  prepared.valid = 1;
  *plan = prepared;
  return kStatusOk;
}

status_kind initialize_collector(
    const read_request_plan_v1 &plan,
    operand_materializer::response_collector_v1 *collector) {
  if (collector == NULL || plan.valid != 1 ||
      plan.request_count != plan.operand_plan.read_count) {
    return kStatusInvalidArgument;
  }
  return operand_materializer::initialize_collector(
             plan.operand_plan, plan.identity, collector) ==
             operand_materializer::kStatusOk
         ? kStatusOk
         : kStatusCollectorRejected;
}

status_kind accept_read_response(
    const backing::state_v1 &state,
    const rtcore_memory_unit_request_snapshot &request,
    operand_materializer::response_collector_v1 *collector) {
  if (collector == NULL) return kStatusInvalidArgument;
  operand_plan::read_plan_v1 plan = {};
  operand_materializer::operation_identity_v1 identity = {};
  if (!request_shape_valid(request, &plan, &identity)) {
    return kStatusMalformedTransport;
  }
  operand_materializer::chunk_response_v1 response = {};
  if (backing::prepare_read_response(
          state, plan, identity,
          request.v04_private_state_384_read.read_index,
          &response) != backing::kStatusOk) {
    return kStatusBackingRejected;
  }
  return operand_materializer::accept_response(response, collector) ==
             operand_materializer::kStatusOk
         ? kStatusOk
         : kStatusCollectorRejected;
}

status_kind commit_sparse_deltas_after_ack(
    backing::state_v1 *state, const write_commit_input_v1 &input,
    const operand_plan::chunk_delta_v1 *deltas, size_t delta_count) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!owner_valid(input.owner) ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidOwner;
  }
  if (input.operation_sequence == 0) return kStatusInvalidOperation;
  if (input.matching_write_ack != 1) return kStatusWriteAckRequired;
  operand_plan::write_request_v1 request = {};
  request.private_layout_profile_id = kPrivateLayoutProfileId;
  request.producer = input.producer;
  return backing::apply_sparse_deltas(
             state, make_identity(input.owner,
                                  input.operation_sequence),
             request, deltas, delta_count) == backing::kStatusOk
         ? kStatusOk
         : kStatusWriteRejected;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidOwner:
      return "invalid_owner";
    case kStatusInvalidOperation:
      return "invalid_operation";
    case kStatusInvalidDestination:
      return "invalid_destination";
    case kStatusMalformedTransport:
      return "malformed_transport";
    case kStatusCollectorRejected:
      return "collector_rejected";
    case kStatusBackingRejected:
      return "backing_rejected";
    case kStatusWriteAckRequired:
      return "write_ack_required";
    case kStatusWriteRejected:
      return "write_rejected";
  }
  return "unknown";
}

}  // namespace live_bridge
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore
