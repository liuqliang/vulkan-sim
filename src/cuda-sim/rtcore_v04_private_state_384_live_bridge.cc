#include "rtcore_v04_private_state_384_live_bridge.h"

#include <cstring>
#include <limits>

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

bool operation_key_valid(const live_operation_key_v1 &key) {
  return key.storage_profile ==
             private_storage::kProfileCompressedShared384 &&
         key.private_layout_profile_id == kPrivateLayoutProfileId &&
         key.bvh_format_profile_id == kGenRtBvhFormatProfileId &&
         key.reservation_generation != 0 &&
         key.identity.operation_sequence != 0 &&
         bytes_are_zero(key.identity.reserved_zero,
                        sizeof(key.identity.reserved_zero));
}

bool owner_valid(const private_frontier::owner_binding_v0 &owner) {
  return owner.request_identity != 0 && owner.generation != 0 &&
         owner.resident_warp_id < backing::kResidentWarpCapacity &&
         owner.private_slot_id < backing::kPrivateSlotCapacity &&
         owner.lane_id < backing::kLaneCapacity &&
         bytes_are_zero(owner.reserved_zero,
                        sizeof(owner.reserved_zero));
}

bool bridge_owners_equal(
    const private_frontier::owner_binding_v0 &left,
    const private_frontier::owner_binding_v0 &right) {
  return owner_valid(left) && owner_valid(right) &&
         left.owner_hw_sid == right.owner_hw_sid &&
         left.resident_warp_id == right.resident_warp_id &&
         left.request_identity == right.request_identity &&
         left.generation == right.generation &&
         left.private_slot_id == right.private_slot_id &&
         left.lane_id == right.lane_id;
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

private_frontier::owner_binding_v0 make_owner(
    const operand_materializer::operation_identity_v1 &identity) {
  private_frontier::owner_binding_v0 owner = {};
  owner.owner_hw_sid = identity.owner_hw_sid;
  owner.resident_warp_id = identity.resident_warp_id;
  owner.request_identity = identity.request_identity;
  owner.generation = identity.request_generation;
  owner.private_slot_id = identity.private_slot_id;
  owner.lane_id = identity.lane_id;
  return owner;
}

bool destination_valid(uint8_t destination) {
  return destination ==
             RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL ||
         destination ==
             RTCORE_MEMORY_DESTINATION_STACK_QUEUE_FILL ||
         destination ==
             RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL ||
         destination ==
             RTCORE_MEMORY_DESTINATION_PRIVATE_BOUNDARY_FILL ||
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
      transport.operation_sequence == 0 ||
      transport.storage_profile !=
          private_storage::kProfileCompressedShared384 ||
      transport.bvh_format_profile_id !=
          kGenRtBvhFormatProfileId ||
      transport.reservation_generation == 0 ||
      transport.read_count != canonical_plan->read_count ||
      transport.read_index >= transport.read_count ||
      request.chunk_id != transport.read_index ||
      request.chunk_count != transport.read_count ||
      transport.memory_op_seq_base == 0 ||
      request.memory_op_seq !=
          static_cast<unsigned>(transport.memory_op_seq_base) +
              transport.read_index) {
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
  if (!owner_valid(input.owner) ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidOwner;
  }
  if (input.storage_profile !=
          private_storage::kProfileCompressedShared384 ||
      input.bvh_format_profile_id != kGenRtBvhFormatProfileId ||
      input.reservation_generation == 0) {
    return kStatusInvalidProfile;
  }
  if (input.operation_sequence == 0) return kStatusInvalidOperation;
  if (!destination_valid(input.destination)) {
    return kStatusInvalidDestination;
  }
  if (input.memory_op_seq_base == 0 ||
      static_cast<unsigned>(input.memory_op_seq_base) +
              operand_materializer::kMaxOperationReadChunks - 1u >
          std::numeric_limits<uint8_t>::max()) {
    return kStatusInvalidOperation;
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
  prepared.key.identity = prepared.identity;
  prepared.key.private_layout_profile_id = kPrivateLayoutProfileId;
  prepared.key.bvh_format_profile_id = input.bvh_format_profile_id;
  prepared.key.reservation_generation = input.reservation_generation;
  prepared.key.storage_profile = input.storage_profile;
  prepared.key.consumer = input.consumer;
  prepared.key.operation = input.operation;
  prepared.key.completion_reason = input.completion_reason;
  if (!operation_key_valid(prepared.key)) {
    return kStatusInvalidProfile;
  }
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
    memory.memory_op_seq =
        static_cast<unsigned>(input.memory_op_seq_base) + index;
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
    transport.bvh_format_profile_id = input.bvh_format_profile_id;
    transport.reservation_generation = input.reservation_generation;
    transport.consumer = input.consumer;
    transport.operation_kind = input.operation;
    transport.completion_reason = input.completion_reason;
    transport.read_index = index;
    transport.read_count = prepared.operand_plan.read_count;
    transport.storage_profile = input.storage_profile;
    transport.valid = 1;
    transport.memory_op_seq_base = input.memory_op_seq_base;
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
      plan.request_count != plan.operand_plan.read_count ||
      !operation_key_valid(plan.key) ||
      std::memcmp(&plan.key.identity, &plan.identity,
                  sizeof(plan.identity)) != 0) {
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

status_kind stage_sparse_commit(
    const write_commit_input_v1 &input,
    const sparse_chunk_delta_v1 *deltas, size_t delta_count,
    pending_sparse_commit_v1 *pending) {
  if (pending == NULL) return kStatusInvalidArgument;
  *pending = pending_sparse_commit_v1();
  if (!owner_valid(input.owner) ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidOwner;
  }
  if (input.operation_sequence == 0 || input.commit_epoch == 0 ||
      input.expected_write_ack_count == 0 ||
      input.expected_write_ack_count > 16) {
    return kStatusInvalidCommit;
  }
  if (input.storage_profile !=
          private_storage::kProfileCompressedShared384 ||
      input.bvh_format_profile_id != kGenRtBvhFormatProfileId) {
    return kStatusInvalidProfile;
  }
  operand_plan::write_request_v1 request = {};
  request.private_layout_profile_id = kPrivateLayoutProfileId;
  request.producer = input.producer;
  operand_plan::unit_sparse_write_plan_v1 merged = {};
  if (operand_plan::merge_sparse_writes(
          request, deltas, delta_count, &merged) !=
          operand_plan::kStatusOk ||
      merged.write_count == 0) {
    return kStatusWriteRejected;
  }
  pending->key.identity =
      make_identity(input.owner, input.operation_sequence);
  pending->key.private_layout_profile_id = kPrivateLayoutProfileId;
  pending->key.bvh_format_profile_id = input.bvh_format_profile_id;
  pending->key.reservation_generation = input.commit_epoch;
  pending->key.storage_profile = input.storage_profile;
  pending->key.consumer = operand_plan::kConsumerInvalid;
  pending->key.operation = operand_plan::kOperationDefault;
  pending->key.completion_reason =
      operand_plan::kCompletionReasonNone;
  pending->merged_write_plan = merged;
  pending->commit_epoch = input.commit_epoch;
  pending->expected_ack_mask = static_cast<uint16_t>(
      input.expected_write_ack_count == 16
          ? 0xffffu
          : (uint16_t{1} << input.expected_write_ack_count) - 1u);
  pending->producer = input.producer;
  pending->expected_write_ack_count =
      static_cast<uint8_t>(input.expected_write_ack_count);
  pending->valid = 1;
  return kStatusOk;
}

status_kind register_modeled_write(
    const private_shared::shared_write_v0 &write,
    pending_sparse_commit_v1 *pending) {
  if (pending == NULL || pending->valid != 1 ||
      pending->committed != 0 || !operation_key_valid(pending->key) ||
      !write.valid ||
      !bytes_are_zero(write.reserved_zero,
                      sizeof(write.reserved_zero)) ||
      write.reserved_zero1 != 0 ||
      write.address_space != private_shared::kAddressSpaceShared ||
      write.address_mode != private_shared::kAddressModePrivateField ||
      write.access_operation != private_shared::kAccessOperationWrite ||
      write.destination != private_shared::kDestinationPrivateCommitAck ||
      !bridge_owners_equal(
          write.owner, make_owner(pending->key.identity)) ||
      write.operation_seq !=
          pending->key.identity.operation_sequence ||
      write.commit_epoch != pending->commit_epoch ||
      write.memory_op_seq == 0 || write.memory_op_seq > 16 ||
      write.chunk_count != pending->expected_write_ack_count ||
      write.chunk_id >= write.chunk_count ||
      write.memory_op_seq != static_cast<uint32_t>(write.chunk_id) + 1u ||
      write.field_kind == 0 || write.aligned_32b_address == 0 ||
      (write.aligned_32b_address % kChunkBytes) != 0 ||
      write.byte_mask == 0) {
    return kStatusInvalidCommit;
  }
  const uint16_t write_bit = static_cast<uint16_t>(
      uint16_t{1} << (write.memory_op_seq - 1u));
  if ((pending->expected_ack_mask & write_bit) == 0 ||
      (pending->registered_ack_mask & write_bit) != 0) {
    return kStatusInvalidCommit;
  }
  pending_sparse_commit_v1::expected_write_ack_v1 &expected =
      pending->expected_writes[write.memory_op_seq - 1u];
  expected.aligned_32b_address = write.aligned_32b_address;
  expected.byte_mask = write.byte_mask;
  expected.memory_operation_seq =
      static_cast<uint16_t>(write.memory_op_seq);
  expected.chunk_id = write.chunk_id;
  expected.chunk_count = write.chunk_count;
  expected.field_kind = write.field_kind;
  expected.valid = 1;
  std::memcpy(expected.payload, write.payload,
              sizeof(expected.payload));
  pending->registered_ack_mask = static_cast<uint16_t>(
      pending->registered_ack_mask | write_bit);
  return kStatusOk;
}

status_kind accept_write_ack_and_maybe_commit(
    backing::state_v1 *state,
    const private_shared::shared_write_v0 &write,
    const private_shared::runtime_write_ack_v0 &ack,
    pending_sparse_commit_v1 *pending, bool *canonical_committed) {
  if (state == NULL || pending == NULL || canonical_committed == NULL) {
    return kStatusInvalidArgument;
  }
  *canonical_committed = false;
  if (pending->valid != 1 || pending->committed != 0 ||
      !operation_key_valid(pending->key) ||
      pending->producer == operand_plan::kProducerInvalid ||
      pending->commit_epoch == 0 ||
      pending->expected_ack_mask == 0 ||
      (pending->registered_ack_mask &
       ~pending->expected_ack_mask) != 0 ||
      (pending->acknowledged_ack_mask &
       ~pending->expected_ack_mask) != 0) {
    return kStatusInvalidCommit;
  }
  if (!ack.valid ||
      !bytes_are_zero(ack.reserved_zero,
                      sizeof(ack.reserved_zero)) ||
      !bridge_owners_equal(
          ack.owner, make_owner(pending->key.identity)) ||
      ack.operation_seq !=
          pending->key.identity.operation_sequence ||
      ack.commit_epoch != pending->commit_epoch ||
      ack.memory_operation_seq == 0 ||
      ack.memory_operation_seq > 16) {
    return kStatusInvalidWriteAck;
  }
  const uint16_t ack_bit = static_cast<uint16_t>(
      uint16_t{1} << (ack.memory_operation_seq - 1u));
  if ((pending->expected_ack_mask & ack_bit) == 0 ||
      (pending->registered_ack_mask & ack_bit) == 0) {
    return kStatusInvalidWriteAck;
  }
  if ((pending->acknowledged_ack_mask & ack_bit) != 0) {
    return kStatusDuplicateWriteAck;
  }
  const pending_sparse_commit_v1::expected_write_ack_v1 &expected =
      pending->expected_writes[ack.memory_operation_seq - 1u];
  if (expected.valid != 1 ||
      expected.memory_operation_seq != ack.memory_operation_seq ||
      expected.field_kind != ack.field_kind ||
      !write.valid ||
      write.address_space != private_shared::kAddressSpaceShared ||
      write.address_mode != private_shared::kAddressModePrivateField ||
      write.access_operation != private_shared::kAccessOperationWrite ||
      write.destination != private_shared::kDestinationPrivateCommitAck ||
      !bridge_owners_equal(write.owner, ack.owner) ||
      write.operation_seq != ack.operation_seq ||
      write.commit_epoch != ack.commit_epoch ||
      write.memory_op_seq != ack.memory_operation_seq ||
      write.chunk_id != expected.chunk_id ||
      write.chunk_count != expected.chunk_count ||
      write.field_kind != ack.field_kind ||
      write.aligned_32b_address != expected.aligned_32b_address ||
      write.byte_mask != expected.byte_mask ||
      std::memcmp(write.payload, expected.payload,
                  sizeof(expected.payload)) != 0) {
    return kStatusInvalidWriteAck;
  }
  pending->acknowledged_ack_mask = static_cast<uint16_t>(
      pending->acknowledged_ack_mask | ack_bit);
  if (pending->acknowledged_ack_mask !=
      pending->expected_ack_mask) {
    return kStatusOk;
  }
  if (pending->registered_ack_mask != pending->expected_ack_mask) {
    return kStatusInvalidCommit;
  }

  operand_plan::chunk_delta_v1 deltas[kChunkCount] = {};
  for (uint8_t index = 0;
       index < pending->merged_write_plan.write_count; ++index) {
    const chunk_write_v1 &write =
        pending->merged_write_plan.writes[index];
    if (write.byte_count != kChunkBytes ||
        (write.slot_byte_offset % kChunkBytes) != 0 ||
        write.slot_byte_offset >= kSlotBytes) {
      return kStatusWriteRejected;
    }
    operand_plan::chunk_delta_v1 &delta = deltas[index];
    delta.chunk_index = static_cast<uint8_t>(
        write.slot_byte_offset / kChunkBytes);
    delta.byte_mask = write.byte_mask;
    std::memcpy(delta.payload, write.payload, sizeof(delta.payload));
  }
  operand_plan::write_request_v1 request = {};
  request.private_layout_profile_id = kPrivateLayoutProfileId;
  request.producer = pending->producer;
  if (backing::apply_sparse_deltas(
          state, pending->key.identity, request, deltas,
          pending->merged_write_plan.write_count) !=
      backing::kStatusOk) {
    return kStatusWriteRejected;
  }
  pending->committed = 1;
  *canonical_committed = true;
  return kStatusOk;
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
    case kStatusInvalidProfile:
      return "invalid_profile";
    case kStatusInvalidCommit:
      return "invalid_commit";
    case kStatusInvalidWriteAck:
      return "invalid_write_ack";
    case kStatusDuplicateWriteAck:
      return "duplicate_write_ack";
    case kStatusWriteRejected:
      return "write_rejected";
  }
  return "unknown";
}

}  // namespace live_bridge
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore
