#include "rtcore_v04_live_global_memory_adapter.h"

#include <cstring>
#include <type_traits>

#include "memory.h"
#include "rtcore_v04_target_shared_memory_bridge.h"

namespace rtcore {
namespace v04 {
namespace live_global_memory {
namespace {

static const unsigned kResponseTargetRtcore = 1;
static const unsigned kFullByteMask = 0xffffffffu;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool common_transport_shape_valid(
    const rtcore_memory_unit_request_snapshot &request) {
  const rtcore_v04_target_raw_read_transport_snapshot &extension =
      request.v04_target_raw_read;
  const bool producer_tag_valid =
      extension.producer_commit_required == 0
          ? extension.producer_operation_seq == 0 &&
                extension.producer_commit_epoch == 0
          : extension.producer_operation_seq != 0 &&
                extension.producer_commit_epoch != 0;
  return request.valid &&
         request.address_space == RTCORE_MEMORY_ADDRESS_SPACE_GLOBAL &&
         request.operation == RTCORE_MEMORY_OPERATION_READ &&
         request.destination ==
             RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL &&
         request.response_target == kResponseTargetRtcore &&
         request.rt_request_id != 0 && request.lane_id < 32 &&
         request.resident_warp_id < 8 &&
         request.request_generation != 0 &&
         request.private_slot_id < 256 &&
         request.chunk_count != 0 &&
         request.chunk_count <= target_memory::kMaxRawReadChunks &&
         request.chunk_id < request.chunk_count &&
         request.memory_op_seq ==
             static_cast<unsigned>(request.chunk_id) + 1 &&
         request.access_kind == RTCORE_MEMORY_ACCESS_TARGET_RAW_READ &&
         (request.aligned_32b_addr &
          (target_memory::kRawReadChunkBytes - 1)) == 0 &&
         request.byte_mask == kFullByteMask && !request.is_write &&
         extension.valid == 1 && extension.reservation_id != 0 &&
         extension.reservation_age != 0 &&
         extension.raw_payload_base_address != 0 &&
         extension.target_operation_seq != 0 &&
         producer_tag_valid &&
         extension.target_slot_generation != 0 &&
         extension.private_layout_profile_id != 0 &&
         extension.bvh_format_profile_id ==
             typed_node::kGenRtDerivedProfileId &&
         extension.private_storage_profile <=
             private_storage::kProfileCompressedShared384 &&
         extension.producer_commit_required <= 1 &&
         extension.operand_kind ==
             RTCORE_MEMORY_TARGET_OPERAND_RAW_GLOBAL &&
         extension.field_kind == 0 &&
         extension.private_chunk_count <=
             target_shared_memory::kMaxPrivateReadChunks &&
         extension.transfer_bytes ==
             target_memory::kRawReadChunkBytes &&
         bytes_are_zero(extension.reserved_zero,
                        sizeof(extension.reserved_zero));
}

}  // namespace

static_assert(
    std::is_standard_layout<
        rtcore_v04_target_raw_read_transport_snapshot>::value,
    "target raw-read transport extension must be standard layout");
static_assert(
    std::is_trivially_copyable<
        rtcore_v04_target_raw_read_transport_snapshot>::value,
    "target raw-read transport extension must be trivially copyable");
static_assert(
    std::is_standard_layout<rtcore_memory_unit_request_snapshot>::value,
    "memory-unit request snapshot must remain standard layout");
static_assert(
    std::is_trivially_copyable<rtcore_memory_unit_request_snapshot>::value,
    "memory-unit request snapshot must remain trivially copyable");
static_assert(
    std::is_standard_layout<target_memory::raw_read_chunk_v0>::value,
    "target raw-read descriptor must be standard layout");
static_assert(
    std::is_trivially_copyable<target_memory::raw_read_chunk_v0>::value,
    "target raw-read descriptor must be trivially copyable");

status_kind lower_raw_read_chunk(
    const target_memory::raw_read_chunk_v0 &chunk,
    uint64_t issue_cycle,
    rtcore_memory_unit_request_snapshot *request) {
  if (request == NULL) return kStatusInvalidArgument;
  *request = rtcore_memory_unit_request_snapshot();
  if (target_memory::validate_raw_read_chunk(chunk) !=
      target_memory::kStatusOk) {
    return kStatusMalformedRawRead;
  }

  const fetch_target::reservation_receipt_v0 &reservation =
      chunk.reservation;
  request->valid = true;
  request->address_space = RTCORE_MEMORY_ADDRESS_SPACE_GLOBAL;
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
  request->memory_op_seq = static_cast<unsigned>(chunk.chunk_id) + 1;
  request->chunk_id = chunk.chunk_id;
  request->chunk_count = chunk.chunk_count;
  request->access_kind = RTCORE_MEMORY_ACCESS_TARGET_RAW_READ;
  request->aligned_32b_addr = chunk.aligned_32b_address;
  request->byte_mask = kFullByteMask;
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
  extension.transfer_bytes = chunk.transfer_bytes;
  extension.operand_kind =
      RTCORE_MEMORY_TARGET_OPERAND_RAW_GLOBAL;
  extension.private_chunk_count =
      reservation.private_chunk_count;
  extension.operation_kind = reservation.operation_kind;
  extension.private_storage_profile =
      reservation.private_storage_profile;
  extension.valid = 1;
  return kStatusOk;
}

status_kind reconstruct_raw_read_chunk(
    const rtcore_memory_unit_request_snapshot &request,
    target_memory::raw_read_chunk_v0 *chunk) {
  if (chunk == NULL) return kStatusInvalidArgument;
  *chunk = target_memory::raw_read_chunk_v0();
  if (!common_transport_shape_valid(request)) {
    return kStatusMalformedTransport;
  }

  const rtcore_v04_target_raw_read_transport_snapshot &extension =
      request.v04_target_raw_read;
  fetch_target::reservation_receipt_v0 &reservation =
      chunk->reservation;
  reservation.owner.owner_hw_sid = request.owner_hw_sid;
  reservation.owner.resident_warp_id = request.resident_warp_id;
  reservation.owner.request_identity = request.rt_request_id;
  reservation.owner.generation = request.request_generation;
  reservation.owner.private_slot_id = request.private_slot_id;
  reservation.owner.lane_id = request.lane_id;
  reservation.reservation_id = extension.reservation_id;
  reservation.reservation_age = extension.reservation_age;
  reservation.raw_payload_base_address =
      extension.raw_payload_base_address;
  reservation.target_operation_seq =
      extension.target_operation_seq;
  reservation.producer_operation_seq =
      extension.producer_operation_seq;
  reservation.producer_commit_epoch =
      extension.producer_commit_epoch;
  reservation.slot_generation =
      extension.target_slot_generation;
  reservation.private_layout_profile_id =
      extension.private_layout_profile_id;
  reservation.bvh_format_profile_id =
      extension.bvh_format_profile_id;
  reservation.raw_payload_bytes = extension.raw_payload_bytes;
  reservation.target_kind = extension.target_kind;
  reservation.slot_index = extension.target_slot_index;
  reservation.raw_chunk_count =
      static_cast<uint8_t>(request.chunk_count);
  reservation.private_chunk_count =
      extension.private_chunk_count;
  reservation.producer_commit_required =
      extension.producer_commit_required;
  reservation.operation_kind = extension.operation_kind;
  reservation.private_storage_profile =
      extension.private_storage_profile;
  reservation.valid = 1;

  chunk->aligned_32b_address = request.aligned_32b_addr;
  chunk->address_space = target_memory::kAddressSpaceGlobal;
  chunk->operation = target_memory::kOperationRead;
  chunk->destination =
      target_memory::kDestinationTargetQueueFill;
  chunk->target_kind = extension.target_kind;
  chunk->chunk_id = static_cast<uint8_t>(request.chunk_id);
  chunk->chunk_count = static_cast<uint8_t>(request.chunk_count);
  chunk->transfer_bytes = extension.transfer_bytes;
  chunk->valid = 1;
  return target_memory::validate_raw_read_chunk(*chunk) ==
                 target_memory::kStatusOk
             ? kStatusOk
             : kStatusMalformedTransport;
}

status_kind prepare_request_plan(
    const target_memory::raw_read_plan_v0 &raw_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan) {
  if (request_plan == NULL) return kStatusInvalidArgument;
  *request_plan = request_plan_v0();
  if (raw_plan.valid != 1 || raw_plan.chunk_count == 0 ||
      raw_plan.chunk_count > kMaxRequestChunks ||
      !bytes_are_zero(raw_plan.reserved_zero,
                      sizeof(raw_plan.reserved_zero))) {
    return kStatusMalformedRawRead;
  }
  for (unsigned index = 0; index < raw_plan.chunk_count; ++index) {
    const status_kind status =
        lower_raw_read_chunk(raw_plan.chunks[index], issue_cycle,
                             &request_plan->requests[index]);
    if (status != kStatusOk) {
      *request_plan = request_plan_v0();
      return status;
    }
  }
  request_plan->request_count = raw_plan.chunk_count;
  request_plan->valid = 1;
  return kStatusOk;
}

status_kind materialize_functional_response(
    const rtcore_memory_unit_request_snapshot &request,
    memory_space *global_memory, uint64_t response_address,
    uint8_t *response_payload, uint8_t response_bytes) {
  if (global_memory == NULL || response_payload == NULL) {
    return kStatusInvalidArgument;
  }
  target_memory::raw_read_chunk_v0 chunk = {};
  if (reconstruct_raw_read_chunk(request, &chunk) != kStatusOk) {
    return kStatusMalformedTransport;
  }
  if (response_address != request.aligned_32b_addr) {
    return kStatusResponseAddressMismatch;
  }
  if (response_bytes != target_memory::kRawReadChunkBytes) {
    return kStatusResponseSizeMismatch;
  }
  global_memory->read(request.aligned_32b_addr, response_bytes,
                      response_payload);
  return kStatusOk;
}

status_kind accept_materialized_response(
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request,
    const uint8_t *response_payload, uint8_t response_bytes) {
  if (target_state == NULL || response_payload == NULL) {
    return kStatusInvalidArgument;
  }
  target_memory::raw_read_chunk_v0 chunk = {};
  if (reconstruct_raw_read_chunk(request, &chunk) != kStatusOk) {
    return kStatusMalformedTransport;
  }
  return target_memory::accept_raw_read_response(
             target_state, chunk, response_payload, response_bytes) ==
                 target_memory::kStatusOk
             ? kStatusOk
             : kStatusTargetFillRejected;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusMalformedRawRead:
      return "malformed_raw_read";
    case kStatusMalformedTransport:
      return "malformed_transport";
    case kStatusResponseAddressMismatch:
      return "response_address_mismatch";
    case kStatusResponseSizeMismatch:
      return "response_size_mismatch";
    case kStatusTargetFillRejected:
      return "target_fill_rejected";
  }
  return "unknown";
}

}  // namespace live_global_memory
}  // namespace v04
}  // namespace rtcore
