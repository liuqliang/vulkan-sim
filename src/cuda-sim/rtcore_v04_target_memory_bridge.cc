#include "rtcore_v04_target_memory_bridge.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace target_memory {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool expected_shape(const fetch_target::reservation_receipt_v0 &reservation,
                    uint16_t *payload_bytes, uint8_t *chunk_count,
                    uint8_t *slot_capacity) {
  if (payload_bytes == NULL || chunk_count == NULL ||
      slot_capacity == NULL) {
    return false;
  }
  switch (static_cast<fetch_target::target_kind>(
      reservation.target_kind)) {
    case fetch_target::kTargetNode:
      *payload_bytes = fetch_target::kNodeRawPayloadBytes;
      *chunk_count = 2;
      *slot_capacity = fetch_target::kMaxNodeSlots;
      return true;
    case fetch_target::kTargetPrimitive:
      *payload_bytes = fetch_target::kPrimitiveRawPayloadBytes;
      *chunk_count = 2;
      *slot_capacity = fetch_target::kMaxPrimitiveSlots;
      return true;
    case fetch_target::kTargetInstance:
      *payload_bytes = fetch_target::kInstanceRawPayloadBytes;
      *chunk_count = 4;
      *slot_capacity = fetch_target::kMaxInstanceSlots;
      return true;
    case fetch_target::kTargetInvalid:
      return false;
  }
  return false;
}

bool valid_reservation_shape(
    const fetch_target::reservation_receipt_v0 &reservation) {
  uint16_t expected_payload_bytes = 0;
  uint8_t expected_chunk_count = 0;
  uint8_t slot_capacity = 0;
  const bool producer_tag_valid =
      reservation.producer_commit_required == 0
          ? reservation.producer_operation_seq == 0 &&
                reservation.producer_commit_epoch == 0
          : reservation.producer_operation_seq != 0 &&
                reservation.producer_commit_epoch != 0;
  const bool profile_valid =
      reservation.bvh_format_profile_id ==
          typed_node::kGenRtDerivedProfileId &&
      ((reservation.private_storage_profile ==
            private_storage::kProfileLegacyShared832 &&
        reservation.private_layout_profile_id ==
            private_frontier::kLayoutProfileId) ||
       ((reservation.private_storage_profile ==
             private_storage::kProfileCompressedShared384 ||
         reservation.private_storage_profile ==
             private_storage::kProfileGlobal384) &&
        reservation.private_layout_profile_id ==
            private_state_384::kPrivateLayoutProfileId));
  return reservation.valid == 1 && reservation.reservation_id != 0 &&
         reservation.reservation_age != 0 &&
         reservation.raw_payload_base_address != 0 &&
         (reservation.raw_payload_base_address &
          (kRawReadChunkBytes - 1)) == 0 &&
         reservation.target_operation_seq != 0 && producer_tag_valid &&
         profile_valid &&
         reservation.slot_generation != 0 &&
         reservation.owner.request_identity != 0 &&
         reservation.owner.generation != 0 &&
         reservation.owner.lane_id < 32 &&
         bytes_are_zero(reservation.owner.reserved_zero,
                        sizeof(reservation.owner.reserved_zero)) &&
         bytes_are_zero(reservation.reserved_zero,
                        sizeof(reservation.reserved_zero)) &&
         expected_shape(reservation, &expected_payload_bytes,
                        &expected_chunk_count, &slot_capacity) &&
         reservation.slot_index < slot_capacity &&
         reservation.producer_commit_required <= 1 &&
         reservation.raw_payload_bytes == expected_payload_bytes &&
         reservation.raw_chunk_count == expected_chunk_count;
}

}  // namespace

status_kind prepare_raw_read_plan(
    const fetch_target::reservation_receipt_v0 &reservation,
    raw_read_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = raw_read_plan_v0();
  if (!valid_reservation_shape(reservation)) {
    return kStatusInvalidReservation;
  }
  if (reservation.raw_payload_base_address >
      std::numeric_limits<uint64_t>::max() -
          (reservation.raw_payload_bytes - 1)) {
    return kStatusAddressOverflow;
  }

  for (unsigned index = 0; index < reservation.raw_chunk_count; ++index) {
    raw_read_chunk_v0 &chunk = plan->chunks[index];
    chunk.reservation = reservation;
    chunk.aligned_32b_address =
        reservation.raw_payload_base_address +
        static_cast<uint64_t>(index) * kRawReadChunkBytes;
    chunk.address_space = kAddressSpaceGlobal;
    chunk.operation = kOperationRead;
    chunk.destination = kDestinationTargetQueueFill;
    chunk.target_kind = reservation.target_kind;
    chunk.chunk_id = static_cast<uint8_t>(index);
    chunk.chunk_count = reservation.raw_chunk_count;
    chunk.transfer_bytes = kRawReadChunkBytes;
    chunk.valid = 1;
  }
  plan->chunk_count = reservation.raw_chunk_count;
  plan->valid = 1;
  return kStatusOk;
}

status_kind validate_raw_read_chunk(const raw_read_chunk_v0 &chunk) {
  if (chunk.valid != 1 ||
      chunk.address_space != kAddressSpaceGlobal ||
      chunk.operation != kOperationRead ||
      chunk.destination != kDestinationTargetQueueFill ||
      chunk.target_kind != chunk.reservation.target_kind ||
      chunk.chunk_count != chunk.reservation.raw_chunk_count ||
      chunk.chunk_count == 0 || chunk.chunk_count > kMaxRawReadChunks ||
      chunk.chunk_id >= chunk.chunk_count ||
      chunk.transfer_bytes != kRawReadChunkBytes ||
      !valid_reservation_shape(chunk.reservation)) {
    return kStatusMalformedDescriptor;
  }
  if (chunk.reservation.raw_payload_base_address >
      std::numeric_limits<uint64_t>::max() -
          static_cast<uint64_t>(chunk.chunk_id) * kRawReadChunkBytes ||
      chunk.aligned_32b_address !=
          chunk.reservation.raw_payload_base_address +
              static_cast<uint64_t>(chunk.chunk_id) *
                  kRawReadChunkBytes) {
    return kStatusMalformedDescriptor;
  }
  return kStatusOk;
}

status_kind accept_raw_read_response(
    fetch_target::engine_state_v0 *target_state,
    const raw_read_chunk_v0 &chunk, const uint8_t *response_payload,
    uint8_t response_bytes) {
  if (target_state == NULL || response_payload == NULL) {
    return kStatusInvalidArgument;
  }
  if (validate_raw_read_chunk(chunk) != kStatusOk ||
      response_bytes != kRawReadChunkBytes) {
    return kStatusMalformedDescriptor;
  }
  const fetch_target::status_kind fill_status =
      fetch_target::fill_raw_payload_chunk(
          target_state, chunk.reservation, chunk.chunk_id,
          chunk.chunk_count, response_payload, response_bytes);
  return fill_status == fetch_target::kStatusOk
             ? kStatusOk
             : kStatusTargetFillRejected;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidReservation:
      return "invalid_reservation";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusMalformedDescriptor:
      return "malformed_descriptor";
    case kStatusTargetFillRejected:
      return "target_fill_rejected";
  }
  return "unknown";
}

}  // namespace target_memory
}  // namespace v04
}  // namespace rtcore
