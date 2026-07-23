#ifndef RTCORE_V04_TARGET_MEMORY_BRIDGE_H
#define RTCORE_V04_TARGET_MEMORY_BRIDGE_H

#include <cstdint>

#include "rtcore_v04_fetch_target_queue.h"

namespace rtcore {
namespace v04 {
namespace target_memory {

static const uint8_t kRawReadChunkBytes = 32;
static const uint8_t kMaxRawReadChunks = 4;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidReservation,
  kStatusAddressOverflow,
  kStatusMalformedDescriptor,
  kStatusTargetFillRejected,
};

enum address_space_kind : uint8_t {
  kAddressSpaceInvalid = 0,
  kAddressSpaceGlobal = 1,
};

enum operation_kind : uint8_t {
  kOperationInvalid = 0,
  kOperationRead = 1,
};

enum destination_kind : uint8_t {
  kDestinationInvalid = 0,
  kDestinationTargetQueueFill = 1,
};

struct raw_read_chunk_v0 {
  fetch_target::reservation_receipt_v0 reservation;
  uint64_t aligned_32b_address;
  uint8_t address_space;
  uint8_t operation;
  uint8_t destination;
  uint8_t target_kind;
  uint8_t chunk_id;
  uint8_t chunk_count;
  uint8_t transfer_bytes;
  uint8_t valid;
};

struct raw_read_plan_v0 {
  raw_read_chunk_v0 chunks[kMaxRawReadChunks];
  uint8_t chunk_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

status_kind prepare_raw_read_plan(
    const fetch_target::reservation_receipt_v0 &reservation,
    raw_read_plan_v0 *plan);

status_kind validate_raw_read_chunk(const raw_read_chunk_v0 &chunk);

status_kind accept_raw_read_response(
    fetch_target::engine_state_v0 *target_state,
    const raw_read_chunk_v0 &chunk, const uint8_t *response_payload,
    uint8_t response_bytes);

const char *status_name(status_kind status);

}  // namespace target_memory
}  // namespace v04
}  // namespace rtcore

#endif
