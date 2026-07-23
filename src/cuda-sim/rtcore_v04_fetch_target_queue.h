#ifndef RTCORE_V04_FETCH_TARGET_QUEUE_H
#define RTCORE_V04_FETCH_TARGET_QUEUE_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_stack_result_commit.h"

namespace rtcore {
namespace v04 {
namespace fetch_target {

static const uint8_t kMaxNodeSlots = 32;
static const uint8_t kMaxPrimitiveSlots = 16;
static const uint8_t kMaxInstanceSlots = 8;
static const uint16_t kNodeRawPayloadBytes = 64;
static const uint16_t kPrimitiveRawPayloadBytes = 64;
static const uint16_t kInstanceRawPayloadBytes = 128;
static const uint16_t kMaxRawPayloadBytes = kInstanceRawPayloadBytes;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusInvalidSelectedFetch,
  kStatusCapacityBackpressure,
  kStatusReservationBudgetBackpressure,
  kStatusReservationSequenceExhausted,
  kStatusUnknownReservation,
  kStatusStaleReservation,
  kStatusPayloadShapeMismatch,
  kStatusDuplicateRawPayload,
  kStatusChunkShapeMismatch,
  kStatusDuplicateRawChunk,
  kStatusUnknownProducerCommit,
  kStatusDuplicateProducerCommit,
  kStatusReadyFifoInvariant,
  kStatusNoReadyOperation,
  kStatusUnitInputBackpressure,
};

enum target_kind : uint8_t {
  kTargetInvalid = 0,
  kTargetNode = 1,
  kTargetPrimitive = 2,
  kTargetInstance = 3,
};

enum slot_state_kind : uint8_t {
  kSlotFree = 0,
  kSlotReservedWaitDataOrCommit = 1,
  kSlotReady = 2,
  kSlotIssued = 3,
};

enum operand_valid_bit : uint8_t {
  kOperandSelectedFetchValid = 1u << 0,
  kOperandRawPayloadValid = 1u << 1,
};

struct config_v0 {
  uint8_t node_capacity;
  uint8_t node_reservation_width;
  uint8_t primitive_capacity;
  uint8_t primitive_reservation_width;
  uint8_t instance_capacity;
  uint8_t instance_reservation_width;
  uint8_t reserved_zero[2];
};

struct reservation_receipt_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint64_t raw_payload_base_address;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint32_t slot_generation;
  uint16_t raw_payload_bytes;
  uint8_t target_kind;
  uint8_t slot_index;
  uint8_t raw_chunk_count;
  uint8_t producer_commit_required;
  uint8_t valid;
  uint8_t reserved_zero;
};

struct operation_packet_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint64_t raw_payload_base_address;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint32_t slot_generation;
  uint16_t raw_payload_bytes;
  uint8_t target_kind;
  uint8_t valid;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  uint8_t raw_payload[kMaxRawPayloadBytes];
};

struct slot_metadata_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint64_t raw_payload_base_address;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint32_t slot_generation;
  uint16_t raw_payload_bytes;
  uint8_t target_kind;
  uint8_t state;
  uint8_t valid_operand_mask;
  uint8_t required_operand_mask;
  uint8_t pending_response_count;
  uint8_t expected_chunk_count;
  uint8_t received_chunk_mask;
  uint8_t producer_commit_required;
  uint8_t producer_commit_complete;
  uint8_t ready_enqueued;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
};

struct node_slot_v0 {
  slot_metadata_v0 metadata;
  uint8_t raw_payload[kNodeRawPayloadBytes];
};

struct primitive_slot_v0 {
  slot_metadata_v0 metadata;
  uint8_t raw_payload[kPrimitiveRawPayloadBytes];
};

struct instance_slot_v0 {
  slot_metadata_v0 metadata;
  uint8_t raw_payload[kInstanceRawPayloadBytes];
};

struct ready_fifo_v0 {
  uint8_t indices[kMaxNodeSlots];
  uint64_t reservation_ages[kMaxNodeSlots];
  uint8_t count;
  uint8_t reserved_zero[7];
};

struct reservation_window_v0 {
  uint64_t last_cycle;
  uint8_t last_cycle_valid;
  uint8_t accepted_this_cycle;
  uint8_t reserved_zero[6];
};

struct engine_state_v0 {
  config_v0 config;
  uint64_t next_reservation_id;
  uint64_t next_reservation_age;
  uint8_t initialized;
  uint8_t reserved_zero[7];
  reservation_window_v0 node_window;
  reservation_window_v0 primitive_window;
  reservation_window_v0 instance_window;
  node_slot_v0 node_slots[kMaxNodeSlots];
  primitive_slot_v0 primitive_slots[kMaxPrimitiveSlots];
  instance_slot_v0 instance_slots[kMaxInstanceSlots];
  ready_fifo_v0 node_ready;
  ready_fifo_v0 primitive_ready;
  ready_fifo_v0 instance_ready;
};

status_kind initialize(engine_state_v0 *state, const config_v0 &config);

status_kind classify_selected_fetch(
    const typed_node::selected_child_fetch_work_item_v0 &selected_fetch,
    target_kind *target, uint16_t *raw_payload_bytes);

status_kind try_reserve_prefill(
    engine_state_v0 *state,
    const stack_commit::forwarding_decision_input_v0 &decision,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt);

status_kind fill_raw_payload(engine_state_v0 *state,
                             const reservation_receipt_v0 &reservation,
                             const uint8_t *raw_payload,
                             uint16_t raw_payload_bytes);

status_kind fill_raw_payload_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, const uint8_t *raw_payload_chunk,
    uint8_t chunk_bytes);

status_kind complete_producer_commit(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    uint32_t commit_epoch, reservation_receipt_v0 *reservation);

status_kind pop_ready_operation(engine_state_v0 *state, target_kind target,
                                bool unit_input_accepts,
                                operation_packet_v0 *packet);

uint8_t active_slot_count(const engine_state_v0 &state, target_kind target);
uint8_t ready_slot_count(const engine_state_v0 &state, target_kind target);

const char *status_name(status_kind status);

}  // namespace fetch_target
}  // namespace v04
}  // namespace rtcore

#endif
