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
  kStatusPrivateOperandShapeMismatch,
  kStatusDuplicatePrivateChunk,
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
  kOperandTargetReferenceValid = 1u << 0,
  kOperandRawPayloadValid = 1u << 1,
  kOperandMutableRayValid = 1u << 2,
  kOperandRayPolicyValid = 1u << 3,
  kOperandDecodeContextValid = 1u << 4,
  kOperandCommittedHitValid = 1u << 5,
};

enum target_reference_source_kind : uint8_t {
  kTargetReferenceInvalid = 0,
  kTargetReferenceRootCompatibilityProxy = 1,
  kTargetReferenceSelectedFetchCompatibilityAdapter = 2,
};

struct target_reference_v0 {
  uint64_t payload_offset;
  uint32_t near_t_bits;
  uint16_t payload_byte_count;
  uint8_t payload_kind;
  uint8_t level;
  uint8_t source_kind;
  uint8_t proxy_delegated;
  uint8_t reserved_zero[6];
};

struct reservation_input_v0 {
  private_frontier::owner_binding_v0 owner;
  target_reference_v0 target_reference;
  typed_node::ray_policy_v0 forwarded_ray_policy;
  uint64_t raw_payload_base_address;
  uint32_t target_operation_seq;
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint16_t raw_payload_bytes;
  uint8_t target_kind;
  uint8_t producer_commit_required;
  uint8_t required_operand_mask;
  uint8_t forwarded_operand_mask;
  uint8_t reserved_zero[6];
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
  uint32_t target_operation_seq;
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint32_t slot_generation;
  uint16_t raw_payload_bytes;
  uint8_t target_kind;
  uint8_t slot_index;
  uint8_t raw_chunk_count;
  uint8_t private_chunk_count;
  uint8_t producer_commit_required;
  uint8_t valid;
  uint8_t reserved_zero[2];
};

struct operation_packet_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint64_t raw_payload_base_address;
  uint32_t target_operation_seq;
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint32_t slot_generation;
  uint16_t raw_payload_bytes;
  uint8_t target_kind;
  uint8_t valid;
  target_reference_v0 target_reference;
  typed_node::ray_policy_v0 ray_policy;
  private_frontier::root_private_operands_v0 private_operands;
  uint8_t raw_payload[kMaxRawPayloadBytes];
};

struct slot_metadata_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint64_t raw_payload_base_address;
  uint32_t target_operation_seq;
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint32_t slot_generation;
  uint16_t raw_payload_bytes;
  uint8_t target_kind;
  uint8_t state;
  uint8_t valid_operand_mask;
  uint8_t required_operand_mask;
  uint8_t pending_raw_response_count;
  uint8_t expected_raw_chunk_count;
  uint8_t received_raw_chunk_mask;
  uint8_t pending_private_response_count;
  uint8_t expected_private_chunk_count;
  uint8_t received_private_chunk_mask;
  uint8_t producer_commit_required;
  uint8_t producer_commit_complete;
  uint8_t ready_enqueued;
  target_reference_v0 target_reference;
  typed_node::ray_policy_v0 ray_policy;
  uint8_t mutable_ray_bytes[private_frontier::kMutableRayStateBytes];
  uint8_t decode_context_bytes[private_frontier::kAsDecodeContextBytes];
  uint8_t committed_hit_bytes[private_frontier::kCommittedHitBytes];
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

status_kind try_reserve(
    engine_state_v0 *state, const reservation_input_v0 &input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt);

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

status_kind fill_private_operand_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint8_t field_kind, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes]);

status_kind complete_producer_commit(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    uint32_t commit_epoch, reservation_receipt_v0 *reservation);

status_kind build_ready_operation_packet(
    const reservation_input_v0 &input, uint64_t reservation_id,
    uint64_t reservation_age, uint32_t slot_generation,
    const private_frontier::root_private_operands_v0 &private_operands,
    const uint8_t *raw_payload, operation_packet_v0 *packet);

status_kind build_ready_node_operation_packet(
    const reservation_input_v0 &input, uint64_t reservation_id,
    uint64_t reservation_age, uint32_t slot_generation,
    const private_frontier::root_private_operands_v0 &private_operands,
    const uint8_t *raw_payload, operation_packet_v0 *packet);

status_kind pop_ready_operation(engine_state_v0 *state, target_kind target,
                                bool unit_input_accepts,
                                operation_packet_v0 *packet);

status_kind peek_ready_operation(const engine_state_v0 &state,
                                 target_kind target,
                                 operation_packet_v0 *packet);

status_kind peek_ready_reservation(
    const engine_state_v0 &state,
    const reservation_receipt_v0 &reservation,
    operation_packet_v0 *packet);

uint8_t active_slot_count(const engine_state_v0 &state, target_kind target);
uint8_t ready_slot_count(const engine_state_v0 &state, target_kind target);

const char *status_name(status_kind status);

}  // namespace fetch_target
}  // namespace v04
}  // namespace rtcore

#endif
