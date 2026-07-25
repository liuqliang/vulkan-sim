#ifndef RTCORE_V04_STACK_OPERATION_QUEUE_H
#define RTCORE_V04_STACK_OPERATION_QUEUE_H

#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"

namespace rtcore {
namespace v04 {
namespace stack_operation {

static const uint8_t kMaxSlots = 4;
static const uint8_t kFrontierMetadataReadChunks = 2;
static const uint8_t kMaxPopOperandReadChunks =
    private_frontier::kMaxNonemptyPopOperandChunks;
static const uint8_t kMaxEmptyOperandReadChunks =
    private_frontier::kMaxEmptyPopOperandChunks;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusInvalidRoute,
  kStatusCapacityBackpressure,
  kStatusReservationBudgetBackpressure,
  kStatusSequenceExhausted,
  kStatusUnknownReservation,
  kStatusStaleReservation,
  kStatusChunkShapeMismatch,
  kStatusDuplicateChunk,
  kStatusInvalidFrontierMetadata,
  kStatusPopOperandPlanRequired,
  kStatusPopOperandPlanMismatch,
  kStatusDuplicatePopOperandChunk,
  kStatusEmptyOperandPlanRequired,
  kStatusEmptyOperandPlanMismatch,
  kStatusDuplicateEmptyOperandChunk,
  kStatusEmptyFrontierBoundary,
  kStatusNoReadyOperation,
  kStatusUnitInputBackpressure,
};

enum slot_state_kind : uint8_t {
  kSlotFree = 0,
  kSlotReservedWaitMetadata = 1,
  kSlotReservedWaitPopPlan = 2,
  kSlotReservedWaitPopOperands = 3,
  kSlotReservedWaitEmptyPlan = 4,
  kSlotReservedWaitEmptyOperands = 5,
  kSlotEmptyFrontierBoundary = 6,
  kSlotReady = 7,
};

struct config_v0 {
  uint8_t capacity;
  uint8_t reservation_width;
  uint8_t reserved_zero[6];
};

struct reservation_input_v0 {
  private_frontier::owner_binding_v0 owner;
  typed_node::route_result_v0 node_route;
  typed_node::ray_policy_v0 ray_policy;
  uint32_t current_traversal_bound_bits;
  uint32_t target_operation_seq;
  union {
    uint32_t producer_operation_seq;
    uint32_t source_node_operation_seq;
  };
  uint8_t operation_kind;
  uint8_t reserved_zero[3];
};

struct reservation_receipt_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint32_t target_operation_seq;
  union {
    uint32_t producer_operation_seq;
    uint32_t source_node_operation_seq;
  };
  uint32_t slot_generation;
  uint8_t slot_index;
  uint8_t metadata_chunk_count;
  uint8_t operation_kind;
  uint8_t valid;
};

struct operation_packet_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint32_t target_operation_seq;
  union {
    uint32_t producer_operation_seq;
    uint32_t source_node_operation_seq;
  };
  uint32_t slot_generation;
  uint8_t operation_kind;
  uint8_t valid;
  uint8_t reserved_zero[2];
  private_frontier::frontier_metadata_image_v0 frontier_metadata;
  typed_node::ray_policy_v0 ray_policy;
  typed_stack::push_input_v0 input;
  typed_stack::pop_input_v0 pop_input;
  typed_stack::empty_input_v0 empty_input;
};

struct slot_v0 {
  reservation_receipt_v0 reservation;
  typed_node::route_result_v0 node_route;
  typed_node::ray_policy_v0 ray_policy;
  uint32_t current_traversal_bound_bits;
  uint32_t metadata_byte_valid_mask;
  uint8_t metadata_bytes[private_frontier::kFrontierMetadataBytes];
  private_frontier::access_plan_v0 pop_operand_read_plan;
  private_frontier::access_plan_v0 empty_operand_read_plan;
  uint8_t top_entry_bytes[private_frontier::kFrontierEntryBytes];
  uint8_t mutable_ray_bytes[private_frontier::kMutableRayStateBytes];
  uint8_t decode_context_bytes[private_frontier::kAsDecodeContextBytes];
  uint8_t committed_hit_bytes[private_frontier::kCommittedHitBytes];
  uint8_t parent_frame_bytes[private_frontier::kParentFrameBytes];
  uint64_t mutable_ray_byte_valid_mask;
  uint64_t decode_context_byte_valid_mask;
  uint64_t committed_hit_byte_valid_mask;
  uint64_t parent_frame_byte_valid_mask[2];
  uint16_t top_entry_byte_valid_mask;
  uint16_t received_pop_operand_chunk_mask;
  uint8_t received_empty_operand_chunk_mask;
  uint8_t received_metadata_chunk_mask;
  uint8_t expected_pop_operand_chunk_count;
  uint8_t expected_empty_operand_chunk_count;
  uint8_t empty_input_valid;
  uint8_t state;
  uint8_t reserved_zero[1];
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
  uint32_t next_slot_generation[kMaxSlots];
  uint64_t total_ready_publications;
  uint8_t initialized;
  uint8_t reserved_zero[7];
  reservation_window_v0 reservation_window;
  slot_v0 slots[kMaxSlots];
};

config_v0 candidate_profile_config();

status_kind initialize(engine_state_v0 *state, const config_v0 &config);

status_kind try_reserve(
    engine_state_v0 *state, const reservation_input_v0 &input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt);

status_kind fill_frontier_metadata_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint16_t slot_chunk_offset, uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes]);

status_kind bind_pop_operand_read_plan(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan);

status_kind fill_pop_operand_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint8_t field_kind, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes]);

status_kind bind_empty_operand_read_plan(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan);

status_kind fill_empty_operand_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint8_t field_kind, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes]);

status_kind peek_frontier_metadata(
    const engine_state_v0 &state,
    const reservation_receipt_v0 &reservation,
    private_frontier::frontier_metadata_image_v0 *metadata);

status_kind peek_ready_reservation(
    const engine_state_v0 &state,
    const reservation_receipt_v0 &reservation,
    operation_packet_v0 *packet);

status_kind peek_ready_operation(const engine_state_v0 &state,
                                 operation_packet_v0 *packet);

status_kind pop_ready_operation(engine_state_v0 *state,
                                bool unit_input_accepts,
                                operation_packet_v0 *packet);

uint8_t active_slot_count(const engine_state_v0 &state);
uint8_t ready_slot_count(const engine_state_v0 &state);
uint8_t empty_boundary_slot_count(const engine_state_v0 &state);

const char *status_name(status_kind status);

}  // namespace stack_operation
}  // namespace v04
}  // namespace rtcore

#endif
