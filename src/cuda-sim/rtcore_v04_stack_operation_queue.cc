#include "rtcore_v04_stack_operation_queue.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace stack_operation {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

float fp32_value(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t load_u32_le(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

bool config_valid(const config_v0 &config) {
  return config.capacity != 0 && config.capacity <= kMaxSlots &&
         config.reservation_width != 0 &&
         config.reservation_width <= config.capacity &&
         bytes_are_zero(config.reserved_zero,
                        sizeof(config.reserved_zero));
}

bool route_valid(const typed_node::route_result_v0 &route) {
  const uint8_t required_mask = static_cast<uint8_t>(
      typed_node::kSelectedFetchValid |
      typed_node::kFrontierItemsValid);
  return route.status == typed_node::kStatusOk &&
         route.result_kind == typed_node::kRouteResultSelected &&
         route.frontier_count != 0 &&
         route.frontier_count <= typed_stack::kMaxRemainderChildren &&
         route.output_valid_mask == required_mask &&
         bytes_are_zero(route.reserved_zero,
                        sizeof(route.reserved_zero)) &&
         bytes_are_zero(route.reserved_zero_tail,
                        sizeof(route.reserved_zero_tail));
}

bool input_valid(const reservation_input_v0 &input) {
  return input.owner.request_identity != 0 &&
         input.owner.generation != 0 &&
         input.owner.resident_warp_id < 8 &&
         input.owner.private_slot_id < 256 &&
         input.owner.lane_id < 32 &&
         bytes_are_zero(input.owner.reserved_zero,
                        sizeof(input.owner.reserved_zero)) &&
         input.target_operation_seq != 0 &&
         input.source_node_operation_seq != 0 &&
         input.target_operation_seq != input.source_node_operation_seq &&
         std::isfinite(fp32_value(input.current_traversal_bound_bits)) &&
         bytes_are_zero(input.reserved_zero,
                        sizeof(input.reserved_zero)) &&
         route_valid(input.node_route);
}

bool receipt_matches_slot(const reservation_receipt_v0 &receipt,
                          const slot_v0 &slot, uint8_t slot_index) {
  return receipt.valid == 1 && receipt.slot_index == slot_index &&
         receipt.metadata_chunk_count == kFrontierMetadataReadChunks &&
         bytes_are_zero(&receipt.reserved_zero,
                        sizeof(receipt.reserved_zero)) &&
         slot.state != kSlotFree &&
         std::memcmp(&receipt, &slot.reservation,
                     sizeof(receipt)) == 0;
}

int find_free_slot(const engine_state_v0 &state) {
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    if (state.slots[index].state == kSlotFree) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_oldest_ready_slot(const engine_state_v0 &state) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    const slot_v0 &slot = state.slots[index];
    if (slot.state == kSlotReady &&
        slot.reservation.reservation_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = slot.reservation.reservation_age;
    }
  }
  return selected;
}

void refresh_reservation_window(engine_state_v0 *state,
                                uint64_t reservation_cycle) {
  if (state->reservation_window.last_cycle_valid == 0 ||
      state->reservation_window.last_cycle != reservation_cycle) {
    state->reservation_window.last_cycle = reservation_cycle;
    state->reservation_window.last_cycle_valid = 1;
    state->reservation_window.accepted_this_cycle = 0;
  }
}

bool expected_chunk_shape(uint8_t chunk_id, uint16_t slot_chunk_offset,
                          uint32_t byte_mask) {
  if (chunk_id == 0) {
    return slot_chunk_offset == 0x020 &&
           byte_mask == UINT32_C(0xfffff000);
  }
  if (chunk_id == 1) {
    return slot_chunk_offset == 0x040 &&
           byte_mask == UINT32_C(0x0000000f);
  }
  return false;
}

bool metadata_valid(const uint8_t *bytes) {
  const uint32_t top = load_u32_le(bytes + 0);
  const uint32_t count = load_u32_le(bytes + 4);
  const uint32_t capacity = load_u32_le(bytes + 8);
  const uint32_t current_level = load_u32_le(bytes + 12);
  const uint32_t level_frame_depth = load_u32_le(bytes + 16);
  const uint32_t max_level_depth = load_u32_le(bytes + 20);
  return top == count && top <= capacity &&
         capacity == private_frontier::kFrontierEntryCapacity &&
         current_level <= 1 && max_level_depth == 1 &&
         level_frame_depth <= max_level_depth;
}

void build_packet(const slot_v0 &slot, operation_packet_v0 *packet) {
  *packet = operation_packet_v0();
  packet->owner = slot.reservation.owner;
  packet->reservation_id = slot.reservation.reservation_id;
  packet->reservation_age = slot.reservation.reservation_age;
  packet->target_operation_seq =
      slot.reservation.target_operation_seq;
  packet->source_node_operation_seq =
      slot.reservation.source_node_operation_seq;
  packet->slot_generation = slot.reservation.slot_generation;
  packet->valid = 1;
  packet->input.profile_id = typed_stack::kGenRtDerivedProfileId;
  packet->input.operation_kind =
      typed_stack::kPushRemainderAndForwardSelected;
  packet->input.frontier.frontier_top =
      load_u32_le(slot.metadata_bytes + 0);
  packet->input.frontier.frontier_count =
      load_u32_le(slot.metadata_bytes + 4);
  packet->input.frontier.frontier_capacity =
      load_u32_le(slot.metadata_bytes + 8);
  packet->input.current_traversal_bound_bits =
      slot.current_traversal_bound_bits;
  packet->input.node_route = slot.node_route;
}

}  // namespace

config_v0 candidate_profile_config() {
  config_v0 config = {};
  config.capacity = kMaxSlots;
  config.reservation_width = 1;
  return config;
}

status_kind initialize(engine_state_v0 *state,
                       const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!config_valid(config)) return kStatusInvalidConfiguration;
  *state = engine_state_v0();
  state->config = config;
  state->next_reservation_id = 1;
  state->next_reservation_age = 1;
  for (unsigned index = 0; index < kMaxSlots; ++index) {
    state->next_slot_generation[index] = 1;
  }
  state->initialized = 1;
  return kStatusOk;
}

status_kind try_reserve(
    engine_state_v0 *state, const reservation_input_v0 &input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *receipt = reservation_receipt_v0();
  if (!input_valid(input)) return kStatusInvalidRoute;
  refresh_reservation_window(state, reservation_cycle);
  if (state->reservation_window.accepted_this_cycle >=
      state->config.reservation_width) {
    return kStatusReservationBudgetBackpressure;
  }
  const int slot_index = find_free_slot(*state);
  if (slot_index < 0) return kStatusCapacityBackpressure;
  if (state->next_reservation_id == 0 ||
      state->next_reservation_age == 0 ||
      state->next_slot_generation[slot_index] == 0) {
    return kStatusSequenceExhausted;
  }

  slot_v0 prepared = {};
  prepared.reservation.owner = input.owner;
  prepared.reservation.reservation_id =
      state->next_reservation_id;
  prepared.reservation.reservation_age =
      state->next_reservation_age;
  prepared.reservation.target_operation_seq =
      input.target_operation_seq;
  prepared.reservation.source_node_operation_seq =
      input.source_node_operation_seq;
  prepared.reservation.slot_generation =
      state->next_slot_generation[slot_index];
  prepared.reservation.slot_index =
      static_cast<uint8_t>(slot_index);
  prepared.reservation.metadata_chunk_count =
      kFrontierMetadataReadChunks;
  prepared.reservation.valid = 1;
  prepared.node_route = input.node_route;
  prepared.current_traversal_bound_bits =
      input.current_traversal_bound_bits;
  prepared.state = kSlotReservedWaitMetadata;
  state->slots[slot_index] = prepared;
  *receipt = prepared.reservation;

  ++state->next_reservation_id;
  ++state->next_reservation_age;
  ++state->next_slot_generation[slot_index];
  ++state->reservation_window.accepted_this_cycle;
  return kStatusOk;
}

status_kind fill_frontier_metadata_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint16_t slot_chunk_offset, uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes]) {
  if (state == NULL || payload == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  if (reservation.slot_index >= state->config.capacity) {
    return kStatusUnknownReservation;
  }
  slot_v0 &slot = state->slots[reservation.slot_index];
  if (!receipt_matches_slot(reservation, slot,
                            reservation.slot_index)) {
    return slot.state == kSlotFree ? kStatusUnknownReservation
                                   : kStatusStaleReservation;
  }
  if (chunk_count != kFrontierMetadataReadChunks ||
      chunk_id >= chunk_count ||
      !expected_chunk_shape(chunk_id, slot_chunk_offset, byte_mask)) {
    return kStatusChunkShapeMismatch;
  }
  const uint8_t chunk_bit = static_cast<uint8_t>(1u << chunk_id);
  if ((slot.received_chunk_mask & chunk_bit) != 0) {
    return kStatusDuplicateChunk;
  }

  for (unsigned byte = 0;
       byte < private_frontier::kSharedAccessChunkBytes; ++byte) {
    if ((byte_mask & (uint32_t{1} << byte)) == 0) continue;
    const uint32_t slot_offset = slot_chunk_offset + byte;
    if (slot_offset < private_frontier::kFrontierMetadataOffset ||
        slot_offset >= private_frontier::kFrontierMetadataOffset +
                           private_frontier::kFrontierMetadataBytes) {
      return kStatusChunkShapeMismatch;
    }
    const uint32_t metadata_offset =
        slot_offset - private_frontier::kFrontierMetadataOffset;
    slot.metadata_bytes[metadata_offset] = payload[byte];
    slot.metadata_byte_valid_mask |= uint32_t{1} << metadata_offset;
  }
  slot.received_chunk_mask =
      static_cast<uint8_t>(slot.received_chunk_mask | chunk_bit);
  const uint8_t all_chunks = static_cast<uint8_t>(
      (1u << kFrontierMetadataReadChunks) - 1u);
  const uint32_t all_metadata_bytes =
      (uint32_t{1} << private_frontier::kFrontierMetadataBytes) - 1u;
  if (slot.received_chunk_mask == all_chunks) {
    if (slot.metadata_byte_valid_mask != all_metadata_bytes ||
        !metadata_valid(slot.metadata_bytes)) {
      return kStatusInvalidFrontierMetadata;
    }
    slot.state = kSlotReady;
    ++state->total_ready_publications;
  }
  return kStatusOk;
}

status_kind peek_ready_reservation(
    const engine_state_v0 &state,
    const reservation_receipt_v0 &reservation,
    operation_packet_v0 *packet) {
  if (packet == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  if (reservation.slot_index >= state.config.capacity) {
    return kStatusUnknownReservation;
  }
  const slot_v0 &slot = state.slots[reservation.slot_index];
  if (!receipt_matches_slot(reservation, slot,
                            reservation.slot_index)) {
    return slot.state == kSlotFree ? kStatusUnknownReservation
                                   : kStatusStaleReservation;
  }
  if (slot.state != kSlotReady) return kStatusNoReadyOperation;
  build_packet(slot, packet);
  return kStatusOk;
}

status_kind pop_ready_operation(engine_state_v0 *state,
                                bool unit_input_accepts,
                                operation_packet_v0 *packet) {
  if (state == NULL || packet == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  const int slot_index = find_oldest_ready_slot(*state);
  if (slot_index < 0) return kStatusNoReadyOperation;
  if (!unit_input_accepts) return kStatusUnitInputBackpressure;
  build_packet(state->slots[slot_index], packet);
  state->slots[slot_index] = slot_v0();
  return kStatusOk;
}

uint8_t active_slot_count(const engine_state_v0 &state) {
  if (state.initialized != 1) return 0;
  uint8_t count = 0;
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    count += state.slots[index].state != kSlotFree ? 1 : 0;
  }
  return count;
}

uint8_t ready_slot_count(const engine_state_v0 &state) {
  if (state.initialized != 1) return 0;
  uint8_t count = 0;
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    count += state.slots[index].state == kSlotReady ? 1 : 0;
  }
  return count;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfiguration:
      return "invalid_configuration";
    case kStatusInvalidRoute:
      return "invalid_route";
    case kStatusCapacityBackpressure:
      return "capacity_backpressure";
    case kStatusReservationBudgetBackpressure:
      return "reservation_budget_backpressure";
    case kStatusSequenceExhausted:
      return "sequence_exhausted";
    case kStatusUnknownReservation:
      return "unknown_reservation";
    case kStatusStaleReservation:
      return "stale_reservation";
    case kStatusChunkShapeMismatch:
      return "chunk_shape_mismatch";
    case kStatusDuplicateChunk:
      return "duplicate_chunk";
    case kStatusInvalidFrontierMetadata:
      return "invalid_frontier_metadata";
    case kStatusNoReadyOperation:
      return "no_ready_operation";
    case kStatusUnitInputBackpressure:
      return "unit_input_backpressure";
  }
  return "unknown";
}

}  // namespace stack_operation
}  // namespace v04
}  // namespace rtcore
