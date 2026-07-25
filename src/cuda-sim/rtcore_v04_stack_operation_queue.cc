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

uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint32_t load_u32_le(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

template <typename T>
bool object_is_zero(const T &value) {
  const T zero = {};
  return std::memcmp(&value, &zero, sizeof(value)) == 0;
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

uint8_t normalized_operation_kind(const reservation_input_v0 &input) {
  if (input.operation_kind == typed_stack::kOperationInvalid &&
      route_valid(input.node_route)) {
    return typed_stack::kPushRemainderAndForwardSelected;
  }
  return input.operation_kind;
}

bool input_valid(const reservation_input_v0 &input) {
  const uint8_t operation_kind = normalized_operation_kind(input);
  const bool operation_payload_valid =
      operation_kind ==
              typed_stack::kPushRemainderAndForwardSelected
          ? std::isfinite(fp32_value(
                input.current_traversal_bound_bits)) &&
                route_valid(input.node_route)
          : operation_kind == typed_stack::kPopNext &&
                input.current_traversal_bound_bits == 0 &&
                object_is_zero(input.node_route) &&
                object_is_zero(input.ray_policy);
  return input.owner.request_identity != 0 &&
         input.owner.generation != 0 &&
         input.owner.resident_warp_id < 8 &&
         input.owner.private_slot_id < 256 &&
         input.owner.lane_id < 32 &&
         bytes_are_zero(input.owner.reserved_zero,
                        sizeof(input.owner.reserved_zero)) &&
         input.target_operation_seq != 0 &&
         input.producer_operation_seq != 0 &&
         input.target_operation_seq != input.producer_operation_seq &&
         operation_payload_valid &&
         bytes_are_zero(input.ray_policy.reserved_zero,
                        sizeof(input.ray_policy.reserved_zero)) &&
         bytes_are_zero(input.reserved_zero,
                        sizeof(input.reserved_zero));
}

bool receipt_matches_slot(const reservation_receipt_v0 &receipt,
                          const slot_v0 &slot, uint8_t slot_index) {
  return receipt.valid == 1 && receipt.slot_index == slot_index &&
         receipt.metadata_chunk_count == kFrontierMetadataReadChunks &&
         receipt.operation_kind ==
             slot.reservation.operation_kind &&
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

private_frontier::frontier_metadata_image_v0 decode_metadata_bytes(
    const uint8_t *bytes) {
  private_frontier::frontier_metadata_image_v0 metadata = {};
  metadata.frontier_top = load_u32_le(bytes + 0);
  metadata.frontier_count = load_u32_le(bytes + 4);
  metadata.frontier_capacity = load_u32_le(bytes + 8);
  metadata.current_level = load_u32_le(bytes + 12);
  metadata.level_frame_depth = load_u32_le(bytes + 16);
  metadata.max_level_depth = load_u32_le(bytes + 20);
  return metadata;
}

uint64_t valid_byte_mask(unsigned byte_count) {
  return byte_count == 64
             ? std::numeric_limits<uint64_t>::max()
             : (uint64_t{1} << byte_count) - 1;
}

bool derive_region_from_plan(
    const reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan,
    private_frontier::region_binding_v0 *region) {
  if (region == NULL || read_plan.access_count == 0) return false;
  const private_frontier::shared_chunk_access_v0 &first =
      read_plan.accesses[0];
  const uint32_t aligned_slot_offset =
      first.slot_byte_offset -
      (first.slot_byte_offset %
       private_frontier::kSharedAccessChunkBytes);
  if (first.aligned_32b_address < aligned_slot_offset) return false;
  const uint64_t slot_base =
      first.aligned_32b_address - aligned_slot_offset;
  const uint64_t lane_offset =
      static_cast<uint64_t>(reservation.owner.private_slot_id) *
      private_frontier::kPrivateDataSlotBytes;
  if (slot_base < lane_offset) return false;
  *region = private_frontier::region_binding_v0();
  region->profile_id = private_frontier::kLayoutProfileId;
  region->slot_count = 256;
  region->private_region_base = slot_base - lane_offset;
  return true;
}

bool pop_operand_plan_valid(
    const reservation_receipt_v0 &reservation,
    const slot_v0 &slot,
    const private_frontier::access_plan_v0 &read_plan) {
  if (slot.state != kSlotReservedWaitPopPlan ||
      reservation.operation_kind != typed_stack::kPopNext ||
      read_plan.access_count == 0 ||
      read_plan.access_count > kMaxPopOperandReadChunks ||
      !private_frontier::owners_equal(read_plan.owner,
                                      reservation.owner)) {
    return false;
  }
  private_frontier::region_binding_v0 region = {};
  if (!derive_region_from_plan(reservation, read_plan, &region)) {
    return false;
  }
  private_frontier::access_plan_v0 expected = {};
  const private_frontier::frontier_metadata_image_v0 metadata =
      decode_metadata_bytes(slot.metadata_bytes);
  return private_frontier::build_nonempty_pop_operand_read_plan(
             reservation.owner, region, metadata, &expected) ==
             private_frontier::kStatusOk &&
         std::memcmp(&read_plan, &expected, sizeof(read_plan)) == 0;
}

bool pop_projection_complete(const slot_v0 &slot) {
  return slot.top_entry_byte_valid_mask ==
             static_cast<uint16_t>(
                 valid_byte_mask(
                     private_frontier::kFrontierEntryBytes)) &&
         slot.mutable_ray_byte_valid_mask ==
             valid_byte_mask(private_frontier::kMutableRayStateBytes) &&
         slot.decode_context_byte_valid_mask ==
             valid_byte_mask(private_frontier::kAsDecodeContextBytes) &&
         slot.committed_hit_byte_valid_mask ==
             valid_byte_mask(private_frontier::kCommittedHitBytes);
}

bool decode_pop_projection(
    const slot_v0 &slot, typed_stack::pop_input_v0 *input) {
  if (input == NULL || !pop_projection_complete(slot)) return false;
  private_frontier::shadow_slot_v0 projection = {};
  projection.owner = slot.reservation.owner;
  std::memcpy(
      projection.bytes + private_frontier::kMutableRayStateOffset,
      slot.mutable_ray_bytes, sizeof(slot.mutable_ray_bytes));
  std::memcpy(
      projection.bytes + private_frontier::kAsDecodeContextOffset,
      slot.decode_context_bytes, sizeof(slot.decode_context_bytes));
  std::memcpy(
      projection.bytes + private_frontier::kCommittedHitOffset,
      slot.committed_hit_bytes, sizeof(slot.committed_hit_bytes));
  private_frontier::root_private_operands_v0 operands = {};
  if (private_frontier::decode_root_private_operands(
          projection, slot.reservation.owner, &operands) !=
      private_frontier::kStatusOk) {
    return false;
  }

  *input = typed_stack::pop_input_v0();
  input->profile_id = typed_stack::kGenRtDerivedProfileId;
  input->operation_kind = typed_stack::kPopNext;
  input->has_top_entry = 1;
  const private_frontier::frontier_metadata_image_v0 metadata =
      decode_metadata_bytes(slot.metadata_bytes);
  input->frontier.frontier_top = metadata.frontier_top;
  input->frontier.frontier_count = metadata.frontier_count;
  input->frontier.frontier_capacity = metadata.frontier_capacity;
  input->current_traversal_bound_bits =
      fp32_bits(operands.committed_hit.valid != 0
                    ? operands.committed_hit.hit_t
                    : operands.mutable_ray.t_max);
  std::memcpy(&input->top_entry, slot.top_entry_bytes,
              sizeof(input->top_entry));
  input->current_decode_context = operands.decode_context;
  return true;
}

void build_packet(const slot_v0 &slot, operation_packet_v0 *packet) {
  *packet = operation_packet_v0();
  packet->owner = slot.reservation.owner;
  packet->reservation_id = slot.reservation.reservation_id;
  packet->reservation_age = slot.reservation.reservation_age;
  packet->target_operation_seq =
      slot.reservation.target_operation_seq;
  packet->producer_operation_seq =
      slot.reservation.producer_operation_seq;
  packet->slot_generation = slot.reservation.slot_generation;
  packet->operation_kind = slot.reservation.operation_kind;
  packet->valid = 1;
  packet->frontier_metadata =
      decode_metadata_bytes(slot.metadata_bytes);
  packet->ray_policy = slot.ray_policy;
  if (slot.reservation.operation_kind ==
      typed_stack::kPushRemainderAndForwardSelected) {
    packet->input.profile_id = typed_stack::kGenRtDerivedProfileId;
    packet->input.operation_kind =
        typed_stack::kPushRemainderAndForwardSelected;
    packet->input.frontier.frontier_top =
        packet->frontier_metadata.frontier_top;
    packet->input.frontier.frontier_count =
        packet->frontier_metadata.frontier_count;
    packet->input.frontier.frontier_capacity =
        packet->frontier_metadata.frontier_capacity;
    packet->input.current_traversal_bound_bits =
        slot.current_traversal_bound_bits;
    packet->input.node_route = slot.node_route;
  } else {
    const bool decoded =
        decode_pop_projection(slot, &packet->pop_input);
    if (!decoded) {
      *packet = operation_packet_v0();
    }
  }
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
  prepared.reservation.producer_operation_seq =
      input.producer_operation_seq;
  prepared.reservation.slot_generation =
      state->next_slot_generation[slot_index];
  prepared.reservation.slot_index =
      static_cast<uint8_t>(slot_index);
  prepared.reservation.metadata_chunk_count =
      kFrontierMetadataReadChunks;
  prepared.reservation.operation_kind =
      normalized_operation_kind(input);
  prepared.reservation.valid = 1;
  prepared.node_route = input.node_route;
  prepared.ray_policy = input.ray_policy;
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
  if ((slot.received_metadata_chunk_mask & chunk_bit) != 0) {
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
  slot.received_metadata_chunk_mask = static_cast<uint8_t>(
      slot.received_metadata_chunk_mask | chunk_bit);
  const uint8_t all_chunks = static_cast<uint8_t>(
      (1u << kFrontierMetadataReadChunks) - 1u);
  const uint32_t all_metadata_bytes =
      (uint32_t{1} << private_frontier::kFrontierMetadataBytes) - 1u;
  if (slot.received_metadata_chunk_mask == all_chunks) {
    if (slot.metadata_byte_valid_mask != all_metadata_bytes ||
        !metadata_valid(slot.metadata_bytes)) {
      return kStatusInvalidFrontierMetadata;
    }
    if (slot.reservation.operation_kind ==
        typed_stack::kPushRemainderAndForwardSelected) {
      slot.state = kSlotReady;
      ++state->total_ready_publications;
    } else {
      const private_frontier::frontier_metadata_image_v0 metadata =
          decode_metadata_bytes(slot.metadata_bytes);
      slot.state = metadata.frontier_count == 0
                       ? kSlotEmptyFrontierBoundary
                       : kSlotReservedWaitPopPlan;
    }
  }
  return kStatusOk;
}

status_kind bind_pop_operand_read_plan(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation,
    const private_frontier::access_plan_v0 &read_plan) {
  if (state == NULL || state->initialized != 1) {
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
  if (slot.state != kSlotReservedWaitPopPlan) {
    return kStatusPopOperandPlanRequired;
  }
  if (!pop_operand_plan_valid(reservation, slot, read_plan)) {
    return kStatusPopOperandPlanMismatch;
  }
  slot.pop_operand_read_plan = read_plan;
  slot.expected_pop_operand_chunk_count = read_plan.access_count;
  slot.state = kSlotReservedWaitPopOperands;
  return kStatusOk;
}

status_kind fill_pop_operand_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint8_t field_kind, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
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
  if (slot.state != kSlotReservedWaitPopOperands ||
      chunk_count != slot.expected_pop_operand_chunk_count ||
      chunk_id >= chunk_count) {
    return kStatusPopOperandPlanMismatch;
  }
  const private_frontier::shared_chunk_access_v0 &expected =
      slot.pop_operand_read_plan.accesses[chunk_id];
  const uint16_t expected_chunk_offset = static_cast<uint16_t>(
      expected.slot_byte_offset -
      (expected.slot_byte_offset %
       private_frontier::kSharedAccessChunkBytes));
  if (field_kind != expected.field_kind ||
      slot_chunk_offset != expected_chunk_offset ||
      byte_mask != expected.byte_mask ||
      expected.access_kind != private_frontier::kAccessRead) {
    return kStatusPopOperandPlanMismatch;
  }
  const uint16_t chunk_bit = static_cast<uint16_t>(1u << chunk_id);
  if ((slot.received_pop_operand_chunk_mask & chunk_bit) != 0) {
    return kStatusDuplicatePopOperandChunk;
  }

  const private_frontier::frontier_metadata_image_v0 metadata =
      decode_metadata_bytes(slot.metadata_bytes);
  const uint32_t top_entry_offset =
      private_frontier::kFrontierEntriesOffset +
      (metadata.frontier_top - 1) *
          private_frontier::kFrontierEntryBytes;
  for (unsigned byte = 0;
       byte < private_frontier::kSharedAccessChunkBytes; ++byte) {
    if ((byte_mask & (uint32_t{1} << byte)) == 0) continue;
    const uint32_t slot_offset = slot_chunk_offset + byte;
    uint8_t *destination = NULL;
    uint64_t *valid_mask = NULL;
    uint32_t field_offset = 0;
    uint32_t field_bytes = 0;
    switch (field_kind) {
      case private_frontier::kFieldFrontierEntry:
        destination = slot.top_entry_bytes;
        field_offset = top_entry_offset;
        field_bytes = private_frontier::kFrontierEntryBytes;
        break;
      case private_frontier::kFieldMutableRayState:
        destination = slot.mutable_ray_bytes;
        valid_mask = &slot.mutable_ray_byte_valid_mask;
        field_offset = private_frontier::kMutableRayStateOffset;
        field_bytes = private_frontier::kMutableRayStateBytes;
        break;
      case private_frontier::kFieldAsDecodeContext:
        destination = slot.decode_context_bytes;
        valid_mask = &slot.decode_context_byte_valid_mask;
        field_offset = private_frontier::kAsDecodeContextOffset;
        field_bytes = private_frontier::kAsDecodeContextBytes;
        break;
      case private_frontier::kFieldCommittedHit:
        destination = slot.committed_hit_bytes;
        valid_mask = &slot.committed_hit_byte_valid_mask;
        field_offset = private_frontier::kCommittedHitOffset;
        field_bytes = private_frontier::kCommittedHitBytes;
        break;
      default:
        return kStatusPopOperandPlanMismatch;
    }
    if (slot_offset < field_offset ||
        slot_offset >= field_offset + field_bytes) {
      return kStatusPopOperandPlanMismatch;
    }
    const uint32_t destination_offset = slot_offset - field_offset;
    destination[destination_offset] = payload[byte];
    if (field_kind == private_frontier::kFieldFrontierEntry) {
      slot.top_entry_byte_valid_mask =
          static_cast<uint16_t>(
              slot.top_entry_byte_valid_mask |
              (uint16_t{1} << destination_offset));
    } else {
      *valid_mask |= uint64_t{1} << destination_offset;
    }
  }
  slot.received_pop_operand_chunk_mask = static_cast<uint16_t>(
      slot.received_pop_operand_chunk_mask | chunk_bit);
  const uint16_t all_chunks = static_cast<uint16_t>(
      (uint16_t{1} << slot.expected_pop_operand_chunk_count) - 1u);
  if (slot.received_pop_operand_chunk_mask == all_chunks) {
    typed_stack::pop_input_v0 input = {};
    if (!decode_pop_projection(slot, &input)) {
      return kStatusPopOperandPlanMismatch;
    }
    slot.state = kSlotReady;
    ++state->total_ready_publications;
  }
  return kStatusOk;
}

status_kind peek_frontier_metadata(
    const engine_state_v0 &state,
    const reservation_receipt_v0 &reservation,
    private_frontier::frontier_metadata_image_v0 *metadata) {
  if (metadata == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *metadata = private_frontier::frontier_metadata_image_v0();
  if (reservation.slot_index >= state.config.capacity) {
    return kStatusUnknownReservation;
  }
  const slot_v0 &slot = state.slots[reservation.slot_index];
  if (!receipt_matches_slot(reservation, slot,
                            reservation.slot_index)) {
    return slot.state == kSlotFree ? kStatusUnknownReservation
                                   : kStatusStaleReservation;
  }
  if (slot.state == kSlotReservedWaitMetadata) {
    return kStatusPopOperandPlanRequired;
  }
  *metadata = decode_metadata_bytes(slot.metadata_bytes);
  return slot.state == kSlotEmptyFrontierBoundary
             ? kStatusEmptyFrontierBoundary
             : kStatusOk;
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

status_kind peek_ready_operation(const engine_state_v0 &state,
                                 operation_packet_v0 *packet) {
  if (packet == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  const int slot_index = find_oldest_ready_slot(state);
  if (slot_index < 0) return kStatusNoReadyOperation;
  build_packet(state.slots[slot_index], packet);
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

uint8_t empty_boundary_slot_count(const engine_state_v0 &state) {
  if (state.initialized != 1) return 0;
  uint8_t count = 0;
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    count += state.slots[index].state ==
                     kSlotEmptyFrontierBoundary
                 ? 1
                 : 0;
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
    case kStatusPopOperandPlanRequired:
      return "pop_operand_plan_required";
    case kStatusPopOperandPlanMismatch:
      return "pop_operand_plan_mismatch";
    case kStatusDuplicatePopOperandChunk:
      return "duplicate_pop_operand_chunk";
    case kStatusEmptyFrontierBoundary:
      return "empty_frontier_boundary";
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
