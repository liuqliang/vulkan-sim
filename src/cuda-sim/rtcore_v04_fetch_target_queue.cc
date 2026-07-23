#include "rtcore_v04_fetch_target_queue.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace fetch_target {
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

bool valid_owner(const private_frontier::owner_binding_v0 &owner) {
  return owner.request_identity != 0 && owner.generation != 0 &&
         owner.lane_id < 32 &&
         bytes_are_zero(owner.reserved_zero, sizeof(owner.reserved_zero));
}

bool valid_config(const config_v0 &config) {
  return config.node_capacity != 0 &&
         config.node_capacity <= kMaxNodeSlots &&
         config.node_reservation_width != 0 &&
         config.node_reservation_width <= config.node_capacity &&
         config.primitive_capacity != 0 &&
         config.primitive_capacity <= kMaxPrimitiveSlots &&
         config.primitive_reservation_width != 0 &&
         config.primitive_reservation_width <= config.primitive_capacity &&
         config.instance_capacity != 0 &&
         config.instance_capacity <= kMaxInstanceSlots &&
         config.instance_reservation_width != 0 &&
         config.instance_reservation_width <= config.instance_capacity &&
         bytes_are_zero(config.reserved_zero, sizeof(config.reserved_zero));
}

bool kind_allowed_for_as_type(uint8_t as_type, uint8_t payload_kind) {
  if (as_type == 1) {
    return payload_kind == typed_node::kInternalPayloadKind ||
           payload_kind == typed_node::kInstancePayloadKind;
  }
  if (as_type == typed_blas::kAsTypeBlas) {
    return payload_kind == typed_node::kInternalPayloadKind ||
           payload_kind == typed_node::kProceduralPayloadKind ||
           payload_kind == typed_node::kQuadPayloadKind;
  }
  return false;
}

bool selected_fetch_shape_valid(
    const typed_node::selected_child_fetch_work_item_v0 &selected,
    target_kind target, uint16_t payload_bytes) {
  const typed_blas::as_decode_context_v0 &context = selected.decode_context;
  const typed_node::compact_child_work_item_v0 &child = selected.child;
  return context.bvh_format_profile_id == typed_node::kGenRtDerivedProfileId &&
         context.reserved_zero == 0 && context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         kind_allowed_for_as_type(context.as_object.as_type,
                                  child.payload_kind) &&
         context.device_base != 0 && context.device_range_bytes >= 64 &&
         context.device_range_bytes <=
             std::numeric_limits<uint64_t>::max() - context.device_base &&
         child.payload_byte_count == payload_bytes && child.child_slot < 6 &&
         (child.payload_offset & uint64_t{0x3f}) == 0 &&
         child.payload_offset <= context.device_range_bytes &&
         static_cast<uint64_t>(payload_bytes) <=
             context.device_range_bytes - child.payload_offset &&
         std::isfinite(fp32_value(child.near_t_bits)) &&
         target != kTargetInvalid;
}

template <typename Slot>
int find_free_slot(const Slot *slots, uint8_t capacity) {
  for (unsigned index = 0; index < capacity; ++index) {
    if (slots[index].metadata.state == kSlotFree) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

bool reservation_budget_available(const reservation_window_v0 &window,
                                  uint8_t width, uint64_t cycle) {
  return window.last_cycle_valid == 0 || window.last_cycle != cycle ||
         window.accepted_this_cycle < width;
}

void consume_reservation_budget(reservation_window_v0 *window,
                                uint64_t cycle) {
  if (window->last_cycle_valid == 0 || window->last_cycle != cycle) {
    window->last_cycle = cycle;
    window->last_cycle_valid = 1;
    window->accepted_this_cycle = 0;
  }
  ++window->accepted_this_cycle;
}

void build_receipt(const slot_metadata_v0 &metadata, uint8_t slot_index,
                   reservation_receipt_v0 *receipt) {
  *receipt = reservation_receipt_v0();
  receipt->owner = metadata.owner;
  receipt->reservation_id = metadata.reservation_id;
  receipt->reservation_age = metadata.reservation_age;
  receipt->operation_seq = metadata.operation_seq;
  receipt->commit_epoch = metadata.commit_epoch;
  receipt->slot_generation = metadata.slot_generation;
  receipt->raw_payload_bytes = metadata.raw_payload_bytes;
  receipt->target_kind = metadata.target_kind;
  receipt->slot_index = slot_index;
  receipt->producer_commit_required =
      metadata.producer_commit_required;
  receipt->valid = 1;
}

template <typename Slot>
status_kind reserve_in_queue(
    Slot *slots, uint8_t capacity, reservation_window_v0 *window,
    uint8_t reservation_width, engine_state_v0 *state,
    const stack_commit::forwarding_decision_input_v0 &decision,
    target_kind target, uint16_t raw_payload_bytes, uint64_t cycle,
    reservation_receipt_v0 *receipt) {
  if (!reservation_budget_available(*window, reservation_width, cycle)) {
    return kStatusReservationBudgetBackpressure;
  }
  const int slot_index = find_free_slot(slots, capacity);
  if (slot_index < 0) return kStatusCapacityBackpressure;
  if (state->next_reservation_id == 0 ||
      state->next_reservation_age == 0 ||
      slots[slot_index].metadata.slot_generation ==
          std::numeric_limits<uint32_t>::max()) {
    return kStatusReservationSequenceExhausted;
  }

  const uint32_t slot_generation =
      slots[slot_index].metadata.slot_generation + 1;
  std::memset(&slots[slot_index], 0, sizeof(slots[slot_index]));
  slot_metadata_v0 &metadata = slots[slot_index].metadata;
  metadata.owner = decision.owner;
  metadata.reservation_id = state->next_reservation_id;
  metadata.reservation_age = state->next_reservation_age;
  metadata.operation_seq = decision.operation_seq;
  metadata.commit_epoch = decision.commit_epoch;
  metadata.slot_generation = slot_generation;
  metadata.raw_payload_bytes = raw_payload_bytes;
  metadata.target_kind = target;
  metadata.state = kSlotReservedWaitDataOrCommit;
  metadata.valid_operand_mask = kOperandSelectedFetchValid;
  metadata.required_operand_mask =
      static_cast<uint8_t>(kOperandSelectedFetchValid |
                           kOperandRawPayloadValid);
  metadata.pending_response_count = 1;
  metadata.producer_commit_required =
      decision.persistent_write_count != 0;
  metadata.selected_fetch = decision.selected_fetch;

  consume_reservation_budget(window, cycle);
  build_receipt(metadata, static_cast<uint8_t>(slot_index), receipt);
  ++state->next_reservation_id;
  ++state->next_reservation_age;
  return kStatusOk;
}

bool receipt_matches(const slot_metadata_v0 &metadata,
                     const reservation_receipt_v0 &receipt) {
  return receipt.valid == 1 &&
         metadata.state != kSlotFree &&
         metadata.reservation_id == receipt.reservation_id &&
         metadata.reservation_age == receipt.reservation_age &&
         metadata.operation_seq == receipt.operation_seq &&
         metadata.commit_epoch == receipt.commit_epoch &&
         metadata.slot_generation == receipt.slot_generation &&
         metadata.target_kind == receipt.target_kind &&
         metadata.raw_payload_bytes == receipt.raw_payload_bytes &&
         private_frontier::owners_equal(metadata.owner, receipt.owner);
}

bool producer_identity_matches(
    const slot_metadata_v0 &metadata,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    uint32_t commit_epoch) {
  return metadata.state != kSlotFree &&
         metadata.operation_seq == operation_seq &&
         metadata.commit_epoch == commit_epoch &&
         private_frontier::owners_equal(metadata.owner, owner);
}

status_kind push_ready_index(ready_fifo_v0 *fifo, uint8_t capacity,
                             uint8_t slot_index, uint64_t reservation_age) {
  if (fifo->count >= capacity || reservation_age == 0) {
    return kStatusReadyFifoInvariant;
  }
  uint8_t insert = fifo->count;
  while (insert != 0 &&
         fifo->reservation_ages[insert - 1] > reservation_age) {
    fifo->indices[insert] = fifo->indices[insert - 1];
    fifo->reservation_ages[insert] =
        fifo->reservation_ages[insert - 1];
    --insert;
  }
  fifo->indices[insert] = slot_index;
  fifo->reservation_ages[insert] = reservation_age;
  ++fifo->count;
  return kStatusOk;
}

status_kind maybe_make_ready(slot_metadata_v0 *metadata,
                             ready_fifo_v0 *fifo, uint8_t capacity,
                             uint8_t slot_index) {
  if (metadata->state != kSlotReservedWaitDataOrCommit ||
      metadata->valid_operand_mask != metadata->required_operand_mask ||
      metadata->pending_response_count != 0 ||
      metadata->producer_commit_complete == 0) {
    return kStatusOk;
  }
  if (metadata->ready_enqueued != 0) return kStatusReadyFifoInvariant;
  const status_kind push_status =
      push_ready_index(fifo, capacity, slot_index,
                       metadata->reservation_age);
  if (push_status != kStatusOk) return push_status;
  metadata->state = kSlotReady;
  metadata->ready_enqueued = 1;
  return kStatusOk;
}

template <typename Slot>
status_kind fill_in_queue(Slot *slots, uint8_t capacity,
                          ready_fifo_v0 *fifo,
                          const reservation_receipt_v0 &reservation,
                          const uint8_t *raw_payload,
                          uint16_t raw_payload_bytes) {
  if (reservation.slot_index >= capacity) return kStatusUnknownReservation;
  Slot &slot = slots[reservation.slot_index];
  if (!receipt_matches(slot.metadata, reservation)) {
    return slot.metadata.state == kSlotFree ? kStatusUnknownReservation
                                            : kStatusStaleReservation;
  }
  if (raw_payload_bytes != slot.metadata.raw_payload_bytes ||
      raw_payload_bytes > sizeof(slot.raw_payload)) {
    return kStatusPayloadShapeMismatch;
  }
  if ((slot.metadata.valid_operand_mask & kOperandRawPayloadValid) != 0) {
    return kStatusDuplicateRawPayload;
  }
  std::memcpy(slot.raw_payload, raw_payload, raw_payload_bytes);
  slot.metadata.valid_operand_mask |= kOperandRawPayloadValid;
  slot.metadata.pending_response_count = 0;
  return maybe_make_ready(&slot.metadata, fifo, capacity,
                          reservation.slot_index);
}

template <typename Slot>
int find_producer_slot(
    Slot *slots, uint8_t capacity,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    uint32_t commit_epoch) {
  int selected = -1;
  for (unsigned index = 0; index < capacity; ++index) {
    if (producer_identity_matches(slots[index].metadata, owner,
                                  operation_seq, commit_epoch)) {
      if (selected >= 0) return -2;
      selected = static_cast<int>(index);
    }
  }
  return selected;
}

template <typename Slot>
status_kind complete_in_queue(
    Slot *slots, uint8_t capacity, ready_fifo_v0 *fifo, int slot_index,
    reservation_receipt_v0 *reservation) {
  Slot &slot = slots[slot_index];
  if (slot.metadata.producer_commit_complete != 0) {
    return kStatusDuplicateProducerCommit;
  }
  slot.metadata.producer_commit_complete = 1;
  build_receipt(slot.metadata, static_cast<uint8_t>(slot_index),
                reservation);
  return maybe_make_ready(&slot.metadata, fifo, capacity,
                          static_cast<uint8_t>(slot_index));
}

template <typename Slot>
status_kind pop_from_queue(Slot *slots, uint8_t capacity,
                           ready_fifo_v0 *fifo, target_kind target,
                           bool unit_input_accepts,
                           operation_packet_v0 *packet) {
  if (fifo->count == 0) return kStatusNoReadyOperation;
  if (!unit_input_accepts) return kStatusUnitInputBackpressure;
  const uint8_t slot_index = fifo->indices[0];
  if (slot_index >= capacity) return kStatusReadyFifoInvariant;
  Slot &slot = slots[slot_index];
  if (slot.metadata.state != kSlotReady ||
      slot.metadata.ready_enqueued != 1 ||
      slot.metadata.target_kind != target ||
      slot.metadata.valid_operand_mask !=
          slot.metadata.required_operand_mask ||
      slot.metadata.producer_commit_complete != 1 ||
      slot.metadata.raw_payload_bytes > sizeof(slot.raw_payload)) {
    return kStatusReadyFifoInvariant;
  }

  *packet = operation_packet_v0();
  packet->owner = slot.metadata.owner;
  packet->reservation_id = slot.metadata.reservation_id;
  packet->reservation_age = slot.metadata.reservation_age;
  packet->operation_seq = slot.metadata.operation_seq;
  packet->commit_epoch = slot.metadata.commit_epoch;
  packet->slot_generation = slot.metadata.slot_generation;
  packet->raw_payload_bytes = slot.metadata.raw_payload_bytes;
  packet->target_kind = target;
  packet->valid = 1;
  packet->selected_fetch = slot.metadata.selected_fetch;
  std::memcpy(packet->raw_payload, slot.raw_payload,
              slot.metadata.raw_payload_bytes);

  slot.metadata.state = kSlotIssued;
  for (unsigned index = 1; index < fifo->count; ++index) {
    fifo->indices[index - 1] = fifo->indices[index];
    fifo->reservation_ages[index - 1] =
        fifo->reservation_ages[index];
  }
  fifo->indices[fifo->count - 1] = 0;
  fifo->reservation_ages[fifo->count - 1] = 0;
  --fifo->count;
  const uint32_t generation = slot.metadata.slot_generation;
  std::memset(&slot, 0, sizeof(slot));
  slot.metadata.slot_generation = generation;
  return kStatusOk;
}

template <typename Slot>
uint8_t count_active(const Slot *slots, uint8_t capacity) {
  uint8_t count = 0;
  for (unsigned index = 0; index < capacity; ++index) {
    count += slots[index].metadata.state != kSlotFree;
  }
  return count;
}

}  // namespace

status_kind initialize(engine_state_v0 *state, const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!valid_config(config)) return kStatusInvalidConfiguration;
  std::memset(state, 0, sizeof(*state));
  state->config = config;
  state->next_reservation_id = 1;
  state->next_reservation_age = 1;
  state->initialized = 1;
  return kStatusOk;
}

status_kind classify_selected_fetch(
    const typed_node::selected_child_fetch_work_item_v0 &selected_fetch,
    target_kind *target, uint16_t *raw_payload_bytes) {
  if (target == NULL || raw_payload_bytes == NULL) {
    return kStatusInvalidArgument;
  }
  *target = kTargetInvalid;
  *raw_payload_bytes = 0;
  switch (selected_fetch.child.payload_kind) {
    case typed_node::kInternalPayloadKind:
      *target = kTargetNode;
      *raw_payload_bytes = kNodeRawPayloadBytes;
      break;
    case typed_node::kProceduralPayloadKind:
    case typed_node::kQuadPayloadKind:
      *target = kTargetPrimitive;
      *raw_payload_bytes = kPrimitiveRawPayloadBytes;
      break;
    case typed_node::kInstancePayloadKind:
      *target = kTargetInstance;
      *raw_payload_bytes = kInstanceRawPayloadBytes;
      break;
    default:
      return kStatusInvalidSelectedFetch;
  }
  if (!selected_fetch_shape_valid(selected_fetch, *target,
                                  *raw_payload_bytes)) {
    *target = kTargetInvalid;
    *raw_payload_bytes = 0;
    return kStatusInvalidSelectedFetch;
  }
  return kStatusOk;
}

status_kind try_reserve_prefill(
    engine_state_v0 *state,
    const stack_commit::forwarding_decision_input_v0 &decision,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      decision.operation_seq == 0 || decision.commit_epoch == 0 ||
      !valid_owner(decision.owner) ||
      !bytes_are_zero(decision.reserved_zero,
                      sizeof(decision.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *receipt = reservation_receipt_v0();
  target_kind target = kTargetInvalid;
  uint16_t raw_payload_bytes = 0;
  const status_kind classify_status =
      classify_selected_fetch(decision.selected_fetch, &target,
                              &raw_payload_bytes);
  if (classify_status != kStatusOk) return classify_status;

  switch (target) {
    case kTargetNode:
      return reserve_in_queue(
          state->node_slots, state->config.node_capacity,
          &state->node_window, state->config.node_reservation_width, state,
          decision, target, raw_payload_bytes, reservation_cycle, receipt);
    case kTargetPrimitive:
      return reserve_in_queue(
          state->primitive_slots, state->config.primitive_capacity,
          &state->primitive_window,
          state->config.primitive_reservation_width, state, decision, target,
          raw_payload_bytes, reservation_cycle, receipt);
    case kTargetInstance:
      return reserve_in_queue(
          state->instance_slots, state->config.instance_capacity,
          &state->instance_window, state->config.instance_reservation_width,
          state, decision, target, raw_payload_bytes, reservation_cycle,
          receipt);
    case kTargetInvalid:
      break;
  }
  return kStatusInvalidSelectedFetch;
}

status_kind fill_raw_payload(engine_state_v0 *state,
                             const reservation_receipt_v0 &reservation,
                             const uint8_t *raw_payload,
                             uint16_t raw_payload_bytes) {
  if (state == NULL || raw_payload == NULL || state->initialized != 1 ||
      reservation.valid != 1) {
    return kStatusInvalidArgument;
  }
  engine_state_v0 staged = *state;
  status_kind status = kStatusInvalidArgument;
  switch (static_cast<target_kind>(reservation.target_kind)) {
    case kTargetNode:
      status = fill_in_queue(
          staged.node_slots, staged.config.node_capacity, &staged.node_ready,
          reservation, raw_payload, raw_payload_bytes);
      break;
    case kTargetPrimitive:
      status = fill_in_queue(
          staged.primitive_slots, staged.config.primitive_capacity,
          &staged.primitive_ready, reservation, raw_payload,
          raw_payload_bytes);
      break;
    case kTargetInstance:
      status = fill_in_queue(
          staged.instance_slots, staged.config.instance_capacity,
          &staged.instance_ready, reservation, raw_payload,
          raw_payload_bytes);
      break;
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  if (status == kStatusOk) *state = staged;
  return status;
}

status_kind complete_producer_commit(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    uint32_t commit_epoch, reservation_receipt_v0 *reservation) {
  if (state == NULL || reservation == NULL || state->initialized != 1 ||
      !valid_owner(owner) || operation_seq == 0 || commit_epoch == 0) {
    return kStatusInvalidArgument;
  }
  *reservation = reservation_receipt_v0();
  engine_state_v0 staged = *state;
  reservation_receipt_v0 staged_reservation = {};
  const int node_slot =
      find_producer_slot(staged.node_slots, staged.config.node_capacity,
                         owner, operation_seq, commit_epoch);
  const int primitive_slot = find_producer_slot(
      staged.primitive_slots, staged.config.primitive_capacity, owner,
      operation_seq, commit_epoch);
  const int instance_slot = find_producer_slot(
      staged.instance_slots, staged.config.instance_capacity, owner,
      operation_seq, commit_epoch);
  const unsigned match_count =
      (node_slot >= 0) + (primitive_slot >= 0) + (instance_slot >= 0);
  if (node_slot == -2 || primitive_slot == -2 || instance_slot == -2 ||
      match_count > 1) {
    return kStatusStaleReservation;
  }
  status_kind status = kStatusUnknownProducerCommit;
  if (node_slot >= 0) {
    status = complete_in_queue(
        staged.node_slots, staged.config.node_capacity, &staged.node_ready,
        node_slot, &staged_reservation);
  } else if (primitive_slot >= 0) {
    status = complete_in_queue(
        staged.primitive_slots, staged.config.primitive_capacity,
        &staged.primitive_ready, primitive_slot, &staged_reservation);
  } else if (instance_slot >= 0) {
    status = complete_in_queue(
        staged.instance_slots, staged.config.instance_capacity,
        &staged.instance_ready, instance_slot, &staged_reservation);
  }
  if (status == kStatusOk) {
    *state = staged;
    *reservation = staged_reservation;
  }
  return status;
}

status_kind pop_ready_operation(engine_state_v0 *state, target_kind target,
                                bool unit_input_accepts,
                                operation_packet_v0 *packet) {
  if (state == NULL || packet == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  switch (target) {
    case kTargetNode:
      return pop_from_queue(
          state->node_slots, state->config.node_capacity, &state->node_ready,
          target, unit_input_accepts, packet);
    case kTargetPrimitive:
      return pop_from_queue(
          state->primitive_slots, state->config.primitive_capacity,
          &state->primitive_ready, target, unit_input_accepts, packet);
    case kTargetInstance:
      return pop_from_queue(
          state->instance_slots, state->config.instance_capacity,
          &state->instance_ready, target, unit_input_accepts, packet);
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  return kStatusInvalidArgument;
}

uint8_t active_slot_count(const engine_state_v0 &state, target_kind target) {
  if (state.initialized != 1) return 0;
  switch (target) {
    case kTargetNode:
      return count_active(state.node_slots, state.config.node_capacity);
    case kTargetPrimitive:
      return count_active(state.primitive_slots,
                          state.config.primitive_capacity);
    case kTargetInstance:
      return count_active(state.instance_slots,
                          state.config.instance_capacity);
    case kTargetInvalid:
      return 0;
  }
  return 0;
}

uint8_t ready_slot_count(const engine_state_v0 &state, target_kind target) {
  if (state.initialized != 1) return 0;
  switch (target) {
    case kTargetNode:
      return state.node_ready.count;
    case kTargetPrimitive:
      return state.primitive_ready.count;
    case kTargetInstance:
      return state.instance_ready.count;
    case kTargetInvalid:
      return 0;
  }
  return 0;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfiguration:
      return "invalid_configuration";
    case kStatusInvalidSelectedFetch:
      return "invalid_selected_fetch";
    case kStatusCapacityBackpressure:
      return "capacity_backpressure";
    case kStatusReservationBudgetBackpressure:
      return "reservation_budget_backpressure";
    case kStatusReservationSequenceExhausted:
      return "reservation_sequence_exhausted";
    case kStatusUnknownReservation:
      return "unknown_reservation";
    case kStatusStaleReservation:
      return "stale_reservation";
    case kStatusPayloadShapeMismatch:
      return "payload_shape_mismatch";
    case kStatusDuplicateRawPayload:
      return "duplicate_raw_payload";
    case kStatusUnknownProducerCommit:
      return "unknown_producer_commit";
    case kStatusDuplicateProducerCommit:
      return "duplicate_producer_commit";
    case kStatusReadyFifoInvariant:
      return "ready_fifo_invariant";
    case kStatusNoReadyOperation:
      return "no_ready_operation";
    case kStatusUnitInputBackpressure:
      return "unit_input_backpressure";
  }
  return "unknown";
}

}  // namespace fetch_target
}  // namespace v04
}  // namespace rtcore
