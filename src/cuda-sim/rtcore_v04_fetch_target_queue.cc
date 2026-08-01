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
  const bool address_aligned =
      child.payload_offset <=
          std::numeric_limits<uint64_t>::max() - context.device_base &&
      ((context.device_base + child.payload_offset) &
       (private_frontier::kSharedAccessChunkBytes - 1)) == 0;
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
         target != kTargetInvalid && address_aligned;
}

bool target_reference_shape_valid(
    const target_reference_v0 &reference, target_kind target,
    uint16_t payload_bytes, uint64_t raw_payload_base_address) {
  typed_node::replay_cursor_v0 replay_cursor = {};
  const bool payload_kind_valid =
      reference.payload_kind == typed_node::kInternalPayloadKind ||
      reference.payload_kind == typed_node::kProceduralPayloadKind ||
      reference.payload_kind == typed_node::kQuadPayloadKind ||
      reference.payload_kind == typed_node::kInstancePayloadKind;
  const bool kind_matches_target =
      reference.payload_kind == typed_node::kInstancePayloadKind
          ? target == kTargetInstance
          : (reference.payload_kind == typed_node::kInternalPayloadKind
                 ? target == kTargetNode
                 : target == kTargetPrimitive);
  if (!payload_kind_valid ||
      reference.payload_byte_count != payload_bytes ||
      !kind_matches_target) {
    return false;
  }
  const bool instance_blas_root_source_valid =
      reference.source_kind !=
          kTargetReferenceInstanceBlasRootProducer ||
      (target == kTargetNode &&
       reference.payload_kind == typed_node::kInternalPayloadKind &&
       reference.payload_byte_count == kNodeRawPayloadBytes &&
       reference.level == typed_node::kLevelBlas &&
       reference.proxy_delegated == 0);
  return reference.level >= typed_node::kLevelTlas &&
         reference.level <= typed_node::kLevelBlas &&
         reference.source_kind >= kTargetReferenceRootCompatibilityProxy &&
         reference.source_kind <=
             kTargetReferenceInstanceBlasRootProducer &&
         reference.proxy_delegated <= 1 &&
         instance_blas_root_source_valid &&
         decode_replay_cursor(reference.replay_control,
                              &replay_cursor) &&
         (target == kTargetNode ||
          replay_cursor.anchor_valid == 0) &&
         bytes_are_zero(reference.reserved_zero,
                        sizeof(reference.reserved_zero)) &&
         (reference.payload_offset & uint64_t{0x3f}) == 0 &&
         (raw_payload_base_address &
          (private_frontier::kSharedAccessChunkBytes - 1)) == 0 &&
         std::isfinite(fp32_value(reference.near_t_bits));
}

bool pending_parent_resume_valid(const reservation_input_v0 &input) {
  if (input.pending_parent_resume_valid > 1) return false;
  if (input.pending_parent_resume_valid == 0) {
    const short_stack::entry_v0 zero = {};
    return std::memcmp(&input.pending_parent_resume, &zero,
                       sizeof(zero)) == 0;
  }
  return input.target_kind == kTargetNode &&
         short_stack::validate_entry(input.pending_parent_resume) &&
         short_stack::control_kind(
             input.pending_parent_resume.control) ==
             short_stack::kEntryParentResume &&
         short_stack::control_domain(
             input.pending_parent_resume.control) ==
             (input.target_reference.level == typed_node::kLevelBlas
                  ? short_stack::kDomainBlas
                  : short_stack::kDomainTlas);
}

bool reservation_identity_valid(const reservation_input_v0 &input) {
  const bool producer_tag_valid =
      input.producer_commit_required == 0
          ? input.producer_operation_seq == 0 &&
                input.producer_commit_epoch == 0
          : input.producer_operation_seq != 0 &&
                input.producer_commit_epoch != 0;
  const uint8_t private_mask = static_cast<uint8_t>(
      input.required_operand_mask &
      (kOperandMutableRayValid | kOperandDecodeContextValid |
       kOperandCommittedHitValid | kOperandCurrentInstanceValid));
  const uint8_t root_private_mask = static_cast<uint8_t>(
      kOperandMutableRayValid | kOperandDecodeContextValid |
      kOperandCommittedHitValid);
  const uint8_t expected_private_mask = static_cast<uint8_t>(
      root_private_mask |
      (input.target_kind == kTargetPrimitive
           ? kOperandCurrentInstanceValid
           : 0));
  return valid_owner(input.owner) && input.target_operation_seq != 0 &&
         producer_tag_valid && input.raw_payload_base_address != 0 &&
         input.raw_payload_bytes != 0 &&
         input.target_kind >= kTargetNode &&
         input.target_kind <= kTargetInstance &&
         input.producer_commit_required <= 1 &&
         (input.required_operand_mask & kOperandTargetReferenceValid) != 0 &&
         (input.required_operand_mask & kOperandRawPayloadValid) != 0 &&
         (input.required_operand_mask & kOperandParentFrameValid) == 0 &&
         (input.forwarded_operand_mask &
          ~input.required_operand_mask) == 0 &&
         (input.forwarded_operand_mask &
          ~(kOperandRayPolicyValid)) == 0 &&
         (private_mask == 0 || private_mask == expected_private_mask) &&
         bytes_are_zero(input.forwarded_ray_policy.reserved_zero,
                        sizeof(input.forwarded_ray_policy.reserved_zero)) &&
         bytes_are_zero(input.reserved_zero, sizeof(input.reserved_zero)) &&
         pending_parent_resume_valid(input) &&
         target_reference_shape_valid(
             input.target_reference,
             static_cast<target_kind>(input.target_kind),
             input.raw_payload_bytes, input.raw_payload_base_address);
}

bool instance_restore_reservation_identity_valid(
    const instance_restore_reservation_input_v0 &input) {
  return valid_owner(input.owner) && input.target_operation_seq != 0 &&
         input.required_operand_mask == kOperandParentFrameValid &&
         bytes_are_zero(input.reserved_zero, sizeof(input.reserved_zero));
}

uint16_t raw_payload_bytes_for_target(target_kind target) {
  switch (target) {
    case kTargetNode:
      return kNodeRawPayloadBytes;
    case kTargetPrimitive:
      return kPrimitiveRawPayloadBytes;
    case kTargetInstance:
      return kInstanceRawPayloadBytes;
    case kTargetInvalid:
      return 0;
  }
  return 0;
}

bool recovery_reservation_identity_valid(
    const recovery_reservation_input_v0 &input) {
  const uint8_t complete_operand_mask = static_cast<uint8_t>(
      kOperandTargetReferenceValid | kOperandRawPayloadValid |
      kOperandMutableRayValid | kOperandRayPolicyValid |
      kOperandDecodeContextValid | kOperandCommittedHitValid);
  const uint8_t expected_operand_mask = static_cast<uint8_t>(
      complete_operand_mask |
      (input.target_kind == kTargetPrimitive
           ? kOperandCurrentInstanceValid
           : 0));
  return valid_owner(input.owner) &&
         input.target_operation_seq != 0 &&
         input.target_kind >= kTargetNode &&
         input.target_kind <= kTargetInstance &&
         input.required_operand_mask == expected_operand_mask &&
         bytes_are_zero(input.reserved_zero,
                        sizeof(input.reserved_zero));
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
  receipt->raw_payload_base_address =
      metadata.raw_payload_base_address;
  receipt->target_operation_seq = metadata.target_operation_seq;
  receipt->producer_operation_seq = metadata.producer_operation_seq;
  receipt->producer_commit_epoch = metadata.producer_commit_epoch;
  receipt->slot_generation = metadata.slot_generation;
  receipt->private_layout_profile_id =
      metadata.private_layout_profile_id;
  receipt->bvh_format_profile_id =
      metadata.bvh_format_profile_id;
  receipt->raw_payload_bytes = metadata.raw_payload_bytes;
  receipt->target_kind = metadata.target_kind;
  receipt->slot_index = slot_index;
  receipt->raw_chunk_count = metadata.expected_raw_chunk_count;
  receipt->private_chunk_count = metadata.expected_private_chunk_count;
  receipt->producer_commit_required =
      metadata.producer_commit_required;
  receipt->operation_kind = metadata.operation_kind;
  receipt->private_storage_profile =
      metadata.private_storage_profile;
  receipt->valid = 1;
}

template <typename Slot>
status_kind reserve_in_queue(
    Slot *slots, uint8_t capacity, reservation_window_v0 *window,
    uint8_t reservation_width, engine_state_v0 *state,
    const reservation_input_v0 &input,
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
  metadata.owner = input.owner;
  metadata.reservation_id = state->next_reservation_id;
  metadata.reservation_age = state->next_reservation_age;
  metadata.raw_payload_base_address = input.raw_payload_base_address;
  metadata.target_operation_seq = input.target_operation_seq;
  metadata.producer_operation_seq = input.producer_operation_seq;
  metadata.producer_commit_epoch = input.producer_commit_epoch;
  metadata.slot_generation = slot_generation;
  metadata.private_layout_profile_id =
      private_frontier::kLayoutProfileId;
  metadata.bvh_format_profile_id =
      typed_node::kGenRtDerivedProfileId;
  metadata.private_storage_profile =
      private_storage::kProfileLegacyShared832;
  metadata.raw_payload_bytes = raw_payload_bytes;
  metadata.target_kind = target;
  metadata.operation_kind = kOperationFetchTarget;
  metadata.state = kSlotReservedWaitDataOrCommit;
  metadata.valid_operand_mask = static_cast<uint8_t>(
      kOperandTargetReferenceValid | input.forwarded_operand_mask);
  metadata.required_operand_mask = input.required_operand_mask;
  metadata.expected_raw_chunk_count = static_cast<uint8_t>(
      raw_payload_bytes / private_frontier::kSharedAccessChunkBytes);
  metadata.pending_raw_response_count = metadata.expected_raw_chunk_count;
  const uint8_t private_required_mask = static_cast<uint8_t>(
      metadata.required_operand_mask &
      (kOperandMutableRayValid | kOperandDecodeContextValid |
       kOperandCommittedHitValid | kOperandCurrentInstanceValid));
  metadata.expected_private_chunk_count =
      private_required_mask == 0
          ? 0
          : static_cast<uint8_t>(
                target == kTargetPrimitive ? 9 : 7);
  metadata.pending_private_response_count =
      metadata.expected_private_chunk_count;
  metadata.producer_commit_required = input.producer_commit_required;
  metadata.producer_commit_complete =
      input.producer_commit_required == 0;
  metadata.target_reference = input.target_reference;
  metadata.pending_parent_resume =
      input.pending_parent_resume;
  metadata.pending_parent_resume_valid =
      input.pending_parent_resume_valid;
  if ((input.forwarded_operand_mask & kOperandRayPolicyValid) != 0) {
    metadata.ray_policy = input.forwarded_ray_policy;
  }

  consume_reservation_budget(window, cycle);
  build_receipt(metadata, static_cast<uint8_t>(slot_index), receipt);
  ++state->next_reservation_id;
  ++state->next_reservation_age;
  return kStatusOk;
}

template <typename Slot>
status_kind reserve_recovery_in_queue(
    Slot *slots, uint8_t capacity, reservation_window_v0 *window,
    uint8_t reservation_width, engine_state_v0 *state,
    const recovery_reservation_input_v0 &input, target_kind target,
    uint64_t cycle, reservation_receipt_v0 *receipt) {
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

  const uint16_t raw_payload_bytes =
      raw_payload_bytes_for_target(target);
  if (raw_payload_bytes == 0) return kStatusInvalidSelectedFetch;
  const uint32_t slot_generation =
      slots[slot_index].metadata.slot_generation + 1;
  std::memset(&slots[slot_index], 0, sizeof(slots[slot_index]));
  slot_metadata_v0 &metadata = slots[slot_index].metadata;
  metadata.owner = input.owner;
  metadata.reservation_id = state->next_reservation_id;
  metadata.reservation_age = state->next_reservation_age;
  metadata.target_operation_seq = input.target_operation_seq;
  metadata.slot_generation = slot_generation;
  metadata.private_layout_profile_id =
      private_frontier::kLayoutProfileId;
  metadata.bvh_format_profile_id =
      typed_node::kGenRtDerivedProfileId;
  metadata.private_storage_profile =
      private_storage::kProfileLegacyShared832;
  metadata.raw_payload_bytes = raw_payload_bytes;
  metadata.target_kind = target;
  metadata.operation_kind = kOperationFetchTarget;
  metadata.state = kSlotReservedWaitDataOrCommit;
  metadata.required_operand_mask = input.required_operand_mask;
  metadata.expected_raw_chunk_count = static_cast<uint8_t>(
      raw_payload_bytes / private_frontier::kSharedAccessChunkBytes);
  metadata.pending_raw_response_count =
      metadata.expected_raw_chunk_count;
  metadata.expected_private_chunk_count =
      target == kTargetPrimitive ? 9 : 7;
  metadata.pending_private_response_count =
      metadata.expected_private_chunk_count;
  metadata.producer_commit_complete = 1;
  metadata.recovery_descriptor_pending = 1;
  metadata.pending_recovery_descriptor_response_count =
      kStackSpillRecoveryChunks;

  consume_reservation_budget(window, cycle);
  build_receipt(metadata, static_cast<uint8_t>(slot_index), receipt);
  ++state->next_reservation_id;
  ++state->next_reservation_age;
  return kStatusOk;
}

status_kind reserve_instance_restore_in_queue(
    instance_slot_v0 *slots, uint8_t capacity,
    reservation_window_v0 *window, uint8_t reservation_width,
    engine_state_v0 *state,
    const instance_restore_reservation_input_v0 &input, uint64_t cycle,
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
  metadata.owner = input.owner;
  metadata.reservation_id = state->next_reservation_id;
  metadata.reservation_age = state->next_reservation_age;
  metadata.target_operation_seq = input.target_operation_seq;
  metadata.slot_generation = slot_generation;
  metadata.private_layout_profile_id =
      private_frontier::kLayoutProfileId;
  metadata.bvh_format_profile_id =
      typed_node::kGenRtDerivedProfileId;
  metadata.private_storage_profile =
      private_storage::kProfileLegacyShared832;
  metadata.target_kind = kTargetInstance;
  metadata.operation_kind = kOperationInstanceRestoreParent;
  metadata.state = kSlotReservedWaitDataOrCommit;
  metadata.required_operand_mask = input.required_operand_mask;
  metadata.expected_private_chunk_count = 5;
  metadata.pending_private_response_count = 5;
  metadata.producer_commit_complete = 1;

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
         metadata.target_operation_seq == receipt.target_operation_seq &&
         metadata.producer_operation_seq ==
             receipt.producer_operation_seq &&
         metadata.producer_commit_epoch ==
             receipt.producer_commit_epoch &&
         metadata.slot_generation == receipt.slot_generation &&
         metadata.private_layout_profile_id ==
             receipt.private_layout_profile_id &&
         metadata.bvh_format_profile_id ==
             receipt.bvh_format_profile_id &&
         metadata.private_storage_profile ==
             receipt.private_storage_profile &&
         metadata.target_kind == receipt.target_kind &&
         metadata.raw_payload_bytes == receipt.raw_payload_bytes &&
         metadata.expected_raw_chunk_count == receipt.raw_chunk_count &&
         metadata.expected_private_chunk_count ==
             receipt.private_chunk_count &&
         metadata.operation_kind == receipt.operation_kind &&
         private_frontier::owners_equal(metadata.owner, receipt.owner);
}

bool producer_identity_matches(
    const slot_metadata_v0 &metadata,
    const private_frontier::owner_binding_v0 &owner, uint32_t operation_seq,
    uint32_t commit_epoch) {
  return metadata.state != kSlotFree &&
         metadata.producer_commit_required != 0 &&
         metadata.producer_operation_seq == operation_seq &&
         metadata.producer_commit_epoch == commit_epoch &&
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
  const uint8_t private_required_mask = static_cast<uint8_t>(
      metadata->required_operand_mask &
      (kOperandMutableRayValid | kOperandDecodeContextValid |
       kOperandCommittedHitValid | kOperandParentFrameValid));
  const bool raw_global_operands_complete =
      (metadata->required_operand_mask & kOperandRawPayloadValid) == 0 ||
      ((metadata->valid_operand_mask & kOperandRawPayloadValid) != 0 &&
       metadata->pending_raw_response_count == 0);
  const bool private_shared_operands_complete =
      (metadata->valid_operand_mask & private_required_mask) ==
          private_required_mask &&
      metadata->pending_private_response_count == 0;
  const bool producer_commit_gate_complete =
      metadata->producer_commit_complete != 0;
  const bool required_field_mask_complete =
      metadata->valid_operand_mask == metadata->required_operand_mask;
  if (metadata->state != kSlotReservedWaitDataOrCommit ||
      metadata->recovery_descriptor_pending != 0 ||
      !raw_global_operands_complete ||
      !private_shared_operands_complete ||
      !producer_commit_gate_complete ||
      !required_field_mask_complete) {
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
  if (slot.metadata.raw_payload_base_address !=
      reservation.raw_payload_base_address) {
    return kStatusStaleReservation;
  }
  if (raw_payload_bytes != slot.metadata.raw_payload_bytes ||
      raw_payload_bytes > sizeof(slot.raw_payload)) {
    return kStatusPayloadShapeMismatch;
  }
  if ((slot.metadata.valid_operand_mask & kOperandRawPayloadValid) != 0) {
    return kStatusDuplicateRawPayload;
  }
  if (slot.metadata.received_raw_chunk_mask != 0) {
    return kStatusDuplicateRawPayload;
  }
  std::memcpy(slot.raw_payload, raw_payload, raw_payload_bytes);
  slot.metadata.received_raw_chunk_mask = static_cast<uint8_t>(
      (uint8_t{1} << slot.metadata.expected_raw_chunk_count) - 1);
  slot.metadata.valid_operand_mask |= kOperandRawPayloadValid;
  slot.metadata.pending_raw_response_count = 0;
  return maybe_make_ready(&slot.metadata, fifo, capacity,
                          reservation.slot_index);
}

template <typename Slot>
status_kind fill_chunk_in_queue(
    Slot *slots, uint8_t capacity, ready_fifo_v0 *fifo,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, const uint8_t *raw_payload_chunk,
    uint8_t chunk_bytes) {
  if (reservation.slot_index >= capacity) return kStatusUnknownReservation;
  Slot &slot = slots[reservation.slot_index];
  if (!receipt_matches(slot.metadata, reservation)) {
    return slot.metadata.state == kSlotFree ? kStatusUnknownReservation
                                            : kStatusStaleReservation;
  }
  if (slot.metadata.raw_payload_base_address !=
      reservation.raw_payload_base_address) {
    return kStatusStaleReservation;
  }
  if (chunk_count != slot.metadata.expected_raw_chunk_count ||
      chunk_count == 0 || chunk_count > 4 || chunk_id >= chunk_count ||
      chunk_bytes != private_frontier::kSharedAccessChunkBytes ||
      static_cast<uint16_t>(chunk_count) * chunk_bytes !=
          slot.metadata.raw_payload_bytes ||
      static_cast<size_t>(chunk_id) * chunk_bytes + chunk_bytes >
          sizeof(slot.raw_payload)) {
    return kStatusChunkShapeMismatch;
  }
  const uint8_t chunk_bit = static_cast<uint8_t>(uint8_t{1} << chunk_id);
  if ((slot.metadata.received_raw_chunk_mask & chunk_bit) != 0 ||
      (slot.metadata.valid_operand_mask & kOperandRawPayloadValid) != 0) {
    return kStatusDuplicateRawChunk;
  }
  std::memcpy(slot.raw_payload +
                  static_cast<size_t>(chunk_id) * chunk_bytes,
              raw_payload_chunk, chunk_bytes);
  slot.metadata.received_raw_chunk_mask |= chunk_bit;
  if (slot.metadata.pending_raw_response_count == 0) {
    return kStatusReadyFifoInvariant;
  }
  --slot.metadata.pending_raw_response_count;
  const uint8_t expected_mask =
      static_cast<uint8_t>((uint8_t{1} << chunk_count) - 1);
  if (slot.metadata.received_raw_chunk_mask == expected_mask) {
    if (slot.metadata.pending_raw_response_count != 0) {
      return kStatusReadyFifoInvariant;
    }
    slot.metadata.valid_operand_mask |= kOperandRawPayloadValid;
  }
  return maybe_make_ready(&slot.metadata, fifo, capacity,
                          reservation.slot_index);
}

struct expected_private_chunk_v0 {
  uint8_t field_kind;
  uint16_t slot_chunk_offset;
  uint32_t byte_mask;
};

const expected_private_chunk_v0 kExpectedRootPrivateChunks[7] = {
    {private_frontier::kFieldMutableRayState, 0x000, 0xffffffffu},
    {private_frontier::kFieldMutableRayState, 0x020, 0x00000fffu},
    {private_frontier::kFieldAsDecodeContext, 0x040, 0xffffff00u},
    {private_frontier::kFieldAsDecodeContext, 0x060, 0x0000ffffu},
    {private_frontier::kFieldCommittedHit, 0x080, 0xffffff00u},
    {private_frontier::kFieldCommittedHit, 0x0a0, 0xffffffffu},
    {private_frontier::kFieldCommittedHit, 0x0c0, 0x000000ffu},
};

const expected_private_chunk_v0 kExpectedPrimitivePrivateChunks[9] = {
    {private_frontier::kFieldMutableRayState, 0x000, 0xffffffffu},
    {private_frontier::kFieldMutableRayState, 0x020, 0x00000fffu},
    {private_frontier::kFieldAsDecodeContext, 0x040, 0xffffff00u},
    {private_frontier::kFieldAsDecodeContext, 0x060, 0x0000ffffu},
    {private_frontier::kFieldCommittedHit, 0x080, 0xffffff00u},
    {private_frontier::kFieldCommittedHit, 0x0a0, 0xffffffffu},
    {private_frontier::kFieldCommittedHit, 0x0c0, 0x000000ffu},
    {private_frontier::kFieldCurrentInstance, 0x060, 0xffff0000u},
    {private_frontier::kFieldCurrentInstance, 0x080, 0x000000ffu},
};

const expected_private_chunk_v0 kExpectedParentFrameChunks[5] = {
    {private_frontier::kFieldParentFrame, 0x200, 0xffffff00u},
    {private_frontier::kFieldParentFrame, 0x220, 0xffffffffu},
    {private_frontier::kFieldParentFrame, 0x240, 0xffffffffu},
    {private_frontier::kFieldParentFrame, 0x260, 0xffffffffu},
    {private_frontier::kFieldParentFrame, 0x280, 0x000000ffu},
};

uint8_t *private_field_bytes(slot_metadata_v0 *metadata, uint8_t field_kind,
                             uint16_t *field_offset,
                             uint16_t *field_bytes) {
  switch (field_kind) {
    case private_frontier::kFieldMutableRayState:
      *field_offset = private_frontier::kMutableRayStateOffset;
      *field_bytes = private_frontier::kMutableRayStateBytes;
      return metadata->mutable_ray_bytes;
    case private_frontier::kFieldAsDecodeContext:
      *field_offset = private_frontier::kAsDecodeContextOffset;
      *field_bytes = private_frontier::kAsDecodeContextBytes;
      return metadata->decode_context_bytes;
    case private_frontier::kFieldCommittedHit:
      *field_offset = private_frontier::kCommittedHitOffset;
      *field_bytes = private_frontier::kCommittedHitBytes;
      return metadata->committed_hit_bytes;
    case private_frontier::kFieldCurrentInstance:
      *field_offset = private_frontier::kCurrentInstanceOffset;
      *field_bytes = private_frontier::kCurrentInstanceBytes;
      return metadata->current_instance_bytes;
    case private_frontier::kFieldParentFrame:
      *field_offset = private_frontier::kParentFrameOffset;
      *field_bytes = private_frontier::kParentFrameBytes;
      return metadata->parent_frame_bytes;
    default:
      break;
  }
  return NULL;
}

template <typename Slot>
status_kind fill_private_chunk_in_queue(
    Slot *slots, uint8_t capacity, ready_fifo_v0 *fifo,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint8_t field_kind, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes]) {
  if (reservation.slot_index >= capacity) return kStatusUnknownReservation;
  Slot &slot = slots[reservation.slot_index];
  if (!receipt_matches(slot.metadata, reservation)) {
    return slot.metadata.state == kSlotFree ? kStatusUnknownReservation
                                            : kStatusStaleReservation;
  }
  const bool parent_restore =
      slot.metadata.operation_kind == kOperationInstanceRestoreParent;
  const bool primitive_fetch =
      !parent_restore &&
      slot.metadata.target_kind == kTargetPrimitive;
  const expected_private_chunk_v0 *expected =
      parent_restore ? kExpectedParentFrameChunks
                     : (primitive_fetch
                            ? kExpectedPrimitivePrivateChunks
                            : kExpectedRootPrivateChunks);
  const uint8_t expected_count =
      parent_restore ? 5 : (primitive_fetch ? 9 : 7);
  if (chunk_count != slot.metadata.expected_private_chunk_count ||
      chunk_count != expected_count || chunk_id >= chunk_count ||
      field_kind != expected[chunk_id].field_kind ||
      slot_chunk_offset != expected[chunk_id].slot_chunk_offset ||
      byte_mask != expected[chunk_id].byte_mask) {
    return kStatusPrivateOperandShapeMismatch;
  }
  const uint16_t chunk_bit =
      static_cast<uint16_t>(uint16_t{1} << chunk_id);
  if ((slot.metadata.received_private_chunk_mask & chunk_bit) != 0) {
    return kStatusDuplicatePrivateChunk;
  }

  uint16_t field_offset = 0;
  uint16_t field_bytes = 0;
  uint8_t *destination = private_field_bytes(
      &slot.metadata, field_kind, &field_offset, &field_bytes);
  if (destination == NULL) return kStatusPrivateOperandShapeMismatch;
  for (unsigned byte = 0; byte < private_frontier::kSharedAccessChunkBytes;
       ++byte) {
    if ((byte_mask & (uint32_t{1} << byte)) == 0) {
      if (payload[byte] != 0) return kStatusPrivateOperandShapeMismatch;
      continue;
    }
    const uint32_t slot_offset = slot_chunk_offset + byte;
    if (slot_offset < field_offset ||
        slot_offset >= static_cast<uint32_t>(field_offset) + field_bytes) {
      return kStatusPrivateOperandShapeMismatch;
    }
    destination[slot_offset - field_offset] = payload[byte];
  }

  slot.metadata.received_private_chunk_mask |= chunk_bit;
  if (slot.metadata.pending_private_response_count == 0) {
    return kStatusReadyFifoInvariant;
  }
  --slot.metadata.pending_private_response_count;
  if (slot.metadata.received_private_chunk_mask ==
      static_cast<uint16_t>((uint16_t{1} << chunk_count) - 1)) {
    if (slot.metadata.pending_private_response_count != 0) {
      return kStatusReadyFifoInvariant;
    }
    slot.metadata.valid_operand_mask |=
        parent_restore
            ? static_cast<uint8_t>(kOperandParentFrameValid)
            : static_cast<uint8_t>(
                  kOperandMutableRayValid | kOperandDecodeContextValid |
                  kOperandCommittedHitValid |
                  (primitive_fetch
                       ? kOperandCurrentInstanceValid
                       : 0));
  }
  return maybe_make_ready(&slot.metadata, fifo, capacity,
                          reservation.slot_index);
}

uint8_t consumer_for_target(target_kind target) {
  switch (target) {
    case kTargetNode:
      return private_state_384::operand_plan::kConsumerNode;
    case kTargetPrimitive:
      return private_state_384::operand_plan::kConsumerPrimitive;
    case kTargetInstance:
      return private_state_384::operand_plan::kConsumerInstance;
    case kTargetInvalid:
      return private_state_384::operand_plan::kConsumerInvalid;
  }
  return private_state_384::operand_plan::kConsumerInvalid;
}

bool is_private_state_384_profile(uint8_t profile) {
  return profile == private_storage::kProfileCompressedShared384 ||
         profile == private_storage::kProfileGlobal384;
}

bool operation_key_matches_metadata(
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    const slot_metadata_v0 &metadata) {
  const private_state_384::operand_materializer::operation_identity_v1
      &identity = key.identity;
  return key.private_slot_base_address != 0 &&
         key.private_slot_base_address % private_state_384::kChunkBytes == 0 &&
         key.private_layout_profile_id ==
             private_state_384::kPrivateLayoutProfileId &&
         key.private_layout_profile_id ==
             metadata.private_layout_profile_id &&
         key.bvh_format_profile_id ==
             metadata.bvh_format_profile_id &&
         key.reservation_generation == metadata.slot_generation &&
         is_private_state_384_profile(key.storage_profile) &&
         key.storage_profile == metadata.private_storage_profile &&
         key.consumer ==
             consumer_for_target(
                 static_cast<target_kind>(metadata.target_kind)) &&
         key.operation ==
             private_state_384::operand_plan::kOperationDefault &&
         key.completion_reason ==
             private_state_384::operand_plan::kCompletionReasonNone &&
         identity.owner_hw_sid == metadata.owner.owner_hw_sid &&
         identity.resident_warp_id ==
             metadata.owner.resident_warp_id &&
         identity.request_identity ==
             metadata.owner.request_identity &&
         identity.request_generation == metadata.owner.generation &&
         identity.private_slot_id == metadata.owner.private_slot_id &&
         identity.operation_sequence ==
             metadata.target_operation_seq &&
         identity.lane_id == metadata.owner.lane_id &&
         bytes_are_zero(identity.reserved_zero,
                        sizeof(identity.reserved_zero));
}

bool collector_identity_matches_key(
    const private_state_384::operand_materializer::
        response_collector_v1 &collector,
    const private_state_384::live_bridge::live_operation_key_v1 &key) {
  const private_state_384::operand_materializer::operation_identity_v1
      &identity = collector.identity;
  return identity.owner_hw_sid == key.identity.owner_hw_sid &&
         identity.resident_warp_id ==
             key.identity.resident_warp_id &&
         identity.request_identity ==
             key.identity.request_identity &&
         identity.request_generation ==
             key.identity.request_generation &&
         identity.private_slot_id ==
             key.identity.private_slot_id &&
         identity.operation_sequence ==
             key.identity.operation_sequence &&
         identity.lane_id == key.identity.lane_id &&
         bytes_are_zero(identity.reserved_zero,
                        sizeof(identity.reserved_zero)) &&
         collector.private_layout_profile_id ==
             key.private_layout_profile_id &&
         collector.consumer == key.consumer &&
         collector.operation == key.operation &&
         collector.completion_reason == key.completion_reason;
}

template <typename Slot>
status_kind configure_private_state_384_slot_in_queue(
    Slot *slots, uint8_t capacity,
    const reservation_receipt_v0 &reservation,
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    const private_state_384::operand_materializer::
        response_collector_v1 &collector,
    reservation_receipt_v0 *updated_reservation) {
  if (reservation.slot_index >= capacity) {
    return kStatusUnknownReservation;
  }
  Slot &slot = slots[reservation.slot_index];
  if (!receipt_matches(slot.metadata, reservation)) {
    return slot.metadata.state == kSlotFree
               ? kStatusUnknownReservation
               : kStatusStaleReservation;
  }
  slot_metadata_v0 &metadata = slot.metadata;
  if (metadata.operation_kind != kOperationFetchTarget ||
      metadata.private_storage_profile !=
          private_storage::kProfileLegacyShared832 ||
      metadata.received_private_chunk_mask != 0 ||
      metadata.private_state_384_projection_valid != 0 ||
      collector.initialized != 1 || collector.failed != 0 ||
      collector.required_count == 0 ||
      collector.required_count >
          private_state_384::operand_materializer::
              kMaxOperationReadChunks ||
      collector.received_count != 0 ||
      collector.received_chunk_mask != 0 ||
      !collector_identity_matches_key(collector, key)) {
    return kStatusPrivate384PlanMismatch;
  }
  metadata.private_layout_profile_id =
      key.private_layout_profile_id;
  metadata.bvh_format_profile_id =
      key.bvh_format_profile_id;
  metadata.private_storage_profile =
      key.storage_profile;
  if (!operation_key_matches_metadata(key, metadata)) {
    return kStatusPrivate384PlanMismatch;
  }
  metadata.private_state_384_key = key;
  metadata.private_state_384_collector = collector;
  metadata.expected_private_chunk_count = collector.required_count;
  metadata.pending_private_response_count = collector.required_count;
  metadata.received_private_chunk_mask = 0;
  std::memset(metadata.mutable_ray_bytes, 0,
              sizeof(metadata.mutable_ray_bytes));
  std::memset(metadata.decode_context_bytes, 0,
              sizeof(metadata.decode_context_bytes));
  std::memset(metadata.committed_hit_bytes, 0,
              sizeof(metadata.committed_hit_bytes));
  std::memset(metadata.current_instance_bytes, 0,
              sizeof(metadata.current_instance_bytes));
  build_receipt(metadata, reservation.slot_index,
                updated_reservation);
  return kStatusOk;
}

template <typename Slot>
status_kind publish_private_state_384_projection_in_queue(
    Slot *slots, uint8_t capacity, ready_fifo_v0 *fifo,
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    const private_state_384::operand_materializer::
        response_collector_v1 &collector,
    const typed_node::ray_policy_v0 &ray_policy,
    const private_frontier::root_private_operands_v0 &root_projection,
    const private_frontier::instance_shader_projection_v0
        &current_instance_projection) {
  int match = -1;
  for (uint8_t index = 0; index < capacity; ++index) {
    const slot_metadata_v0 &metadata = slots[index].metadata;
    if (metadata.state == kSlotFree ||
        !is_private_state_384_profile(
            metadata.private_storage_profile) ||
        !operation_key_matches_metadata(key, metadata)) {
      continue;
    }
    if (match >= 0) return kStatusPrivate384ResponseRejected;
    match = index;
  }
  if (match < 0) return kStatusPrivate384ResponseRejected;
  Slot &slot = slots[match];
  slot_metadata_v0 &metadata = slot.metadata;
  if (!operation_key_matches_metadata(
          metadata.private_state_384_key, metadata) ||
      metadata.private_state_384_projection_valid != 0 ||
      collector.initialized != 1 || collector.failed != 0 ||
      collector.required_count !=
          metadata.expected_private_chunk_count ||
      collector.received_count != collector.required_count ||
      collector.received_chunk_mask != collector.required_chunk_mask ||
      !collector_identity_matches_key(collector, key) ||
      !bytes_are_zero(ray_policy.reserved_zero,
                      sizeof(ray_policy.reserved_zero))) {
    return kStatusPrivate384MaterializeRejected;
  }
  metadata.private_state_384_collector = collector;
  metadata.pending_private_response_count = 0;
  metadata.received_private_chunk_mask =
      collector.received_chunk_mask;
  metadata.ray_policy = ray_policy;
  metadata.private_state_384_root_projection =
      root_projection;
  metadata.private_state_384_current_instance_projection =
      current_instance_projection;
  metadata.private_state_384_projection_valid = 1;
  const target_kind target =
      static_cast<target_kind>(metadata.target_kind);
  metadata.valid_operand_mask |= static_cast<uint8_t>(
      kOperandMutableRayValid | kOperandRayPolicyValid |
      kOperandDecodeContextValid | kOperandCommittedHitValid |
      (target == kTargetPrimitive
           ? kOperandCurrentInstanceValid
           : 0));
  return maybe_make_ready(&metadata, fifo, capacity,
                          static_cast<uint8_t>(match));
}

struct expected_recovery_descriptor_chunk_v0 {
  uint16_t slot_chunk_offset;
  uint32_t byte_mask;
};

const expected_recovery_descriptor_chunk_v0
    kExpectedRecoveryDescriptorChunks[kStackSpillRecoveryChunks] = {
        {0x280, 0xffffff00u},
        {0x2a0, 0xffffffffu},
        {0x2c0, 0xffffffffu},
        {0x2e0, 0xffffffffu},
        {0x300, 0x000000ffu},
};

template <typename Slot>
status_kind fill_recovery_descriptor_chunk_in_queue(
    Slot *slots, uint8_t capacity, ready_fifo_v0 *fifo,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes],
    reservation_receipt_v0 *updated_reservation,
    typed_node::selected_child_fetch_work_item_v0 *selected_fetch) {
  if (reservation.slot_index >= capacity) return kStatusUnknownReservation;
  Slot &slot = slots[reservation.slot_index];
  if (!receipt_matches(slot.metadata, reservation)) {
    return slot.metadata.state == kSlotFree ? kStatusUnknownReservation
                                            : kStatusStaleReservation;
  }
  if (slot.metadata.recovery_descriptor_pending != 1 ||
      chunk_count != kStackSpillRecoveryChunks ||
      chunk_id >= chunk_count ||
      slot_chunk_offset !=
          kExpectedRecoveryDescriptorChunks[chunk_id].slot_chunk_offset ||
      byte_mask !=
          kExpectedRecoveryDescriptorChunks[chunk_id].byte_mask) {
    return kStatusRecoveryDescriptorShapeMismatch;
  }
  const uint8_t chunk_bit =
      static_cast<uint8_t>(uint8_t{1} << chunk_id);
  if ((slot.metadata.received_recovery_descriptor_chunk_mask &
       chunk_bit) != 0) {
    return kStatusDuplicateRecoveryDescriptorChunk;
  }

  for (unsigned byte = 0;
       byte < private_frontier::kSharedAccessChunkBytes; ++byte) {
    if ((byte_mask & (uint32_t{1} << byte)) == 0) {
      if (payload[byte] != 0) {
        return kStatusRecoveryDescriptorShapeMismatch;
      }
      continue;
    }
    const uint32_t private_slot_offset = slot_chunk_offset + byte;
    if (private_slot_offset < private_frontier::kTransitionSpillOffset ||
        private_slot_offset >=
            private_frontier::kTransitionSpillOffset +
                private_frontier::kStackTransitionSpillBytes) {
      return kStatusRecoveryDescriptorShapeMismatch;
    }
    slot.metadata.recovery_descriptor_bytes[
        private_slot_offset -
        private_frontier::kTransitionSpillOffset] = payload[byte];
  }
  slot.metadata.received_recovery_descriptor_chunk_mask |= chunk_bit;
  if (slot.metadata.pending_recovery_descriptor_response_count == 0) {
    return kStatusReadyFifoInvariant;
  }
  --slot.metadata.pending_recovery_descriptor_response_count;
  const uint8_t complete_mask = static_cast<uint8_t>(
      (uint8_t{1} << kStackSpillRecoveryChunks) - 1);
  if (slot.metadata.received_recovery_descriptor_chunk_mask !=
      complete_mask) {
    return kStatusOk;
  }
  if (slot.metadata.pending_recovery_descriptor_response_count != 0 ||
      !bytes_are_zero(
          slot.metadata.recovery_descriptor_bytes +
              private_frontier::kStackSelectedFetchBytes,
          private_frontier::kStackTransitionSpillBytes -
              private_frontier::kStackSelectedFetchBytes)) {
    return kStatusRecoveryDescriptorShapeMismatch;
  }

  private_frontier::shadow_slot_v0 spill_slot = {};
  spill_slot.owner = slot.metadata.owner;
  std::memcpy(
      spill_slot.bytes + private_frontier::kTransitionSpillOffset,
      slot.metadata.recovery_descriptor_bytes,
      private_frontier::kStackTransitionSpillBytes);
  typed_node::selected_child_fetch_work_item_v0 decoded = {};
  target_kind decoded_target = kTargetInvalid;
  uint16_t decoded_raw_bytes = 0;
  if (private_frontier::decode_stack_selected_fetch_spill(
          spill_slot, slot.metadata.owner, &decoded) !=
          private_frontier::kStatusOk ||
      classify_selected_fetch(
          decoded, &decoded_target, &decoded_raw_bytes) != kStatusOk ||
      decoded_target !=
          static_cast<target_kind>(slot.metadata.target_kind) ||
      decoded_raw_bytes != slot.metadata.raw_payload_bytes) {
    return kStatusRecoveryDescriptorKindMismatch;
  }

  slot.metadata.target_reference.payload_offset =
      decoded.child.payload_offset;
  slot.metadata.target_reference.near_t_bits =
      decoded.child.near_t_bits;
  slot.metadata.target_reference.payload_byte_count =
      decoded_raw_bytes;
  slot.metadata.target_reference.payload_kind =
      decoded.child.payload_kind;
  slot.metadata.target_reference.level =
      decoded.decode_context.as_object.as_type == 1
          ? typed_node::kLevelTlas
          : typed_node::kLevelBlas;
  slot.metadata.target_reference.source_kind =
      kTargetReferenceSelectedFetchCompatibilityAdapter;
  slot.metadata.target_reference.proxy_delegated = 1;
  slot.metadata.raw_payload_base_address =
      decoded.decode_context.device_base +
      decoded.child.payload_offset;
  slot.metadata.valid_operand_mask |=
      kOperandTargetReferenceValid;
  slot.metadata.recovery_descriptor_pending = 0;
  build_receipt(slot.metadata, reservation.slot_index,
                updated_reservation);
  *selected_fetch = decoded;
  return maybe_make_ready(&slot.metadata, fifo, capacity,
                          reservation.slot_index);
}

template <typename Slot>
status_kind fill_recovery_ray_policy_in_queue(
    Slot *slots, uint8_t capacity, ready_fifo_v0 *fifo,
    const reservation_receipt_v0 &reservation,
    const typed_node::ray_policy_v0 &ray_policy) {
  if (reservation.slot_index >= capacity) return kStatusUnknownReservation;
  Slot &slot = slots[reservation.slot_index];
  if (!receipt_matches(slot.metadata, reservation)) {
    return slot.metadata.state == kSlotFree ? kStatusUnknownReservation
                                            : kStatusStaleReservation;
  }
  if (!bytes_are_zero(ray_policy.reserved_zero,
                      sizeof(ray_policy.reserved_zero))) {
    return kStatusRecoveryDescriptorShapeMismatch;
  }
  if ((slot.metadata.valid_operand_mask &
       kOperandRayPolicyValid) != 0) {
    return kStatusDuplicateRecoveryRayPolicy;
  }
  slot.metadata.ray_policy = ray_policy;
  slot.metadata.valid_operand_mask |= kOperandRayPolicyValid;
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
status_kind build_operation_packet(const Slot &slot, target_kind target,
                                   operation_packet_v0 *packet) {
  if (packet == NULL) return kStatusInvalidArgument;
  if (slot.metadata.operation_kind ==
      kOperationInstanceRestoreParent) {
    if (target != kTargetInstance ||
        slot.metadata.required_operand_mask !=
            kOperandParentFrameValid ||
        slot.metadata.raw_payload_bytes != 0) {
      return kStatusPrivateOperandShapeMismatch;
    }
    private_frontier::shadow_slot_v0 private_slot = {};
    private_slot.owner = slot.metadata.owner;
    std::memcpy(
        private_slot.bytes + private_frontier::kParentFrameOffset,
        slot.metadata.parent_frame_bytes,
        sizeof(slot.metadata.parent_frame_bytes));
    private_frontier::traversal_frame_projection_v0 parent_frame = {};
    if (private_frontier::decode_parent_frame(
            private_slot, slot.metadata.owner, &parent_frame) !=
        private_frontier::kStatusOk) {
      return kStatusPrivateOperandShapeMismatch;
    }
    *packet = operation_packet_v0();
    packet->owner = slot.metadata.owner;
    packet->reservation_id = slot.metadata.reservation_id;
    packet->reservation_age = slot.metadata.reservation_age;
    packet->target_operation_seq =
        slot.metadata.target_operation_seq;
    packet->slot_generation = slot.metadata.slot_generation;
    packet->target_kind = kTargetInstance;
    packet->operation_kind = kOperationInstanceRestoreParent;
    packet->parent_frame = parent_frame;
    packet->valid = 1;
    return kStatusOk;
  }
  reservation_input_v0 input = {};
  input.owner = slot.metadata.owner;
  input.target_reference = slot.metadata.target_reference;
  input.forwarded_ray_policy = slot.metadata.ray_policy;
  input.raw_payload_base_address = slot.metadata.raw_payload_base_address;
  input.target_operation_seq = slot.metadata.target_operation_seq;
  input.producer_operation_seq = slot.metadata.producer_operation_seq;
  input.producer_commit_epoch = slot.metadata.producer_commit_epoch;
  input.raw_payload_bytes = slot.metadata.raw_payload_bytes;
  input.target_kind = target;
  input.producer_commit_required =
      slot.metadata.producer_commit_required;
  input.required_operand_mask = slot.metadata.required_operand_mask;
  input.pending_parent_resume =
      slot.metadata.pending_parent_resume;
  input.pending_parent_resume_valid =
      slot.metadata.pending_parent_resume_valid;
  const uint8_t root_private_mask = static_cast<uint8_t>(
      kOperandMutableRayValid | kOperandDecodeContextValid |
      kOperandCommittedHitValid);
  private_frontier::root_private_operands_v0 private_operands =
      slot.metadata.private_state_384_root_projection;
  if (is_private_state_384_profile(
          slot.metadata.private_storage_profile)) {
    if (slot.metadata.private_state_384_projection_valid != 1) {
      return kStatusPrivate384MaterializeRejected;
    }
  } else if ((slot.metadata.required_operand_mask & root_private_mask) ==
             root_private_mask) {
    private_frontier::shadow_slot_v0 private_slot = {};
    private_slot.owner = slot.metadata.owner;
    std::memcpy(
        private_slot.bytes + private_frontier::kMutableRayStateOffset,
        slot.metadata.mutable_ray_bytes,
        sizeof(slot.metadata.mutable_ray_bytes));
    std::memcpy(
        private_slot.bytes + private_frontier::kAsDecodeContextOffset,
        slot.metadata.decode_context_bytes,
        sizeof(slot.metadata.decode_context_bytes));
    std::memcpy(
        private_slot.bytes + private_frontier::kCommittedHitOffset,
        slot.metadata.committed_hit_bytes,
        sizeof(slot.metadata.committed_hit_bytes));
    if (private_frontier::decode_root_private_operands(
            private_slot, slot.metadata.owner,
            &private_operands) != private_frontier::kStatusOk) {
      return kStatusPrivateOperandShapeMismatch;
    }
  }
  private_frontier::instance_shader_projection_v0 current_instance =
      slot.metadata.private_state_384_current_instance_projection;
  if (is_private_state_384_profile(
          slot.metadata.private_storage_profile)) {
    if (slot.metadata.private_state_384_projection_valid != 1) {
      return kStatusPrivate384MaterializeRejected;
    }
  } else if ((slot.metadata.required_operand_mask &
              kOperandCurrentInstanceValid) != 0) {
    private_frontier::shadow_slot_v0 private_slot = {};
    private_slot.owner = slot.metadata.owner;
    std::memcpy(
        private_slot.bytes + private_frontier::kCurrentInstanceOffset,
        slot.metadata.current_instance_bytes,
        sizeof(slot.metadata.current_instance_bytes));
    if (private_frontier::decode_current_instance(
            private_slot, slot.metadata.owner,
            &current_instance) != private_frontier::kStatusOk) {
      return kStatusPrivateOperandShapeMismatch;
    }
  }
  const status_kind status = build_ready_operation_packet(
      input, slot.metadata.reservation_id, slot.metadata.reservation_age,
      slot.metadata.slot_generation, private_operands, current_instance,
      slot.raw_payload, packet);
  if (status != kStatusOk) return status;
  packet->private_layout_profile_id =
      slot.metadata.private_layout_profile_id;
  packet->bvh_format_profile_id =
      slot.metadata.bvh_format_profile_id;
  packet->private_storage_profile =
      slot.metadata.private_storage_profile;
  return kStatusOk;
}

template <typename Slot>
status_kind validate_ready_position(const Slot *slots, uint8_t capacity,
                                    const ready_fifo_v0 &fifo,
                                    target_kind target, uint8_t position) {
  if (position >= fifo.count) return kStatusNoReadyOperation;
  const uint8_t slot_index = fifo.indices[position];
  if (slot_index >= capacity) return kStatusReadyFifoInvariant;
  const Slot &slot = slots[slot_index];
  if (slot.metadata.state != kSlotReady ||
      slot.metadata.ready_enqueued != 1 ||
      slot.metadata.target_kind != target ||
      slot.metadata.valid_operand_mask !=
          slot.metadata.required_operand_mask ||
      slot.metadata.producer_commit_complete != 1 ||
      slot.metadata.pending_raw_response_count != 0 ||
      slot.metadata.pending_private_response_count != 0 ||
      slot.metadata.raw_payload_bytes > sizeof(slot.raw_payload)) {
    return kStatusReadyFifoInvariant;
  }
  return kStatusOk;
}

template <typename Slot>
status_kind find_ready_position_kind(
    const Slot *slots, uint8_t capacity, const ready_fifo_v0 &fifo,
    target_kind target, operation_kind required_operation_kind,
    uint8_t *position) {
  if (position == NULL) return kStatusInvalidArgument;
  *position = 0;
  if (fifo.count == 0) return kStatusNoReadyOperation;
  for (uint8_t index = 0; index < fifo.count; ++index) {
    const status_kind status =
        validate_ready_position(slots, capacity, fifo, target, index);
    if (status != kStatusOk) return status;
    if (slots[fifo.indices[index]].metadata.operation_kind ==
        required_operation_kind) {
      *position = index;
      return kStatusOk;
    }
  }
  return kStatusNoReadyOperation;
}

template <typename Slot>
status_kind pop_from_queue_position(
    Slot *slots, uint8_t capacity, ready_fifo_v0 *fifo,
    target_kind target, uint8_t position, bool unit_input_accepts,
    operation_packet_v0 *packet) {
  const status_kind ready_status =
      validate_ready_position(slots, capacity, *fifo, target, position);
  if (ready_status != kStatusOk) return ready_status;
  if (!unit_input_accepts) return kStatusUnitInputBackpressure;
  const uint8_t slot_index = fifo->indices[position];
  Slot &slot = slots[slot_index];

  const status_kind packet_status =
      build_operation_packet(slot, target, packet);
  if (packet_status != kStatusOk) return packet_status;

  slot.metadata.state = kSlotIssued;
  for (unsigned index = position + 1; index < fifo->count; ++index) {
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
status_kind pop_from_queue(Slot *slots, uint8_t capacity,
                           ready_fifo_v0 *fifo, target_kind target,
                           bool unit_input_accepts,
                           operation_packet_v0 *packet) {
  return pop_from_queue_position(slots, capacity, fifo, target, 0,
                                 unit_input_accepts, packet);
}

template <typename Slot>
status_kind peek_from_queue_position(
    const Slot *slots, uint8_t capacity, const ready_fifo_v0 &fifo,
    target_kind target, uint8_t position, operation_packet_v0 *packet) {
  const status_kind ready_status =
      validate_ready_position(slots, capacity, fifo, target, position);
  if (ready_status != kStatusOk) return ready_status;
  const Slot &slot = slots[fifo.indices[position]];
  return build_operation_packet(slot, target, packet);
}

template <typename Slot>
status_kind peek_from_queue(const Slot *slots, uint8_t capacity,
                            const ready_fifo_v0 &fifo,
                            target_kind target,
                            operation_packet_v0 *packet) {
  return peek_from_queue_position(slots, capacity, fifo, target, 0,
                                  packet);
}

template <typename Slot>
status_kind peek_reserved_slot(
    const Slot *slots, uint8_t capacity, const ready_fifo_v0 &fifo,
    const reservation_receipt_v0 &reservation, target_kind target,
    operation_packet_v0 *packet) {
  if (reservation.slot_index >= capacity) return kStatusUnknownReservation;
  const Slot &slot = slots[reservation.slot_index];
  if (!receipt_matches(slot.metadata, reservation)) {
    return slot.metadata.state == kSlotFree ? kStatusUnknownReservation
                                            : kStatusStaleReservation;
  }
  if (slot.metadata.state != kSlotReady) return kStatusNoReadyOperation;
  if (slot.metadata.ready_enqueued != 1 ||
      slot.metadata.target_kind != target ||
      slot.metadata.valid_operand_mask !=
          slot.metadata.required_operand_mask ||
      slot.metadata.producer_commit_complete != 1 ||
      slot.metadata.pending_raw_response_count != 0 ||
      slot.metadata.pending_private_response_count != 0 ||
      slot.metadata.raw_payload_bytes > sizeof(slot.raw_payload)) {
    return kStatusReadyFifoInvariant;
  }
  unsigned fifo_match_count = 0;
  for (unsigned index = 0; index < fifo.count; ++index) {
    fifo_match_count +=
        fifo.indices[index] == reservation.slot_index &&
        fifo.reservation_ages[index] == reservation.reservation_age;
  }
  if (fifo_match_count != 1) return kStatusReadyFifoInvariant;
  return build_operation_packet(slot, target, packet);
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

bool encode_replay_cursor(const typed_node::replay_cursor_v0 &cursor,
                          uint8_t *control) {
  if (control == NULL || cursor.anchor_valid > 1 ||
      cursor.inclusive > 1 ||
      !bytes_are_zero(cursor.reserved_zero,
                      sizeof(cursor.reserved_zero)) ||
      (cursor.anchor_valid == 0 &&
       (cursor.child_anchor != 0 || cursor.inclusive != 0)) ||
      (cursor.anchor_valid != 0 &&
       cursor.child_anchor >= typed_node::kMaxChildren)) {
    return false;
  }
  *control =
      cursor.anchor_valid == 0
          ? 0
          : static_cast<uint8_t>(
                cursor.child_anchor | uint8_t{1u << 3} |
                (cursor.inclusive != 0 ? uint8_t{1u << 4}
                                       : uint8_t{0}));
  return true;
}

bool decode_replay_cursor(uint8_t control,
                          typed_node::replay_cursor_v0 *cursor) {
  if (cursor == NULL || (control & uint8_t{0xe0}) != 0) {
    return false;
  }
  *cursor = typed_node::replay_cursor_v0();
  if (control == 0) return true;
  cursor->child_anchor =
      static_cast<uint8_t>(control & uint8_t{0x07});
  cursor->anchor_valid =
      static_cast<uint8_t>((control >> 3) & uint8_t{0x01});
  cursor->inclusive =
      static_cast<uint8_t>((control >> 4) & uint8_t{0x01});
  return cursor->anchor_valid != 0 &&
         cursor->child_anchor < typed_node::kMaxChildren;
}

bool validate_target_reference_shape(
    const target_reference_v0 &reference, target_kind target,
    uint16_t payload_bytes, uint64_t raw_payload_base_address) {
  return target_reference_shape_valid(
      reference, target, payload_bytes, raw_payload_base_address);
}

status_kind build_ready_operation_packet(
    const reservation_input_v0 &input, uint64_t reservation_id,
    uint64_t reservation_age, uint32_t slot_generation,
    const private_frontier::root_private_operands_v0 &private_operands,
    const private_frontier::instance_shader_projection_v0 &current_instance,
    const uint8_t *raw_payload, operation_packet_v0 *packet) {
  if (packet == NULL || raw_payload == NULL ||
      !reservation_identity_valid(input) || reservation_id == 0 ||
      reservation_age == 0 || slot_generation == 0 ||
      input.raw_payload_bytes > kMaxRawPayloadBytes) {
    return kStatusInvalidArgument;
  }

  *packet = operation_packet_v0();
  packet->owner = input.owner;
  packet->reservation_id = reservation_id;
  packet->reservation_age = reservation_age;
  packet->raw_payload_base_address = input.raw_payload_base_address;
  packet->target_operation_seq = input.target_operation_seq;
  packet->producer_operation_seq = input.producer_operation_seq;
  packet->producer_commit_epoch = input.producer_commit_epoch;
  packet->slot_generation = slot_generation;
  packet->private_layout_profile_id =
      private_frontier::kLayoutProfileId;
  packet->bvh_format_profile_id =
      typed_node::kGenRtDerivedProfileId;
  packet->raw_payload_bytes = input.raw_payload_bytes;
  packet->target_kind = input.target_kind;
  packet->operation_kind = kOperationFetchTarget;
  packet->private_storage_profile =
      private_storage::kProfileLegacyShared832;
  packet->valid = 1;
  packet->target_reference = input.target_reference;
  packet->pending_parent_resume =
      input.pending_parent_resume;
  packet->pending_parent_resume_valid =
      input.pending_parent_resume_valid;
  packet->ray_policy = input.forwarded_ray_policy;
  packet->private_operands = private_operands;
  packet->current_instance = current_instance;
  std::memcpy(packet->raw_payload, raw_payload, input.raw_payload_bytes);
  return kStatusOk;
}

status_kind build_ready_node_operation_packet(
    const reservation_input_v0 &input, uint64_t reservation_id,
    uint64_t reservation_age, uint32_t slot_generation,
    const private_frontier::root_private_operands_v0 &private_operands,
    const uint8_t *raw_payload, operation_packet_v0 *packet) {
  const uint8_t complete_node_mask = static_cast<uint8_t>(
      kOperandTargetReferenceValid | kOperandRawPayloadValid |
      kOperandMutableRayValid | kOperandRayPolicyValid |
      kOperandDecodeContextValid | kOperandCommittedHitValid);
  if (input.target_kind != kTargetNode ||
      input.required_operand_mask != complete_node_mask) {
    return kStatusInvalidArgument;
  }
  return build_ready_operation_packet(
      input, reservation_id, reservation_age, slot_generation,
      private_operands,
      private_frontier::instance_shader_projection_v0(),
      raw_payload, packet);
}

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

static status_kind try_reserve_internal(
    engine_state_v0 *state, const reservation_input_v0 &input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt,
    bool allow_instance_blas_root_source) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      !reservation_identity_valid(input) ||
      (input.target_reference.source_kind ==
           kTargetReferenceInstanceBlasRootProducer &&
       (!allow_instance_blas_root_source ||
        input.producer_commit_required != 1))) {
    return kStatusInvalidArgument;
  }
  *receipt = reservation_receipt_v0();
  const target_kind target =
      static_cast<target_kind>(input.target_kind);
  const uint16_t raw_payload_bytes = input.raw_payload_bytes;

  switch (target) {
    case kTargetNode:
      return reserve_in_queue(
          state->node_slots, state->config.node_capacity,
          &state->node_window, state->config.node_reservation_width, state,
          input, target, raw_payload_bytes, reservation_cycle, receipt);
    case kTargetPrimitive:
      return reserve_in_queue(
          state->primitive_slots, state->config.primitive_capacity,
          &state->primitive_window,
          state->config.primitive_reservation_width, state, input, target,
          raw_payload_bytes, reservation_cycle, receipt);
    case kTargetInstance:
      return reserve_in_queue(
          state->instance_slots, state->config.instance_capacity,
          &state->instance_window, state->config.instance_reservation_width,
          state, input, target, raw_payload_bytes, reservation_cycle,
          receipt);
    case kTargetInvalid:
      break;
  }
  return kStatusInvalidSelectedFetch;
}

status_kind try_reserve(
    engine_state_v0 *state, const reservation_input_v0 &input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  return try_reserve_internal(state, input, reservation_cycle, receipt,
                              false);
}

status_kind try_reserve_selected_fetch(
    engine_state_v0 *state,
    const selected_fetch_reservation_input_v0 &selected_input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  reservation_input_v0 input = {};
  const status_kind lower_status =
      lower_selected_fetch(selected_input, &input);
  if (lower_status != kStatusOk) return lower_status;
  return try_reserve(state, input, reservation_cycle, receipt);
}

status_kind lower_selected_fetch(
    const selected_fetch_reservation_input_v0 &selected_input,
    reservation_input_v0 *input) {
  const bool producer_tag_valid =
      selected_input.producer_commit_required == 0
          ? selected_input.producer_operation_seq == 0 &&
                selected_input.producer_commit_epoch == 0
          : selected_input.producer_operation_seq != 0 &&
                selected_input.producer_commit_epoch != 0 &&
                selected_input.producer_operation_seq !=
                    selected_input.target_operation_seq;
  if (input == NULL ||
      selected_input.target_operation_seq == 0 ||
      selected_input.producer_commit_required > 1 ||
      !producer_tag_valid ||
      !bytes_are_zero(selected_input.reserved_zero,
                      sizeof(selected_input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *input = reservation_input_v0();

  target_kind target = kTargetInvalid;
  uint16_t raw_payload_bytes = 0;
  const status_kind classify_status =
      classify_selected_fetch(selected_input.selected_fetch, &target,
                              &raw_payload_bytes);
  if (classify_status != kStatusOk) return classify_status;

  input->owner = selected_input.owner;
  input->target_reference.payload_offset =
      selected_input.selected_fetch.child.payload_offset;
  input->target_reference.near_t_bits =
      selected_input.selected_fetch.child.near_t_bits;
  input->target_reference.build_generation =
      selected_input.build_generation;
  input->target_reference.replay_control =
      selected_input.replay_control;
  input->pending_parent_resume =
      selected_input.pending_parent_resume;
  input->pending_parent_resume_valid =
      selected_input.pending_parent_resume_valid;
  input->target_reference.payload_byte_count = raw_payload_bytes;
  input->target_reference.payload_kind =
      selected_input.selected_fetch.child.payload_kind;
  input->target_reference.level =
      selected_input.selected_fetch.decode_context.as_object.as_type == 1
          ? typed_node::kLevelTlas
          : typed_node::kLevelBlas;
  input->target_reference.source_kind =
      kTargetReferenceSelectedFetchCompatibilityAdapter;
  input->target_reference.proxy_delegated = 1;
  input->forwarded_ray_policy = selected_input.forwarded_ray_policy;
  if (selected_input.selected_fetch.decode_context.device_base >
      std::numeric_limits<uint64_t>::max() -
          selected_input.selected_fetch.child.payload_offset) {
    return kStatusInvalidSelectedFetch;
  }
  input->raw_payload_base_address =
      selected_input.selected_fetch.decode_context.device_base +
      selected_input.selected_fetch.child.payload_offset;
  input->target_operation_seq = selected_input.target_operation_seq;
  input->producer_operation_seq =
      selected_input.producer_operation_seq;
  input->producer_commit_epoch =
      selected_input.producer_commit_epoch;
  input->raw_payload_bytes = raw_payload_bytes;
  input->target_kind = target;
  input->producer_commit_required =
      selected_input.producer_commit_required;
  input->required_operand_mask = static_cast<uint8_t>(
      selected_input.required_operand_mask |
      (target == kTargetPrimitive
           ? kOperandCurrentInstanceValid
           : 0));
  input->forwarded_operand_mask =
      selected_input.forwarded_operand_mask;
  return kStatusOk;
}

status_kind try_reserve_instance_blas_root(
    engine_state_v0 *state,
    const instance_blas_root_reservation_input_v0 &root_input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  reservation_input_v0 input = {};
  const status_kind lower_status =
      lower_instance_blas_root(root_input, &input);
  if (lower_status != kStatusOk) return lower_status;
  return try_reserve_internal(state, input, reservation_cycle, receipt,
                              true);
}

status_kind lower_instance_blas_root(
    const instance_blas_root_reservation_input_v0 &root_input,
    reservation_input_v0 *input) {
  const typed_node::compact_child_work_item_v0 &root =
      root_input.root_fetch.child;
  const typed_blas::as_decode_context_v0 &context =
      root_input.root_fetch.decode_context;
  if (input == NULL ||
      root_input.target_operation_seq == 0 ||
      root_input.producer_operation_seq == 0 ||
      root_input.producer_commit_epoch == 0 ||
      root_input.target_operation_seq ==
          root_input.producer_operation_seq ||
      root.payload_kind != typed_node::kInternalPayloadKind ||
      root.payload_byte_count != kNodeRawPayloadBytes ||
      root.child_slot != 0 ||
      context.as_object.as_type != typed_blas::kAsTypeBlas ||
      !bytes_are_zero(root_input.reserved_zero,
                      sizeof(root_input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *input = reservation_input_v0();

  input->owner = root_input.owner;
  input->target_reference.payload_offset = root.payload_offset;
  input->target_reference.near_t_bits = root.near_t_bits;
  input->target_reference.build_generation =
      root_input.build_generation;
  input->target_reference.payload_byte_count = kNodeRawPayloadBytes;
  input->target_reference.payload_kind = root.payload_kind;
  input->target_reference.level = typed_node::kLevelBlas;
  input->target_reference.source_kind =
      kTargetReferenceInstanceBlasRootProducer;
  input->target_reference.proxy_delegated = 0;
  input->forwarded_ray_policy = root_input.forwarded_ray_policy;
  if (context.device_base >
      std::numeric_limits<uint64_t>::max() - root.payload_offset) {
    return kStatusInvalidSelectedFetch;
  }
  input->raw_payload_base_address =
      context.device_base + root.payload_offset;
  input->target_operation_seq = root_input.target_operation_seq;
  input->producer_operation_seq = root_input.producer_operation_seq;
  input->producer_commit_epoch = root_input.producer_commit_epoch;
  input->raw_payload_bytes = kNodeRawPayloadBytes;
  input->target_kind = kTargetNode;
  input->producer_commit_required = 1;
  input->required_operand_mask = static_cast<uint8_t>(
      kOperandTargetReferenceValid | kOperandRawPayloadValid |
      kOperandMutableRayValid | kOperandRayPolicyValid |
      kOperandDecodeContextValid | kOperandCommittedHitValid);
  input->forwarded_operand_mask = kOperandRayPolicyValid;
  return kStatusOk;
}

status_kind try_reserve_recovery(
    engine_state_v0 *state,
    const recovery_reservation_input_v0 &input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      !recovery_reservation_identity_valid(input)) {
    return kStatusInvalidArgument;
  }
  *receipt = reservation_receipt_v0();
  const target_kind target =
      static_cast<target_kind>(input.target_kind);
  switch (target) {
    case kTargetNode:
      return reserve_recovery_in_queue(
          state->node_slots, state->config.node_capacity,
          &state->node_window, state->config.node_reservation_width,
          state, input, target, reservation_cycle, receipt);
    case kTargetPrimitive:
      return reserve_recovery_in_queue(
          state->primitive_slots, state->config.primitive_capacity,
          &state->primitive_window,
          state->config.primitive_reservation_width, state, input,
          target, reservation_cycle, receipt);
    case kTargetInstance:
      return reserve_recovery_in_queue(
          state->instance_slots, state->config.instance_capacity,
          &state->instance_window,
          state->config.instance_reservation_width, state, input,
          target, reservation_cycle, receipt);
    case kTargetInvalid:
      break;
  }
  return kStatusInvalidSelectedFetch;
}

status_kind try_reserve_instance_restore_parent(
    engine_state_v0 *state,
    const instance_restore_reservation_input_v0 &input,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      !instance_restore_reservation_identity_valid(input)) {
    return kStatusInvalidArgument;
  }
  *receipt = reservation_receipt_v0();
  return reserve_instance_restore_in_queue(
      state->instance_slots, state->config.instance_capacity,
      &state->instance_window, state->config.instance_reservation_width,
      state, input, reservation_cycle, receipt);
}

status_kind try_reserve_prefill(
    engine_state_v0 *state,
    const stack_commit::forwarding_decision_input_v0 &decision,
    uint64_t reservation_cycle, reservation_receipt_v0 *receipt) {
  if (decision.producer_operation_seq == 0 ||
      decision.target_operation_seq == 0 ||
      decision.commit_epoch == 0 ||
      !valid_owner(decision.owner) ||
      !bytes_are_zero(decision.reserved_zero,
                      sizeof(decision.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  target_kind target = kTargetInvalid;
  uint16_t raw_payload_bytes = 0;
  const status_kind classify_status =
      classify_selected_fetch(decision.selected_fetch, &target,
                              &raw_payload_bytes);
  if (classify_status != kStatusOk) return classify_status;

  reservation_input_v0 input = {};
  input.owner = decision.owner;
  input.target_reference.payload_offset =
      decision.selected_fetch.child.payload_offset;
  input.target_reference.near_t_bits =
      decision.selected_fetch.child.near_t_bits;
  input.target_reference.payload_byte_count = raw_payload_bytes;
  input.target_reference.payload_kind =
      decision.selected_fetch.child.payload_kind;
  input.target_reference.level =
      decision.selected_fetch.decode_context.as_object.as_type == 1
          ? typed_node::kLevelTlas
          : typed_node::kLevelBlas;
  input.target_reference.source_kind =
      kTargetReferenceSelectedFetchCompatibilityAdapter;
  input.target_reference.proxy_delegated = 1;
  input.raw_payload_base_address =
      decision.selected_fetch.decode_context.device_base +
      decision.selected_fetch.child.payload_offset;
  input.forwarded_ray_policy = decision.forwarded_ray_policy;
  input.target_operation_seq = decision.target_operation_seq;
  input.producer_operation_seq = decision.producer_operation_seq;
  input.producer_commit_epoch = decision.commit_epoch;
  input.raw_payload_bytes = raw_payload_bytes;
  input.target_kind = target;
  input.producer_commit_required = 1;
  input.required_operand_mask = static_cast<uint8_t>(
      kOperandTargetReferenceValid | kOperandRawPayloadValid);
  return try_reserve(state, input, reservation_cycle, receipt);
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

status_kind fill_raw_payload_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, const uint8_t *raw_payload_chunk,
    uint8_t chunk_bytes) {
  if (state == NULL || raw_payload_chunk == NULL ||
      state->initialized != 1 || reservation.valid != 1) {
    return kStatusInvalidArgument;
  }
  engine_state_v0 staged = *state;
  status_kind status = kStatusInvalidArgument;
  switch (static_cast<target_kind>(reservation.target_kind)) {
    case kTargetNode:
      status = fill_chunk_in_queue(
          staged.node_slots, staged.config.node_capacity,
          &staged.node_ready, reservation, chunk_id, chunk_count,
          raw_payload_chunk, chunk_bytes);
      break;
    case kTargetPrimitive:
      status = fill_chunk_in_queue(
          staged.primitive_slots, staged.config.primitive_capacity,
          &staged.primitive_ready, reservation, chunk_id, chunk_count,
          raw_payload_chunk, chunk_bytes);
      break;
    case kTargetInstance:
      status = fill_chunk_in_queue(
          staged.instance_slots, staged.config.instance_capacity,
          &staged.instance_ready, reservation, chunk_id, chunk_count,
          raw_payload_chunk, chunk_bytes);
      break;
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  if (status == kStatusOk) *state = staged;
  return status;
}

status_kind fill_private_operand_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint8_t field_kind, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes]) {
  if (state == NULL || payload == NULL || state->initialized != 1 ||
      reservation.valid != 1) {
    return kStatusInvalidArgument;
  }
  engine_state_v0 staged = *state;
  status_kind status = kStatusInvalidArgument;
  switch (static_cast<target_kind>(reservation.target_kind)) {
    case kTargetNode:
      status = fill_private_chunk_in_queue(
          staged.node_slots, staged.config.node_capacity, &staged.node_ready,
          reservation, chunk_id, chunk_count, field_kind, slot_chunk_offset,
          byte_mask, payload);
      break;
    case kTargetPrimitive:
      status = fill_private_chunk_in_queue(
          staged.primitive_slots, staged.config.primitive_capacity,
          &staged.primitive_ready, reservation, chunk_id, chunk_count,
          field_kind, slot_chunk_offset, byte_mask, payload);
      break;
    case kTargetInstance:
      status = fill_private_chunk_in_queue(
          staged.instance_slots, staged.config.instance_capacity,
          &staged.instance_ready, reservation, chunk_id, chunk_count,
          field_kind, slot_chunk_offset, byte_mask, payload);
      break;
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  if (status == kStatusOk) *state = staged;
  return status;
}

status_kind configure_private_state_384_slot(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation,
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    const private_state_384::operand_materializer::
        response_collector_v1 &collector,
    reservation_receipt_v0 *updated_reservation) {
  if (state == NULL || updated_reservation == NULL ||
      state->initialized != 1 || reservation.valid != 1) {
    return kStatusInvalidArgument;
  }
  *updated_reservation = reservation_receipt_v0();
  engine_state_v0 staged = *state;
  status_kind status = kStatusInvalidArgument;
  switch (static_cast<target_kind>(reservation.target_kind)) {
    case kTargetNode:
      status = configure_private_state_384_slot_in_queue(
          staged.node_slots, staged.config.node_capacity,
          reservation, key, collector, updated_reservation);
      break;
    case kTargetPrimitive:
      status = configure_private_state_384_slot_in_queue(
          staged.primitive_slots, staged.config.primitive_capacity,
          reservation, key, collector, updated_reservation);
      break;
    case kTargetInstance:
      status = configure_private_state_384_slot_in_queue(
          staged.instance_slots, staged.config.instance_capacity,
          reservation, key, collector, updated_reservation);
      break;
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  if (status == kStatusOk) *state = staged;
  return status;
}

status_kind publish_private_state_384_projection(
    engine_state_v0 *state,
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    const private_state_384::operand_materializer::
        response_collector_v1 &collector,
    const typed_node::ray_policy_v0 &ray_policy,
    const private_frontier::root_private_operands_v0 &root_projection,
    const private_frontier::instance_shader_projection_v0
        &current_instance_projection) {
  if (state == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  engine_state_v0 staged = *state;
  status_kind status = kStatusPrivate384MaterializeRejected;
  switch (key.consumer) {
    case private_state_384::operand_plan::kConsumerNode:
      status = publish_private_state_384_projection_in_queue(
          staged.node_slots, staged.config.node_capacity,
          &staged.node_ready, key, collector, ray_policy,
          root_projection, current_instance_projection);
      break;
    case private_state_384::operand_plan::kConsumerPrimitive:
      status = publish_private_state_384_projection_in_queue(
          staged.primitive_slots, staged.config.primitive_capacity,
          &staged.primitive_ready, key, collector, ray_policy,
          root_projection, current_instance_projection);
      break;
    case private_state_384::operand_plan::kConsumerInstance:
      status = publish_private_state_384_projection_in_queue(
          staged.instance_slots, staged.config.instance_capacity,
          &staged.instance_ready, key, collector, ray_policy,
          root_projection, current_instance_projection);
      break;
    default:
      return kStatusPrivate384MaterializeRejected;
  }
  if (status == kStatusOk) *state = staged;
  return status;
}

status_kind fill_recovery_descriptor_chunk(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation, uint8_t chunk_id,
    uint8_t chunk_count, uint16_t slot_chunk_offset,
    uint32_t byte_mask,
    const uint8_t payload[private_frontier::kSharedAccessChunkBytes],
    reservation_receipt_v0 *updated_reservation,
    typed_node::selected_child_fetch_work_item_v0 *selected_fetch) {
  if (state == NULL || payload == NULL ||
      updated_reservation == NULL || selected_fetch == NULL ||
      state->initialized != 1 || reservation.valid != 1) {
    return kStatusInvalidArgument;
  }
  *updated_reservation = reservation_receipt_v0();
  *selected_fetch =
      typed_node::selected_child_fetch_work_item_v0();
  engine_state_v0 staged = *state;
  status_kind status = kStatusInvalidArgument;
  switch (static_cast<target_kind>(reservation.target_kind)) {
    case kTargetNode:
      status = fill_recovery_descriptor_chunk_in_queue(
          staged.node_slots, staged.config.node_capacity,
          &staged.node_ready, reservation, chunk_id, chunk_count,
          slot_chunk_offset, byte_mask, payload, updated_reservation,
          selected_fetch);
      break;
    case kTargetPrimitive:
      status = fill_recovery_descriptor_chunk_in_queue(
          staged.primitive_slots, staged.config.primitive_capacity,
          &staged.primitive_ready, reservation, chunk_id, chunk_count,
          slot_chunk_offset, byte_mask, payload, updated_reservation,
          selected_fetch);
      break;
    case kTargetInstance:
      status = fill_recovery_descriptor_chunk_in_queue(
          staged.instance_slots, staged.config.instance_capacity,
          &staged.instance_ready, reservation, chunk_id, chunk_count,
          slot_chunk_offset, byte_mask, payload, updated_reservation,
          selected_fetch);
      break;
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  if (status == kStatusOk) *state = staged;
  return status;
}

status_kind fill_recovery_ray_policy(
    engine_state_v0 *state,
    const reservation_receipt_v0 &reservation,
    const typed_node::ray_policy_v0 &ray_policy) {
  if (state == NULL || state->initialized != 1 ||
      reservation.valid != 1) {
    return kStatusInvalidArgument;
  }
  engine_state_v0 staged = *state;
  status_kind status = kStatusInvalidArgument;
  switch (static_cast<target_kind>(reservation.target_kind)) {
    case kTargetNode:
      status = fill_recovery_ray_policy_in_queue(
          staged.node_slots, staged.config.node_capacity,
          &staged.node_ready, reservation, ray_policy);
      break;
    case kTargetPrimitive:
      status = fill_recovery_ray_policy_in_queue(
          staged.primitive_slots, staged.config.primitive_capacity,
          &staged.primitive_ready, reservation, ray_policy);
      break;
    case kTargetInstance:
      status = fill_recovery_ray_policy_in_queue(
          staged.instance_slots, staged.config.instance_capacity,
          &staged.instance_ready, reservation, ray_policy);
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

status_kind peek_ready_operation(const engine_state_v0 &state,
                                 target_kind target,
                                 operation_packet_v0 *packet) {
  if (packet == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  switch (target) {
    case kTargetNode:
      return peek_from_queue(
          state.node_slots, state.config.node_capacity, state.node_ready,
          target, packet);
    case kTargetPrimitive:
      return peek_from_queue(
          state.primitive_slots, state.config.primitive_capacity,
          state.primitive_ready, target, packet);
    case kTargetInstance:
      return peek_from_queue(
          state.instance_slots, state.config.instance_capacity,
          state.instance_ready, target, packet);
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  return kStatusInvalidArgument;
}

status_kind pop_ready_operation_kind(
    engine_state_v0 *state, target_kind target,
    operation_kind required_operation_kind, bool unit_input_accepts,
    operation_packet_v0 *packet) {
  if (state == NULL || packet == NULL || state->initialized != 1 ||
      (required_operation_kind != kOperationFetchTarget &&
       required_operation_kind != kOperationInstanceRestoreParent)) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  uint8_t position = 0;
  status_kind status = kStatusInvalidArgument;
  switch (target) {
    case kTargetNode:
      status = find_ready_position_kind(
          state->node_slots, state->config.node_capacity,
          state->node_ready, target, required_operation_kind, &position);
      if (status != kStatusOk) return status;
      return pop_from_queue_position(
          state->node_slots, state->config.node_capacity,
          &state->node_ready, target, position, unit_input_accepts,
          packet);
    case kTargetPrimitive:
      status = find_ready_position_kind(
          state->primitive_slots, state->config.primitive_capacity,
          state->primitive_ready, target, required_operation_kind,
          &position);
      if (status != kStatusOk) return status;
      return pop_from_queue_position(
          state->primitive_slots, state->config.primitive_capacity,
          &state->primitive_ready, target, position, unit_input_accepts,
          packet);
    case kTargetInstance:
      status = find_ready_position_kind(
          state->instance_slots, state->config.instance_capacity,
          state->instance_ready, target, required_operation_kind,
          &position);
      if (status != kStatusOk) return status;
      return pop_from_queue_position(
          state->instance_slots, state->config.instance_capacity,
          &state->instance_ready, target, position, unit_input_accepts,
          packet);
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  return kStatusInvalidArgument;
}

status_kind peek_ready_operation_kind(
    const engine_state_v0 &state, target_kind target,
    operation_kind required_operation_kind, operation_packet_v0 *packet) {
  if (packet == NULL || state.initialized != 1 ||
      (required_operation_kind != kOperationFetchTarget &&
       required_operation_kind != kOperationInstanceRestoreParent)) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  uint8_t position = 0;
  status_kind status = kStatusInvalidArgument;
  switch (target) {
    case kTargetNode:
      status = find_ready_position_kind(
          state.node_slots, state.config.node_capacity, state.node_ready,
          target, required_operation_kind, &position);
      if (status != kStatusOk) return status;
      return peek_from_queue_position(
          state.node_slots, state.config.node_capacity, state.node_ready,
          target, position, packet);
    case kTargetPrimitive:
      status = find_ready_position_kind(
          state.primitive_slots, state.config.primitive_capacity,
          state.primitive_ready, target, required_operation_kind,
          &position);
      if (status != kStatusOk) return status;
      return peek_from_queue_position(
          state.primitive_slots, state.config.primitive_capacity,
          state.primitive_ready, target, position, packet);
    case kTargetInstance:
      status = find_ready_position_kind(
          state.instance_slots, state.config.instance_capacity,
          state.instance_ready, target, required_operation_kind,
          &position);
      if (status != kStatusOk) return status;
      return peek_from_queue_position(
          state.instance_slots, state.config.instance_capacity,
          state.instance_ready, target, position, packet);
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  return kStatusInvalidArgument;
}

status_kind peek_ready_reservation(
    const engine_state_v0 &state,
    const reservation_receipt_v0 &reservation,
    operation_packet_v0 *packet) {
  if (packet == NULL || state.initialized != 1 ||
      reservation.valid != 1 ||
      reservation.target_kind < kTargetNode ||
      reservation.target_kind > kTargetInstance) {
    return kStatusInvalidArgument;
  }
  *packet = operation_packet_v0();
  switch (static_cast<target_kind>(reservation.target_kind)) {
    case kTargetNode:
      return peek_reserved_slot(
          state.node_slots, state.config.node_capacity, state.node_ready,
          reservation, kTargetNode, packet);
    case kTargetPrimitive:
      return peek_reserved_slot(
          state.primitive_slots, state.config.primitive_capacity,
          state.primitive_ready, reservation, kTargetPrimitive, packet);
    case kTargetInstance:
      return peek_reserved_slot(
          state.instance_slots, state.config.instance_capacity,
          state.instance_ready, reservation, kTargetInstance, packet);
    case kTargetInvalid:
      return kStatusInvalidArgument;
  }
  return kStatusInvalidArgument;
}

template <typename Slot>
status_kind find_private_state_384_reservation_in_queue(
    const Slot *slots, uint8_t capacity,
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    reservation_receipt_v0 *reservation) {
  int match = -1;
  for (uint8_t index = 0; index < capacity; ++index) {
    const slot_metadata_v0 &metadata = slots[index].metadata;
    if (metadata.state == kSlotFree ||
        !is_private_state_384_profile(
            metadata.private_storage_profile) ||
        !operation_key_matches_metadata(key, metadata)) {
      continue;
    }
    if (match >= 0) return kStatusPrivate384PlanMismatch;
    match = index;
  }
  if (match < 0) return kStatusUnknownReservation;
  build_receipt(slots[match].metadata, static_cast<uint8_t>(match),
                reservation);
  return kStatusOk;
}

status_kind find_private_state_384_reservation(
    const engine_state_v0 &state,
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    reservation_receipt_v0 *reservation) {
  if (reservation == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *reservation = reservation_receipt_v0();
  switch (key.consumer) {
    case private_state_384::operand_plan::kConsumerNode:
      return find_private_state_384_reservation_in_queue(
          state.node_slots, state.config.node_capacity, key,
          reservation);
    case private_state_384::operand_plan::kConsumerPrimitive:
      return find_private_state_384_reservation_in_queue(
          state.primitive_slots, state.config.primitive_capacity, key,
          reservation);
    case private_state_384::operand_plan::kConsumerInstance:
      return find_private_state_384_reservation_in_queue(
          state.instance_slots, state.config.instance_capacity, key,
          reservation);
    default:
      return kStatusInvalidArgument;
  }
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
    case kStatusChunkShapeMismatch:
      return "chunk_shape_mismatch";
    case kStatusDuplicateRawChunk:
      return "duplicate_raw_chunk";
    case kStatusPrivateOperandShapeMismatch:
      return "private_operand_shape_mismatch";
    case kStatusDuplicatePrivateChunk:
      return "duplicate_private_chunk";
    case kStatusRecoveryDescriptorShapeMismatch:
      return "recovery_descriptor_shape_mismatch";
    case kStatusDuplicateRecoveryDescriptorChunk:
      return "duplicate_recovery_descriptor_chunk";
    case kStatusRecoveryDescriptorKindMismatch:
      return "recovery_descriptor_kind_mismatch";
    case kStatusDuplicateRecoveryRayPolicy:
      return "duplicate_recovery_ray_policy";
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
    case kStatusPrivate384PlanMismatch:
      return "private_384_plan_mismatch";
    case kStatusPrivate384ResponseRejected:
      return "private_384_response_rejected";
    case kStatusPrivate384MaterializeRejected:
      return "private_384_materialize_rejected";
  }
  return "unknown";
}

}  // namespace fetch_target
}  // namespace v04
}  // namespace rtcore
