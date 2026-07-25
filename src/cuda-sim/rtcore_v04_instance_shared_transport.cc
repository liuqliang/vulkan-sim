#include "rtcore_v04_instance_shared_transport.h"

#include <cstring>
#include <limits>

#include "rtcore_v04_private_shared_backing_internal.h"

namespace rtcore {
namespace v04 {
namespace instance_shared {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool valid_config(const config_v0 &config) {
  return config.result_commit_capacity != 0 &&
         config.result_commit_capacity <= kMaxResultCommitEntries &&
         config.tracker_capacity != 0 &&
         config.tracker_capacity <= kMaxCommitTrackers &&
         bytes_are_zero(config.reserved_zero,
                        sizeof(config.reserved_zero));
}

uint16_t expected_mask(uint16_t count) {
  return count == 0
             ? 0
             : static_cast<uint16_t>((uint32_t{1} << count) - 1);
}

int find_free_result(const engine_state_v0 &state) {
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    if (state.result_entries[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_free_tracker(const engine_state_v0 &state) {
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    if (state.trackers[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_tracker(const engine_state_v0 &state,
                 const private_frontier::owner_binding_v0 &owner,
                 uint32_t operation_seq, uint32_t commit_epoch) {
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    const commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid != 0 &&
        tracker.operation_seq == operation_seq &&
        tracker.commit_epoch == commit_epoch &&
        private_frontier::owners_equal(tracker.owner, owner)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

bool operation_live(const engine_state_v0 &state,
                    const private_frontier::owner_binding_v0 &owner,
                    uint32_t operation_seq) {
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    const commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid != 0 &&
        tracker.operation_seq == operation_seq &&
        private_frontier::owners_equal(tracker.owner, owner)) {
      return true;
    }
  }
  return false;
}

int find_oldest_result(const engine_state_v0 &state) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    const result_commit_entry_v0 &entry = state.result_entries[index];
    if (entry.valid != 0 &&
        entry.next_write_index < entry.write_count &&
        entry.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = entry.issue_age;
    }
  }
  return selected;
}

void build_offer(const result_commit_entry_v0 &entry,
                 write_offer_v0 *offer) {
  *offer = write_offer_v0();
  offer->owner = entry.owner;
  offer->operation_seq = entry.operation_seq;
  offer->commit_epoch = entry.commit_epoch;
  offer->memory_operation_seq =
      static_cast<uint16_t>(entry.next_write_index + 1);
  offer->write_count = entry.write_count;
  offer->fragment = entry.writes[entry.next_write_index];
}

int find_result(const engine_state_v0 &state,
                const write_offer_v0 &offer) {
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    const result_commit_entry_v0 &entry = state.result_entries[index];
    if (entry.valid != 0 &&
        entry.operation_seq == offer.operation_seq &&
        entry.commit_epoch == offer.commit_epoch &&
        private_frontier::owners_equal(entry.owner, offer.owner)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

status_kind prepare_shared_write(
    const write_offer_v0 &offer, uint64_t enqueue_cycle,
    private_shared::shared_write_v0 *operation) {
  if (operation == NULL || offer.operation_seq == 0 ||
      offer.commit_epoch == 0 || offer.memory_operation_seq == 0 ||
      offer.write_count != instance_semantic::kRestoreWriteFragmentCount ||
      offer.memory_operation_seq > offer.write_count ||
      !bytes_are_zero(offer.reserved_zero,
                      sizeof(offer.reserved_zero)) ||
      !instance_semantic::validate_private_write_fragment(
          offer.fragment)) {
    return kStatusWriteOfferMismatch;
  }
  *operation = private_shared::shared_write_v0();
  operation->valid = true;
  operation->address_space = private_shared::kAddressSpaceShared;
  operation->address_mode = private_shared::kAddressModePrivateField;
  operation->access_operation = private_shared::kAccessOperationWrite;
  operation->destination = private_shared::kDestinationPrivateCommitAck;
  operation->owner = offer.owner;
  operation->operation_seq = offer.operation_seq;
  operation->commit_epoch = offer.commit_epoch;
  operation->memory_op_seq = offer.memory_operation_seq;
  operation->chunk_id =
      static_cast<uint8_t>(offer.memory_operation_seq - 1);
  operation->chunk_count = static_cast<uint8_t>(offer.write_count);
  operation->field_kind = offer.fragment.field_kind;
  operation->aligned_32b_address =
      offer.fragment.aligned_32b_address;
  operation->byte_mask = offer.fragment.byte_mask;
  std::memcpy(operation->payload, offer.fragment.payload,
              sizeof(operation->payload));
  operation->enqueue_cycle = enqueue_cycle;
  return kStatusOk;
}

}  // namespace

status_kind initialize(engine_state_v0 *state, const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!valid_config(config)) return kStatusInvalidConfiguration;
  *state = engine_state_v0();
  state->config = config;
  state->next_issue_age = 1;
  state->initialized = 1;
  return kStatusOk;
}

status_kind capture_restore_parent_result(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_instance::restore_parent_input_v0 &input,
    const typed_instance::restore_parent_result_v0 &result,
    capture_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      producer_operation_seq == 0 || commit_epoch == 0 ||
      target_operation_seq == 0 ||
      target_operation_seq == producer_operation_seq ||
      !typed_instance::validate_restore_parent_result(input, result)) {
    return kStatusInvalidArgument;
  }
  *receipt = capture_receipt_v0();
  if (operation_live(*state, owner, producer_operation_seq)) {
    return kStatusDuplicateOperation;
  }
  const int result_index = find_free_result(*state);
  if (result_index < 0) return kStatusResultBackpressure;
  const int tracker_index = find_free_tracker(*state);
  if (tracker_index < 0) return kStatusTrackerBackpressure;
  if (state->next_issue_age == 0) return kStatusInvalidArgument;

  instance_semantic::restore_commit_plan_v0 plan = {};
  if (instance_semantic::prepare_restore_parent(
          owner, producer_operation_seq, region, canonical_slot,
          result, &plan) != instance_semantic::kStatusOk ||
      plan.write_fragment_count !=
          instance_semantic::kRestoreWriteFragmentCount) {
    return kStatusSemanticPlanRejected;
  }

  result_commit_entry_v0 entry = {};
  entry.owner = owner;
  entry.issue_age = state->next_issue_age;
  entry.operation_seq = producer_operation_seq;
  entry.target_operation_seq = target_operation_seq;
  entry.commit_epoch = commit_epoch;
  entry.write_count = plan.write_fragment_count;
  entry.tracker_slot = static_cast<uint8_t>(tracker_index);
  entry.valid = 1;
  for (unsigned index = 0; index < plan.write_fragment_count; ++index) {
    entry.writes[index] = plan.write_fragments[index];
  }

  commit_tracker_v0 tracker = {};
  tracker.owner = owner;
  tracker.issue_age = state->next_issue_age;
  tracker.operation_seq = producer_operation_seq;
  tracker.target_operation_seq = target_operation_seq;
  tracker.commit_epoch = commit_epoch;
  tracker.expected_write_count = plan.write_fragment_count;
  tracker.valid = 1;

  state->result_entries[result_index] = entry;
  state->trackers[tracker_index] = tracker;
  ++state->next_issue_age;
  receipt->producer_operation_seq = producer_operation_seq;
  receipt->target_operation_seq = target_operation_seq;
  receipt->commit_epoch = commit_epoch;
  receipt->write_count = plan.write_fragment_count;
  receipt->valid = 1;
  return kStatusOk;
}

status_kind peek_write_offer(const engine_state_v0 &state,
                             write_offer_v0 *offer) {
  if (offer == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *offer = write_offer_v0();
  const int index = find_oldest_result(state);
  if (index < 0) return kStatusNoWriteOffer;
  build_offer(state.result_entries[index], offer);
  return kStatusOk;
}

status_kind transfer_next_write(
    engine_state_v0 *state, private_shared::backing_state_v0 *shared_state,
    uint64_t enqueue_cycle, transfer_receipt_v0 *receipt) {
  if (state == NULL || shared_state == NULL || receipt == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *receipt = transfer_receipt_v0();
  write_offer_v0 offer = {};
  const status_kind peek_status = peek_write_offer(*state, &offer);
  if (peek_status != kStatusOk) return peek_status;
  private_shared::shared_write_v0 operation = {};
  const status_kind prepare_status =
      prepare_shared_write(offer, enqueue_cycle, &operation);
  if (prepare_status != kStatusOk) return prepare_status;
  const private_shared::status_kind memory_status =
      private_shared::validate_runtime_write(*shared_state, operation);
  if (memory_status == private_shared::kStatusQueueFull) {
    return kStatusSharedQueueBackpressure;
  }
  if (memory_status != private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }

  engine_state_v0 staged = *state;
  const int result_index = find_result(staged, offer);
  if (result_index < 0) return kStatusWriteOfferMismatch;
  result_commit_entry_v0 &entry = staged.result_entries[result_index];
  if (entry.next_write_index + 1 != offer.memory_operation_seq ||
      entry.next_write_index >= entry.write_count) {
    return kStatusWriteOfferMismatch;
  }
  commit_tracker_v0 &tracker = staged.trackers[entry.tracker_slot];
  const uint16_t bit =
      static_cast<uint16_t>(uint16_t{1} << entry.next_write_index);
  if (tracker.valid == 0 ||
      (tracker.accepted_write_mask & bit) != 0) {
    return kStatusWriteOfferMismatch;
  }
  tracker.accepted_write_mask |= bit;
  ++entry.next_write_index;
  if (entry.next_write_index == entry.write_count) {
    entry = result_commit_entry_v0();
  }
  if (private_shared::enqueue_runtime_write(shared_state, operation) !=
      private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }
  *state = staged;
  receipt->valid = true;
  receipt->instance_offer = offer;
  receipt->shared_write = operation;
  return kStatusOk;
}

bool owns_ack(const engine_state_v0 &state,
              const private_shared::runtime_write_ack_v0 &ack) {
  return state.initialized == 1 && ack.valid &&
         find_tracker(state, ack.owner, ack.operation_seq,
                      ack.commit_epoch) >= 0;
}

status_kind service_next_ack(
    engine_state_v0 *state, private_shared::backing_state_v0 *shared_state,
    uint64_t service_cycle, ack_receipt_v0 *receipt) {
  if (state == NULL || shared_state == NULL || receipt == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *receipt = ack_receipt_v0();
  private_shared::runtime_write_ack_v0 ack = {};
  const private_shared::status_kind peek_status =
      private_shared::detail::peek_runtime_write_ack(
          *shared_state, service_cycle, &ack);
  if (peek_status == private_shared::kStatusNoAckReady) {
    return kStatusNoAckReady;
  }
  if (peek_status != private_shared::kStatusOk) {
    return kStatusSharedAckRejected;
  }
  const int tracker_index =
      find_tracker(*state, ack.owner, ack.operation_seq,
                   ack.commit_epoch);
  if (tracker_index < 0) return kStatusUnknownAck;
  commit_tracker_v0 &tracker = state->trackers[tracker_index];
  if (ack.memory_operation_seq == 0 ||
      ack.memory_operation_seq > tracker.expected_write_count) {
    return kStatusStaleAck;
  }
  const uint16_t bit = static_cast<uint16_t>(
      uint16_t{1} << (ack.memory_operation_seq - 1));
  if ((tracker.accepted_write_mask & bit) == 0) {
    return kStatusAckBeforeTransfer;
  }
  if ((tracker.acknowledged_write_mask & bit) != 0) {
    return kStatusDuplicateAck;
  }

  engine_state_v0 staged = *state;
  commit_tracker_v0 &staged_tracker =
      staged.trackers[tracker_index];
  staged_tracker.acknowledged_write_mask |= bit;
  if (staged_tracker.acknowledged_write_mask ==
      expected_mask(staged_tracker.expected_write_count)) {
    staged_tracker.ready = 1;
  }
  if (private_shared::detail::commit_runtime_write_ack(
          shared_state, service_cycle, ack) !=
      private_shared::kStatusOk) {
    return kStatusSharedAckRejected;
  }
  *state = staged;
  receipt->valid = true;
  receipt->shared_ack = ack;
  return kStatusOk;
}

status_kind pop_ready_event(engine_state_v0 *state,
                            ready_event_v0 *event) {
  if (state == NULL || event == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *event = ready_event_v0();
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state->config.tracker_capacity;
       ++index) {
    const commit_tracker_v0 &tracker = state->trackers[index];
    if (tracker.valid != 0 && tracker.ready != 0 &&
        tracker.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = tracker.issue_age;
    }
  }
  if (selected < 0) return kStatusNoReadyEvent;
  const commit_tracker_v0 tracker = state->trackers[selected];
  if (tracker.accepted_write_mask !=
          expected_mask(tracker.expected_write_count) ||
      tracker.acknowledged_write_mask !=
          expected_mask(tracker.expected_write_count)) {
    return kStatusStaleAck;
  }
  event->owner = tracker.owner;
  event->producer_operation_seq = tracker.operation_seq;
  event->target_operation_seq = tracker.target_operation_seq;
  event->commit_epoch = tracker.commit_epoch;
  event->valid = 1;
  state->trackers[selected] = commit_tracker_v0();
  return kStatusOk;
}

uint8_t active_result_entry_count(const engine_state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return count;
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    count += state.result_entries[index].valid != 0;
  }
  return count;
}

uint8_t active_tracker_count(const engine_state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return count;
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    count += state.trackers[index].valid != 0;
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
    case kStatusResultBackpressure:
      return "result_backpressure";
    case kStatusTrackerBackpressure:
      return "tracker_backpressure";
    case kStatusDuplicateOperation:
      return "duplicate_operation";
    case kStatusSemanticPlanRejected:
      return "semantic_plan_rejected";
    case kStatusNoWriteOffer:
      return "no_write_offer";
    case kStatusWriteOfferMismatch:
      return "write_offer_mismatch";
    case kStatusSharedQueueBackpressure:
      return "shared_queue_backpressure";
    case kStatusSharedWriteRejected:
      return "shared_write_rejected";
    case kStatusNoAckReady:
      return "no_ack_ready";
    case kStatusUnknownAck:
      return "unknown_ack";
    case kStatusStaleAck:
      return "stale_ack";
    case kStatusAckBeforeTransfer:
      return "ack_before_transfer";
    case kStatusDuplicateAck:
      return "duplicate_ack";
    case kStatusSharedAckRejected:
      return "shared_ack_rejected";
    case kStatusNoReadyEvent:
      return "no_ready_event";
  }
  return "unknown";
}

}  // namespace instance_shared
}  // namespace v04
}  // namespace rtcore
