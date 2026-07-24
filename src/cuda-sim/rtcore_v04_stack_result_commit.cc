#include "rtcore_v04_stack_result_commit.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace stack_commit {
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
         config.tracker_capacity <= kMaxRequestCommitTrackers &&
         bytes_are_zero(config.reserved_zero, sizeof(config.reserved_zero));
}

uint16_t expected_mask(uint16_t write_count) {
  if (write_count == 0) return 0;
  return static_cast<uint16_t>((uint32_t{1} << write_count) - 1);
}

int find_free_result_entry(const engine_state_v0 &state) {
  for (unsigned index = 0; index < state.config.result_commit_capacity;
       ++index) {
    if (state.result_entries[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_free_tracker(const engine_state_v0 &state) {
  for (unsigned index = 0; index < state.config.tracker_capacity; ++index) {
    if (state.trackers[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

bool operation_is_live(const engine_state_v0 &state,
                       const private_frontier::owner_binding_v0 &owner,
                       uint32_t operation_seq) {
  for (unsigned index = 0; index < state.config.tracker_capacity; ++index) {
    const request_commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid != 0 && tracker.operation_seq == operation_seq &&
        private_frontier::owners_equal(tracker.owner, owner)) {
      return true;
    }
  }
  return false;
}

bool materialize_fragment(
    const private_frontier::shadow_slot_v0 &updated_slot,
    const private_frontier::shared_chunk_access_v0 &access,
    stack_semantic::private_write_fragment_v0 *fragment) {
  if (fragment == NULL ||
      access.access_kind != private_frontier::kAccessWrite ||
      access.slot_byte_offset >= private_frontier::kPrivateDataSlotBytes) {
    return false;
  }
  const uint32_t aligned_slot_offset =
      access.slot_byte_offset -
      (access.slot_byte_offset % private_frontier::kSharedAccessChunkBytes);
  if (aligned_slot_offset > private_frontier::kPrivateDataSlotBytes -
                                private_frontier::kSharedAccessChunkBytes) {
    return false;
  }

  stack_semantic::private_write_fragment_v0 prepared = {};
  prepared.aligned_32b_address = access.aligned_32b_address;
  prepared.byte_mask = access.byte_mask;
  prepared.slot_byte_offset = access.slot_byte_offset;
  prepared.byte_count = access.byte_count;
  prepared.field_kind = access.field_kind;
  for (unsigned byte = 0; byte < private_frontier::kSharedAccessChunkBytes;
       ++byte) {
    if ((access.byte_mask & (uint32_t{1} << byte)) != 0) {
      prepared.payload[byte] = updated_slot.bytes[aligned_slot_offset + byte];
    }
  }
  if (!stack_semantic::validate_private_write_fragment(prepared)) {
    return false;
  }
  *fragment = prepared;
  return true;
}

int find_oldest_offer_entry(const engine_state_v0 &state) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.result_commit_capacity;
       ++index) {
    const result_commit_entry_v0 &entry = state.result_entries[index];
    if (entry.valid != 0 && entry.next_write_index < entry.write_count &&
        entry.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = entry.issue_age;
    }
  }
  return selected;
}

void build_offer(const result_commit_entry_v0 &entry, write_offer_v0 *offer) {
  std::memset(offer, 0, sizeof(*offer));
  offer->owner = entry.owner;
  offer->operation_seq = entry.operation_seq;
  offer->commit_epoch = entry.commit_epoch;
  offer->memory_operation_seq =
      static_cast<uint16_t>(entry.next_write_index + 1);
  offer->write_count = entry.write_count;
  offer->fragment = entry.writes[entry.next_write_index];
}

int find_result_entry(const engine_state_v0 &state,
                      const write_offer_v0 &offer) {
  for (unsigned index = 0; index < state.config.result_commit_capacity;
       ++index) {
    const result_commit_entry_v0 &entry = state.result_entries[index];
    if (entry.valid != 0 && entry.operation_seq == offer.operation_seq &&
        entry.commit_epoch == offer.commit_epoch &&
        private_frontier::owners_equal(entry.owner, offer.owner)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_exact_tracker(const engine_state_v0 &state,
                       const private_frontier::owner_binding_v0 &owner,
                       uint32_t operation_seq, uint32_t commit_epoch) {
  for (unsigned index = 0; index < state.config.tracker_capacity; ++index) {
    const request_commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid != 0 && tracker.operation_seq == operation_seq &&
        tracker.commit_epoch == commit_epoch &&
        private_frontier::owners_equal(tracker.owner, owner)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

bool has_other_epoch(const engine_state_v0 &state,
                     const private_frontier::owner_binding_v0 &owner,
                     uint32_t operation_seq) {
  for (unsigned index = 0; index < state.config.tracker_capacity; ++index) {
    const request_commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid != 0 && tracker.operation_seq == operation_seq &&
        private_frontier::owners_equal(tracker.owner, owner)) {
      return true;
    }
  }
  return false;
}

forwarding_kind select_fixed_forwarding(
    void *context, const forwarding_decision_input_v0 &) {
  if (context == NULL) return kForwardingInvalid;
  return *static_cast<const forwarding_kind *>(context);
}

}  // namespace

status_kind initialize(engine_state_v0 *state, const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!valid_config(config)) return kStatusInvalidConfiguration;
  std::memset(state, 0, sizeof(*state));
  state->config = config;
  state->next_commit_epoch = 1;
  state->next_issue_age = 1;
  state->initialized = 1;
  return kStatusOk;
}

status_kind issue_stack_push(
    engine_state_v0 *state, const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_input_v0 &input, forwarding_kind forwarding,
    issue_receipt_v0 *receipt) {
  if (forwarding != kForwardingRegistered &&
      forwarding != kForwardingSpillToMemory) {
    return kStatusInvalidArgument;
  }
  return issue_stack_push_with_selector(
      state, owner, operation_seq, region, canonical_slot, input,
      select_fixed_forwarding, &forwarding, receipt);
}

status_kind issue_stack_push_with_selector(
    engine_state_v0 *state, const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_input_v0 &input, forwarding_selector_v0 selector,
    void *selector_context, issue_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      selector == NULL || operation_seq == 0 ||
      state->next_commit_epoch == 0) {
    return kStatusInvalidArgument;
  }
  const typed_stack::push_result_v0 result =
      typed_stack::execute_push(input);
  const typed_node::ray_policy_v0 compatibility_ray_policy = {};
  const status_kind status = capture_stack_push_result_with_selector(
      state, owner, operation_seq, state->next_commit_epoch,
      operation_seq, region, canonical_slot, input, result,
      compatibility_ray_policy, selector, selector_context, receipt);
  if (status == kStatusOk) {
    ++state->next_commit_epoch;
  }
  return status;
}

status_kind capture_stack_push_result_with_selector(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_input_v0 &input,
    const typed_stack::push_result_v0 &result,
    const typed_node::ray_policy_v0 &forwarded_ray_policy,
    forwarding_selector_v0 selector, void *selector_context,
    issue_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      selector == NULL || producer_operation_seq == 0 ||
      target_operation_seq == 0 || commit_epoch == 0 ||
      !bytes_are_zero(forwarded_ray_policy.reserved_zero,
                      sizeof(forwarded_ray_policy.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  std::memset(receipt, 0, sizeof(*receipt));
  if (operation_is_live(*state, owner, producer_operation_seq)) {
    return kStatusDuplicateOperation;
  }
  const int result_slot = find_free_result_entry(*state);
  if (result_slot < 0) return kStatusResultCommitBackpressure;
  const int tracker_slot = find_free_tracker(*state);
  if (tracker_slot < 0) return kStatusTrackerBackpressure;
  if (state->next_issue_age == 0) {
    return kStatusOperationSequenceExhausted;
  }

  if (input.profile_id != typed_stack::kGenRtDerivedProfileId ||
      input.operation_kind !=
          typed_stack::kPushRemainderAndForwardSelected ||
      !bytes_are_zero(input.reserved_zero0,
                      sizeof(input.reserved_zero0)) ||
      !bytes_are_zero(input.reserved_zero1,
                      sizeof(input.reserved_zero1)) ||
      std::memcmp(&input.node_route.selected_fetch,
                  &result.selected_fetch,
                  sizeof(result.selected_fetch)) != 0 ||
      !typed_stack::validate_push_result(result)) {
    return kStatusInvalidTypedOperation;
  }

  stack_semantic::append_commit_plan_v0 semantic_plan = {};
  if (stack_semantic::prepare_stack_pushed_and_selected(
          owner, producer_operation_seq, region, canonical_slot, result,
          &semantic_plan) != stack_semantic::kStatusOk) {
    return kStatusSemanticPlanRejected;
  }

  forwarding_decision_input_v0 decision_input = {};
  decision_input.owner = owner;
  decision_input.producer_operation_seq =
      producer_operation_seq;
  decision_input.target_operation_seq = target_operation_seq;
  decision_input.commit_epoch = commit_epoch;
  decision_input.persistent_write_count =
      semantic_plan.write_fragment_count;
  decision_input.forwarded_ray_policy = forwarded_ray_policy;
  decision_input.selected_fetch = result.selected_fetch;
  const forwarding_kind forwarding =
      selector(selector_context, decision_input);
  if (forwarding != kForwardingRegistered &&
      forwarding != kForwardingSpillToMemory) {
    return kStatusForwardingDecisionRejected;
  }

  result_commit_entry_v0 prepared_entry = {};
  prepared_entry.owner = owner;
  prepared_entry.issue_age = state->next_issue_age;
  prepared_entry.operation_seq = producer_operation_seq;
  prepared_entry.target_operation_seq = target_operation_seq;
  prepared_entry.commit_epoch = commit_epoch;
  prepared_entry.forwarding_kind = forwarding;
  prepared_entry.tracker_slot = static_cast<uint8_t>(tracker_slot);
  prepared_entry.write_count = semantic_plan.write_fragment_count;
  for (unsigned index = 0; index < semantic_plan.write_fragment_count;
       ++index) {
    prepared_entry.writes[index] = semantic_plan.write_fragments[index];
  }

  if (forwarding == kForwardingSpillToMemory) {
    private_frontier::shadow_slot_v0 spill_slot = canonical_slot;
    private_frontier::access_plan_v0 spill_plan = {};
    if (private_frontier::apply_stack_selected_fetch_spill(
            &spill_slot, owner, region, result, &spill_plan) !=
            private_frontier::kStatusOk ||
        spill_plan.access_count != kStackSpillWriteFragments ||
        !private_frontier::owners_equal(spill_plan.owner, owner) ||
        prepared_entry.write_count >
            kMaxWritesPerTransaction - spill_plan.access_count) {
      return kStatusSpillPlanRejected;
    }
    for (unsigned index = 0; index < spill_plan.access_count; ++index) {
      stack_semantic::private_write_fragment_v0 fragment = {};
      if (spill_plan.accesses[index].field_kind !=
              private_frontier::kFieldTransitionSpill ||
          !materialize_fragment(spill_slot, spill_plan.accesses[index],
                                &fragment)) {
        return kStatusSpillPlanRejected;
      }
      prepared_entry.writes[prepared_entry.write_count++] = fragment;
    }
  }
  if (prepared_entry.write_count > kMaxWritesPerTransaction) {
    return kStatusSemanticPlanRejected;
  }

  request_commit_tracker_v0 prepared_tracker = {};
  prepared_tracker.owner = owner;
  prepared_tracker.issue_age = state->next_issue_age;
  prepared_tracker.operation_seq = producer_operation_seq;
  prepared_tracker.target_operation_seq = target_operation_seq;
  prepared_tracker.commit_epoch = commit_epoch;
  prepared_tracker.expected_write_count = prepared_entry.write_count;
  prepared_tracker.forwarding_kind = forwarding;
  prepared_tracker.valid = 1;
  if (prepared_entry.write_count == 0) {
    prepared_tracker.payload_transferred = 1;
    prepared_tracker.ready = 1;
  } else {
    prepared_entry.valid = 1;
  }

  state->trackers[tracker_slot] = prepared_tracker;
  if (prepared_entry.valid != 0) {
    state->result_entries[result_slot] = prepared_entry;
  }

  receipt->producer_operation_seq =
      prepared_tracker.operation_seq;
  receipt->target_operation_seq =
      prepared_tracker.target_operation_seq;
  receipt->commit_epoch = prepared_tracker.commit_epoch;
  receipt->write_count = prepared_tracker.expected_write_count;
  receipt->forwarding_kind = forwarding;
  receipt->immediate_ready = prepared_tracker.ready;
  if (forwarding == kForwardingRegistered) {
    receipt->registered_prefill = result.selected_fetch;
  }

  ++state->next_issue_age;
  return kStatusOk;
}

status_kind capture_stack_push_result_from_projection_with_selector(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::frontier_metadata_image_v0 &frontier_metadata,
    const typed_stack::push_input_v0 &input,
    const typed_stack::push_result_v0 &result,
    const typed_node::ray_policy_v0 &forwarded_ray_policy,
    forwarding_selector_v0 selector, void *selector_context,
    issue_receipt_v0 *receipt) {
  if (frontier_metadata.frontier_top !=
          input.frontier.frontier_top ||
      frontier_metadata.frontier_count !=
          input.frontier.frontier_count ||
      frontier_metadata.frontier_capacity !=
          input.frontier.frontier_capacity) {
    return kStatusSemanticPlanRejected;
  }
  private_frontier::shadow_slot_v0 projected_slot = {};
  private_frontier::access_plan_v0 ignored_init_plan = {};
  if (private_frontier::initialize_shadow_slot(
          &projected_slot, owner, region, frontier_metadata,
          &ignored_init_plan) != private_frontier::kStatusOk) {
    return kStatusSemanticPlanRejected;
  }
  return capture_stack_push_result_with_selector(
      state, owner, producer_operation_seq, commit_epoch,
      target_operation_seq, region, projected_slot, input, result,
      forwarded_ray_policy, selector, selector_context, receipt);
}

status_kind peek_write_offer(const engine_state_v0 &state,
                             write_offer_v0 *offer) {
  if (offer == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  std::memset(offer, 0, sizeof(*offer));
  const int entry_slot = find_oldest_offer_entry(state);
  if (entry_slot < 0) return kStatusNoWriteOffer;
  build_offer(state.result_entries[entry_slot], offer);
  return kStatusOk;
}

status_kind validate_write_offer(const engine_state_v0 &state,
                                 const write_offer_v0 &offer) {
  if (state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  if (!bytes_are_zero(offer.reserved_zero, sizeof(offer.reserved_zero))) {
    return kStatusWriteOfferMismatch;
  }
  const int entry_slot = find_result_entry(state, offer);
  if (entry_slot < 0) return kStatusWriteOfferMismatch;
  const result_commit_entry_v0 &entry = state.result_entries[entry_slot];
  write_offer_v0 expected = {};
  build_offer(entry, &expected);
  if (std::memcmp(&offer, &expected, sizeof(offer)) != 0) {
    return kStatusWriteOfferMismatch;
  }
  if (entry.tracker_slot >= state.config.tracker_capacity) {
    return kStatusWriteOfferMismatch;
  }
  const request_commit_tracker_v0 &tracker = state.trackers[entry.tracker_slot];
  if (tracker.valid != 1 || tracker.operation_seq != entry.operation_seq ||
      tracker.target_operation_seq != entry.target_operation_seq ||
      tracker.commit_epoch != entry.commit_epoch ||
      !private_frontier::owners_equal(tracker.owner, entry.owner)) {
    return kStatusWriteOfferMismatch;
  }
  return kStatusOk;
}

status_kind accept_write_offer(engine_state_v0 *state,
                               const write_offer_v0 &offer) {
  if (state == NULL) return kStatusInvalidArgument;
  const status_kind validation = validate_write_offer(*state, offer);
  if (validation != kStatusOk) return validation;
  const int entry_slot = find_result_entry(*state, offer);
  result_commit_entry_v0 &entry = state->result_entries[entry_slot];
  request_commit_tracker_v0 &tracker = state->trackers[entry.tracker_slot];
  const uint16_t write_bit =
      static_cast<uint16_t>(uint16_t{1} << entry.next_write_index);
  tracker.accepted_write_mask |= write_bit;
  ++entry.next_write_index;
  if (entry.next_write_index == entry.write_count) {
    tracker.payload_transferred = 1;
    std::memset(&entry, 0, sizeof(entry));
  }
  return kStatusOk;
}

status_kind validate_write_ack(const engine_state_v0 &state,
                               const private_frontier::owner_binding_v0 &owner,
                               uint32_t operation_seq, uint32_t commit_epoch,
                               uint16_t memory_operation_seq) {
  if (state.initialized != 1 || operation_seq == 0 || commit_epoch == 0 ||
      memory_operation_seq == 0) {
    return kStatusInvalidArgument;
  }
  const int tracker_slot =
      find_exact_tracker(state, owner, operation_seq, commit_epoch);
  if (tracker_slot < 0) {
    return has_other_epoch(state, owner, operation_seq) ? kStatusStaleAck
                                                        : kStatusUnknownAck;
  }
  const request_commit_tracker_v0 &tracker = state.trackers[tracker_slot];
  if (memory_operation_seq > tracker.expected_write_count) {
    return kStatusUnknownAck;
  }
  const uint16_t write_bit =
      static_cast<uint16_t>(uint16_t{1} << (memory_operation_seq - 1));
  if ((tracker.accepted_write_mask & write_bit) == 0) {
    return kStatusAckBeforeTransfer;
  }
  if ((tracker.acknowledged_write_mask & write_bit) != 0) {
    return kStatusDuplicateAck;
  }
  return kStatusOk;
}

status_kind accept_write_ack(engine_state_v0 *state,
                             const private_frontier::owner_binding_v0 &owner,
                             uint32_t operation_seq, uint32_t commit_epoch,
                             uint16_t memory_operation_seq) {
  if (state == NULL) return kStatusInvalidArgument;
  const status_kind validation = validate_write_ack(
      *state, owner, operation_seq, commit_epoch, memory_operation_seq);
  if (validation != kStatusOk) return validation;
  const int tracker_slot =
      find_exact_tracker(*state, owner, operation_seq, commit_epoch);
  request_commit_tracker_v0 &tracker = state->trackers[tracker_slot];
  const uint16_t write_bit =
      static_cast<uint16_t>(uint16_t{1} << (memory_operation_seq - 1));
  tracker.acknowledged_write_mask |= write_bit;
  if (tracker.payload_transferred != 0 &&
      tracker.acknowledged_write_mask ==
          expected_mask(tracker.expected_write_count)) {
    tracker.ready = 1;
  }
  return kStatusOk;
}

status_kind pop_ready_event(engine_state_v0 *state, ready_event_v0 *event) {
  if (state == NULL || event == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  std::memset(event, 0, sizeof(*event));
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state->config.tracker_capacity; ++index) {
    const request_commit_tracker_v0 &tracker = state->trackers[index];
    if (tracker.valid != 0 && tracker.ready != 0 &&
        tracker.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = tracker.issue_age;
    }
  }
  if (selected < 0) return kStatusNoReadyEvent;

  const request_commit_tracker_v0 tracker = state->trackers[selected];
  event->owner = tracker.owner;
  event->producer_operation_seq = tracker.operation_seq;
  event->target_operation_seq = tracker.target_operation_seq;
  event->commit_epoch = tracker.commit_epoch;
  event->ready_kind = tracker.forwarding_kind == kForwardingRegistered
                          ? kReadyForwardedTarget
                          : kReadySpillRecovery;
  std::memset(&state->trackers[selected], 0, sizeof(state->trackers[selected]));
  return kStatusOk;
}

uint8_t active_result_entry_count(const engine_state_v0 &state) {
  if (state.initialized != 1) return 0;
  uint8_t count = 0;
  for (unsigned index = 0; index < state.config.result_commit_capacity;
       ++index) {
    count += state.result_entries[index].valid != 0;
  }
  return count;
}

uint8_t active_tracker_count(const engine_state_v0 &state) {
  if (state.initialized != 1) return 0;
  uint8_t count = 0;
  for (unsigned index = 0; index < state.config.tracker_capacity; ++index) {
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
    case kStatusResultCommitBackpressure:
      return "result_commit_backpressure";
    case kStatusTrackerBackpressure:
      return "tracker_backpressure";
    case kStatusDuplicateOperation:
      return "duplicate_operation";
    case kStatusOperationSequenceExhausted:
      return "operation_sequence_exhausted";
    case kStatusInvalidTypedOperation:
      return "invalid_typed_operation";
    case kStatusSemanticPlanRejected:
      return "semantic_plan_rejected";
    case kStatusForwardingDecisionRejected:
      return "forwarding_decision_rejected";
    case kStatusSpillPlanRejected:
      return "spill_plan_rejected";
    case kStatusNoWriteOffer:
      return "no_write_offer";
    case kStatusWriteOfferMismatch:
      return "write_offer_mismatch";
    case kStatusUnknownAck:
      return "unknown_ack";
    case kStatusStaleAck:
      return "stale_ack";
    case kStatusAckBeforeTransfer:
      return "ack_before_transfer";
    case kStatusDuplicateAck:
      return "duplicate_ack";
    case kStatusNoReadyEvent:
      return "no_ready_event";
  }
  return "unknown";
}

}  // namespace stack_commit
}  // namespace v04
}  // namespace rtcore
