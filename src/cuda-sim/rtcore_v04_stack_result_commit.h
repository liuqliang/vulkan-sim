#ifndef RTCORE_V04_STACK_RESULT_COMMIT_H
#define RTCORE_V04_STACK_RESULT_COMMIT_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_stack_result_semantic_applier.h"

namespace rtcore {
namespace v04 {
namespace stack_commit {

static const uint8_t kMaxResultCommitEntries = 8;
static const uint8_t kMaxRequestCommitTrackers = 16;
static const uint8_t kStackSpillWriteFragments =
    (private_frontier::kTransitionSpillOffset %
         private_frontier::kSharedAccessChunkBytes +
     private_frontier::kStackTransitionSpillBytes +
     private_frontier::kSharedAccessChunkBytes - 1) /
    private_frontier::kSharedAccessChunkBytes;
static const uint8_t kMaxWritesPerTransaction =
    stack_semantic::kMaxPrivateWriteFragments + kStackSpillWriteFragments;

static_assert(kStackSpillWriteFragments == 5,
              "Stack transition spill fragment count changed");
static_assert(kMaxWritesPerTransaction == 11,
              "Stack commit write bound changed");
static_assert(kMaxWritesPerTransaction <= 16,
              "Stack commit ACK mask exceeds 16 bits");

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusResultCommitBackpressure,
  kStatusTrackerBackpressure,
  kStatusDuplicateOperation,
  kStatusOperationSequenceExhausted,
  kStatusInvalidTypedOperation,
  kStatusSemanticPlanRejected,
  kStatusForwardingDecisionRejected,
  kStatusSpillPlanRejected,
  kStatusNoWriteOffer,
  kStatusWriteOfferMismatch,
  kStatusUnknownAck,
  kStatusStaleAck,
  kStatusAckBeforeTransfer,
  kStatusDuplicateAck,
  kStatusNoReadyEvent,
};

enum forwarding_kind : uint8_t {
  kForwardingInvalid = 0,
  kForwardingRegistered = 1,
  kForwardingSpillToMemory = 2,
};

enum ready_kind : uint8_t {
  kReadyInvalid = 0,
  kReadyForwardedTarget = 1,
  kReadySpillRecovery = 2,
};

struct forwarding_decision_input_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint16_t persistent_write_count;
  uint8_t reserved_zero[6];
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
};

typedef forwarding_kind (*forwarding_selector_v0)(
    void *context, const forwarding_decision_input_v0 &input);

struct config_v0 {
  uint8_t result_commit_capacity;
  uint8_t tracker_capacity;
  uint8_t reserved_zero[6];
};

struct write_offer_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint16_t memory_operation_seq;
  uint16_t write_count;
  uint8_t reserved_zero[4];
  stack_semantic::private_write_fragment_v0 fragment;
};

struct issue_receipt_v0 {
  uint32_t commit_epoch;
  uint16_t write_count;
  uint8_t forwarding_kind;
  uint8_t immediate_ready;
  typed_node::selected_child_fetch_work_item_v0 registered_prefill;
};

struct ready_event_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint8_t ready_kind;
  uint8_t reserved_zero[7];
};

struct result_commit_entry_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t issue_age;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint16_t write_count;
  uint16_t next_write_index;
  uint8_t valid;
  uint8_t forwarding_kind;
  uint8_t tracker_slot;
  uint8_t reserved_zero;
  stack_semantic::private_write_fragment_v0 writes[kMaxWritesPerTransaction];
};

struct request_commit_tracker_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t issue_age;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint16_t expected_write_count;
  uint16_t accepted_write_mask;
  uint16_t acknowledged_write_mask;
  uint8_t valid;
  uint8_t forwarding_kind;
  uint8_t payload_transferred;
  uint8_t ready;
};

struct engine_state_v0 {
  config_v0 config;
  uint32_t next_commit_epoch;
  uint64_t next_issue_age;
  uint8_t initialized;
  uint8_t reserved_zero[3];
  result_commit_entry_v0 result_entries[kMaxResultCommitEntries];
  request_commit_tracker_v0 trackers[kMaxRequestCommitTrackers];
};

static_assert(sizeof(write_offer_v0) == 88, "Stack write offer layout changed");
static_assert(offsetof(write_offer_v0, fragment) == 40,
              "Stack write offer fragment offset changed");

status_kind initialize(engine_state_v0 *state, const config_v0 &config);

status_kind issue_stack_push(
    engine_state_v0 *state, const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_input_v0 &input, forwarding_kind forwarding,
    issue_receipt_v0 *receipt);

status_kind issue_stack_push_with_selector(
    engine_state_v0 *state, const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_stack::push_input_v0 &input, forwarding_selector_v0 selector,
    void *selector_context, issue_receipt_v0 *receipt);

status_kind peek_write_offer(const engine_state_v0 &state,
                             write_offer_v0 *offer);

status_kind validate_write_offer(const engine_state_v0 &state,
                                 const write_offer_v0 &offer);

status_kind accept_write_offer(engine_state_v0 *state,
                               const write_offer_v0 &offer);

status_kind validate_write_ack(const engine_state_v0 &state,
                               const private_frontier::owner_binding_v0 &owner,
                               uint32_t operation_seq, uint32_t commit_epoch,
                               uint16_t memory_operation_seq);

status_kind accept_write_ack(engine_state_v0 *state,
                             const private_frontier::owner_binding_v0 &owner,
                             uint32_t operation_seq, uint32_t commit_epoch,
                             uint16_t memory_operation_seq);

status_kind pop_ready_event(engine_state_v0 *state, ready_event_v0 *event);

uint8_t active_result_entry_count(const engine_state_v0 &state);
uint8_t active_tracker_count(const engine_state_v0 &state);

const char *status_name(status_kind status);

}  // namespace stack_commit
}  // namespace v04
}  // namespace rtcore

#endif
