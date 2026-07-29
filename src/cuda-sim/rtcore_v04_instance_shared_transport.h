#ifndef RTCORE_V04_INSTANCE_SHARED_TRANSPORT_H
#define RTCORE_V04_INSTANCE_SHARED_TRANSPORT_H

#include <cstdint>

#include "rtcore_v04_instance_result_semantic_applier.h"
#include "rtcore_v04_private_shared_backing.h"

namespace rtcore {
namespace v04 {
namespace instance_shared {

static const uint8_t kMaxResultCommitEntries = 16;
static const uint8_t kMaxCommitTrackers = 16;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusResultBackpressure,
  kStatusTrackerBackpressure,
  kStatusDuplicateOperation,
  kStatusSemanticPlanRejected,
  kStatusNoWriteOffer,
  kStatusWriteOfferMismatch,
  kStatusSharedQueueBackpressure,
  kStatusSharedWriteRejected,
  kStatusNoAckReady,
  kStatusUnknownAck,
  kStatusStaleAck,
  kStatusAckBeforeTransfer,
  kStatusDuplicateAck,
  kStatusSharedAckRejected,
  kStatusNoReadyEvent,
};

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
  instance_semantic::private_write_fragment_v0 fragment;
};

struct capture_receipt_v0 {
  uint32_t producer_operation_seq;
  uint32_t target_operation_seq;
  uint32_t commit_epoch;
  uint16_t write_count;
  uint8_t valid;
  uint8_t reserved_zero;
};

struct transfer_receipt_v0 {
  bool valid;
  uint8_t reserved_zero[7];
  write_offer_v0 instance_offer;
  private_shared::shared_write_v0 shared_write;
};

struct ack_receipt_v0 {
  bool valid;
  uint8_t reserved_zero[7];
  private_shared::runtime_write_ack_v0 shared_ack;
};

struct ready_event_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t producer_operation_seq;
  uint32_t target_operation_seq;
  uint32_t commit_epoch;
  uint32_t root_build_generation;
  uint8_t valid;
  uint8_t route_kind;
  uint8_t reserved_zero[2];
  typed_node::selected_child_fetch_work_item_v0 root_fetch;
  typed_node::ray_policy_v0 ray_policy;
};

struct result_commit_entry_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t issue_age;
  uint32_t operation_seq;
  uint32_t target_operation_seq;
  uint32_t commit_epoch;
  uint16_t write_count;
  uint16_t next_write_index;
  uint8_t tracker_slot;
  uint8_t valid;
  uint8_t reserved_zero[2];
  instance_semantic::private_write_fragment_v0
      writes[instance_semantic::kMaxWriteFragmentCount];
};

struct commit_tracker_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t issue_age;
  uint32_t operation_seq;
  uint32_t target_operation_seq;
  uint32_t commit_epoch;
  uint32_t root_build_generation;
  uint16_t expected_write_count;
  uint16_t accepted_write_mask;
  uint16_t acknowledged_write_mask;
  uint8_t valid;
  uint8_t ready;
  uint8_t route_kind;
  uint8_t reserved_zero[3];
  typed_node::selected_child_fetch_work_item_v0 root_fetch;
  typed_node::ray_policy_v0 ray_policy;
};

struct engine_state_v0 {
  config_v0 config;
  uint64_t next_issue_age;
  uint8_t initialized;
  uint8_t reserved_zero[7];
  result_commit_entry_v0 result_entries[kMaxResultCommitEntries];
  commit_tracker_v0 trackers[kMaxCommitTrackers];
};

status_kind initialize(engine_state_v0 *state, const config_v0 &config);

status_kind capture_restore_parent_result(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_instance::restore_parent_input_v0 &input,
    const typed_instance::restore_parent_result_v0 &result,
    capture_receipt_v0 *receipt);

status_kind capture_enter_result(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_instance::enter_input_v0 &input,
    const typed_instance::enter_result_v0 &result,
    capture_receipt_v0 *receipt);

status_kind peek_write_offer(const engine_state_v0 &state,
                             write_offer_v0 *offer);

status_kind transfer_next_write(
    engine_state_v0 *state, private_shared::backing_state_v0 *shared_state,
    uint64_t enqueue_cycle, transfer_receipt_v0 *receipt);

bool owns_ack(const engine_state_v0 &state,
              const private_shared::runtime_write_ack_v0 &ack);

status_kind service_next_ack(
    engine_state_v0 *state, private_shared::backing_state_v0 *shared_state,
    uint64_t service_cycle, ack_receipt_v0 *receipt);

status_kind pop_ready_event(engine_state_v0 *state,
                            ready_event_v0 *event);

uint8_t active_result_entry_count(const engine_state_v0 &state);
uint8_t active_tracker_count(const engine_state_v0 &state);

const char *status_name(status_kind status);

}  // namespace instance_shared
}  // namespace v04
}  // namespace rtcore

#endif
