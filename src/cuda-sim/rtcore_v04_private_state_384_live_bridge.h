#ifndef RTCORE_V04_PRIVATE_STATE_384_LIVE_BRIDGE_H
#define RTCORE_V04_PRIVATE_STATE_384_LIVE_BRIDGE_H

#include <cstddef>
#include <cstdint>

#include "rtcore_replay_interface.h"
#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_private_state_384_backing.h"
#include "rtcore_v04_private_storage_profile.h"

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace live_bridge {

static const uint8_t kMaxReadRequests =
    operand_materializer::kMaxOperationReadChunks;

typedef operand_plan::chunk_delta_v1 sparse_chunk_delta_v1;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOwner,
  kStatusInvalidOperation,
  kStatusInvalidDestination,
  kStatusMalformedTransport,
  kStatusCollectorRejected,
  kStatusBackingRejected,
  kStatusInvalidProfile,
  kStatusInvalidCommit,
  kStatusInvalidWriteAck,
  kStatusDuplicateWriteAck,
  kStatusWriteRejected,
};

struct live_operation_key_v1 {
  operand_materializer::operation_identity_v1 identity;
  uint64_t private_slot_base_address;
  uint32_t private_layout_profile_id;
  uint32_t bvh_format_profile_id;
  uint32_t reservation_generation;
  uint8_t storage_profile;
  uint8_t consumer;
  uint8_t operation;
  uint8_t completion_reason;
};

struct read_input_v1 {
  private_frontier::owner_binding_v0 owner;
  uint64_t private_slot_base_address;
  uint64_t issue_cycle;
  uint32_t operation_sequence;
  uint32_t bvh_format_profile_id;
  uint32_t reservation_generation;
  uint8_t storage_profile;
  uint8_t consumer;
  uint8_t operation;
  uint8_t completion_reason;
  uint8_t destination;
  uint8_t memory_op_seq_base;
  uint8_t reserved_zero[2];
};

struct read_request_plan_v1 {
  live_operation_key_v1 key;
  operand_plan::read_plan_v1 operand_plan;
  operand_materializer::operation_identity_v1 identity;
  rtcore_memory_unit_request_snapshot requests[kMaxReadRequests];
  uint8_t request_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

struct write_commit_input_v1 {
  private_frontier::owner_binding_v0 owner;
  uint64_t private_slot_base_address;
  uint32_t operation_sequence;
  uint32_t commit_epoch;
  uint32_t bvh_format_profile_id;
  uint16_t expected_write_ack_count;
  uint8_t storage_profile;
  uint8_t producer;
  uint8_t reserved_zero[4];
};

struct pending_sparse_commit_v1 {
  struct expected_write_ack_v1 {
    uint64_t aligned_32b_address;
    uint32_t byte_mask;
    uint16_t memory_operation_seq;
    uint8_t chunk_id;
    uint8_t chunk_count;
    uint8_t field_kind;
    uint8_t valid;
    uint8_t payload[kChunkBytes];
  };

  live_operation_key_v1 key;
  operand_plan::unit_sparse_write_plan_v1 merged_write_plan;
  uint32_t commit_epoch;
  uint16_t expected_ack_mask;
  uint16_t registered_ack_mask;
  uint16_t acknowledged_ack_mask;
  uint8_t producer;
  uint8_t expected_write_ack_count;
  uint8_t valid;
  uint8_t committed;
  uint8_t reserved_zero[2];
  expected_write_ack_v1 expected_writes[16];
};

status_kind prepare_read_requests(const read_input_v1 &input,
                                  read_request_plan_v1 *plan);

status_kind initialize_collector(
    const read_request_plan_v1 &plan,
    operand_materializer::response_collector_v1 *collector);

status_kind accept_read_response(
    const backing::state_v1 &state,
    const rtcore_memory_unit_request_snapshot &request,
    operand_materializer::response_collector_v1 *collector);

status_kind accept_read_response_bytes(
    const rtcore_memory_unit_request_snapshot &request,
    const uint8_t *payload, size_t payload_byte_count,
    operand_materializer::response_collector_v1 *collector);

status_kind stage_sparse_commit(
    const write_commit_input_v1 &input,
    const sparse_chunk_delta_v1 *deltas, size_t delta_count,
    pending_sparse_commit_v1 *pending);

status_kind register_modeled_write(
    const private_shared::shared_write_v0 &write,
    pending_sparse_commit_v1 *pending);

status_kind prepare_global_modeled_write(
    const pending_sparse_commit_v1 &pending, uint8_t write_index,
    uint64_t enqueue_cycle, private_shared::shared_write_v0 *write);

status_kind accept_global_write_ack(
    const private_shared::shared_write_v0 &write,
    const private_shared::runtime_write_ack_v0 &ack,
    pending_sparse_commit_v1 *pending, bool *all_acknowledged);

status_kind accept_write_ack_and_maybe_commit(
    backing::state_v1 *state,
    const private_shared::shared_write_v0 &write,
    const private_shared::runtime_write_ack_v0 &ack,
    pending_sparse_commit_v1 *pending, bool *canonical_committed);

uint64_t private_slot_base(
    const private_frontier::owner_binding_v0 &owner);

const char *status_name(status_kind status);

}  // namespace live_bridge
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore

#endif
