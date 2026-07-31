#ifndef RTCORE_V04_PRIVATE_STATE_384_LIVE_BRIDGE_H
#define RTCORE_V04_PRIVATE_STATE_384_LIVE_BRIDGE_H

#include <cstddef>
#include <cstdint>

#include "rtcore_replay_interface.h"
#include "rtcore_v04_private_state_384_backing.h"

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace live_bridge {

static const uint8_t kMaxReadRequests =
    operand_materializer::kMaxOperationReadChunks;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOwner,
  kStatusInvalidOperation,
  kStatusInvalidDestination,
  kStatusMalformedTransport,
  kStatusCollectorRejected,
  kStatusBackingRejected,
  kStatusWriteAckRequired,
  kStatusWriteRejected,
};

struct read_input_v1 {
  private_frontier::owner_binding_v0 owner;
  uint64_t issue_cycle;
  uint32_t operation_sequence;
  uint8_t consumer;
  uint8_t operation;
  uint8_t completion_reason;
  uint8_t destination;
};

struct read_request_plan_v1 {
  operand_plan::read_plan_v1 operand_plan;
  operand_materializer::operation_identity_v1 identity;
  rtcore_memory_unit_request_snapshot requests[kMaxReadRequests];
  uint8_t request_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

struct write_commit_input_v1 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_sequence;
  uint8_t producer;
  uint8_t matching_write_ack;
  uint8_t reserved_zero[2];
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

status_kind commit_sparse_deltas_after_ack(
    backing::state_v1 *state, const write_commit_input_v1 &input,
    const operand_plan::chunk_delta_v1 *deltas, size_t delta_count);

uint64_t private_slot_base(
    const private_frontier::owner_binding_v0 &owner);

const char *status_name(status_kind status);

}  // namespace live_bridge
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore

#endif
