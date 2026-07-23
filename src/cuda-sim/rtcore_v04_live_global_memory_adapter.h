#ifndef RTCORE_V04_LIVE_GLOBAL_MEMORY_ADAPTER_H
#define RTCORE_V04_LIVE_GLOBAL_MEMORY_ADAPTER_H

#include <cstdint>

#include "rtcore_replay_interface.h"
#include "rtcore_v04_target_memory_bridge.h"

class memory_space;

namespace rtcore {
namespace v04 {
namespace live_global_memory {

static const uint8_t kMaxRequestChunks =
    target_memory::kMaxRawReadChunks;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusMalformedRawRead,
  kStatusMalformedTransport,
  kStatusResponseAddressMismatch,
  kStatusResponseSizeMismatch,
  kStatusTargetFillRejected,
};

struct request_plan_v0 {
  rtcore_memory_unit_request_snapshot requests[kMaxRequestChunks];
  uint8_t request_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

status_kind lower_raw_read_chunk(
    const target_memory::raw_read_chunk_v0 &chunk,
    uint64_t issue_cycle,
    rtcore_memory_unit_request_snapshot *request);

status_kind reconstruct_raw_read_chunk(
    const rtcore_memory_unit_request_snapshot &request,
    target_memory::raw_read_chunk_v0 *chunk);

status_kind prepare_request_plan(
    const target_memory::raw_read_plan_v0 &raw_plan,
    uint64_t issue_cycle, request_plan_v0 *request_plan);

status_kind materialize_functional_response(
    const rtcore_memory_unit_request_snapshot &request,
    memory_space *global_memory, uint64_t response_address,
    uint8_t *response_payload, uint8_t response_bytes);

status_kind accept_materialized_response(
    fetch_target::engine_state_v0 *target_state,
    const rtcore_memory_unit_request_snapshot &request,
    const uint8_t *response_payload, uint8_t response_bytes);

const char *status_name(status_kind status);

}  // namespace live_global_memory
}  // namespace v04
}  // namespace rtcore

extern "C" bool rtcore_v04_live_global_memory_adapter_active();

extern "C" bool rtcore_v04_live_target_reserve_and_enqueue_raw_reads(
    unsigned owner_hw_sid,
    const rtcore::v04::stack_commit::forwarding_decision_input_v0 *decision,
    unsigned long long reservation_cycle,
    rtcore::v04::fetch_target::reservation_receipt_v0 *reservation);

extern "C" bool rtcore_v04_live_target_complete_producer(
    unsigned owner_hw_sid,
    const rtcore::v04::private_frontier::owner_binding_v0 *owner,
    unsigned operation_seq, unsigned commit_epoch,
    rtcore::v04::fetch_target::reservation_receipt_v0 *reservation);

extern "C" bool rtcore_v04_live_target_pop_ready_operation(
    unsigned owner_hw_sid, unsigned target_kind, bool unit_accepts,
    rtcore::v04::fetch_target::operation_packet_v0 *packet);

#endif
