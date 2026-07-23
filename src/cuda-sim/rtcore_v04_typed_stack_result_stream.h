#ifndef RTCORE_V04_TYPED_STACK_RESULT_STREAM_H
#define RTCORE_V04_TYPED_STACK_RESULT_STREAM_H

#include <cstdint>
#include <vector>

#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_typed_stack_kernel.h"

namespace rtcore {
namespace v04 {
namespace typed_stack_stream {

static const uint32_t kMaxPushResultsPerLane = 256;
static const uint16_t kPushResultPayloadBytes =
    sizeof(typed_stack::push_result_v0);
static_assert(sizeof(typed_stack::push_result_v0) == 176,
              "typed Stack push result ABI must remain 176 bytes");

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusCapacityExceeded,
  kStatusInvalidSequence,
  kStatusInvalidResult,
  kStatusInvalidOwner,
  kStatusAlreadyBound,
  kStatusOwnerMismatch,
};

struct push_result_record_v0 {
  bool valid;
  bool owner_bound;
  uint16_t reserved_zero;
  uint32_t operation_seq;
  uint64_t producer_node_reference;
  uint8_t operation_kind;
  uint8_t result_kind;
  uint16_t payload_bytes;
  private_frontier::owner_binding_v0 owner;
  typed_stack::push_result_v0 payload;
};

status_kind append_push_result(
    std::vector<push_result_record_v0> *records,
    uint64_t producer_node_reference,
    const typed_stack::push_result_v0 &result);

status_kind validate_unbound(
    const std::vector<push_result_record_v0> &records);

status_kind validate_bound(
    const std::vector<push_result_record_v0> &records,
    const private_frontier::owner_binding_v0 &owner);

status_kind bind_owner(
    std::vector<push_result_record_v0> *records,
    const private_frontier::owner_binding_v0 &owner);

bool streams_equal(
    const std::vector<push_result_record_v0> &lhs,
    const std::vector<push_result_record_v0> &rhs);

const char *status_name(status_kind status);

}  // namespace typed_stack_stream
}  // namespace v04
}  // namespace rtcore

#endif
