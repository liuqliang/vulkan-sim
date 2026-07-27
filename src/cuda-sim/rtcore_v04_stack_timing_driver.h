#ifndef RTCORE_V04_STACK_TIMING_DRIVER_H
#define RTCORE_V04_STACK_TIMING_DRIVER_H

#include <cstdint>

#include "rtcore_v04_stack_operation_queue.h"
#include "rtcore_v04_timing_driver.h"

namespace rtcore {
namespace v04 {
namespace stack_timing {

static const uint8_t kMaxStackUnits = 1;
static const uint8_t kMaxPipelineEntries = 4;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusCycleRegression,
  kStatusInvalidOperationPacket,
  kStatusOwnerMismatch,
  kStatusOperatorFailed,
  kStatusFrontierCapacityExceeded,
  kStatusTimingControlRejected,
  kStatusQueueInvariant,
  kStatusResultSinkRejected,
};

enum stall_bit : uint8_t {
  kStallNone = 0,
  kStallUnitUnavailable = 1u << 0,
  kStallPipelineFull = 1u << 1,
  kStallResultSinkBackpressure = 1u << 2,
};

struct config_v0 {
  uint8_t stack_unit_count;
  uint8_t stack_latency;
  uint8_t stack_initiation_interval;
  uint8_t stack_issue_width;
  uint8_t reserved_zero[4];
};

struct unit_state_v0 {
  uint64_t next_issue_cycle;
};

struct pipeline_entry_v0 {
  stack_operation::operation_packet_v0 operation_packet;
  typed_stack::push_result_v0 typed_result;
  typed_stack::pop_result_v0 typed_pop_result;
  typed_stack::empty_result_v0 typed_empty_result;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint8_t valid;
  uint8_t unit_index;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[5];
};

struct completed_push_receipt_v0 {
  stack_operation::operation_packet_v0 operation_packet;
  typed_stack::push_result_v0 typed_result;
  typed_stack::pop_result_v0 typed_pop_result;
  typed_stack::empty_result_v0 typed_empty_result;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint64_t capture_cycle;
  uint32_t producer_operation_seq;
  uint32_t commit_epoch;
  uint32_t target_operation_seq;
  uint8_t valid;
  uint8_t operation_kind;
  uint8_t operator_invocation_count;
  uint8_t used_transition_spill;
  uint8_t target_materialized;
  uint8_t terminal_boundary;
  uint8_t reserved_zero[2];
};

enum result_sink_kind : uint8_t {
  kResultSinkAccepted = 0,
  kResultSinkBackpressure = 1,
  kResultSinkRejected = 2,
};

typedef result_sink_kind (*result_sink_accept_fn)(
    completed_push_receipt_v0 *push,
    timing_driver::state_v0 *staged_timing_state, void *context);

struct result_sink_v0 {
  result_sink_accept_fn accept;
  void *context;
};

struct cycle_result_v0 {
  completed_push_receipt_v0 completed_push;
  uint8_t issued_count;
  uint8_t captured_result_count;
  uint8_t stall_mask;
  uint8_t active_pipeline_entries;
  uint8_t ready_stack_entries;
  uint8_t reserved_zero[3];
};

struct state_v0 {
  config_v0 config;
  uint64_t next_issue_age;
  uint64_t last_service_cycle;
  uint64_t total_operator_invocations;
  uint64_t total_results_captured;
  uint8_t initialized;
  uint8_t last_service_cycle_valid;
  uint8_t reserved_zero[6];
  unit_state_v0 units[kMaxStackUnits];
  pipeline_entry_v0 pipeline[kMaxPipelineEntries];
};

config_v0 candidate_profile_config();

status_kind initialize(state_v0 *state, const config_v0 &config);

status_kind service_cycle(
    state_v0 *state, stack_operation::engine_state_v0 *operation_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    const result_sink_v0 *result_sink, cycle_result_v0 *result);

uint8_t active_pipeline_count(const state_v0 &state);

const char *status_name(status_kind status);

}  // namespace stack_timing
}  // namespace v04
}  // namespace rtcore

#endif
