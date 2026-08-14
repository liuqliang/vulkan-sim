#ifndef RTCORE_V04_PRIMITIVE_TIMING_DRIVER_H
#define RTCORE_V04_PRIMITIVE_TIMING_DRIVER_H

#include <cstdint>

#include "rtcore_v04_functional_driver.h"
#include "rtcore_v04_timing_driver.h"

namespace rtcore {
namespace v04 {
namespace primitive_timing {

static const uint8_t kMaxPrimitiveUnits = 4;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusCycleRegression,
  kStatusCounterExhausted,
  kStatusInvalidOperationPacket,
  kStatusOwnerMismatch,
  kStatusOperatorFailed,
  kStatusTimingControlRejected,
  kStatusQueueInvariant,
  kStatusResultSinkRejected,
};

enum stall_bit : uint8_t {
  kStallNone = 0,
  kStallUnitUnavailable = 1u << 0,
  kStallResultSinkBackpressure = 1u << 1,
};

enum unit_phase_kind : uint8_t {
  kUnitIdle = 0,
  kUnitExecutingLeaf = 1,
  kUnitOutputPending = 2,
};

struct config_v0 {
  uint8_t primitive_unit_count;
  uint8_t primitive_first_batch_latency;
  uint8_t primitive_batch_width;
  uint8_t primitive_batch_interval;
  uint8_t primitive_issue_width;
  uint8_t reserved_zero[3];
};

struct unit_state_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_primitive::route_input_v0 typed_input;
  typed_primitive::route_result_v0 typed_result;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint8_t phase;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[6];
};

struct completed_receipt_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_primitive::route_input_v0 typed_input;
  typed_primitive::route_result_v0 typed_result;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint64_t capture_cycle;
  uint32_t producer_operation_seq;
  uint32_t commit_epoch;
  uint32_t target_operation_seq;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t unit_index;
  uint8_t reserved_zero[5];
};

enum result_sink_kind : uint8_t {
  kResultSinkAccepted = 0,
  kResultSinkBackpressure = 1,
  kResultSinkRejected = 2,
};

typedef result_sink_kind (*result_sink_accept_fn)(
    completed_receipt_v0 *receipt,
    timing_driver::state_v0 *staged_timing_state, void *context);

struct result_sink_v0 {
  result_sink_accept_fn accept;
  void *context;
};

struct cycle_result_v0 {
  completed_receipt_v0 completed[kMaxPrimitiveUnits];
  uint8_t issued_count;
  uint8_t captured_result_count;
  uint8_t stall_mask;
  uint8_t active_unit_count;
  uint8_t executing_unit_count;
  uint8_t output_pending_count;
  uint8_t ready_primitive_entries;
  uint8_t typed_operator_status;
};

struct state_v0 {
  config_v0 config;
  uint64_t next_issue_age;
  uint64_t last_service_cycle;
  uint64_t total_operator_invocations;
  uint64_t total_results_captured;
  uint64_t total_service_cycles;
  uint64_t total_issued;
  uint64_t total_stall_unit_unavailable;
  uint64_t total_stall_result_sink_backpressure;
  uint64_t total_issue_width_limited;
  uint8_t max_active_unit_count;
  uint8_t max_executing_unit_count;
  uint8_t max_output_pending_count;
  uint8_t max_ready_primitive_entries;
  uint8_t initialized;
  uint8_t last_service_cycle_valid;
  uint8_t reserved_zero[2];
  unit_state_v0 units[kMaxPrimitiveUnits];
};

config_v0 candidate_profile_config();

status_kind initialize(state_v0 *state, const config_v0 &config);

status_kind service_cycle(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    const result_sink_v0 *result_sink, cycle_result_v0 *result);

uint8_t active_unit_count(const state_v0 &state);
uint8_t executing_unit_count(const state_v0 &state);
uint8_t output_pending_count(const state_v0 &state);

const char *status_name(status_kind status);

}  // namespace primitive_timing
}  // namespace v04
}  // namespace rtcore

#endif
