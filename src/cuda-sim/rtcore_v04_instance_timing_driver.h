#ifndef RTCORE_V04_INSTANCE_TIMING_DRIVER_H
#define RTCORE_V04_INSTANCE_TIMING_DRIVER_H

#include <cstdint>

#include "rtcore_v04_fetch_target_queue.h"
#include "rtcore_v04_timing_driver.h"
#include "rtcore_v04_typed_instance_kernel.h"

namespace rtcore {
namespace v04 {
namespace instance_timing {

static const uint8_t kMaxInstanceUnits = 2;
static const uint8_t kMaxPipelineEntries = 8;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusCycleRegression,
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
  kStallPipelineFull = 1u << 1,
  kStallResultSinkBackpressure = 1u << 2,
};

struct config_v0 {
  uint8_t instance_unit_count;
  uint8_t instance_latency;
  uint8_t instance_initiation_interval;
  uint8_t instance_issue_width;
  uint8_t reserved_zero[4];
};

struct unit_state_v0 {
  uint64_t next_issue_cycle;
};

struct pipeline_entry_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_instance::enter_input_v0 enter_input;
  typed_instance::enter_result_v0 enter_result;
  typed_instance::restore_parent_result_v0 typed_result;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint8_t valid;
  uint8_t unit_index;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[5];
};

struct completed_enter_receipt_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_instance::enter_input_v0 typed_input;
  typed_instance::enter_result_v0 typed_result;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint64_t capture_cycle;
  uint32_t producer_operation_seq;
  uint32_t commit_epoch;
  uint32_t target_operation_seq;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[6];
};

struct completed_restore_receipt_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_instance::restore_parent_result_v0 typed_result;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint64_t capture_cycle;
  uint32_t producer_operation_seq;
  uint32_t commit_epoch;
  uint32_t target_operation_seq;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[6];
};

enum result_sink_kind : uint8_t {
  kResultSinkAccepted = 0,
  kResultSinkBackpressure = 1,
  kResultSinkRejected = 2,
};

typedef result_sink_kind (*result_sink_accept_fn)(
    completed_restore_receipt_v0 *restore,
    timing_driver::state_v0 *staged_timing_state, void *context);

typedef result_sink_kind (*enter_input_prepare_fn)(
    const fetch_target::operation_packet_v0 *operation_packet,
    typed_instance::enter_input_v0 *input, void *context);

typedef result_sink_kind (*enter_result_sink_accept_fn)(
    completed_enter_receipt_v0 *enter,
    timing_driver::state_v0 *staged_timing_state, void *context);

struct result_sink_v0 {
  result_sink_accept_fn accept;
  enter_input_prepare_fn prepare_enter;
  enter_result_sink_accept_fn accept_enter;
  void *context;
};

struct cycle_result_v0 {
  completed_restore_receipt_v0 completed_restores[kMaxInstanceUnits];
  completed_enter_receipt_v0 completed_enters[kMaxInstanceUnits];
  uint8_t issued_count;
  uint8_t captured_result_count;
  uint8_t captured_restore_count;
  uint8_t captured_enter_count;
  uint8_t stall_mask;
  uint8_t active_pipeline_entries;
  uint8_t ready_instance_entries;
  uint8_t reserved_zero[1];
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
  unit_state_v0 units[kMaxInstanceUnits];
  pipeline_entry_v0 pipeline[kMaxPipelineEntries];
};

config_v0 candidate_profile_config();

status_kind initialize(state_v0 *state, const config_v0 &config);

status_kind service_cycle(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    const result_sink_v0 *result_sink, cycle_result_v0 *result);

uint8_t active_pipeline_count(const state_v0 &state);

const char *status_name(status_kind status);

}  // namespace instance_timing
}  // namespace v04
}  // namespace rtcore

#endif
