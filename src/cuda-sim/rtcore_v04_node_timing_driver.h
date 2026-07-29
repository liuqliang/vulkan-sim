#ifndef RTCORE_V04_NODE_TIMING_DRIVER_H
#define RTCORE_V04_NODE_TIMING_DRIVER_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_functional_driver.h"
#include "rtcore_v04_timing_driver.h"

namespace rtcore {
namespace v04 {
namespace node_timing {

static const uint8_t kMaxNodeUnits = 8;
static const uint8_t kMaxNodePipelineEntries = 16;
static const uint8_t kMaxResultCommitEntries = 16;
static const uint8_t kMaxCycleRouteReceipts = 16;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusCycleRegression,
  kStatusInvalidOperationPacket,
  kStatusOwnerMismatch,
  kStatusOperatorFailed,
  kStatusResultIdentityMismatch,
  kStatusSemanticApplyFailed,
  kStatusTimingControlRejected,
  kStatusQueueInvariant,
  kStatusRouteSinkRejected,
};

enum stall_bit : uint8_t {
  kStallNone = 0,
  kStallNodeUnitUnavailable = 1u << 0,
  kStallNodePipelineFull = 1u << 1,
  kStallResultCommitFull = 1u << 2,
  kStallResultCommitNotAccepted = 1u << 3,
  kStallRouteSinkBackpressure = 1u << 4,
};

struct config_v0 {
  uint8_t node_unit_count;
  uint8_t node_latency;
  uint8_t node_initiation_interval;
  uint8_t node_issue_width;
  uint8_t result_commit_capacity;
  uint8_t result_commit_width;
  uint8_t reserved_zero[2];
};

struct result_identity_envelope_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint32_t target_operation_seq;
  uint32_t slot_generation;
};

struct node_unit_state_v0 {
  uint64_t next_issue_cycle;
};

struct node_pipeline_entry_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_node::route_result_v0 typed_result;
  result_identity_envelope_v0 result_identity;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint8_t valid;
  uint8_t unit_index;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[5];
};

struct result_commit_entry_v0 {
  result_identity_envelope_v0 result_identity;
  result_semantic::node_commit_plan_v0 semantic_plan;
  typed_node::ray_policy_v0 ray_policy;
  fetch_target::target_reference_v0 current_target_reference;
  typed_blas::as_decode_context_v0 current_decode_context;
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint64_t capture_cycle;
  uint32_t commit_epoch;
  uint32_t current_traversal_bound_bits;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[2];
};

struct committed_route_receipt_v0 {
  result_identity_envelope_v0 result_identity;
  typed_node::route_result_v0 typed_result;
  result_semantic::node_commit_plan_v0 semantic_plan;
  typed_node::ray_policy_v0 ray_policy;
  fetch_target::target_reference_v0 current_target_reference;
  typed_blas::as_decode_context_v0 current_decode_context;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint64_t capture_cycle;
  uint64_t commit_cycle;
  uint32_t commit_epoch;
  uint32_t current_traversal_bound_bits;
  uint32_t next_target_operation_seq;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t next_target_kind;
  uint8_t next_target_materialized;
};

enum route_sink_result_kind : uint8_t {
  kRouteSinkAccepted = 0,
  kRouteSinkBackpressure = 1,
  kRouteSinkRejected = 2,
};

enum materialized_route_target_kind : uint8_t {
  kMaterializedRouteTargetInvalid = 0,
  kMaterializedRouteTargetStackOperation = 4,
};

typedef route_sink_result_kind (*route_sink_accept_fn)(
    committed_route_receipt_v0 *route,
    timing_driver::state_v0 *staged_timing_state, void *context);

struct route_sink_v0 {
  route_sink_accept_fn accept;
  void *context;
};

struct cycle_result_v0 {
  committed_route_receipt_v0
      committed_routes[kMaxCycleRouteReceipts];
  uint8_t issued_count;
  uint8_t captured_result_count;
  uint8_t committed_route_count;
  uint8_t stall_mask;
  uint8_t active_pipeline_entries;
  uint8_t active_result_entries;
  uint8_t ready_node_entries;
  uint8_t reserved_zero;
};

struct state_v0 {
  config_v0 config;
  uint64_t next_issue_age;
  uint64_t last_service_cycle;
  uint64_t total_operator_invocations;
  uint64_t total_routes_committed;
  uint8_t initialized;
  uint8_t last_service_cycle_valid;
  uint8_t reserved_zero[6];
  node_unit_state_v0 node_units[kMaxNodeUnits];
  node_pipeline_entry_v0
      node_pipeline[kMaxNodePipelineEntries];
  result_commit_entry_v0
      result_commits[kMaxResultCommitEntries];
};

static_assert(sizeof(result_commit_entry_v0) <= 432,
              "Node Result Commit entry exceeds candidate 432-byte profile");

config_v0 candidate_profile_config();

status_kind initialize(state_v0 *state, const config_v0 &config);

bool validate_result_identity(
    const result_identity_envelope_v0 &identity,
    const fetch_target::operation_packet_v0 &packet);

status_kind service_cycle(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    bool result_commit_accepts, cycle_result_v0 *result);

status_kind service_cycle(
    state_v0 *state, fetch_target::engine_state_v0 *target_state,
    timing_driver::state_v0 *timing_state, uint64_t service_cycle,
    bool result_commit_accepts, const route_sink_v0 *route_sink,
    cycle_result_v0 *result);

uint8_t active_pipeline_count(const state_v0 &state);
uint8_t active_result_count(const state_v0 &state);

const char *status_name(status_kind status);

}  // namespace node_timing
}  // namespace v04
}  // namespace rtcore

#endif
