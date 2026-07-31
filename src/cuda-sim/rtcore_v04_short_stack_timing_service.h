#ifndef RTCORE_V04_SHORT_STACK_TIMING_SERVICE_H
#define RTCORE_V04_SHORT_STACK_TIMING_SERVICE_H

#include <cstdint>

#include "rtcore_replay_interface.h"
#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_private_state_384_live_bridge.h"
#include "rtcore_v04_short_stack_transition.h"
#include "rtcore_v04_timing_driver.h"
#include "rtcore_v04_typed_instance_kernel.h"

namespace rtcore {
namespace v04 {
namespace short_stack_timing {

static const uint8_t kMaxSlots = 16;
static const uint8_t kMaxUnits = 1;
static const uint8_t kReadChunkCount =
    short_stack_shared::kStateAccessChunkCount;
static const uint8_t kMaxReadChunkCount =
    private_state_384::operand_materializer::kMaxOperationReadChunks;
static const uint8_t kWriteChunkCount =
    short_stack_shared::kStateAccessChunkCount;
static const uint8_t kMaxWriteChunkCount = 12;
static const uint8_t kReturnInstanceReadChunkCount = 4;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfig,
  kStatusCapacityBackpressure,
  kStatusReservationBudgetBackpressure,
  kStatusOwnerMismatch,
  kStatusTimingControlRejected,
  kStatusSharedPlanRejected,
  kStatusMalformedTransport,
  kStatusDuplicateResponse,
  kStatusReturnInstanceRejected,
  kStatusTransitionRejected,
  kStatusParentResolveRejected,
  kStatusSharedQueueBackpressure,
  kStatusSharedWriteRejected,
  kStatusNoAckOwned,
  kStatusAckRejected,
  kStatusNoFollowupRead,
  kStatusNoReadyResult,
};

enum phase_kind : uint8_t {
  kPhaseInvalid = 0,
  kPhaseReading,
  kPhasePrivate384FollowupPlanReady,
  kPhasePrivate384CompletionPlanReady,
  kPhaseReturnInstancePlanReady,
  kPhaseReturnInstanceReading,
  kPhaseReadyToIssue,
  kPhaseExecuting,
  kPhaseParentLookup,
  kPhaseWriting,
  kPhaseResultReady,
};

enum operation_kind : uint8_t {
  kOperationInvalid = 0,
  kOperationNodeTransition = 1,
  kOperationResumeTransition = 2,
  kOperationEnterBlasTransition = 3,
};

struct config_v0 {
  uint8_t capacity;
  uint8_t reservation_width;
  uint8_t unit_count;
  uint8_t stack_latency;
  uint8_t initiation_interval;
  uint8_t issue_width;
  uint8_t parent_lookup_latency;
  uint8_t reserved_zero;
};

struct reservation_input_v0 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 private_region;
  typed_node::route_result_v0 node_route;
  typed_node::ray_policy_v0 ray_policy;
  private_frontier::root_private_operands_v0 immutable_trace_input;
  fetch_target::target_reference_v0 current_target;
  typed_blas::as_decode_context_v0 current_decode_context;
  typed_blas::as_decode_context_v0 active_decode_context;
  fetch_target::target_reference_v0 tlas_instance_target;
  typed_node::selected_child_fetch_work_item_v0 blas_root;
  typed_instance::mutable_ray_state_v0 instance_object_ray;
  typed_instance::instance_shader_projection_v0
      instance_projection;
  short_stack::entry_v0 pending_parent_resume;
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint32_t target_operation_seq;
  uint32_t blas_build_generation;
  uint8_t operation_kind;
  uint8_t pending_parent_resume_valid;
  uint8_t private_storage_profile;
  uint8_t recovery_target_inflight;
  uint8_t deferred_instance_valid;
};

struct reservation_receipt_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint32_t operation_seq;
  uint32_t producer_operation_seq;
  uint32_t slot_generation;
  uint8_t slot_index;
  uint8_t read_chunk_count;
  uint8_t valid;
  uint8_t reserved_zero;
};

struct request_plan_v0 {
  rtcore_memory_unit_request_snapshot requests[kMaxReadChunkCount];
  uint8_t request_count;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

struct operation_entry_v0 {
  reservation_input_v0 input;
  reservation_receipt_v0 reservation;
  private_frontier::shadow_slot_v0 read_slot;
  private_frontier::access_plan_v0 read_plan;
  private_state_384::operand_materializer::response_collector_v1
      private_state_384_collector;
  private_state_384::operand_materializer::stack_operands_v1
      private_state_384_stack_operands;
  private_state_384::operand_materializer::stack_terminal_operands_v1
      private_state_384_terminal_operands;
  private_state_384::operand_materializer::stack_cross_as_operands_v1
      private_state_384_cross_as_operands;
  private_state_384::operand_materializer::final_completion_operands_v1
      private_state_384_completion_operands;
  private_state_384::live_bridge::pending_sparse_commit_v1
      private_state_384_commit;
  short_stack_transition::result_v0 transition;
  short_stack::parent_edge_v0 parent_edge;
  private_frontier::instance_shader_projection_v0
      return_instance_projection;
  uint8_t return_instance_payload[
      fetch_target::kInstanceRawPayloadBytes];
  uint64_t issue_age;
  uint64_t issue_cycle;
  uint64_t result_ready_cycle;
  uint64_t parent_lookup_ready_cycle;
  uint32_t commit_epoch;
  uint16_t received_read_mask;
  uint8_t received_return_instance_mask;
  uint16_t enqueued_write_mask;
  uint16_t acknowledged_write_mask;
  uint8_t phase;
  uint8_t parent_edge_valid;
  uint8_t private_state_384_operands_valid;
  uint8_t private_state_384_selected_operation;
  uint8_t private_state_384_terminal_operands_valid;
  uint8_t private_state_384_cross_as_operands_valid;
  uint8_t private_state_384_completion_operands_valid;
  uint8_t private_state_384_completion_reason;
  uint8_t return_instance_projection_valid;
  uint8_t valid;
};

struct unit_state_v0 {
  uint64_t next_issue_cycle;
};

struct engine_state_v0 {
  uint8_t initialized;
  uint8_t reserved_zero[7];
  config_v0 config;
  uint64_t next_reservation_id;
  uint64_t next_age;
  uint64_t reservation_cycle;
  uint8_t reservations_this_cycle;
  uint8_t reserved_zero1[7];
  unit_state_v0 units[kMaxUnits];
  operation_entry_v0 slots[kMaxSlots];
};

struct cycle_result_v0 {
  uint8_t issued;
  uint8_t transition_committed;
  uint8_t parent_lookup_completed;
  uint8_t return_instance_requested;
  uint8_t active_operations;
  uint8_t ready_results;
  uint8_t rejected_operation_kind;
  uint8_t rejected_transition_status;
};

struct ready_result_v0 {
  private_frontier::owner_binding_v0 owner;
  typed_node::ray_policy_v0 ray_policy;
  short_stack_transition::result_v0 transition;
  uint64_t reservation_id;
  uint32_t operation_seq;
  uint32_t producer_operation_seq;
  uint32_t commit_epoch;
  uint32_t slot_generation;
  uint8_t slot_index;
  uint8_t private_storage_profile;
  uint8_t terminal_committed_hit_valid;
  uint8_t valid;
  private_frontier::committed_hit_projection_v0
      terminal_committed_hit;
};

typedef bool (*parent_resolver_fn)(
    void *context,
    const typed_blas::as_decode_context_v0 &decode_context,
    uint32_t build_generation, uint64_t payload_offset,
    uint8_t payload_kind,
    short_stack::parent_edge_v0 *parent);

struct parent_resolver_v0 {
  parent_resolver_fn resolve;
  void *context;
};

config_v0 candidate_profile_config();
status_kind initialize(engine_state_v0 *state, const config_v0 &config);

status_kind reserve(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    reservation_receipt_v0 *reservation, request_plan_v0 *requests);

status_kind reserve_existing_target(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    reservation_receipt_v0 *reservation, request_plan_v0 *requests);

status_kind reserve_private_state_384(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_state_384::backing::state_v1 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    reservation_receipt_v0 *reservation, request_plan_v0 *requests);

status_kind accept_read_response(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const rtcore_memory_unit_request_snapshot &request,
    uint64_t response_cycle,
    const uint8_t *return_instance_payload = NULL,
    uint8_t return_instance_payload_bytes = 0);

status_kind accept_private_state_384_read_response(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_state_384::backing::state_v1 &private_backing,
    const rtcore_memory_unit_request_snapshot &request,
    uint64_t response_cycle);

status_kind take_private_state_384_followup_read_plan(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t issue_cycle, request_plan_v0 *requests);

status_kind take_return_instance_read_plan(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t issue_cycle, request_plan_v0 *requests);

status_kind service_cycle(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    const parent_resolver_v0 &parent_resolver, uint64_t service_cycle,
    cycle_result_v0 *result);

status_kind service_write_enqueue(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle, uint8_t write_enqueue_budget,
    uint8_t *writes_enqueued, bool *shared_queue_blocked);

bool owns_ack(const engine_state_v0 &state,
              const private_shared::runtime_write_ack_v0 &ack);

status_kind accept_write_ack(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle,
    const private_shared::runtime_write_ack_v0 &ack);

status_kind accept_write_ack_with_private_state_384(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    private_state_384::backing::state_v1 *private_state_384_backing,
    uint64_t service_cycle,
    const private_shared::runtime_write_ack_v0 &ack);

status_kind peek_ready_result(const engine_state_v0 &state,
                              ready_result_v0 *result);
status_kind consume_ready_result(engine_state_v0 *state,
                                 const ready_result_v0 &result);

bool find_live_reservation(
    const engine_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, reservation_receipt_v0 *reservation,
    uint8_t *operation_kind);

uint8_t active_operation_count(const engine_state_v0 &state);
const char *status_name(status_kind status);

}  // namespace short_stack_timing
}  // namespace v04
}  // namespace rtcore

#endif
