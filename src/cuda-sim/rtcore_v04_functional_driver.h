#ifndef RTCORE_V04_FUNCTIONAL_DRIVER_H
#define RTCORE_V04_FUNCTIONAL_DRIVER_H

#include <cstdint>

#include "rtcore_v04_instance_result_semantic_applier.h"
#include "rtcore_v04_result_semantic_applier.h"
#include "rtcore_v04_stack_operation_queue.h"
#include "rtcore_v04_stack_result_semantic_applier.h"

namespace rtcore {
namespace v04 {
namespace functional_driver {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOperationPacket,
  kStatusCanonicalInputMismatch,
  kStatusTypedOperatorFailed,
  kStatusSemanticApplyFailed,
};

enum semantic_plan_kind : uint8_t {
  kSemanticPlanInvalid = 0,
  kSemanticPlanNode = 1,
  kSemanticPlanStackAppend = 2,
  kSemanticPlanStackPop = 3,
  kSemanticPlanStackRestoreParent = 4,
  kSemanticPlanStackTerminalHit = 5,
  kSemanticPlanStackTerminalMiss = 6,
  kSemanticPlanInstanceEnter = 7,
  kSemanticPlanInstanceRestore = 8,
};

struct node_execution_v0 {
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[6];
  typed_node::route_input_v0 operator_input;
  typed_node::route_result_v0 operator_result;
  result_semantic::node_commit_plan_v0 semantic_plan;
};

struct stack_execution_v0 {
  stack_operation::operation_packet_v0 operation_packet;
  typed_stack::push_result_v0 push_result;
  typed_stack::pop_result_v0 pop_result;
  typed_stack::empty_result_v0 empty_result;
  stack_semantic::append_commit_plan_v0 append_plan;
  stack_semantic::pop_commit_plan_v0 pop_plan;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t semantic_plan_kind;
  uint8_t terminal_boundary;
  uint8_t reserved_zero[4];
};

struct instance_enter_execution_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_instance::enter_input_v0 operator_input;
  typed_instance::enter_result_v0 operator_result;
  instance_semantic::enter_commit_plan_v0 semantic_plan;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t semantic_plan_kind;
  uint8_t reserved_zero[5];
};

struct instance_restore_execution_v0 {
  fetch_target::operation_packet_v0 operation_packet;
  typed_instance::restore_parent_input_v0 operator_input;
  typed_instance::restore_parent_result_v0 operator_result;
  instance_semantic::restore_commit_plan_v0 semantic_plan;
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t semantic_plan_kind;
  uint8_t reserved_zero[5];
};

bool mode_selection_valid(bool functional_only_enabled,
                          bool timing_driver_enabled,
                          bool root_packet_enabled,
                          bool live_node_timing_enabled = false);

status_kind prepare_node_operator_input(
    const fetch_target::operation_packet_v0 &packet,
    typed_node::route_input_v0 *input);

status_kind execute_node_operator_once(
    const fetch_target::operation_packet_v0 &packet,
    typed_node::route_result_v0 *result,
    uint8_t *operator_invocation_count);

status_kind execute_one_node(
    const fetch_target::operation_packet_v0 &packet,
    node_execution_v0 *execution);

status_kind execute_one_stack(
    const stack_operation::operation_packet_v0 &packet,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    stack_execution_v0 *execution);

status_kind execute_one_instance_enter(
    const fetch_target::operation_packet_v0 &packet,
    const typed_instance::enter_input_v0 &input,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    instance_enter_execution_v0 *execution);

status_kind execute_one_instance_restore(
    const fetch_target::operation_packet_v0 &packet,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    instance_restore_execution_v0 *execution);

const char *status_name(status_kind status);

}  // namespace functional_driver
}  // namespace v04
}  // namespace rtcore

#endif
