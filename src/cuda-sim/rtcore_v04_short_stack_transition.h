#ifndef RTCORE_V04_SHORT_STACK_TRANSITION_H
#define RTCORE_V04_SHORT_STACK_TRANSITION_H

#include <cstdint>

#include "rtcore_v04_fetch_target_queue.h"
#include "rtcore_v04_short_stack_shared_codec.h"

namespace rtcore {
namespace v04 {
namespace short_stack_transition {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusPrivateStateRejected,
  kStatusShortStackRejected,
  kStatusGenerationMismatch,
  kStatusDecodeContextMismatch,
  kStatusParentLookupRequired,
  kStatusReturnInstanceRequired,
};

struct node_input_v0 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 region;
  private_frontier::shadow_slot_v0 canonical_slot;
  typed_node::route_result_v0 node_route;
  fetch_target::target_reference_v0 current_target;
  typed_blas::as_decode_context_v0 current_decode_context;
  short_stack::entry_v0 pending_parent_resume;
  short_stack::parent_edge_v0 parent_edge;
  uint8_t pending_parent_resume_valid;
  uint8_t parent_edge_valid;
  uint8_t reserved_zero[6];
};

// Profile-selected Stack operands. Unlike node_input_v0 this input never
// carries a full private-slot image.
struct node_operands_input_v1 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 region;
  short_stack_shared::persistent_state_v0 persistent_state;
  typed_node::route_result_v0 node_route;
  fetch_target::target_reference_v0 current_target;
  typed_blas::as_decode_context_v0 current_decode_context;
  short_stack::entry_v0 pending_parent_resume;
  short_stack::parent_edge_v0 parent_edge;
  uint8_t pending_parent_resume_valid;
  uint8_t parent_edge_valid;
  uint8_t reserved_zero[6];
};

struct resume_input_v0 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 region;
  private_frontier::shadow_slot_v0 canonical_slot;
  private_frontier::root_private_operands_v0 immutable_trace_input;
  typed_blas::as_decode_context_v0 active_decode_context;
  short_stack::parent_edge_v0 parent_edge;
  uint8_t parent_edge_valid;
  uint8_t reserved_zero[7];
};

struct resume_operands_input_v1 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 region;
  short_stack_shared::persistent_state_v0 persistent_state;
  private_frontier::mutable_ray_state_v0 parent_ray;
  typed_blas::as_decode_context_v0 parent_decode_context;
  private_frontier::instance_shader_projection_v0 parent_instance;
  typed_blas::as_decode_context_v0 active_decode_context;
  short_stack::parent_edge_v0 parent_edge;
  uint8_t parent_restore_valid;
  uint8_t parent_edge_valid;
  uint8_t recovery_target_completed;
  uint8_t reserved_zero[5];
};

struct enter_blas_input_v0 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 region;
  private_frontier::shadow_slot_v0 canonical_slot;
  fetch_target::target_reference_v0 tlas_instance_target;
  typed_node::selected_child_fetch_work_item_v0 blas_root;
  uint32_t blas_build_generation;
  uint8_t reserved_zero[4];
};

struct enter_blas_operands_input_v1 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 region;
  short_stack_shared::persistent_state_v0 persistent_state;
  fetch_target::target_reference_v0 tlas_instance_target;
  typed_node::selected_child_fetch_work_item_v0 blas_root;
  uint32_t blas_build_generation;
  uint8_t reserved_zero[4];
};

struct result_v0 {
  private_frontier::shadow_slot_v0 updated_slot;
  private_frontier::access_plan_v0 write_plan;
  short_stack_shared::persistent_state_v0 persistent_state;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  short_stack::entry_v0 selected_entry;
  typed_node::replay_cursor_v0 replay_cursor;
  short_stack::entry_v0 pending_parent_resume;
  typed_blas::as_decode_context_v0 parent_lookup_decode_context;
  uint64_t parent_lookup_payload_offset;
  uint32_t next_build_generation;
  uint32_t parent_lookup_build_generation;
  uint8_t selected_valid;
  uint8_t terminal;
  uint8_t restore_parent_required;
  uint8_t parent_lookup_required;
  uint8_t pending_parent_resume_valid;
  uint8_t compressed_to_replay;
  uint8_t overflowed_bottom;
  uint8_t parent_lookup_payload_kind;
};

status_kind prepare_node_transition(const node_input_v0 &input,
                                    result_v0 *result);
status_kind prepare_node_transition_from_operands(
    const node_operands_input_v1 &input, result_v0 *result);
status_kind prepare_resume_transition(const resume_input_v0 &input,
                                      result_v0 *result);
status_kind prepare_resume_transition_from_operands(
    const resume_operands_input_v1 &input, result_v0 *result);
status_kind prepare_enter_blas_transition(
    const enter_blas_input_v0 &input, result_v0 *result);
status_kind prepare_enter_blas_transition_from_operands(
    const enter_blas_operands_input_v1 &input, result_v0 *result);

const char *status_name(status_kind status);

}  // namespace short_stack_transition
}  // namespace v04
}  // namespace rtcore

#endif
