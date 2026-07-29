#include "rtcore_v04_functional_engine.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "rtcore_v04_request_owner_binding.h"

namespace rtcore {
namespace v04 {
namespace functional_engine {
namespace {

enum next_action_kind : uint8_t {
  kActionInvalid = 0,
  kActionTarget,
  kActionStackPush,
  kActionStackPop,
};

struct next_action_v0 {
  uint8_t kind;
  uint8_t instance_blas_root;
  uint8_t short_entry_valid;
  uint8_t parent_resume_valid;
  uint8_t reserved_zero[4];
  uint32_t producer_operation_seq;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  typed_node::route_result_v0 node_route;
  typed_node::replay_cursor_v0 replay_cursor;
  short_stack::entry_v0 short_entry;
  short_stack::entry_v0 parent_resume;
};

uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool provider_valid(const provider_v0 &provider) {
  return provider.read_raw_payload != NULL &&
         provider.prepare_instance_enter != NULL;
}

status_kind allocate_operation(state_v0 *state, uint32_t *operation_seq) {
  if (state == NULL || operation_seq == NULL || !state->valid) {
    return kStatusInvalidState;
  }
  if (state->next_operation_seq == 0 ||
      state->next_operation_seq ==
          std::numeric_limits<uint32_t>::max()) {
    return kStatusOperationSequenceExhausted;
  }
  *operation_seq = state->next_operation_seq++;
  ++state->operation_count;
  return state->operation_count <= kMaxOperationsPerRun
             ? kStatusOk
             : kStatusOperationWatchdog;
}

status_kind record_driver_rejection(
    state_v0 *state, driver_unit_kind unit,
    functional_driver::status_kind status) {
  if (state != NULL) {
    state->last_driver_unit = unit;
    state->last_driver_status = status;
  }
  return kStatusFunctionalDriverRejected;
}

status_kind decode_private(
    const state_v0 &state,
    private_frontier::root_private_operands_v0 *operands,
    private_frontier::instance_shader_projection_v0 *current_instance) {
  if (operands == NULL || current_instance == NULL ||
      private_frontier::decode_root_private_operands(
          state.canonical_slot, state.owner, operands) !=
          private_frontier::kStatusOk ||
      private_frontier::decode_current_instance(
          state.canonical_slot, state.owner, current_instance) !=
          private_frontier::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  return kStatusOk;
}

short_stack::entry_v0 short_entry_from_fetch(
    const typed_node::selected_child_fetch_work_item_v0 &fetch,
    short_stack::domain_kind domain) {
  short_stack::entry_v0 entry = {};
  entry.payload_offset = fetch.child.payload_offset;
  entry.near_t_bits = fetch.child.near_t_bits;
  entry.payload_byte_count = fetch.child.payload_byte_count;
  entry.payload_kind = fetch.child.payload_kind;
  entry.control = short_stack::make_control(
      short_stack::kEntryDirectTarget, domain,
      fetch.child.child_slot, true, false);
  return entry;
}

bool selected_fetch_from_short_entry(
    const short_stack::entry_v0 &entry,
    const typed_blas::as_decode_context_v0 &decode_context,
    typed_node::selected_child_fetch_work_item_v0 *fetch) {
  if (fetch == NULL || !short_stack::validate_entry(entry) ||
      short_stack::control_kind(entry.control) ==
          short_stack::kEntryCrossAsReturn) {
    return false;
  }
  *fetch = typed_node::selected_child_fetch_work_item_v0();
  fetch->child.payload_offset = entry.payload_offset;
  fetch->child.near_t_bits = entry.near_t_bits;
  fetch->child.payload_byte_count = entry.payload_byte_count;
  fetch->child.payload_kind = entry.payload_kind;
  fetch->child.child_slot =
      short_stack::control_child_anchor(entry.control);
  fetch->decode_context = decode_context;
  return true;
}

typed_node::replay_cursor_v0 replay_cursor_from_entry(
    const short_stack::entry_v0 &entry) {
  typed_node::replay_cursor_v0 cursor = {};
  const short_stack::entry_kind kind =
      short_stack::control_kind(entry.control);
  if (kind == short_stack::kEntrySameNodeReplay ||
      kind == short_stack::kEntryParentResume) {
    cursor.child_anchor =
        short_stack::control_child_anchor(entry.control);
    cursor.anchor_valid =
        short_stack::control_anchor_valid(entry.control) ? 1 : 0;
    cursor.inclusive =
        short_stack::control_inclusive(entry.control) ? 1 : 0;
  }
  return cursor;
}

uint32_t active_build_generation(const state_v0 &state) {
  return state.short_stack.active_domain ==
                 short_stack::kDomainBlas
             ? state.blas_build_generation
             : state.tlas_build_generation;
}

status_kind build_target_packet(
    state_v0 *state, const provider_v0 &provider,
    const typed_node::selected_child_fetch_work_item_v0 &selected_fetch,
    bool instance_blas_root, uint32_t producer_operation_seq,
    fetch_target::operation_packet_v0 *packet) {
  if (state == NULL || packet == NULL) return kStatusInvalidArgument;
  uint32_t operation_seq = 0;
  status_kind status = allocate_operation(state, &operation_seq);
  if (status != kStatusOk) return status;

  fetch_target::reservation_input_v0 lowered = {};
  if (instance_blas_root) {
    fetch_target::instance_blas_root_reservation_input_v0 input = {};
    input.owner = state->owner;
    input.root_fetch = selected_fetch;
    input.forwarded_ray_policy = state->ray_policy;
    input.target_operation_seq = operation_seq;
    input.producer_operation_seq = producer_operation_seq;
    input.producer_commit_epoch = producer_operation_seq;
    if (fetch_target::lower_instance_blas_root(input, &lowered) !=
        fetch_target::kStatusOk) {
      return kStatusTargetRejected;
    }
  } else {
    fetch_target::selected_fetch_reservation_input_v0 input = {};
    input.owner = state->owner;
    input.selected_fetch = selected_fetch;
    input.forwarded_ray_policy = state->ray_policy;
    input.target_operation_seq = operation_seq;
    input.required_operand_mask = static_cast<uint8_t>(
        fetch_target::kOperandTargetReferenceValid |
        fetch_target::kOperandRawPayloadValid |
        fetch_target::kOperandMutableRayValid |
        fetch_target::kOperandRayPolicyValid |
        fetch_target::kOperandDecodeContextValid |
        fetch_target::kOperandCommittedHitValid);
    input.forwarded_operand_mask =
        fetch_target::kOperandRayPolicyValid;
    if (fetch_target::lower_selected_fetch(input, &lowered) !=
        fetch_target::kStatusOk) {
      return kStatusTargetRejected;
    }
  }

  private_frontier::root_private_operands_v0 operands = {};
  private_frontier::instance_shader_projection_v0 current_instance = {};
  status = decode_private(*state, &operands, &current_instance);
  if (status != kStatusOk) return status;

  uint8_t raw_payload[fetch_target::kMaxRawPayloadBytes] = {};
  if (!provider.read_raw_payload(
          provider.context, selected_fetch.decode_context,
          lowered.raw_payload_base_address, lowered.raw_payload_bytes,
          raw_payload)) {
    return kStatusPayloadReadRejected;
  }
  if (fetch_target::build_ready_operation_packet(
          lowered, operation_seq, operation_seq, 1, operands,
          current_instance, raw_payload, packet) !=
      fetch_target::kStatusOk) {
    return kStatusTargetRejected;
  }
  return kStatusOk;
}

status_kind build_root_packet(
    state_v0 *state, const root_input_v0 &input,
    const provider_v0 &provider,
    fetch_target::operation_packet_v0 *packet) {
  uint32_t operation_seq = 0;
  status_kind status = allocate_operation(state, &operation_seq);
  if (status != kStatusOk) return status;

  fetch_target::reservation_input_v0 lowered = {};
  lowered.owner = input.owner;
  lowered.target_reference = input.root_reference;
  lowered.forwarded_ray_policy = input.ray_policy;
  lowered.raw_payload_base_address = input.raw_payload_base_address;
  lowered.target_operation_seq = operation_seq;
  lowered.raw_payload_bytes = fetch_target::kNodeRawPayloadBytes;
  lowered.target_kind = fetch_target::kTargetNode;
  lowered.required_operand_mask = static_cast<uint8_t>(
      fetch_target::kOperandTargetReferenceValid |
      fetch_target::kOperandRawPayloadValid |
      fetch_target::kOperandMutableRayValid |
      fetch_target::kOperandRayPolicyValid |
      fetch_target::kOperandDecodeContextValid |
      fetch_target::kOperandCommittedHitValid);
  lowered.forwarded_operand_mask =
      fetch_target::kOperandRayPolicyValid;

  uint8_t raw_payload[fetch_target::kNodeRawPayloadBytes] = {};
  if (!provider.read_raw_payload(
          provider.context, input.private_operands.decode_context,
          input.raw_payload_base_address, sizeof(raw_payload),
          raw_payload)) {
    return kStatusPayloadReadRejected;
  }
  if (fetch_target::build_ready_node_operation_packet(
          lowered, operation_seq, operation_seq, 1,
          input.private_operands, raw_payload, packet) !=
      fetch_target::kStatusOk) {
    return kStatusTargetRejected;
  }
  return kStatusOk;
}

status_kind build_stack_packet(
    state_v0 *state, const next_action_v0 &action,
    stack_operation::operation_packet_v0 *packet) {
  if (state == NULL || packet == NULL) return kStatusInvalidArgument;
  uint32_t operation_seq = 0;
  status_kind status = allocate_operation(state, &operation_seq);
  if (status != kStatusOk) return status;
  if (action.producer_operation_seq == 0 ||
      action.producer_operation_seq == operation_seq) {
    return kStatusInvalidState;
  }

  private_frontier::frontier_metadata_image_v0 metadata = {};
  private_frontier::root_private_operands_v0 operands = {};
  if (private_frontier::decode_metadata(
          state->canonical_slot, state->owner, &metadata) !=
          private_frontier::kStatusOk ||
      private_frontier::decode_root_private_operands(
          state->canonical_slot, state->owner, &operands) !=
          private_frontier::kStatusOk) {
    return kStatusPrivateStateRejected;
  }

  *packet = stack_operation::operation_packet_v0();
  packet->owner = state->owner;
  packet->reservation_id = operation_seq;
  packet->reservation_age = operation_seq;
  packet->target_operation_seq = operation_seq;
  packet->producer_operation_seq = action.producer_operation_seq;
  packet->slot_generation = 1;
  packet->valid = 1;
  packet->frontier_metadata = metadata;
  packet->ray_policy = state->ray_policy;
  if (action.kind == kActionStackPush) {
    packet->operation_kind =
        typed_stack::kPushRemainderAndForwardSelected;
    packet->input.profile_id = typed_stack::kGenRtDerivedProfileId;
    packet->input.operation_kind =
        typed_stack::kPushRemainderAndForwardSelected;
    packet->input.frontier.frontier_top = metadata.frontier_top;
    packet->input.frontier.frontier_count = metadata.frontier_count;
    packet->input.frontier.frontier_capacity =
        metadata.frontier_capacity;
    packet->input.current_traversal_bound_bits = fp32_bits(
        operands.committed_hit.valid != 0
            ? operands.committed_hit.hit_t
            : operands.mutable_ray.t_max);
    packet->input.node_route = action.node_route;
    return kStatusOk;
  }
  if (action.kind != kActionStackPop) return kStatusInvalidState;

  packet->operation_kind = typed_stack::kPopNext;
  bool current_level_frontier_empty = false;
  if (metadata.current_level == 0 &&
      metadata.level_frame_depth == 0) {
    current_level_frontier_empty =
        metadata.frontier_top == 0 &&
        metadata.frontier_count == 0;
  } else if (metadata.current_level ==
                 typed_node::kLevelBlas - 1 &&
             metadata.level_frame_depth == 1) {
    private_frontier::traversal_frame_projection_v0 parent = {};
    if (private_frontier::decode_parent_frame(
            state->canonical_slot, state->owner, &parent) !=
        private_frontier::kStatusOk) {
      return kStatusPrivateStateRejected;
    }
    current_level_frontier_empty =
        metadata.frontier_top ==
            parent.frontier_marker.frontier_top &&
        metadata.frontier_count ==
            parent.frontier_marker.frontier_count;
  }
  if (!current_level_frontier_empty) {
    packet->pop_input.profile_id =
        typed_stack::kGenRtDerivedProfileId;
    packet->pop_input.operation_kind = typed_stack::kPopNext;
    packet->pop_input.has_top_entry = 1;
    packet->pop_input.frontier.frontier_top = metadata.frontier_top;
    packet->pop_input.frontier.frontier_count =
        metadata.frontier_count;
    packet->pop_input.frontier.frontier_capacity =
        metadata.frontier_capacity;
    packet->pop_input.current_traversal_bound_bits = fp32_bits(
        operands.committed_hit.valid != 0
            ? operands.committed_hit.hit_t
            : operands.mutable_ray.t_max);
    if (private_frontier::decode_entry(
            state->canonical_slot, state->owner,
            metadata.frontier_top - 1,
            &packet->pop_input.top_entry) !=
        private_frontier::kStatusOk) {
      return kStatusPrivateStateRejected;
    }
    packet->pop_input.current_decode_context =
        operands.decode_context;
    return kStatusOk;
  }

  packet->empty_input.profile_id =
      typed_stack::kGenRtDerivedProfileId;
  packet->empty_input.operation_kind = typed_stack::kPopNext;
  packet->empty_input.frontier.frontier_top = metadata.frontier_top;
  packet->empty_input.frontier.frontier_count =
      metadata.frontier_count;
  packet->empty_input.frontier.frontier_capacity =
      metadata.frontier_capacity;
  packet->empty_input.frontier.current_level = metadata.current_level;
  packet->empty_input.frontier.level_frame_depth =
      metadata.level_frame_depth;
  packet->empty_input.frontier.max_level_depth =
      metadata.max_level_depth;
  if (metadata.current_level == typed_node::kLevelBlas - 1 &&
      metadata.level_frame_depth == 1) {
    packet->empty_input.parent_frame_available = 1;
    if (private_frontier::decode_parent_frame(
            state->canonical_slot, state->owner,
            &packet->empty_input.parent_frame) !=
        private_frontier::kStatusOk) {
      return kStatusPrivateStateRejected;
    }
  } else if (metadata.current_level == 0 &&
             metadata.level_frame_depth == 0) {
    if (private_frontier::decode_committed_hit(
            state->canonical_slot, state->owner,
            &packet->empty_input.current_committed_hit) !=
        private_frontier::kStatusOk) {
      return kStatusPrivateStateRejected;
    }
  } else {
    return kStatusPrivateStateRejected;
  }
  return kStatusOk;
}

status_kind apply_instance_enter(
    state_v0 *state,
    const functional_driver::instance_enter_execution_v0 &execution) {
  if (execution.semantic_plan.route_kind ==
      instance_semantic::kRouteStackPopNext) {
    return kStatusOk;
  }
  if (execution.semantic_plan.route_kind !=
      instance_semantic::kRouteBlasRootNode) {
    return kStatusSemanticApplyRejected;
  }
  private_frontier::traversal_frame_projection_v0 parent = {};
  private_frontier::access_plan_v0 ignored = {};
  if (private_frontier::capture_parent_frame(
          state->canonical_slot, state->owner, &parent) !=
          private_frontier::kStatusOk ||
      private_frontier::apply_parent_frame_push(
          &state->canonical_slot, state->owner, state->private_region,
          parent, &ignored) != private_frontier::kStatusOk) {
    return kStatusSemanticApplyRejected;
  }
  private_frontier::mutable_ray_state_v0 object_ray = {};
  std::memcpy(&object_ray, &execution.operator_result.object_ray,
              sizeof(object_ray));
  private_frontier::instance_shader_projection_v0 instance = {};
  instance.instance_metadata_ref =
      execution.operator_result.instance_projection
          .instance_metadata_reference;
  instance.instance_index =
      execution.operator_result.instance_projection.instance_index;
  instance.instance_custom_index =
      execution.operator_result.instance_projection
          .instance_custom_index;
  instance.instance_sbt_contribution =
      execution.operator_result.instance_projection
          .instance_sbt_contribution;
  instance.instance_policy_flags =
      execution.operator_result.instance_projection.instance_flags;
  if (private_frontier::apply_instance_enter_state(
          &state->canonical_slot, state->owner, state->private_region,
          object_ray,
          execution.operator_result.root_fetch.decode_context,
          instance, &ignored) != private_frontier::kStatusOk) {
    return kStatusSemanticApplyRejected;
  }
  return kStatusOk;
}

status_kind apply_primitive(
    state_v0 *state,
    const primitive_semantic::semantic_plan_v0 &plan) {
  const private_frontier::committed_hit_projection_v0 *committed =
      plan.committed_hit_valid ? &plan.committed_hit : NULL;
  const private_frontier::retained_candidate_projection_v0 *retained =
      plan.retained_candidate_valid ? &plan.retained_candidate : NULL;
  const typed_primitive::primitive_resume_data_v0 *resume_data =
      plan.primitive_resume_valid ? &plan.primitive_resume : NULL;
  if (committed == NULL && retained == NULL && resume_data == NULL) {
    return kStatusOk;
  }
  private_frontier::access_plan_v0 ignored = {};
  return private_frontier::apply_primitive_result_state(
             &state->canonical_slot, state->owner,
             state->private_region, committed, retained, resume_data,
             &ignored) == private_frontier::kStatusOk
             ? kStatusOk
             : kStatusSemanticApplyRejected;
}

status_kind advance_short_stack_after_node(
    state_v0 *state,
    const functional_driver::node_execution_v0 &execution,
    const next_action_v0 &current, next_action_v0 *next) {
  if (state == NULL || next == NULL ||
      state->short_stack_replay_enabled == 0 ||
      current.short_entry_valid == 0 ||
      current.short_entry.payload_kind !=
          typed_node::kInternalPayloadKind) {
    return kStatusInvalidState;
  }
  uint32_t stack_operation_seq = 0;
  status_kind status =
      allocate_operation(state, &stack_operation_seq);
  if (status != kStatusOk) return status;

  short_stack::route_push_input_v0 input = {};
  input.state = state->short_stack;
  input.route = execution.operator_result;
  input.current_node_payload_offset =
      current.short_entry.payload_offset;
  input.active_domain = state->short_stack.active_domain;
  input.parent_resume = current.parent_resume;
  input.parent_resume_valid = current.parent_resume_valid;
  const short_stack::route_push_result_v0 pushed =
      short_stack::push_node_route(input);
  if (pushed.status != short_stack::kStatusOk) {
    return kStatusShortStackRejected;
  }
  state->short_stack = pushed.state;
  state->same_node_replays += pushed.compressed_to_replay;
  state->bottom_overflows += pushed.overflowed_bottom;
  if (current.parent_resume_valid != 0) {
    ++state->parent_bailouts;
  }

  *next = next_action_v0();
  next->producer_operation_seq = stack_operation_seq;
  if (pushed.selected_valid == 0) {
    next->kind = kActionStackPop;
    return kStatusOk;
  }
  next->kind = kActionTarget;
  next->selected_fetch = pushed.selected;
  next->short_entry = short_entry_from_fetch(
      pushed.selected,
      static_cast<short_stack::domain_kind>(
          state->short_stack.active_domain));
  if (!short_stack::validate_entry(next->short_entry)) {
    return kStatusShortStackRejected;
  }
  next->short_entry_valid = 1;
  return kStatusOk;
}

status_kind publish_output(state_v0 *state, boundary_kind boundary,
                           uint32_t operation_seq,
                           output_v0 *output) {
  if (state == NULL || output == NULL ||
      boundary == kBoundaryInvalid || operation_seq == 0) {
    return kStatusInvalidArgument;
  }
  uint32_t reason = abi_v04::kReasonNoneOrInvalid;
  switch (boundary) {
    case kBoundaryFinalMiss:
      reason = abi_v04::kReasonMiss;
      break;
    case kBoundaryFinalHit:
      reason = abi_v04::kReasonClosestHitReady;
      break;
    case kBoundaryAnyHit:
      reason = abi_v04::kReasonAnyHitRequired;
      break;
    case kBoundaryIntersection:
      reason = abi_v04::kReasonIntersectionRequired;
      break;
    case kBoundaryInvalid:
      return kStatusInvalidArgument;
  }
  state->boundary_kind = boundary;
  state->boundary_operation_seq = operation_seq;
  state->boundary_reason = reason;
  state->waiting_shader =
      boundary == kBoundaryAnyHit ||
      boundary == kBoundaryIntersection;
  state->terminal = !state->waiting_shader;

  *output = output_v0();
  output->valid = 1;
  output->boundary_kind = boundary;
  output->waiting_shader = state->waiting_shader;
  output->terminal = state->terminal;
  output->owner = state->owner;
  output->boundary_operation_seq = operation_seq;
  output->boundary_reason = reason;
  output->operation_count = state->operation_count;
  output->node_visits = state->node_visits;
  output->primitive_tests = state->primitive_tests;
  output->same_node_replays = state->same_node_replays;
  output->parent_bailouts = state->parent_bailouts;
  output->bottom_overflows = state->bottom_overflows;
  private_frontier::root_private_operands_v0 operands = {};
  private_frontier::instance_shader_projection_v0 current_instance = {};
  const status_kind decode_status =
      decode_private(*state, &operands, &current_instance);
  if (decode_status != kStatusOk) return decode_status;
  output->ray = operands.mutable_ray;
  output->committed_hit = operands.committed_hit;
  if (private_frontier::decode_retained_candidate(
          state->canonical_slot, state->owner,
          &output->retained_candidate) !=
      private_frontier::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  return kStatusOk;
}

status_kind restore_parent_for_short_stack(
    state_v0 *state, uint32_t *restore_operation_seq) {
  if (state == NULL || restore_operation_seq == NULL) {
    return kStatusInvalidArgument;
  }
  private_frontier::traversal_frame_projection_v0 parent = {};
  private_frontier::frontier_metadata_image_v0 metadata = {};
  if (private_frontier::decode_parent_frame(
          state->canonical_slot, state->owner, &parent) !=
          private_frontier::kStatusOk ||
      private_frontier::decode_metadata(
          state->canonical_slot, state->owner, &metadata) !=
          private_frontier::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  typed_stack::frontier_level_delta_v0 delta = {};
  delta.action = typed_stack::kFrontierActionPopFrame;
  delta.new_frontier_top = parent.frontier_marker.frontier_top;
  delta.new_frontier_count =
      parent.frontier_marker.frontier_count;
  delta.new_current_level = parent.traversal_level;
  delta.new_level_frame_depth =
      parent.frontier_marker.level_frame_depth;
  delta.max_level_depth = metadata.max_level_depth;
  private_frontier::access_plan_v0 ignored = {};
  if (private_frontier::apply_parent_restore_delta(
          &state->canonical_slot, state->owner,
          state->private_region, delta, &ignored) !=
      private_frontier::kStatusOk) {
    return kStatusSemanticApplyRejected;
  }

  status_kind status =
      allocate_operation(state, restore_operation_seq);
  if (status != kStatusOk) return status;
  fetch_target::operation_packet_v0 restore_packet = {};
  restore_packet.owner = state->owner;
  restore_packet.reservation_id = *restore_operation_seq;
  restore_packet.reservation_age = *restore_operation_seq;
  restore_packet.target_operation_seq = *restore_operation_seq;
  restore_packet.slot_generation = 1;
  restore_packet.target_kind = fetch_target::kTargetInstance;
  restore_packet.operation_kind =
      fetch_target::kOperationInstanceRestoreParent;
  restore_packet.valid = 1;
  restore_packet.parent_frame = parent;
  functional_driver::instance_restore_execution_v0 restore = {};
  const functional_driver::status_kind driver_status =
      functional_driver::execute_one_instance_restore(
          restore_packet, state->private_region,
          state->canonical_slot, &restore);
  if (driver_status != functional_driver::kStatusOk ||
      !restore.valid) {
    return record_driver_rejection(
        state, kDriverUnitInstance, driver_status);
  }
  if (private_frontier::apply_parent_state_restore(
          &state->canonical_slot, state->owner,
          state->private_region,
          restore.operator_result.restored_parent,
          &ignored) != private_frontier::kStatusOk) {
    return kStatusSemanticApplyRejected;
  }
  state->blas_build_generation = 0;
  return kStatusOk;
}

status_kind select_next_short_stack_target(
    state_v0 *state, const provider_v0 &provider,
    next_action_v0 *action, output_v0 *output) {
  if (state == NULL || action == NULL || output == NULL ||
      state->short_stack_replay_enabled == 0 ||
      provider.resolve_parent_edge == NULL) {
    return kStatusInvalidState;
  }
  while (true) {
    if (state->short_stack.stack_count == 0) {
      if (!short_stack::terminal_allowed(
              state->short_stack, false, false, false)) {
        return kStatusShortStackRejected;
      }
      uint32_t terminal_operation_seq = 0;
      status_kind status =
          allocate_operation(state, &terminal_operation_seq);
      if (status != kStatusOk) return status;
      private_frontier::root_private_operands_v0 operands = {};
      private_frontier::instance_shader_projection_v0 instance = {};
      status = decode_private(*state, &operands, &instance);
      if (status != kStatusOk) return status;
      return publish_output(
          state,
          operands.committed_hit.valid != 0
              ? kBoundaryFinalHit
              : kBoundaryFinalMiss,
          terminal_operation_seq, output);
    }

    if (state->short_stack.cross_as != 0 &&
        state->short_stack.stack_count == 1) {
      if (state->short_stack.lost != 0) {
        return kStatusShortStackRejected;
      }
      short_stack::entry_v0 return_entry = {};
      if (!short_stack::read_logical_entry(
              state->short_stack, 0, &return_entry) ||
          short_stack::control_kind(return_entry.control) !=
              short_stack::kEntryCrossAsReturn) {
        return kStatusShortStackRejected;
      }
      private_frontier::traversal_frame_projection_v0 parent_frame = {};
      if (private_frontier::decode_parent_frame(
              state->canonical_slot, state->owner,
              &parent_frame) != private_frontier::kStatusOk) {
        return kStatusPrivateStateRejected;
      }
      short_stack::parent_edge_v0 parent = {};
      if (!provider.resolve_parent_edge(
              provider.context, parent_frame.current_decode_context,
              state->tlas_build_generation,
              return_entry.payload_offset, &parent) ||
          short_stack::return_to_tlas(
              &state->short_stack, parent) !=
              short_stack::kStatusOk) {
        return kStatusParentResolveRejected;
      }
      uint32_t restore_operation_seq = 0;
      status_kind status = restore_parent_for_short_stack(
          state, &restore_operation_seq);
      if (status != kStatusOk) return status;
      continue;
    }

    const bool need_parent =
        short_stack::parent_bailout_required(state->short_stack);
    const short_stack::pop_result_v0 popped =
        short_stack::pop_top(state->short_stack);
    if (popped.status != short_stack::kStatusOk ||
        popped.entry_valid == 0 ||
        short_stack::control_kind(popped.entry.control) ==
            short_stack::kEntryCrossAsReturn) {
      return kStatusShortStackRejected;
    }
    uint32_t stack_operation_seq = 0;
    status_kind status =
        allocate_operation(state, &stack_operation_seq);
    if (status != kStatusOk) return status;

    short_stack::parent_bailout_result_v0 bailout = {};
    if (need_parent) {
      private_frontier::root_private_operands_v0 operands = {};
      private_frontier::instance_shader_projection_v0 instance = {};
      status = decode_private(*state, &operands, &instance);
      if (status != kStatusOk) return status;
      short_stack::parent_edge_v0 parent = {};
      if (!provider.resolve_parent_edge(
              provider.context, operands.decode_context,
              active_build_generation(*state),
              popped.entry.payload_offset, &parent)) {
        return kStatusParentResolveRejected;
      }
      bailout = short_stack::prepare_parent_bailout(popped, parent);
      if (bailout.status != short_stack::kStatusOk) {
        return kStatusShortStackRejected;
      }
      state->short_stack = bailout.state;
    } else {
      state->short_stack = popped.state;
    }

    private_frontier::root_private_operands_v0 operands = {};
    private_frontier::instance_shader_projection_v0 instance = {};
    status = decode_private(*state, &operands, &instance);
    if (status != kStatusOk) return status;
    *action = next_action_v0();
    action->kind = kActionTarget;
    action->producer_operation_seq = stack_operation_seq;
    action->short_entry = popped.entry;
    action->short_entry_valid = 1;
    action->replay_cursor =
        replay_cursor_from_entry(popped.entry);
    action->parent_resume = bailout.parent_resume;
    action->parent_resume_valid =
        bailout.parent_resume_valid;
    if (!selected_fetch_from_short_entry(
            popped.entry, operands.decode_context,
            &action->selected_fetch)) {
      return kStatusShortStackRejected;
    }
    return kStatusOk;
  }
}

status_kind run_loop(state_v0 *state, const provider_v0 &provider,
                     next_action_v0 action, output_v0 *output) {
  if (state == NULL || output == NULL || !provider_valid(provider) ||
      !state->valid || state->waiting_shader || state->terminal) {
    return kStatusInvalidState;
  }

  while (true) {
    if (state->operation_count >= kMaxOperationsPerRun) {
      return kStatusOperationWatchdog;
    }
    if (action.kind == kActionTarget) {
      fetch_target::operation_packet_v0 packet = {};
      status_kind status = build_target_packet(
          state, provider, action.selected_fetch,
          action.instance_blas_root != 0,
          action.producer_operation_seq, &packet);
      if (status != kStatusOk) return status;

      if (packet.target_kind == fetch_target::kTargetNode) {
        functional_driver::node_execution_v0 execution = {};
        const functional_driver::status_kind driver_status =
            state->short_stack_replay_enabled != 0
                ? functional_driver::execute_one_node(
                      packet, action.replay_cursor, &execution)
                : functional_driver::execute_one_node(
                      packet, &execution);
        if (driver_status != functional_driver::kStatusOk ||
            !execution.valid) {
          return record_driver_rejection(
              state, kDriverUnitNode, driver_status);
        }
        ++state->node_visits;
        if (state->short_stack_replay_enabled != 0) {
          next_action_v0 next = {};
          status = advance_short_stack_after_node(
              state, execution, action, &next);
          if (status != kStatusOk) return status;
          action = next;
          continue;
        }
        action = next_action_v0();
        action.producer_operation_seq =
            packet.target_operation_seq;
        if (execution.semantic_plan.route_kind ==
            result_semantic::kNodeRouteDirectChild) {
          action.kind = kActionTarget;
          action.selected_fetch =
              execution.semantic_plan.selected_fetch;
        } else if (execution.semantic_plan.route_kind ==
                   result_semantic::kNodeRouteMultiChildToStack) {
          action.kind = kActionStackPush;
          action.node_route = execution.operator_result;
        } else if (execution.semantic_plan.route_kind ==
                   result_semantic::kNodeRouteNoChild) {
          action.kind = kActionStackPop;
        } else {
          return kStatusSemanticApplyRejected;
        }
        continue;
      }

      if (packet.target_kind == fetch_target::kTargetInstance) {
        typed_instance::enter_input_v0 input = {};
        if (!provider.prepare_instance_enter(
                provider.context, packet, &input)) {
          return kStatusInstanceProducerRejected;
        }
        functional_driver::instance_enter_execution_v0 execution = {};
        const functional_driver::status_kind driver_status =
            functional_driver::execute_one_instance_enter(
                packet, input, state->private_region,
                state->canonical_slot, &execution);
        if (driver_status != functional_driver::kStatusOk ||
            !execution.valid) {
          return record_driver_rejection(
              state, kDriverUnitInstance, driver_status);
        }
        status = apply_instance_enter(state, execution);
        if (status != kStatusOk) return status;
        if (state->short_stack_replay_enabled != 0) {
          const instance_semantic::route_kind route =
              static_cast<instance_semantic::route_kind>(
                  execution.semantic_plan.route_kind);
          if (route ==
              instance_semantic::kRouteBlasRootNode) {
            if (action.short_entry_valid == 0 ||
                action.short_entry.payload_kind !=
                    typed_node::kInstancePayloadKind ||
                execution.operator_result.root_fetch
                        .build_generation == 0) {
              return kStatusShortStackRejected;
            }
            short_stack::cross_as_input_v0 crossing = {};
            crossing.state = state->short_stack;
            crossing.tlas_instance = action.short_entry;
            crossing.blas_root = short_entry_from_fetch(
                execution.semantic_plan.root_fetch,
                short_stack::kDomainBlas);
            const short_stack::cross_as_result_v0 crossed =
                short_stack::enter_blas(crossing);
            if (crossed.status != short_stack::kStatusOk) {
              return kStatusShortStackRejected;
            }
            state->short_stack = crossed.state;
            state->blas_build_generation =
                execution.operator_result.root_fetch
                    .build_generation;
            next_action_v0 next = {};
            status = select_next_short_stack_target(
                state, provider, &next, output);
            if (status != kStatusOk || output->valid != 0) {
              return status;
            }
            next.instance_blas_root = 1;
            action = next;
            continue;
          }
          if (route !=
              instance_semantic::kRouteStackPopNext) {
            return kStatusSemanticApplyRejected;
          }
          next_action_v0 next = {};
          status = select_next_short_stack_target(
              state, provider, &next, output);
          if (status != kStatusOk || output->valid != 0) {
            return status;
          }
          action = next;
          continue;
        }
        action = next_action_v0();
        action.producer_operation_seq =
            packet.target_operation_seq;
        if (execution.semantic_plan.route_kind ==
            instance_semantic::kRouteStackPopNext) {
          action.kind = kActionStackPop;
        } else if (execution.semantic_plan.route_kind ==
                   instance_semantic::kRouteBlasRootNode) {
          action.kind = kActionTarget;
          action.instance_blas_root = 1;
          action.selected_fetch = execution.semantic_plan.root_fetch;
          state->ray_policy = execution.semantic_plan.ray_policy;
        } else {
          return kStatusSemanticApplyRejected;
        }
        continue;
      }

      if (packet.target_kind == fetch_target::kTargetPrimitive) {
        typed_primitive::route_input_v0 input = {};
        functional_driver::status_kind driver_status =
            functional_driver::prepare_primitive_operator_input(
                packet, &input);
        if (driver_status != functional_driver::kStatusOk) {
          return record_driver_rejection(
              state, kDriverUnitPrimitive, driver_status);
        }
        functional_driver::primitive_execution_v0 execution = {};
        driver_status = functional_driver::execute_one_primitive(
            packet, input, state->private_region,
            state->canonical_slot, &execution);
        if (driver_status != functional_driver::kStatusOk ||
            !execution.valid) {
          return record_driver_rejection(
              state, kDriverUnitPrimitive, driver_status);
        }
        ++state->primitive_tests;
        status = apply_primitive(state, execution.semantic_plan);
        if (status != kStatusOk) return status;
        const primitive_semantic::route_kind route =
            static_cast<primitive_semantic::route_kind>(
                execution.semantic_plan.route_kind);
        if (route == primitive_semantic::kRouteAnyHitBoundary) {
          return publish_output(state, kBoundaryAnyHit,
                                packet.target_operation_seq, output);
        }
        if (route == primitive_semantic::kRouteIntersectionBoundary) {
          status = publish_output(
              state, kBoundaryIntersection,
              packet.target_operation_seq, output);
          if (status == kStatusOk) {
            output->intersection_boundary =
                execution.semantic_plan.intersection_boundary;
          }
          return status;
        }
        if (route == primitive_semantic::kRouteFinalHitBoundary) {
          return publish_output(state, kBoundaryFinalHit,
                                packet.target_operation_seq, output);
        }
        if (route != primitive_semantic::kRouteStackPopNext) {
          return kStatusSemanticApplyRejected;
        }
        if (state->short_stack_replay_enabled != 0) {
          next_action_v0 next = {};
          status = select_next_short_stack_target(
              state, provider, &next, output);
          if (status != kStatusOk || output->valid != 0) {
            return status;
          }
          action = next;
          continue;
        }
        action = next_action_v0();
        action.kind = kActionStackPop;
        action.producer_operation_seq =
            packet.target_operation_seq;
        continue;
      }
      return kStatusTargetRejected;
    }

    if (state->short_stack_replay_enabled != 0 &&
        action.kind == kActionStackPop) {
      next_action_v0 next = {};
      status_kind status = select_next_short_stack_target(
          state, provider, &next, output);
      if (status != kStatusOk || output->valid != 0) {
        return status;
      }
      action = next;
      continue;
    }
    if (action.kind == kActionStackPush ||
        action.kind == kActionStackPop) {
      stack_operation::operation_packet_v0 packet = {};
      status_kind status = build_stack_packet(state, action, &packet);
      if (status != kStatusOk) return status;
      functional_driver::stack_execution_v0 execution = {};
      const functional_driver::status_kind driver_status =
          functional_driver::execute_one_stack(
              packet, state->private_region, state->canonical_slot,
              &execution);
      if (driver_status != functional_driver::kStatusOk ||
          !execution.valid) {
        return record_driver_rejection(
            state, kDriverUnitStack, driver_status);
      }

      action = next_action_v0();
      action.producer_operation_seq =
          packet.target_operation_seq;
      if (execution.semantic_plan_kind ==
          functional_driver::kSemanticPlanStackAppend) {
        private_frontier::access_plan_v0 ignored = {};
        if (private_frontier::apply_append_delta(
                &state->canonical_slot, state->owner,
                state->private_region,
                execution.push_result.frontier_delta,
                &ignored) != private_frontier::kStatusOk) {
          return kStatusSemanticApplyRejected;
        }
        action.kind = kActionTarget;
        action.selected_fetch = execution.push_result.selected_fetch;
        continue;
      }
      if (execution.semantic_plan_kind ==
          functional_driver::kSemanticPlanStackPop) {
        private_frontier::access_plan_v0 ignored = {};
        if (private_frontier::apply_pop_delta(
                &state->canonical_slot, state->owner,
                state->private_region,
                execution.pop_result.frontier_delta,
                &ignored) != private_frontier::kStatusOk) {
          return kStatusSemanticApplyRejected;
        }
        if (execution.pop_result.result_kind ==
            typed_stack::kStackSelectedNext) {
          action.kind = kActionTarget;
          action.selected_fetch =
              execution.pop_result.selected_fetch;
        } else if (execution.pop_result.result_kind ==
                   typed_stack::kStackPrunedRetryPop) {
          action.kind = kActionStackPop;
        } else {
          return kStatusSemanticApplyRejected;
        }
        continue;
      }
      if (execution.semantic_plan_kind ==
          functional_driver::kSemanticPlanStackRestoreParent) {
        private_frontier::access_plan_v0 ignored = {};
        if (private_frontier::apply_parent_restore_delta(
                &state->canonical_slot, state->owner,
                state->private_region,
                execution.empty_result.frontier_delta,
                &ignored) != private_frontier::kStatusOk) {
          return kStatusSemanticApplyRejected;
        }
        uint32_t restore_operation_seq = 0;
        status = allocate_operation(state, &restore_operation_seq);
        if (status != kStatusOk) return status;
        fetch_target::operation_packet_v0 restore_packet = {};
        restore_packet.owner = state->owner;
        restore_packet.reservation_id = restore_operation_seq;
        restore_packet.reservation_age = restore_operation_seq;
        restore_packet.target_operation_seq = restore_operation_seq;
        restore_packet.slot_generation = 1;
        restore_packet.target_kind = fetch_target::kTargetInstance;
        restore_packet.operation_kind =
            fetch_target::kOperationInstanceRestoreParent;
        restore_packet.valid = 1;
        restore_packet.parent_frame =
            execution.empty_result.parent_frame;
        functional_driver::instance_restore_execution_v0 restore = {};
        const functional_driver::status_kind driver_status =
            functional_driver::execute_one_instance_restore(
                restore_packet, state->private_region,
                state->canonical_slot, &restore);
        if (driver_status != functional_driver::kStatusOk ||
            !restore.valid) {
          return record_driver_rejection(
              state, kDriverUnitInstance, driver_status);
        }
        if (private_frontier::apply_parent_state_restore(
                &state->canonical_slot, state->owner,
                state->private_region,
                restore.operator_result.restored_parent,
                &ignored) != private_frontier::kStatusOk) {
          return kStatusSemanticApplyRejected;
        }
        action.kind = kActionStackPop;
        action.producer_operation_seq = restore_operation_seq;
        continue;
      }
      if (execution.semantic_plan_kind ==
          functional_driver::kSemanticPlanStackTerminalHit) {
        return publish_output(state, kBoundaryFinalHit,
                              packet.target_operation_seq, output);
      }
      if (execution.semantic_plan_kind ==
          functional_driver::kSemanticPlanStackTerminalMiss) {
        return publish_output(state, kBoundaryFinalMiss,
                              packet.target_operation_seq, output);
      }
      return kStatusSemanticApplyRejected;
    }
    return kStatusInvalidState;
  }
}

}  // namespace

status_kind run_new(const root_input_v0 &input,
                    const provider_v0 &provider, state_v0 *state,
                    output_v0 *output) {
  if (state == NULL || output == NULL || !provider_valid(provider) ||
      !request_owner::validate_private_frontier_owner_identity(
          input.owner) ||
      input.private_region.profile_id !=
          private_frontier::kLayoutProfileId ||
      input.private_region.private_region_base == 0 ||
      input.private_region.slot_count == 0 ||
      input.owner.private_slot_id >= input.private_region.slot_count ||
      input.root_reference.source_kind !=
          fetch_target::kTargetReferenceRootCompatibilityProxy ||
      input.root_reference.proxy_delegated != 1 ||
      input.raw_payload_base_address == 0 ||
      input.short_stack_replay_enabled > 1 ||
      (input.short_stack_replay_enabled != 0 &&
       (input.root_build_generation == 0 ||
        provider.resolve_parent_edge == NULL)) ||
      !std::all_of(input.reserved_zero,
                   input.reserved_zero +
                       sizeof(input.reserved_zero),
                   [](uint8_t value) { return value == 0; })) {
    return kStatusInvalidArgument;
  }

  *state = state_v0();
  *output = output_v0();
  state->valid = 1;
  state->owner = input.owner;
  state->private_region = input.private_region;
  state->ray_policy = input.ray_policy;
  state->short_stack_replay_enabled =
      input.short_stack_replay_enabled;
  state->short_stack.active_domain =
      short_stack::kDomainTlas;
  state->tlas_build_generation =
      input.root_build_generation;
  state->next_operation_seq = 1;
  private_frontier::frontier_metadata_image_v0 metadata = {};
  metadata.frontier_capacity =
      private_frontier::kFrontierEntryCapacity;
  metadata.max_level_depth = 1;
  private_frontier::access_plan_v0 ignored = {};
  if (private_frontier::initialize_root_shadow_slot(
          &state->canonical_slot, input.owner, input.private_region,
          metadata, input.private_operands, &ignored) !=
      private_frontier::kStatusOk) {
    return kStatusPrivateStateRejected;
  }

  fetch_target::operation_packet_v0 root_packet = {};
  status_kind status =
      build_root_packet(state, input, provider, &root_packet);
  if (status != kStatusOk) return status;
  functional_driver::node_execution_v0 execution = {};
  if (functional_driver::execute_one_node(
          root_packet, &execution) != functional_driver::kStatusOk ||
      !execution.valid) {
    return kStatusFunctionalDriverRejected;
  }
  ++state->node_visits;

  next_action_v0 action = {};
  action.producer_operation_seq =
      root_packet.target_operation_seq;
  if (state->short_stack_replay_enabled != 0) {
    action.kind = kActionTarget;
    action.short_entry.payload_offset =
        input.root_reference.payload_offset;
    action.short_entry.near_t_bits =
        input.root_reference.near_t_bits;
    action.short_entry.payload_byte_count =
        input.root_reference.payload_byte_count;
    action.short_entry.payload_kind =
        input.root_reference.payload_kind;
    action.short_entry.control = short_stack::make_control(
        short_stack::kEntryDirectTarget,
        short_stack::kDomainTlas, 0, true, false);
    action.short_entry_valid = 1;
    if (!short_stack::validate_entry(action.short_entry)) {
      return kStatusShortStackRejected;
    }
    next_action_v0 next = {};
    status = advance_short_stack_after_node(
        state, execution, action, &next);
    if (status != kStatusOk) return status;
    return run_loop(state, provider, next, output);
  }
  if (execution.semantic_plan.route_kind ==
      result_semantic::kNodeRouteDirectChild) {
    action.kind = kActionTarget;
    action.selected_fetch = execution.semantic_plan.selected_fetch;
  } else if (execution.semantic_plan.route_kind ==
             result_semantic::kNodeRouteMultiChildToStack) {
    action.kind = kActionStackPush;
    action.node_route = execution.operator_result;
  } else if (execution.semantic_plan.route_kind ==
             result_semantic::kNodeRouteNoChild) {
    action.kind = kActionStackPop;
  } else {
    return kStatusSemanticApplyRejected;
  }
  return run_loop(state, provider, action, output);
}

status_kind resume(
    state_v0 *state, uint32_t boundary_reason,
    const std::array<uint32_t, abi_v04::kWordCount> &shader_return_words,
    const provider_v0 &provider, output_v0 *output) {
  if (state == NULL || output == NULL || !provider_valid(provider) ||
      !state->valid || !state->waiting_shader || state->terminal ||
      state->boundary_operation_seq == 0) {
    return kStatusInvalidState;
  }
  uint32_t operation_seq = 0;
  status_kind status = allocate_operation(state, &operation_seq);
  if (status != kStatusOk) return status;
  primitive_semantic::semantic_plan_v0 plan = {};
  if (continuation_lifecycle::prepare_shader_return_semantic_plan(
          state->owner, operation_seq, boundary_reason,
          shader_return_words, state->canonical_slot, &plan) !=
      continuation_lifecycle::kStatusOk) {
    return kStatusShaderReturnRejected;
  }
  private_frontier::access_plan_v0 ignored = {};
  const private_frontier::committed_hit_projection_v0 *committed =
      plan.committed_hit_valid ? &plan.committed_hit : NULL;
  if (private_frontier::apply_shader_return_state(
          &state->canonical_slot, state->owner, state->private_region,
          committed, &ignored) != private_frontier::kStatusOk) {
    return kStatusSemanticApplyRejected;
  }
  state->waiting_shader = 0;
  state->boundary_kind = kBoundaryInvalid;
  state->boundary_reason = 0;
  if (plan.route_kind ==
      primitive_semantic::kRouteFinalHitBoundary) {
    return publish_output(
        state, kBoundaryFinalHit, operation_seq, output);
  }
  if (plan.route_kind !=
      primitive_semantic::kRouteStackPopNext) {
    return kStatusSemanticApplyRejected;
  }
  next_action_v0 action = {};
  action.kind = kActionStackPop;
  action.producer_operation_seq = operation_seq;
  return run_loop(state, provider, action, output);
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk: return "ok";
    case kStatusInvalidArgument: return "invalid_argument";
    case kStatusInvalidState: return "invalid_state";
    case kStatusOperationSequenceExhausted:
      return "operation_sequence_exhausted";
    case kStatusOperationWatchdog: return "operation_watchdog";
    case kStatusPrivateStateRejected: return "private_state_rejected";
    case kStatusTargetRejected: return "target_rejected";
    case kStatusPayloadReadRejected: return "payload_read_rejected";
    case kStatusInstanceProducerRejected:
      return "instance_producer_rejected";
    case kStatusFunctionalDriverRejected:
      return "functional_driver_rejected";
    case kStatusSemanticApplyRejected:
      return "semantic_apply_rejected";
    case kStatusShaderReturnRejected:
      return "shader_return_rejected";
    case kStatusParentResolveRejected:
      return "parent_resolve_rejected";
    case kStatusShortStackRejected:
      return "short_stack_rejected";
  }
  return "unknown";
}

const char *boundary_name(boundary_kind boundary) {
  switch (boundary) {
    case kBoundaryFinalMiss: return "final_miss";
    case kBoundaryFinalHit: return "final_hit";
    case kBoundaryAnyHit: return "any_hit";
    case kBoundaryIntersection: return "intersection";
    case kBoundaryInvalid: break;
  }
  return "invalid";
}

const char *driver_unit_name(driver_unit_kind unit) {
  switch (unit) {
    case kDriverUnitNode: return "node";
    case kDriverUnitInstance: return "instance";
    case kDriverUnitPrimitive: return "primitive";
    case kDriverUnitStack: return "stack";
    case kDriverUnitInvalid: break;
  }
  return "invalid";
}

}  // namespace functional_engine
}  // namespace v04
}  // namespace rtcore
