#include "rtcore_v04_short_stack_transition.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace short_stack_transition {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

uint32_t active_build_generation(
    const short_stack_shared::persistent_state_v0 &state) {
  return state.stack.active_domain == short_stack::kDomainBlas
             ? state.blas_build_generation
             : state.tlas_build_generation;
}

uint8_t expected_as_type(uint8_t active_domain) {
  return active_domain == short_stack::kDomainBlas
             ? typed_blas::kAsTypeBlas
             : 1;
}

bool selected_fetch_from_entry(
    const short_stack::entry_v0 &entry,
    const typed_blas::as_decode_context_v0 &decode_context,
    typed_node::selected_child_fetch_work_item_v0 *selected,
    typed_node::replay_cursor_v0 *cursor) {
  if (selected == NULL || cursor == NULL ||
      !short_stack::validate_entry(entry) ||
      short_stack::control_kind(entry.control) ==
          short_stack::kEntryCrossAsReturn) {
    return false;
  }
  *selected = typed_node::selected_child_fetch_work_item_v0();
  selected->child.payload_offset = entry.payload_offset;
  selected->child.near_t_bits = entry.near_t_bits;
  selected->child.payload_byte_count = entry.payload_byte_count;
  selected->child.payload_kind = entry.payload_kind;
  selected->child.child_slot =
      short_stack::control_child_anchor(entry.control);
  selected->decode_context = decode_context;

  *cursor = typed_node::replay_cursor_v0();
  const short_stack::entry_kind kind =
      short_stack::control_kind(entry.control);
  if (kind == short_stack::kEntrySameNodeReplay ||
      kind == short_stack::kEntryParentResume) {
    cursor->child_anchor =
        short_stack::control_child_anchor(entry.control);
    cursor->anchor_valid =
        short_stack::control_anchor_valid(entry.control) ? 1 : 0;
    cursor->inclusive =
        short_stack::control_inclusive(entry.control) ? 1 : 0;
  }
  return true;
}

bool bind_recovery_target_inflight(
    short_stack_shared::persistent_state_v0 *persistent,
    const short_stack::entry_v0 &selected) {
  if (persistent == NULL) return false;
  if (!short_stack::validate_drained_recovery_state(
          persistent->stack)) {
    return persistent->recovery_target_inflight == 0;
  }
  const short_stack::entry_kind kind =
      short_stack::control_kind(selected.control);
  if (!short_stack::validate_entry(selected) ||
      selected.payload_kind != typed_node::kInternalPayloadKind ||
      (kind != short_stack::kEntryDirectTarget &&
       kind != short_stack::kEntrySameNodeReplay &&
       kind != short_stack::kEntryParentResume)) {
    return false;
  }
  persistent->recovery_target_inflight = 1;
  return true;
}

short_stack::entry_v0 direct_entry_from_target(
    const fetch_target::target_reference_v0 &target,
    short_stack::domain_kind domain) {
  short_stack::entry_v0 entry = {};
  entry.payload_offset = target.payload_offset;
  entry.near_t_bits = target.near_t_bits;
  entry.payload_byte_count = target.payload_byte_count;
  entry.payload_kind = target.payload_kind;
  entry.control = short_stack::make_control(
      short_stack::kEntryDirectTarget, domain, 0, true, false);
  return entry;
}

short_stack::entry_v0 direct_entry_from_fetch(
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

status_kind finalize_persistent_state(
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const short_stack_shared::persistent_state_v0 &persistent,
    result_v0 *result) {
  result->updated_slot = canonical_slot;
  if (short_stack_shared::apply_persistent_state(
          &result->updated_slot, owner, region, persistent,
          &result->write_plan) != short_stack_shared::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  result->persistent_state = persistent;
  return kStatusOk;
}

bool append_access_plan(
    private_frontier::access_plan_v0 *destination,
    const private_frontier::access_plan_v0 &source) {
  if (destination == NULL ||
      destination->access_count + source.access_count >
          private_frontier::kMaxAccessChunks) {
    return false;
  }
  if (destination->access_count == 0) {
    destination->owner = source.owner;
  } else if (!private_frontier::owners_equal(
                 destination->owner, source.owner)) {
    return false;
  }
  for (unsigned index = 0; index < source.access_count; ++index) {
    destination->accesses[destination->access_count++] =
        source.accesses[index];
  }
  return true;
}

}  // namespace

static status_kind prepare_node_transition_impl(
    const node_operands_input_v1 &input,
    const private_frontier::shadow_slot_v0 &write_seed,
    result_v0 *result) {
  if (result == NULL ||
      input.pending_parent_resume_valid > 1 ||
      input.parent_edge_valid > 1 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *result = result_v0();

  short_stack_shared::persistent_state_v0 persistent =
      input.persistent_state;
  if (!short_stack_shared::validate_persistent_state(persistent)) {
    return kStatusPrivateStateRejected;
  }
  const uint32_t build_generation =
      active_build_generation(persistent);
  if (input.current_target.build_generation != build_generation) {
    return kStatusGenerationMismatch;
  }
  if (input.current_target.payload_kind !=
          typed_node::kInternalPayloadKind ||
      input.current_target.payload_byte_count !=
          fetch_target::kNodeRawPayloadBytes ||
      input.current_decode_context.as_object.as_type !=
          expected_as_type(persistent.stack.active_domain) ||
      input.current_target.level !=
          (persistent.stack.active_domain ==
                   short_stack::kDomainBlas
               ? typed_node::kLevelBlas
               : typed_node::kLevelTlas)) {
    return kStatusDecodeContextMismatch;
  }
  if (persistent.recovery_target_inflight != 0 &&
      input.current_target.payload_kind !=
          typed_node::kInternalPayloadKind) {
    return kStatusShortStackRejected;
  }

  short_stack::entry_v0 pending_parent_resume =
      input.pending_parent_resume;
  uint8_t pending_parent_resume_valid =
      input.pending_parent_resume_valid;
  bool current_parent_bailout_consumed = false;
  if (persistent.recovery_target_inflight != 0 &&
      pending_parent_resume_valid == 0) {
    if (input.parent_edge_valid == 0) {
      result->parent_lookup_decode_context =
          input.current_decode_context;
      result->parent_lookup_build_generation =
          build_generation;
      result->parent_lookup_payload_offset =
          input.current_target.payload_offset;
      result->parent_lookup_payload_kind =
          input.current_target.payload_kind;
      result->parent_lookup_required = 1;
      result->persistent_state = persistent;
      return kStatusParentLookupRequired;
    }
    short_stack::pop_result_v0 current = {};
    current.status = short_stack::kStatusOk;
    current.state = persistent.stack;
    current.entry = direct_entry_from_target(
        input.current_target,
        static_cast<short_stack::domain_kind>(
            persistent.stack.active_domain));
    current.entry_valid = 1;
    const short_stack::parent_bailout_result_v0 bailout =
        short_stack::prepare_parent_bailout(
            current, input.parent_edge);
    if (bailout.status != short_stack::kStatusOk) {
      return kStatusShortStackRejected;
    }
    persistent.stack = bailout.state;
    persistent.recovery_target_inflight = 0;
    pending_parent_resume = bailout.parent_resume;
    pending_parent_resume_valid =
        bailout.parent_resume_valid;
    current_parent_bailout_consumed = true;
  } else if (persistent.recovery_target_inflight != 0) {
    persistent.recovery_target_inflight = 0;
  }

  short_stack::route_push_input_v0 push = {};
  push.state = persistent.stack;
  push.route = input.node_route;
  push.current_node_payload_offset =
      input.current_target.payload_offset;
  push.parent_resume = pending_parent_resume;
  push.active_domain = persistent.stack.active_domain;
  push.parent_resume_valid =
      pending_parent_resume_valid;
  const short_stack::route_push_result_v0 pushed =
      short_stack::push_node_route(push);
  if (pushed.status != short_stack::kStatusOk) {
    return kStatusShortStackRejected;
  }

  persistent.stack = pushed.state;
  result->compressed_to_replay = pushed.compressed_to_replay;
  result->overflowed_bottom = pushed.overflowed_bottom;
  if (pushed.selected_valid != 0) {
    result->selected_fetch = pushed.selected;
    result->selected_valid = 1;
    result->next_build_generation = build_generation;
  } else if (persistent.stack.cross_as != 0 &&
             persistent.stack.stack_count == 1) {
    if (persistent.stack.active_domain != short_stack::kDomainBlas ||
        persistent.stack.lost != 0 ||
        input.node_route.result_kind !=
            typed_node::kRouteResultMiss) {
      return kStatusShortStackRejected;
    }
    result->persistent_state = persistent;
    return kStatusReturnInstanceRequired;
  } else if (persistent.stack.stack_count == 0) {
    if (!short_stack::terminal_allowed(
            persistent.stack, false, false, false)) {
      return kStatusShortStackRejected;
    }
    result->terminal = 1;
  } else {
    const bool need_parent =
        !current_parent_bailout_consumed &&
        short_stack::parent_bailout_required(persistent.stack);
    if (need_parent && input.parent_edge_valid == 0) {
      short_stack::entry_v0 top = {};
      if (!short_stack::read_logical_entry(
              persistent.stack, 0, &top)) {
        return kStatusShortStackRejected;
      }
      result->parent_lookup_decode_context =
          input.current_decode_context;
      result->parent_lookup_build_generation = build_generation;
      result->parent_lookup_payload_offset = top.payload_offset;
      result->parent_lookup_payload_kind = top.payload_kind;
      result->parent_lookup_required = 1;
      result->persistent_state = persistent;
      return kStatusParentLookupRequired;
    }

    const short_stack::pop_result_v0 popped =
        short_stack::pop_top(persistent.stack);
    if (popped.status != short_stack::kStatusOk ||
        popped.entry_valid == 0) {
      return kStatusShortStackRejected;
    }
    short_stack::parent_bailout_result_v0 bailout = {};
    if (need_parent) {
      bailout = short_stack::prepare_parent_bailout(
          popped, input.parent_edge);
      if (bailout.status != short_stack::kStatusOk) {
        return kStatusShortStackRejected;
      }
      persistent.stack = bailout.state;
      result->pending_parent_resume = bailout.parent_resume;
      result->pending_parent_resume_valid =
          bailout.parent_resume_valid;
    } else {
      persistent.stack = popped.state;
    }
    if (!bind_recovery_target_inflight(
            &persistent, popped.entry)) {
      return kStatusShortStackRejected;
    }
    if (!selected_fetch_from_entry(
            popped.entry, input.current_decode_context,
            &result->selected_fetch, &result->replay_cursor)) {
      return kStatusShortStackRejected;
    }
    result->selected_valid = 1;
    result->next_build_generation = build_generation;
  }

  return finalize_persistent_state(
      input.owner, input.region, write_seed, persistent, result);
}

status_kind prepare_node_transition(const node_input_v0 &input,
                                    result_v0 *result) {
  short_stack_shared::persistent_state_v0 persistent = {};
  if (short_stack_shared::decode_persistent_state(
          input.canonical_slot, input.owner, &persistent) !=
      short_stack_shared::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  node_operands_input_v1 operands = {};
  operands.owner = input.owner;
  operands.region = input.region;
  operands.persistent_state = persistent;
  operands.node_route = input.node_route;
  operands.current_target = input.current_target;
  operands.current_decode_context = input.current_decode_context;
  operands.pending_parent_resume = input.pending_parent_resume;
  operands.parent_edge = input.parent_edge;
  operands.pending_parent_resume_valid =
      input.pending_parent_resume_valid;
  operands.parent_edge_valid = input.parent_edge_valid;
  return prepare_node_transition_impl(
      operands, input.canonical_slot, result);
}

status_kind prepare_node_transition_from_operands(
    const node_operands_input_v1 &input, result_v0 *result) {
  private_frontier::shadow_slot_v0 empty_write_seed = {};
  empty_write_seed.owner = input.owner;
  return prepare_node_transition_impl(
      input, empty_write_seed, result);
}

static status_kind prepare_resume_transition_impl(
    const resume_operands_input_v1 &input,
    const private_frontier::shadow_slot_v0 &write_seed,
    result_v0 *result) {
  if (result == NULL || input.parent_restore_valid > 1 ||
      input.parent_edge_valid > 1 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *result = result_v0();
  short_stack_shared::persistent_state_v0 persistent =
      input.persistent_state;
  if (!short_stack_shared::validate_persistent_state(persistent)) {
    return kStatusPrivateStateRejected;
  }
  if (persistent.recovery_target_inflight != 0) {
    return kStatusShortStackRejected;
  }
  const uint32_t build_generation =
      active_build_generation(persistent);
  if (input.active_decode_context.as_object.as_type !=
      expected_as_type(persistent.stack.active_domain)) {
    return kStatusDecodeContextMismatch;
  }

  if (persistent.stack.stack_count == 0) {
    if (!short_stack::terminal_allowed(
            persistent.stack, false, false, false)) {
      return kStatusShortStackRejected;
    }
    result->terminal = 1;
    return finalize_persistent_state(
        input.owner, input.region, write_seed,
        persistent, result);
  }

  if (persistent.stack.cross_as != 0 &&
      persistent.stack.stack_count == 1) {
    short_stack::entry_v0 return_entry = {};
    if (persistent.stack.lost != 0 ||
        input.parent_restore_valid == 0 ||
        input.parent_decode_context.as_object.as_type != 1 ||
        !short_stack::read_logical_entry(
            persistent.stack, 0, &return_entry) ||
        short_stack::control_kind(return_entry.control) !=
            short_stack::kEntryCrossAsReturn) {
      return kStatusShortStackRejected;
    }
    if (input.parent_edge_valid == 0) {
      result->parent_lookup_decode_context =
          input.parent_decode_context;
      result->parent_lookup_build_generation =
          persistent.tlas_build_generation;
      result->parent_lookup_payload_offset =
          return_entry.payload_offset;
      result->parent_lookup_payload_kind =
          return_entry.payload_kind;
      result->parent_lookup_required = 1;
      result->persistent_state = persistent;
      return kStatusParentLookupRequired;
    }
    if (short_stack::return_to_tlas(
            &persistent.stack, input.parent_edge) !=
        short_stack::kStatusOk) {
      return kStatusShortStackRejected;
    }
    persistent.blas_build_generation = 0;
    result->next_build_generation =
        persistent.tlas_build_generation;

    private_frontier::traversal_frame_projection_v0 root = {};
    root.ray = input.parent_ray;
    root.current_decode_context =
        input.parent_decode_context;
    root.current_instance = input.parent_instance;
    private_frontier::shadow_slot_v0 restored_slot =
        write_seed;
    private_frontier::access_plan_v0 restore_plan = {};
    if (private_frontier::apply_parent_state_restore(
            &restored_slot, input.owner, input.region, root,
            &restore_plan) != private_frontier::kStatusOk) {
      return kStatusPrivateStateRejected;
    }

    if (persistent.stack.stack_count == 0) {
      if (!short_stack::terminal_allowed(
              persistent.stack, false, false, false)) {
        return kStatusShortStackRejected;
      }
      result->terminal = 1;
    } else {
      const short_stack::pop_result_v0 popped =
          short_stack::pop_top(persistent.stack);
      if (popped.status != short_stack::kStatusOk ||
          popped.entry_valid == 0 ||
          short_stack::control_kind(popped.entry.control) !=
              short_stack::kEntryParentResume ||
          !selected_fetch_from_entry(
              popped.entry,
              input.parent_decode_context,
              &result->selected_fetch, &result->replay_cursor)) {
        return kStatusShortStackRejected;
      }
      persistent.stack = popped.state;
      if (!bind_recovery_target_inflight(
              &persistent, popped.entry)) {
        return kStatusShortStackRejected;
      }
      result->selected_valid = 1;
    }

    private_frontier::access_plan_v0 stack_plan = {};
    result->updated_slot = restored_slot;
    if (short_stack_shared::apply_persistent_state(
            &result->updated_slot, input.owner, input.region,
            persistent, &stack_plan) !=
            short_stack_shared::kStatusOk ||
        !append_access_plan(&result->write_plan, restore_plan) ||
        !append_access_plan(&result->write_plan, stack_plan)) {
      return kStatusPrivateStateRejected;
    }
    result->persistent_state = persistent;
    return kStatusOk;
  }

  const bool need_parent =
      short_stack::parent_bailout_required(persistent.stack);
  short_stack::entry_v0 top = {};
  if (need_parent && input.parent_edge_valid == 0) {
    if (!short_stack::read_logical_entry(
            persistent.stack, 0, &top)) {
      return kStatusShortStackRejected;
    }
    result->parent_lookup_decode_context =
        input.active_decode_context;
    result->parent_lookup_build_generation = build_generation;
    result->parent_lookup_payload_offset = top.payload_offset;
    result->parent_lookup_payload_kind = top.payload_kind;
    result->parent_lookup_required = 1;
    result->persistent_state = persistent;
    return kStatusParentLookupRequired;
  }

  const short_stack::pop_result_v0 popped =
      short_stack::pop_top(persistent.stack);
  if (popped.status != short_stack::kStatusOk ||
      popped.entry_valid == 0 ||
      short_stack::control_kind(popped.entry.control) ==
          short_stack::kEntryCrossAsReturn) {
    return kStatusShortStackRejected;
  }
  short_stack::parent_bailout_result_v0 bailout = {};
  if (need_parent) {
    bailout = short_stack::prepare_parent_bailout(
        popped, input.parent_edge);
    if (bailout.status != short_stack::kStatusOk) {
      return kStatusShortStackRejected;
    }
    persistent.stack = bailout.state;
    result->pending_parent_resume = bailout.parent_resume;
    result->pending_parent_resume_valid =
        bailout.parent_resume_valid;
  } else {
    persistent.stack = popped.state;
  }
  if (!bind_recovery_target_inflight(
          &persistent, popped.entry)) {
    return kStatusShortStackRejected;
  }
  if (!selected_fetch_from_entry(
          popped.entry, input.active_decode_context,
          &result->selected_fetch, &result->replay_cursor)) {
    return kStatusShortStackRejected;
  }
  result->selected_valid = 1;
  result->next_build_generation = build_generation;
  return finalize_persistent_state(
      input.owner, input.region, write_seed,
      persistent, result);
}

status_kind prepare_resume_transition(const resume_input_v0 &input,
                                      result_v0 *result) {
  if (result == NULL ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  resume_operands_input_v1 operands = {};
  operands.owner = input.owner;
  operands.region = input.region;
  if (short_stack_shared::decode_persistent_state(
          input.canonical_slot, input.owner,
          &operands.persistent_state) !=
      short_stack_shared::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  operands.parent_ray =
      input.immutable_trace_input.mutable_ray;
  operands.parent_decode_context =
      input.immutable_trace_input.decode_context;
  operands.active_decode_context =
      input.active_decode_context;
  operands.parent_edge = input.parent_edge;
  operands.parent_restore_valid = 1;
  operands.parent_edge_valid = input.parent_edge_valid;
  return prepare_resume_transition_impl(
      operands, input.canonical_slot, result);
}

status_kind prepare_resume_transition_from_operands(
    const resume_operands_input_v1 &input, result_v0 *result) {
  private_frontier::shadow_slot_v0 empty_write_seed = {};
  empty_write_seed.owner = input.owner;
  return prepare_resume_transition_impl(
      input, empty_write_seed, result);
}

static status_kind prepare_enter_blas_transition_impl(
    const enter_blas_operands_input_v1 &input,
    const private_frontier::shadow_slot_v0 &write_seed,
    result_v0 *result) {
  if (result == NULL ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *result = result_v0();
  short_stack_shared::persistent_state_v0 persistent =
      input.persistent_state;
  if (!short_stack_shared::validate_persistent_state(persistent)) {
    return kStatusPrivateStateRejected;
  }
  if (persistent.recovery_target_inflight != 0) {
    return kStatusShortStackRejected;
  }
  if (persistent.stack.active_domain != short_stack::kDomainTlas ||
      persistent.stack.cross_as != 0 ||
      input.tlas_instance_target.build_generation !=
          persistent.tlas_build_generation ||
      input.tlas_instance_target.payload_kind !=
          typed_node::kInstancePayloadKind ||
      input.tlas_instance_target.level != typed_node::kLevelTlas ||
      input.blas_root.child.payload_kind !=
          typed_node::kInternalPayloadKind ||
      input.blas_root.child.payload_byte_count !=
          fetch_target::kNodeRawPayloadBytes ||
      input.blas_root.decode_context.as_object.as_type !=
          typed_blas::kAsTypeBlas ||
      input.blas_build_generation == 0) {
    return kStatusDecodeContextMismatch;
  }

  short_stack::cross_as_input_v0 crossing = {};
  crossing.state = persistent.stack;
  crossing.tlas_instance = direct_entry_from_target(
      input.tlas_instance_target, short_stack::kDomainTlas);
  crossing.blas_root = direct_entry_from_fetch(
      input.blas_root, short_stack::kDomainBlas);
  const short_stack::cross_as_result_v0 crossed =
      short_stack::enter_blas(crossing);
  if (crossed.status != short_stack::kStatusOk) {
    return kStatusShortStackRejected;
  }
  const short_stack::pop_result_v0 popped =
      short_stack::pop_top(crossed.state);
  if (popped.status != short_stack::kStatusOk ||
      popped.entry_valid == 0 ||
      short_stack::control_domain(popped.entry.control) !=
          short_stack::kDomainBlas ||
      !selected_fetch_from_entry(
          popped.entry, input.blas_root.decode_context,
          &result->selected_fetch, &result->replay_cursor)) {
    return kStatusShortStackRejected;
  }
  persistent.stack = popped.state;
  persistent.blas_build_generation =
      input.blas_build_generation;
  result->selected_valid = 1;
  result->next_build_generation =
      input.blas_build_generation;
  return finalize_persistent_state(
      input.owner, input.region, write_seed,
      persistent, result);
}

status_kind prepare_enter_blas_transition(
    const enter_blas_input_v0 &input, result_v0 *result) {
  if (result == NULL ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  enter_blas_operands_input_v1 operands = {};
  operands.owner = input.owner;
  operands.region = input.region;
  if (short_stack_shared::decode_persistent_state(
          input.canonical_slot, input.owner,
          &operands.persistent_state) !=
      short_stack_shared::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  operands.tlas_instance_target =
      input.tlas_instance_target;
  operands.blas_root = input.blas_root;
  operands.blas_build_generation =
      input.blas_build_generation;
  return prepare_enter_blas_transition_impl(
      operands, input.canonical_slot, result);
}

status_kind prepare_enter_blas_transition_from_operands(
    const enter_blas_operands_input_v1 &input, result_v0 *result) {
  private_frontier::shadow_slot_v0 empty_write_seed = {};
  empty_write_seed.owner = input.owner;
  return prepare_enter_blas_transition_impl(
      input, empty_write_seed, result);
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusPrivateStateRejected:
      return "private_state_rejected";
    case kStatusShortStackRejected:
      return "short_stack_rejected";
    case kStatusGenerationMismatch:
      return "generation_mismatch";
    case kStatusDecodeContextMismatch:
      return "decode_context_mismatch";
    case kStatusParentLookupRequired:
      return "parent_lookup_required";
    case kStatusReturnInstanceRequired:
      return "return_instance_required";
  }
  return "unknown";
}

}  // namespace short_stack_transition
}  // namespace v04
}  // namespace rtcore
