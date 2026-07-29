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

}  // namespace

status_kind prepare_node_transition(const node_input_v0 &input,
                                    result_v0 *result) {
  if (result == NULL ||
      input.pending_parent_resume_valid > 1 ||
      input.parent_edge_valid > 1 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *result = result_v0();

  short_stack_shared::persistent_state_v0 persistent = {};
  if (short_stack_shared::decode_persistent_state(
          input.canonical_slot, input.owner, &persistent) !=
      short_stack_shared::kStatusOk) {
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

  short_stack::route_push_input_v0 push = {};
  push.state = persistent.stack;
  push.route = input.node_route;
  push.current_node_payload_offset =
      input.current_target.payload_offset;
  push.parent_resume = input.pending_parent_resume;
  push.active_domain = persistent.stack.active_domain;
  push.parent_resume_valid =
      input.pending_parent_resume_valid;
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
  } else if (persistent.stack.stack_count == 0) {
    if (!short_stack::terminal_allowed(
            persistent.stack, false, false, false)) {
      return kStatusShortStackRejected;
    }
    result->terminal = 1;
  } else {
    const bool need_parent =
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
    if (!selected_fetch_from_entry(
            popped.entry, input.current_decode_context,
            &result->selected_fetch, &result->replay_cursor)) {
      return kStatusShortStackRejected;
    }
    result->selected_valid = 1;
    result->next_build_generation = build_generation;
  }

  result->updated_slot = input.canonical_slot;
  if (short_stack_shared::apply_persistent_state(
          &result->updated_slot, input.owner, input.region,
          persistent, &result->write_plan) !=
      short_stack_shared::kStatusOk) {
    return kStatusPrivateStateRejected;
  }
  result->persistent_state = persistent;
  return kStatusOk;
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
  }
  return "unknown";
}

}  // namespace short_stack_transition
}  // namespace v04
}  // namespace rtcore
