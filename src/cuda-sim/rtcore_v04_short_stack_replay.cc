#include "rtcore_v04_short_stack_replay.h"

#include <cmath>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace short_stack {
namespace {

static const uint8_t kKindMask = 0x03u;
static const uint8_t kDomainBit = 0x04u;
static const uint8_t kAnchorValidBit = 0x08u;
static const uint8_t kInclusiveBit = 0x10u;
static const uint8_t kAnchorShift = 5u;

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static float fp32_value(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static bool expected_payload_bytes(uint8_t kind, uint16_t *bytes) {
  if (bytes == NULL) return false;
  switch (kind) {
    case typed_node::kInternalPayloadKind:
    case typed_node::kProceduralPayloadKind:
    case typed_node::kQuadPayloadKind:
      *bytes = 64;
      return true;
    case typed_node::kInstancePayloadKind:
      *bytes = 128;
      return true;
  }
  return false;
}

static bool valid_domain_payload(domain_kind domain, uint8_t kind) {
  if (domain == kDomainTlas) {
    return kind == typed_node::kInternalPayloadKind ||
           kind == typed_node::kInstancePayloadKind;
  }
  return kind == typed_node::kInternalPayloadKind ||
         kind == typed_node::kProceduralPayloadKind ||
         kind == typed_node::kQuadPayloadKind;
}

static uint8_t physical_index(const state_v0 &state,
                              uint8_t logical_index) {
  return static_cast<uint8_t>(
      (state.stack_top_ptr + logical_index) % kLogicalCapacity);
}

static entry_v0 direct_entry(
    const typed_node::compact_child_work_item_v0 &child,
    domain_kind domain) {
  entry_v0 entry = {};
  entry.payload_offset = child.payload_offset;
  entry.near_t_bits = child.near_t_bits;
  entry.payload_byte_count = child.payload_byte_count;
  entry.payload_kind = child.payload_kind;
  entry.control =
      make_control(kEntryDirectTarget, domain, child.child_slot,
                   true, false);
  return entry;
}

static bool write_logical_entries(state_v0 *state,
                                  const entry_v0 *entries,
                                  uint8_t count, bool lost) {
  if (state == NULL || count > kLogicalCapacity ||
      (count != 0 && entries == NULL)) {
    return false;
  }
  std::memset(state->entries, 0, sizeof(state->entries));
  state->stack_top_ptr = 0;
  state->stack_count = count;
  state->lost = lost ? 1 : 0;
  for (uint8_t index = 0; index < count; ++index) {
    state->entries[index] = entries[index];
  }
  return true;
}

static bool route_shape_valid(const typed_node::route_result_v0 &route,
                              domain_kind domain) {
  if (route.status != typed_node::kStatusOk) {
    return false;
  }
  if (route.result_kind == typed_node::kRouteResultMiss) {
    return route.frontier_count == 0 && route.output_valid_mask == 0;
  }
  if (route.result_kind != typed_node::kRouteResultSelected ||
      route.frontier_count > typed_node::kMaxChildren - 1 ||
      route.output_valid_mask !=
          static_cast<uint8_t>(
              typed_node::kSelectedFetchValid |
              (route.frontier_count != 0
                   ? typed_node::kFrontierItemsValid
                   : 0)) ||
      route.selected_fetch.decode_context.as_object.as_type !=
          (domain == kDomainTlas ? 1 : typed_blas::kAsTypeBlas)) {
    return false;
  }
  return true;
}

static uint8_t ordinary_count(const state_v0 &state) {
  return static_cast<uint8_t>(
      state.stack_count - (state.cross_as != 0 ? 1 : 0));
}

static bool append_active_ordinary_entries(
    const state_v0 &state, entry_v0 *destination, uint8_t *count) {
  if (destination == NULL || count == NULL) return false;
  const uint8_t active_count = ordinary_count(state);
  for (uint8_t index = 0; index < active_count; ++index) {
    if (!read_logical_entry(state, index, &destination[*count])) {
      return false;
    }
    ++*count;
  }
  return true;
}

static bool read_protected_return(const state_v0 &state,
                                  entry_v0 *entry) {
  if (entry == NULL || state.cross_as == 0 ||
      state.stack_count == 0) {
    return false;
  }
  return read_logical_entry(
      state, static_cast<uint8_t>(state.stack_count - 1), entry) &&
         control_kind(entry->control) == kEntryCrossAsReturn;
}

}  // namespace

static_assert(sizeof(entry_v0) == 16,
              "Simulator short-stack entry must remain 16 bytes");
static_assert(sizeof(parent_edge_v0) == 16,
              "Simulator replay parent edge must remain 16 bytes");

uint8_t make_control(entry_kind kind, domain_kind domain,
                     uint8_t child_anchor, bool anchor_valid,
                     bool inclusive) {
  if (kind > kEntryCrossAsReturn || domain > kDomainBlas ||
      child_anchor >= typed_node::kMaxChildren) {
    return 0xffu;
  }
  return static_cast<uint8_t>(
      static_cast<uint8_t>(kind) |
      (domain == kDomainBlas ? kDomainBit : 0u) |
      (anchor_valid ? kAnchorValidBit : 0u) |
      (inclusive ? kInclusiveBit : 0u) |
      static_cast<uint8_t>(child_anchor << kAnchorShift));
}

entry_kind control_kind(uint8_t control) {
  return static_cast<entry_kind>(control & kKindMask);
}

domain_kind control_domain(uint8_t control) {
  return (control & kDomainBit) != 0 ? kDomainBlas : kDomainTlas;
}

uint8_t control_child_anchor(uint8_t control) {
  return static_cast<uint8_t>(control >> kAnchorShift);
}

bool control_anchor_valid(uint8_t control) {
  return (control & kAnchorValidBit) != 0;
}

bool control_inclusive(uint8_t control) {
  return (control & kInclusiveBit) != 0;
}

bool validate_entry(const entry_v0 &entry) {
  const entry_kind kind = control_kind(entry.control);
  const domain_kind domain = control_domain(entry.control);
  const uint8_t anchor = control_child_anchor(entry.control);
  const bool anchor_valid = control_anchor_valid(entry.control);
  const bool inclusive = control_inclusive(entry.control);
  uint16_t expected_bytes = 0;
  if (kind > kEntryCrossAsReturn || domain > kDomainBlas ||
      anchor >= typed_node::kMaxChildren ||
      entry.payload_offset == 0 ||
      (entry.payload_offset & uint64_t{0x3f}) != 0 ||
      !expected_payload_bytes(entry.payload_kind, &expected_bytes) ||
      entry.payload_byte_count != expected_bytes ||
      !valid_domain_payload(domain, entry.payload_kind)) {
    return false;
  }
  if (kind == kEntryDirectTarget) {
    return anchor_valid && !inclusive &&
           std::isfinite(fp32_value(entry.near_t_bits));
  }
  if (kind == kEntrySameNodeReplay ||
      kind == kEntryParentResume) {
    return entry.payload_kind == typed_node::kInternalPayloadKind &&
           anchor_valid && std::isfinite(fp32_value(entry.near_t_bits));
  }
  return domain == kDomainTlas &&
         entry.payload_kind == typed_node::kInstancePayloadKind &&
         !anchor_valid && !inclusive && entry.near_t_bits == 0;
}

bool validate_state(const state_v0 &state) {
  if (state.stack_count > kLogicalCapacity ||
      state.stack_top_ptr >= kLogicalCapacity ||
      state.cross_as > 1 || state.lost > 1 ||
      state.active_domain > kDomainBlas ||
      !bytes_are_zero(state.reserved_zero,
                      sizeof(state.reserved_zero))) {
    return false;
  }
  if (state.cross_as !=
      (state.active_domain == kDomainBlas ? 1 : 0)) {
    return false;
  }
  unsigned return_count = 0;
  for (uint8_t index = 0; index < state.stack_count; ++index) {
    entry_v0 entry = {};
    if (!read_logical_entry(state, index, &entry) ||
        !validate_entry(entry)) {
      return false;
    }
    if (control_kind(entry.control) == kEntryCrossAsReturn) {
      ++return_count;
      if (state.cross_as == 0 ||
          index != state.stack_count - 1) {
        return false;
      }
    } else if (control_domain(entry.control) !=
               static_cast<domain_kind>(state.active_domain)) {
      return false;
    }
  }
  if (state.cross_as != 0) {
    return state.stack_count >= 1 && return_count == 1;
  }
  return return_count == 0;
}

bool read_logical_entry(const state_v0 &state, uint8_t logical_index,
                        entry_v0 *entry) {
  if (entry == NULL || logical_index >= state.stack_count ||
      state.stack_top_ptr >= kLogicalCapacity) {
    return false;
  }
  *entry = state.entries[physical_index(state, logical_index)];
  return true;
}

route_push_result_v0 push_node_route(
    const route_push_input_v0 &input) {
  route_push_result_v0 result = {};
  result.status = kStatusInvalidArgument;
  if (!validate_state(input.state) ||
      input.active_domain > kDomainBlas ||
      input.active_domain != input.state.active_domain ||
      input.current_node_payload_offset == 0 ||
      (input.current_node_payload_offset & uint64_t{0x3f}) != 0 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    result.status = kStatusInvalidState;
    return result;
  }
  const domain_kind domain =
      static_cast<domain_kind>(input.active_domain);
  if (!route_shape_valid(input.route, domain)) {
    result.status = kStatusInvalidRoute;
    return result;
  }

  const bool selected_valid =
      input.route.result_kind == typed_node::kRouteResultSelected;
  typed_node::compact_child_work_item_v0 selected_child = {};
  if (selected_valid) {
    selected_child = input.route.selected_fetch.child;
    if (!validate_entry(direct_entry(selected_child, domain))) {
      result.status = kStatusInvalidRoute;
      return result;
    }
    for (uint8_t index = 0; index < input.route.frontier_count;
         ++index) {
      if (!validate_entry(direct_entry(input.route.frontier[index],
                                       domain))) {
        result.status = kStatusInvalidRoute;
        return result;
      }
    }
  }
  if (input.parent_resume_valid > 1 ||
      (input.parent_resume_valid != 0 &&
       (ordinary_count(input.state) != 0 ||
        input.state.lost == 0 ||
        !validate_entry(input.parent_resume) ||
        control_kind(input.parent_resume.control) !=
            kEntryParentResume ||
        control_domain(input.parent_resume.control) != domain))) {
    result.status = kStatusInvalidEntry;
    return result;
  }

  entry_v0 merged[kLogicalCapacity +
                  typed_node::kMaxChildren] = {};
  uint8_t merged_count = 0;
  const uint8_t active_count = ordinary_count(input.state);
  const uint8_t ordinary_capacity = static_cast<uint8_t>(
      kLogicalCapacity - (input.state.cross_as != 0 ? 1 : 0));
  const bool direct_remainder_fits =
      static_cast<unsigned>(active_count) +
          input.route.frontier_count +
          input.parent_resume_valid <= ordinary_capacity;
  if (selected_valid && direct_remainder_fits) {
    for (uint8_t index = 0; index < input.route.frontier_count;
         ++index) {
      merged[merged_count++] =
          direct_entry(input.route.frontier[index], domain);
    }
  } else if (selected_valid && input.route.frontier_count != 0) {
    entry_v0 replay = {};
    replay.payload_offset = input.current_node_payload_offset;
    replay.near_t_bits = selected_child.near_t_bits;
    replay.payload_byte_count = 64;
    replay.payload_kind = typed_node::kInternalPayloadKind;
    replay.control =
        make_control(kEntrySameNodeReplay, domain,
                     selected_child.child_slot, true, false);
    if (!validate_entry(replay)) {
      result.status = kStatusInvalidRoute;
      return result;
    }
    merged[merged_count++] = replay;
    result.compressed_to_replay = 1;
  }
  if (input.parent_resume_valid != 0) {
    if (merged_count < ordinary_capacity) {
      merged[merged_count++] = input.parent_resume;
      result.parent_resume_appended = 1;
    } else {
      result.parent_resume_deferred = 1;
    }
  }
  if (!append_active_ordinary_entries(input.state, merged,
                                      &merged_count)) {
    result.status = kStatusInvalidState;
    return result;
  }

  const bool overflowed =
      merged_count > ordinary_capacity ||
      result.parent_resume_deferred != 0;
  const uint8_t retained_count =
      merged_count > ordinary_capacity ? ordinary_capacity
                                       : merged_count;
  entry_v0 retained[kLogicalCapacity] = {};
  for (uint8_t index = 0; index < retained_count; ++index) {
    retained[index] = merged[index];
  }
  uint8_t final_count = retained_count;
  if (input.state.cross_as != 0) {
    entry_v0 return_entry = {};
    if (!read_protected_return(input.state, &return_entry)) {
      result.status = kStatusInvalidState;
      return result;
    }
    retained[final_count++] = return_entry;
  }
  result.state = input.state;
  if (!write_logical_entries(
          &result.state, retained, final_count,
          input.state.lost != 0 || overflowed)) {
    result.status = kStatusInvalidState;
    return result;
  }
  result.status = kStatusOk;
  result.selected_valid = selected_valid ? 1 : 0;
  result.overflowed_bottom = overflowed ? 1 : 0;
  if (selected_valid) result.selected = input.route.selected_fetch;
  return result;
}

pop_result_v0 pop_top(const state_v0 &state) {
  pop_result_v0 result = {};
  result.status = kStatusInvalidArgument;
  if (!validate_state(state) || state.stack_count == 0) {
    result.status = kStatusInvalidState;
    return result;
  }
  if (!read_logical_entry(state, 0, &result.entry)) {
    result.status = kStatusInvalidEntry;
    return result;
  }
  result.state = state;
  std::memset(
      &result.state.entries[result.state.stack_top_ptr], 0,
      sizeof(result.state.entries[result.state.stack_top_ptr]));
  result.state.stack_top_ptr = static_cast<uint8_t>(
      (result.state.stack_top_ptr + 1) % kLogicalCapacity);
  --result.state.stack_count;
  if (result.state.stack_count == 0) {
    result.state.stack_top_ptr = 0;
  }
  result.status = kStatusOk;
  result.entry_valid = 1;
  return result;
}

bool parent_bailout_required(const state_v0 &state) {
  if (!validate_state(state) || state.lost == 0) return false;
  const uint8_t active_count = static_cast<uint8_t>(
      state.stack_count - (state.cross_as != 0 ? 1 : 0));
  if (active_count != 1) return false;
  entry_v0 top = {};
  if (!read_logical_entry(state, 0, &top)) return false;
  const entry_kind kind = control_kind(top.control);
  return top.payload_kind == typed_node::kInternalPayloadKind &&
         (kind == kEntryDirectTarget ||
          kind == kEntrySameNodeReplay ||
          kind == kEntryParentResume);
}

parent_bailout_result_v0 prepare_parent_bailout(
    const pop_result_v0 &popped, const parent_edge_v0 &parent) {
  parent_bailout_result_v0 result = {};
  result.status = kStatusInvalidArgument;
  if (popped.status != kStatusOk || popped.entry_valid == 0 ||
      !validate_state(popped.state) || popped.state.lost == 0 ||
      ordinary_count(popped.state) != 0 ||
      !validate_entry(popped.entry) ||
      popped.entry.payload_kind !=
          typed_node::kInternalPayloadKind ||
      !bytes_are_zero(parent.reserved_zero,
                      sizeof(parent.reserved_zero)) ||
      parent.root > 1) {
    return result;
  }
  result.state = popped.state;
  if (parent.root != 0) {
    result.state.lost = 0;
    result.root_replay = 1;
    result.status = kStatusOk;
    return result;
  }
  if (parent.parent_payload_offset == 0 ||
      (parent.parent_payload_offset & uint64_t{0x3f}) != 0 ||
      parent.parent_child_slot >= typed_node::kMaxChildren) {
    result.status = kStatusInvalidParentEdge;
    return result;
  }
  result.parent_resume.payload_offset =
      parent.parent_payload_offset;
  result.parent_resume.near_t_bits = popped.entry.near_t_bits;
  result.parent_resume.payload_byte_count = 64;
  result.parent_resume.payload_kind =
      typed_node::kInternalPayloadKind;
  result.parent_resume.control =
      make_control(kEntryParentResume,
                   static_cast<domain_kind>(
                       popped.state.active_domain),
                   parent.parent_child_slot, true, false);
  if (!validate_entry(result.parent_resume)) {
    result.status = kStatusInvalidParentEdge;
    return result;
  }
  result.parent_resume_valid = 1;
  result.status = kStatusOk;
  return result;
}

cross_as_result_v0 enter_blas(const cross_as_input_v0 &input) {
  cross_as_result_v0 result = {};
  result.status = kStatusInvalidArgument;
  if (!validate_state(input.state) || input.state.cross_as != 0 ||
      input.state.active_domain != kDomainTlas ||
      !validate_entry(input.tlas_instance) ||
      control_kind(input.tlas_instance.control) !=
          kEntryDirectTarget ||
      control_domain(input.tlas_instance.control) != kDomainTlas ||
      input.tlas_instance.payload_kind !=
          typed_node::kInstancePayloadKind ||
      !validate_entry(input.blas_root) ||
      control_kind(input.blas_root.control) !=
          kEntryDirectTarget ||
      control_domain(input.blas_root.control) != kDomainBlas) {
    result.status = kStatusCrossAsInvariant;
    return result;
  }
  entry_v0 logical[2] = {};
  logical[0] = input.blas_root;
  logical[1] = input.tlas_instance;
  logical[1].near_t_bits = 0;
  logical[1].control =
      make_control(kEntryCrossAsReturn, kDomainTlas, 0, false,
                   false);
  result.state = state_v0();
  result.state.cross_as = 1;
  result.state.active_domain = kDomainBlas;
  if (!write_logical_entries(&result.state, logical, 2, false) ||
      !validate_state(result.state)) {
    result.status = kStatusCrossAsInvariant;
    return result;
  }
  result.status = kStatusOk;
  return result;
}

status_kind return_to_tlas(state_v0 *state,
                           const parent_edge_v0 &instance_parent) {
  if (state == NULL || !validate_state(*state) ||
      state->cross_as == 0 || state->active_domain != kDomainBlas ||
      state->lost != 0 || state->stack_count != 1 ||
      !bytes_are_zero(instance_parent.reserved_zero,
                      sizeof(instance_parent.reserved_zero)) ||
      instance_parent.root > 1) {
    return kStatusCrossAsInvariant;
  }
  entry_v0 return_entry = {};
  if (!read_logical_entry(*state, 0, &return_entry) ||
      control_kind(return_entry.control) !=
          kEntryCrossAsReturn) {
    return kStatusCrossAsInvariant;
  }
  entry_v0 logical[1] = {};
  uint8_t count = 0;
  bool lost = false;
  if (instance_parent.root == 0) {
    if (instance_parent.parent_payload_offset == 0 ||
        (instance_parent.parent_payload_offset &
         uint64_t{0x3f}) != 0 ||
        instance_parent.parent_child_slot >=
            typed_node::kMaxChildren) {
      return kStatusInvalidParentEdge;
    }
    logical[0].payload_offset =
        instance_parent.parent_payload_offset;
    logical[0].near_t_bits = 0;
    logical[0].payload_byte_count = 64;
    logical[0].payload_kind =
        typed_node::kInternalPayloadKind;
    logical[0].control =
        make_control(kEntryParentResume, kDomainTlas,
                     instance_parent.parent_child_slot, true, false);
    count = 1;
    lost = true;
  }
  state->cross_as = 0;
  state->active_domain = kDomainTlas;
  if (!write_logical_entries(state, logical, count, lost) ||
      !validate_state(*state)) {
    return kStatusCrossAsInvariant;
  }
  return kStatusOk;
}

bool terminal_allowed(const state_v0 &state, bool pending_candidate,
                      bool pending_memory, bool pending_commit) {
  return validate_state(state) && state.stack_count == 0 &&
         state.cross_as == 0 && state.lost == 0 &&
         !pending_candidate && !pending_memory && !pending_commit;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidState:
      return "invalid_state";
    case kStatusInvalidEntry:
      return "invalid_entry";
    case kStatusInvalidRoute:
      return "invalid_route";
    case kStatusInvalidParentEdge:
      return "invalid_parent_edge";
    case kStatusCrossAsInvariant:
      return "cross_as_invariant";
  }
  return "unknown";
}

}  // namespace short_stack
}  // namespace v04
}  // namespace rtcore
