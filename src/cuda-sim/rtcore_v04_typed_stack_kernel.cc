#include "rtcore_v04_typed_stack_kernel.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace typed_stack {
namespace {

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

static bool expected_child_layout(uint8_t kind, uint16_t *bytes) {
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

static bool child_kind_allowed_for_as_type(uint8_t as_type, uint8_t kind) {
  if (as_type == 1) {
    return kind == typed_node::kInternalPayloadKind ||
           kind == typed_node::kInstancePayloadKind;
  }
  if (as_type == typed_blas::kAsTypeBlas) {
    return kind == typed_node::kInternalPayloadKind ||
           kind == typed_node::kProceduralPayloadKind ||
           kind == typed_node::kQuadPayloadKind;
  }
  return false;
}

static bool valid_decode_context(
    const typed_blas::as_decode_context_v0 &context) {
  return context.bvh_format_profile_id == kGenRtDerivedProfileId &&
         context.reserved_zero == 0 && context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         (context.as_object.as_type == 1 ||
          context.as_object.as_type == typed_blas::kAsTypeBlas) &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         context.device_base != 0 && context.device_range_bytes >= 64 &&
         context.device_range_bytes <=
             std::numeric_limits<uint64_t>::max() - context.device_base;
}

static bool valid_child_item(
    const typed_node::compact_child_work_item_v0 &item,
    const typed_blas::as_decode_context_v0 &context) {
  uint16_t expected_bytes = 0;
  const float near_t = fp32_value(item.near_t_bits);
  return expected_child_layout(item.payload_kind, &expected_bytes) &&
         child_kind_allowed_for_as_type(context.as_object.as_type,
                                        item.payload_kind) &&
         item.payload_byte_count == expected_bytes && item.child_slot < 6 &&
         (item.payload_offset & uint64_t{0x3f}) == 0 &&
         item.payload_offset <= context.device_range_bytes &&
         static_cast<uint64_t>(item.payload_byte_count) <=
             context.device_range_bytes - item.payload_offset &&
         std::isfinite(near_t);
}

static bool item_precedes(
    const typed_node::compact_child_work_item_v0 &lhs,
    const typed_node::compact_child_work_item_v0 &rhs) {
  const float lhs_near = fp32_value(lhs.near_t_bits);
  const float rhs_near = fp32_value(rhs.near_t_bits);
  return lhs_near < rhs_near ||
         (lhs_near == rhs_near && lhs.child_slot < rhs.child_slot);
}

static bool checked_add_u32(uint32_t lhs, uint32_t rhs,
                            uint32_t *result) {
  if (result == NULL || lhs > std::numeric_limits<uint32_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

static bool valid_committed_hit(
    const committed_hit_projection_v0 &hit) {
  if (hit.valid > 1 || hit.attribute_word_count > 4 ||
      !bytes_are_zero(hit.reserved_zero0,
                      sizeof(hit.reserved_zero0)) ||
      hit.reserved_zero1 != 0) {
    return false;
  }
  if (hit.valid == 0) {
    const committed_hit_projection_v0 empty = {};
    return std::memcmp(&hit, &empty, sizeof(empty)) == 0;
  }
  return std::isfinite(hit.hit_t);
}

static bool valid_parent_frame(
    const traversal_frame_projection_v0 &frame,
    uint32_t frontier_capacity) {
  const mutable_ray_state_v0 &ray = frame.ray;
  for (unsigned component = 0; component < 3; ++component) {
    if (!std::isfinite(ray.origin[component]) ||
        !std::isfinite(ray.direction[component])) {
      return false;
    }
  }
  return std::isfinite(ray.t_min) && std::isfinite(ray.t_max) &&
         ray.t_min <= ray.t_max && frame.traversal_level == 0 &&
         frame.frontier_marker.frontier_top ==
             frame.frontier_marker.frontier_count &&
         frame.frontier_marker.frontier_top <= frontier_capacity &&
         frame.frontier_marker.level_frame_depth == 0 &&
         frame.frontier_marker.reserved_zero == 0 &&
         valid_decode_context(frame.current_decode_context) &&
         frame.current_decode_context.as_object.as_type == 1 &&
         bytes_are_zero(frame.current_instance.reserved_zero,
                        sizeof(frame.current_instance.reserved_zero));
}

}  // namespace

push_result_v0 execute_push(const push_input_v0 &input) {
  push_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (input.operation_kind != kPushRemainderAndForwardSelected ||
      !bytes_are_zero(input.reserved_zero0,
                      sizeof(input.reserved_zero0)) ||
      !bytes_are_zero(input.reserved_zero1,
                      sizeof(input.reserved_zero1))) {
    return result;
  }
  if (input.frontier.frontier_top != input.frontier.frontier_count ||
      input.frontier.frontier_top > input.frontier.frontier_capacity ||
      input.frontier.frontier_count > input.frontier.frontier_capacity) {
    result.status = kStatusInvalidFrontierMetadata;
    return result;
  }

  const typed_node::route_result_v0 &route = input.node_route;
  const uint8_t required_route_mask =
      static_cast<uint8_t>(typed_node::kSelectedFetchValid |
                           typed_node::kFrontierItemsValid);
  if (route.status != typed_node::kStatusOk ||
      route.result_kind != typed_node::kRouteResultSelected ||
      route.output_valid_mask != required_route_mask ||
      route.frontier_count == 0 ||
      route.frontier_count > kMaxRemainderChildren ||
      !bytes_are_zero(route.reserved_zero, sizeof(route.reserved_zero)) ||
      !bytes_are_zero(route.reserved_zero_tail,
                      sizeof(route.reserved_zero_tail))) {
    result.status = kStatusInvalidRoutePacket;
    return result;
  }

  const float traversal_bound =
      fp32_value(input.current_traversal_bound_bits);
  if (!std::isfinite(traversal_bound)) {
    result.status = kStatusInvalidTraversalBound;
    return result;
  }
  if (!valid_decode_context(route.selected_fetch.decode_context) ||
      !valid_child_item(route.selected_fetch.child,
                        route.selected_fetch.decode_context) ||
      fp32_value(route.selected_fetch.child.near_t_bits) > traversal_bound) {
    result.status = kStatusInvalidWorkItem;
    return result;
  }

  typed_node::compact_child_work_item_v0
      survivors[kMaxRemainderChildren] = {};
  unsigned survivor_count = 0;
  uint8_t seen_child_slots =
      static_cast<uint8_t>(1u << route.selected_fetch.child.child_slot);
  typed_node::compact_child_work_item_v0 previous =
      route.selected_fetch.child;
  for (unsigned index = 0; index < route.frontier_count; ++index) {
    const typed_node::compact_child_work_item_v0 &item =
        route.frontier[index];
    if (!valid_child_item(item, route.selected_fetch.decode_context) ||
        (seen_child_slots & (1u << item.child_slot)) != 0) {
      result.status = kStatusInvalidWorkItem;
      return result;
    }
    seen_child_slots |= static_cast<uint8_t>(1u << item.child_slot);
    if (!item_precedes(previous, item)) {
      result.status = kStatusInvalidRemainderOrder;
      return result;
    }
    previous = item;
    if (fp32_value(item.near_t_bits) <= traversal_bound) {
      survivors[survivor_count++] = item;
    }
  }
  for (unsigned index = route.frontier_count;
       index < kMaxRemainderChildren; ++index) {
    const typed_node::compact_child_work_item_v0 empty = {};
    if (std::memcmp(&route.frontier[index], &empty, sizeof(empty)) != 0) {
      result.status = kStatusInvalidRoutePacket;
      return result;
    }
  }

  uint32_t new_top = 0;
  uint32_t new_count = 0;
  if (!checked_add_u32(input.frontier.frontier_top, survivor_count,
                       &new_top) ||
      !checked_add_u32(input.frontier.frontier_count, survivor_count,
                       &new_count) ||
      new_top > input.frontier.frontier_capacity ||
      new_count > input.frontier.frontier_capacity) {
    result.status = kStatusFrontierCapacityExceeded;
    return result;
  }

  result.status = kStatusOk;
  result.result_kind = kStackPushedAndSelected;
  result.output_valid_mask =
      static_cast<uint8_t>(kFrontierDeltaValid | kSelectedFetchValid);
  result.pruned_count =
      static_cast<uint8_t>(route.frontier_count - survivor_count);
  result.frontier_delta.action = kFrontierActionAppendChildren;
  result.frontier_delta.write_count =
      static_cast<uint8_t>(survivor_count);
  result.frontier_delta.append_base_index = input.frontier.frontier_top;
  result.frontier_delta.new_frontier_top = new_top;
  result.frontier_delta.new_frontier_count = new_count;
  for (unsigned index = 0; index < survivor_count; ++index) {
    result.frontier_delta.written_items[index] =
        survivors[survivor_count - index - 1];
  }
  result.selected_fetch = route.selected_fetch;
  return result;
}

bool validate_push_result(const push_result_v0 &result) {
  const uint8_t expected_valid =
      static_cast<uint8_t>(kFrontierDeltaValid | kSelectedFetchValid);
  const frontier_append_delta_v0 &delta = result.frontier_delta;
  if (result.status != kStatusOk ||
      result.result_kind != kStackPushedAndSelected ||
      result.output_valid_mask != expected_valid ||
      result.pruned_count > kMaxRemainderChildren ||
      !bytes_are_zero(result.reserved_zero0,
                      sizeof(result.reserved_zero0)) ||
      !bytes_are_zero(result.reserved_zero_tail,
                      sizeof(result.reserved_zero_tail)) ||
      delta.action != kFrontierActionAppendChildren ||
      delta.write_count > kMaxRemainderChildren ||
      delta.reserved_zero != 0) {
    return false;
  }

  const unsigned route_remainder_count =
      static_cast<unsigned>(delta.write_count) + result.pruned_count;
  if (route_remainder_count == 0 ||
      route_remainder_count > kMaxRemainderChildren) {
    return false;
  }

  uint32_t expected_top = 0;
  if (!checked_add_u32(delta.append_base_index, delta.write_count,
                       &expected_top) ||
      delta.new_frontier_top != expected_top ||
      delta.new_frontier_count != expected_top) {
    return false;
  }

  const typed_node::selected_child_fetch_work_item_v0 &selected =
      result.selected_fetch;
  if (!valid_decode_context(selected.decode_context) ||
      !valid_child_item(selected.child, selected.decode_context)) {
    return false;
  }

  uint8_t seen_child_slots =
      static_cast<uint8_t>(1u << selected.child.child_slot);
  for (unsigned index = 0; index < delta.write_count; ++index) {
    const typed_node::compact_child_work_item_v0 &item =
        delta.written_items[index];
    if (!valid_child_item(item, selected.decode_context) ||
        (seen_child_slots & (1u << item.child_slot)) != 0 ||
        !item_precedes(selected.child, item)) {
      return false;
    }
    if (index != 0 &&
        !item_precedes(item, delta.written_items[index - 1])) {
      return false;
    }
    seen_child_slots |= static_cast<uint8_t>(1u << item.child_slot);
  }
  for (unsigned index = delta.write_count;
       index < kMaxRemainderChildren; ++index) {
    const typed_node::compact_child_work_item_v0 zero = {};
    if (std::memcmp(&delta.written_items[index], &zero, sizeof(zero)) != 0) {
      return false;
    }
  }
  return true;
}

pop_result_v0 execute_pop(const pop_input_v0 &input) {
  pop_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (input.operation_kind != kPopNext || input.has_top_entry > 1 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return result;
  }
  if (input.has_top_entry == 0) {
    result.status = kStatusMissingFrontierOperand;
    return result;
  }
  if (input.frontier.frontier_top == 0 ||
      input.frontier.frontier_count == 0 ||
      input.frontier.frontier_top != input.frontier.frontier_count ||
      input.frontier.frontier_top > input.frontier.frontier_capacity ||
      input.frontier.frontier_count > input.frontier.frontier_capacity) {
    result.status = kStatusInvalidFrontierMetadata;
    return result;
  }

  const float traversal_bound =
      fp32_value(input.current_traversal_bound_bits);
  if (!std::isfinite(traversal_bound)) {
    result.status = kStatusInvalidTraversalBound;
    return result;
  }
  if (!valid_decode_context(input.current_decode_context) ||
      !valid_child_item(input.top_entry, input.current_decode_context)) {
    result.status = kStatusInvalidWorkItem;
    return result;
  }

  result.status = kStatusOk;
  result.output_valid_mask = kFrontierDeltaValid;
  result.frontier_delta.action = kFrontierActionPopChild;
  result.frontier_delta.pop_count = 1;
  result.frontier_delta.popped_index = input.frontier.frontier_top - 1;
  result.frontier_delta.new_frontier_top =
      input.frontier.frontier_top - 1;
  result.frontier_delta.new_frontier_count =
      input.frontier.frontier_count - 1;

  if (fp32_value(input.top_entry.near_t_bits) > traversal_bound) {
    result.result_kind = kStackPrunedRetryPop;
    return result;
  }

  result.result_kind = kStackSelectedNext;
  result.output_valid_mask =
      static_cast<uint8_t>(kFrontierDeltaValid | kSelectedFetchValid);
  result.selected_fetch.child = input.top_entry;
  result.selected_fetch.decode_context = input.current_decode_context;
  return result;
}

bool validate_pop_result(const pop_input_v0 &input,
                         const pop_result_v0 &result) {
  if (input.profile_id != kGenRtDerivedProfileId ||
      input.operation_kind != kPopNext || input.has_top_entry != 1 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero)) ||
      input.frontier.frontier_top == 0 ||
      input.frontier.frontier_count == 0 ||
      input.frontier.frontier_top != input.frontier.frontier_count ||
      input.frontier.frontier_top > input.frontier.frontier_capacity ||
      !std::isfinite(fp32_value(input.current_traversal_bound_bits)) ||
      !valid_decode_context(input.current_decode_context) ||
      !valid_child_item(input.top_entry,
                        input.current_decode_context) ||
      result.status != kStatusOk ||
      !bytes_are_zero(result.reserved_zero0,
                      sizeof(result.reserved_zero0)) ||
      !bytes_are_zero(result.reserved_zero_tail,
                      sizeof(result.reserved_zero_tail)) ||
      result.frontier_delta.action != kFrontierActionPopChild ||
      result.frontier_delta.pop_count != 1 ||
      result.frontier_delta.reserved_zero != 0 ||
      result.frontier_delta.popped_index !=
          input.frontier.frontier_top - 1 ||
      result.frontier_delta.new_frontier_top !=
          input.frontier.frontier_top - 1 ||
      result.frontier_delta.new_frontier_count !=
          input.frontier.frontier_count - 1) {
    return false;
  }

  const bool selected =
      fp32_value(input.top_entry.near_t_bits) <=
      fp32_value(input.current_traversal_bound_bits);
  const uint8_t expected_mask = static_cast<uint8_t>(
      kFrontierDeltaValid |
      (selected ? kSelectedFetchValid : 0));
  if (result.result_kind !=
          (selected ? kStackSelectedNext
                    : kStackPrunedRetryPop) ||
      result.output_valid_mask != expected_mask) {
    return false;
  }
  typed_node::selected_child_fetch_work_item_v0 expected_selected = {};
  if (selected) {
    expected_selected.child = input.top_entry;
    expected_selected.decode_context =
        input.current_decode_context;
  }
  return std::memcmp(&result.selected_fetch, &expected_selected,
                     sizeof(expected_selected)) == 0;
}

empty_result_v0 execute_empty(const empty_input_v0 &input) {
  empty_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if (input.operation_kind != kPopNext ||
      input.parent_frame_available > 1 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return result;
  }
  if (input.frontier.frontier_top !=
          input.frontier.frontier_count ||
      input.frontier.frontier_capacity != 16 ||
      input.frontier.max_level_depth != 1 ||
      input.frontier.level_frame_depth >
          input.frontier.max_level_depth) {
    result.status = kStatusInvalidFrontierMetadata;
    return result;
  }
  if (!valid_committed_hit(input.current_committed_hit)) {
    result.status = kStatusInvalidCommittedHit;
    return result;
  }

  if (input.frontier.current_level == 1 &&
      input.frontier.level_frame_depth == 1) {
    if (input.parent_frame_available != 1 ||
        !valid_parent_frame(input.parent_frame,
                            input.frontier.frontier_capacity) ||
        input.frontier.frontier_top !=
            input.parent_frame.frontier_marker.frontier_top ||
        input.frontier.frontier_count !=
            input.parent_frame.frontier_marker.frontier_count) {
      result.status = kStatusInvalidParentFrame;
      return result;
    }
    result.status = kStatusOk;
    result.result_kind = kStackRestoreParent;
    result.output_valid_mask =
        static_cast<uint8_t>(kFrontierDeltaValid |
                             kParentFrameValid);
    result.frontier_delta.action = kFrontierActionPopFrame;
    result.frontier_delta.new_frontier_top =
        input.parent_frame.frontier_marker.frontier_top;
    result.frontier_delta.new_frontier_count =
        input.parent_frame.frontier_marker.frontier_count;
    result.frontier_delta.new_current_level =
        input.parent_frame.traversal_level;
    result.frontier_delta.new_level_frame_depth =
        input.parent_frame.frontier_marker.level_frame_depth;
    result.frontier_delta.max_level_depth =
        input.frontier.max_level_depth;
    result.parent_frame = input.parent_frame;
    return result;
  }

  if (input.frontier.current_level != 0 ||
      input.frontier.level_frame_depth != 0 ||
      input.parent_frame_available != 0 ||
      input.frontier.frontier_top != 0 ||
      input.frontier.frontier_count != 0) {
    result.status = kStatusInvalidLevelTransition;
    return result;
  }

  result.status = kStatusOk;
  if (input.current_committed_hit.valid != 0) {
    result.result_kind = kStackFinalHit;
    result.output_valid_mask = kTerminalHitValid;
    result.terminal_hit = input.current_committed_hit;
  } else {
    result.result_kind = kStackFinalMiss;
  }
  return result;
}

bool validate_empty_result(const empty_input_v0 &input,
                           const empty_result_v0 &result) {
  if (input.profile_id != kGenRtDerivedProfileId ||
      input.operation_kind != kPopNext ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero)) ||
      input.frontier.frontier_top !=
          input.frontier.frontier_count ||
      input.frontier.frontier_capacity != 16 ||
      input.frontier.max_level_depth != 1 ||
      input.frontier.level_frame_depth >
          input.frontier.max_level_depth ||
      !valid_committed_hit(input.current_committed_hit) ||
      result.status != kStatusOk ||
      !bytes_are_zero(result.reserved_zero0,
                      sizeof(result.reserved_zero0)) ||
      !bytes_are_zero(result.reserved_zero_tail,
                      sizeof(result.reserved_zero_tail))) {
    return false;
  }

  if (input.frontier.current_level == 1 &&
      input.frontier.level_frame_depth == 1) {
    return input.parent_frame_available == 1 &&
           valid_parent_frame(input.parent_frame,
                              input.frontier.frontier_capacity) &&
           input.frontier.frontier_top ==
               input.parent_frame.frontier_marker.frontier_top &&
           input.frontier.frontier_count ==
               input.parent_frame.frontier_marker.frontier_count &&
           result.result_kind == kStackRestoreParent &&
           result.output_valid_mask ==
               static_cast<uint8_t>(kFrontierDeltaValid |
                                    kParentFrameValid) &&
           result.frontier_delta.action ==
               kFrontierActionPopFrame &&
           bytes_are_zero(result.frontier_delta.reserved_zero,
                          sizeof(result.frontier_delta.reserved_zero)) &&
           result.frontier_delta.new_frontier_top ==
               input.parent_frame.frontier_marker.frontier_top &&
           result.frontier_delta.new_frontier_count ==
               input.parent_frame.frontier_marker.frontier_count &&
           result.frontier_delta.new_current_level ==
               input.parent_frame.traversal_level &&
           result.frontier_delta.new_level_frame_depth ==
               input.parent_frame.frontier_marker.level_frame_depth &&
           result.frontier_delta.max_level_depth ==
               input.frontier.max_level_depth &&
           std::memcmp(&result.parent_frame, &input.parent_frame,
                       sizeof(result.parent_frame)) == 0 &&
           result.terminal_hit.valid == 0;
  }

  const frontier_level_delta_v0 empty_delta = {};
  const traversal_frame_projection_v0 empty_frame = {};
  const bool terminal_common =
      input.frontier.current_level == 0 &&
      input.frontier.level_frame_depth == 0 &&
      input.parent_frame_available == 0 &&
      valid_committed_hit(input.current_committed_hit) &&
      std::memcmp(&result.frontier_delta, &empty_delta,
                  sizeof(empty_delta)) == 0 &&
      std::memcmp(&result.parent_frame, &empty_frame,
                  sizeof(empty_frame)) == 0;
  if (!terminal_common) return false;
  if (input.current_committed_hit.valid != 0) {
    return result.result_kind == kStackFinalHit &&
           result.output_valid_mask == kTerminalHitValid &&
           std::memcmp(&result.terminal_hit,
                       &input.current_committed_hit,
                       sizeof(result.terminal_hit)) == 0;
  }
  const committed_hit_projection_v0 empty_hit = {};
  return result.result_kind == kStackFinalMiss &&
         result.output_valid_mask == 0 &&
         std::memcmp(&result.terminal_hit, &empty_hit,
                     sizeof(empty_hit)) == 0;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusUnsupportedProfile:
      return "unsupported_profile";
    case kStatusInvalidFrontierMetadata:
      return "invalid_frontier_metadata";
    case kStatusInvalidRoutePacket:
      return "invalid_route_packet";
    case kStatusInvalidTraversalBound:
      return "invalid_traversal_bound";
    case kStatusInvalidWorkItem:
      return "invalid_work_item";
    case kStatusInvalidRemainderOrder:
      return "invalid_remainder_order";
    case kStatusFrontierCapacityExceeded:
      return "frontier_capacity_exceeded";
    case kStatusMissingFrontierOperand:
      return "missing_frontier_operand";
    case kStatusInvalidLevelTransition:
      return "invalid_level_transition";
    case kStatusInvalidParentFrame:
      return "invalid_parent_frame";
    case kStatusInvalidCommittedHit:
      return "invalid_committed_hit";
  }
  return "unknown";
}

}  // namespace typed_stack
}  // namespace v04
}  // namespace rtcore
