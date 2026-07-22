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
  }
  return "unknown";
}

}  // namespace typed_stack
}  // namespace v04
}  // namespace rtcore
