#include "rtcore_v04_result_semantic_applier.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "rtcore_v04_canonical_ray.h"

namespace rtcore {
namespace v04 {
namespace result_semantic {
namespace {

bool bytes_are_zero(const void *value, size_t byte_count) {
  const uint8_t *bytes = static_cast<const uint8_t *>(value);
  for (size_t index = 0; index < byte_count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

float fp32_value(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool operation_packet_valid_internal(
    const fetch_target::operation_packet_v0 &packet) {
  const typed_blas::as_decode_context_v0 &context =
      packet.private_operands.decode_context;
  const fetch_target::target_reference_v0 &reference =
      packet.target_reference;
  const private_frontier::mutable_ray_state_v0 &ray =
      packet.private_operands.mutable_ray;
  const private_frontier::committed_hit_projection_v0 &hit =
      packet.private_operands.committed_hit;
  bool ray_valid = std::isfinite(ray.t_min) &&
                   std::isfinite(ray.t_max) &&
                   ray.t_min <= ray.t_max;
  for (unsigned axis = 0; axis < 3; ++axis) {
    ray_valid =
        ray_valid && std::isfinite(ray.origin[axis]) &&
        canonical_ray::inverse_direction_matches(
            ray.direction[axis], ray.inverse_direction[axis]);
  }
  return packet.valid == 1 &&
         packet.target_kind == fetch_target::kTargetNode &&
         packet.target_operation_seq != 0 &&
         packet.reservation_id != 0 && packet.reservation_age != 0 &&
         packet.slot_generation != 0 &&
         packet.raw_payload_base_address != 0 &&
         packet.raw_payload_bytes == fetch_target::kNodeRawPayloadBytes &&
         reference.payload_byte_count ==
             fetch_target::kNodeRawPayloadBytes &&
         reference.payload_kind == typed_node::kInternalPayloadKind &&
         reference.level >= typed_node::kLevelTlas &&
         reference.level <= typed_node::kLevelBlas &&
         reference.source_kind >=
             fetch_target::kTargetReferenceRootCompatibilityProxy &&
         reference.source_kind <=
             fetch_target::
                 kTargetReferenceInstanceBlasRootProducer &&
         reference.proxy_delegated <= 1 &&
         (reference.source_kind !=
              fetch_target::kTargetReferenceInstanceBlasRootProducer ||
          reference.proxy_delegated == 0) &&
         bytes_are_zero(reference.reserved_zero,
                        sizeof(reference.reserved_zero)) &&
         (reference.payload_offset & uint64_t{0x3f}) == 0 &&
         std::isfinite(fp32_value(reference.near_t_bits)) &&
         context.bvh_format_profile_id ==
             typed_node::kGenRtDerivedProfileId &&
         context.reserved_zero == 0 &&
         context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         context.as_object.as_type ==
             (reference.level == typed_node::kLevelTlas
                  ? uint8_t{1}
                  : typed_blas::kAsTypeBlas) &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         context.device_base != 0 &&
         context.device_range_bytes >=
             fetch_target::kNodeRawPayloadBytes &&
         context.device_range_bytes <=
             std::numeric_limits<uint64_t>::max() -
                 context.device_base &&
         packet.raw_payload_base_address >= context.device_base &&
         reference.payload_offset ==
             packet.raw_payload_base_address - context.device_base &&
         reference.payload_offset <= context.device_range_bytes &&
         uint64_t{fetch_target::kNodeRawPayloadBytes} <=
             context.device_range_bytes -
                 reference.payload_offset &&
         packet.owner.request_identity != 0 &&
         packet.owner.generation != 0 && packet.owner.lane_id < 32 &&
         bytes_are_zero(packet.owner.reserved_zero,
                        sizeof(packet.owner.reserved_zero)) &&
         bytes_are_zero(packet.ray_policy.reserved_zero,
                        sizeof(packet.ray_policy.reserved_zero)) &&
         ray_valid && hit.valid <= 1 &&
         hit.attribute_word_count <= 4 &&
         bytes_are_zero(hit.reserved_zero0,
                        sizeof(hit.reserved_zero0)) &&
         hit.reserved_zero1 == 0 &&
         (hit.valid == 0 || std::isfinite(hit.hit_t)) &&
         bytes_are_zero(
             packet.raw_payload + fetch_target::kNodeRawPayloadBytes,
             fetch_target::kMaxRawPayloadBytes -
                 fetch_target::kNodeRawPayloadBytes);
}

bool child_valid(
    const typed_node::compact_child_work_item_v0 &child,
    const typed_blas::as_decode_context_v0 &context,
    fetch_target::target_kind *target) {
  typed_node::selected_child_fetch_work_item_v0 selected = {};
  selected.child = child;
  selected.decode_context = context;
  uint16_t payload_bytes = 0;
  return fetch_target::classify_selected_fetch(
             selected, target, &payload_bytes) == fetch_target::kStatusOk &&
         payload_bytes == child.payload_byte_count;
}

bool child_precedes(
    const typed_node::compact_child_work_item_v0 &lhs,
    const typed_node::compact_child_work_item_v0 &rhs) {
  const float lhs_near = fp32_value(lhs.near_t_bits);
  const float rhs_near = fp32_value(rhs.near_t_bits);
  return lhs_near < rhs_near ||
         (lhs_near == rhs_near && lhs.child_slot < rhs.child_slot);
}

bool result_tail_zero(const typed_node::route_result_v0 &result,
                      unsigned first_unused_frontier) {
  if (!bytes_are_zero(result.reserved_zero, sizeof(result.reserved_zero)) ||
      !bytes_are_zero(result.reserved_zero_tail,
                      sizeof(result.reserved_zero_tail))) {
    return false;
  }
  for (unsigned index = first_unused_frontier;
       index < typed_node::kMaxChildren - 1; ++index) {
    if (!bytes_are_zero(&result.frontier[index],
                        sizeof(result.frontier[index]))) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool validate_node_operation_packet(
    const fetch_target::operation_packet_v0 &packet) {
  return operation_packet_valid_internal(packet);
}

status_kind prepare_node_result(
    const fetch_target::operation_packet_v0 &packet,
    const typed_node::route_result_v0 &result,
    node_commit_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = node_commit_plan_v0();
  if (!validate_node_operation_packet(packet)) {
    return kStatusInvalidOperationPacket;
  }
  if (result.status != typed_node::kStatusOk ||
      result.frontier_count > typed_node::kMaxChildren - 1 ||
      !result_tail_zero(result, result.frontier_count)) {
    return kStatusInvalidTypedResult;
  }

  node_route_kind route = kNodeRouteInvalid;
  fetch_target::target_kind next_target = fetch_target::kTargetInvalid;
  if (result.result_kind == typed_node::kRouteResultMiss) {
    if (result.output_valid_mask != 0 || result.frontier_count != 0 ||
        !bytes_are_zero(&result.selected_fetch,
                        sizeof(result.selected_fetch))) {
      return kStatusInvalidTypedResult;
    }
    route = kNodeRouteNoChild;
  } else if (result.result_kind == typed_node::kRouteResultSelected) {
    const uint8_t expected_mask =
        result.frontier_count == 0
            ? static_cast<uint8_t>(typed_node::kSelectedFetchValid)
            : static_cast<uint8_t>(typed_node::kSelectedFetchValid |
                                   typed_node::kFrontierItemsValid);
    if (result.output_valid_mask != expected_mask ||
        std::memcmp(&result.selected_fetch.decode_context,
                    &packet.private_operands.decode_context,
                    sizeof(result.selected_fetch.decode_context)) != 0 ||
        !child_valid(result.selected_fetch.child,
                     result.selected_fetch.decode_context, &next_target)) {
      return kStatusInvalidSelectedFetch;
    }
    uint8_t seen_child_slots = static_cast<uint8_t>(
        1u << result.selected_fetch.child.child_slot);
    typed_node::compact_child_work_item_v0 previous =
        result.selected_fetch.child;
    for (unsigned index = 0; index < result.frontier_count; ++index) {
      fetch_target::target_kind frontier_target =
          fetch_target::kTargetInvalid;
      const typed_node::compact_child_work_item_v0 &current =
          result.frontier[index];
      const uint8_t child_bit =
          static_cast<uint8_t>(1u << current.child_slot);
      if (!child_valid(result.frontier[index],
                       result.selected_fetch.decode_context,
                       &frontier_target) ||
          (seen_child_slots & child_bit) != 0 ||
          !child_precedes(previous, current)) {
        return kStatusInvalidFrontier;
      }
      seen_child_slots = static_cast<uint8_t>(
          seen_child_slots | child_bit);
      previous = current;
    }
    route = result.frontier_count == 0
                ? kNodeRouteDirectChild
                : kNodeRouteMultiChildToStack;
  } else {
    return kStatusInvalidTypedResult;
  }

  plan->owner = packet.owner;
  plan->producer_operation_seq = packet.target_operation_seq;
  plan->valid = 1;
  plan->route_kind = route;
  plan->next_target_kind = next_target;
  plan->frontier_count = result.frontier_count;
  plan->selected_fetch = result.selected_fetch;
  std::memcpy(plan->frontier, result.frontier,
              sizeof(result.frontier));
  return kStatusOk;
}

bool validate_node_commit_plan(
    const node_commit_plan_v0 &plan,
    const fetch_target::operation_packet_v0 &packet,
    const typed_node::route_result_v0 &result) {
  node_commit_plan_v0 expected = {};
  return prepare_node_result(packet, result, &expected) == kStatusOk &&
         std::memcmp(&plan, &expected, sizeof(plan)) == 0;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidOperationPacket:
      return "invalid_operation_packet";
    case kStatusInvalidTypedResult:
      return "invalid_typed_result";
    case kStatusInvalidSelectedFetch:
      return "invalid_selected_fetch";
    case kStatusInvalidFrontier:
      return "invalid_frontier";
  }
  return "unknown";
}

const char *node_route_name(node_route_kind route) {
  switch (route) {
    case kNodeRouteInvalid:
      return "NODE_INVALID";
    case kNodeRouteNoChild:
      return "NODE_NO_CHILD";
    case kNodeRouteDirectChild:
      return "NODE_DIRECT_CHILD";
    case kNodeRouteMultiChildToStack:
      return "NODE_MULTI_CHILD_TO_STACK";
  }
  return "NODE_UNKNOWN";
}

}  // namespace result_semantic
}  // namespace v04
}  // namespace rtcore
