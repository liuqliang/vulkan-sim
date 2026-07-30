#include "rtcore_v04_functional_driver.h"
#include "rtcore_v04_request_owner_binding.h"
#include "rtcore_v04_typed_diagnostic_collector.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace functional_driver {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t size) {
  if (bytes == NULL) return false;
  for (size_t index = 0; index < size; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool region_shape_valid(
    const private_frontier::region_binding_v0 &region,
    const private_frontier::owner_binding_v0 &owner) {
  return request_owner::validate_private_frontier_owner_identity(owner) &&
         region.profile_id == private_frontier::kLayoutProfileId &&
         region.private_region_base != 0 &&
         region.slot_count != 0 &&
         owner.private_slot_id < region.slot_count;
}

bool metadata_equal(
    const private_frontier::frontier_metadata_image_v0 &lhs,
    const private_frontier::frontier_metadata_image_v0 &rhs) {
  return lhs.frontier_top == rhs.frontier_top &&
         lhs.frontier_count == rhs.frontier_count &&
         lhs.frontier_capacity == rhs.frontier_capacity &&
         lhs.current_level == rhs.current_level &&
         lhs.level_frame_depth == rhs.level_frame_depth &&
         lhs.max_level_depth == rhs.max_level_depth;
}

bool stack_frontier_matches(
    const typed_stack::frontier_metadata_v0 &frontier,
    const private_frontier::frontier_metadata_image_v0 &metadata) {
  return frontier.frontier_top == metadata.frontier_top &&
         frontier.frontier_count == metadata.frontier_count &&
         frontier.frontier_capacity == metadata.frontier_capacity;
}

uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool traversal_bound_matches(
    uint32_t supplied_bits,
    const private_frontier::root_private_operands_v0 &operands) {
  const float bound = operands.committed_hit.valid != 0
                          ? operands.committed_hit.hit_t
                          : operands.mutable_ray.t_max;
  return supplied_bits == fp32_bits(bound);
}

bool stack_packet_shape_valid(
    const stack_operation::operation_packet_v0 &packet) {
  const typed_stack::push_input_v0 zero_push = {};
  const typed_stack::pop_input_v0 zero_pop = {};
  const typed_stack::empty_input_v0 zero_empty = {};
  const bool push =
      packet.operation_kind ==
          typed_stack::kPushRemainderAndForwardSelected &&
      packet.input.operation_kind ==
          typed_stack::kPushRemainderAndForwardSelected &&
      stack_frontier_matches(packet.input.frontier,
                             packet.frontier_metadata) &&
      std::memcmp(&packet.pop_input, &zero_pop,
                  sizeof(zero_pop)) == 0 &&
      std::memcmp(&packet.empty_input, &zero_empty,
                  sizeof(zero_empty)) == 0;
  const bool pop =
      packet.operation_kind == typed_stack::kPopNext &&
      packet.pop_input.operation_kind == typed_stack::kPopNext &&
      packet.pop_input.has_top_entry == 1 &&
      stack_frontier_matches(packet.pop_input.frontier,
                             packet.frontier_metadata) &&
      std::memcmp(&packet.input, &zero_push,
                  sizeof(zero_push)) == 0 &&
      std::memcmp(&packet.empty_input, &zero_empty,
                  sizeof(zero_empty)) == 0;
  const bool empty =
      packet.operation_kind == typed_stack::kPopNext &&
      packet.empty_input.operation_kind == typed_stack::kPopNext &&
      packet.empty_input.parent_frame_available <= 1 &&
      metadata_equal(packet.frontier_metadata,
                     private_frontier::frontier_metadata_image_v0{
                         packet.empty_input.frontier.frontier_top,
                         packet.empty_input.frontier.frontier_count,
                         packet.empty_input.frontier.frontier_capacity,
                         packet.empty_input.frontier.current_level,
                         packet.empty_input.frontier.level_frame_depth,
                         packet.empty_input.frontier.max_level_depth}) &&
      std::memcmp(&packet.input, &zero_push,
                  sizeof(zero_push)) == 0 &&
      std::memcmp(&packet.pop_input, &zero_pop,
                  sizeof(zero_pop)) == 0;
  return packet.valid == 1 && packet.reservation_id != 0 &&
         packet.reservation_age != 0 &&
         packet.target_operation_seq != 0 &&
         packet.producer_operation_seq != 0 &&
         packet.target_operation_seq != packet.producer_operation_seq &&
         packet.slot_generation != 0 && (push || pop || empty) &&
         bytes_are_zero(packet.reserved_zero,
                        sizeof(packet.reserved_zero)) &&
         bytes_are_zero(packet.ray_policy.reserved_zero,
                        sizeof(packet.ray_policy.reserved_zero));
}

bool canonical_stack_input_matches(
    const stack_operation::operation_packet_v0 &packet,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot) {
  if (!region_shape_valid(region, packet.owner)) return false;
  private_frontier::frontier_metadata_image_v0 metadata = {};
  if (private_frontier::decode_metadata(
          canonical_slot, packet.owner, &metadata) !=
          private_frontier::kStatusOk ||
      !metadata_equal(metadata, packet.frontier_metadata)) {
    return false;
  }

  if (packet.operation_kind ==
      typed_stack::kPushRemainderAndForwardSelected) {
    private_frontier::root_private_operands_v0 operands = {};
    return private_frontier::decode_root_private_operands(
               canonical_slot, packet.owner, &operands) ==
               private_frontier::kStatusOk &&
           traversal_bound_matches(
               packet.input.current_traversal_bound_bits, operands);
  }

  if (packet.pop_input.operation_kind == typed_stack::kPopNext) {
    if (packet.frontier_metadata.frontier_top == 0) return false;
    private_frontier::root_private_operands_v0 operands = {};
    typed_node::compact_child_work_item_v0 top = {};
    return private_frontier::decode_root_private_operands(
               canonical_slot, packet.owner, &operands) ==
               private_frontier::kStatusOk &&
           private_frontier::decode_entry(
               canonical_slot, packet.owner,
               packet.frontier_metadata.frontier_top - 1, &top) ==
               private_frontier::kStatusOk &&
           traversal_bound_matches(
               packet.pop_input.current_traversal_bound_bits,
               operands) &&
           std::memcmp(&packet.pop_input.top_entry, &top,
                       sizeof(top)) == 0 &&
           std::memcmp(&packet.pop_input.current_decode_context,
                       &operands.decode_context,
                       sizeof(operands.decode_context)) == 0;
  }

  if (packet.empty_input.parent_frame_available != 0) {
    private_frontier::traversal_frame_projection_v0 parent = {};
    return private_frontier::decode_parent_frame(
               canonical_slot, packet.owner, &parent) ==
               private_frontier::kStatusOk &&
           std::memcmp(&packet.empty_input.parent_frame, &parent,
                       sizeof(parent)) == 0;
  }
  private_frontier::committed_hit_projection_v0 committed_hit = {};
  return private_frontier::decode_committed_hit(
             canonical_slot, packet.owner, &committed_hit) ==
             private_frontier::kStatusOk &&
         std::memcmp(&packet.empty_input.current_committed_hit,
                     &committed_hit, sizeof(committed_hit)) == 0;
}

bool instance_enter_packet_shape_valid(
    const fetch_target::operation_packet_v0 &packet) {
  const bool producer_identity_valid =
      (packet.producer_operation_seq == 0 &&
       packet.producer_commit_epoch == 0) ||
      (packet.producer_operation_seq != 0 &&
       packet.producer_commit_epoch != 0 &&
       packet.producer_operation_seq !=
           packet.target_operation_seq);
  return packet.valid == 1 &&
         packet.target_kind == fetch_target::kTargetInstance &&
         packet.operation_kind == fetch_target::kOperationFetchTarget &&
         packet.reservation_id != 0 && packet.reservation_age != 0 &&
         packet.target_operation_seq != 0 &&
         producer_identity_valid &&
         packet.slot_generation != 0 &&
         packet.raw_payload_base_address != 0 &&
         packet.raw_payload_bytes ==
             fetch_target::kInstanceRawPayloadBytes &&
         packet.target_reference.payload_kind ==
             typed_node::kInstancePayloadKind &&
         packet.target_reference.payload_byte_count ==
             fetch_target::kInstanceRawPayloadBytes &&
         packet.target_reference.level == typed_node::kLevelTlas &&
         packet.target_reference.source_kind ==
             fetch_target::
                 kTargetReferenceSelectedFetchCompatibilityAdapter &&
         packet.target_reference.proxy_delegated == 1 &&
         fetch_target::validate_target_reference_shape(
             packet.target_reference, fetch_target::kTargetInstance,
             packet.raw_payload_bytes,
             packet.raw_payload_base_address) &&
         bytes_are_zero(packet.ray_policy.reserved_zero,
                        sizeof(packet.ray_policy.reserved_zero)) &&
         bytes_are_zero(packet.reserved_zero,
                        sizeof(packet.reserved_zero));
}

bool instance_restore_packet_shape_valid(
    const fetch_target::operation_packet_v0 &packet) {
  return packet.valid == 1 &&
         packet.target_kind == fetch_target::kTargetInstance &&
         packet.operation_kind ==
             fetch_target::kOperationInstanceRestoreParent &&
         packet.reservation_id != 0 && packet.reservation_age != 0 &&
         packet.target_operation_seq != 0 &&
         packet.producer_operation_seq == 0 &&
         packet.producer_commit_epoch == 0 &&
         packet.slot_generation != 0 &&
         packet.raw_payload_base_address == 0 &&
         packet.raw_payload_bytes == 0 &&
         bytes_are_zero(packet.reserved_zero,
                        sizeof(packet.reserved_zero));
}

bool canonical_instance_enter_input_matches(
    const fetch_target::operation_packet_v0 &packet,
    const typed_instance::enter_input_v0 &input,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot) {
  static_assert(
      sizeof(input.world_ray) ==
          sizeof(packet.private_operands.mutable_ray),
      "Instance and private mutable rays must remain byte-compatible");
  private_frontier::root_private_operands_v0 operands = {};
  return region_shape_valid(region, packet.owner) &&
         private_frontier::decode_root_private_operands(
             canonical_slot, packet.owner, &operands) ==
             private_frontier::kStatusOk &&
         std::memcmp(&operands, &packet.private_operands,
                     sizeof(operands)) == 0 &&
         std::memcmp(&input.world_ray,
                     &packet.private_operands.mutable_ray,
                     sizeof(input.world_ray)) == 0 &&
         std::memcmp(&input.tlas_decode_context,
                     &packet.private_operands.decode_context,
                     sizeof(input.tlas_decode_context)) == 0 &&
         input.policy.ray_flags == packet.ray_policy.ray_flags &&
         input.policy.cull_mask == packet.ray_policy.cull_mask &&
         typed_instance::validate_enter_transition_binding(input) &&
         input.instance_blas_reference.instance_metadata_reference ==
             packet.raw_payload_base_address &&
         std::memcmp(input.raw_instance.raw_bytes,
                     packet.raw_payload,
                     fetch_target::kInstanceRawPayloadBytes) == 0;
}

bool canonical_instance_restore_input_matches(
    const fetch_target::operation_packet_v0 &packet,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot) {
  private_frontier::traversal_frame_projection_v0 parent = {};
  return region_shape_valid(region, packet.owner) &&
         private_frontier::decode_parent_frame(
             canonical_slot, packet.owner, &parent) ==
             private_frontier::kStatusOk &&
         std::memcmp(&parent, &packet.parent_frame,
                     sizeof(parent)) == 0;
}

}  // namespace

status_kind execute_one_stack(
    const stack_operation::operation_packet_v0 &packet,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    stack_execution_v0 *execution) {
  if (execution == NULL) return kStatusInvalidArgument;
  *execution = stack_execution_v0();
  if (!stack_packet_shape_valid(packet)) {
    return kStatusInvalidOperationPacket;
  }
  if (!canonical_stack_input_matches(packet, region,
                                     canonical_slot)) {
    return kStatusCanonicalInputMismatch;
  }
  execution->operation_packet = packet;

  if (packet.operation_kind ==
      typed_stack::kPushRemainderAndForwardSelected) {
    execution->push_result = typed_stack::execute_push(packet.input);
    execution->operator_invocation_count = 1;
    if (!typed_stack::validate_push_result(
            execution->push_result)) {
      if (execution->push_result.status ==
          typed_stack::kStatusFrontierCapacityExceeded) {
        return kStatusFrontierCapacityExceeded;
      }
      return kStatusTypedOperatorFailed;
    }
    if (stack_semantic::prepare_stack_pushed_and_selected(
            packet.owner, packet.target_operation_seq, region,
            canonical_slot, execution->push_result,
            &execution->append_plan) != stack_semantic::kStatusOk) {
      return kStatusSemanticApplyFailed;
    }
    execution->semantic_plan_kind = kSemanticPlanStackAppend;
  } else if (packet.empty_input.operation_kind ==
             typed_stack::kPopNext) {
    execution->empty_result =
        typed_stack::execute_empty(packet.empty_input);
    execution->operator_invocation_count = 1;
    if (!typed_stack::validate_empty_result(
            packet.empty_input, execution->empty_result)) {
      return kStatusTypedOperatorFailed;
    }
    if (execution->empty_result.result_kind ==
        typed_stack::kStackRestoreParent) {
      if (stack_semantic::prepare_stack_restore_parent(
              packet.owner, packet.target_operation_seq, region,
              packet.empty_input, execution->empty_result,
              &execution->pop_plan) != stack_semantic::kStatusOk) {
        return kStatusSemanticApplyFailed;
      }
      execution->semantic_plan_kind =
          kSemanticPlanStackRestoreParent;
    } else if (execution->empty_result.result_kind ==
               typed_stack::kStackFinalHit) {
      execution->semantic_plan_kind =
          kSemanticPlanStackTerminalHit;
      execution->terminal_boundary = 1;
    } else if (execution->empty_result.result_kind ==
               typed_stack::kStackFinalMiss) {
      execution->semantic_plan_kind =
          kSemanticPlanStackTerminalMiss;
      execution->terminal_boundary = 1;
    } else {
      return kStatusSemanticApplyFailed;
    }
  } else {
    execution->pop_result =
        typed_stack::execute_pop(packet.pop_input);
    execution->operator_invocation_count = 1;
    if (!typed_stack::validate_pop_result(
            packet.pop_input, execution->pop_result)) {
      return kStatusTypedOperatorFailed;
    }
    if (stack_semantic::prepare_stack_pop_next(
            packet.owner, packet.target_operation_seq, region,
            packet.frontier_metadata, packet.pop_input,
            execution->pop_result,
            &execution->pop_plan) != stack_semantic::kStatusOk) {
      return kStatusSemanticApplyFailed;
    }
    execution->semantic_plan_kind = kSemanticPlanStackPop;
  }
  typed_diagnostic::record_v0 diagnostic = {};
  diagnostic.owner = packet.owner;
  diagnostic.operation_seq = packet.target_operation_seq;
  diagnostic.driver = typed_diagnostic::kDriverFunctionalOnly;
  diagnostic.unit = typed_diagnostic::kUnitStack;
  diagnostic.operation_kind = packet.operation_kind;
  diagnostic.semantic_plan_kind = execution->semantic_plan_kind;
  diagnostic.boundary_kind = execution->terminal_boundary;
  if (packet.operation_kind ==
      typed_stack::kPushRemainderAndForwardSelected) {
    diagnostic.route_kind = execution->append_plan.route_kind;
    diagnostic.typed_input = &packet.input;
    diagnostic.typed_input_bytes = sizeof(packet.input);
    diagnostic.typed_result = &execution->push_result;
    diagnostic.typed_result_bytes = sizeof(execution->push_result);
    diagnostic.semantic_plan = &execution->append_plan;
    diagnostic.semantic_plan_bytes = sizeof(execution->append_plan);
  } else if (packet.empty_input.operation_kind ==
             typed_stack::kPopNext) {
    diagnostic.route_kind = execution->semantic_plan_kind;
    diagnostic.typed_input = &packet.empty_input;
    diagnostic.typed_input_bytes = sizeof(packet.empty_input);
    diagnostic.typed_result = &execution->empty_result;
    diagnostic.typed_result_bytes = sizeof(execution->empty_result);
    if (!execution->terminal_boundary) {
      diagnostic.semantic_plan = &execution->pop_plan;
      diagnostic.semantic_plan_bytes = sizeof(execution->pop_plan);
      diagnostic.route_kind = execution->pop_plan.route_kind;
    }
  } else {
    diagnostic.route_kind = execution->pop_plan.route_kind;
    diagnostic.typed_input = &packet.pop_input;
    diagnostic.typed_input_bytes = sizeof(packet.pop_input);
    diagnostic.typed_result = &execution->pop_result;
    diagnostic.typed_result_bytes = sizeof(execution->pop_result);
    diagnostic.semantic_plan = &execution->pop_plan;
    diagnostic.semantic_plan_bytes = sizeof(execution->pop_plan);
  }
  if (!typed_diagnostic::emit_record(diagnostic)) {
    return kStatusDiagnosticRejected;
  }
  execution->valid = 1;
  return kStatusOk;
}

status_kind execute_one_instance_enter(
    const fetch_target::operation_packet_v0 &packet,
    const typed_instance::enter_input_v0 &input,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    instance_enter_execution_v0 *execution,
    bool short_stack_mode) {
  if (execution == NULL) return kStatusInvalidArgument;
  *execution = instance_enter_execution_v0();
  if (!instance_enter_packet_shape_valid(packet)) {
    return kStatusInvalidOperationPacket;
  }
  if (!canonical_instance_enter_input_matches(
          packet, input, region, canonical_slot)) {
    return kStatusCanonicalInputMismatch;
  }
  execution->operation_packet = packet;
  execution->operator_input = input;
  execution->operator_result =
      typed_instance::execute_enter(execution->operator_input);
  execution->operator_invocation_count = 1;
  if (execution->operator_result.status !=
      typed_instance::kStatusOk) {
    return kStatusTypedOperatorFailed;
  }
  if (instance_semantic::prepare_enter(
          packet.owner, packet.target_operation_seq, region,
          canonical_slot, execution->operator_input,
          execution->operator_result,
          &execution->semantic_plan, short_stack_mode) !=
      instance_semantic::kStatusOk) {
    return kStatusSemanticApplyFailed;
  }
  execution->semantic_plan_kind = kSemanticPlanInstanceEnter;
  typed_diagnostic::record_v0 diagnostic = {};
  diagnostic.owner = packet.owner;
  diagnostic.operation_seq = packet.target_operation_seq;
  diagnostic.driver = typed_diagnostic::kDriverFunctionalOnly;
  diagnostic.unit = typed_diagnostic::kUnitInstance;
  diagnostic.operation_kind = packet.operation_kind;
  diagnostic.semantic_plan_kind = execution->semantic_plan_kind;
  diagnostic.route_kind = execution->semantic_plan.route_kind;
  diagnostic.typed_input = &execution->operator_input;
  diagnostic.typed_input_bytes = sizeof(execution->operator_input);
  diagnostic.typed_result = &execution->operator_result;
  diagnostic.typed_result_bytes = sizeof(execution->operator_result);
  diagnostic.semantic_plan = &execution->semantic_plan;
  diagnostic.semantic_plan_bytes = sizeof(execution->semantic_plan);
  if (!typed_diagnostic::emit_record(diagnostic)) {
    return kStatusDiagnosticRejected;
  }
  execution->valid = 1;
  return kStatusOk;
}

status_kind execute_one_instance_restore(
    const fetch_target::operation_packet_v0 &packet,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    instance_restore_execution_v0 *execution) {
  if (execution == NULL) return kStatusInvalidArgument;
  *execution = instance_restore_execution_v0();
  if (!instance_restore_packet_shape_valid(packet)) {
    return kStatusInvalidOperationPacket;
  }
  if (!canonical_instance_restore_input_matches(
          packet, region, canonical_slot)) {
    return kStatusCanonicalInputMismatch;
  }
  execution->operation_packet = packet;
  execution->operator_input.profile_id =
      typed_instance::kGenRtDerivedProfileId;
  execution->operator_input.operation_kind =
      typed_instance::kRestoreParent;
  execution->operator_input.parent_frame = packet.parent_frame;
  execution->operator_result =
      typed_instance::execute_restore_parent(
          execution->operator_input);
  execution->operator_invocation_count = 1;
  if (!typed_instance::validate_restore_parent_result(
          execution->operator_input,
          execution->operator_result)) {
    return kStatusTypedOperatorFailed;
  }
  if (instance_semantic::prepare_restore_parent(
          packet.owner, packet.target_operation_seq, region,
          canonical_slot, execution->operator_result,
          &execution->semantic_plan) !=
      instance_semantic::kStatusOk) {
    return kStatusSemanticApplyFailed;
  }
  execution->semantic_plan_kind = kSemanticPlanInstanceRestore;
  typed_diagnostic::record_v0 diagnostic = {};
  diagnostic.owner = packet.owner;
  diagnostic.operation_seq = packet.target_operation_seq;
  diagnostic.driver = typed_diagnostic::kDriverFunctionalOnly;
  diagnostic.unit = typed_diagnostic::kUnitInstance;
  diagnostic.operation_kind = packet.operation_kind;
  diagnostic.semantic_plan_kind = execution->semantic_plan_kind;
  diagnostic.route_kind = execution->semantic_plan.route_kind;
  diagnostic.typed_input = &execution->operator_input;
  diagnostic.typed_input_bytes = sizeof(execution->operator_input);
  diagnostic.typed_result = &execution->operator_result;
  diagnostic.typed_result_bytes = sizeof(execution->operator_result);
  diagnostic.semantic_plan = &execution->semantic_plan;
  diagnostic.semantic_plan_bytes = sizeof(execution->semantic_plan);
  if (!typed_diagnostic::emit_record(diagnostic)) {
    return kStatusDiagnosticRejected;
  }
  execution->valid = 1;
  return kStatusOk;
}

}  // namespace functional_driver
}  // namespace v04
}  // namespace rtcore
