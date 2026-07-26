#include "rtcore_v04_functional_engine.h"

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
  uint8_t reserved_zero[6];
  uint32_t producer_operation_seq;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  typed_node::route_result_v0 node_route;
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
  if (metadata.frontier_count != 0) {
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
        if (functional_driver::execute_one_node(
                packet, &execution) != functional_driver::kStatusOk ||
            !execution.valid) {
          return kStatusFunctionalDriverRejected;
        }
        ++state->node_visits;
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
        if (functional_driver::execute_one_instance_enter(
                packet, input, state->private_region,
                state->canonical_slot, &execution) !=
                functional_driver::kStatusOk ||
            !execution.valid) {
          return kStatusFunctionalDriverRejected;
        }
        status = apply_instance_enter(state, execution);
        if (status != kStatusOk) return status;
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
        if (functional_driver::prepare_primitive_operator_input(
                packet, &input) != functional_driver::kStatusOk) {
          return kStatusFunctionalDriverRejected;
        }
        functional_driver::primitive_execution_v0 execution = {};
        if (functional_driver::execute_one_primitive(
                packet, input, state->private_region,
                state->canonical_slot, &execution) !=
                functional_driver::kStatusOk ||
            !execution.valid) {
          return kStatusFunctionalDriverRejected;
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
        action = next_action_v0();
        action.kind = kActionStackPop;
        action.producer_operation_seq =
            packet.target_operation_seq;
        continue;
      }
      return kStatusTargetRejected;
    }

    if (action.kind == kActionStackPush ||
        action.kind == kActionStackPop) {
      stack_operation::operation_packet_v0 packet = {};
      status_kind status = build_stack_packet(state, action, &packet);
      if (status != kStatusOk) return status;
      functional_driver::stack_execution_v0 execution = {};
      if (functional_driver::execute_one_stack(
              packet, state->private_region, state->canonical_slot,
              &execution) != functional_driver::kStatusOk ||
          !execution.valid) {
        return kStatusFunctionalDriverRejected;
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
        if (functional_driver::execute_one_instance_restore(
                restore_packet, state->private_region,
                state->canonical_slot, &restore) !=
                functional_driver::kStatusOk ||
            !restore.valid) {
          return kStatusFunctionalDriverRejected;
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
      input.raw_payload_base_address == 0) {
    return kStatusInvalidArgument;
  }

  *state = state_v0();
  *output = output_v0();
  state->valid = 1;
  state->owner = input.owner;
  state->private_region = input.private_region;
  state->ray_policy = input.ray_policy;
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

}  // namespace functional_engine
}  // namespace v04
}  // namespace rtcore
