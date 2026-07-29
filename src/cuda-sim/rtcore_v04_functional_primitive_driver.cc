#include "rtcore_v04_functional_driver.h"
#include "rtcore_v04_request_owner_binding.h"
#include "rtcore_v04_typed_diagnostic_collector.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace functional_driver {
namespace {

bool bytes_are_zero(const void *value, size_t byte_count) {
  const uint8_t *bytes = static_cast<const uint8_t *>(value);
  for (size_t index = 0; index < byte_count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool candidate_diagnostics_enabled() {
  const char *value =
      std::getenv("VULKAN_SIM_RTCORE_ABI_V04_HIT_CANDIDATE_DIAGNOSTICS");
  return value != NULL && value[0] != '\0' &&
         std::strcmp(value, "0") != 0;
}

uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void emit_candidate_diagnostic(
    const fetch_target::operation_packet_v0 &packet,
    const typed_primitive::route_input_v0 &input,
    const typed_primitive::candidate_result_v0 &candidate,
    const typed_primitive::route_result_v0 &route) {
  if (!candidate_diagnostics_enabled() ||
      packet.target_reference.payload_kind !=
          typed_node::kQuadPayloadKind) {
    return;
  }
  const typed_stack::committed_hit_projection_v0 &committed =
      input.current_committed_hit;
  const char *distance_relation = "no_geometric_hit";
  if (candidate.geometric_hit != 0) {
    const float candidate_t = [&candidate]() {
      float value = 0.0f;
      std::memcpy(&value, &candidate.world_t_bits, sizeof(value));
      return value;
    }();
    if (committed.valid == 0) {
      distance_relation = "no_committed_hit";
    } else if (candidate_t < committed.hit_t) {
      distance_relation = "less";
    } else if (candidate_t == committed.hit_t) {
      distance_relation = "equal";
    } else {
      distance_relation = "greater";
    }
  }
  std::printf(
      "GPGPU-Sim RTCORE_V04_HIT_CANDIDATE_DIAGNOSTIC "
      "owner_hw_sid=%u resident_warp_slot=%u request_identity=%u "
      "request_generation=%u private_slot_id=%u lane_id=%u "
      "operation_seq=%u payload_offset=0x%llx "
      "geometric_hit=%u candidate_hit=%u candidate_status=%u "
      "candidate_t_bits=0x%08x candidate_primitive=%u "
      "candidate_geometry=%u candidate_instance=%u "
      "candidate_instance_custom=%u candidate_instance_sbt=%u "
      "candidate_hit_kind=%u committed_valid=%u "
      "committed_t_bits=0x%08x committed_primitive=%u "
      "committed_geometry=%u committed_instance=%u "
      "committed_instance_custom=%u committed_instance_sbt=%u "
      "committed_hit_kind=%u distance_relation=%s route_kind=%u "
      "route_identity_valid=%u route_primitive=%u route_geometry=%u "
      "route_instance=%u\n",
      packet.owner.owner_hw_sid, packet.owner.resident_warp_id,
      packet.owner.request_identity, packet.owner.generation,
      packet.owner.private_slot_id, packet.owner.lane_id,
      packet.target_operation_seq,
      static_cast<unsigned long long>(
          packet.target_reference.payload_offset),
      candidate.geometric_hit, candidate.candidate_hit,
      candidate.status, candidate.world_t_bits,
      candidate.primitive_index, candidate.geometry_index,
      input.current_instance.instance_index,
      input.current_instance.instance_custom_index,
      input.current_instance.instance_sbt_contribution,
      candidate.hit_kind, committed.valid,
      fp32_bits(committed.hit_t), committed.primitive_index,
      committed.geometry_index, committed.instance_index,
      committed.instance_custom_index,
      committed.instance_sbt_contribution, committed.hit_kind,
      distance_relation, route.result_kind,
      (route.output_valid_mask &
       typed_primitive::kIdentityAndPolicyValid) != 0
          ? 1
          : 0,
      route.identity_and_policy.primitive_index,
      route.identity_and_policy.geometry_index,
      route.identity_and_policy.instance_index);
  std::fflush(stdout);
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

bool primitive_packet_shape_valid(
    const fetch_target::operation_packet_v0 &packet) {
  const bool producer_identity_valid =
      (packet.producer_operation_seq == 0 &&
       packet.producer_commit_epoch == 0) ||
      (packet.producer_operation_seq != 0 &&
       packet.producer_commit_epoch != 0 &&
       packet.producer_operation_seq !=
           packet.target_operation_seq);
  const bool primitive_kind =
      packet.target_reference.payload_kind ==
          typed_node::kQuadPayloadKind ||
      packet.target_reference.payload_kind ==
          typed_node::kProceduralPayloadKind;
  const private_frontier::traversal_frame_projection_v0 empty_parent = {};
  return packet.valid == 1 &&
         packet.target_kind == fetch_target::kTargetPrimitive &&
         packet.operation_kind == fetch_target::kOperationFetchTarget &&
         packet.reservation_id != 0 && packet.reservation_age != 0 &&
         packet.target_operation_seq != 0 && producer_identity_valid &&
         packet.slot_generation != 0 &&
         packet.raw_payload_base_address != 0 &&
         packet.raw_payload_bytes ==
             fetch_target::kPrimitiveRawPayloadBytes &&
         primitive_kind &&
         packet.target_reference.payload_byte_count ==
             fetch_target::kPrimitiveRawPayloadBytes &&
         packet.target_reference.level == typed_node::kLevelBlas &&
         packet.target_reference.source_kind ==
             fetch_target::
                 kTargetReferenceSelectedFetchCompatibilityAdapter &&
         packet.target_reference.proxy_delegated == 1 &&
         fetch_target::validate_target_reference_shape(
             packet.target_reference, fetch_target::kTargetPrimitive,
             packet.raw_payload_bytes,
             packet.raw_payload_base_address) &&
         bytes_are_zero(packet.ray_policy.reserved_zero,
                        sizeof(packet.ray_policy.reserved_zero)) &&
         bytes_are_zero(packet.reserved_zero,
                        sizeof(packet.reserved_zero)) &&
         std::memcmp(&packet.parent_frame, &empty_parent,
                     sizeof(empty_parent)) == 0 &&
         bytes_are_zero(
             packet.raw_payload +
                 fetch_target::kPrimitiveRawPayloadBytes,
             fetch_target::kMaxRawPayloadBytes -
                 fetch_target::kPrimitiveRawPayloadBytes);
}

bool canonical_primitive_input_matches(
    const fetch_target::operation_packet_v0 &packet,
    const typed_primitive::route_input_v0 &input,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot) {
  static_assert(sizeof(input.ray) ==
                    sizeof(packet.private_operands.mutable_ray),
                "Primitive and private rays must remain byte-compatible");
  private_frontier::root_private_operands_v0 operands = {};
  private_frontier::instance_shader_projection_v0 current_instance = {};
  if (!region_shape_valid(region, packet.owner) ||
      private_frontier::decode_root_private_operands(
          canonical_slot, packet.owner, &operands) !=
          private_frontier::kStatusOk ||
      private_frontier::decode_current_instance(
          canonical_slot, packet.owner, &current_instance) !=
          private_frontier::kStatusOk) {
    return false;
  }
  if (packet.raw_payload_base_address <
      packet.private_operands.decode_context.device_base) {
    return false;
  }
  return std::memcmp(&packet.private_operands, &operands,
                     sizeof(operands)) == 0 &&
         input.profile_id ==
             typed_primitive::kGenRtDerivedProfileId &&
         input.operation_kind == typed_primitive::kOperationTestLeaf &&
         input.leaf_fetch_address ==
             packet.raw_payload_base_address &&
         input.input_slot_mask == uint64_t{1} &&
         std::memcmp(&input.ray,
                     &packet.private_operands.mutable_ray,
                     sizeof(input.ray)) == 0 &&
         std::memcmp(&input.ray_policy, &packet.ray_policy,
                     sizeof(input.ray_policy)) == 0 &&
         std::memcmp(&input.current_instance, &current_instance,
                     sizeof(current_instance)) == 0 &&
         std::memcmp(&packet.current_instance, &current_instance,
                     sizeof(current_instance)) == 0 &&
         std::memcmp(&input.current_committed_hit,
                     &packet.private_operands.committed_hit,
                     sizeof(input.current_committed_hit)) == 0 &&
         std::memcmp(&input.decode_context,
                     &packet.private_operands.decode_context,
                     sizeof(input.decode_context)) == 0 &&
         input.raw_primitive.header.expected_payload_kind ==
             packet.target_reference.payload_kind &&
         input.raw_primitive.header.expected_chunk_count == 2 &&
         input.raw_primitive.header.payload_byte_count ==
             fetch_target::kPrimitiveRawPayloadBytes &&
         input.raw_primitive.header.received_chunk_mask == 0x03 &&
         bytes_are_zero(
             input.raw_primitive.header.reserved_zero,
             sizeof(input.raw_primitive.header.reserved_zero)) &&
         std::memcmp(input.raw_primitive.raw_bytes,
                     packet.raw_payload,
                     fetch_target::kPrimitiveRawPayloadBytes) == 0 &&
         packet.target_reference.payload_offset ==
             packet.raw_payload_base_address -
                 packet.private_operands.decode_context.device_base &&
         bytes_are_zero(input.reserved_zero,
                        sizeof(input.reserved_zero));
}

}  // namespace

status_kind prepare_primitive_operator_input(
    const fetch_target::operation_packet_v0 &packet,
    typed_primitive::route_input_v0 *input) {
  if (input == NULL) return kStatusInvalidArgument;
  *input = typed_primitive::route_input_v0();
  if (!primitive_packet_shape_valid(packet)) {
    return kStatusInvalidOperationPacket;
  }
  typed_primitive::route_input_v0 prepared = {};
  prepared.profile_id = typed_primitive::kGenRtDerivedProfileId;
  prepared.operation_kind = typed_primitive::kOperationTestLeaf;
  prepared.ray = packet.private_operands.mutable_ray;
  prepared.ray_policy = packet.ray_policy;
  prepared.leaf_fetch_address = packet.raw_payload_base_address;
  const bool raw_ready =
      packet.target_reference.payload_kind ==
              typed_node::kQuadPayloadKind
          ? typed_primitive::make_raw_primitive_payload(
                packet.raw_payload, &prepared.raw_primitive)
          : typed_primitive::make_raw_procedural_payload(
                packet.raw_payload, &prepared.raw_primitive);
  if (!raw_ready ||
      !typed_primitive::extract_geometry_policy(
          prepared.raw_primitive, &prepared.geometry_policy)) {
    return kStatusInvalidOperationPacket;
  }
  prepared.input_slot_mask = uint64_t{1};
  prepared.decode_context = packet.private_operands.decode_context;
  prepared.current_instance = packet.current_instance;
  prepared.current_committed_hit =
      packet.private_operands.committed_hit;
  *input = prepared;
  return kStatusOk;
}

status_kind execute_one_primitive(
    const fetch_target::operation_packet_v0 &packet,
    const typed_primitive::route_input_v0 &input,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    primitive_execution_v0 *execution) {
  if (execution == NULL) return kStatusInvalidArgument;
  *execution = primitive_execution_v0();
  if (!primitive_packet_shape_valid(packet)) {
    return kStatusInvalidOperationPacket;
  }
  if (!canonical_primitive_input_matches(
          packet, input, region, canonical_slot)) {
    return kStatusCanonicalInputMismatch;
  }

  execution->operation_packet = packet;
  execution->operator_input = input;
  typed_primitive::candidate_result_v0 triangle_candidate = {};
  execution->operator_result = typed_primitive::execute_route(
      execution->operator_input, &triangle_candidate);
  execution->operator_invocation_count = 1;
  if (execution->operator_result.status !=
          typed_primitive::kStatusOk ||
      execution->operator_result.typed_operator_invocation_count != 1) {
    return kStatusTypedOperatorFailed;
  }
  emit_candidate_diagnostic(
      packet, execution->operator_input, triangle_candidate,
      execution->operator_result);
  if (primitive_semantic::prepare_result(
          packet.owner, packet.target_operation_seq,
          execution->operator_input, execution->operator_result,
          &execution->semantic_plan) !=
      primitive_semantic::kStatusOk) {
    return kStatusSemanticApplyFailed;
  }
  execution->semantic_plan_kind = kSemanticPlanPrimitive;
  typed_diagnostic::record_v0 diagnostic = {};
  diagnostic.owner = packet.owner;
  diagnostic.operation_seq = packet.target_operation_seq;
  diagnostic.driver = typed_diagnostic::kDriverFunctionalOnly;
  diagnostic.unit = typed_diagnostic::kUnitPrimitive;
  diagnostic.operation_kind = packet.operation_kind;
  diagnostic.semantic_plan_kind = execution->semantic_plan_kind;
  diagnostic.route_kind = execution->semantic_plan.route_kind;
  diagnostic.boundary_kind =
      execution->semantic_plan.route_kind ==
              primitive_semantic::kRouteAnyHitBoundary ||
          execution->semantic_plan.route_kind ==
              primitive_semantic::kRouteIntersectionBoundary ||
          execution->semantic_plan.route_kind ==
              primitive_semantic::kRouteFinalHitBoundary
          ? 1
          : 0;
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
