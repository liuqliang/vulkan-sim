#include "rtcore_v04_functional_driver.h"

#include <cstdio>
#include <cstring>

#include "rtcore_v04_typed_diagnostic_collector.h"

namespace rtcore {
namespace v04 {
namespace functional_driver {

bool mode_selection_valid(bool functional_only_enabled,
                          bool timing_driver_enabled,
                          bool root_packet_enabled,
                          bool live_node_timing_enabled) {
  if (functional_only_enabled) {
    return !timing_driver_enabled && !root_packet_enabled &&
           !live_node_timing_enabled;
  }
  return !live_node_timing_enabled ||
         (timing_driver_enabled && root_packet_enabled);
}

status_kind prepare_node_operator_input(
    const fetch_target::operation_packet_v0 &packet,
    typed_node::route_input_v0 *input) {
  if (input == NULL) return kStatusInvalidArgument;
  *input = typed_node::route_input_v0();
  if (!result_semantic::validate_node_operation_packet(packet)) {
    return kStatusInvalidOperationPacket;
  }

  input->candidate.profile_id =
      packet.private_operands.decode_context.bvh_format_profile_id;
  input->candidate.level = packet.target_reference.level;
  const private_frontier::mutable_ray_state_v0 &ray =
      packet.private_operands.mutable_ray;
  std::memcpy(input->candidate.ray.origin, ray.origin, sizeof(ray.origin));
  std::memcpy(input->candidate.ray.direction, ray.direction,
              sizeof(ray.direction));
  std::memcpy(input->candidate.ray.inverse_direction,
              ray.inverse_direction, sizeof(ray.inverse_direction));
  input->candidate.ray.t_min = ray.t_min;
  input->candidate.ray.t_max = ray.t_max;
  input->candidate.policy = packet.ray_policy;
  input->candidate.committed_t =
      packet.private_operands.committed_hit.valid != 0
          ? packet.private_operands.committed_hit.hit_t
          : ray.t_max;
  if (!typed_node::make_raw_node_payload(
          packet.raw_payload, &input->candidate.raw_node)) {
    return kStatusInvalidOperationPacket;
  }
  input->decode_context = packet.private_operands.decode_context;
  input->current_payload_offset =
      packet.target_reference.payload_offset;
  return kStatusOk;
}

status_kind execute_node_operator_once(
    const fetch_target::operation_packet_v0 &packet,
    typed_node::route_result_v0 *result,
    uint8_t *operator_invocation_count) {
  if (result == NULL || operator_invocation_count == NULL) {
    return kStatusInvalidArgument;
  }
  *result = typed_node::route_result_v0();
  *operator_invocation_count = 0;
  typed_node::route_input_v0 input = {};
  const status_kind status = prepare_node_operator_input(packet, &input);
  if (status != kStatusOk) return status;
  *result = typed_node::execute_route(input);
  *operator_invocation_count = 1;
  return result->status == typed_node::kStatusOk
             ? kStatusOk
             : kStatusTypedOperatorFailed;
}

status_kind execute_one_node(
    const fetch_target::operation_packet_v0 &packet,
    node_execution_v0 *execution) {
  typed_node::replay_cursor_v0 replay_cursor = {};
  return execute_one_node(packet, replay_cursor, execution);
}

status_kind execute_one_node(
    const fetch_target::operation_packet_v0 &packet,
    const typed_node::replay_cursor_v0 &replay_cursor,
    node_execution_v0 *execution) {
  if (execution == NULL) return kStatusInvalidArgument;
  *execution = node_execution_v0();
  status_kind status = prepare_node_operator_input(
      packet, &execution->operator_input);
  if (status != kStatusOk) return status;
  typed_node::candidate_result_v0 candidate_result = {};
  execution->operator_result = typed_node::execute_route(
      execution->operator_input, replay_cursor, &candidate_result);
  execution->operator_invocation_count = 1;
  if (execution->operator_result.status != typed_node::kStatusOk) {
    return kStatusTypedOperatorFailed;
  }

  const result_semantic::status_kind semantic_status =
      result_semantic::prepare_node_result(
          packet, execution->operator_result, &execution->semantic_plan);
  if (semantic_status != result_semantic::kStatusOk) {
    return kStatusSemanticApplyFailed;
  }
  if (typed_diagnostic::recording_required()) {
    const typed_node::candidate_input_v0 &candidate =
        execution->operator_input.candidate;
    std::printf(
        "GPGPU-Sim RTCORE_V04_TYPED_NODE_SUMMARY "
        "driver=functional_only operation_seq=%u level=%u "
        "current_payload_offset=%llu ray_origin=(%.9g,%.9g,%.9g) "
        "ray_direction=(%.9g,%.9g,%.9g) ray_tmin=%.9g ray_tmax=%.9g "
        "committed_t=%.9g evaluated_child_mask=0x%02x "
        "hit_child_mask=0x%02x candidate_count=%u result_kind=%u "
        "semantic_route_kind=%u\n",
        packet.target_operation_seq, candidate.level,
        static_cast<unsigned long long>(
            execution->operator_input.current_payload_offset),
        candidate.ray.origin[0], candidate.ray.origin[1],
        candidate.ray.origin[2], candidate.ray.direction[0],
        candidate.ray.direction[1], candidate.ray.direction[2],
        candidate.ray.t_min, candidate.ray.t_max,
        candidate.committed_t,
        candidate_result.evaluated_child_mask,
        candidate_result.hit_child_mask,
        candidate_result.candidate_count,
        execution->operator_result.result_kind,
        execution->semantic_plan.route_kind);
    std::fflush(stdout);
  }
  typed_diagnostic::record_v0 diagnostic = {};
  diagnostic.owner = packet.owner;
  diagnostic.operation_seq = packet.target_operation_seq;
  diagnostic.driver = typed_diagnostic::kDriverFunctionalOnly;
  diagnostic.unit = typed_diagnostic::kUnitNode;
  diagnostic.operation_kind = packet.operation_kind;
  diagnostic.semantic_plan_kind = kSemanticPlanNode;
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

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidOperationPacket:
      return "invalid_operation_packet";
    case kStatusCanonicalInputMismatch:
      return "canonical_input_mismatch";
    case kStatusTypedOperatorFailed:
      return "typed_operator_failed";
    case kStatusFrontierCapacityExceeded:
      return "frontier_capacity_exceeded";
    case kStatusSemanticApplyFailed:
      return "semantic_apply_failed";
    case kStatusDiagnosticRejected:
      return "diagnostic_rejected";
  }
  return "unknown";
}

}  // namespace functional_driver
}  // namespace v04
}  // namespace rtcore
