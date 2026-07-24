#include "rtcore_v04_functional_driver.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace functional_driver {

bool mode_selection_valid(bool functional_only_enabled,
                          bool timing_driver_enabled,
                          bool root_packet_enabled) {
  return !functional_only_enabled ||
         (!timing_driver_enabled && !root_packet_enabled);
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

status_kind execute_one_node(
    const fetch_target::operation_packet_v0 &packet,
    node_execution_v0 *execution) {
  if (execution == NULL) return kStatusInvalidArgument;
  *execution = node_execution_v0();
  status_kind status =
      prepare_node_operator_input(packet, &execution->operator_input);
  if (status != kStatusOk) return status;

  execution->operator_result =
      typed_node::execute_route(execution->operator_input);
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
    case kStatusTypedOperatorFailed:
      return "typed_operator_failed";
    case kStatusSemanticApplyFailed:
      return "semantic_apply_failed";
  }
  return "unknown";
}

}  // namespace functional_driver
}  // namespace v04
}  // namespace rtcore
