#ifndef RTCORE_V04_FUNCTIONAL_DRIVER_H
#define RTCORE_V04_FUNCTIONAL_DRIVER_H

#include <cstdint>

#include "rtcore_v04_result_semantic_applier.h"

namespace rtcore {
namespace v04 {
namespace functional_driver {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOperationPacket,
  kStatusTypedOperatorFailed,
  kStatusSemanticApplyFailed,
};

struct node_execution_v0 {
  uint8_t valid;
  uint8_t operator_invocation_count;
  uint8_t reserved_zero[6];
  typed_node::route_input_v0 operator_input;
  typed_node::route_result_v0 operator_result;
  result_semantic::node_commit_plan_v0 semantic_plan;
};

bool mode_selection_valid(bool functional_only_enabled,
                          bool timing_driver_enabled,
                          bool root_packet_enabled,
                          bool live_node_timing_enabled = false);

status_kind prepare_node_operator_input(
    const fetch_target::operation_packet_v0 &packet,
    typed_node::route_input_v0 *input);

status_kind execute_node_operator_once(
    const fetch_target::operation_packet_v0 &packet,
    typed_node::route_result_v0 *result,
    uint8_t *operator_invocation_count);

status_kind execute_one_node(
    const fetch_target::operation_packet_v0 &packet,
    node_execution_v0 *execution);

const char *status_name(status_kind status);

}  // namespace functional_driver
}  // namespace v04
}  // namespace rtcore

#endif
