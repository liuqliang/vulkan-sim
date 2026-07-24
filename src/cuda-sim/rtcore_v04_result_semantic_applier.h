#ifndef RTCORE_V04_RESULT_SEMANTIC_APPLIER_H
#define RTCORE_V04_RESULT_SEMANTIC_APPLIER_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_fetch_target_queue.h"

namespace rtcore {
namespace v04 {
namespace result_semantic {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOperationPacket,
  kStatusInvalidTypedResult,
  kStatusInvalidSelectedFetch,
  kStatusInvalidFrontier,
};

enum node_route_kind : uint8_t {
  kNodeRouteInvalid = 0,
  kNodeRouteNoChild = 1,
  kNodeRouteDirectChild = 2,
  kNodeRouteMultiChildToStack = 3,
};

struct node_commit_plan_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t producer_operation_seq;
  uint8_t valid;
  uint8_t route_kind;
  uint8_t next_target_kind;
  uint8_t frontier_count;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  typed_node::compact_child_work_item_v0
      frontier[typed_node::kMaxChildren - 1];
};

static_assert(offsetof(node_commit_plan_v0, selected_fetch) == 32,
              "Node semantic plan selected-fetch offset changed");

bool validate_node_operation_packet(
    const fetch_target::operation_packet_v0 &packet);

status_kind prepare_node_result(
    const fetch_target::operation_packet_v0 &packet,
    const typed_node::route_result_v0 &result,
    node_commit_plan_v0 *plan);

bool validate_node_commit_plan(
    const node_commit_plan_v0 &plan,
    const fetch_target::operation_packet_v0 &packet,
    const typed_node::route_result_v0 &result);

const char *status_name(status_kind status);
const char *node_route_name(node_route_kind route);

}  // namespace result_semantic
}  // namespace v04
}  // namespace rtcore

#endif
