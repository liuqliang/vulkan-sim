#ifndef RTCORE_V04_PRIMITIVE_RESULT_SEMANTIC_APPLIER_H
#define RTCORE_V04_PRIMITIVE_RESULT_SEMANTIC_APPLIER_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_typed_primitive_kernel.h"

namespace rtcore {
namespace v04 {
namespace primitive_semantic {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOperationIdentity,
  kStatusInvalidTypedResult,
};

enum route_kind : uint8_t {
  kRouteInvalid = 0,
  kRouteStackPopNext = 1,
  kRouteAnyHitBoundary = 2,
  kRouteIntersectionBoundary = 3,
  kRouteFinalHitBoundary = 4,
};

struct retained_candidate_projection_v0 {
  typed_primitive::primitive_identity_policy_facts_v0
      identity_and_policy;
  typed_primitive::triangle_hit_facts_v0 triangle_hit;
};

struct semantic_plan_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint8_t valid;
  uint8_t route_kind;
  uint8_t committed_hit_valid;
  uint8_t retained_candidate_valid;
  uint8_t primitive_resume_valid;
  uint8_t intersection_boundary_valid;
  uint8_t reserved_zero[2];
  typed_stack::committed_hit_projection_v0 committed_hit;
  retained_candidate_projection_v0 retained_candidate;
  typed_primitive::primitive_resume_data_v0 primitive_resume;
  typed_primitive::intersection_boundary_facts_v0
      intersection_boundary;
};

static_assert(sizeof(retained_candidate_projection_v0) == 48,
              "Primitive retained candidate must remain 48 bytes");
static_assert(offsetof(retained_candidate_projection_v0, triangle_hit) == 32,
              "Primitive retained triangle facts offset changed");

status_kind prepare_result(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const typed_primitive::route_input_v0 &input,
    const typed_primitive::route_result_v0 &result,
    semantic_plan_v0 *plan);

const char *status_name(status_kind status);
const char *route_name(route_kind route);

}  // namespace primitive_semantic
}  // namespace v04
}  // namespace rtcore

#endif
