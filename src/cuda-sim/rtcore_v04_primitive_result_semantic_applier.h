#ifndef RTCORE_V04_PRIMITIVE_RESULT_SEMANTIC_APPLIER_H
#define RTCORE_V04_PRIMITIVE_RESULT_SEMANTIC_APPLIER_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_typed_primitive_kernel.h"

namespace rtcore {
namespace v04 {
namespace primitive_semantic {

static const uint8_t kMaxWriteFragmentCount = 5;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOperationIdentity,
  kStatusInvalidTypedResult,
  kStatusLayoutRejected,
  kStatusInvalidWriteFragment,
};

enum route_kind : uint8_t {
  kRouteInvalid = 0,
  kRouteStackPopNext = 1,
  kRouteAnyHitBoundary = 2,
  kRouteIntersectionBoundary = 3,
  kRouteFinalHitBoundary = 4,
};

typedef private_frontier::retained_candidate_projection_v0
    retained_candidate_projection_v0;

struct private_write_fragment_v0 {
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
  uint16_t slot_byte_offset;
  uint8_t byte_count;
  uint8_t field_kind;
  uint8_t payload[private_frontier::kSharedAccessChunkBytes];
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
  uint8_t shader_return_valid;
  uint8_t reserved_zero;
  typed_stack::committed_hit_projection_v0 committed_hit;
  retained_candidate_projection_v0 retained_candidate;
  typed_primitive::primitive_resume_data_v0 primitive_resume;
  typed_primitive::intersection_boundary_facts_v0
      intersection_boundary;
};

struct private_commit_plan_v0 {
  semantic_plan_v0 semantic_plan;
  uint8_t valid;
  uint8_t write_fragment_count;
  uint8_t required_ack_count;
  uint8_t reserved_zero[5];
  private_write_fragment_v0
      write_fragments[kMaxWriteFragmentCount];
};

static_assert(sizeof(retained_candidate_projection_v0) == 48,
              "Primitive retained candidate must remain 48 bytes");
static_assert(offsetof(retained_candidate_projection_v0, triangle_hit) == 32,
              "Primitive retained triangle facts offset changed");
static_assert(sizeof(private_write_fragment_v0) == 48,
              "Primitive private write fragment must remain 48 bytes");
static_assert(offsetof(private_write_fragment_v0, payload) == 16,
              "Primitive private write payload offset changed");

status_kind prepare_result(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const typed_primitive::route_input_v0 &input,
    const typed_primitive::route_result_v0 &result,
    semantic_plan_v0 *plan);

bool validate_private_write_fragment(
    const private_write_fragment_v0 &fragment);

status_kind prepare_private_commit(
    const semantic_plan_v0 &semantic_plan,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    private_commit_plan_v0 *plan);

const char *status_name(status_kind status);
const char *route_name(route_kind route);

}  // namespace primitive_semantic
}  // namespace v04
}  // namespace rtcore

#endif
