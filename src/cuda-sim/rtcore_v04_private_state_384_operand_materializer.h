#ifndef RTCORE_V04_PRIVATE_STATE_384_OPERAND_MATERIALIZER_H
#define RTCORE_V04_PRIVATE_STATE_384_OPERAND_MATERIALIZER_H

#include <cstdint>

#include "rtcore_v04_private_state_384_operand_plan.h"
#include "rtcore_v04_typed_instance_kernel.h"

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace operand_materializer {

static const uint8_t kMaxOperationReadChunks = 9;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidPlan,
  kStatusInvalidOperationIdentity,
  kStatusInvalidCollector,
  kStatusUnsupportedPrivateLayout,
  kStatusUnsupportedBvhProfile,
  kStatusNotInitialized,
  kStatusResponseIdentityMismatch,
  kStatusResponseShapeMismatch,
  kStatusUnrequestedChunk,
  kStatusDuplicateChunk,
  kStatusCollectorFailed,
  kStatusIncompleteResponses,
  kStatusInvalidConsumerOperation,
  kStatusInvalidEncoding,
};

struct operation_identity_v1 {
  uint32_t owner_hw_sid;
  uint32_t resident_warp_id;
  uint32_t request_identity;
  uint32_t request_generation;
  uint32_t private_slot_id;
  uint32_t operation_sequence;
  uint8_t lane_id;
  uint8_t reserved_zero[3];
};

struct chunk_response_v1 {
  operation_identity_v1 identity;
  uint32_t private_layout_profile_id;
  uint8_t consumer;
  uint8_t operation;
  uint8_t completion_reason;
  uint8_t chunk_index;
  uint16_t slot_byte_offset;
  uint16_t byte_count;
  uint8_t reserved_zero[4];
  uint8_t payload[kChunkBytes];
};

struct collected_chunk_v1 {
  uint8_t chunk_index;
  uint8_t valid;
  uint8_t reserved_zero[2];
  uint8_t payload[kChunkBytes];
};

// This is an operation-selected response set, not a 384-byte slot mirror.
struct response_collector_v1 {
  operation_identity_v1 identity;
  uint32_t private_layout_profile_id;
  uint16_t required_chunk_mask;
  uint16_t received_chunk_mask;
  uint8_t consumer;
  uint8_t operation;
  uint8_t completion_reason;
  uint8_t required_count;
  uint8_t received_count;
  uint8_t initialized;
  uint8_t failed;
  uint8_t reserved_zero;
  collected_chunk_v1 chunks[kMaxOperationReadChunks];
};

struct materialize_context_v1 {
  uint32_t bvh_format_profile_id;
  uint32_t reserved_zero;
  uint8_t recovery_target_inflight;
  uint8_t reserved_zero1[3];
};

struct node_operands_v1 {
  typed_node::ray_state_v0 ray;
  typed_node::ray_policy_v0 ray_policy;
  typed_blas::as_decode_context_v0 decode_context;
  float effective_traversal_bound;
  uint8_t committed_valid;
  uint8_t reserved_zero[3];
};

// Completion/resubmit transport these producer-validated facts but cannot
// issue a leaf read. Primitive-resume materialization revalidates the leaf
// address against its C1 active AS context before execution.
struct software_boundary_v1 {
  uint8_t reason;
  uint8_t reserved_zero[7];
  typed_primitive::primitive_identity_policy_facts_v0 identity_and_policy;
  typed_primitive::primitive_resume_data_v0 primitive_resume;
  union {
    typed_primitive::triangle_hit_facts_v0 triangle_hit;
    typed_primitive::intersection_boundary_facts_v0 intersection;
    uint8_t raw[16];
  } reason_facts;
};

struct primitive_operands_v1 {
  typed_stack::mutable_ray_state_v0 ray;
  typed_node::ray_policy_v0 ray_policy;
  typed_blas::as_decode_context_v0 decode_context;
  typed_stack::instance_shader_projection_v0 current_instance;
  typed_stack::committed_hit_projection_v0 committed_hit;
  float effective_traversal_bound;
  uint32_t reserved_zero;
};

struct primitive_resume_operands_v1 {
  primitive_operands_v1 persistent;
  software_boundary_v1 boundary;
};

struct instance_operands_v1 {
  typed_instance::mutable_ray_state_v0 world_ray;
  typed_instance::ray_policy_v0 ray_policy;
  typed_blas::as_decode_context_v0 tlas_decode_context;
  uint32_t tlas_build_generation;
  uint32_t reserved_zero;
};

struct stack_operands_v1 {
  typed_stack::mutable_ray_state_v0 ray;
  typed_node::ray_policy_v0 ray_policy;
  typed_blas::as_decode_context_v0 active_decode_context;
  short_stack::state_v0 stack;
  float effective_traversal_bound;
  uint32_t tlas_build_generation;
  uint32_t blas_build_generation;
  uint8_t committed_valid;
  uint8_t reserved_zero[3];
};

struct stack_terminal_operands_v1 {
  stack_operands_v1 base;
  typed_stack::committed_hit_projection_v0 committed_hit;
};

struct parent_restore_operands_v1 {
  typed_stack::mutable_ray_state_v0 ray;
  typed_node::ray_policy_v0 ray_policy;
  typed_blas::as_decode_context_v0 tlas_decode_context;
};

struct stack_cross_as_operands_v1 {
  stack_operands_v1 base;
  parent_restore_operands_v1 parent;
};

struct final_completion_operands_v1 {
  uint8_t completion_reason;
  uint8_t reserved_zero[7];
  typed_stack::committed_hit_projection_v0 committed_hit;
};

status_kind initialize_collector(
    const operand_plan::read_plan_v1 &plan,
    const operation_identity_v1 &identity,
    response_collector_v1 *collector);

status_kind accept_response(const chunk_response_v1 &response,
                            response_collector_v1 *collector);

bool responses_complete(const response_collector_v1 &collector);

status_kind promote_stack_collector(
    const response_collector_v1 &base_collector,
    uint8_t selected_operation,
    response_collector_v1 *selected_collector);

status_kind materialize_node(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    node_operands_v1 *operands);

status_kind materialize_primitive(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    primitive_operands_v1 *operands);

status_kind materialize_primitive_resume(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    primitive_resume_operands_v1 *operands);

status_kind materialize_instance(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    instance_operands_v1 *operands);

status_kind materialize_stack_base(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    stack_operands_v1 *operands);

status_kind materialize_stack_terminal(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    stack_terminal_operands_v1 *operands);

status_kind materialize_stack_cross_as(
    const response_collector_v1 &collector,
    const materialize_context_v1 &context,
    stack_cross_as_operands_v1 *operands);

status_kind materialize_final_completion(
    const response_collector_v1 &collector,
    final_completion_operands_v1 *operands);

status_kind materialize_boundary_completion(
    const response_collector_v1 &collector,
    software_boundary_v1 *operands);

status_kind materialize_resubmit(
    const response_collector_v1 &collector,
    software_boundary_v1 *operands);

const char *status_name(status_kind status);

static_assert(kMaxOperationReadChunks < kChunkCount,
              "operation collector must not become a full-slot mirror");
static_assert(sizeof(response_collector_v1) < kSlotBytes,
              "operation collector must remain smaller than one slot image");
static_assert(sizeof(collected_chunk_v1) == 36,
              "collected response shape changed");
static_assert(sizeof(operation_identity_v1) == 28,
              "operation response identity changed");
static_assert(sizeof(chunk_response_v1) == 76,
              "384B response envelope changed");
static_assert(sizeof(software_boundary_v1::reason_facts) == 16,
              "reason-selected boundary facts changed");

}  // namespace operand_materializer
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore

#endif
