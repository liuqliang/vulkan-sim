#ifndef RTCORE_V04_TYPED_PRIMITIVE_KERNEL_H
#define RTCORE_V04_TYPED_PRIMITIVE_KERNEL_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_typed_stack_kernel.h"

namespace rtcore {
namespace v04 {
namespace typed_primitive {

static const uint32_t kGenRtDerivedProfileId = 0x00010001u;
static const uint8_t kProceduralPayloadKind = 3u;
static const uint8_t kQuadPayloadKind = 4u;
static const uint32_t kTriangleFrontCounterclockwise = 0x2u;
static const uint8_t kHitKindFrontFacing = 0xfeu;
static const uint8_t kHitKindBackFacing = 0xffu;
static const uint32_t kRayFlagOpaque = 0x00000001u;
static const uint32_t kRayFlagNoOpaque = 0x00000002u;
static const uint32_t kRayFlagTerminateOnFirstHit = 0x00000004u;
static const uint32_t kRayFlagSkipClosestHitShader = 0x00000008u;
static const uint32_t kRayFlagCullBackFacingTriangles = 0x00000010u;
static const uint32_t kRayFlagCullFrontFacingTriangles = 0x00000020u;
static const uint32_t kRayFlagCullOpaque = 0x00000040u;
static const uint32_t kRayFlagCullNoOpaque = 0x00000080u;
static const uint32_t kRayFlagSkipTriangles = 0x00000100u;
static const uint32_t kRayFlagSkipAabbs = 0x00000200u;
static const uint32_t kSupportedRayFlagMask = 0x000003ffu;
static const uint8_t kInstanceTriangleFacingCullDisable = 0x01u;
static const uint8_t kInstanceTriangleFrontCounterclockwise = 0x02u;
static const uint8_t kInstanceForceOpaque = 0x04u;
static const uint8_t kInstanceForceNoOpaque = 0x08u;
static const uint8_t kSupportedInstancePolicyMask = 0x0fu;
static const uint32_t kGeometryOpaque = 0x01u;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedProfile,
  kStatusMalformedEnvelope,
  kStatusMalformedLeaf,
  kStatusUnsupportedLeafEncoding,
  kStatusInvalidNumericInput,
  kStatusInvalidRouteInput,
  kStatusInvalidPolicy,
};

enum operation_kind : uint8_t {
  kOperationInvalid = 0,
  kOperationTestLeaf = 1,
};

enum route_result_kind : uint8_t {
  kRouteResultInvalid = 0,
  kRouteResultNoCandidate = 1,
  kRouteResultCommitOpaque = 2,
  kRouteResultAnyHitBoundary = 3,
  kRouteResultIntersectionBoundary = 4,
  kRouteResultFinalHit = 5,
};

enum route_output_valid_bit : uint8_t {
  kIdentityAndPolicyValid = 1u << 0,
  kTriangleHitValid = 1u << 1,
  kPrimitiveResumeValid = 1u << 2,
  kInternalFaultValid = 1u << 3,
  kIntersectionBoundaryValid = 1u << 4,
};

enum effective_policy_bit : uint8_t {
  kPolicyEffectiveOpaque = 1u << 0,
  kPolicyProceduralAnyHitEligible = 1u << 1,
};

enum geometry_type_kind : uint8_t {
  kGeometryTypeInvalid = 0,
  kGeometryTypeTriangle = 1,
  kGeometryTypeProcedural = 2,
};

struct raw_payload_header_v0 {
  uint8_t expected_payload_kind;
  uint8_t expected_chunk_count;
  uint16_t payload_byte_count;
  uint8_t received_chunk_mask;
  uint8_t reserved_zero[3];
};

struct raw_primitive_payload_v0 {
  raw_payload_header_v0 header;
  uint8_t raw_bytes[64];
};

struct ray_state_v0 {
  float origin[3];
  float direction[3];
  float t_min;
  float t_max;
};

struct alignas(16) candidate_input_v0 {
  uint32_t profile_id;
  uint32_t instance_flags;
  ray_state_v0 object_ray;
  float world_to_object_t_multiplier;
  float committed_world_t;
  float world_t_min;
  float world_t_max;
  raw_primitive_payload_v0 raw_primitive;
};

struct candidate_result_v0 {
  uint8_t status;
  uint8_t geometric_hit;
  uint8_t candidate_hit;
  uint8_t counter_clockwise_facing;
  uint8_t front_facing;
  uint8_t hit_kind;
  uint8_t geometry_ray_mask;
  uint8_t geometry_flags;
  uint32_t shader_index;
  uint32_t geometry_index;
  uint32_t primitive_index;
  uint32_t raw_quad_control;
  uint32_t object_t_bits;
  uint32_t world_t_bits;
  uint32_t bary_vertex1_bits;
  uint32_t bary_vertex2_bits;
  uint32_t determinant_bits;
  uint32_t reserved_zero;
};

struct alignas(16) procedural_input_v0 {
  uint32_t profile_id;
  uint32_t cull_mask;
  uint32_t reserved_zero[2];
  raw_primitive_payload_v0 raw_primitive;
};

struct procedural_result_v0 {
  uint8_t status;
  uint8_t mask_visible;
  uint8_t primitive_count;
  uint8_t leaf_type;
  uint8_t geometry_ray_mask;
  uint8_t geometry_flags;
  uint16_t last_primitive;
  uint32_t shader_index;
  uint32_t geometry_index;
  uint32_t primitive_index;
  uint32_t raw_control;
  uint32_t reserved_zero[2];
};

struct geometry_policy_projection_v0 {
  uint32_t geometry_flags;
  uint32_t pipeline_policy_bits;
};

struct primitive_identity_policy_facts_v0 {
  uint64_t instance_metadata_ref;
  uint32_t primitive_index;
  uint32_t geometry_index;
  uint32_t instance_index;
  uint32_t instance_custom_index;
  uint32_t instance_sbt_contribution;
  uint8_t geometry_type;
  uint8_t geometry_policy_flags;
  uint8_t instance_policy_flags;
  uint8_t effective_policy_flags;
};

struct triangle_hit_facts_v0 {
  uint32_t hit_t_bits;
  uint32_t bary_vertex1_bits;
  uint32_t bary_vertex2_bits;
  uint8_t hit_kind;
  uint8_t reserved_zero[3];
};

struct primitive_resume_data_v0 {
  uint64_t leaf_fetch_address;
  uint64_t remaining_slot_mask;
};

struct intersection_boundary_facts_v0 {
  uint32_t boundary_ray_tmax_bits;
  uint32_t reserved_zero;
};

struct internal_fault_facts_v0 {
  uint32_t fault_code;
  uint32_t reserved_zero;
};

struct alignas(16) route_input_v0 {
  uint32_t profile_id;
  uint8_t operation_kind;
  uint8_t reserved_zero[7];
  typed_stack::mutable_ray_state_v0 ray;
  typed_node::ray_policy_v0 ray_policy;
  uint64_t leaf_fetch_address;
  raw_primitive_payload_v0 raw_primitive;
  uint64_t input_slot_mask;
  typed_blas::as_decode_context_v0 decode_context;
  typed_stack::instance_shader_projection_v0 current_instance;
  typed_stack::committed_hit_projection_v0 current_committed_hit;
  geometry_policy_projection_v0 geometry_policy;
};

struct alignas(16) route_result_v0 {
  uint8_t status;
  uint8_t result_kind;
  uint8_t output_valid_mask;
  uint8_t typed_operator_invocation_count;
  uint8_t reserved_zero[12];
  primitive_identity_policy_facts_v0 identity_and_policy;
  triangle_hit_facts_v0 triangle_hit;
  primitive_resume_data_v0 primitive_resume;
  intersection_boundary_facts_v0 intersection_boundary;
  internal_fault_facts_v0 internal_fault;
};

static_assert(sizeof(raw_payload_header_v0) == 8,
              "raw payload header must remain 8 bytes");
static_assert(sizeof(raw_primitive_payload_v0) == 72,
              "GEN_RT raw primitive envelope must remain 72 bytes");
static_assert(sizeof(ray_state_v0) == 32,
              "local typed object ray input must remain 32 bytes");
static_assert(sizeof(candidate_input_v0) == 128,
              "typed primitive candidate input must remain 128 bytes");
static_assert(sizeof(candidate_result_v0) == 48,
              "typed primitive candidate result must remain 48 bytes");
static_assert(sizeof(procedural_input_v0) == 96,
              "typed procedural boundary input must remain 96 bytes");
static_assert(alignof(procedural_input_v0) == 16,
              "typed procedural boundary input must remain 16-byte aligned");
static_assert(offsetof(procedural_input_v0, profile_id) == 0,
              "typed procedural profile id offset changed");
static_assert(offsetof(procedural_input_v0, cull_mask) == 4,
              "typed procedural cull mask offset changed");
static_assert(offsetof(procedural_input_v0, raw_primitive) == 16,
              "typed procedural raw payload offset changed");
static_assert(sizeof(procedural_result_v0) == 32,
              "typed procedural boundary result must remain 32 bytes");
static_assert(offsetof(procedural_result_v0, status) == 0,
              "typed procedural status offset changed");
static_assert(offsetof(procedural_result_v0, last_primitive) == 6,
              "typed procedural last marker offset changed");
static_assert(offsetof(procedural_result_v0, shader_index) == 8,
              "typed procedural shader index offset changed");
static_assert(offsetof(procedural_result_v0, geometry_index) == 12,
              "typed procedural geometry index offset changed");
static_assert(offsetof(procedural_result_v0, primitive_index) == 16,
              "typed procedural primitive index offset changed");
static_assert(offsetof(procedural_result_v0, raw_control) == 20,
              "typed procedural raw control offset changed");
static_assert(sizeof(geometry_policy_projection_v0) == 8,
              "Primitive geometry policy must remain 8 bytes");
static_assert(sizeof(primitive_identity_policy_facts_v0) == 32,
              "Primitive identity/policy facts must remain 32 bytes");
static_assert(sizeof(triangle_hit_facts_v0) == 16,
              "Primitive triangle facts must remain 16 bytes");
static_assert(sizeof(primitive_resume_data_v0) == 16,
              "Primitive resume data must remain 16 bytes");
static_assert(sizeof(intersection_boundary_facts_v0) == 8,
              "Primitive intersection boundary must remain 8 bytes");
static_assert(sizeof(internal_fault_facts_v0) == 8,
              "Primitive internal fault facts must remain 8 bytes");
static_assert(sizeof(route_input_v0) == 288,
              "Primitive route input must remain 288 bytes");
static_assert(alignof(route_input_v0) == 16,
              "Primitive route input must remain aligned");
static_assert(offsetof(route_input_v0, ray) == 12,
              "Primitive route ray offset changed");
static_assert(offsetof(route_input_v0, ray_policy) == 56,
              "Primitive route ray policy offset changed");
static_assert(offsetof(route_input_v0, leaf_fetch_address) == 64,
              "Primitive route leaf address offset changed");
static_assert(offsetof(route_input_v0, raw_primitive) == 72,
              "Primitive route raw payload offset changed");
static_assert(offsetof(route_input_v0, input_slot_mask) == 144,
              "Primitive route input mask offset changed");
static_assert(offsetof(route_input_v0, decode_context) == 152,
              "Primitive route decode context offset changed");
static_assert(offsetof(route_input_v0, current_instance) == 192,
              "Primitive route current Instance offset changed");
static_assert(offsetof(route_input_v0, current_committed_hit) == 216,
              "Primitive route committed hit offset changed");
static_assert(offsetof(route_input_v0, geometry_policy) == 280,
              "Primitive route geometry policy offset changed");
static_assert(sizeof(route_result_v0) == 96,
              "Primitive route result must remain 96 bytes");
static_assert(alignof(route_result_v0) == 16,
              "Primitive route result must remain aligned");
static_assert(offsetof(route_result_v0, identity_and_policy) == 16,
              "Primitive identity/policy result offset changed");
static_assert(offsetof(route_result_v0, triangle_hit) == 48,
              "Primitive triangle result offset changed");
static_assert(offsetof(route_result_v0, primitive_resume) == 64,
              "Primitive resume result offset changed");

bool make_raw_primitive_payload(const void *raw_primitive_bytes,
                                raw_primitive_payload_v0 *payload);

bool make_raw_procedural_payload(const void *raw_primitive_bytes,
                                 raw_primitive_payload_v0 *payload);

candidate_result_v0 execute(const candidate_input_v0 &input);

procedural_result_v0 execute_procedural(const procedural_input_v0 &input);

route_result_v0 execute_route(const route_input_v0 &input);

const char *status_name(status_kind status);

}  // namespace typed_primitive
}  // namespace v04
}  // namespace rtcore

#endif
