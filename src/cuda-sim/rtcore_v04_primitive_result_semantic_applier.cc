#include "rtcore_v04_primitive_result_semantic_applier.h"

#include <cmath>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace primitive_semantic {
namespace {

static const uint8_t kInlineAttributeLocation = 1;
static const uint8_t kTriangleBarycentricFormat = 1;

bool bytes_are_zero(const void *value, size_t byte_count) {
  const uint8_t *bytes = static_cast<const uint8_t *>(value);
  for (size_t index = 0; index < byte_count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

float fp32_value(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool expected_effective_opaque(
    const typed_primitive::route_input_v0 &input) {
  if ((input.ray_policy.ray_flags &
       typed_primitive::kRayFlagOpaque) != 0) {
    return true;
  }
  if ((input.ray_policy.ray_flags &
       typed_primitive::kRayFlagNoOpaque) != 0) {
    return false;
  }
  if ((input.current_instance.instance_policy_flags &
       typed_primitive::kInstanceForceOpaque) != 0) {
    return true;
  }
  if ((input.current_instance.instance_policy_flags &
       typed_primitive::kInstanceForceNoOpaque) != 0) {
    return false;
  }
  return (input.geometry_policy.geometry_flags &
          typed_primitive::kGeometryOpaque) != 0;
}

bool identity_matches_input(
    const typed_primitive::primitive_identity_policy_facts_v0 &facts,
    const typed_primitive::route_input_v0 &input,
    uint8_t expected_geometry_type) {
  const bool opaque = expected_effective_opaque(input);
  uint8_t expected_effective =
      opaque ? typed_primitive::kPolicyEffectiveOpaque : 0;
  if (expected_geometry_type ==
          typed_primitive::kGeometryTypeProcedural &&
      !opaque) {
    expected_effective = static_cast<uint8_t>(
        expected_effective |
        typed_primitive::kPolicyProceduralAnyHitEligible);
  }
  return facts.instance_metadata_ref ==
             input.current_instance.instance_metadata_ref &&
         facts.instance_index ==
             input.current_instance.instance_index &&
         facts.instance_custom_index ==
             input.current_instance.instance_custom_index &&
         facts.instance_sbt_contribution ==
             input.current_instance.instance_sbt_contribution &&
         facts.geometry_type == expected_geometry_type &&
         facts.geometry_policy_flags ==
             input.geometry_policy.geometry_flags &&
         facts.instance_policy_flags ==
             input.current_instance.instance_policy_flags &&
         facts.effective_policy_flags == expected_effective;
}

bool triangle_facts_valid(
    const typed_primitive::triangle_hit_facts_v0 &facts,
    const typed_primitive::route_input_v0 &input) {
  const float hit_t = fp32_value(facts.hit_t_bits);
  return std::isfinite(hit_t) && hit_t >= input.ray.t_min &&
         hit_t <= input.ray.t_max &&
         (input.current_committed_hit.valid == 0 ||
          hit_t < input.current_committed_hit.hit_t) &&
         std::isfinite(fp32_value(facts.bary_vertex1_bits)) &&
         std::isfinite(fp32_value(facts.bary_vertex2_bits)) &&
         (facts.hit_kind == typed_primitive::kHitKindFrontFacing ||
          facts.hit_kind == typed_primitive::kHitKindBackFacing) &&
         bytes_are_zero(facts.reserved_zero,
                        sizeof(facts.reserved_zero));
}

bool common_result_shape_valid(
    const typed_primitive::route_result_v0 &result) {
  return result.status == typed_primitive::kStatusOk &&
         result.typed_operator_invocation_count == 1 &&
         bytes_are_zero(result.reserved_zero,
                        sizeof(result.reserved_zero)) &&
         bytes_are_zero(&result.primitive_resume,
                        sizeof(result.primitive_resume)) &&
         bytes_are_zero(&result.internal_fault,
                        sizeof(result.internal_fault));
}

void form_triangle_committed_hit(
    const typed_primitive::route_result_v0 &result,
    typed_stack::committed_hit_projection_v0 *hit) {
  *hit = typed_stack::committed_hit_projection_v0();
  hit->valid = 1;
  hit->geometry_type = typed_primitive::kGeometryTypeTriangle;
  hit->hit_kind = result.triangle_hit.hit_kind;
  hit->attribute_word_count = 2;
  hit->attribute_location = kInlineAttributeLocation;
  hit->attribute_format = kTriangleBarycentricFormat;
  hit->hit_t = fp32_value(result.triangle_hit.hit_t_bits);
  hit->policy_flags =
      result.identity_and_policy.effective_policy_flags;
  hit->instance_metadata_ref =
      result.identity_and_policy.instance_metadata_ref;
  hit->primitive_index = result.identity_and_policy.primitive_index;
  hit->geometry_index = result.identity_and_policy.geometry_index;
  hit->instance_index = result.identity_and_policy.instance_index;
  hit->instance_custom_index =
      result.identity_and_policy.instance_custom_index;
  hit->instance_sbt_contribution =
      result.identity_and_policy.instance_sbt_contribution;
  hit->inline_attributes[0] =
      result.triangle_hit.bary_vertex1_bits;
  hit->inline_attributes[1] =
      result.triangle_hit.bary_vertex2_bits;
}

}  // namespace

status_kind prepare_result(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq,
    const typed_primitive::route_input_v0 &input,
    const typed_primitive::route_result_v0 &result,
    semantic_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = semantic_plan_v0();
  if (operation_seq == 0) return kStatusInvalidOperationIdentity;
  if (!common_result_shape_valid(result)) {
    return kStatusInvalidTypedResult;
  }

  semantic_plan_v0 prepared = {};
  prepared.owner = owner;
  prepared.operation_seq = operation_seq;
  const typed_primitive::primitive_identity_policy_facts_v0
      empty_identity = {};
  const typed_primitive::triangle_hit_facts_v0 empty_triangle = {};
  const typed_primitive::intersection_boundary_facts_v0
      empty_intersection = {};

  switch (result.result_kind) {
    case typed_primitive::kRouteResultNoCandidate:
      if (result.output_valid_mask != 0 ||
          std::memcmp(&result.identity_and_policy, &empty_identity,
                      sizeof(empty_identity)) != 0 ||
          std::memcmp(&result.triangle_hit, &empty_triangle,
                      sizeof(empty_triangle)) != 0 ||
          std::memcmp(&result.intersection_boundary,
                      &empty_intersection,
                      sizeof(empty_intersection)) != 0) {
        return kStatusInvalidTypedResult;
      }
      prepared.route_kind = kRouteStackPopNext;
      break;

    case typed_primitive::kRouteResultCommitOpaque:
    case typed_primitive::kRouteResultFinalHit:
    case typed_primitive::kRouteResultAnyHitBoundary:
      if (result.output_valid_mask !=
              static_cast<uint8_t>(
                  typed_primitive::kIdentityAndPolicyValid |
                  typed_primitive::kTriangleHitValid) ||
          !identity_matches_input(
              result.identity_and_policy, input,
              typed_primitive::kGeometryTypeTriangle) ||
          !triangle_facts_valid(result.triangle_hit, input) ||
          !bytes_are_zero(&result.intersection_boundary,
                          sizeof(result.intersection_boundary))) {
        return kStatusInvalidTypedResult;
      }
      if (result.result_kind ==
          typed_primitive::kRouteResultAnyHitBoundary) {
        if ((result.identity_and_policy.effective_policy_flags &
             typed_primitive::kPolicyEffectiveOpaque) != 0) {
          return kStatusInvalidTypedResult;
        }
        prepared.route_kind = kRouteAnyHitBoundary;
        prepared.retained_candidate_valid = 1;
        prepared.retained_candidate.identity_and_policy =
            result.identity_and_policy;
        prepared.retained_candidate.triangle_hit =
            result.triangle_hit;
      } else {
        if ((result.identity_and_policy.effective_policy_flags &
             typed_primitive::kPolicyEffectiveOpaque) == 0) {
          return kStatusInvalidTypedResult;
        }
        prepared.route_kind =
            result.result_kind ==
                    typed_primitive::kRouteResultFinalHit
                ? kRouteFinalHitBoundary
                : kRouteStackPopNext;
        prepared.committed_hit_valid = 1;
        form_triangle_committed_hit(result,
                                    &prepared.committed_hit);
      }
      break;

    case typed_primitive::kRouteResultIntersectionBoundary: {
      const float expected_boundary_tmax =
          input.current_committed_hit.valid != 0
              ? input.current_committed_hit.hit_t
              : input.ray.t_max;
      if (result.output_valid_mask !=
              static_cast<uint8_t>(
                  typed_primitive::kIdentityAndPolicyValid |
                  typed_primitive::kIntersectionBoundaryValid) ||
          !identity_matches_input(
              result.identity_and_policy, input,
              typed_primitive::kGeometryTypeProcedural) ||
          !bytes_are_zero(&result.triangle_hit,
                          sizeof(result.triangle_hit)) ||
          result.intersection_boundary.reserved_zero != 0 ||
          result.intersection_boundary.boundary_ray_tmax_bits !=
              fp32_bits(expected_boundary_tmax)) {
        return kStatusInvalidTypedResult;
      }
      prepared.route_kind = kRouteIntersectionBoundary;
      prepared.retained_candidate_valid = 1;
      prepared.intersection_boundary_valid = 1;
      prepared.retained_candidate.identity_and_policy =
          result.identity_and_policy;
      prepared.intersection_boundary =
          result.intersection_boundary;
      break;
    }

    default:
      return kStatusInvalidTypedResult;
  }

  prepared.valid = 1;
  *plan = prepared;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidOperationIdentity:
      return "invalid_operation_identity";
    case kStatusInvalidTypedResult:
      return "invalid_typed_result";
  }
  return "unknown";
}

const char *route_name(route_kind route) {
  switch (route) {
    case kRouteInvalid:
      return "invalid";
    case kRouteStackPopNext:
      return "stack_pop_next";
    case kRouteAnyHitBoundary:
      return "any_hit_boundary";
    case kRouteIntersectionBoundary:
      return "intersection_boundary";
    case kRouteFinalHitBoundary:
      return "final_hit_boundary";
  }
  return "unknown";
}

}  // namespace primitive_semantic
}  // namespace v04
}  // namespace rtcore
