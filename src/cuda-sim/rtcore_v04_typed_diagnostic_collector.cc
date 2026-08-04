#include "rtcore_v04_typed_diagnostic_collector.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rtcore_v04_conservation_recorder.h"
#include "rtcore_v04_functional_driver.h"
#include "rtcore_v04_request_owner_binding.h"

namespace rtcore {
namespace v04 {
namespace typed_diagnostic {
namespace {

static const uint64_t kFnvOffsetBasis = UINT64_C(14695981039346656037);
static const uint64_t kFnvPrime = UINT64_C(1099511628211);

bool env_value_is_true(const char *value) {
  return value != NULL &&
         (std::strcmp(value, "1") == 0 ||
          std::strcmp(value, "true") == 0 ||
          std::strcmp(value, "on") == 0 ||
          std::strcmp(value, "yes") == 0);
}

bool optional_u32_filter_matches(const char *name, uint32_t actual) {
  const char *value = std::getenv(name);
  if (value == NULL || value[0] == '\0') return true;
  char *end = NULL;
  const unsigned long parsed = std::strtoul(value, &end, 0);
  return end != value && *end == '\0' && parsed <= UINT32_MAX &&
         static_cast<uint32_t>(parsed) == actual;
}

bool diagnostic_filter_matches_owner(
    const private_frontier::owner_binding_v0 &owner) {
  return optional_u32_filter_matches(
             "VULKAN_SIM_RTCORE_ABI_V04_TYPED_DIAGNOSTIC_OWNER_HW_SID",
             owner.owner_hw_sid) &&
         optional_u32_filter_matches(
             "VULKAN_SIM_RTCORE_ABI_V04_TYPED_DIAGNOSTIC_REQUEST_IDENTITY",
             owner.request_identity) &&
         optional_u32_filter_matches(
             "VULKAN_SIM_RTCORE_ABI_V04_TYPED_DIAGNOSTIC_LANE_ID",
             owner.lane_id);
}

bool diagnostic_filter_matches(const record_v0 &record) {
  return diagnostic_filter_matches_owner(record.owner);
}

uint64_t hash_bytes(const void *data, size_t byte_count) {
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  uint64_t hash = kFnvOffsetBasis;
  for (size_t index = 0; index < byte_count; ++index) {
    hash ^= bytes[index];
    hash *= kFnvPrime;
  }
  return hash;
}

bool bytes_are_zero(const uint8_t *bytes, size_t byte_count) {
  for (size_t index = 0; index < byte_count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool valid_driver(uint8_t driver) {
  return driver == kDriverFunctionalOnly || driver == kDriverTiming;
}

bool valid_unit(uint8_t unit) {
  return unit >= kUnitNode && unit <= kUnitPrimitive;
}

bool owner_matches(const private_frontier::owner_binding_v0 &lhs,
                   const private_frontier::owner_binding_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.resident_warp_id == rhs.resident_warp_id &&
         lhs.request_identity == rhs.request_identity &&
         lhs.generation == rhs.generation &&
         lhs.private_slot_id == rhs.private_slot_id &&
         lhs.lane_id == rhs.lane_id;
}

template <typename Plan>
void clear_plan_identity(Plan *plan) {
  std::memset(&plan->owner, 0, sizeof(plan->owner));
  plan->operation_seq = 0;
}

template <typename Fragment>
void clear_fragment_addresses(Fragment *fragments, uint8_t count,
                              uint8_t capacity) {
  const uint8_t bounded_count = count < capacity ? count : capacity;
  for (uint8_t index = 0; index < bounded_count; ++index) {
    fragments[index].aligned_32b_address = 0;
  }
}

void clear_decode_context_transport(
    typed_blas::as_decode_context_v0 *context) {
  context->device_base = 0;
}

void clear_selected_fetch_transport(
    typed_node::selected_child_fetch_work_item_v0 *selected) {
  clear_decode_context_transport(&selected->decode_context);
}

void clear_instance_projection_transport(
    typed_stack::instance_shader_projection_v0 *instance) {
  instance->instance_metadata_ref = 0;
}

void clear_committed_hit_transport(
    typed_stack::committed_hit_projection_v0 *hit) {
  hit->instance_metadata_ref = 0;
}

void clear_frame_transport(
    typed_stack::traversal_frame_projection_v0 *frame) {
  clear_decode_context_transport(&frame->current_decode_context);
  clear_instance_projection_transport(&frame->current_instance);
}

void clear_raw_instance_transport(
    typed_instance::raw_instance_payload_v0 *payload) {
  static const size_t kStartNodeAddressOffset = 8;
  static const size_t kStartNodeAddressBytes = 6;
  static const size_t kBvhAddressOffset = 64;
  static const size_t kBvhAddressBytes = 8;
  std::memset(
      payload->raw_bytes + kStartNodeAddressOffset, 0,
      kStartNodeAddressBytes);
  std::memset(
      payload->raw_bytes + kBvhAddressOffset, 0, kBvhAddressBytes);
}

uint64_t normalized_input_hash(const record_v0 &record) {
  namespace fd = functional_driver;
  if (record.unit == kUnitNode) {
    typed_node::route_input_v0 input =
        *static_cast<const typed_node::route_input_v0 *>(
            record.typed_input);
    clear_decode_context_transport(&input.decode_context);
    return hash_bytes(&input, sizeof(input));
  }
  if (record.unit == kUnitStack) {
    if (record.semantic_plan_kind == fd::kSemanticPlanStackAppend) {
      typed_stack::push_input_v0 input =
          *static_cast<const typed_stack::push_input_v0 *>(
              record.typed_input);
      clear_selected_fetch_transport(&input.node_route.selected_fetch);
      return hash_bytes(&input, sizeof(input));
    }
    if (record.semantic_plan_kind == fd::kSemanticPlanStackPop) {
      typed_stack::pop_input_v0 input =
          *static_cast<const typed_stack::pop_input_v0 *>(
              record.typed_input);
      clear_decode_context_transport(&input.current_decode_context);
      return hash_bytes(&input, sizeof(input));
    }
    typed_stack::empty_input_v0 input =
        *static_cast<const typed_stack::empty_input_v0 *>(
            record.typed_input);
    clear_committed_hit_transport(&input.current_committed_hit);
    clear_frame_transport(&input.parent_frame);
    return hash_bytes(&input, sizeof(input));
  }
  if (record.unit == kUnitInstance) {
    if (record.semantic_plan_kind == fd::kSemanticPlanInstanceEnter) {
      typed_instance::enter_input_v0 input =
          *static_cast<const typed_instance::enter_input_v0 *>(
              record.typed_input);
      clear_raw_instance_transport(&input.raw_instance);
      input.instance_blas_reference.instance_metadata_reference = 0;
      clear_decode_context_transport(&input.tlas_decode_context);
      clear_decode_context_transport(&input.blas_decode_context);
      return hash_bytes(&input, sizeof(input));
    }
    typed_instance::restore_parent_input_v0 input =
        *static_cast<const typed_instance::restore_parent_input_v0 *>(
            record.typed_input);
    clear_frame_transport(&input.parent_frame);
    return hash_bytes(&input, sizeof(input));
  }
  typed_primitive::route_input_v0 input =
      *static_cast<const typed_primitive::route_input_v0 *>(
          record.typed_input);
  input.leaf_fetch_address = 0;
  clear_decode_context_transport(&input.decode_context);
  clear_instance_projection_transport(&input.current_instance);
  clear_committed_hit_transport(&input.current_committed_hit);
  return hash_bytes(&input, sizeof(input));
}

uint64_t normalized_result_hash(const record_v0 &record) {
  namespace fd = functional_driver;
  if (record.unit == kUnitNode) {
    typed_node::route_result_v0 result =
        *static_cast<const typed_node::route_result_v0 *>(
            record.typed_result);
    clear_selected_fetch_transport(&result.selected_fetch);
    return hash_bytes(&result, sizeof(result));
  }
  if (record.unit == kUnitStack) {
    if (record.semantic_plan_kind == fd::kSemanticPlanStackAppend) {
      typed_stack::push_result_v0 result =
          *static_cast<const typed_stack::push_result_v0 *>(
              record.typed_result);
      clear_selected_fetch_transport(&result.selected_fetch);
      return hash_bytes(&result, sizeof(result));
    }
    if (record.semantic_plan_kind == fd::kSemanticPlanStackPop) {
      typed_stack::pop_result_v0 result =
          *static_cast<const typed_stack::pop_result_v0 *>(
              record.typed_result);
      clear_selected_fetch_transport(&result.selected_fetch);
      return hash_bytes(&result, sizeof(result));
    }
    typed_stack::empty_result_v0 result =
        *static_cast<const typed_stack::empty_result_v0 *>(
            record.typed_result);
    clear_frame_transport(&result.parent_frame);
    clear_committed_hit_transport(&result.terminal_hit);
    return hash_bytes(&result, sizeof(result));
  }
  if (record.unit == kUnitInstance) {
    if (record.semantic_plan_kind == fd::kSemanticPlanInstanceEnter) {
      typed_instance::enter_result_v0 result =
          *static_cast<const typed_instance::enter_result_v0 *>(
              record.typed_result);
      result.instance_projection.instance_metadata_reference = 0;
      clear_decode_context_transport(&result.root_fetch.decode_context);
      return hash_bytes(&result, sizeof(result));
    }
    typed_instance::restore_parent_result_v0 result =
        *static_cast<const typed_instance::restore_parent_result_v0 *>(
            record.typed_result);
    clear_frame_transport(&result.restored_parent);
    return hash_bytes(&result, sizeof(result));
  }
  typed_primitive::route_result_v0 result =
      *static_cast<const typed_primitive::route_result_v0 *>(
          record.typed_result);
  result.identity_and_policy.instance_metadata_ref = 0;
  result.primitive_resume.leaf_fetch_address = 0;
  return hash_bytes(&result, sizeof(result));
}

template <typename Input, typename Result, typename Plan>
bool fixed_record_shape(const record_v0 &record) {
  return record.typed_input_bytes == sizeof(Input) &&
         record.typed_result_bytes == sizeof(Result) &&
         record.semantic_plan_bytes == sizeof(Plan);
}

template <typename Plan>
bool plan_identity_matches(const record_v0 &record, const Plan &plan) {
  return owner_matches(record.owner, plan.owner) &&
         record.operation_seq == plan.operation_seq &&
         plan.valid != 0 && plan.route_kind == record.route_kind;
}

bool record_shape_valid(const record_v0 &record) {
  namespace fd = functional_driver;
  if (record.unit == kUnitNode) {
    if (record.operation_kind != fetch_target::kOperationFetchTarget ||
        record.semantic_plan_kind != fd::kSemanticPlanNode ||
        record.boundary_kind != 0 ||
        !fixed_record_shape<
            typed_node::route_input_v0, typed_node::route_result_v0,
            result_semantic::node_commit_plan_v0>(record)) {
      return false;
    }
    const result_semantic::node_commit_plan_v0 &plan =
        *static_cast<const result_semantic::node_commit_plan_v0 *>(
            record.semantic_plan);
    return owner_matches(record.owner, plan.owner) &&
           record.operation_seq == plan.producer_operation_seq &&
           plan.valid != 0 && plan.route_kind == record.route_kind;
  }

  if (record.unit == kUnitStack) {
    if (record.operation_kind ==
        typed_stack::kPushRemainderAndForwardSelected) {
      if (record.semantic_plan_kind != fd::kSemanticPlanStackAppend ||
          record.boundary_kind != 0 ||
          !fixed_record_shape<
              typed_stack::push_input_v0, typed_stack::push_result_v0,
              stack_semantic::append_commit_plan_v0>(record)) {
        return false;
      }
      const stack_semantic::append_commit_plan_v0 &plan =
          *static_cast<const stack_semantic::append_commit_plan_v0 *>(
              record.semantic_plan);
      return plan_identity_matches(record, plan);
    }
    if (record.operation_kind != typed_stack::kPopNext) return false;
    if (record.semantic_plan_kind == fd::kSemanticPlanStackPop ||
        record.semantic_plan_kind ==
            fd::kSemanticPlanStackRestoreParent) {
      const bool pop = record.semantic_plan_kind ==
                       fd::kSemanticPlanStackPop;
      const bool shape_valid =
          pop
              ? fixed_record_shape<
                    typed_stack::pop_input_v0,
                    typed_stack::pop_result_v0,
                    stack_semantic::pop_commit_plan_v0>(record)
              : fixed_record_shape<
                    typed_stack::empty_input_v0,
                    typed_stack::empty_result_v0,
                    stack_semantic::pop_commit_plan_v0>(record);
      if (!shape_valid || record.boundary_kind != 0) return false;
      const stack_semantic::pop_commit_plan_v0 &plan =
          *static_cast<const stack_semantic::pop_commit_plan_v0 *>(
              record.semantic_plan);
      return plan_identity_matches(record, plan);
    }
    const bool terminal_hit =
        record.semantic_plan_kind == fd::kSemanticPlanStackTerminalHit;
    const bool terminal_miss =
        record.semantic_plan_kind == fd::kSemanticPlanStackTerminalMiss;
    return (terminal_hit || terminal_miss) &&
           record.boundary_kind == 1 &&
           record.route_kind == record.semantic_plan_kind &&
           record.typed_input_bytes ==
               sizeof(typed_stack::empty_input_v0) &&
           record.typed_result_bytes ==
               sizeof(typed_stack::empty_result_v0) &&
           record.semantic_plan == NULL &&
           record.semantic_plan_bytes == 0;
  }

  if (record.unit == kUnitInstance) {
    if (record.boundary_kind != 0) return false;
    if (record.operation_kind == fetch_target::kOperationFetchTarget &&
        record.semantic_plan_kind ==
            fd::kSemanticPlanInstanceEnter &&
        fixed_record_shape<
            typed_instance::enter_input_v0,
            typed_instance::enter_result_v0,
            instance_semantic::enter_commit_plan_v0>(record)) {
      const instance_semantic::enter_commit_plan_v0 &plan =
          *static_cast<const instance_semantic::enter_commit_plan_v0 *>(
              record.semantic_plan);
      return plan_identity_matches(record, plan);
    }
    if (record.operation_kind ==
            fetch_target::kOperationInstanceRestoreParent &&
        record.semantic_plan_kind ==
            fd::kSemanticPlanInstanceRestore &&
        fixed_record_shape<
            typed_instance::restore_parent_input_v0,
            typed_instance::restore_parent_result_v0,
            instance_semantic::restore_commit_plan_v0>(record)) {
      const instance_semantic::restore_commit_plan_v0 &plan =
          *static_cast<const instance_semantic::restore_commit_plan_v0 *>(
              record.semantic_plan);
      return plan_identity_matches(record, plan);
    }
    return false;
  }

  if (record.unit == kUnitPrimitive) {
    if (record.operation_kind != fetch_target::kOperationFetchTarget ||
        record.semantic_plan_kind !=
            fd::kSemanticPlanPrimitive ||
        !fixed_record_shape<
            typed_primitive::route_input_v0,
            typed_primitive::route_result_v0,
            primitive_semantic::semantic_plan_v0>(record)) {
      return false;
    }
    const primitive_semantic::semantic_plan_v0 &plan =
        *static_cast<const primitive_semantic::semantic_plan_v0 *>(
            record.semantic_plan);
    const bool boundary =
        plan.route_kind == primitive_semantic::kRouteAnyHitBoundary ||
        plan.route_kind ==
            primitive_semantic::kRouteIntersectionBoundary ||
        plan.route_kind == primitive_semantic::kRouteFinalHitBoundary;
    return plan_identity_matches(record, plan) &&
           record.boundary_kind == (boundary ? 1 : 0);
  }
  return false;
}

uint64_t normalized_plan_hash(const record_v0 &record) {
  namespace fd = functional_driver;
  if (record.semantic_plan_bytes == 0) return kFnvOffsetBasis;
  if (record.semantic_plan_kind == fd::kSemanticPlanNode) {
    result_semantic::node_commit_plan_v0 plan =
        *static_cast<const result_semantic::node_commit_plan_v0 *>(
            record.semantic_plan);
    std::memset(&plan.owner, 0, sizeof(plan.owner));
    plan.producer_operation_seq = 0;
    return hash_bytes(&plan, sizeof(plan));
  }
  if (record.semantic_plan_kind ==
      fd::kSemanticPlanStackAppend) {
    stack_semantic::append_commit_plan_v0 plan =
        *static_cast<const stack_semantic::append_commit_plan_v0 *>(
            record.semantic_plan);
    clear_plan_identity(&plan);
    clear_fragment_addresses(
        plan.write_fragments, plan.write_fragment_count,
        stack_semantic::kMaxPrivateWriteFragments);
    return hash_bytes(&plan, sizeof(plan));
  }
  if (record.semantic_plan_kind == fd::kSemanticPlanStackPop ||
      record.semantic_plan_kind ==
          fd::kSemanticPlanStackRestoreParent) {
    stack_semantic::pop_commit_plan_v0 plan =
        *static_cast<const stack_semantic::pop_commit_plan_v0 *>(
            record.semantic_plan);
    clear_plan_identity(&plan);
    clear_fragment_addresses(
        plan.write_fragments, plan.write_fragment_count,
        stack_semantic::kMaxPopPrivateWriteFragments);
    return hash_bytes(&plan, sizeof(plan));
  }
  if (record.semantic_plan_kind ==
      fd::kSemanticPlanInstanceEnter) {
    instance_semantic::enter_commit_plan_v0 plan =
        *static_cast<const instance_semantic::enter_commit_plan_v0 *>(
            record.semantic_plan);
    clear_plan_identity(&plan);
    clear_fragment_addresses(
        plan.write_fragments, plan.write_fragment_count,
        instance_semantic::kMaxWriteFragmentCount);
    return hash_bytes(&plan, sizeof(plan));
  }
  if (record.semantic_plan_kind ==
      fd::kSemanticPlanInstanceRestore) {
    instance_semantic::restore_commit_plan_v0 plan =
        *static_cast<const instance_semantic::restore_commit_plan_v0 *>(
            record.semantic_plan);
    clear_plan_identity(&plan);
    clear_fragment_addresses(
        plan.write_fragments, plan.write_fragment_count,
        instance_semantic::kRestoreWriteFragmentCount);
    return hash_bytes(&plan, sizeof(plan));
  }
  primitive_semantic::semantic_plan_v0 plan =
      *static_cast<const primitive_semantic::semantic_plan_v0 *>(
          record.semantic_plan);
  clear_plan_identity(&plan);
  return hash_bytes(&plan, sizeof(plan));
}

}  // namespace

bool enabled() {
  static const bool value = env_value_is_true(
      std::getenv("VULKAN_SIM_RTCORE_ABI_V04_TYPED_DIAGNOSTIC_COLLECTOR"));
  return value;
}

bool recording_required() {
  return enabled() || conservation::enabled();
}

bool emit_record(const record_v0 &record) {
  if (!recording_required()) return true;
  if (!valid_driver(record.driver) || !valid_unit(record.unit) ||
      record.operation_seq == 0 ||
      !request_owner::validate_private_frontier_owner_identity(
          record.owner) ||
      record.typed_input == NULL || record.typed_input_bytes == 0 ||
      record.typed_result == NULL || record.typed_result_bytes == 0 ||
      (record.semantic_plan_bytes != 0 &&
       record.semantic_plan == NULL) ||
      !bytes_are_zero(record.reserved_zero,
                      sizeof(record.reserved_zero)) ||
      !record_shape_valid(record)) {
    return false;
  }

  if (conservation::enabled()) {
    conservation::lane_event_v0 conservation_record = {};
    conservation_record.owner = record.owner;
    conservation_record.operation_seq = record.operation_seq;
    if (!conservation::ensure_target_ready_before_operation(
            record.owner, record.operation_seq)) {
      return false;
    }
    conservation_record.event = conservation::kEventTypedOperation;
    conservation_record.detail_kind = record.unit;
    if (!conservation::emit_lane_event(conservation_record)) {
      return false;
    }
    if (record.driver == kDriverFunctionalOnly ||
        record.unit == kUnitNode) {
      conservation_record.event = conservation::kEventResultCommitZero;
      if (!conservation::emit_lane_event(conservation_record)) {
        return false;
      }
    }
  }
  if (!enabled() || !diagnostic_filter_matches(record)) return true;

  const uint64_t raw_input_hash =
      hash_bytes(record.typed_input, record.typed_input_bytes);
  const uint64_t raw_result_hash =
      hash_bytes(record.typed_result, record.typed_result_bytes);
  const uint64_t input_hash = normalized_input_hash(record);
  const uint64_t result_hash = normalized_result_hash(record);
  const uint64_t plan_hash = normalized_plan_hash(record);
  std::printf(
      "GPGPU-Sim RTCORE_V04_TYPED_DIAGNOSTIC "
      "schema=2 normalization=identity_and_address_stripped "
      "driver=%s unit=%s owner_hw_sid=%u "
      "resident_warp_slot=%u request_identity=%u "
      "request_generation=%u private_slot_id=%u lane_id=%u "
      "operation_seq=%u operation_kind=%u semantic_plan_kind=%u "
      "route_kind=%u boundary_kind=%u input_bytes=%zu "
      "input_hash=%016llx raw_input_hash=%016llx "
      "result_bytes=%zu result_hash=%016llx raw_result_hash=%016llx "
      "plan_bytes=%zu plan_hash=%016llx\n",
      driver_name(static_cast<driver_kind>(record.driver)),
      unit_name(static_cast<unit_kind>(record.unit)),
      record.owner.owner_hw_sid, record.owner.resident_warp_id,
      record.owner.request_identity, record.owner.generation,
      record.owner.private_slot_id, record.owner.lane_id,
      record.operation_seq, record.operation_kind,
      record.semantic_plan_kind, record.route_kind,
      record.boundary_kind, record.typed_input_bytes,
      static_cast<unsigned long long>(input_hash),
      static_cast<unsigned long long>(raw_input_hash),
      record.typed_result_bytes,
      static_cast<unsigned long long>(result_hash),
      static_cast<unsigned long long>(raw_result_hash),
      record.semantic_plan_bytes,
      static_cast<unsigned long long>(plan_hash));
  std::fflush(stdout);
  return true;
}

const char *driver_name(driver_kind driver) {
  switch (driver) {
    case kDriverFunctionalOnly:
      return "functional_only";
    case kDriverTiming:
      return "timing";
    case kDriverInvalid:
      break;
  }
  return "invalid";
}

const char *unit_name(unit_kind unit) {
  switch (unit) {
    case kUnitNode:
      return "node";
    case kUnitStack:
      return "stack";
    case kUnitInstance:
      return "instance";
    case kUnitPrimitive:
      return "primitive";
    case kUnitInvalid:
      break;
  }
  return "invalid";
}

}  // namespace typed_diagnostic
}  // namespace v04
}  // namespace rtcore
