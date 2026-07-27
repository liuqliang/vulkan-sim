#ifndef RTCORE_V04_FUNCTIONAL_ENGINE_H
#define RTCORE_V04_FUNCTIONAL_ENGINE_H

#include <array>
#include <cstdint>

#include "rtcore_v04_continuation_lifecycle.h"
#include "rtcore_v04_functional_driver.h"

namespace rtcore {
namespace v04 {
namespace functional_engine {

static const uint32_t kMaxOperationsPerRun = 1u << 20;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidState,
  kStatusOperationSequenceExhausted,
  kStatusOperationWatchdog,
  kStatusPrivateStateRejected,
  kStatusTargetRejected,
  kStatusPayloadReadRejected,
  kStatusInstanceProducerRejected,
  kStatusFunctionalDriverRejected,
  kStatusSemanticApplyRejected,
  kStatusShaderReturnRejected,
};

enum boundary_kind : uint8_t {
  kBoundaryInvalid = 0,
  kBoundaryFinalMiss,
  kBoundaryFinalHit,
  kBoundaryAnyHit,
  kBoundaryIntersection,
};

enum driver_unit_kind : uint8_t {
  kDriverUnitInvalid = 0,
  kDriverUnitNode,
  kDriverUnitInstance,
  kDriverUnitPrimitive,
  kDriverUnitStack,
};

struct provider_v0 {
  void *context;
  bool (*read_raw_payload)(
      void *context,
      const typed_blas::as_decode_context_v0 &decode_context,
      uint64_t address, uint16_t byte_count, uint8_t *bytes);
  bool (*prepare_instance_enter)(
      void *context, const fetch_target::operation_packet_v0 &packet,
      typed_instance::enter_input_v0 *input);
};

struct root_input_v0 {
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 private_region;
  fetch_target::target_reference_v0 root_reference;
  typed_node::ray_policy_v0 ray_policy;
  private_frontier::root_private_operands_v0 private_operands;
  uint64_t raw_payload_base_address;
};

struct state_v0 {
  uint8_t valid;
  uint8_t waiting_shader;
  uint8_t terminal;
  uint8_t boundary_kind;
  private_frontier::owner_binding_v0 owner;
  private_frontier::region_binding_v0 private_region;
  private_frontier::shadow_slot_v0 canonical_slot;
  typed_node::ray_policy_v0 ray_policy;
  uint32_t next_operation_seq;
  uint32_t boundary_operation_seq;
  uint32_t boundary_reason;
  uint32_t operation_count;
  uint32_t node_visits;
  uint32_t primitive_tests;
  uint8_t last_driver_unit;
  uint8_t last_driver_status;
  uint8_t reserved_zero[2];
};

struct output_v0 {
  uint8_t valid;
  uint8_t boundary_kind;
  uint8_t waiting_shader;
  uint8_t terminal;
  private_frontier::owner_binding_v0 owner;
  private_frontier::mutable_ray_state_v0 ray;
  private_frontier::committed_hit_projection_v0 committed_hit;
  private_frontier::retained_candidate_projection_v0 retained_candidate;
  typed_primitive::intersection_boundary_facts_v0 intersection_boundary;
  uint32_t boundary_operation_seq;
  uint32_t boundary_reason;
  uint32_t operation_count;
  uint32_t node_visits;
  uint32_t primitive_tests;
};

status_kind run_new(const root_input_v0 &input,
                    const provider_v0 &provider, state_v0 *state,
                    output_v0 *output);

status_kind resume(
    state_v0 *state, uint32_t boundary_reason,
    const std::array<uint32_t, abi_v04::kWordCount> &shader_return_words,
    const provider_v0 &provider, output_v0 *output);

const char *status_name(status_kind status);
const char *boundary_name(boundary_kind boundary);
const char *driver_unit_name(driver_unit_kind unit);

}  // namespace functional_engine
}  // namespace v04
}  // namespace rtcore

#endif
