#include "rtcore_v04_target_private_state_384_bridge.h"

#include <cstddef>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace target_private_state_384 {
namespace {

static const unsigned kResponseTargetRtcore = 1;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool identities_equal(
    const private_state_384::operand_materializer::
        operation_identity_v1 &left,
    const private_state_384::operand_materializer::
        operation_identity_v1 &right) {
  return left.owner_hw_sid == right.owner_hw_sid &&
         left.resident_warp_id == right.resident_warp_id &&
         left.request_identity == right.request_identity &&
         left.request_generation == right.request_generation &&
         left.private_slot_id == right.private_slot_id &&
         left.operation_sequence == right.operation_sequence &&
         left.lane_id == right.lane_id &&
         bytes_are_zero(left.reserved_zero,
                        sizeof(left.reserved_zero)) &&
         bytes_are_zero(right.reserved_zero,
                        sizeof(right.reserved_zero));
}

bool keys_equal(
    const private_state_384::live_bridge::live_operation_key_v1 &left,
    const private_state_384::live_bridge::live_operation_key_v1 &right) {
  return identities_equal(left.identity, right.identity) &&
         left.private_layout_profile_id ==
             right.private_layout_profile_id &&
         left.bvh_format_profile_id ==
             right.bvh_format_profile_id &&
         left.reservation_generation ==
             right.reservation_generation &&
         left.storage_profile == right.storage_profile &&
         left.consumer == right.consumer &&
         left.operation == right.operation &&
         left.completion_reason == right.completion_reason;
}

fetch_target::target_kind target_for_consumer(uint8_t consumer) {
  switch (consumer) {
    case private_state_384::operand_plan::kConsumerNode:
      return fetch_target::kTargetNode;
    case private_state_384::operand_plan::kConsumerPrimitive:
      return fetch_target::kTargetPrimitive;
    case private_state_384::operand_plan::kConsumerInstance:
      return fetch_target::kTargetInstance;
    default:
      return fetch_target::kTargetInvalid;
  }
}

bool plan_is_canonical(
    const private_state_384::live_bridge::read_request_plan_v1 &plan) {
  if (plan.valid != 1 || plan.request_count == 0 ||
      plan.request_count != plan.operand_plan.read_count ||
      !bytes_are_zero(plan.reserved_zero,
                      sizeof(plan.reserved_zero)) ||
      plan.key.storage_profile !=
          private_storage::kProfileCompressedShared384 ||
      plan.key.private_layout_profile_id !=
          private_state_384::kPrivateLayoutProfileId ||
      plan.key.bvh_format_profile_id !=
          private_state_384::kGenRtBvhFormatProfileId ||
      plan.key.reservation_generation == 0 ||
      plan.key.operation !=
          private_state_384::operand_plan::kOperationDefault ||
      plan.key.completion_reason !=
          private_state_384::operand_plan::kCompletionReasonNone ||
      target_for_consumer(plan.key.consumer) ==
          fetch_target::kTargetInvalid ||
      !identities_equal(plan.key.identity, plan.identity)) {
    return false;
  }
  private_state_384::operand_plan::read_request_v1 request = {};
  request.private_layout_profile_id =
      plan.key.private_layout_profile_id;
  request.consumer = plan.key.consumer;
  request.operation = plan.key.operation;
  request.completion_reason = plan.key.completion_reason;
  private_state_384::operand_plan::read_plan_v1 canonical = {};
  if (private_state_384::operand_plan::make_read_plan(
          request, &canonical) !=
          private_state_384::operand_plan::kStatusOk ||
      std::memcmp(&canonical, &plan.operand_plan,
                  sizeof(canonical)) != 0) {
    return false;
  }
  for (uint8_t index = 0; index < plan.request_count; ++index) {
    const rtcore_memory_unit_request_snapshot &memory =
        plan.requests[index];
    const rtcore_v04_private_state_384_read_transport_snapshot
        &transport = memory.v04_private_state_384_read;
    private_frontier::owner_binding_v0 owner = {};
    owner.owner_hw_sid = plan.identity.owner_hw_sid;
    owner.resident_warp_id =
        static_cast<uint8_t>(plan.identity.resident_warp_id);
    owner.request_identity = plan.identity.request_identity;
    owner.generation = plan.identity.request_generation;
    owner.private_slot_id = plan.identity.private_slot_id;
    owner.lane_id = plan.identity.lane_id;
    if (!memory.valid ||
        memory.address_space != RTCORE_MEMORY_ADDRESS_SPACE_SHARED ||
        memory.operation != RTCORE_MEMORY_OPERATION_READ ||
        memory.destination !=
            RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL ||
        memory.response_target != kResponseTargetRtcore ||
        memory.rt_request_id == 0 ||
        memory.resident_warp_id >=
            private_state_384::backing::kResidentWarpCapacity ||
        memory.request_generation == 0 ||
        memory.private_slot_id >=
            private_state_384::backing::kPrivateSlotCapacity ||
        memory.lane_id >=
            private_state_384::backing::kLaneCapacity ||
        memory.access_kind !=
            RTCORE_MEMORY_ACCESS_PRIVATE_STATE_384_READ ||
        memory.is_write ||
        memory.byte_mask !=
            private_state_384::backing::kFullChunkByteMask ||
        memory.owner_hw_sid != owner.owner_hw_sid ||
        memory.resident_warp_id != owner.resident_warp_id ||
        memory.rt_request_id != owner.request_identity ||
        memory.request_generation != owner.generation ||
        memory.private_slot_id != owner.private_slot_id ||
        memory.lane_id != owner.lane_id ||
        memory.chunk_id != index ||
        memory.chunk_count != plan.request_count ||
        memory.memory_op_seq != static_cast<unsigned>(index) + 1u ||
        memory.aligned_32b_addr !=
            private_state_384::live_bridge::private_slot_base(owner) +
                canonical.reads[index].slot_byte_offset ||
        transport.operation_sequence !=
            plan.identity.operation_sequence ||
        transport.private_layout_profile_id !=
            plan.key.private_layout_profile_id ||
        transport.bvh_format_profile_id !=
            plan.key.bvh_format_profile_id ||
        transport.reservation_generation !=
            plan.key.reservation_generation ||
        transport.consumer != plan.key.consumer ||
        transport.operation_kind != plan.key.operation ||
        transport.completion_reason !=
            plan.key.completion_reason ||
        transport.read_index != index ||
        transport.read_count != plan.request_count ||
        transport.storage_profile != plan.key.storage_profile ||
        transport.valid != 1 || transport.reserved_zero != 0) {
      return false;
    }
  }
  return true;
}

template <typename Slot>
fetch_target::slot_metadata_v0 *find_metadata(
    Slot *slots, uint8_t capacity,
    const private_state_384::live_bridge::live_operation_key_v1 &key) {
  fetch_target::slot_metadata_v0 *match = NULL;
  for (uint8_t index = 0; index < capacity; ++index) {
    fetch_target::slot_metadata_v0 &metadata =
        slots[index].metadata;
    if (metadata.state == fetch_target::kSlotFree ||
        metadata.private_state_384_projection_valid != 0 ||
        !keys_equal(metadata.private_state_384_key, key)) {
      continue;
    }
    if (match != NULL) return NULL;
    match = &metadata;
  }
  return match;
}

fetch_target::slot_metadata_v0 *find_metadata(
    fetch_target::engine_state_v0 *state,
    const private_state_384::live_bridge::live_operation_key_v1 &key) {
  switch (target_for_consumer(key.consumer)) {
    case fetch_target::kTargetNode:
      return find_metadata(
          state->node_slots, state->config.node_capacity, key);
    case fetch_target::kTargetPrimitive:
      return find_metadata(
          state->primitive_slots, state->config.primitive_capacity,
          key);
    case fetch_target::kTargetInstance:
      return find_metadata(
          state->instance_slots, state->config.instance_capacity,
          key);
    case fetch_target::kTargetInvalid:
      return NULL;
  }
  return NULL;
}

private_state_384::live_bridge::live_operation_key_v1
key_from_request(const rtcore_memory_unit_request_snapshot &request) {
  private_state_384::live_bridge::live_operation_key_v1 key = {};
  key.identity.owner_hw_sid = request.owner_hw_sid;
  key.identity.resident_warp_id = request.resident_warp_id;
  key.identity.request_identity = request.rt_request_id;
  key.identity.request_generation = request.request_generation;
  key.identity.private_slot_id = request.private_slot_id;
  key.identity.operation_sequence =
      request.v04_private_state_384_read.operation_sequence;
  key.identity.lane_id = static_cast<uint8_t>(request.lane_id);
  key.private_layout_profile_id =
      request.v04_private_state_384_read.private_layout_profile_id;
  key.bvh_format_profile_id =
      request.v04_private_state_384_read.bvh_format_profile_id;
  key.reservation_generation =
      request.v04_private_state_384_read.reservation_generation;
  key.storage_profile =
      request.v04_private_state_384_read.storage_profile;
  key.consumer = request.v04_private_state_384_read.consumer;
  key.operation = request.v04_private_state_384_read.operation_kind;
  key.completion_reason =
      request.v04_private_state_384_read.completion_reason;
  return key;
}

template <typename DestinationRay, typename SourceRay>
void copy_ray(const SourceRay &source, DestinationRay *destination) {
  for (uint8_t component = 0; component < 3; ++component) {
    destination->origin[component] = source.origin[component];
    destination->direction[component] =
        source.direction[component];
    destination->inverse_direction[component] =
        source.inverse_direction[component];
  }
  destination->t_min = source.t_min;
  destination->t_max = source.t_max;
}

status_kind materialize_projection(
    const private_state_384::live_bridge::live_operation_key_v1 &key,
    const private_state_384::operand_materializer::
        response_collector_v1 &collector,
    typed_node::ray_policy_v0 *ray_policy,
    private_frontier::root_private_operands_v0 *root,
    private_frontier::instance_shader_projection_v0
        *current_instance) {
  if (ray_policy == NULL || root == NULL ||
      current_instance == NULL) {
    return kStatusInvalidArgument;
  }
  *ray_policy = typed_node::ray_policy_v0();
  *root = private_frontier::root_private_operands_v0();
  *current_instance =
      private_frontier::instance_shader_projection_v0();
  private_state_384::operand_materializer::materialize_context_v1
      context = {};
  context.bvh_format_profile_id = key.bvh_format_profile_id;
  switch (target_for_consumer(key.consumer)) {
    case fetch_target::kTargetNode: {
      private_state_384::operand_materializer::node_operands_v1
          operands = {};
      if (private_state_384::operand_materializer::materialize_node(
              collector, context, &operands) !=
          private_state_384::operand_materializer::kStatusOk) {
        return kStatusMaterializeRejected;
      }
      copy_ray(operands.ray, &root->mutable_ray);
      root->decode_context = operands.decode_context;
      root->committed_hit.valid = operands.committed_valid;
      root->committed_hit.hit_t =
          operands.committed_valid != 0
              ? operands.effective_traversal_bound
              : 0.0f;
      *ray_policy = operands.ray_policy;
      return kStatusOk;
    }
    case fetch_target::kTargetPrimitive: {
      private_state_384::operand_materializer::primitive_operands_v1
          operands = {};
      if (private_state_384::operand_materializer::
              materialize_primitive(collector, context, &operands) !=
          private_state_384::operand_materializer::kStatusOk) {
        return kStatusMaterializeRejected;
      }
      root->mutable_ray = operands.ray;
      root->decode_context = operands.decode_context;
      root->committed_hit = operands.committed_hit;
      *current_instance = operands.current_instance;
      *ray_policy = operands.ray_policy;
      return kStatusOk;
    }
    case fetch_target::kTargetInstance: {
      private_state_384::operand_materializer::instance_operands_v1
          operands = {};
      if (private_state_384::operand_materializer::
              materialize_instance(collector, context, &operands) !=
          private_state_384::operand_materializer::kStatusOk) {
        return kStatusMaterializeRejected;
      }
      copy_ray(operands.world_ray, &root->mutable_ray);
      root->decode_context = operands.tlas_decode_context;
      ray_policy->ray_flags = operands.ray_policy.ray_flags;
      ray_policy->cull_mask = operands.ray_policy.cull_mask;
      return kStatusOk;
    }
    case fetch_target::kTargetInvalid:
      return kStatusMaterializeRejected;
  }
  return kStatusMaterializeRejected;
}

}  // namespace

status_kind configure_read(
    fetch_target::engine_state_v0 *state,
    const fetch_target::reservation_receipt_v0 &reservation,
    const private_state_384::live_bridge::read_request_plan_v1 &plan,
    fetch_target::reservation_receipt_v0 *updated_reservation) {
  if (state == NULL || updated_reservation == NULL ||
      state->initialized != 1 || !plan_is_canonical(plan) ||
      reservation.slot_generation !=
          plan.key.reservation_generation ||
      reservation.target_operation_seq !=
          plan.identity.operation_sequence) {
    return kStatusPlanMismatch;
  }
  private_state_384::operand_materializer::response_collector_v1
      collector = {};
  if (private_state_384::live_bridge::initialize_collector(
          plan, &collector) !=
      private_state_384::live_bridge::kStatusOk) {
    return kStatusPlanMismatch;
  }
  return fetch_target::configure_private_state_384_slot(
             state, reservation, plan.key, collector,
             updated_reservation) == fetch_target::kStatusOk
             ? kStatusOk
             : kStatusTargetRejected;
}

status_kind accept_response(
    fetch_target::engine_state_v0 *state,
    const private_state_384::backing::state_v1 &backing,
    const rtcore_memory_unit_request_snapshot &request) {
  if (state == NULL || state->initialized != 1 || !request.valid ||
      request.destination !=
          RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL ||
      request.v04_private_state_384_read.valid != 1) {
    return kStatusInvalidArgument;
  }
  fetch_target::engine_state_v0 staged = *state;
  const private_state_384::live_bridge::live_operation_key_v1 key =
      key_from_request(request);
  fetch_target::slot_metadata_v0 *metadata =
      find_metadata(&staged, key);
  if (metadata == NULL ||
      metadata->pending_private_response_count == 0 ||
      metadata->private_state_384_projection_valid != 0) {
    return kStatusResponseRejected;
  }
  private_state_384::operand_materializer::response_collector_v1
      collector = metadata->private_state_384_collector;
  if (private_state_384::live_bridge::accept_read_response(
          backing, request, &collector) !=
      private_state_384::live_bridge::kStatusOk) {
    return kStatusResponseRejected;
  }
  metadata->private_state_384_collector = collector;
  metadata->received_private_chunk_mask =
      collector.received_chunk_mask;
  metadata->pending_private_response_count = static_cast<uint8_t>(
      collector.required_count - collector.received_count);
  if (!private_state_384::operand_materializer::responses_complete(
          collector)) {
    *state = staged;
    return kStatusOk;
  }
  typed_node::ray_policy_v0 ray_policy = {};
  private_frontier::root_private_operands_v0 root = {};
  private_frontier::instance_shader_projection_v0 current_instance = {};
  if (materialize_projection(
          key, collector, &ray_policy, &root, &current_instance) !=
      kStatusOk) {
    return kStatusMaterializeRejected;
  }
  if (fetch_target::publish_private_state_384_projection(
          &staged, key, collector, ray_policy, root,
          current_instance) != fetch_target::kStatusOk) {
    return kStatusTargetRejected;
  }
  *state = staged;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusPlanMismatch:
      return "plan_mismatch";
    case kStatusTargetRejected:
      return "target_rejected";
    case kStatusResponseRejected:
      return "response_rejected";
    case kStatusMaterializeRejected:
      return "materialize_rejected";
  }
  return "unknown";
}

}  // namespace target_private_state_384
}  // namespace v04
}  // namespace rtcore
