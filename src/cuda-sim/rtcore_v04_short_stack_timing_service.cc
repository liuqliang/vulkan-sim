#include "rtcore_v04_short_stack_timing_service.h"

#include <cstring>
#include <limits>

#include "rtcore_v04_private_shared_backing_internal.h"
#include "rtcore_v04_stall_attribution.h"
#include "rtcore_v04_typed_instance_kernel.h"

namespace rtcore {
namespace v04 {
namespace short_stack_timing {
namespace {

static const uint8_t kAllReadChunks =
    static_cast<uint8_t>((1u << kReadChunkCount) - 1u);
static const uint8_t kReadPhaseShortStackState = 4;
static const uint8_t kReadPhaseReturnInstance = 5;
static const uint8_t kAllReturnInstanceReadChunks =
    static_cast<uint8_t>(
        (1u << kReturnInstanceReadChunkCount) - 1u);
static const unsigned kResponseTargetRtcore = 1;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

uint16_t all_write_chunks(uint8_t write_count) {
  return write_count == 0 || write_count > kMaxWriteChunkCount
             ? 0
             : static_cast<uint16_t>((uint16_t{1} << write_count) - 1u);
}

bool make_stack_sparse_deltas(
    const short_stack_shared::persistent_state_v0 &persistent,
    private_state_384::live_bridge::sparse_chunk_delta_v1
        deltas[private_state_384::kChunkCount],
    uint8_t *delta_count) {
  if (deltas == NULL || delta_count == NULL ||
      !short_stack_shared::validate_persistent_state(persistent)) {
    return false;
  }
  private_state_384::stack_sparse_projection_v1 projection = {};
  if (private_state_384::encode_stack_sparse_projection(
          persistent.stack, &projection,
          persistent.recovery_target_inflight) !=
      private_state_384::kStatusOk) {
    return false;
  }
  std::memset(
      deltas, 0,
      sizeof(*deltas) * private_state_384::kChunkCount);
  deltas[0].chunk_index = 4;
  deltas[0].byte_mask = 0x0f000000u;
  std::memcpy(
      deltas[0].payload + 24, projection.metadata,
      sizeof(projection.metadata));
  for (uint8_t index = 0; index < 3; ++index) {
    deltas[index + 1].chunk_index =
        static_cast<uint8_t>(5 + index);
    deltas[index + 1].byte_mask =
        private_state_384::backing::kFullChunkByteMask;
    std::memcpy(
        deltas[index + 1].payload,
        projection.entries +
            index * private_state_384::kChunkBytes,
        private_state_384::kChunkBytes);
  }
  *delta_count = 4;
  return true;
}

template <typename SourceRay>
void copy_codec_ray(const SourceRay &source,
                    private_state_384::ray_v1 *destination) {
  for (uint8_t component = 0; component < 3; ++component) {
    destination->origin[component] = source.origin[component];
    destination->direction[component] = source.direction[component];
  }
  destination->t_min = source.t_min;
  destination->t_max = source.t_max;
}

bool copy_codec_as(
    const typed_blas::as_decode_context_v0 &source,
    uint8_t cull_mask,
    private_state_384::as_context_v1 *destination) {
  if (destination == NULL ||
      source.bvh_format_profile_id !=
          private_state_384::kGenRtBvhFormatProfileId ||
      source.as_object.object_id == 0 ||
      source.as_object.generation == 0 ||
      source.device_range_bytes == 0) {
    return false;
  }
  *destination = private_state_384::as_context_v1();
  destination->as_object_id = source.as_object.object_id;
  destination->device_base = source.device_base;
  destination->device_range_bytes = source.device_range_bytes;
  destination->as_object_generation =
      source.as_object.generation;
  destination->as_type = source.as_object.as_type;
  destination->cull_mask = cull_mask;
  return true;
}

void copy_sparse_chunk(
    uint8_t chunk_index, uint32_t byte_mask,
    const private_state_384::image_v1 &image,
    private_state_384::live_bridge::sparse_chunk_delta_v1 *delta) {
  *delta =
      private_state_384::live_bridge::sparse_chunk_delta_v1();
  delta->chunk_index = chunk_index;
  delta->byte_mask = byte_mask;
  const uint16_t byte_offset =
      static_cast<uint16_t>(
          chunk_index * private_state_384::kChunkBytes);
  for (uint8_t byte = 0; byte < private_state_384::kChunkBytes;
       ++byte) {
    if ((byte_mask & (uint32_t{1} << byte)) != 0) {
      delta->payload[byte] =
          image.bytes[byte_offset + byte];
    }
  }
}

bool make_instance_enter_sparse_deltas(
    const operation_entry_v0 &entry,
    private_state_384::live_bridge::sparse_chunk_delta_v1
        deltas[private_state_384::kChunkCount],
    uint8_t *delta_count) {
  if (deltas == NULL || delta_count == NULL ||
      entry.input.deferred_instance_valid != 1 ||
      entry.input.operation_kind != kOperationEnterBlasTransition ||
      !short_stack_shared::validate_persistent_state(
          entry.transition.persistent_state)) {
    return false;
  }
  private_state_384::state_v1 state = {};
  copy_codec_ray(entry.input.instance_object_ray, &state.ray);
  const uint8_t cull_mask =
      entry.private_state_384_stack_operands.ray_policy.cull_mask;
  if (!copy_codec_as(
          entry.input.blas_root.decode_context, cull_mask,
          &state.active_as) ||
      !copy_codec_as(
          entry.private_state_384_stack_operands
              .active_decode_context,
          cull_mask, &state.parent_restore.tlas_context)) {
    return false;
  }
  state.current_instance.instance_metadata_ref =
      entry.input.instance_projection
          .instance_metadata_reference;
  state.current_instance.instance_index =
      entry.input.instance_projection.instance_index;
  state.current_instance.instance_custom_index =
      entry.input.instance_projection.instance_custom_index;
  state.current_instance.instance_sbt_contribution =
      entry.input.instance_projection
          .instance_sbt_contribution;
  state.current_instance.instance_policy_flags =
      entry.input.instance_projection.instance_flags;
  state.ray_flags =
      entry.private_state_384_stack_operands.ray_policy.ray_flags;
  state.tlas_build_generation =
      entry.private_state_384_stack_operands
          .tlas_build_generation;
  state.blas_build_generation =
      entry.input.blas_build_generation;
  state.stack = entry.transition.persistent_state.stack;
  copy_codec_ray(
      entry.private_state_384_stack_operands.ray,
      &state.parent_restore.ray);
  private_state_384::control_tags_v1 control = {};
  control.parent_restore_valid = 1;
  private_state_384::image_v1 image = {};
  if (private_state_384::encode_image(
          private_state_384::kPrivateLayoutProfileId,
          private_state_384::kGenRtBvhFormatProfileId,
          state, control, &image) !=
      private_state_384::kStatusOk) {
    return false;
  }
  std::memset(
      deltas, 0,
      sizeof(*deltas) * private_state_384::kChunkCount);
  static const uint8_t chunks[] = {0, 1, 3, 4, 5, 6, 7, 10, 11};
  static const uint32_t masks[] = {
      0xffffffffu, 0xffffffffu, 0xff000000u,
      0x0ff00fffu, 0xffffffffu, 0xffffffffu,
      0xffffffffu, 0xffffffffu, 0xffffffffu};
  static const uint8_t kInstanceEnterDeltaCount = 9;
  for (uint8_t index = 0; index < kInstanceEnterDeltaCount;
       ++index) {
    copy_sparse_chunk(
        chunks[index], masks[index], image, &deltas[index]);
  }
  *delta_count = kInstanceEnterDeltaCount;
  return true;
}

bool make_cross_as_return_sparse_deltas(
    const operation_entry_v0 &entry,
    private_state_384::live_bridge::sparse_chunk_delta_v1
        deltas[private_state_384::kChunkCount],
    uint8_t *delta_count) {
  if (deltas == NULL || delta_count == NULL ||
      entry.private_state_384_cross_as_operands_valid != 1 ||
      entry.return_instance_projection_valid != 1 ||
      entry.input.operation_kind != kOperationResumeTransition ||
      !short_stack_shared::validate_persistent_state(
          entry.transition.persistent_state) ||
      entry.transition.persistent_state.stack.active_domain !=
          short_stack::kDomainTlas ||
      entry.transition.persistent_state.blas_build_generation != 0) {
    return false;
  }
  private_state_384::state_v1 state = {};
  copy_codec_ray(
      entry.private_state_384_cross_as_operands.parent.ray,
      &state.ray);
  const uint8_t cull_mask =
      entry.private_state_384_cross_as_operands
          .parent.ray_policy.cull_mask;
  if (!copy_codec_as(
          entry.private_state_384_cross_as_operands
              .parent.tlas_decode_context,
          cull_mask, &state.active_as)) {
    return false;
  }
  state.current_instance.instance_metadata_ref =
      entry.return_instance_projection.instance_metadata_ref;
  state.current_instance.instance_index =
      entry.return_instance_projection.instance_index;
  state.current_instance.instance_custom_index =
      entry.return_instance_projection.instance_custom_index;
  state.current_instance.instance_sbt_contribution =
      entry.return_instance_projection
          .instance_sbt_contribution;
  state.current_instance.instance_policy_flags =
      entry.return_instance_projection.instance_policy_flags;
  state.ray_flags =
      entry.private_state_384_cross_as_operands
          .base.ray_policy.ray_flags;
  state.tlas_build_generation =
      entry.transition.persistent_state.tlas_build_generation;
  state.blas_build_generation = 0;
  state.stack = entry.transition.persistent_state.stack;
  private_state_384::control_tags_v1 control = {};
  control.recovery_target_inflight =
      entry.transition.persistent_state.recovery_target_inflight;
  private_state_384::image_v1 image = {};
  if (private_state_384::encode_image(
          private_state_384::kPrivateLayoutProfileId,
          private_state_384::kGenRtBvhFormatProfileId,
          state, control, &image) !=
      private_state_384::kStatusOk) {
    return false;
  }
  std::memset(
      deltas, 0,
      sizeof(*deltas) * private_state_384::kChunkCount);
  static const uint8_t chunks[] = {0, 1, 3, 4, 5, 6, 7};
  static const uint32_t masks[] = {
      0xffffffffu, 0xffffffffu, 0xff000000u,
      0x0ff00fffu, 0xffffffffu, 0xffffffffu,
      0xffffffffu};
  static const uint8_t kCrossAsReturnDeltaCount = 7;
  for (uint8_t index = 0; index < kCrossAsReturnDeltaCount;
       ++index) {
    copy_sparse_chunk(
        chunks[index], masks[index], image, &deltas[index]);
  }
  *delta_count = kCrossAsReturnDeltaCount;
  return true;
}

bool make_return_instance_projection(
    const typed_instance::boundary_result_v0 &decoded,
    const typed_blas::as_decode_context_v0 &tlas,
    const short_stack::entry_v0 &return_entry,
    private_frontier::instance_shader_projection_v0 *projection) {
  if (projection == NULL ||
      decoded.status != typed_instance::kStatusOk ||
      tlas.as_object.as_type != typed_instance::kAsTypeTlas ||
      return_entry.payload_offset >
          tlas.device_range_bytes ||
      return_entry.payload_byte_count !=
          fetch_target::kInstanceRawPayloadBytes ||
      return_entry.payload_offset >
          tlas.device_range_bytes -
              fetch_target::kInstanceRawPayloadBytes) {
    return false;
  }
  *projection =
      private_frontier::instance_shader_projection_v0();
  projection->instance_metadata_ref =
      tlas.device_base + return_entry.payload_offset;
  projection->instance_index = decoded.instance_index;
  projection->instance_custom_index =
      decoded.instance_custom_index;
  projection->instance_sbt_contribution =
      decoded.instance_sbt_contribution;
  projection->instance_policy_flags =
      decoded.instance_flags;
  return true;
}

uint64_t private_slot_base(
    const private_frontier::owner_binding_v0 &owner) {
  return UINT64_C(0xff00000000000000) +
         static_cast<uint64_t>(owner.owner_hw_sid) * UINT64_C(0x1000000) +
         static_cast<uint64_t>(owner.private_slot_id) *
             private_frontier::kPrivateDataSlotBytes;
}

bool make_request_owner(
    const private_frontier::owner_binding_v0 &private_owner,
    request_owner::lane_binding_v0 *request_binding) {
  if (request_binding == NULL || private_owner.request_identity == 0 ||
      private_owner.generation == 0 ||
      private_owner.generation > request_owner::kRequestGenerationMax ||
      private_owner.resident_warp_id >=
          request_owner::kResidentWarpCapacity ||
      private_owner.private_slot_id >
          std::numeric_limits<uint16_t>::max() ||
      private_owner.lane_id >= request_owner::kLaneCapacity ||
      !bytes_are_zero(private_owner.reserved_zero,
                      sizeof(private_owner.reserved_zero))) {
    return false;
  }
  request_owner::internal_request_key_fields_v0 fields = {};
  if (request_owner::unpack_internal_request_key(
          private_owner.request_identity, &fields) !=
          request_owner::kStatusOk ||
      fields.resident_warp_slot != private_owner.resident_warp_id ||
      fields.lane_id != private_owner.lane_id ||
      fields.request_generation != private_owner.generation) {
    return false;
  }
  *request_binding = request_owner::lane_binding_v0();
  request_binding->packed_request_key =
      private_owner.request_identity;
  request_binding->owner_hw_sid = private_owner.owner_hw_sid;
  request_binding->request_control_slot =
      fields.request_control_slot;
  request_binding->request_generation =
      fields.request_generation;
  request_binding->private_slot_id =
      static_cast<uint16_t>(private_owner.private_slot_id);
  request_binding->resident_warp_slot =
      fields.resident_warp_slot;
  request_binding->lane_id = fields.lane_id;
  return private_frontier::owners_equal(
      request_owner::make_private_frontier_owner(*request_binding),
      private_owner);
}

void emit_attempt(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, uint64_t service_cycle,
    uint16_t arbitration_slot,
    stall_attribution::stage_kind stage,
    stall_attribution::outcome_kind outcome,
    stall_attribution::action_kind action,
    stall_attribution::reason_kind reason) {
  if (!stall_attribution::enabled()) return;
  stall_attribution::attempt_record_v0 record = {};
  record.service_cycle = service_cycle;
  record.owner_hw_sid = owner.owner_hw_sid;
  record.request_identity = owner.request_identity;
  record.request_generation = owner.generation;
  record.operation_seq = operation_seq;
  record.chunk_id = 0;
  record.chunk_count = 1;
  record.arbitration_slot = arbitration_slot;
  record.lane_id = owner.lane_id;
  record.unit = stall_attribution::kUnitShortStack;
  record.stage = stage;
  record.outcome = outcome;
  record.action = action;
  record.reason = reason;
  stall_attribution::emit_attempt(record);
}

bool valid_config(const config_v0 &config) {
  return config.capacity != 0 && config.capacity <= kMaxSlots &&
         config.reservation_width != 0 &&
         config.reservation_width <= kMaxSlots &&
         config.unit_count != 0 && config.unit_count <= kMaxUnits &&
         config.stack_latency != 0 &&
         config.initiation_interval != 0 &&
         config.issue_width != 0 &&
         config.issue_width <= config.unit_count &&
         config.parent_lookup_latency != 0 &&
         config.reserved_zero == 0;
}

bool request_matches_entry(
    const rtcore_memory_unit_request_snapshot &request,
    const operation_entry_v0 &entry) {
  const rtcore_v04_stack_private_read_transport_snapshot &transport =
      request.v04_stack_private_read;
  const bool state_read =
      transport.read_phase == kReadPhaseShortStackState &&
      entry.phase == kPhaseReading &&
      request.access_kind ==
          RTCORE_MEMORY_ACCESS_SHORT_STACK_STATE_READ &&
      request.chunk_count == kReadChunkCount &&
      request.memory_op_seq == request.chunk_id + 1;
  const bool return_instance_read =
      transport.read_phase == kReadPhaseReturnInstance &&
      entry.phase == kPhaseReturnInstanceReading &&
      request.access_kind ==
          RTCORE_MEMORY_ACCESS_SHORT_STACK_RETURN_INSTANCE_READ &&
      request.chunk_count == kReturnInstanceReadChunkCount &&
      request.memory_op_seq ==
          (entry.input.private_storage_profile ==
                   private_storage::kProfileCompressedShared384 ||
               entry.input.private_storage_profile ==
                   private_storage::kProfileGlobal384
               ? entry.reservation.read_chunk_count
               : kReadChunkCount) +
              request.chunk_id + 1;
  return entry.valid != 0 && (state_read || return_instance_read) &&
         request.valid &&
         request.address_space ==
             (return_instance_read
                  ? RTCORE_MEMORY_ADDRESS_SPACE_GLOBAL
                  : RTCORE_MEMORY_ADDRESS_SPACE_SHARED) &&
         request.operation == RTCORE_MEMORY_OPERATION_READ &&
         request.destination ==
             RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL &&
         request.response_target == kResponseTargetRtcore &&
         !request.is_write &&
         request.owner_hw_sid == entry.input.owner.owner_hw_sid &&
         request.rt_request_id == entry.input.owner.request_identity &&
         request.resident_warp_id ==
             entry.input.owner.resident_warp_id &&
         request.request_generation == entry.input.owner.generation &&
         request.private_slot_id == entry.input.owner.private_slot_id &&
         request.lane_id == entry.input.owner.lane_id &&
         request.chunk_id < request.chunk_count &&
         request.byte_mask != 0 &&
         transport.valid == 1 &&
         transport.reservation_id ==
             entry.reservation.reservation_id &&
         transport.reservation_age ==
             entry.reservation.reservation_age &&
         transport.target_operation_seq ==
             entry.reservation.operation_seq &&
         transport.producer_operation_seq ==
             entry.reservation.producer_operation_seq &&
         transport.target_slot_generation ==
             entry.reservation.slot_generation &&
         transport.target_slot_index ==
             entry.reservation.slot_index &&
         transport.operation_kind == entry.input.operation_kind &&
         bytes_are_zero(transport.reserved_zero,
                        sizeof(transport.reserved_zero));
}

int find_free_slot(const engine_state_v0 &state) {
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    if (state.slots[index].valid == 0) return static_cast<int>(index);
  }
  return -1;
}

int find_oldest_phase(const engine_state_v0 &state, phase_kind phase,
                      uint64_t service_cycle, bool require_mature) {
  int selected = -1;
  uint64_t oldest = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    const operation_entry_v0 &entry = state.slots[index];
    if (entry.valid == 0 || entry.phase != phase) continue;
    const uint64_t ready_cycle =
        phase == kPhaseParentLookup ? entry.parent_lookup_ready_cycle
                                    : entry.result_ready_cycle;
    if (require_mature && ready_cycle > service_cycle) continue;
    if (entry.issue_age < oldest) {
      selected = static_cast<int>(index);
      oldest = entry.issue_age;
    }
  }
  return selected;
}

status_kind begin_transition_commit(
    operation_entry_v0 *entry,
    timing_driver::state_v0 *timing_state) {
  if (entry == NULL || timing_state == NULL ||
      entry->transition.write_plan.access_count == 0 ||
      entry->transition.write_plan.access_count >
          kMaxWriteChunkCount) {
    return kStatusInvalidArgument;
  }

  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry->input.owner, &request_binding)) {
    return kStatusOwnerMismatch;
  }
  timing_driver::state_v0 staged_timing = *timing_state;
  uint32_t commit_epoch = 0;
  if (timing_driver::begin_result_commit(
          &staged_timing, request_binding,
          entry->reservation.operation_seq, &commit_epoch) !=
      timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  const bool private_state_384 =
      entry->input.private_storage_profile ==
          private_storage::kProfileCompressedShared384 ||
      entry->input.private_storage_profile ==
          private_storage::kProfileGlobal384;
  if (private_state_384) {
    private_state_384::live_bridge::sparse_chunk_delta_v1
        deltas[private_state_384::kChunkCount] = {};
    uint8_t delta_count = 0;
    private_state_384::live_bridge::write_commit_input_v1 input = {};
    input.owner = entry->input.owner;
    input.private_slot_base_address =
        entry->input.private_slot_base_address;
    input.operation_sequence =
        entry->reservation.operation_seq;
    input.commit_epoch = commit_epoch;
    input.bvh_format_profile_id =
        private_state_384::kGenRtBvhFormatProfileId;
    input.expected_write_ack_count =
        entry->input.private_storage_profile ==
                private_storage::kProfileGlobal384
            ? 0
            : entry->transition.write_plan.access_count;
    input.storage_profile = entry->input.private_storage_profile;
    const bool instance_enter =
        entry->input.operation_kind ==
        kOperationEnterBlasTransition;
    const bool cross_as_return =
        entry->input.operation_kind ==
            kOperationResumeTransition &&
        entry->private_state_384_cross_as_operands_valid == 1;
    input.producer =
        instance_enter
            ? private_state_384::operand_plan::
                  kProducerInstanceEnter
            : cross_as_return
                  ? private_state_384::operand_plan::
                        kProducerStackCrossAsReturn
                  : private_state_384::operand_plan::
                        kProducerStack;
    const bool deltas_valid =
        instance_enter
            ? make_instance_enter_sparse_deltas(
                  *entry, deltas, &delta_count)
            : cross_as_return
                  ? make_cross_as_return_sparse_deltas(
                        *entry, deltas, &delta_count)
                  : make_stack_sparse_deltas(
                        entry->transition.persistent_state, deltas,
                        &delta_count);
    if (!deltas_valid ||
        private_state_384::live_bridge::stage_sparse_commit(
            input, deltas, delta_count,
            &entry->private_state_384_commit) !=
            private_state_384::live_bridge::kStatusOk) {
      return kStatusSharedPlanRejected;
    }
  }
  entry->commit_epoch = commit_epoch;
  entry->phase = kPhaseWriting;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind enqueue_transition_write(
    operation_entry_v0 *entry, uint8_t chunk_index,
    timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle) {
  if (entry == NULL || timing_state == NULL || private_backing == NULL ||
      entry->phase != kPhaseWriting ||
      chunk_index >= kMaxWriteChunkCount ||
      chunk_index >= entry->transition.write_plan.access_count ||
      (entry->enqueued_write_mask & (1u << chunk_index)) != 0) {
    return kStatusInvalidArgument;
  }
  if (private_backing->shared_queue.size() >=
      private_shared::kSharedQueueCapacity) {
    return kStatusSharedQueueBackpressure;
  }

  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry->input.owner, &request_binding)) {
    return kStatusOwnerMismatch;
  }
  timing_driver::state_v0 staged_timing = *timing_state;
  private_shared::backing_state_v0 staged_backing = *private_backing;
  const uint64_t slot_base = private_slot_base(entry->input.owner);
  const private_frontier::shared_chunk_access_v0 &access =
      entry->transition.write_plan.accesses[chunk_index];
  if (access.access_kind != private_frontier::kAccessWrite ||
      access.aligned_32b_address < slot_base ||
      access.aligned_32b_address - slot_base >
          private_frontier::kPrivateDataSlotBytes -
              private_frontier::kSharedAccessChunkBytes ||
      access.byte_mask == 0) {
    return kStatusSharedPlanRejected;
  }
  const uint32_t chunk_offset = static_cast<uint32_t>(
      access.aligned_32b_address - slot_base);
  private_shared::shared_write_v0 write = {};
  write.valid = true;
  write.address_space = private_shared::kAddressSpaceShared;
  write.address_mode = private_shared::kAddressModePrivateField;
  write.access_operation = private_shared::kAccessOperationWrite;
  write.destination = private_shared::kDestinationPrivateCommitAck;
  write.owner = entry->input.owner;
  write.operation_seq = entry->reservation.operation_seq;
  write.commit_epoch = entry->commit_epoch;
  write.memory_op_seq = chunk_index + 1;
  write.chunk_id = chunk_index;
  write.chunk_count =
      entry->transition.write_plan.access_count;
  write.field_kind = access.field_kind;
  write.aligned_32b_address = access.aligned_32b_address;
  write.byte_mask = access.byte_mask;
  for (unsigned byte = 0; byte < sizeof(write.payload); ++byte) {
    if ((write.byte_mask & (uint32_t{1} << byte)) != 0) {
      write.payload[byte] =
          entry->transition.updated_slot.bytes[
              chunk_offset + byte];
    }
  }
  write.enqueue_cycle = service_cycle;
  if (timing_driver::begin_memory_transaction(
          &staged_timing, request_binding,
          entry->reservation.operation_seq) !=
          timing_driver::kStatusOk ||
      private_shared::enqueue_runtime_write(
          &staged_backing, write) != private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }
  private_state_384::live_bridge::pending_sparse_commit_v1
      staged_private_commit = entry->private_state_384_commit;
  if (entry->input.private_storage_profile ==
          private_storage::kProfileCompressedShared384 &&
      private_state_384::live_bridge::register_modeled_write(
          write, &staged_private_commit) !=
          private_state_384::live_bridge::kStatusOk) {
    return kStatusSharedWriteRejected;
  }
  entry->enqueued_write_mask = static_cast<uint16_t>(
      entry->enqueued_write_mask | (1u << chunk_index));
  entry->private_state_384_commit = staged_private_commit;
  *timing_state = staged_timing;
  *private_backing = staged_backing;
  return kStatusOk;
}

uint8_t modeled_write_count(const operation_entry_v0 &entry) {
  return entry.input.private_storage_profile ==
                 private_storage::kProfileGlobal384
             ? entry.private_state_384_commit.expected_write_ack_count
             : entry.transition.write_plan.access_count;
}

int find_oldest_pending_write(const engine_state_v0 &state,
                              bool global) {
  int selected = -1;
  uint64_t oldest = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    const operation_entry_v0 &entry = state.slots[index];
    if (entry.valid == 0 || entry.phase != kPhaseWriting ||
        (entry.input.private_storage_profile ==
             private_storage::kProfileGlobal384) != global ||
        entry.enqueued_write_mask ==
            all_write_chunks(modeled_write_count(entry))) {
      continue;
    }
    if (entry.issue_age < oldest) {
      selected = static_cast<int>(index);
      oldest = entry.issue_age;
    }
  }
  return selected;
}

uint8_t first_missing_write_chunk(const operation_entry_v0 &entry) {
  const uint8_t count = modeled_write_count(entry);
  for (uint8_t index = 0; index < count; ++index) {
    if ((entry.enqueued_write_mask & (1u << index)) == 0) {
      return index;
    }
  }
  return count;
}

bool valid_operation_input(const reservation_input_v0 &input,
                           bool existing_target) {
  if (input.producer_operation_seq == 0 ||
      input.pending_parent_resume_valid > 1 ||
      input.recovery_target_inflight > 1 ||
      input.deferred_instance_valid > 1 ||
      input.recovery_target_completed > 1 ||
      (input.private_storage_profile !=
           private_storage::kProfileLegacyShared832 &&
       input.private_storage_profile !=
           private_storage::kProfileCompressedShared384 &&
       input.private_storage_profile !=
           private_storage::kProfileGlobal384)) {
    return false;
  }
  const bool global = input.private_storage_profile ==
                      private_storage::kProfileGlobal384;
  if (global &&
      (input.private_slot_base_address == 0 ||
       input.private_slot_base_address %
               private_state_384::kChunkBytes !=
           0)) {
    return false;
  }
  if (existing_target != (input.target_operation_seq != 0) ||
      existing_target != (input.producer_commit_epoch != 0)) {
    return false;
  }
  if (input.operation_kind == kOperationNodeTransition) {
    return !existing_target && input.recovery_target_completed == 0;
  }
  if (input.operation_kind == kOperationResumeTransition) {
    return existing_target && input.deferred_instance_valid == 0 &&
           (input.recovery_target_completed == 0 ||
            input.recovery_target_inflight != 0);
  }
  if (input.operation_kind == kOperationEnterBlasTransition) {
    return existing_target && input.blas_build_generation != 0 &&
           input.recovery_target_completed == 0 &&
           (input.private_storage_profile ==
                    private_storage::kProfileLegacyShared832 ||
            input.deferred_instance_valid == 1);
  }
  return false;
}

status_kind reserve_impl(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    bool existing_target, reservation_receipt_v0 *reservation,
    request_plan_v0 *requests) {
  if (state == NULL || timing_state == NULL || reservation == NULL ||
      requests == NULL || state->initialized != 1 ||
      !valid_config(state->config) ||
      !valid_operation_input(input, existing_target) ||
      input.private_storage_profile !=
          private_storage::kProfileLegacyShared832) {
    return kStatusInvalidArgument;
  }
  *reservation = reservation_receipt_v0();
  *requests = request_plan_v0();
  request_owner::lane_binding_v0 request_binding = {};
  const private_shared::lane_slot_state_v0 *lane =
      private_shared::find_live_lane(private_backing, input.owner);
  if (!make_request_owner(input.owner, &request_binding) ||
      lane == NULL) {
    return kStatusOwnerMismatch;
  }
  if (state->reservation_cycle != reservation_cycle) {
    state->reservation_cycle = reservation_cycle;
    state->reservations_this_cycle = 0;
  }
  const uint32_t attempted_operation_seq =
      input.target_operation_seq != 0
          ? input.target_operation_seq
          : input.producer_operation_seq;
  if (state->reservations_this_cycle >=
      state->config.reservation_width) {
    emit_attempt(
        input.owner, attempted_operation_seq, reservation_cycle,
        state->reservations_this_cycle,
        stall_attribution::kStageAdmission,
        stall_attribution::kOutcomeStall,
        stall_attribution::kActionNone,
        stall_attribution::kReasonReservationBudget);
    return kStatusReservationBudgetBackpressure;
  }
  const int slot_index = find_free_slot(*state);
  if (slot_index < 0) {
    emit_attempt(
        input.owner, attempted_operation_seq, reservation_cycle,
        state->reservations_this_cycle,
        stall_attribution::kStageAdmission,
        stall_attribution::kOutcomeStall,
        stall_attribution::kActionNone,
        stall_attribution::kReasonQueueCapacity);
    return kStatusCapacityBackpressure;
  }

  private_frontier::access_plan_v0 read_plan = {};
  if (short_stack_shared::build_persistent_state_read_plan(
          lane->canonical_slot, input.owner, input.private_region,
          &read_plan) != short_stack_shared::kStatusOk ||
      read_plan.access_count != kReadChunkCount) {
    return kStatusSharedPlanRejected;
  }

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  uint32_t operation_seq = input.target_operation_seq;
  if (existing_target) {
    const timing_driver::lane_control_state_v0 *control =
        timing_driver::find_live_lane_control(
            staged_timing, request_binding);
    if (control == NULL ||
        control->live_target_operation_seq != operation_seq ||
        control->live_commit_producer_operation_seq !=
            input.producer_operation_seq ||
        control->live_commit_epoch != input.producer_commit_epoch ||
        control->pending_recovery_operation_seq != 0 ||
        control->pending_terminal_kind !=
            timing_driver::kTerminalBoundaryInvalid ||
        control->live_memory_transaction_count != 0 ||
        control->live_commit_memory_transaction_count != 0) {
      return kStatusTimingControlRejected;
    }
  } else if (timing_driver::allocate_target_operation(
                 &staged_timing, request_binding, &operation_seq) !=
             timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }

  operation_entry_v0 &entry = staged_state.slots[slot_index];
  const uint32_t next_generation =
      entry.reservation.slot_generation == UINT32_MAX
          ? 1
          : entry.reservation.slot_generation + 1;
  entry = operation_entry_v0();
  entry.input = input;
  entry.read_slot.owner = input.owner;
  entry.read_plan = read_plan;
  entry.issue_age = staged_state.next_age++;
  entry.phase = kPhaseReading;
  entry.valid = 1;
  entry.reservation.owner = input.owner;
  entry.reservation.reservation_id =
      staged_state.next_reservation_id++;
  entry.reservation.reservation_age = entry.issue_age;
  entry.reservation.operation_seq = operation_seq;
  entry.reservation.producer_operation_seq =
      input.producer_operation_seq;
  entry.reservation.slot_generation = next_generation;
  entry.reservation.slot_index = static_cast<uint8_t>(slot_index);
  entry.reservation.read_chunk_count = kReadChunkCount;
  entry.reservation.valid = 1;
  if (timing_driver::bind_target_operation(
          &staged_timing, request_binding, operation_seq,
          timing_driver::kTargetOperationClassShortStack,
          input.operation_kind, entry.reservation.reservation_id) !=
      timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }

  const uint64_t slot_base = private_slot_base(input.owner);
  for (unsigned index = 0; index < kReadChunkCount; ++index) {
    const private_frontier::shared_chunk_access_v0 &access =
        read_plan.accesses[index];
    if (access.access_kind != private_frontier::kAccessRead ||
        access.aligned_32b_address < slot_base ||
        access.aligned_32b_address - slot_base > UINT16_MAX ||
        access.byte_mask == 0) {
      return kStatusSharedPlanRejected;
    }
    rtcore_memory_unit_request_snapshot &request =
        requests->requests[index];
    request.valid = true;
    request.address_space = RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
    request.operation = RTCORE_MEMORY_OPERATION_READ;
    request.destination =
        RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL;
    request.response_target = kResponseTargetRtcore;
    request.owner_hw_sid = input.owner.owner_hw_sid;
    request.rt_request_id = input.owner.request_identity;
    request.lane_id = input.owner.lane_id;
    request.resident_warp_id = input.owner.resident_warp_id;
    request.request_generation = input.owner.generation;
    request.private_slot_id = input.owner.private_slot_id;
    request.memory_op_seq = index + 1;
    request.chunk_id = index;
    request.chunk_count = kReadChunkCount;
    request.access_kind =
        RTCORE_MEMORY_ACCESS_SHORT_STACK_STATE_READ;
    request.aligned_32b_addr = access.aligned_32b_address;
    request.byte_mask = access.byte_mask;
    request.is_write = false;
    request.issue_cycle = reservation_cycle;
    rtcore_v04_stack_private_read_transport_snapshot &transport =
        request.v04_stack_private_read;
    transport.reservation_id = entry.reservation.reservation_id;
    transport.reservation_age = entry.reservation.reservation_age;
    transport.target_operation_seq = operation_seq;
    transport.producer_operation_seq =
        input.producer_operation_seq;
    transport.target_slot_generation =
        entry.reservation.slot_generation;
    transport.slot_chunk_offset = static_cast<uint16_t>(
        access.aligned_32b_address - slot_base);
    transport.target_slot_index =
        entry.reservation.slot_index;
    transport.field_kind = access.field_kind;
    transport.operation_kind = input.operation_kind;
    transport.read_phase = kReadPhaseShortStackState;
    transport.valid = 1;
    if (!request_matches_entry(request, entry) ||
        timing_driver::begin_memory_transaction(
            &staged_timing, request_binding, operation_seq) !=
            timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
  }
  requests->request_count = kReadChunkCount;
  requests->valid = 1;
  ++staged_state.reservations_this_cycle;
  *reservation = entry.reservation;
  *state = staged_state;
  *timing_state = staged_timing;
  emit_attempt(
      input.owner, operation_seq, reservation_cycle,
      static_cast<uint16_t>(
          staged_state.reservations_this_cycle - 1),
      stall_attribution::kStageAdmission,
      stall_attribution::kOutcomeProgress,
      stall_attribution::kActionAdmissionAccept,
      stall_attribution::kReasonNone);
  return kStatusOk;
}

}  // namespace

config_v0 candidate_profile_config() {
  config_v0 config = {};
  config.capacity = 16;
  config.reservation_width = 4;
  config.unit_count = 1;
  config.stack_latency = 2;
  config.initiation_interval = 1;
  config.issue_width = 1;
  config.parent_lookup_latency = 2;
  return config;
}

status_kind initialize(engine_state_v0 *state, const config_v0 &config) {
  if (state == NULL || !valid_config(config)) {
    return kStatusInvalidConfig;
  }
  *state = engine_state_v0();
  state->initialized = 1;
  state->config = config;
  state->next_reservation_id = 1;
  state->next_age = 1;
  return kStatusOk;
}

status_kind reserve(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    reservation_receipt_v0 *reservation, request_plan_v0 *requests) {
  reservation_input_v0 normalized = input;
  if (normalized.operation_kind == kOperationInvalid) {
    normalized.operation_kind = kOperationNodeTransition;
  }
  if (normalized.private_storage_profile == 0) {
    normalized.private_storage_profile =
        private_storage::kProfileLegacyShared832;
  }
  return reserve_impl(
      state, timing_state, private_backing, normalized,
      reservation_cycle, false, reservation, requests);
}

status_kind reserve_existing_target(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    reservation_receipt_v0 *reservation, request_plan_v0 *requests) {
  reservation_input_v0 normalized = input;
  if (normalized.private_storage_profile == 0) {
    normalized.private_storage_profile =
        private_storage::kProfileLegacyShared832;
  }
  return reserve_impl(
      state, timing_state, private_backing, normalized,
      reservation_cycle, true, reservation, requests);
}

status_kind reserve_private_state_384(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_state_384::backing::state_v1 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    reservation_receipt_v0 *reservation, request_plan_v0 *requests) {
  const bool existing_target = input.target_operation_seq != 0;
  if (state == NULL || timing_state == NULL || reservation == NULL ||
      requests == NULL || state->initialized != 1 ||
      !valid_config(state->config) ||
      !valid_operation_input(input, existing_target) ||
      (input.private_storage_profile !=
           private_storage::kProfileCompressedShared384 &&
       input.private_storage_profile !=
           private_storage::kProfileGlobal384)) {
    return kStatusInvalidArgument;
  }
  *reservation = reservation_receipt_v0();
  *requests = request_plan_v0();
  request_owner::lane_binding_v0 request_binding = {};
  const bool global = input.private_storage_profile ==
                      private_storage::kProfileGlobal384;
  if (!make_request_owner(input.owner, &request_binding) ||
      (!global &&
       private_state_384::backing::find_live_lane(
           private_backing, input.owner) == NULL)) {
    return kStatusOwnerMismatch;
  }
  if (state->reservation_cycle != reservation_cycle) {
    state->reservation_cycle = reservation_cycle;
    state->reservations_this_cycle = 0;
  }
  if (state->reservations_this_cycle >=
      state->config.reservation_width) {
    emit_attempt(
        input.owner, input.producer_operation_seq, reservation_cycle,
        state->reservations_this_cycle,
        stall_attribution::kStageAdmission,
        stall_attribution::kOutcomeStall,
        stall_attribution::kActionNone,
        stall_attribution::kReasonReservationBudget);
    return kStatusReservationBudgetBackpressure;
  }
  const int slot_index = find_free_slot(*state);
  if (slot_index < 0) {
    emit_attempt(
        input.owner, input.producer_operation_seq, reservation_cycle,
        state->reservations_this_cycle,
        stall_attribution::kStageAdmission,
        stall_attribution::kOutcomeStall,
        stall_attribution::kActionNone,
        stall_attribution::kReasonQueueCapacity);
    return kStatusCapacityBackpressure;
  }

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  uint32_t operation_seq = input.target_operation_seq;
  if (existing_target) {
    const timing_driver::lane_control_state_v0 *control =
        timing_driver::find_live_lane_control(
            staged_timing, request_binding);
    if (control == NULL ||
        control->live_target_operation_seq != operation_seq ||
        control->live_commit_producer_operation_seq !=
            input.producer_operation_seq ||
        control->live_commit_epoch != input.producer_commit_epoch ||
        control->pending_recovery_operation_seq != 0 ||
        control->pending_terminal_kind !=
            timing_driver::kTerminalBoundaryInvalid ||
        control->live_memory_transaction_count != 0 ||
        control->live_commit_memory_transaction_count != 0) {
      return kStatusTimingControlRejected;
    }
  } else if (timing_driver::allocate_target_operation(
                 &staged_timing, request_binding, &operation_seq) !=
             timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  const uint32_t next_generation =
      entry.reservation.slot_generation == UINT32_MAX
          ? 1
          : entry.reservation.slot_generation + 1;
  entry = operation_entry_v0();
  entry.input = input;
  entry.issue_age = staged_state.next_age++;
  entry.phase = kPhaseReading;
  entry.valid = 1;
  entry.reservation.owner = input.owner;
  entry.reservation.reservation_id =
      staged_state.next_reservation_id++;
  entry.reservation.reservation_age = entry.issue_age;
  entry.reservation.operation_seq = operation_seq;
  entry.reservation.producer_operation_seq =
      input.producer_operation_seq;
  entry.reservation.slot_generation = next_generation;
  entry.reservation.slot_index = static_cast<uint8_t>(slot_index);
  entry.reservation.valid = 1;

  private_state_384::live_bridge::read_input_v1 read_input = {};
  read_input.owner = input.owner;
  read_input.private_slot_base_address =
      input.private_slot_base_address;
  read_input.issue_cycle = reservation_cycle;
  read_input.operation_sequence = operation_seq;
  read_input.bvh_format_profile_id =
      private_state_384::kGenRtBvhFormatProfileId;
  read_input.reservation_generation = next_generation;
  read_input.storage_profile = input.private_storage_profile;
  read_input.consumer =
      private_state_384::operand_plan::kConsumerStack;
  read_input.operation =
      private_state_384::operand_plan::kOperationDefault;
  read_input.completion_reason =
      private_state_384::operand_plan::kCompletionReasonNone;
  read_input.destination =
      RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL;
  read_input.memory_op_seq_base = 1;
  private_state_384::live_bridge::read_request_plan_v1 read_plan = {};
  if (private_state_384::live_bridge::prepare_read_requests(
          read_input, &read_plan) !=
          private_state_384::live_bridge::kStatusOk ||
      read_plan.request_count == 0 ||
      read_plan.request_count > kMaxReadChunkCount ||
      private_state_384::live_bridge::initialize_collector(
          read_plan, &entry.private_state_384_collector) !=
          private_state_384::live_bridge::kStatusOk) {
    return kStatusSharedPlanRejected;
  }
  entry.reservation.read_chunk_count = read_plan.request_count;
  if (timing_driver::bind_target_operation(
          &staged_timing, request_binding, operation_seq,
          timing_driver::kTargetOperationClassShortStack,
          input.operation_kind, entry.reservation.reservation_id) !=
      timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  for (uint8_t index = 0; index < read_plan.request_count; ++index) {
    if (timing_driver::begin_memory_transaction(
            &staged_timing, request_binding, operation_seq) !=
        timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
    requests->requests[index] = read_plan.requests[index];
  }
  requests->request_count = read_plan.request_count;
  requests->valid = 1;
  ++staged_state.reservations_this_cycle;
  *reservation = entry.reservation;
  *state = staged_state;
  *timing_state = staged_timing;
  emit_attempt(
      input.owner, operation_seq, reservation_cycle,
      static_cast<uint16_t>(
          staged_state.reservations_this_cycle - 1),
      stall_attribution::kStageAdmission,
      stall_attribution::kOutcomeProgress,
      stall_attribution::kActionAdmissionAccept,
      stall_attribution::kReasonNone);
  return kStatusOk;
}

status_kind accept_read_response(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const rtcore_memory_unit_request_snapshot &request,
    uint64_t response_cycle, const uint8_t *return_instance_payload,
    uint8_t return_instance_payload_bytes) {
  if (state == NULL || timing_state == NULL ||
      state->initialized != 1 ||
      request.v04_stack_private_read.target_slot_index >=
          state->config.capacity) {
    return kStatusInvalidArgument;
  }
  const uint8_t slot_index =
      request.v04_stack_private_read.target_slot_index;
  const operation_entry_v0 &current = state->slots[slot_index];
  if (!request_matches_entry(request, current)) {
    return kStatusMalformedTransport;
  }
  const bool return_instance_read =
      request.v04_stack_private_read.read_phase ==
      kReadPhaseReturnInstance;
  if (return_instance_read !=
          (return_instance_payload != NULL) ||
      (return_instance_read &&
       return_instance_payload_bytes !=
           private_frontier::kSharedAccessChunkBytes) ||
      (!return_instance_read &&
       return_instance_payload_bytes != 0)) {
    return kStatusMalformedTransport;
  }
  const uint8_t chunk_bit =
      static_cast<uint8_t>(1u << request.chunk_id);
  const uint8_t current_mask =
      return_instance_read
          ? current.received_return_instance_mask
          : current.received_read_mask;
  if ((current_mask & chunk_bit) != 0) {
    return kStatusDuplicateResponse;
  }

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  if (!return_instance_read) {
    private_frontier::shared_chunk_access_v0 access = {};
    access.aligned_32b_address = request.aligned_32b_addr;
    access.byte_mask = request.byte_mask;
    access.field_kind =
        request.v04_stack_private_read.field_kind;
    access.access_kind = private_frontier::kAccessRead;
    uint8_t payload[private_frontier::kSharedAccessChunkBytes] = {};
    if (private_shared::read_canonical_chunk(
            private_backing, entry.input.owner, access, payload) !=
        private_shared::kStatusOk) {
      return kStatusSharedPlanRejected;
    }
    const uint32_t chunk_offset =
        request.v04_stack_private_read.slot_chunk_offset;
    if (chunk_offset >
        private_frontier::kPrivateDataSlotBytes -
            private_frontier::kSharedAccessChunkBytes) {
      return kStatusMalformedTransport;
    }
    for (unsigned byte = 0;
         byte < private_frontier::kSharedAccessChunkBytes; ++byte) {
      if ((request.byte_mask & (uint32_t{1} << byte)) != 0) {
        entry.read_slot.bytes[chunk_offset + byte] = payload[byte];
      }
    }
  }
  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry.input.owner, &request_binding) ||
      timing_driver::complete_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  if (return_instance_read) {
    std::memcpy(
        entry.return_instance_payload +
            request.chunk_id *
                private_frontier::kSharedAccessChunkBytes,
        return_instance_payload,
        private_frontier::kSharedAccessChunkBytes);
    entry.received_return_instance_mask = static_cast<uint8_t>(
        entry.received_return_instance_mask | chunk_bit);
    if (entry.received_return_instance_mask ==
        kAllReturnInstanceReadChunks) {
      typed_instance::raw_instance_payload_v0 raw_instance = {};
      typed_instance::boundary_input_v0 decode_input = {};
      decode_input.profile_id =
          typed_instance::kGenRtDerivedProfileId;
      if (!typed_instance::make_raw_instance_payload(
              entry.return_instance_payload, &raw_instance)) {
        return kStatusReturnInstanceRejected;
      }
      decode_input.raw_instance = raw_instance;
      const typed_instance::boundary_result_v0 decoded =
          typed_instance::execute(decode_input);
      if (decoded.status != typed_instance::kStatusOk) {
        return kStatusReturnInstanceRejected;
      }
      if (entry.input.private_storage_profile ==
              private_storage::kProfileCompressedShared384 ||
          entry.input.private_storage_profile ==
              private_storage::kProfileGlobal384) {
        short_stack::entry_v0 return_entry = {};
        if (entry.private_state_384_cross_as_operands_valid != 1 ||
            !short_stack::read_logical_entry(
                entry.private_state_384_cross_as_operands
                    .base.stack,
                0, &return_entry) ||
            !make_return_instance_projection(
                decoded,
                entry.private_state_384_cross_as_operands
                    .parent.tlas_decode_context,
                return_entry,
                &entry.return_instance_projection)) {
          return kStatusReturnInstanceRejected;
        }
        entry.return_instance_projection_valid = 1;
      }
      entry.phase = kPhaseReadyToIssue;
      entry.result_ready_cycle = response_cycle;
    }
  } else {
    entry.received_read_mask =
        static_cast<uint8_t>(entry.received_read_mask | chunk_bit);
    if (entry.received_read_mask == kAllReadChunks) {
      short_stack_shared::persistent_state_v0 persistent = {};
      short_stack::entry_v0 return_entry = {};
      if (short_stack_shared::decode_persistent_state(
              entry.read_slot, entry.input.owner, &persistent) !=
          short_stack_shared::kStatusOk) {
        return kStatusSharedPlanRejected;
      }
      const bool return_instance_required =
          entry.input.operation_kind ==
                  kOperationResumeTransition &&
              persistent.stack.cross_as != 0 &&
              persistent.stack.stack_count == 1 &&
              short_stack::read_logical_entry(
                  persistent.stack, 0, &return_entry) &&
              short_stack::control_kind(return_entry.control) ==
                  short_stack::kEntryCrossAsReturn;
      entry.phase = return_instance_required
                        ? kPhaseReturnInstancePlanReady
                        : kPhaseReadyToIssue;
      entry.result_ready_cycle = response_cycle;
    }
  }
  *state = staged_state;
  *timing_state = staged_timing;
  return kStatusOk;
}

static status_kind accept_private_state_384_read_response_impl(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_state_384::backing::state_v1 *private_backing,
    const rtcore_memory_unit_request_snapshot &request,
    const uint8_t *payload, size_t payload_byte_count,
    uint64_t response_cycle) {
  const bool compressed =
      request.v04_private_state_384_read.storage_profile ==
      private_storage::kProfileCompressedShared384;
  const bool global =
      request.v04_private_state_384_read.storage_profile ==
      private_storage::kProfileGlobal384;
  if (state == NULL || timing_state == NULL ||
      state->initialized != 1 ||
      request.access_kind !=
          RTCORE_MEMORY_ACCESS_PRIVATE_STATE_384_READ ||
      request.destination !=
          RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL ||
      (!compressed && !global) ||
      (compressed &&
       (private_backing == NULL || payload != NULL ||
        payload_byte_count != 0)) ||
      (global &&
       (private_backing != NULL || payload == NULL ||
        payload_byte_count != private_state_384::kChunkBytes)) ||
      (request.v04_private_state_384_read.consumer !=
           private_state_384::operand_plan::kConsumerStack &&
       request.v04_private_state_384_read.consumer !=
           private_state_384::operand_plan::
               kConsumerCompletionPublisher)) {
    return kStatusInvalidArgument;
  }
  int slot_index = -1;
  for (unsigned index = 0; index < state->config.capacity; ++index) {
    const operation_entry_v0 &entry = state->slots[index];
    if (entry.valid != 0 && entry.phase == kPhaseReading &&
        entry.input.private_storage_profile ==
            request.v04_private_state_384_read.storage_profile &&
        request.v04_private_state_384_read.consumer ==
            entry.private_state_384_collector.consumer &&
        request.v04_private_state_384_read.operation_kind ==
            entry.private_state_384_collector.operation &&
        request.v04_private_state_384_read.completion_reason ==
            entry.private_state_384_collector.completion_reason &&
        entry.reservation.operation_seq ==
            request.v04_private_state_384_read.operation_sequence &&
        entry.reservation.slot_generation ==
            request.v04_private_state_384_read.reservation_generation &&
        entry.input.owner.owner_hw_sid == request.owner_hw_sid &&
        entry.input.owner.request_identity == request.rt_request_id &&
        entry.input.owner.generation == request.request_generation &&
        entry.input.owner.private_slot_id == request.private_slot_id &&
        entry.input.owner.resident_warp_id ==
            request.resident_warp_id &&
        entry.input.owner.lane_id == request.lane_id) {
      if (slot_index >= 0) return kStatusMalformedTransport;
      slot_index = static_cast<int>(index);
    }
  }
  if (slot_index < 0) return kStatusMalformedTransport;

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  const private_state_384::live_bridge::status_kind bridge_status =
      compressed
          ? private_state_384::live_bridge::accept_read_response(
                *private_backing, request,
                &entry.private_state_384_collector)
          : private_state_384::live_bridge::accept_read_response_bytes(
                request, payload, payload_byte_count,
                &entry.private_state_384_collector);
  if (bridge_status != private_state_384::live_bridge::kStatusOk) {
    return bridge_status ==
                   private_state_384::live_bridge::kStatusCollectorRejected
               ? kStatusDuplicateResponse
               : kStatusMalformedTransport;
  }
  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry.input.owner, &request_binding) ||
      timing_driver::complete_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  entry.received_read_mask =
      entry.private_state_384_collector.received_chunk_mask;
  if (private_state_384::operand_materializer::responses_complete(
          entry.private_state_384_collector)) {
    if (entry.private_state_384_collector.consumer ==
        private_state_384::operand_plan::
            kConsumerCompletionPublisher) {
      if (private_state_384::operand_materializer::
              materialize_final_completion(
                  entry.private_state_384_collector,
                  &entry.private_state_384_completion_operands) !=
              private_state_384::operand_materializer::kStatusOk ||
          entry.private_state_384_completion_operands
                  .completion_reason !=
              entry.private_state_384_completion_reason) {
        return kStatusSharedPlanRejected;
      }
      entry.private_state_384_completion_operands_valid = 1;
      entry.phase = kPhaseResultReady;
      entry.result_ready_cycle = response_cycle;
      *state = staged_state;
      *timing_state = staged_timing;
      return kStatusOk;
    }
    if (entry.private_state_384_collector.consumer !=
        private_state_384::operand_plan::kConsumerStack) {
      return kStatusSharedPlanRejected;
    }
    private_state_384::operand_materializer::materialize_context_v1
        context = {};
    context.bvh_format_profile_id =
        private_state_384::kGenRtBvhFormatProfileId;
    context.recovery_target_inflight =
        entry.input.recovery_target_inflight;
    const uint8_t selected_operation =
        entry.private_state_384_collector.operation;
    if (selected_operation ==
        private_state_384::operand_plan::kOperationDefault) {
      private_state_384::operand_materializer::stack_metadata_v1
          metadata = {};
      if (private_state_384::operand_materializer::
              materialize_stack_metadata(
                  entry.private_state_384_collector,
                  &metadata) !=
          private_state_384::operand_materializer::kStatusOk) {
        return kStatusSharedPlanRejected;
      }
      if (metadata.stack_count != 0) {
        entry.private_state_384_selected_operation =
            private_state_384::operand_plan::
                kOperationStackEntries;
        entry.phase = kPhasePrivate384FollowupPlanReady;
        entry.result_ready_cycle = response_cycle;
        *state = staged_state;
        *timing_state = staged_timing;
        return kStatusOk;
      }
    }
    if (selected_operation ==
            private_state_384::operand_plan::kOperationDefault ||
        selected_operation ==
            private_state_384::operand_plan::
                kOperationStackEntries) {
      if (private_state_384::operand_materializer::
              materialize_stack_base(
                  entry.private_state_384_collector, context,
                  &entry.private_state_384_stack_operands) !=
          private_state_384::operand_materializer::kStatusOk) {
        return kStatusSharedPlanRejected;
      }
    } else if (
        selected_operation ==
        private_state_384::operand_plan::kOperationStackTerminal) {
      if (private_state_384::operand_materializer::
              materialize_stack_terminal(
                  entry.private_state_384_collector, context,
                  &entry.private_state_384_terminal_operands) !=
          private_state_384::operand_materializer::kStatusOk) {
        return kStatusSharedPlanRejected;
      }
      entry.private_state_384_stack_operands =
          entry.private_state_384_terminal_operands.base;
      entry.private_state_384_terminal_operands_valid = 1;
    } else if (
        selected_operation ==
        private_state_384::operand_plan::
            kOperationStackCrossAsReturn) {
      if (private_state_384::operand_materializer::
              materialize_stack_cross_as(
                  entry.private_state_384_collector, context,
                  &entry.private_state_384_cross_as_operands) !=
          private_state_384::operand_materializer::kStatusOk) {
        return kStatusSharedPlanRejected;
      }
      if (entry.return_transition_state_carry_valid != 0) {
        if (entry.input.operation_kind !=
                kOperationResumeTransition ||
            !short_stack_shared::validate_persistent_state(
                entry.transition.persistent_state)) {
          return kStatusSharedPlanRejected;
        }
        entry.private_state_384_cross_as_operands.base
            .tlas_build_generation =
            entry.transition.persistent_state
                .tlas_build_generation;
        entry.private_state_384_cross_as_operands.base
            .blas_build_generation =
            entry.transition.persistent_state
                .blas_build_generation;
        entry.private_state_384_cross_as_operands.base.stack =
            entry.transition.persistent_state.stack;
        entry.input.recovery_target_inflight =
            entry.transition.persistent_state
                .recovery_target_inflight;
        entry.return_transition_state_carry_valid = 0;
      }
      entry.private_state_384_stack_operands =
          entry.private_state_384_cross_as_operands.base;
      entry.private_state_384_cross_as_operands_valid = 1;
    } else {
      return kStatusSharedPlanRejected;
    }
    short_stack_shared::persistent_state_v0 persistent = {};
    persistent.tlas_build_generation =
        entry.private_state_384_stack_operands
            .tlas_build_generation;
    persistent.blas_build_generation =
        entry.private_state_384_stack_operands
            .blas_build_generation;
    persistent.stack =
        entry.private_state_384_stack_operands.stack;
    persistent.recovery_target_inflight =
        entry.input.recovery_target_inflight;
    if (!short_stack_shared::validate_persistent_state(persistent)) {
      return kStatusSharedPlanRejected;
    }
    entry.input.ray_policy =
        entry.private_state_384_stack_operands.ray_policy;
    entry.input.current_decode_context =
        entry.private_state_384_stack_operands
            .active_decode_context;
    entry.input.active_decode_context =
        entry.private_state_384_stack_operands
            .active_decode_context;
    entry.private_state_384_operands_valid = 1;
    if (selected_operation ==
        private_state_384::operand_plan::kOperationDefault) {
      const bool terminal_followup =
          entry.input.operation_kind ==
              kOperationResumeTransition &&
          persistent.stack.stack_count == 0;
      if (terminal_followup) {
        entry.private_state_384_selected_operation =
            private_state_384::operand_plan::
                kOperationStackTerminal;
        entry.phase = kPhasePrivate384FollowupPlanReady;
      } else {
        entry.phase = kPhaseReadyToIssue;
      }
    } else if (
        selected_operation ==
        private_state_384::operand_plan::kOperationStackEntries) {
      const bool cross_as_followup =
          entry.input.operation_kind ==
              kOperationResumeTransition &&
          persistent.stack.cross_as != 0 &&
          persistent.stack.stack_count == 1;
      if (cross_as_followup) {
        entry.private_state_384_selected_operation =
            private_state_384::operand_plan::
                kOperationStackCrossAsReturn;
        entry.phase = kPhasePrivate384FollowupPlanReady;
      } else {
        entry.phase = kPhaseReadyToIssue;
      }
    } else if (
        selected_operation ==
        private_state_384::operand_plan::
            kOperationStackCrossAsReturn) {
      entry.phase = kPhaseReturnInstancePlanReady;
    } else {
      entry.phase = kPhaseReadyToIssue;
    }
    entry.result_ready_cycle = response_cycle;
  }
  *state = staged_state;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind accept_private_state_384_read_response(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_state_384::backing::state_v1 &private_backing,
    const rtcore_memory_unit_request_snapshot &request,
    uint64_t response_cycle) {
  return accept_private_state_384_read_response_impl(
      state, timing_state, &private_backing, request, NULL, 0,
      response_cycle);
}

status_kind accept_private_state_384_read_response_bytes(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const rtcore_memory_unit_request_snapshot &request,
    const uint8_t *payload, size_t payload_byte_count,
    uint64_t response_cycle) {
  return accept_private_state_384_read_response_impl(
      state, timing_state, NULL, request, payload,
      payload_byte_count, response_cycle);
}

status_kind take_private_state_384_followup_read_plan(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t issue_cycle, request_plan_v0 *requests) {
  if (state == NULL || timing_state == NULL || requests == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *requests = request_plan_v0();
  const int stack_followup_index = find_oldest_phase(
      *state, kPhasePrivate384FollowupPlanReady, 0, false);
  const int completion_followup_index = find_oldest_phase(
      *state, kPhasePrivate384CompletionPlanReady, 0, false);
  const bool completion_followup =
      completion_followup_index >= 0 &&
      (stack_followup_index < 0 ||
       state->slots[completion_followup_index].issue_age <
           state->slots[stack_followup_index].issue_age);
  const int slot_index = completion_followup
                             ? completion_followup_index
                             : stack_followup_index;
  if (slot_index < 0) return kStatusNoFollowupRead;

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  const bool stack_entries_followup =
      !completion_followup &&
      entry.private_state_384_selected_operation ==
          private_state_384::operand_plan::
              kOperationStackEntries;
  if ((entry.input.private_storage_profile !=
           private_storage::kProfileCompressedShared384 &&
       entry.input.private_storage_profile !=
           private_storage::kProfileGlobal384) ||
      (stack_entries_followup
           ? entry.private_state_384_operands_valid != 0
           : entry.private_state_384_operands_valid != 1)) {
    return kStatusSharedPlanRejected;
  }
  if (completion_followup) {
    const uint8_t expected_reason =
        entry.private_state_384_stack_operands.committed_valid != 0
            ? private_state_384::operand_plan::
                  kCompletionReasonClosestHitReady
            : private_state_384::operand_plan::
                  kCompletionReasonMiss;
    if (entry.transition.terminal != 1 ||
        entry.transition.selected_valid != 0 ||
        entry.commit_epoch == 0 ||
        entry.private_state_384_completion_operands_valid != 0 ||
        entry.private_state_384_completion_reason !=
            expected_reason) {
      return kStatusSharedPlanRejected;
    }
  } else if (
      entry.private_state_384_selected_operation !=
          private_state_384::operand_plan::
              kOperationStackEntries &&
      entry.private_state_384_selected_operation !=
          private_state_384::operand_plan::
              kOperationStackTerminal &&
      entry.private_state_384_selected_operation !=
          private_state_384::operand_plan::
              kOperationStackCrossAsReturn) {
    return kStatusSharedPlanRejected;
  }

  private_state_384::live_bridge::read_input_v1 read_input = {};
  read_input.owner = entry.input.owner;
  read_input.private_slot_base_address =
      entry.input.private_slot_base_address;
  read_input.issue_cycle = issue_cycle;
  read_input.operation_sequence =
      entry.reservation.operation_seq;
  read_input.bvh_format_profile_id =
      private_state_384::kGenRtBvhFormatProfileId;
  read_input.reservation_generation =
      entry.reservation.slot_generation;
  read_input.storage_profile =
      entry.input.private_storage_profile;
  read_input.consumer =
      completion_followup
          ? private_state_384::operand_plan::
                kConsumerCompletionPublisher
          : private_state_384::operand_plan::kConsumerStack;
  if (completion_followup) {
    read_input.operation =
        private_state_384::operand_plan::kOperationDefault;
    read_input.completion_reason =
        entry.private_state_384_completion_reason;
  } else {
    read_input.operation =
        entry.private_state_384_selected_operation;
    read_input.completion_reason =
        private_state_384::operand_plan::kCompletionReasonNone;
  }
  read_input.destination =
      RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL;
  read_input.memory_op_seq_base =
      completion_followup ? 24 : 16;
  private_state_384::live_bridge::read_request_plan_v1 full_plan = {};
  private_state_384::operand_materializer::response_collector_v1
      promoted = {};
  if (private_state_384::live_bridge::prepare_read_requests(
          read_input, &full_plan) !=
      private_state_384::live_bridge::kStatusOk) {
    return kStatusSharedPlanRejected;
  }
  if (completion_followup) {
    if (private_state_384::live_bridge::initialize_collector(
            full_plan, &promoted) !=
        private_state_384::live_bridge::kStatusOk) {
      return kStatusSharedPlanRejected;
    }
  } else if (
      private_state_384::operand_materializer::
          promote_stack_collector(
              entry.private_state_384_collector,
              entry.private_state_384_selected_operation,
              &promoted) !=
      private_state_384::operand_materializer::kStatusOk) {
    return kStatusSharedPlanRejected;
  }

  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry.input.owner, &request_binding)) {
    return kStatusOwnerMismatch;
  }
  for (uint8_t index = 0; index < full_plan.request_count; ++index) {
    const uint8_t chunk_index =
        full_plan.operand_plan.reads[index].chunk_index;
    const uint16_t chunk_bit = static_cast<uint16_t>(
        uint16_t{1} << chunk_index);
    if (!completion_followup &&
        (promoted.received_chunk_mask & chunk_bit) != 0) {
      continue;
    }
    if (requests->request_count >= kMaxReadChunkCount ||
        timing_driver::begin_memory_transaction(
            &staged_timing, request_binding,
            entry.reservation.operation_seq) !=
            timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
    requests->requests[requests->request_count++] =
        full_plan.requests[index];
  }
  const uint8_t expected_missing =
      completion_followup
          ? 2
          : entry.private_state_384_selected_operation ==
                    private_state_384::operand_plan::
                        kOperationStackEntries
          ? 3
          : entry.private_state_384_selected_operation ==
                    private_state_384::operand_plan::
                        kOperationStackTerminal
          ? 1
          : 2;
  if (requests->request_count != expected_missing) {
    return kStatusSharedPlanRejected;
  }
  requests->valid = 1;
  entry.private_state_384_collector = promoted;
  entry.received_read_mask = promoted.received_chunk_mask;
  entry.reservation.read_chunk_count =
      full_plan.request_count;
  entry.phase = kPhaseReading;
  *state = staged_state;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind take_return_instance_read_plan(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t issue_cycle, request_plan_v0 *requests) {
  if (state == NULL || timing_state == NULL || requests == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *requests = request_plan_v0();
  const int slot_index = find_oldest_phase(
      *state, kPhaseReturnInstancePlanReady, 0, false);
  if (slot_index < 0) return kStatusNoFollowupRead;

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  short_stack_shared::persistent_state_v0 persistent = {};
  short_stack::entry_v0 return_entry = {};
  typed_blas::as_decode_context_v0 tlas = {};
  if (entry.input.private_storage_profile ==
          private_storage::kProfileCompressedShared384 ||
      entry.input.private_storage_profile ==
          private_storage::kProfileGlobal384) {
    if (entry.private_state_384_cross_as_operands_valid != 1) {
      return kStatusSharedPlanRejected;
    }
    persistent.tlas_build_generation =
        entry.private_state_384_cross_as_operands
            .base.tlas_build_generation;
    persistent.blas_build_generation =
        entry.private_state_384_cross_as_operands
            .base.blas_build_generation;
    persistent.stack =
        entry.private_state_384_cross_as_operands.base.stack;
    persistent.recovery_target_inflight =
        entry.input.recovery_target_inflight;
    tlas = entry.private_state_384_cross_as_operands
               .parent.tlas_decode_context;
  } else {
    if (short_stack_shared::decode_persistent_state(
            entry.read_slot, entry.input.owner, &persistent) !=
        short_stack_shared::kStatusOk) {
      return kStatusSharedPlanRejected;
    }
    tlas = entry.input.immutable_trace_input.decode_context;
  }
  if (!short_stack_shared::validate_persistent_state(persistent) ||
      persistent.stack.cross_as == 0 ||
      persistent.stack.stack_count != 1 ||
      !short_stack::read_logical_entry(
          persistent.stack, 0, &return_entry) ||
      short_stack::control_kind(return_entry.control) !=
          short_stack::kEntryCrossAsReturn ||
      return_entry.payload_byte_count !=
          fetch_target::kInstanceRawPayloadBytes ||
      tlas.as_object.as_type != 1 ||
      tlas.device_range_bytes <
          fetch_target::kInstanceRawPayloadBytes ||
      return_entry.payload_offset >
          tlas.device_range_bytes -
              fetch_target::kInstanceRawPayloadBytes) {
    return kStatusSharedPlanRejected;
  }
  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry.input.owner, &request_binding)) {
    return kStatusOwnerMismatch;
  }
  const uint64_t instance_base =
      tlas.device_base + return_entry.payload_offset;
  entry.phase = kPhaseReturnInstanceReading;
  for (unsigned index = 0; index < kReturnInstanceReadChunkCount;
       ++index) {
    rtcore_memory_unit_request_snapshot &request =
        requests->requests[index];
    request.valid = true;
    request.address_space = RTCORE_MEMORY_ADDRESS_SPACE_GLOBAL;
    request.operation = RTCORE_MEMORY_OPERATION_READ;
    request.destination =
        RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL;
    request.response_target = kResponseTargetRtcore;
    request.owner_hw_sid = entry.input.owner.owner_hw_sid;
    request.rt_request_id = entry.input.owner.request_identity;
    request.lane_id = entry.input.owner.lane_id;
    request.resident_warp_id =
        entry.input.owner.resident_warp_id;
    request.request_generation = entry.input.owner.generation;
    request.private_slot_id =
        entry.input.owner.private_slot_id;
    request.memory_op_seq = kReadChunkCount + index + 1;
    if (entry.input.private_storage_profile ==
            private_storage::kProfileCompressedShared384 ||
        entry.input.private_storage_profile ==
            private_storage::kProfileGlobal384) {
      request.memory_op_seq =
          entry.reservation.read_chunk_count + index + 1;
    }
    request.chunk_id = index;
    request.chunk_count = kReturnInstanceReadChunkCount;
    request.access_kind =
        RTCORE_MEMORY_ACCESS_SHORT_STACK_RETURN_INSTANCE_READ;
    request.aligned_32b_addr =
        instance_base +
        index * private_frontier::kSharedAccessChunkBytes;
    request.byte_mask = UINT32_MAX;
    request.is_write = false;
    request.issue_cycle = issue_cycle;
    rtcore_v04_stack_private_read_transport_snapshot &transport =
        request.v04_stack_private_read;
    transport.reservation_id =
        entry.reservation.reservation_id;
    transport.reservation_age =
        entry.reservation.reservation_age;
    transport.target_operation_seq =
        entry.reservation.operation_seq;
    transport.producer_operation_seq =
        entry.reservation.producer_operation_seq;
    transport.target_slot_generation =
        entry.reservation.slot_generation;
    transport.slot_chunk_offset = static_cast<uint16_t>(
        index * private_frontier::kSharedAccessChunkBytes);
    transport.target_slot_index =
        entry.reservation.slot_index;
    transport.field_kind = private_frontier::kFieldInvalid;
    transport.operation_kind = entry.input.operation_kind;
    transport.read_phase = kReadPhaseReturnInstance;
    transport.valid = 1;
    if (!request_matches_entry(request, entry) ||
        timing_driver::begin_memory_transaction(
            &staged_timing, request_binding,
            entry.reservation.operation_seq) !=
            timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
  }
  requests->request_count = kReturnInstanceReadChunkCount;
  requests->valid = 1;
  *state = staged_state;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind service_cycle(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    const parent_resolver_v0 &parent_resolver, uint64_t service_cycle,
    cycle_result_v0 *result) {
  if (state == NULL || timing_state == NULL ||
      private_backing == NULL || result == NULL ||
      state->initialized != 1 || parent_resolver.resolve == NULL) {
    return kStatusInvalidArgument;
  }
  *result = cycle_result_v0();

  const int lookup_index = find_oldest_phase(
      *state, kPhaseParentLookup, service_cycle, true);
  if (lookup_index >= 0) {
    operation_entry_v0 &entry = state->slots[lookup_index];
    short_stack::parent_edge_v0 parent = {};
    if (!parent_resolver.resolve(
            parent_resolver.context,
            entry.transition.parent_lookup_decode_context,
            entry.transition.parent_lookup_build_generation,
            entry.transition.parent_lookup_payload_offset,
            entry.transition.parent_lookup_payload_kind,
            &parent)) {
      return kStatusParentResolveRejected;
    }
    entry.parent_edge = parent;
    entry.parent_edge_valid = 1;
    entry.phase = kPhaseExecuting;
    entry.result_ready_cycle = service_cycle;
    ++result->parent_lookup_completed;
  }

  uint8_t issued_unit_mask = 0;
  for (unsigned issued = 0; issued < state->config.issue_width; ++issued) {
    const int ready_index = find_oldest_phase(
        *state, kPhaseReadyToIssue, service_cycle, false);
    if (ready_index < 0) break;
    int unit_index = -1;
    for (unsigned unit = 0; unit < state->config.unit_count; ++unit) {
      if ((issued_unit_mask & (1u << unit)) == 0 &&
          state->units[unit].next_issue_cycle <= service_cycle) {
        unit_index = static_cast<int>(unit);
        break;
      }
    }
    if (unit_index < 0) {
      const operation_entry_v0 &entry = state->slots[ready_index];
      emit_attempt(
          entry.input.owner, entry.reservation.operation_seq,
          service_cycle, static_cast<uint16_t>(issued),
          stall_attribution::kStageIssue,
          stall_attribution::kOutcomeStall,
          stall_attribution::kActionNone,
          stall_attribution::kReasonUnitBusy);
      break;
    }
    operation_entry_v0 &entry = state->slots[ready_index];
    entry.phase = kPhaseExecuting;
    entry.issue_cycle = service_cycle;
    entry.result_ready_cycle =
        service_cycle + state->config.stack_latency;
    state->units[unit_index].next_issue_cycle =
        service_cycle + state->config.initiation_interval;
    issued_unit_mask = static_cast<uint8_t>(
        issued_unit_mask | (1u << unit_index));
    ++result->issued;
    emit_attempt(
        entry.input.owner, entry.reservation.operation_seq,
        service_cycle, static_cast<uint16_t>(issued),
        stall_attribution::kStageIssue,
        stall_attribution::kOutcomeProgress,
        stall_attribution::kActionIssue,
        stall_attribution::kReasonNone);
  }

  const int execute_index = find_oldest_phase(
      *state, kPhaseExecuting, service_cycle, true);
  if (execute_index >= 0) {
    operation_entry_v0 &entry = state->slots[execute_index];
    short_stack_transition::result_v0 transition = {};
    short_stack_transition::status_kind transition_status =
        short_stack_transition::kStatusInvalidArgument;
    if (entry.input.operation_kind == kOperationNodeTransition) {
      if (entry.input.private_storage_profile ==
              private_storage::kProfileCompressedShared384 ||
          entry.input.private_storage_profile ==
              private_storage::kProfileGlobal384) {
        if (entry.private_state_384_operands_valid != 1) {
          return kStatusSharedPlanRejected;
        }
        short_stack_transition::node_operands_input_v1
            transition_input = {};
        transition_input.owner = entry.input.owner;
        transition_input.region = entry.input.private_region;
        transition_input.persistent_state.tlas_build_generation =
            entry.private_state_384_stack_operands
                .tlas_build_generation;
        transition_input.persistent_state.blas_build_generation =
            entry.private_state_384_stack_operands
                .blas_build_generation;
        transition_input.persistent_state.stack =
            entry.private_state_384_stack_operands.stack;
        transition_input.persistent_state
            .recovery_target_inflight =
            entry.input.recovery_target_inflight;
        transition_input.node_route = entry.input.node_route;
        transition_input.current_target =
            entry.input.current_target;
        transition_input.current_decode_context =
            entry.private_state_384_stack_operands
                .active_decode_context;
        transition_input.pending_parent_resume =
            entry.input.pending_parent_resume;
        transition_input.pending_parent_resume_valid =
            entry.input.pending_parent_resume_valid;
        transition_input.parent_edge = entry.parent_edge;
        transition_input.parent_edge_valid =
            entry.parent_edge_valid;
        transition_status =
            short_stack_transition::
                prepare_node_transition_from_operands(
                    transition_input, &transition);
      } else {
        short_stack_transition::node_input_v0 transition_input = {};
        transition_input.owner = entry.input.owner;
        transition_input.region = entry.input.private_region;
        transition_input.canonical_slot = entry.read_slot;
        transition_input.node_route = entry.input.node_route;
        transition_input.current_target = entry.input.current_target;
        transition_input.current_decode_context =
            entry.input.current_decode_context;
        transition_input.pending_parent_resume =
            entry.input.pending_parent_resume;
        transition_input.pending_parent_resume_valid =
            entry.input.pending_parent_resume_valid;
        transition_input.parent_edge = entry.parent_edge;
        transition_input.parent_edge_valid = entry.parent_edge_valid;
        transition_status =
            short_stack_transition::prepare_node_transition(
                transition_input, &transition);
      }
    } else if (entry.input.operation_kind ==
               kOperationResumeTransition) {
      if (entry.input.private_storage_profile ==
              private_storage::kProfileCompressedShared384 ||
          entry.input.private_storage_profile ==
              private_storage::kProfileGlobal384) {
        if (entry.private_state_384_operands_valid != 1) {
          return kStatusSharedPlanRejected;
        }
        short_stack_transition::resume_operands_input_v1
            transition_input = {};
        transition_input.owner = entry.input.owner;
        transition_input.region = entry.input.private_region;
        transition_input.persistent_state.tlas_build_generation =
            entry.private_state_384_stack_operands
                .tlas_build_generation;
        transition_input.persistent_state.blas_build_generation =
            entry.private_state_384_stack_operands
                .blas_build_generation;
        transition_input.persistent_state.stack =
            entry.private_state_384_stack_operands.stack;
        transition_input.persistent_state
            .recovery_target_inflight =
            entry.input.recovery_target_inflight;
        transition_input.recovery_target_completed =
            entry.input.recovery_target_completed;
        transition_input.active_decode_context =
            entry.private_state_384_stack_operands
                .active_decode_context;
        if (entry.private_state_384_cross_as_operands_valid != 0) {
          if (entry.return_instance_projection_valid != 1) {
            return kStatusSharedPlanRejected;
          }
          transition_input.parent_ray =
              entry.private_state_384_cross_as_operands
                  .parent.ray;
          transition_input.parent_decode_context =
              entry.private_state_384_cross_as_operands
                  .parent.tlas_decode_context;
          transition_input.parent_instance =
              entry.return_instance_projection;
          transition_input.parent_restore_valid = 1;
        }
        transition_input.parent_edge = entry.parent_edge;
        transition_input.parent_edge_valid =
            entry.parent_edge_valid;
        transition_status =
            short_stack_transition::
                prepare_resume_transition_from_operands(
                    transition_input, &transition);
      } else {
        short_stack_transition::resume_input_v0 transition_input = {};
        transition_input.owner = entry.input.owner;
        transition_input.region = entry.input.private_region;
        transition_input.canonical_slot = entry.read_slot;
        transition_input.immutable_trace_input =
            entry.input.immutable_trace_input;
        transition_input.active_decode_context =
            entry.input.active_decode_context;
        transition_input.parent_edge = entry.parent_edge;
        transition_input.parent_edge_valid =
            entry.parent_edge_valid;
        transition_status =
            short_stack_transition::prepare_resume_transition(
                transition_input, &transition);
      }
    } else if (entry.input.operation_kind ==
               kOperationEnterBlasTransition) {
      if (entry.input.private_storage_profile ==
              private_storage::kProfileCompressedShared384 ||
          entry.input.private_storage_profile ==
              private_storage::kProfileGlobal384) {
        if (entry.private_state_384_operands_valid != 1 ||
            entry.input.deferred_instance_valid != 1) {
          return kStatusSharedPlanRejected;
        }
        short_stack_transition::enter_blas_operands_input_v1
            transition_input = {};
        transition_input.owner = entry.input.owner;
        transition_input.region = entry.input.private_region;
        transition_input.persistent_state.tlas_build_generation =
            entry.private_state_384_stack_operands
                .tlas_build_generation;
        transition_input.persistent_state.blas_build_generation =
            entry.private_state_384_stack_operands
                .blas_build_generation;
        transition_input.persistent_state.stack =
            entry.private_state_384_stack_operands.stack;
        transition_input.persistent_state
            .recovery_target_inflight =
            entry.input.recovery_target_inflight;
        transition_input.tlas_instance_target =
            entry.input.tlas_instance_target;
        transition_input.blas_root = entry.input.blas_root;
        transition_input.blas_build_generation =
            entry.input.blas_build_generation;
        transition_status =
            short_stack_transition::
                prepare_enter_blas_transition_from_operands(
                    transition_input, &transition);
      } else {
        short_stack_transition::enter_blas_input_v0 transition_input = {};
        transition_input.owner = entry.input.owner;
        transition_input.region = entry.input.private_region;
        transition_input.canonical_slot = entry.read_slot;
        transition_input.tlas_instance_target =
            entry.input.tlas_instance_target;
        transition_input.blas_root = entry.input.blas_root;
        transition_input.blas_build_generation =
            entry.input.blas_build_generation;
        transition_status =
            short_stack_transition::prepare_enter_blas_transition(
                transition_input, &transition);
      }
    }
    if (transition_status ==
        short_stack_transition::kStatusParentLookupRequired) {
      entry.transition = transition;
      entry.phase = kPhaseParentLookup;
      entry.parent_lookup_ready_cycle =
          service_cycle + state->config.parent_lookup_latency;
    } else if (
        transition_status ==
        short_stack_transition::kStatusReturnInstanceRequired) {
      entry.transition = transition;
      entry.input.operation_kind = kOperationResumeTransition;
      if (entry.input.private_storage_profile ==
              private_storage::kProfileCompressedShared384 ||
          entry.input.private_storage_profile ==
              private_storage::kProfileGlobal384) {
        if (entry.private_state_384_operands_valid != 1 ||
            entry.private_state_384_cross_as_operands_valid != 0) {
          return kStatusSharedPlanRejected;
        }
        // A recovery-backed node miss can normalize lost/recovery state before
        // discovering that only the cross-AS return entry remains. Keep that
        // state transaction-local until the old Global384 response has been
        // materialized, then apply it before resume. The final resume commit
        // remains the sole writeback to the 384-byte backing image.
        entry.return_transition_state_carry_valid = 1;
        entry.private_state_384_selected_operation =
            private_state_384::operand_plan::
                kOperationStackCrossAsReturn;
        entry.phase = kPhasePrivate384FollowupPlanReady;
      } else {
        entry.phase = kPhaseReturnInstancePlanReady;
      }
      entry.result_ready_cycle = service_cycle;
      ++result->return_instance_requested;
    } else if (transition_status !=
               short_stack_transition::kStatusOk) {
      result->rejected_operation_kind =
          entry.input.operation_kind;
      result->rejected_transition_status =
          static_cast<uint8_t>(transition_status);
      return kStatusTransitionRejected;
    } else {
      entry.transition = transition;
      const status_kind commit_status =
          begin_transition_commit(&entry, timing_state);
      if (commit_status != kStatusOk) return commit_status;
      ++result->transition_committed;
      emit_attempt(
          entry.input.owner, entry.reservation.operation_seq,
          service_cycle, 0,
          stall_attribution::kStageCapture,
          stall_attribution::kOutcomeProgress,
          stall_attribution::kActionCapture,
          stall_attribution::kReasonNone);
    }
  }

  result->active_operations = active_operation_count(*state);
  for (unsigned index = 0; index < state->config.capacity; ++index) {
    result->ready_results +=
        state->slots[index].valid != 0 &&
        state->slots[index].phase == kPhaseResultReady;
  }
  return kStatusOk;
}

status_kind service_write_enqueue(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle, uint8_t write_enqueue_budget,
    uint8_t *writes_enqueued, bool *shared_queue_blocked) {
  if (state == NULL || timing_state == NULL ||
      private_backing == NULL || writes_enqueued == NULL ||
      shared_queue_blocked == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *writes_enqueued = 0;
  *shared_queue_blocked = false;
  while (*writes_enqueued < write_enqueue_budget) {
    const int write_index = find_oldest_pending_write(*state, false);
    if (write_index < 0) break;
    operation_entry_v0 &entry = state->slots[write_index];
    const uint8_t chunk_index = first_missing_write_chunk(entry);
    const status_kind write_status = enqueue_transition_write(
        &entry, chunk_index, timing_state, private_backing,
        service_cycle);
    if (write_status == kStatusSharedQueueBackpressure) {
      *shared_queue_blocked = true;
      break;
    }
    if (write_status != kStatusOk) return write_status;
    ++*writes_enqueued;
  }
  return kStatusOk;
}

status_kind transfer_next_global_write(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    uint64_t service_cycle,
    private_shared::shared_write_v0 *write) {
  if (state == NULL || timing_state == NULL || write == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *write = private_shared::shared_write_v0();
  const int slot_index = find_oldest_pending_write(*state, true);
  if (slot_index < 0) return kStatusNoWriteOffer;

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  const uint8_t write_index = first_missing_write_chunk(entry);
  private_shared::shared_write_v0 prepared = {};
  request_owner::lane_binding_v0 request_binding = {};
  if (write_index >= modeled_write_count(entry) ||
      !make_request_owner(entry.input.owner, &request_binding) ||
      private_state_384::live_bridge::prepare_global_modeled_write(
          entry.private_state_384_commit, write_index, service_cycle,
          &prepared) != private_state_384::live_bridge::kStatusOk ||
      private_state_384::live_bridge::register_modeled_write(
          prepared, &entry.private_state_384_commit) !=
          private_state_384::live_bridge::kStatusOk ||
      timing_driver::begin_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk) {
    return kStatusSharedWriteRejected;
  }
  entry.enqueued_write_mask = static_cast<uint16_t>(
      entry.enqueued_write_mask | (uint16_t{1} << write_index));
  *state = staged_state;
  *timing_state = staged_timing;
  *write = prepared;
  return kStatusOk;
}

bool owns_ack(const engine_state_v0 &state,
              const private_shared::runtime_write_ack_v0 &ack) {
  if (state.initialized != 1 || !ack.valid ||
      ack.memory_operation_seq == 0 ||
      ack.memory_operation_seq > kMaxWriteChunkCount) {
    return false;
  }
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    const operation_entry_v0 &entry = state.slots[index];
    if (entry.valid != 0 && entry.phase == kPhaseWriting &&
        entry.reservation.operation_seq == ack.operation_seq &&
        entry.commit_epoch == ack.commit_epoch &&
        private_frontier::owners_equal(entry.input.owner, ack.owner) &&
        (entry.enqueued_write_mask &
         (1u << (ack.memory_operation_seq - 1))) != 0 &&
        ack.memory_operation_seq <= modeled_write_count(entry) &&
        (entry.acknowledged_write_mask &
         (1u << (ack.memory_operation_seq - 1))) == 0) {
      return true;
    }
  }
  return false;
}

status_kind accept_write_ack(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle,
    const private_shared::runtime_write_ack_v0 &ack) {
  if (state == NULL || timing_state == NULL ||
      private_backing == NULL || !owns_ack(*state, ack)) {
    return kStatusNoAckOwned;
  }
  int slot_index = -1;
  for (unsigned index = 0; index < state->config.capacity; ++index) {
    const operation_entry_v0 &entry = state->slots[index];
    if (entry.valid != 0 && entry.phase == kPhaseWriting &&
        entry.reservation.operation_seq == ack.operation_seq &&
        entry.commit_epoch == ack.commit_epoch &&
        private_frontier::owners_equal(entry.input.owner, ack.owner)) {
      slot_index = static_cast<int>(index);
      break;
    }
  }
  if (slot_index < 0) return kStatusNoAckOwned;

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  private_shared::backing_state_v0 staged_backing = *private_backing;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry.input.owner, &request_binding) ||
      timing_driver::complete_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk ||
      private_shared::detail::commit_runtime_write_ack(
          &staged_backing, service_cycle, ack) !=
          private_shared::kStatusOk) {
    return kStatusAckRejected;
  }
  entry.acknowledged_write_mask = static_cast<uint16_t>(
      entry.acknowledged_write_mask |
      (1u << (ack.memory_operation_seq - 1)));
  if (entry.acknowledged_write_mask ==
      all_write_chunks(entry.transition.write_plan.access_count)) {
    entry.phase = kPhaseResultReady;
  }
  *state = staged_state;
  *timing_state = staged_timing;
  *private_backing = staged_backing;
  return kStatusOk;
}

status_kind accept_write_ack_with_private_state_384(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    private_state_384::backing::state_v1 *private_state_384_backing,
    uint64_t service_cycle,
    const private_shared::runtime_write_ack_v0 &ack) {
  if (state == NULL || timing_state == NULL ||
      private_backing == NULL ||
      private_state_384_backing == NULL ||
      !owns_ack(*state, ack)) {
    return kStatusNoAckOwned;
  }
  int slot_index = -1;
  for (unsigned index = 0; index < state->config.capacity; ++index) {
    const operation_entry_v0 &entry = state->slots[index];
    if (entry.valid != 0 && entry.phase == kPhaseWriting &&
        entry.input.private_storage_profile ==
            private_storage::kProfileCompressedShared384 &&
        entry.reservation.operation_seq == ack.operation_seq &&
        entry.commit_epoch == ack.commit_epoch &&
        private_frontier::owners_equal(
            entry.input.owner, ack.owner)) {
      if (slot_index >= 0) return kStatusNoAckOwned;
      slot_index = static_cast<int>(index);
    }
  }
  if (slot_index < 0 || private_backing->outstanding.empty()) {
    return kStatusNoAckOwned;
  }

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  private_shared::backing_state_v0 staged_backing =
      *private_backing;
  private_state_384::backing::state_v1 staged_private_384 =
      *private_state_384_backing;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  request_owner::lane_binding_v0 request_binding = {};
  const private_shared::shared_write_v0 modeled_write =
      staged_backing.outstanding.front();
  bool canonical_committed = false;
  if (!make_request_owner(entry.input.owner, &request_binding) ||
      timing_driver::complete_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk ||
      private_shared::detail::commit_runtime_write_ack(
          &staged_backing, service_cycle, ack) !=
          private_shared::kStatusOk ||
      private_state_384::live_bridge::
          accept_write_ack_and_maybe_commit(
              &staged_private_384, modeled_write, ack,
              &entry.private_state_384_commit,
              &canonical_committed) !=
          private_state_384::live_bridge::kStatusOk) {
    return kStatusAckRejected;
  }
  entry.acknowledged_write_mask = static_cast<uint16_t>(
      entry.acknowledged_write_mask |
      (1u << (ack.memory_operation_seq - 1)));
  if (entry.acknowledged_write_mask ==
      all_write_chunks(entry.transition.write_plan.access_count)) {
    if (!canonical_committed ||
        timing_driver::commit_private_recovery_target_state(
            &staged_timing, request_binding,
            entry.reservation.operation_seq, entry.commit_epoch,
            entry.transition.persistent_state
                .recovery_target_inflight) !=
            timing_driver::kStatusOk) {
      return kStatusAckRejected;
    }
    if (entry.transition.terminal != 0) {
      if (entry.transition.selected_valid != 0 ||
          entry.private_state_384_operands_valid != 1) {
        return kStatusAckRejected;
      }
      entry.private_state_384_completion_reason =
          entry.private_state_384_stack_operands
                      .committed_valid != 0
              ? private_state_384::operand_plan::
                    kCompletionReasonClosestHitReady
              : private_state_384::operand_plan::
                    kCompletionReasonMiss;
      entry.phase = kPhasePrivate384CompletionPlanReady;
    } else {
      entry.phase = kPhaseResultReady;
    }
  } else if (canonical_committed) {
    return kStatusAckRejected;
  }
  *state = staged_state;
  *timing_state = staged_timing;
  *private_backing = staged_backing;
  *private_state_384_backing = staged_private_384;
  return kStatusOk;
}

status_kind accept_global_write_ack(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::shared_write_v0 &write,
    const private_shared::runtime_write_ack_v0 &ack) {
  if (state == NULL || timing_state == NULL ||
      !owns_ack(*state, ack)) {
    return kStatusNoAckOwned;
  }
  int slot_index = -1;
  for (unsigned index = 0; index < state->config.capacity; ++index) {
    const operation_entry_v0 &entry = state->slots[index];
    if (entry.valid != 0 && entry.phase == kPhaseWriting &&
        entry.input.private_storage_profile ==
            private_storage::kProfileGlobal384 &&
        entry.reservation.operation_seq == ack.operation_seq &&
        entry.commit_epoch == ack.commit_epoch &&
        private_frontier::owners_equal(
            entry.input.owner, ack.owner)) {
      if (slot_index >= 0) return kStatusNoAckOwned;
      slot_index = static_cast<int>(index);
    }
  }
  if (slot_index < 0) return kStatusNoAckOwned;

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  request_owner::lane_binding_v0 request_binding = {};
  bool all_acknowledged = false;
  if (!make_request_owner(entry.input.owner, &request_binding) ||
      timing_driver::complete_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk ||
      private_state_384::live_bridge::accept_global_write_ack(
          write, ack, &entry.private_state_384_commit,
          &all_acknowledged) !=
          private_state_384::live_bridge::kStatusOk) {
    return kStatusAckRejected;
  }
  entry.acknowledged_write_mask = static_cast<uint16_t>(
      entry.acknowledged_write_mask |
      (uint16_t{1} << (ack.memory_operation_seq - 1)));
  const uint16_t expected_mask =
      all_write_chunks(modeled_write_count(entry));
  if (entry.acknowledged_write_mask == expected_mask) {
    if (!all_acknowledged || expected_mask == 0 ||
        timing_driver::commit_private_recovery_target_state(
            &staged_timing, request_binding,
            entry.reservation.operation_seq, entry.commit_epoch,
            entry.transition.persistent_state
                .recovery_target_inflight) !=
            timing_driver::kStatusOk) {
      return kStatusAckRejected;
    }
    if (entry.transition.terminal != 0) {
      if (entry.transition.selected_valid != 0 ||
          entry.private_state_384_operands_valid != 1) {
        return kStatusAckRejected;
      }
      entry.private_state_384_completion_reason =
          entry.private_state_384_stack_operands.committed_valid != 0
              ? private_state_384::operand_plan::
                    kCompletionReasonClosestHitReady
              : private_state_384::operand_plan::
                    kCompletionReasonMiss;
      entry.phase = kPhasePrivate384CompletionPlanReady;
    } else {
      if (entry.transition.selected_valid != 1) {
        return kStatusAckRejected;
      }
      entry.phase = kPhaseResultReady;
    }
  } else if (all_acknowledged || expected_mask == 0) {
    return kStatusAckRejected;
  }
  *state = staged_state;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind peek_ready_result(const engine_state_v0 &state,
                              ready_result_v0 *result) {
  if (state.initialized != 1 || result == NULL) {
    return kStatusInvalidArgument;
  }
  *result = ready_result_v0();
  const int index = find_oldest_phase(
      state, kPhaseResultReady, 0, false);
  if (index < 0) return kStatusNoReadyResult;
  const operation_entry_v0 &entry = state.slots[index];
  result->owner = entry.input.owner;
  result->ray_policy = entry.input.ray_policy;
  result->transition = entry.transition;
  result->reservation_id = entry.reservation.reservation_id;
  result->operation_seq = entry.reservation.operation_seq;
  result->producer_operation_seq =
      entry.reservation.producer_operation_seq;
  result->commit_epoch = entry.commit_epoch;
  result->slot_index = entry.reservation.slot_index;
  result->slot_generation =
      entry.reservation.slot_generation;
  result->private_storage_profile =
      entry.input.private_storage_profile;
  if ((entry.input.private_storage_profile ==
           private_storage::kProfileCompressedShared384 ||
       entry.input.private_storage_profile ==
           private_storage::kProfileGlobal384) &&
      entry.transition.terminal != 0 &&
      entry.private_state_384_completion_operands_valid != 0) {
    result->terminal_committed_hit =
        entry.private_state_384_completion_operands.committed_hit;
    result->terminal_committed_hit_valid = 1;
  }
  result->valid = 1;
  return kStatusOk;
}

status_kind consume_ready_result(engine_state_v0 *state,
                                 const ready_result_v0 &result) {
  if (state == NULL || state->initialized != 1 || !result.valid ||
      result.slot_index >= state->config.capacity) {
    return kStatusInvalidArgument;
  }
  operation_entry_v0 &entry = state->slots[result.slot_index];
  if (entry.valid == 0 || entry.phase != kPhaseResultReady ||
      entry.reservation.reservation_id != result.reservation_id ||
      entry.reservation.operation_seq != result.operation_seq ||
      entry.commit_epoch != result.commit_epoch ||
      !private_frontier::owners_equal(entry.input.owner,
                                      result.owner)) {
    return kStatusNoReadyResult;
  }
  const uint32_t generation = entry.reservation.slot_generation;
  entry = operation_entry_v0();
  entry.reservation.slot_generation = generation;
  return kStatusOk;
}

bool find_live_reservation(
    const engine_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq, reservation_receipt_v0 *reservation,
    uint8_t *operation_kind) {
  if (!state.initialized || operation_seq == 0 ||
      reservation == NULL || operation_kind == NULL) {
    return false;
  }
  *reservation = reservation_receipt_v0();
  *operation_kind = kOperationInvalid;
  bool found = false;
  for (uint8_t index = 0; index < state.config.capacity; ++index) {
    const operation_entry_v0 &entry = state.slots[index];
    if (!entry.valid || !entry.reservation.valid ||
        entry.reservation.operation_seq != operation_seq ||
        !private_frontier::owners_equal(entry.input.owner, owner)) {
      continue;
    }
    if (found || entry.input.operation_kind == kOperationInvalid ||
        entry.reservation.reservation_id == 0) {
      return false;
    }
    *reservation = entry.reservation;
    *operation_kind = entry.input.operation_kind;
    found = true;
  }
  return found;
}

uint8_t active_operation_count(const engine_state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return 0;
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    count += state.slots[index].valid != 0;
  }
  return count;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfig:
      return "invalid_config";
    case kStatusCapacityBackpressure:
      return "capacity_backpressure";
    case kStatusReservationBudgetBackpressure:
      return "reservation_budget_backpressure";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusTimingControlRejected:
      return "timing_control_rejected";
    case kStatusSharedPlanRejected:
      return "shared_plan_rejected";
    case kStatusMalformedTransport:
      return "malformed_transport";
    case kStatusDuplicateResponse:
      return "duplicate_response";
    case kStatusReturnInstanceRejected:
      return "return_instance_rejected";
    case kStatusTransitionRejected:
      return "transition_rejected";
    case kStatusParentResolveRejected:
      return "parent_resolve_rejected";
    case kStatusSharedQueueBackpressure:
      return "shared_queue_backpressure";
    case kStatusSharedWriteRejected:
      return "shared_write_rejected";
    case kStatusNoAckOwned:
      return "no_ack_owned";
    case kStatusAckRejected:
      return "ack_rejected";
    case kStatusNoFollowupRead:
      return "no_followup_read";
    case kStatusNoWriteOffer:
      return "no_write_offer";
    case kStatusNoReadyResult:
      return "no_ready_result";
  }
  return "unknown";
}

}  // namespace short_stack_timing
}  // namespace v04
}  // namespace rtcore
