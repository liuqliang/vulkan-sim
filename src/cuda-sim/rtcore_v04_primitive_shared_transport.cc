#include "rtcore_v04_primitive_shared_transport.h"

#include <cstring>
#include <limits>

#include "rtcore_v04_private_shared_backing_internal.h"

namespace rtcore {
namespace v04 {
namespace primitive_shared {
namespace {

static const uint8_t kMaxPrivateState384DeltaCount = 4;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool valid_config(const config_v0 &config) {
  return config.result_commit_capacity != 0 &&
         config.result_commit_capacity <= kMaxResultCommitEntries &&
         config.tracker_capacity != 0 &&
         config.tracker_capacity <= kMaxCommitTrackers &&
         config.boundary_capacity != 0 &&
         config.boundary_capacity <= kMaxBoundaryReceipts &&
         bytes_are_zero(config.reserved_zero,
                        sizeof(config.reserved_zero));
}

bool is_boundary_route(uint8_t route) {
  return route == primitive_semantic::kRouteAnyHitBoundary ||
         route == primitive_semantic::kRouteIntersectionBoundary ||
         route == primitive_semantic::kRouteFinalHitBoundary;
}

uint16_t expected_mask(uint16_t count) {
  return count == 0
             ? 0
             : static_cast<uint16_t>((uint32_t{1} << count) - 1);
}

int find_free_result(const engine_state_v0 &state) {
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    if (state.result_entries[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_free_tracker(const engine_state_v0 &state) {
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    if (state.trackers[index].valid == 0) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int find_tracker(const engine_state_v0 &state,
                 const private_frontier::owner_binding_v0 &owner,
                 uint32_t operation_seq, uint32_t commit_epoch) {
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    const commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid != 0 &&
        tracker.operation_seq == operation_seq &&
        tracker.commit_epoch == commit_epoch &&
        private_frontier::owners_equal(tracker.owner, owner)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

bool operation_live(const engine_state_v0 &state,
                    const private_frontier::owner_binding_v0 &owner,
                    uint32_t operation_seq) {
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    const commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid != 0 &&
        tracker.operation_seq == operation_seq &&
        private_frontier::owners_equal(tracker.owner, owner)) {
      return true;
    }
  }
  return false;
}

int find_oldest_result(const engine_state_v0 &state) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    const result_commit_entry_v0 &entry = state.result_entries[index];
    if (entry.valid != 0 &&
        entry.next_write_index < entry.write_count &&
        (selected < 0 || entry.issue_age < selected_age)) {
      selected = static_cast<int>(index);
      selected_age = entry.issue_age;
    }
  }
  return selected;
}

int find_oldest_global_write_tracker(const engine_state_v0 &state) {
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    const commit_tracker_v0 &tracker = state.trackers[index];
    if (tracker.valid == 0 ||
        tracker.private_storage_profile !=
            private_storage::kProfileGlobal384 ||
        tracker.expected_write_count == 0 ||
        tracker.accepted_write_mask ==
            expected_mask(tracker.expected_write_count)) {
      continue;
    }
    if (selected < 0 || tracker.issue_age < selected_age) {
      selected = static_cast<int>(index);
      selected_age = tracker.issue_age;
    }
  }
  return selected;
}

uint8_t first_missing_write(uint16_t mask, uint16_t count) {
  for (uint8_t index = 0; index < count; ++index) {
    if ((mask & (uint16_t{1} << index)) == 0) return index;
  }
  return static_cast<uint8_t>(count);
}

int find_result(const engine_state_v0 &state,
                const write_offer_v0 &offer) {
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    const result_commit_entry_v0 &entry = state.result_entries[index];
    if (entry.valid != 0 &&
        entry.operation_seq == offer.operation_seq &&
        entry.commit_epoch == offer.commit_epoch &&
        private_frontier::owners_equal(entry.owner, offer.owner)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

void build_offer(const result_commit_entry_v0 &entry,
                 write_offer_v0 *offer) {
  *offer = write_offer_v0();
  offer->owner = entry.owner;
  offer->operation_seq = entry.operation_seq;
  offer->commit_epoch = entry.commit_epoch;
  offer->memory_operation_seq =
      static_cast<uint16_t>(entry.next_write_index + 1);
  offer->write_count = entry.write_count;
  offer->fragment = entry.writes[entry.next_write_index];
}

status_kind prepare_shared_write(
    const write_offer_v0 &offer, uint64_t enqueue_cycle,
    private_shared::shared_write_v0 *operation) {
  if (operation == NULL || offer.operation_seq == 0 ||
      offer.commit_epoch == 0 || offer.memory_operation_seq == 0 ||
      offer.write_count == 0 ||
      offer.write_count > primitive_semantic::kMaxWriteFragmentCount ||
      offer.memory_operation_seq > offer.write_count ||
      !bytes_are_zero(offer.reserved_zero,
                      sizeof(offer.reserved_zero)) ||
      !primitive_semantic::validate_private_write_fragment(
          offer.fragment)) {
    return kStatusWriteOfferMismatch;
  }
  *operation = private_shared::shared_write_v0();
  operation->valid = true;
  operation->address_space = private_shared::kAddressSpaceShared;
  operation->address_mode = private_shared::kAddressModePrivateField;
  operation->access_operation = private_shared::kAccessOperationWrite;
  operation->destination = private_shared::kDestinationPrivateCommitAck;
  operation->owner = offer.owner;
  operation->operation_seq = offer.operation_seq;
  operation->commit_epoch = offer.commit_epoch;
  operation->memory_op_seq = offer.memory_operation_seq;
  operation->chunk_id =
      static_cast<uint8_t>(offer.memory_operation_seq - 1);
  operation->chunk_count = static_cast<uint8_t>(offer.write_count);
  operation->field_kind = offer.fragment.field_kind;
  operation->aligned_32b_address =
      offer.fragment.aligned_32b_address;
  operation->byte_mask = offer.fragment.byte_mask;
  std::memcpy(operation->payload, offer.fragment.payload,
              sizeof(operation->payload));
  operation->enqueue_cycle = enqueue_cycle;
  return kStatusOk;
}

void append_private_state_384_delta(
    uint8_t chunk_index, uint32_t byte_mask, const uint8_t *payload,
    private_state_384::live_bridge::sparse_chunk_delta_v1
        deltas[kMaxPrivateState384DeltaCount],
    uint8_t *delta_count) {
  private_state_384::live_bridge::sparse_chunk_delta_v1 &delta =
      deltas[(*delta_count)++];
  delta.chunk_index = chunk_index;
  delta.byte_mask = byte_mask;
  for (uint8_t byte = 0; byte < private_state_384::kChunkBytes; ++byte) {
    if ((byte_mask & (uint32_t{1} << byte)) != 0) {
      delta.payload[byte] = payload[byte];
    }
  }
}

bool make_private_state_384_as_context(
    const typed_blas::as_decode_context_v0 &typed,
    uint8_t cull_mask, private_state_384::as_context_v1 *context) {
  if (context == NULL ||
      typed.bvh_format_profile_id !=
          private_state_384::kGenRtBvhFormatProfileId ||
      typed.reserved_zero != 0 ||
      !bytes_are_zero(typed.as_object.reserved_zero,
                      sizeof(typed.as_object.reserved_zero))) {
    return false;
  }
  *context = private_state_384::as_context_v1();
  context->as_object_id = typed.as_object.object_id;
  context->device_base = typed.device_base;
  context->device_range_bytes = typed.device_range_bytes;
  context->as_object_generation = typed.as_object.generation;
  context->as_type = typed.as_object.as_type;
  context->cull_mask = cull_mask;
  return true;
}

bool prepare_private_state_384_deltas(
    const primitive_semantic::semantic_plan_v0 &semantic_plan,
    const typed_blas::as_decode_context_v0 &active_decode_context,
    private_state_384::live_bridge::sparse_chunk_delta_v1
        deltas[kMaxPrivateState384DeltaCount],
    uint8_t *delta_count) {
  if (deltas == NULL || delta_count == NULL) return false;
  std::memset(
      deltas, 0,
      sizeof(*deltas) * kMaxPrivateState384DeltaCount);
  *delta_count = 0;
  if (semantic_plan.valid != 1 || semantic_plan.operation_seq == 0 ||
      semantic_plan.reserved_zero != 0 ||
      semantic_plan.committed_hit_valid > 1 ||
      semantic_plan.retained_candidate_valid > 1 ||
      semantic_plan.primitive_resume_valid > 1 ||
      semantic_plan.intersection_boundary_valid > 1 ||
      semantic_plan.shader_return_valid > 1) {
    return false;
  }

  if (semantic_plan.committed_hit_valid != 0) {
    uint8_t payload[
        private_state_384::kCommittedHitProjectionBytes] = {};
    if (private_state_384::encode_committed_hit_sparse_projection(
            semantic_plan.committed_hit, payload) !=
        private_state_384::kStatusOk) {
      return false;
    }
    append_private_state_384_delta(
        2, 0xffffffffu, payload, deltas, delta_count);
    append_private_state_384_delta(
        3, 0x00ffffffu,
        payload + private_state_384::kChunkBytes, deltas,
        delta_count);
  }

  if (semantic_plan.retained_candidate_valid != 0) {
    private_state_384::boundary_state_v1 boundary = {};
    boundary.identity_and_policy =
        semantic_plan.retained_candidate.identity_and_policy;
    boundary.primitive_resume = semantic_plan.primitive_resume;
    uint8_t reason = private_state_384::kBoundaryReasonNone;
    if (semantic_plan.route_kind ==
            primitive_semantic::kRouteAnyHitBoundary &&
        semantic_plan.intersection_boundary_valid == 0) {
      reason = private_state_384::kBoundaryReasonAnyHit;
      boundary.triangle_hit =
          semantic_plan.retained_candidate.triangle_hit;
    } else if (
        semantic_plan.route_kind ==
            primitive_semantic::kRouteIntersectionBoundary &&
        semantic_plan.intersection_boundary_valid == 1) {
      reason =
          private_state_384::kBoundaryReasonProceduralIntersection;
      boundary.intersection = semantic_plan.intersection_boundary;
    } else {
      return false;
    }
    private_state_384::as_context_v1 active_as = {};
    uint8_t payload[private_state_384::kBoundaryProjectionBytes] = {};
    if (!make_private_state_384_as_context(
            active_decode_context, semantic_plan.ray_policy.cull_mask,
            &active_as) ||
        private_state_384::encode_boundary_sparse_projection(
            boundary, reason, active_as, payload) !=
            private_state_384::kStatusOk) {
      return false;
    }
    append_private_state_384_delta(
        8, 0xffffffffu, payload, deltas, delta_count);
    append_private_state_384_delta(
        9, 0xffffffffu,
        payload + private_state_384::kChunkBytes, deltas,
        delta_count);
  } else if (semantic_plan.shader_return_valid != 0) {
    const uint8_t cleared[private_state_384::kBoundaryProjectionBytes] = {};
    append_private_state_384_delta(
        8, 0xffffffffu, cleared, deltas, delta_count);
    append_private_state_384_delta(
        9, 0xffffffffu,
        cleared + private_state_384::kChunkBytes, deltas,
        delta_count);
  }

  return *delta_count <= kMaxPrivateState384DeltaCount &&
         !(semantic_plan.retained_candidate_valid != 0 &&
           semantic_plan.shader_return_valid != 0) &&
         !(semantic_plan.primitive_resume_valid != 0 &&
           semantic_plan.retained_candidate_valid == 0);
}

}  // namespace

status_kind initialize(engine_state_v0 *state, const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!valid_config(config)) return kStatusInvalidConfiguration;
  *state = engine_state_v0();
  state->config = config;
  state->next_issue_age = 1;
  state->initialized = 1;
  return kStatusOk;
}

static status_kind capture_semantic_plan_impl(
    engine_state_v0 *state,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const primitive_semantic::semantic_plan_v0 &semantic_plan,
    uint8_t private_storage_profile,
    const typed_blas::as_decode_context_v0 *explicit_active_decode_context,
    uint8_t private_state_384_producer,
    capture_receipt_v0 *receipt) {
  const bool internal_successor =
      semantic_plan.route_kind ==
      primitive_semantic::kRouteStackPopNext;
  const bool boundary = is_boundary_route(semantic_plan.route_kind);
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      producer_operation_seq == 0 || commit_epoch == 0 ||
      (internal_successor &&
       (target_operation_seq == 0 ||
        target_operation_seq == producer_operation_seq)) ||
      (boundary && target_operation_seq != 0) ||
      (!internal_successor && !boundary) ||
      semantic_plan.operation_seq != producer_operation_seq ||
      !private_frontier::owners_equal(
          semantic_plan.owner, canonical_slot.owner) ||
      (private_storage_profile !=
           private_storage::kProfileLegacyShared832 &&
       private_storage_profile !=
           private_storage::kProfileCompressedShared384) ||
      (private_storage_profile ==
           private_storage::kProfileCompressedShared384 &&
       (explicit_active_decode_context == NULL ||
        (private_state_384_producer !=
             private_state_384::operand_plan::kProducerPrimitive &&
         private_state_384_producer !=
             private_state_384::operand_plan::
                 kProducerResubmitApply)))) {
    return kStatusInvalidArgument;
  }
  *receipt = capture_receipt_v0();
  if (operation_live(
          *state, semantic_plan.owner, producer_operation_seq)) {
    return kStatusDuplicateOperation;
  }
  const int tracker_index = find_free_tracker(*state);
  if (tracker_index < 0) return kStatusTrackerBackpressure;
  if (state->next_issue_age == 0 ||
      state->next_issue_age ==
          std::numeric_limits<uint64_t>::max()) {
    return kStatusCounterExhausted;
  }

  primitive_semantic::private_commit_plan_v0 commit = {};
  private_frontier::root_private_operands_v0 active_operands = {};
  const bool allow_terminal_shader_return =
      private_storage_profile ==
          private_storage::kProfileCompressedShared384 &&
      private_state_384_producer ==
          private_state_384::operand_plan::kProducerResubmitApply;
  if (primitive_semantic::prepare_private_commit(
          semantic_plan, region, canonical_slot, &commit,
          allow_terminal_shader_return) !=
          primitive_semantic::kStatusOk ||
      (explicit_active_decode_context == NULL &&
       private_frontier::decode_root_private_operands(
           canonical_slot, semantic_plan.owner,
           &active_operands) != private_frontier::kStatusOk) ||
      commit.valid != 1 ||
      commit.semantic_plan.route_kind == primitive_semantic::kRouteInvalid) {
    return kStatusSemanticPlanRejected;
  }
  if (explicit_active_decode_context != NULL) {
    active_operands.decode_context = *explicit_active_decode_context;
  }

  private_state_384::live_bridge::pending_sparse_commit_v1
      private_state_384_commit = {};
  if (private_storage_profile ==
      private_storage::kProfileCompressedShared384) {
    private_state_384::live_bridge::sparse_chunk_delta_v1
        deltas[kMaxPrivateState384DeltaCount] = {};
    uint8_t delta_count = 0;
    if (!prepare_private_state_384_deltas(
            semantic_plan, active_operands.decode_context, deltas,
            &delta_count) ||
        ((commit.write_fragment_count == 0) != (delta_count == 0))) {
      return kStatusPrivateState384Rejected;
    }
    if (delta_count != 0) {
      private_state_384::live_bridge::write_commit_input_v1 input = {};
      input.owner = semantic_plan.owner;
      input.operation_sequence = producer_operation_seq;
      input.commit_epoch = commit_epoch;
      input.bvh_format_profile_id =
          private_state_384::kGenRtBvhFormatProfileId;
      input.expected_write_ack_count =
          commit.write_fragment_count;
      input.storage_profile =
          private_storage::kProfileCompressedShared384;
      input.producer = private_state_384_producer;
      if (private_state_384::live_bridge::stage_sparse_commit(
              input, deltas, delta_count,
              &private_state_384_commit) !=
          private_state_384::live_bridge::kStatusOk) {
        return kStatusPrivateState384Rejected;
      }
    }
  }

  const int result_index =
      commit.write_fragment_count == 0 ? -1 : find_free_result(*state);
  if (commit.write_fragment_count != 0 && result_index < 0) {
    return kStatusResultBackpressure;
  }
  if (result_index >= 0) {
    result_commit_entry_v0 entry = {};
    entry.owner = semantic_plan.owner;
    entry.issue_age = state->next_issue_age;
    entry.operation_seq = producer_operation_seq;
    entry.target_operation_seq = target_operation_seq;
    entry.commit_epoch = commit_epoch;
    entry.write_count = commit.write_fragment_count;
    entry.tracker_slot = static_cast<uint8_t>(tracker_index);
    entry.valid = 1;
    for (unsigned index = 0;
         index < commit.write_fragment_count; ++index) {
      entry.writes[index] = commit.write_fragments[index];
    }
    state->result_entries[result_index] = entry;
  }

  commit_tracker_v0 tracker = {};
  tracker.owner = semantic_plan.owner;
  tracker.issue_age = state->next_issue_age;
  tracker.operation_seq = producer_operation_seq;
  tracker.target_operation_seq = target_operation_seq;
  tracker.commit_epoch = commit_epoch;
  tracker.active_decode_context =
      active_operands.decode_context;
  tracker.expected_write_count = commit.write_fragment_count;
  tracker.route_kind = commit.semantic_plan.route_kind;
  tracker.private_storage_profile = private_storage_profile;
  tracker.semantic_plan = commit.semantic_plan;
  tracker.private_state_384_commit = private_state_384_commit;
  tracker.valid = 1;
  tracker.ready = commit.write_fragment_count == 0 ? 1 : 0;
  state->trackers[tracker_index] = tracker;
  ++state->next_issue_age;

  receipt->producer_operation_seq = producer_operation_seq;
  receipt->target_operation_seq = target_operation_seq;
  receipt->commit_epoch = commit_epoch;
  receipt->write_count = commit.write_fragment_count;
  receipt->route_kind = commit.semantic_plan.route_kind;
  receipt->valid = 1;
  return kStatusOk;
}

status_kind capture_result(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const typed_primitive::route_input_v0 &input,
    const typed_primitive::route_result_v0 &result,
    capture_receipt_v0 *receipt) {
  primitive_semantic::semantic_plan_v0 semantic = {};
  if (primitive_semantic::prepare_result(
          owner, producer_operation_seq, input, result, &semantic) !=
      primitive_semantic::kStatusOk) {
    return kStatusSemanticPlanRejected;
  }
  return capture_semantic_plan_impl(
      state, producer_operation_seq, commit_epoch,
      target_operation_seq, region, canonical_slot, semantic,
      private_storage::kProfileLegacyShared832,
      &input.decode_context,
      private_state_384::operand_plan::kProducerInvalid, receipt);
}

status_kind capture_result_with_private_state_384(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &compatibility_slot,
    const typed_primitive::route_input_v0 &input,
    const typed_primitive::route_result_v0 &result,
    capture_receipt_v0 *receipt) {
  primitive_semantic::semantic_plan_v0 semantic = {};
  if (primitive_semantic::prepare_result(
          owner, producer_operation_seq, input, result, &semantic) !=
      primitive_semantic::kStatusOk) {
    return kStatusSemanticPlanRejected;
  }
  return capture_semantic_plan_impl(
      state, producer_operation_seq, commit_epoch,
      target_operation_seq, region, compatibility_slot, semantic,
      private_storage::kProfileCompressedShared384,
      &input.decode_context,
      private_state_384::operand_plan::kProducerPrimitive, receipt);
}

status_kind capture_result_global384(
    engine_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    uint64_t private_slot_base_address,
    const typed_primitive::route_input_v0 &input,
    const typed_primitive::route_result_v0 &result,
    capture_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      producer_operation_seq == 0 || commit_epoch == 0 ||
      private_slot_base_address == 0 ||
      private_slot_base_address % private_state_384::kChunkBytes != 0) {
    return kStatusInvalidArgument;
  }
  *receipt = capture_receipt_v0();
  primitive_semantic::semantic_plan_v0 semantic = {};
  if (primitive_semantic::prepare_result(
          owner, producer_operation_seq, input, result, &semantic) !=
          primitive_semantic::kStatusOk ||
      semantic.valid != 1 ||
      semantic.primitive_resume_valid != 0 ||
      semantic.shader_return_valid != 0) {
    return kStatusSemanticPlanRejected;
  }
  const bool boundary_route = is_boundary_route(semantic.route_kind);
  const bool successor_route =
      semantic.route_kind == primitive_semantic::kRouteStackPopNext;
  if ((!boundary_route && !successor_route) ||
      (boundary_route && target_operation_seq != 0) ||
      (successor_route &&
       (target_operation_seq == 0 ||
        target_operation_seq == producer_operation_seq))) {
    return kStatusSemanticPlanRejected;
  }
  if (operation_live(*state, owner, producer_operation_seq)) {
    return kStatusDuplicateOperation;
  }
  const int tracker_index = find_free_tracker(*state);
  if (tracker_index < 0) return kStatusTrackerBackpressure;
  if (state->next_issue_age == 0 ||
      state->next_issue_age == std::numeric_limits<uint64_t>::max()) {
    return kStatusCounterExhausted;
  }

  private_state_384::live_bridge::sparse_chunk_delta_v1
      deltas[kMaxPrivateState384DeltaCount] = {};
  uint8_t delta_count = 0;
  if (!prepare_private_state_384_deltas(
          semantic, input.decode_context, deltas, &delta_count)) {
    return kStatusPrivateState384Rejected;
  }
  private_state_384::live_bridge::pending_sparse_commit_v1 pending = {};
  if (delta_count != 0) {
    private_state_384::live_bridge::write_commit_input_v1 commit = {};
    commit.owner = owner;
    commit.private_slot_base_address = private_slot_base_address;
    commit.operation_sequence = producer_operation_seq;
    commit.commit_epoch = commit_epoch;
    commit.bvh_format_profile_id =
        private_state_384::kGenRtBvhFormatProfileId;
    commit.storage_profile = private_storage::kProfileGlobal384;
    commit.producer =
        private_state_384::operand_plan::kProducerPrimitive;
    if (private_state_384::live_bridge::stage_sparse_commit(
            commit, deltas, delta_count, &pending) !=
        private_state_384::live_bridge::kStatusOk) {
      return kStatusPrivateState384Rejected;
    }
  }

  commit_tracker_v0 tracker = {};
  tracker.owner = owner;
  tracker.active_decode_context = input.decode_context;
  tracker.issue_age = state->next_issue_age;
  tracker.operation_seq = producer_operation_seq;
  tracker.target_operation_seq = target_operation_seq;
  tracker.commit_epoch = commit_epoch;
  tracker.expected_write_count =
      pending.expected_write_ack_count;
  tracker.route_kind = semantic.route_kind;
  tracker.private_storage_profile =
      private_storage::kProfileGlobal384;
  tracker.semantic_plan = semantic;
  tracker.private_state_384_commit = pending;
  tracker.valid = 1;
  tracker.ready = tracker.expected_write_count == 0 ? 1 : 0;
  state->trackers[tracker_index] = tracker;
  ++state->next_issue_age;

  receipt->producer_operation_seq = producer_operation_seq;
  receipt->target_operation_seq = target_operation_seq;
  receipt->commit_epoch = commit_epoch;
  receipt->write_count = tracker.expected_write_count;
  receipt->route_kind = semantic.route_kind;
  receipt->valid = 1;
  return kStatusOk;
}

status_kind capture_semantic_plan(
    engine_state_v0 *state,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    const primitive_semantic::semantic_plan_v0 &semantic_plan,
    capture_receipt_v0 *receipt) {
  return capture_semantic_plan_impl(
      state, producer_operation_seq, commit_epoch,
      target_operation_seq, region, canonical_slot, semantic_plan,
      private_storage::kProfileLegacyShared832, NULL,
      private_state_384::operand_plan::kProducerInvalid, receipt);
}

status_kind capture_resubmit_semantic_plan_with_private_state_384(
    engine_state_v0 *state,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t target_operation_seq,
    const private_frontier::region_binding_v0 &region,
    const private_frontier::shadow_slot_v0 &compatibility_slot,
    const primitive_semantic::semantic_plan_v0 &semantic_plan,
    capture_receipt_v0 *receipt) {
  if (semantic_plan.shader_return_valid != 1 ||
      semantic_plan.retained_candidate_valid != 0 ||
      semantic_plan.primitive_resume_valid != 0 ||
      semantic_plan.intersection_boundary_valid != 0) {
    return kStatusSemanticPlanRejected;
  }
  const typed_blas::as_decode_context_v0 unused_decode_context = {};
  return capture_semantic_plan_impl(
      state, producer_operation_seq, commit_epoch,
      target_operation_seq, region, compatibility_slot,
      semantic_plan,
      private_storage::kProfileCompressedShared384,
      &unused_decode_context,
      private_state_384::operand_plan::kProducerResubmitApply,
      receipt);
}

status_kind peek_write_offer(const engine_state_v0 &state,
                             write_offer_v0 *offer) {
  if (offer == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *offer = write_offer_v0();
  const int index = find_oldest_result(state);
  if (index < 0) return kStatusNoWriteOffer;
  build_offer(state.result_entries[index], offer);
  return kStatusOk;
}

status_kind transfer_next_write(
    engine_state_v0 *state, private_shared::backing_state_v0 *shared_state,
    uint64_t enqueue_cycle, transfer_receipt_v0 *receipt) {
  if (state == NULL || shared_state == NULL || receipt == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *receipt = transfer_receipt_v0();
  write_offer_v0 offer = {};
  const status_kind peek_status = peek_write_offer(*state, &offer);
  if (peek_status != kStatusOk) return peek_status;
  private_shared::shared_write_v0 operation = {};
  const status_kind prepare_status =
      prepare_shared_write(offer, enqueue_cycle, &operation);
  if (prepare_status != kStatusOk) return prepare_status;
  const private_shared::status_kind memory_status =
      private_shared::validate_runtime_write(*shared_state, operation);
  if (memory_status == private_shared::kStatusQueueFull) {
    return kStatusSharedQueueBackpressure;
  }
  if (memory_status != private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }

  engine_state_v0 staged = *state;
  const int result_index = find_result(staged, offer);
  if (result_index < 0) return kStatusWriteOfferMismatch;
  result_commit_entry_v0 &entry = staged.result_entries[result_index];
  if (entry.next_write_index + 1 != offer.memory_operation_seq ||
      entry.next_write_index >= entry.write_count) {
    return kStatusWriteOfferMismatch;
  }
  commit_tracker_v0 &tracker = staged.trackers[entry.tracker_slot];
  const uint16_t bit =
      static_cast<uint16_t>(uint16_t{1} << entry.next_write_index);
  if (tracker.valid == 0 ||
      (tracker.accepted_write_mask & bit) != 0) {
    return kStatusWriteOfferMismatch;
  }
  if (tracker.private_storage_profile ==
          private_storage::kProfileCompressedShared384 &&
      private_state_384::live_bridge::register_modeled_write(
          operation, &tracker.private_state_384_commit) !=
          private_state_384::live_bridge::kStatusOk) {
    return kStatusPrivateState384Rejected;
  }
  tracker.accepted_write_mask |= bit;
  ++entry.next_write_index;
  if (entry.next_write_index == entry.write_count) {
    entry = result_commit_entry_v0();
  }
  if (private_shared::enqueue_runtime_write(shared_state, operation) !=
      private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }
  *state = staged;
  receipt->valid = true;
  receipt->primitive_offer = offer;
  receipt->shared_write = operation;
  return kStatusOk;
}

status_kind transfer_next_global_write(
    engine_state_v0 *state, uint64_t enqueue_cycle,
    private_shared::shared_write_v0 *write) {
  if (state == NULL || write == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *write = private_shared::shared_write_v0();
  const int tracker_index = find_oldest_global_write_tracker(*state);
  if (tracker_index < 0) return kStatusNoWriteOffer;

  engine_state_v0 staged = *state;
  commit_tracker_v0 &tracker = staged.trackers[tracker_index];
  const uint8_t write_index = first_missing_write(
      tracker.accepted_write_mask, tracker.expected_write_count);
  private_shared::shared_write_v0 prepared = {};
  if (write_index >= tracker.expected_write_count ||
      private_state_384::live_bridge::prepare_global_modeled_write(
          tracker.private_state_384_commit, write_index,
          enqueue_cycle, &prepared) !=
          private_state_384::live_bridge::kStatusOk ||
      private_state_384::live_bridge::register_modeled_write(
          prepared, &tracker.private_state_384_commit) !=
          private_state_384::live_bridge::kStatusOk) {
    return kStatusPrivateState384Rejected;
  }
  tracker.accepted_write_mask = static_cast<uint16_t>(
      tracker.accepted_write_mask | (uint16_t{1} << write_index));
  *state = staged;
  *write = prepared;
  return kStatusOk;
}

bool owns_ack(const engine_state_v0 &state,
              const private_shared::runtime_write_ack_v0 &ack) {
  return state.initialized == 1 && ack.valid &&
         find_tracker(state, ack.owner, ack.operation_seq,
                      ack.commit_epoch) >= 0;
}

bool profile_for_write(
    const engine_state_v0 &state,
    const private_shared::shared_write_v0 &write,
    uint8_t *private_storage_profile) {
  if (private_storage_profile == NULL || state.initialized != 1 ||
      !write.valid) {
    return false;
  }
  const int tracker_index =
      find_tracker(state, write.owner, write.operation_seq,
                   write.commit_epoch);
  if (tracker_index < 0) return false;
  const uint8_t profile =
      state.trackers[tracker_index].private_storage_profile;
  if (profile != private_storage::kProfileLegacyShared832 &&
      profile != private_storage::kProfileCompressedShared384 &&
      profile != private_storage::kProfileGlobal384) {
    return false;
  }
  *private_storage_profile = profile;
  return true;
}

status_kind service_next_ack(
    engine_state_v0 *state, private_shared::backing_state_v0 *shared_state,
    uint64_t service_cycle, ack_receipt_v0 *receipt) {
  if (state == NULL || shared_state == NULL || receipt == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *receipt = ack_receipt_v0();
  private_shared::runtime_write_ack_v0 ack = {};
  const private_shared::status_kind peek_status =
      private_shared::detail::peek_runtime_write_ack(
          *shared_state, service_cycle, &ack);
  if (peek_status == private_shared::kStatusNoAckReady) {
    return kStatusNoAckReady;
  }
  if (peek_status != private_shared::kStatusOk) {
    return kStatusSharedAckRejected;
  }
  const int tracker_index =
      find_tracker(*state, ack.owner, ack.operation_seq,
                   ack.commit_epoch);
  if (tracker_index < 0) return kStatusUnknownAck;
  commit_tracker_v0 &tracker = state->trackers[tracker_index];
  if (tracker.private_storage_profile !=
      private_storage::kProfileLegacyShared832) {
    return kStatusPrivateState384Rejected;
  }
  if (ack.memory_operation_seq == 0 ||
      ack.memory_operation_seq > tracker.expected_write_count) {
    return kStatusStaleAck;
  }
  const uint16_t bit = static_cast<uint16_t>(
      uint16_t{1} << (ack.memory_operation_seq - 1));
  if ((tracker.accepted_write_mask & bit) == 0) {
    return kStatusAckBeforeTransfer;
  }
  if ((tracker.acknowledged_write_mask & bit) != 0) {
    return kStatusDuplicateAck;
  }

  engine_state_v0 staged = *state;
  commit_tracker_v0 &staged_tracker =
      staged.trackers[tracker_index];
  staged_tracker.acknowledged_write_mask |= bit;
  if (staged_tracker.acknowledged_write_mask ==
      expected_mask(staged_tracker.expected_write_count)) {
    staged_tracker.ready = 1;
  }
  if (private_shared::detail::commit_runtime_write_ack(
          shared_state, service_cycle, ack) !=
      private_shared::kStatusOk) {
    return kStatusSharedAckRejected;
  }
  *state = staged;
  receipt->valid = true;
  receipt->shared_ack = ack;
  return kStatusOk;
}

status_kind service_next_ack_with_private_state_384(
    engine_state_v0 *state,
    private_shared::backing_state_v0 *shared_state,
    private_state_384::backing::state_v1 *private_state_384_backing,
    uint64_t service_cycle, ack_receipt_v0 *receipt) {
  if (state == NULL || shared_state == NULL ||
      private_state_384_backing == NULL || receipt == NULL ||
      state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *receipt = ack_receipt_v0();
  private_shared::runtime_write_ack_v0 ack = {};
  const private_shared::status_kind peek_status =
      private_shared::detail::peek_runtime_write_ack(
          *shared_state, service_cycle, &ack);
  if (peek_status == private_shared::kStatusNoAckReady) {
    return kStatusNoAckReady;
  }
  if (peek_status != private_shared::kStatusOk ||
      shared_state->outstanding.empty()) {
    return kStatusSharedAckRejected;
  }
  const int tracker_index =
      find_tracker(*state, ack.owner, ack.operation_seq,
                   ack.commit_epoch);
  if (tracker_index < 0) return kStatusUnknownAck;
  const commit_tracker_v0 &tracker = state->trackers[tracker_index];
  if (tracker.private_storage_profile !=
          private_storage::kProfileCompressedShared384 ||
      tracker.private_state_384_commit.valid != 1 ||
      ack.memory_operation_seq == 0 ||
      ack.memory_operation_seq > tracker.expected_write_count) {
    return kStatusPrivateState384Rejected;
  }
  const uint16_t bit = static_cast<uint16_t>(
      uint16_t{1} << (ack.memory_operation_seq - 1));
  if ((tracker.accepted_write_mask & bit) == 0) {
    return kStatusAckBeforeTransfer;
  }
  if ((tracker.acknowledged_write_mask & bit) != 0) {
    return kStatusDuplicateAck;
  }

  engine_state_v0 staged = *state;
  private_shared::backing_state_v0 staged_shared = *shared_state;
  private_state_384::backing::state_v1 staged_private_384 =
      *private_state_384_backing;
  commit_tracker_v0 &staged_tracker =
      staged.trackers[tracker_index];
  const private_shared::shared_write_v0 modeled_write =
      staged_shared.outstanding.front();
  bool canonical_committed = false;
  staged_tracker.acknowledged_write_mask |= bit;
  const bool all_acked =
      staged_tracker.acknowledged_write_mask ==
      expected_mask(staged_tracker.expected_write_count);
  if (private_shared::detail::commit_runtime_write_ack(
          &staged_shared, service_cycle, ack) !=
          private_shared::kStatusOk ||
      private_state_384::live_bridge::
              accept_write_ack_and_maybe_commit(
                  &staged_private_384, modeled_write, ack,
                  &staged_tracker.private_state_384_commit,
                  &canonical_committed) !=
          private_state_384::live_bridge::kStatusOk ||
      canonical_committed != all_acked) {
    return kStatusPrivateState384Rejected;
  }
  if (all_acked) staged_tracker.ready = 1;
  *state = staged;
  *shared_state = staged_shared;
  *private_state_384_backing = staged_private_384;
  receipt->valid = true;
  receipt->shared_ack = ack;
  return kStatusOk;
}

status_kind accept_global_write_ack(
    engine_state_v0 *state,
    const private_shared::shared_write_v0 &write,
    const private_shared::runtime_write_ack_v0 &ack) {
  if (state == NULL || state->initialized != 1 || !ack.valid) {
    return kStatusInvalidArgument;
  }
  const int tracker_index = find_tracker(
      *state, ack.owner, ack.operation_seq, ack.commit_epoch);
  if (tracker_index < 0) return kStatusUnknownAck;
  const commit_tracker_v0 &tracker = state->trackers[tracker_index];
  if (tracker.private_storage_profile !=
          private_storage::kProfileGlobal384 ||
      tracker.private_state_384_commit.valid != 1 ||
      ack.memory_operation_seq == 0 ||
      ack.memory_operation_seq > tracker.expected_write_count) {
    return kStatusPrivateState384Rejected;
  }
  const uint16_t bit = static_cast<uint16_t>(
      uint16_t{1} << (ack.memory_operation_seq - 1));
  if ((tracker.accepted_write_mask & bit) == 0) {
    return kStatusAckBeforeTransfer;
  }
  if ((tracker.acknowledged_write_mask & bit) != 0) {
    return kStatusDuplicateAck;
  }

  engine_state_v0 staged = *state;
  commit_tracker_v0 &staged_tracker =
      staged.trackers[tracker_index];
  bool all_acknowledged = false;
  if (private_state_384::live_bridge::accept_global_write_ack(
          write, ack, &staged_tracker.private_state_384_commit,
          &all_acknowledged) !=
      private_state_384::live_bridge::kStatusOk) {
    return kStatusPrivateState384Rejected;
  }
  staged_tracker.acknowledged_write_mask = static_cast<uint16_t>(
      staged_tracker.acknowledged_write_mask | bit);
  const bool all_acked =
      staged_tracker.acknowledged_write_mask ==
      expected_mask(staged_tracker.expected_write_count);
  if (all_acked != all_acknowledged) {
    return kStatusPrivateState384Rejected;
  }
  if (all_acked) staged_tracker.ready = 1;
  *state = staged;
  return kStatusOk;
}

status_kind pop_ready_event(engine_state_v0 *state,
                            ready_event_v0 *event) {
  if (state == NULL || event == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *event = ready_event_v0();
  int selected = -1;
  uint64_t selected_age = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state->config.tracker_capacity;
       ++index) {
    const commit_tracker_v0 &tracker = state->trackers[index];
    if (tracker.valid != 0 && tracker.ready != 0 &&
        (selected < 0 || tracker.issue_age < selected_age)) {
      selected = static_cast<int>(index);
      selected_age = tracker.issue_age;
    }
  }
  if (selected < 0) return kStatusNoReadyEvent;
  const commit_tracker_v0 tracker = state->trackers[selected];
  if (tracker.accepted_write_mask !=
          expected_mask(tracker.expected_write_count) ||
      tracker.acknowledged_write_mask !=
          expected_mask(tracker.expected_write_count)) {
    return kStatusStaleAck;
  }
  event->owner = tracker.owner;
  event->active_decode_context =
      tracker.active_decode_context;
  event->producer_operation_seq = tracker.operation_seq;
  event->target_operation_seq = tracker.target_operation_seq;
  event->commit_epoch = tracker.commit_epoch;
  event->route_kind = tracker.route_kind;
  event->private_storage_profile =
      tracker.private_storage_profile;
  event->semantic_plan = tracker.semantic_plan;
  event->valid = 1;
  state->trackers[selected] = commit_tracker_v0();
  return kStatusOk;
}

status_kind enqueue_boundary_receipt(
    engine_state_v0 *state, const ready_event_v0 &event) {
  if (state == NULL || state->initialized != 1 || event.valid != 1 ||
      event.producer_operation_seq == 0 || event.commit_epoch == 0 ||
      event.target_operation_seq != 0 ||
      event.route_kind != event.semantic_plan.route_kind) {
    return kStatusInvalidArgument;
  }
  if (!is_boundary_route(event.route_kind)) return kStatusInvalidRoute;
  if (state->boundary_count >= state->config.boundary_capacity) {
    return kStatusBoundaryBackpressure;
  }
  ready_event_v0 queued = event;
  if (event.private_storage_profile ==
          private_storage::kProfileCompressedShared384 ||
      event.private_storage_profile ==
          private_storage::kProfileGlobal384) {
    queued.semantic_plan = primitive_semantic::semantic_plan_v0();
    queued.semantic_plan.owner = event.owner;
    queued.semantic_plan.operation_seq =
        event.producer_operation_seq;
    queued.semantic_plan.route_kind = event.route_kind;
    queued.semantic_plan.valid = 1;
  }
  state->boundary_receipts[state->boundary_count++] = queued;
  return kStatusOk;
}

status_kind peek_boundary_receipt(
    const engine_state_v0 &state, boundary_receipt_v0 *receipt) {
  if (receipt == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  *receipt = boundary_receipt_v0();
  if (state.boundary_count == 0) return kStatusNoBoundaryReceipt;
  *receipt = state.boundary_receipts[0];
  return kStatusOk;
}

status_kind pop_boundary_receipt(
    engine_state_v0 *state, boundary_receipt_v0 *receipt) {
  if (state == NULL || receipt == NULL) {
    return kStatusInvalidArgument;
  }
  const status_kind status = peek_boundary_receipt(*state, receipt);
  if (status != kStatusOk) return status;
  for (unsigned index = 1; index < state->boundary_count; ++index) {
    state->boundary_receipts[index - 1] =
        state->boundary_receipts[index];
  }
  --state->boundary_count;
  state->boundary_receipts[state->boundary_count] =
      boundary_receipt_v0();
  return kStatusOk;
}

uint8_t active_result_entry_count(const engine_state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return count;
  for (unsigned index = 0;
       index < state.config.result_commit_capacity; ++index) {
    count += state.result_entries[index].valid != 0;
  }
  return count;
}

uint8_t active_tracker_count(const engine_state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return count;
  for (unsigned index = 0; index < state.config.tracker_capacity;
       ++index) {
    count += state.trackers[index].valid != 0;
  }
  return count;
}

uint8_t boundary_receipt_count(const engine_state_v0 &state) {
  return state.initialized == 1 ? state.boundary_count : 0;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfiguration:
      return "invalid_configuration";
    case kStatusCounterExhausted:
      return "counter_exhausted";
    case kStatusResultBackpressure:
      return "result_backpressure";
    case kStatusTrackerBackpressure:
      return "tracker_backpressure";
    case kStatusBoundaryBackpressure:
      return "boundary_backpressure";
    case kStatusDuplicateOperation:
      return "duplicate_operation";
    case kStatusSemanticPlanRejected:
      return "semantic_plan_rejected";
    case kStatusNoWriteOffer:
      return "no_write_offer";
    case kStatusWriteOfferMismatch:
      return "write_offer_mismatch";
    case kStatusSharedQueueBackpressure:
      return "shared_queue_backpressure";
    case kStatusSharedWriteRejected:
      return "shared_write_rejected";
    case kStatusNoAckReady:
      return "no_ack_ready";
    case kStatusUnknownAck:
      return "unknown_ack";
    case kStatusStaleAck:
      return "stale_ack";
    case kStatusAckBeforeTransfer:
      return "ack_before_transfer";
    case kStatusDuplicateAck:
      return "duplicate_ack";
    case kStatusSharedAckRejected:
      return "shared_ack_rejected";
    case kStatusNoReadyEvent:
      return "no_ready_event";
    case kStatusNoBoundaryReceipt:
      return "no_boundary_receipt";
    case kStatusInvalidRoute:
      return "invalid_route";
    case kStatusPrivateState384Rejected:
      return "private_state_384_rejected";
  }
  return "unknown";
}

}  // namespace primitive_shared
}  // namespace v04
}  // namespace rtcore
