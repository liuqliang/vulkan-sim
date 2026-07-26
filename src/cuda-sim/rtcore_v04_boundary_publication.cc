#include "rtcore_v04_boundary_publication.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace boundary_publication {
namespace {

unsigned count_bits(uint32_t mask) {
  unsigned count = 0;
  while (mask != 0) {
    mask &= mask - 1;
    ++count;
  }
  return count;
}

uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint8_t publication_chunk_mask(uint32_t word_mask) {
  uint8_t chunk_mask = 0;
  for (unsigned word = 0; word < abi_v04::kWordCount; ++word) {
    if ((word_mask & (uint32_t{1} << word)) != 0) {
      chunk_mask = static_cast<uint8_t>(
          chunk_mask | (uint8_t{1} << (word / (kChunkBytes / 4))));
    }
  }
  return chunk_mask;
}

bool plan_shape_valid(
    const primitive_semantic::semantic_plan_v0 &plan) {
  if (plan.valid != 1 || plan.operation_seq == 0 ||
      plan.primitive_resume_valid != 0 ||
      plan.reserved_zero != 0) {
    return false;
  }
  if (plan.route_kind ==
      primitive_semantic::kRouteAnyHitBoundary) {
    return plan.retained_candidate_valid == 1 &&
           plan.committed_hit_valid == 0 &&
           plan.intersection_boundary_valid == 0;
  }
  if (plan.route_kind ==
      primitive_semantic::kRouteIntersectionBoundary) {
    return plan.retained_candidate_valid == 1 &&
           plan.committed_hit_valid == 0 &&
           plan.intersection_boundary_valid == 1;
  }
  if (plan.route_kind ==
      primitive_semantic::kRouteFinalHitBoundary) {
    return plan.retained_candidate_valid == 0 &&
           plan.committed_hit_valid == 1 &&
           plan.intersection_boundary_valid == 0;
  }
  return false;
}

bool make_boundary_values(
    const primitive_semantic::semantic_plan_v0 &plan, uint32_t *reason,
    abi_v04::shadow::boundary_values *values) {
  if (reason == NULL || values == NULL || !plan_shape_valid(plan)) {
    return false;
  }
  *values = abi_v04::shadow::boundary_values();
  values->candidate_valid = true;

  if (plan.route_kind ==
      primitive_semantic::kRouteFinalHitBoundary) {
    const typed_stack::committed_hit_projection_v0 &hit =
        plan.committed_hit;
    if (hit.valid != 1) return false;
    *reason = abi_v04::kReasonClosestHitReady;
    values->instance_metadata_reference = hit.instance_metadata_ref;
    values->instance_sbt_contribution =
        hit.instance_sbt_contribution;
    values->geometry_index = hit.geometry_index;
    values->boundary_ray_tmax_fp32 = fp32_bits(hit.hit_t);
    values->primitive_index = hit.primitive_index;
    values->instance_index = hit.instance_index;
    values->instance_custom_index = hit.instance_custom_index;
    values->geometry_type = hit.geometry_type;
    values->hit_kind = hit.hit_kind;
    values->input_attribute_word_count = hit.attribute_word_count;
    values->input_attribute_location = hit.attribute_location;
    values->input_attribute_format = hit.attribute_format;
    for (unsigned word = 0;
         word < values->inline_attribute_words.size(); ++word) {
      values->inline_attribute_words[word] = hit.inline_attributes[word];
    }
    return true;
  }

  const primitive_semantic::retained_candidate_projection_v0 &candidate =
      plan.retained_candidate;
  const typed_primitive::primitive_identity_policy_facts_v0 &identity =
      candidate.identity_and_policy;
  values->instance_metadata_reference = identity.instance_metadata_ref;
  values->instance_sbt_contribution =
      identity.instance_sbt_contribution;
  values->geometry_index = identity.geometry_index;
  values->primitive_index = identity.primitive_index;
  values->instance_index = identity.instance_index;
  values->instance_custom_index = identity.instance_custom_index;
  values->geometry_type = identity.geometry_type;

  if (plan.route_kind ==
      primitive_semantic::kRouteAnyHitBoundary) {
    *reason = abi_v04::kReasonAnyHitRequired;
    values->boundary_ray_tmax_fp32 =
        candidate.triangle_hit.hit_t_bits;
    values->hit_kind = candidate.triangle_hit.hit_kind;
    values->input_attribute_word_count = 2;
    values->input_attribute_location = 1;
    values->input_attribute_format = 1;
    values->inline_attribute_words[0] =
        candidate.triangle_hit.bary_vertex1_bits;
    values->inline_attribute_words[1] =
        candidate.triangle_hit.bary_vertex2_bits;
    return true;
  }

  *reason = abi_v04::kReasonIntersectionRequired;
  values->boundary_ray_tmax_fp32 =
      plan.intersection_boundary.boundary_ray_tmax_bits;
  values->procedural_any_hit_eligible =
      (identity.effective_policy_flags &
       typed_primitive::kPolicyProceduralAnyHitEligible) != 0;
  return true;
}

}  // namespace

status_kind initialize(warp_state_v0 *state, uint32_t owner_hw_sid,
                       uint32_t warp_uid, uint32_t warp_id,
                       uint32_t active_mask, uint8_t resident_warp_slot) {
  if (state == NULL) return kStatusInvalidArgument;
  *state = warp_state_v0();
  if (active_mask == 0 || resident_warp_slot >=
                              request_owner::kResidentWarpCapacity) {
    return kStatusInvalidConfiguration;
  }
  state->next_arm_age = 1;
  state->owner_hw_sid = owner_hw_sid;
  state->warp_uid = warp_uid;
  state->warp_id = warp_id;
  state->active_mask = active_mask;
  state->resident_warp_slot = resident_warp_slot;
  state->initialized = 1;
  return kStatusOk;
}

status_kind arm_boundary(
    warp_state_v0 *state,
    const primitive_semantic::semantic_plan_v0 &semantic_plan,
    const std::array<uint32_t, abi_v04::kWordCount> &preimage_words,
    uint32_t commit_epoch, uint64_t arm_cycle, arm_receipt_v0 *receipt) {
  if (receipt != NULL) *receipt = arm_receipt_v0();
  if (state == NULL || receipt == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  const private_frontier::owner_binding_v0 &owner =
      semantic_plan.owner;
  if (owner.owner_hw_sid != state->owner_hw_sid ||
      owner.resident_warp_id != state->resident_warp_slot ||
      owner.lane_id >= kLaneCapacity ||
      (state->active_mask & (uint32_t{1} << owner.lane_id)) == 0) {
    return kStatusOwnerMismatch;
  }
  if (!plan_shape_valid(semantic_plan) || commit_epoch == 0) {
    return kStatusInvalidSemanticPlan;
  }
  lane_publication_v0 &lane = state->lanes[owner.lane_id];
  if (lane.valid != 0 ||
      (state->publication_armed_mask &
       (uint32_t{1} << owner.lane_id)) != 0) {
    return kStatusLaneBusy;
  }
  if (state->next_arm_age == std::numeric_limits<uint64_t>::max()) {
    return kStatusInvalidConfiguration;
  }

  uint32_t reason = abi_v04::kReasonNoneOrInvalid;
  abi_v04::shadow::boundary_values values;
  if (!make_boundary_values(semantic_plan, &reason, &values)) {
    return kStatusInvalidSemanticPlan;
  }
  const abi_v04::shadow::boundary_publication publication =
      abi_v04::shadow::build_boundary_publication(
          preimage_words, reason, values);
  uint32_t compact_result = 0;
  if (!publication.valid() ||
      !abi_v04::pack_compact_result(reason, true, &compact_result)) {
    return kStatusBoundaryEncodingRejected;
  }
  const uint8_t chunk_mask =
      publication_chunk_mask(publication.written_word_mask);
  if (chunk_mask == 0) return kStatusBoundaryEncodingRejected;

  lane = lane_publication_v0();
  lane.owner = owner;
  lane.arm_age = state->next_arm_age++;
  lane.arm_cycle = arm_cycle;
  lane.producer_operation_seq = semantic_plan.operation_seq;
  lane.commit_epoch = commit_epoch;
  lane.reason = reason;
  lane.compact_result = compact_result;
  lane.written_word_mask = publication.written_word_mask;
  lane.route_kind = semantic_plan.route_kind;
  lane.pending_chunk_mask = chunk_mask;
  lane.valid = 1;
  lane.words = publication.words;
  lane.preimage_words = preimage_words;
  state->publication_armed_mask |= uint32_t{1} << owner.lane_id;

  receipt->reason = reason;
  receipt->compact_result = compact_result;
  receipt->written_word_mask = publication.written_word_mask;
  receipt->pending_chunk_mask = chunk_mask;
  receipt->lane_id = owner.lane_id;
  receipt->valid = 1;
  return kStatusOk;
}

status_kind accept_chunk_ack(
    warp_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint8_t chunk_id, uint64_t response_cycle, ack_receipt_v0 *receipt) {
  if (receipt != NULL) *receipt = ack_receipt_v0();
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      owner.lane_id >= kLaneCapacity || chunk_id >= kChunkCount) {
    return kStatusInvalidArgument;
  }
  lane_publication_v0 &lane = state->lanes[owner.lane_id];
  if (lane.valid != 1) return kStatusNoArmedPublication;
  if (!private_frontier::owners_equal(lane.owner, owner) ||
      lane.producer_operation_seq != producer_operation_seq ||
      lane.commit_epoch != commit_epoch ||
      (lane.pending_chunk_mask & (uint8_t{1} << chunk_id)) == 0) {
    return kStatusAckIdentityMismatch;
  }
  if ((lane.acknowledged_chunk_mask &
       (uint8_t{1} << chunk_id)) != 0) {
    return kStatusDuplicateAck;
  }
  lane.acknowledged_chunk_mask = static_cast<uint8_t>(
      lane.acknowledged_chunk_mask | (uint8_t{1} << chunk_id));
  if (lane.acknowledged_chunk_mask == lane.pending_chunk_mask) {
    lane.ack_complete = 1;
    lane.final_ack_cycle = response_cycle;
    state->publication_acked_mask |= uint32_t{1} << owner.lane_id;
  }
  receipt->completed_lane_mask = state->completed_lane_mask;
  receipt->lane_id = owner.lane_id;
  receipt->acknowledged_chunk_mask = lane.acknowledged_chunk_mask;
  receipt->lane_ack_complete = lane.ack_complete;
  receipt->valid = 1;
  return kStatusOk;
}

status_kind select_oldest_completion_ready_lane(
    const warp_state_v0 &state, uint8_t *lane_id) {
  if (lane_id == NULL || state.initialized != 1) {
    return kStatusInvalidArgument;
  }
  uint64_t oldest_age = std::numeric_limits<uint64_t>::max();
  unsigned oldest_lane = kLaneCapacity;
  for (unsigned lane = 0; lane < kLaneCapacity; ++lane) {
    const lane_publication_v0 &candidate = state.lanes[lane];
    if (candidate.valid != 1 || candidate.ack_complete != 1 ||
        candidate.completion_published != 0) {
      continue;
    }
    if (candidate.arm_age < oldest_age ||
        (candidate.arm_age == oldest_age && lane < oldest_lane)) {
      oldest_age = candidate.arm_age;
      oldest_lane = lane;
    }
  }
  if (oldest_lane == kLaneCapacity) return kStatusNoCompletionReady;
  *lane_id = static_cast<uint8_t>(oldest_lane);
  return kStatusOk;
}

status_kind publish_lane_completion(warp_state_v0 *state, uint8_t lane_id,
                                    completion_receipt_v0 *receipt) {
  if (receipt != NULL) *receipt = completion_receipt_v0();
  if (state == NULL || receipt == NULL || state->initialized != 1 ||
      lane_id >= kLaneCapacity) {
    return kStatusInvalidArgument;
  }
  lane_publication_v0 &lane = state->lanes[lane_id];
  if (lane.valid != 1 || lane.ack_complete != 1) {
    return kStatusNoCompletionReady;
  }
  if (lane.completion_published != 0) {
    return kStatusCompletionAlreadyPublished;
  }
  const uint32_t lane_mask = uint32_t{1} << lane_id;
  lane.completion_published = 1;
  state->completed_lane_mask |= lane_mask;
  state->result_valid_mask |= lane_mask;
  if (lane.route_kind ==
      primitive_semantic::kRouteFinalHitBoundary) {
    state->terminal_lane_mask |= lane_mask;
  } else {
    state->continuation_lane_mask |= lane_mask;
  }
  state->scoreboard_handoff_ready =
      all_active_lanes_complete(*state) &&
      transaction_quiescent(*state);

  receipt->completed_lane_mask = state->completed_lane_mask;
  receipt->result_valid_mask = state->result_valid_mask;
  receipt->terminal_lane_mask = state->terminal_lane_mask;
  receipt->continuation_lane_mask = state->continuation_lane_mask;
  receipt->lane_id = lane_id;
  receipt->warp_completion_ready = state->scoreboard_handoff_ready;
  receipt->valid = 1;
  return kStatusOk;
}

status_kind mark_scoreboard_handoff_delivered(warp_state_v0 *state,
                                              uint64_t service_cycle) {
  if (state == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  if (state->scoreboard_handoff_delivered != 0) {
    return kStatusScoreboardAlreadyDelivered;
  }
  if (!all_active_lanes_complete(*state) ||
      !transaction_quiescent(*state) ||
      state->scoreboard_handoff_ready != 1) {
    return kStatusWarpCompletionNotReady;
  }
  state->scoreboard_handoff_ready = 0;
  state->scoreboard_handoff_delivered = 1;
  state->scoreboard_handoff_cycle = service_cycle;
  return kStatusOk;
}

status_kind consume_completion_transaction(warp_state_v0 *state) {
  if (state == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  if (state->scoreboard_handoff_delivered != 1 ||
      !all_active_lanes_complete(*state)) {
    return kStatusWarpCompletionNotReady;
  }
  if (state->completion_consumed != 0) {
    return kStatusCompletionAlreadyConsumed;
  }
  state->completion_consumed = 1;
  return kStatusOk;
}

bool all_active_lanes_complete(const warp_state_v0 &state) {
  return state.initialized == 1 && state.active_mask != 0 &&
         (state.completed_lane_mask & state.active_mask) ==
             state.active_mask;
}

bool transaction_quiescent(const warp_state_v0 &state) {
  if (state.initialized != 1) return false;
  for (unsigned lane = 0; lane < kLaneCapacity; ++lane) {
    if ((state.active_mask & (uint32_t{1} << lane)) == 0) continue;
    const lane_publication_v0 &publication = state.lanes[lane];
    if (publication.valid != 1 ||
        publication.ack_complete != 1 ||
        publication.acknowledged_chunk_mask !=
            publication.pending_chunk_mask) {
      return false;
    }
  }
  return count_bits(state.completed_lane_mask & state.active_mask) ==
         count_bits(state.active_mask);
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfiguration:
      return "invalid_configuration";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusInvalidSemanticPlan:
      return "invalid_semantic_plan";
    case kStatusBoundaryEncodingRejected:
      return "boundary_encoding_rejected";
    case kStatusLaneBusy:
      return "lane_busy";
    case kStatusNoArmedPublication:
      return "no_armed_publication";
    case kStatusAckIdentityMismatch:
      return "ack_identity_mismatch";
    case kStatusDuplicateAck:
      return "duplicate_ack";
    case kStatusNoCompletionReady:
      return "no_completion_ready";
    case kStatusCompletionAlreadyPublished:
      return "completion_already_published";
    case kStatusWarpCompletionNotReady:
      return "warp_completion_not_ready";
    case kStatusScoreboardAlreadyDelivered:
      return "scoreboard_already_delivered";
    case kStatusCompletionAlreadyConsumed:
      return "completion_already_consumed";
  }
  return "unknown";
}

}  // namespace boundary_publication
}  // namespace v04
}  // namespace rtcore
