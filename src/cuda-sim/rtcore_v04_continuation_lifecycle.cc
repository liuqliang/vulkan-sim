#include "rtcore_v04_continuation_lifecycle.h"

#include <cmath>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace continuation_lifecycle {
namespace {

uint32_t lane_bit(uint32_t lane) {
  return uint32_t{1} << lane;
}

float fp32_value(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool owner_matches(
    const private_frontier::owner_binding_v0 &lhs,
    const private_frontier::owner_binding_v0 &rhs) {
  return private_frontier::owners_equal(lhs, rhs);
}

uint64_t extract_u64(
    const std::array<uint32_t, abi_v04::kWordCount> &words,
    const abi_v04::field_spec &low, const abi_v04::field_spec &high) {
  return static_cast<uint64_t>(abi_v04::extract_field(words, low)) |
         (static_cast<uint64_t>(abi_v04::extract_field(words, high))
          << 32);
}

bool identity_matches_handoff(
    const private_frontier::retained_candidate_projection_v0 &retained,
    uint32_t reason,
    const std::array<uint32_t, abi_v04::kWordCount> &words) {
  const typed_primitive::primitive_identity_policy_facts_v0 &identity =
      retained.identity_and_policy;
  const uint8_t expected_geometry =
      reason == abi_v04::kReasonAnyHitRequired
          ? typed_primitive::kGeometryTypeTriangle
          : typed_primitive::kGeometryTypeProcedural;
  if (identity.geometry_type != expected_geometry ||
      identity.instance_metadata_ref !=
          extract_u64(words, abi_v04::kInstanceMetadataReferenceLow32,
                      abi_v04::kInstanceMetadataReferenceHigh32) ||
      identity.instance_sbt_contribution !=
          abi_v04::extract_field(words, abi_v04::kInstanceSbtContribution) ||
      identity.geometry_index !=
          abi_v04::extract_field(words, abi_v04::kGeometryIndex) ||
      identity.primitive_index !=
          abi_v04::extract_field(words, abi_v04::kPrimitiveIndex) ||
      identity.instance_index !=
          abi_v04::extract_field(words, abi_v04::kInstanceIndex) ||
      identity.instance_custom_index !=
          abi_v04::extract_field(words, abi_v04::kInstanceCustomIndex)) {
    return false;
  }
  if (reason != abi_v04::kReasonAnyHitRequired) return true;
  return retained.triangle_hit.hit_t_bits ==
             abi_v04::extract_field(words, abi_v04::kBoundaryRayTmaxFp32) &&
         retained.triangle_hit.hit_kind ==
             abi_v04::extract_field(words, abi_v04::kHitKind) &&
         abi_v04::extract_field(
             words, abi_v04::kInputAttributeWordCount) == 2u &&
         abi_v04::extract_field(
             words, abi_v04::kInputAttributeLocation) == 1u &&
         abi_v04::extract_field(
             words, abi_v04::kInputAttributeFormat) == 1u &&
         retained.triangle_hit.bary_vertex1_bits ==
             abi_v04::extract_field(
                 words, abi_v04::kInlineAttributeWord0) &&
         retained.triangle_hit.bary_vertex2_bits ==
             abi_v04::extract_field(
                 words, abi_v04::kInlineAttributeWord1);
}

private_frontier::committed_hit_projection_v0
triangle_committed_hit(
    const private_frontier::retained_candidate_projection_v0 &retained) {
  private_frontier::committed_hit_projection_v0 hit = {};
  const typed_primitive::primitive_identity_policy_facts_v0 &identity =
      retained.identity_and_policy;
  hit.valid = 1;
  hit.geometry_type = typed_primitive::kGeometryTypeTriangle;
  hit.hit_kind = retained.triangle_hit.hit_kind;
  hit.attribute_word_count = 2;
  hit.attribute_location = 1;
  hit.attribute_format = 1;
  hit.hit_t = fp32_value(retained.triangle_hit.hit_t_bits);
  hit.policy_flags = identity.effective_policy_flags;
  hit.instance_metadata_ref = identity.instance_metadata_ref;
  hit.primitive_index = identity.primitive_index;
  hit.geometry_index = identity.geometry_index;
  hit.instance_index = identity.instance_index;
  hit.instance_custom_index = identity.instance_custom_index;
  hit.instance_sbt_contribution = identity.instance_sbt_contribution;
  hit.inline_attributes[0] = retained.triangle_hit.bary_vertex1_bits;
  hit.inline_attributes[1] = retained.triangle_hit.bary_vertex2_bits;
  return hit;
}

private_frontier::committed_hit_projection_v0
procedural_committed_hit(
    const private_frontier::retained_candidate_projection_v0 &retained,
    const abi_v04::shadow::boundary_return_update &update) {
  private_frontier::committed_hit_projection_v0 hit = {};
  const typed_primitive::primitive_identity_policy_facts_v0 &identity =
      retained.identity_and_policy;
  hit.valid = 1;
  hit.geometry_type = typed_primitive::kGeometryTypeProcedural;
  hit.hit_kind = static_cast<uint8_t>(update.reported_hit_kind);
  hit.attribute_word_count =
      static_cast<uint8_t>(update.reported_attribute_word_count);
  hit.attribute_location =
      update.reported_attribute_word_count == 0 ? 0 : 1;
  hit.attribute_format =
      static_cast<uint8_t>(update.reported_attribute_format);
  hit.hit_t = fp32_value(update.reported_t_fp32);
  hit.policy_flags = identity.effective_policy_flags;
  hit.instance_metadata_ref = identity.instance_metadata_ref;
  hit.primitive_index = identity.primitive_index;
  hit.geometry_index = identity.geometry_index;
  hit.instance_index = identity.instance_index;
  hit.instance_custom_index = identity.instance_custom_index;
  hit.instance_sbt_contribution = identity.instance_sbt_contribution;
  for (uint32_t word = 0;
       word < update.reported_attribute_word_count; ++word) {
    hit.inline_attributes[word] = update.reported_attribute_words[word];
  }
  return hit;
}

}  // namespace

status_kind initialize(
    warp_state_v0 *state, uint32_t owner_hw_sid, uint32_t warp_uid,
    uint32_t warp_id, uint32_t resident_generation,
    uint32_t active_mask) {
  if (state == NULL || active_mask == 0 || resident_generation == 0) {
    return kStatusInvalidArgument;
  }
  *state = warp_state_v0();
  state->owner_hw_sid = owner_hw_sid;
  state->current_warp_uid = warp_uid;
  state->warp_id = warp_id;
  state->resident_generation = resident_generation;
  state->active_mask = active_mask;
  state->completion_transaction_generation = 1;
  state->initialized = 1;
  for (uint32_t lane = 0; lane < 32; ++lane) {
    if ((active_mask & lane_bit(lane)) != 0) {
      state->lanes[lane].state = kLaneActiveTraversal;
    }
  }
  return kStatusOk;
}

status_kind consume_completion(
    warp_state_v0 *state, uint32_t continuation_mask,
    uint32_t terminal_mask) {
  if (state == NULL || state->initialized != 1 || state->released != 0) {
    return kStatusInvalidArgument;
  }
  if (state->completion_consumed != 0) {
    return kStatusCompletionAlreadyConsumed;
  }
  if ((continuation_mask & terminal_mask) != 0 ||
      (continuation_mask | terminal_mask) != state->active_mask) {
    return kStatusCompletionMaskMismatch;
  }
  state->continuation_mask = continuation_mask;
  state->terminal_mask = terminal_mask;
  state->dispatch_pending_mask = state->active_mask;
  state->dispatch_completed_mask = 0;
  state->completion_consumed = 1;
  for (uint32_t lane = 0; lane < 32; ++lane) {
    if ((continuation_mask & lane_bit(lane)) != 0) {
      state->lanes[lane].state = kLaneWaitingShader;
    } else if ((terminal_mask & lane_bit(lane)) != 0) {
      state->lanes[lane].state = kLaneFinalWaitRelease;
    }
  }
  return kStatusOk;
}

status_kind mark_dispatch_complete(
    warp_state_v0 *state, uint32_t completed_lane_mask) {
  if (state == NULL || state->initialized != 1 ||
      state->completion_consumed != 1 || state->released != 0 ||
      completed_lane_mask == 0 ||
      (completed_lane_mask & ~state->active_mask) != 0) {
    return kStatusInvalidArgument;
  }
  if ((completed_lane_mask & state->dispatch_completed_mask) != 0) {
    return kStatusDispatchAlreadyCompleted;
  }
  if ((completed_lane_mask & state->dispatch_pending_mask) !=
      completed_lane_mask) {
    return kStatusInvalidArgument;
  }
  state->dispatch_pending_mask &= ~completed_lane_mask;
  state->dispatch_completed_mask |= completed_lane_mask;
  return kStatusOk;
}

status_kind mark_shader_terminal_publication(
    warp_state_v0 *state, uint32_t terminal_lane_mask) {
  if (state == NULL || state->initialized != 1 ||
      state->completion_consumed != 1 || state->released != 0 ||
      terminal_lane_mask == 0 ||
      (terminal_lane_mask & ~state->continuation_mask) != 0) {
    return kStatusInvalidArgument;
  }
  for (uint32_t lane = 0; lane < 32; ++lane) {
    const uint32_t bit = lane_bit(lane);
    if ((terminal_lane_mask & bit) == 0) continue;
    if (state->lanes[lane].state != kLaneWaitingShader) {
      return kStatusLaneNotWaitingShader;
    }
  }
  for (uint32_t lane = 0; lane < 32; ++lane) {
    const uint32_t bit = lane_bit(lane);
    if ((terminal_lane_mask & bit) == 0) continue;
    state->lanes[lane].state = kLaneFinalWaitRelease;
  }
  state->continuation_mask &= ~terminal_lane_mask;
  state->terminal_mask |= terminal_lane_mask;
  return kStatusOk;
}

status_kind stage_resubmit(
    warp_state_v0 *state, uint32_t next_warp_uid,
    uint32_t next_static_inst_uid, uint32_t next_active_mask,
    uint32_t resume_mask, uint32_t shader_terminal_mask) {
  if (state == NULL || state->initialized != 1 ||
      state->completion_consumed != 1 || state->released != 0) {
    return kStatusInvalidArgument;
  }
  if (state->dispatch_pending_mask != 0 ||
      state->dispatch_completed_mask != state->active_mask) {
    return kStatusDispatchNotComplete;
  }
  if (state->resubmit_pending != 0) {
    return kStatusResubmitAlreadyPending;
  }
  if (next_warp_uid == 0 ||
      next_warp_uid == state->current_warp_uid ||
      next_static_inst_uid == 0 || next_active_mask == 0 ||
      (next_active_mask & ~state->active_mask) != 0 ||
      next_active_mask != state->continuation_mask ||
      (resume_mask & shader_terminal_mask) != 0 ||
      (resume_mask | shader_terminal_mask) != next_active_mask) {
    return kStatusInvalidResubmitMask;
  }
  for (uint32_t lane = 0; lane < 32; ++lane) {
    if ((next_active_mask & lane_bit(lane)) != 0 &&
        state->lanes[lane].state != kLaneWaitingShader) {
      return kStatusLaneNotWaitingShader;
    }
  }
  state->pending_next_warp_uid = next_warp_uid;
  state->pending_next_static_inst_uid = next_static_inst_uid;
  state->pending_next_active_mask = next_active_mask;
  state->pending_resume_mask = resume_mask;
  state->pending_shader_terminal_mask = shader_terminal_mask;
  state->pending_release_mask = state->active_mask & ~next_active_mask;
  state->return_staged_mask = 0;
  state->commit_enqueued_mask = 0;
  state->commit_ready_mask = 0;
  state->resubmit_pending = 1;
  return kStatusOk;
}

status_kind stage_lane_return(
    warp_state_v0 *state, uint8_t lane_id,
    const primitive_semantic::semantic_plan_v0 &semantic_plan) {
  if (state == NULL || lane_id >= 32 || state->resubmit_pending != 1 ||
      semantic_plan.valid != 1 || semantic_plan.shader_return_valid != 1 ||
      semantic_plan.owner.lane_id != lane_id) {
    return kStatusInvalidArgument;
  }
  const uint32_t bit = lane_bit(lane_id);
  const bool terminal_return =
      (state->pending_shader_terminal_mask & bit) != 0;
  if ((state->pending_next_active_mask & bit) == 0 ||
      state->lanes[lane_id].state != kLaneWaitingShader ||
      semantic_plan.route_kind !=
          (terminal_return
               ? primitive_semantic::kRouteFinalHitBoundary
               : primitive_semantic::kRouteStackPopNext)) {
    return kStatusLaneNotWaitingShader;
  }
  if ((state->return_staged_mask & bit) != 0) {
    return kStatusLaneReturnAlreadyStaged;
  }
  state->lanes[lane_id].semantic_plan = semantic_plan;
  state->lanes[lane_id].return_staged = 1;
  state->return_staged_mask |= bit;
  return kStatusOk;
}

status_kind mark_lane_commit_enqueued(
    warp_state_v0 *state, uint8_t lane_id,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t successor_operation_seq) {
  if (state == NULL || lane_id >= 32 || producer_operation_seq == 0 ||
      commit_epoch == 0) {
    return kStatusInvalidArgument;
  }
  const uint32_t bit = lane_bit(lane_id);
  const bool terminal_return =
      (state->pending_shader_terminal_mask & bit) != 0;
  if ((!terminal_return &&
       (successor_operation_seq == 0 ||
        successor_operation_seq == producer_operation_seq)) ||
      (terminal_return && successor_operation_seq != 0)) {
    return kStatusInvalidArgument;
  }
  if ((state->return_staged_mask & bit) == 0) {
    return kStatusLaneReturnNotStaged;
  }
  if ((state->commit_enqueued_mask & bit) != 0) {
    return kStatusLaneCommitAlreadyEnqueued;
  }
  lane_state_v0 &lane = state->lanes[lane_id];
  lane.producer_operation_seq = producer_operation_seq;
  lane.commit_epoch = commit_epoch;
  lane.successor_operation_seq = successor_operation_seq;
  lane.commit_enqueued = 1;
  state->commit_enqueued_mask |= bit;
  return kStatusOk;
}

status_kind mark_lane_commit_ready(
    warp_state_v0 *state, uint8_t lane_id,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t successor_operation_seq) {
  if (state == NULL || lane_id >= 32) {
    return kStatusInvalidArgument;
  }
  const uint32_t bit = lane_bit(lane_id);
  const lane_state_v0 &lane = state->lanes[lane_id];
  if ((state->commit_enqueued_mask & bit) == 0 ||
      lane.producer_operation_seq != producer_operation_seq ||
      lane.commit_epoch != commit_epoch ||
      lane.successor_operation_seq != successor_operation_seq) {
    return kStatusLaneCommitNotEnqueued;
  }
  if ((state->commit_ready_mask & bit) != 0) {
    return kStatusLaneCommitAlreadyReady;
  }
  state->lanes[lane_id].commit_ready = 1;
  state->commit_ready_mask |= bit;
  return kStatusOk;
}

bool resubmit_ready(const warp_state_v0 &state) {
  return state.initialized == 1 && state.resubmit_pending == 1 &&
         state.return_staged_mask == state.pending_next_active_mask &&
         state.commit_enqueued_mask == state.pending_next_active_mask &&
         state.commit_ready_mask == state.pending_next_active_mask;
}

status_kind commit_resubmit(
    warp_state_v0 *state, uint32_t next_warp_uid) {
  if (state == NULL || next_warp_uid == 0 ||
      next_warp_uid == state->current_warp_uid ||
      next_warp_uid != state->pending_next_warp_uid) {
    return kStatusInvalidArgument;
  }
  if (!resubmit_ready(*state)) return kStatusResubmitNotReady;
  if (state->completion_transaction_generation == UINT32_MAX) {
    return kStatusInvalidArgument;
  }
  const uint32_t previous_active_mask = state->active_mask;
  const uint32_t next_active_mask = state->pending_next_active_mask;
  for (uint32_t lane = 0; lane < 32; ++lane) {
    const uint32_t bit = lane_bit(lane);
    if ((previous_active_mask & bit) == 0) continue;
    state->lanes[lane] = lane_state_v0();
    state->lanes[lane].state =
        (next_active_mask & bit) != 0
            ? kLaneActiveTraversal
            : kLaneReleased;
  }
  state->current_warp_uid = next_warp_uid;
  state->active_mask = next_active_mask;
  state->continuation_mask = 0;
  state->terminal_mask = 0;
  state->dispatch_pending_mask = 0;
  state->dispatch_completed_mask = 0;
  state->pending_next_warp_uid = 0;
  state->pending_next_static_inst_uid = 0;
  state->pending_next_active_mask = 0;
  state->pending_resume_mask = 0;
  state->pending_shader_terminal_mask = 0;
  state->pending_release_mask = 0;
  state->return_staged_mask = 0;
  state->commit_enqueued_mask = 0;
  state->commit_ready_mask = 0;
  state->completion_consumed = 0;
  state->resubmit_pending = 0;
  ++state->completion_transaction_generation;
  return kStatusOk;
}

status_kind prepare_release(const warp_state_v0 &state) {
  if (state.initialized != 1 || state.released != 0 ||
      state.completion_consumed != 1 || state.resubmit_pending != 0 ||
      state.active_mask == 0 || state.terminal_mask != state.active_mask ||
      state.continuation_mask != 0 || state.dispatch_pending_mask != 0 ||
      state.dispatch_completed_mask != state.active_mask) {
    return kStatusReleaseNotQuiescent;
  }
  for (uint32_t lane = 0; lane < 32; ++lane) {
    if ((state.active_mask & lane_bit(lane)) != 0 &&
        state.lanes[lane].state != kLaneFinalWaitRelease) {
      return kStatusReleaseNotQuiescent;
    }
  }
  return kStatusOk;
}

status_kind commit_release(warp_state_v0 *state) {
  if (state == NULL) return kStatusInvalidArgument;
  const status_kind status = prepare_release(*state);
  if (status != kStatusOk) return status;
  for (uint32_t lane = 0; lane < 32; ++lane) {
    if ((state->active_mask & lane_bit(lane)) != 0) {
      state->lanes[lane] = lane_state_v0();
      state->lanes[lane].state = kLaneReleased;
    }
  }
  state->active_mask = 0;
  state->released = 1;
  return kStatusOk;
}

status_kind prepare_shader_return_semantic_plan(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t reason,
    const std::array<uint32_t, abi_v04::kWordCount> &words,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    primitive_semantic::semantic_plan_v0 *semantic_plan) {
  if (semantic_plan == NULL || producer_operation_seq == 0 ||
      !owner_matches(owner, canonical_slot.owner)) {
    return kStatusInvalidArgument;
  }
  *semantic_plan = primitive_semantic::semantic_plan_v0();
  const abi_v04::shadow::shader_return_observation observation =
      abi_v04::shadow::decode_shader_return_words(words, reason);
  if (!observation.valid()) return kStatusShaderReturnRejected;

  private_frontier::retained_candidate_projection_v0 retained = {};
  if (private_frontier::decode_retained_candidate(
          canonical_slot, owner, &retained) !=
          private_frontier::kStatusOk ||
      !identity_matches_handoff(retained, reason, words)) {
    return kStatusRetainedCandidateMismatch;
  }

  primitive_semantic::semantic_plan_v0 prepared = {};
  prepared.owner = owner;
  prepared.ray_policy.ray_flags =
      abi_v04::extract_field(words, abi_v04::kRayFlags);
  prepared.ray_policy.cull_mask = static_cast<uint8_t>(
      abi_v04::extract_field(words, abi_v04::kCullMask));
  prepared.operation_seq = producer_operation_seq;
  const bool terminate_search =
      (observation.traversal_effect & (uint32_t{1} << 2)) != 0;
  prepared.route_kind =
      terminate_search ? primitive_semantic::kRouteFinalHitBoundary
                       : primitive_semantic::kRouteStackPopNext;
  prepared.shader_return_valid = 1;
  if (observation.update.action ==
      abi_v04::shadow::kBoundaryReturnCommitAnyHit) {
    prepared.committed_hit_valid = 1;
    prepared.committed_hit = triangle_committed_hit(retained);
  } else if (observation.update.action ==
             abi_v04::shadow::kBoundaryReturnCommitIntersection) {
    const float reported_t = fp32_value(observation.update.reported_t_fp32);
    const float boundary_tmax = fp32_value(
        abi_v04::extract_field(words, abi_v04::kBoundaryRayTmaxFp32));
    if (!std::isfinite(reported_t) || !std::isfinite(boundary_tmax) ||
        reported_t > boundary_tmax) {
      return kStatusShaderReturnRejected;
    }
    prepared.committed_hit_valid = 1;
    prepared.committed_hit =
        procedural_committed_hit(retained, observation.update);
  }
  prepared.valid = 1;
  *semantic_plan = prepared;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk: return "ok";
    case kStatusInvalidArgument: return "invalid_argument";
    case kStatusOwnerMismatch: return "owner_mismatch";
    case kStatusCompletionAlreadyConsumed:
      return "completion_already_consumed";
    case kStatusCompletionMaskMismatch:
      return "completion_mask_mismatch";
    case kStatusDispatchAlreadyCompleted:
      return "dispatch_already_completed";
    case kStatusDispatchNotComplete:
      return "dispatch_not_complete";
    case kStatusResubmitAlreadyPending:
      return "resubmit_already_pending";
    case kStatusInvalidResubmitMask: return "invalid_resubmit_mask";
    case kStatusLaneNotWaitingShader:
      return "lane_not_waiting_shader";
    case kStatusLaneReturnAlreadyStaged:
      return "lane_return_already_staged";
    case kStatusLaneReturnNotStaged:
      return "lane_return_not_staged";
    case kStatusLaneCommitAlreadyEnqueued:
      return "lane_commit_already_enqueued";
    case kStatusLaneCommitNotEnqueued:
      return "lane_commit_not_enqueued";
    case kStatusLaneCommitAlreadyReady:
      return "lane_commit_already_ready";
    case kStatusResubmitNotReady: return "resubmit_not_ready";
    case kStatusReleaseNotQuiescent: return "release_not_quiescent";
    case kStatusShaderReturnRejected: return "shader_return_rejected";
    case kStatusRetainedCandidateMismatch:
      return "retained_candidate_mismatch";
  }
  return "unknown";
}

const char *lane_state_name(lane_state_kind state) {
  switch (state) {
    case kLaneUnused: return "unused";
    case kLaneActiveTraversal: return "active_traversal";
    case kLaneWaitingShader: return "waiting_shader";
    case kLaneFinalWaitRelease: return "final_wait_release";
    case kLaneReleased: return "released";
  }
  return "unknown";
}

}  // namespace continuation_lifecycle
}  // namespace v04
}  // namespace rtcore
