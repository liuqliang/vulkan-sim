#ifndef RTCORE_V04_CONTINUATION_LIFECYCLE_H
#define RTCORE_V04_CONTINUATION_LIFECYCLE_H

#include <array>
#include <cstdint>

#include "rtcore_v04_primitive_result_semantic_applier.h"
#include "rtcore_v04_shadow_shader_return.h"

namespace rtcore {
namespace v04 {
namespace continuation_lifecycle {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusOwnerMismatch,
  kStatusCompletionAlreadyConsumed,
  kStatusCompletionMaskMismatch,
  kStatusDispatchAlreadyCompleted,
  kStatusDispatchNotComplete,
  kStatusResubmitAlreadyPending,
  kStatusInvalidResubmitMask,
  kStatusLaneNotWaitingShader,
  kStatusLaneReturnAlreadyStaged,
  kStatusLaneReturnNotStaged,
  kStatusLaneCommitAlreadyEnqueued,
  kStatusLaneCommitNotEnqueued,
  kStatusLaneCommitAlreadyReady,
  kStatusResubmitNotReady,
  kStatusReleaseNotQuiescent,
  kStatusShaderReturnRejected,
  kStatusRetainedCandidateMismatch,
};

enum lane_state_kind : uint8_t {
  kLaneUnused = 0,
  kLaneActiveTraversal,
  kLaneWaitingShader,
  kLaneFinalWaitRelease,
  kLaneReleased,
};

struct lane_state_v0 {
  uint32_t producer_operation_seq;
  uint32_t commit_epoch;
  uint32_t successor_operation_seq;
  uint8_t state;
  uint8_t return_staged;
  uint8_t commit_enqueued;
  uint8_t commit_ready;
  primitive_semantic::semantic_plan_v0 semantic_plan;
};

struct warp_state_v0 {
  uint32_t owner_hw_sid;
  uint32_t current_warp_uid;
  uint32_t warp_id;
  uint32_t resident_generation;
  uint32_t active_mask;
  uint32_t continuation_mask;
  uint32_t terminal_mask;
  uint32_t dispatch_pending_mask;
  uint32_t dispatch_completed_mask;
  uint32_t pending_next_warp_uid;
  uint32_t pending_next_static_inst_uid;
  uint32_t pending_next_active_mask;
  uint32_t pending_resume_mask;
  uint32_t pending_shader_terminal_mask;
  uint32_t pending_release_mask;
  uint32_t return_staged_mask;
  uint32_t commit_enqueued_mask;
  uint32_t commit_ready_mask;
  uint32_t completion_transaction_generation;
  uint8_t initialized;
  uint8_t completion_consumed;
  uint8_t resubmit_pending;
  uint8_t released;
  lane_state_v0 lanes[32];
};

status_kind initialize(
    warp_state_v0 *state, uint32_t owner_hw_sid, uint32_t warp_uid,
    uint32_t warp_id, uint32_t resident_generation,
    uint32_t active_mask);

status_kind consume_completion(
    warp_state_v0 *state, uint32_t continuation_mask,
    uint32_t terminal_mask);

status_kind mark_dispatch_complete(
    warp_state_v0 *state, uint32_t completed_lane_mask);

status_kind stage_resubmit(
    warp_state_v0 *state, uint32_t next_warp_uid,
    uint32_t next_static_inst_uid, uint32_t next_active_mask,
    uint32_t resume_mask, uint32_t shader_terminal_mask);

status_kind stage_lane_return(
    warp_state_v0 *state, uint8_t lane_id,
    const primitive_semantic::semantic_plan_v0 &semantic_plan);

status_kind mark_lane_commit_enqueued(
    warp_state_v0 *state, uint8_t lane_id,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t successor_operation_seq);

status_kind mark_lane_commit_ready(
    warp_state_v0 *state, uint8_t lane_id,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint32_t successor_operation_seq);

bool resubmit_ready(const warp_state_v0 &state);

status_kind commit_resubmit(
    warp_state_v0 *state, uint32_t next_warp_uid);

status_kind prepare_release(const warp_state_v0 &state);
status_kind commit_release(warp_state_v0 *state);

status_kind prepare_shader_return_semantic_plan(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t reason,
    const std::array<uint32_t, abi_v04::kWordCount> &words,
    const private_frontier::shadow_slot_v0 &canonical_slot,
    primitive_semantic::semantic_plan_v0 *semantic_plan);

const char *status_name(status_kind status);
const char *lane_state_name(lane_state_kind state);

}  // namespace continuation_lifecycle
}  // namespace v04
}  // namespace rtcore

#endif
