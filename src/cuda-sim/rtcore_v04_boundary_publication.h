#ifndef RTCORE_V04_BOUNDARY_PUBLICATION_H
#define RTCORE_V04_BOUNDARY_PUBLICATION_H

#include <array>
#include <cstdint>

#include "rtcore_v04_primitive_result_semantic_applier.h"
#include "rtcore_v04_private_state_384_operand_materializer.h"
#include "rtcore_v04_request_owner_binding.h"
#include "rtcore_v04_shadow_boundary.h"

namespace rtcore {
namespace v04 {
namespace boundary_publication {

static const uint8_t kLaneCapacity = 32;
static const uint8_t kChunkBytes = 32;
static const uint8_t kChunkCount =
    static_cast<uint8_t>(abi_v04::kLaneSlotBytes / kChunkBytes);

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusOwnerMismatch,
  kStatusInvalidSemanticPlan,
  kStatusBoundaryEncodingRejected,
  kStatusLaneBusy,
  kStatusNoArmedPublication,
  kStatusAckIdentityMismatch,
  kStatusDuplicateAck,
  kStatusNoCompletionReady,
  kStatusCompletionAlreadyPublished,
  kStatusWarpCompletionNotReady,
  kStatusScoreboardAlreadyDelivered,
  kStatusCompletionAlreadyConsumed,
};

enum publication_route_kind : uint8_t {
  kPublicationRouteInvalid = 0,
  kPublicationRouteAnyHit = 2,
  kPublicationRouteIntersection = 3,
  kPublicationRouteFinalHit = 4,
  kPublicationRouteFinalMiss = 5,
};

enum terminal_kind : uint8_t {
  kTerminalInvalid = 0,
  kTerminalFinalHit = 1,
  kTerminalFinalMiss = 2,
};

struct lane_publication_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t arm_age;
  uint64_t arm_cycle;
  uint64_t final_ack_cycle;
  uint32_t producer_operation_seq;
  uint32_t commit_epoch;
  uint32_t reason;
  uint32_t compact_result;
  uint32_t written_word_mask;
  uint8_t route_kind;
  uint8_t pending_chunk_mask;
  uint8_t acknowledged_chunk_mask;
  uint8_t valid;
  uint8_t ack_complete;
  uint8_t completion_published;
  uint8_t reserved_zero[2];
  std::array<uint32_t, abi_v04::kWordCount> words;
  std::array<uint32_t, abi_v04::kWordCount> preimage_words;
};

struct warp_state_v0 {
  uint64_t next_arm_age;
  uint64_t scoreboard_handoff_cycle;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
  uint32_t publication_armed_mask;
  uint32_t publication_acked_mask;
  uint32_t completed_lane_mask;
  uint32_t result_valid_mask;
  uint32_t terminal_lane_mask;
  uint32_t continuation_lane_mask;
  uint8_t resident_warp_slot;
  uint8_t initialized;
  uint8_t scoreboard_handoff_ready;
  uint8_t scoreboard_handoff_delivered;
  uint8_t completion_consumed;
  uint8_t reserved_zero[3];
  lane_publication_v0 lanes[kLaneCapacity];
};

struct arm_receipt_v0 {
  uint32_t reason;
  uint32_t compact_result;
  uint32_t written_word_mask;
  uint8_t pending_chunk_mask;
  uint8_t lane_id;
  uint8_t valid;
  uint8_t reserved_zero;
};

struct ack_receipt_v0 {
  uint32_t completed_lane_mask;
  uint8_t lane_id;
  uint8_t acknowledged_chunk_mask;
  uint8_t lane_ack_complete;
  uint8_t valid;
};

struct completion_receipt_v0 {
  uint32_t completed_lane_mask;
  uint32_t result_valid_mask;
  uint32_t terminal_lane_mask;
  uint32_t continuation_lane_mask;
  uint8_t lane_id;
  uint8_t warp_completion_ready;
  uint8_t valid;
  uint8_t reserved_zero;
};

status_kind initialize(warp_state_v0 *state, uint32_t owner_hw_sid,
                       uint32_t warp_uid, uint32_t warp_id,
                       uint32_t active_mask, uint8_t resident_warp_slot);

status_kind arm_boundary(
    warp_state_v0 *state,
    const primitive_semantic::semantic_plan_v0 &semantic_plan,
    const std::array<uint32_t, abi_v04::kWordCount> &preimage_words,
    uint32_t commit_epoch, uint64_t arm_cycle, arm_receipt_v0 *receipt);

status_kind arm_private_state_384_boundary(
    warp_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq,
    const private_state_384::operand_materializer::software_boundary_v1
        &boundary,
    const std::array<uint32_t, abi_v04::kWordCount> &preimage_words,
    uint32_t commit_epoch, uint64_t arm_cycle,
    arm_receipt_v0 *receipt);

status_kind arm_terminal_boundary(
    warp_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, terminal_kind terminal,
    const typed_stack::committed_hit_projection_v0 &committed_hit,
    const std::array<uint32_t, abi_v04::kWordCount> &preimage_words,
    uint32_t commit_epoch, uint64_t arm_cycle, arm_receipt_v0 *receipt);

status_kind accept_chunk_ack(
    warp_state_v0 *state,
    const private_frontier::owner_binding_v0 &owner,
    uint32_t producer_operation_seq, uint32_t commit_epoch,
    uint8_t chunk_id, uint64_t response_cycle, ack_receipt_v0 *receipt);

status_kind select_oldest_completion_ready_lane(
    const warp_state_v0 &state, uint8_t *lane_id);

status_kind publish_lane_completion(warp_state_v0 *state, uint8_t lane_id,
                                    completion_receipt_v0 *receipt);

status_kind mark_scoreboard_handoff_delivered(warp_state_v0 *state,
                                              uint64_t service_cycle);

status_kind consume_completion_transaction(warp_state_v0 *state);

bool all_active_lanes_complete(const warp_state_v0 &state);
bool transaction_quiescent(const warp_state_v0 &state);

const char *status_name(status_kind status);

}  // namespace boundary_publication
}  // namespace v04
}  // namespace rtcore

#endif
