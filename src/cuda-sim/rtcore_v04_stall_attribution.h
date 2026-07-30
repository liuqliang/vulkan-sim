#ifndef RTCORE_V04_STALL_ATTRIBUTION_H
#define RTCORE_V04_STALL_ATTRIBUTION_H

#include <cstdint>

namespace rtcore {
namespace v04 {
namespace stall_attribution {

enum unit_kind : uint8_t {
  kUnitInvalid = 0,
  kUnitNode,
  kUnitStack,
  kUnitInstance,
  kUnitPrimitive,
  kUnitMemory,
  kUnitShortStack,
};

enum stage_kind : uint8_t {
  kStageInvalid = 0,
  kStageIssue,
  kStageCapture,
  kStageCommit,
  kStageInputPrepare,
  kStageFrontend,
  kStageBackend,
  kStageAdmission,
};

enum outcome_kind : uint8_t {
  kOutcomeInvalid = 0,
  kOutcomeProgress,
  kOutcomeStall,
};

enum action_kind : uint8_t {
  kActionNone = 0,
  kActionIssue,
  kActionCapture,
  kActionCommit,
  kActionInputPrepareAccept,
  kActionFrontendAccept,
  kActionMemoryAccept,
  kActionAdmissionAccept,
};

enum reason_kind : uint8_t {
  kReasonNone = 0,
  kReasonResultSinkCapacity,
  kReasonResultCommitCapacity,
  kReasonPipelineCapacity,
  kReasonUnitBusy,
  kReasonInputProvider,
  kReasonIssueBudget,
  kReasonFrontendBudget,
  kReasonL1dReservation,
  kReasonReservationBudget,
  kReasonQueueCapacity,
  kReasonInvalidOffer,
};

struct attempt_record_v0 {
  uint64_t service_cycle;
  uint32_t owner_hw_sid;
  uint32_t request_identity;
  uint32_t request_generation;
  uint32_t operation_seq;
  uint32_t chunk_id;
  uint32_t chunk_count;
  uint16_t arbitration_slot;
  uint8_t lane_id;
  uint8_t unit;
  uint8_t stage;
  uint8_t outcome;
  uint8_t action;
  uint8_t reason;
};

bool enabled();
bool valid_attempt(const attempt_record_v0 &record);
void emit_attempt(const attempt_record_v0 &record);

const char *unit_name(unit_kind unit);
const char *stage_name(stage_kind stage);
const char *outcome_name(outcome_kind outcome);
const char *action_name(action_kind action);
const char *reason_name(reason_kind reason);

}  // namespace stall_attribution
}  // namespace v04
}  // namespace rtcore

#endif
