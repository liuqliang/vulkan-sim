#include "rtcore_v04_stall_attribution.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rtcore_v04_request_owner_binding.h"

namespace rtcore {
namespace v04 {
namespace stall_attribution {
namespace {

bool canonical_true(const char *value) {
  return value != NULL &&
         (std::strcmp(value, "1") == 0 ||
          std::strcmp(value, "true") == 0 ||
          std::strcmp(value, "on") == 0 ||
          std::strcmp(value, "yes") == 0);
}

bool valid_unit(uint8_t unit) {
  return unit > kUnitInvalid && unit <= kUnitShortStack;
}

bool valid_stage(uint8_t stage) {
  return stage > kStageInvalid && stage <= kStageAdmission;
}

bool progress_action_matches_stage(uint8_t action, uint8_t stage) {
  switch (static_cast<action_kind>(action)) {
    case kActionIssue:
      return stage == kStageIssue;
    case kActionCapture:
      return stage == kStageCapture;
    case kActionCommit:
      return stage == kStageCommit;
    case kActionInputPrepareAccept:
      return stage == kStageInputPrepare;
    case kActionFrontendAccept:
      return stage == kStageFrontend;
    case kActionMemoryAccept:
      return stage == kStageBackend;
    case kActionAdmissionAccept:
      return stage == kStageAdmission;
    case kActionNone:
      break;
  }
  return false;
}

bool stall_reason_matches_stage(uint8_t reason, uint8_t stage) {
  switch (static_cast<reason_kind>(reason)) {
    case kReasonResultSinkCapacity:
      return stage == kStageCapture || stage == kStageCommit;
    case kReasonResultCommitCapacity:
      return stage == kStageCapture;
    case kReasonPipelineCapacity:
    case kReasonUnitBusy:
      return stage == kStageIssue;
    case kReasonInputProvider:
      return stage == kStageInputPrepare;
    case kReasonIssueBudget:
    case kReasonFrontendBudget:
    case kReasonL1dRequestArbiter:
      return stage == kStageFrontend;
    case kReasonL1dReservation:
    case kReasonL1dDataPort:
      return stage == kStageBackend;
    case kReasonReservationBudget:
    case kReasonQueueCapacity:
      return stage == kStageAdmission;
    case kReasonNone:
    case kReasonInvalidOffer:
      break;
  }
  return false;
}

}  // namespace

bool enabled() {
  static const bool value = canonical_true(
      std::getenv("VULKAN_SIM_RTCORE_V04_EXCLUSIVE_STALL_ACCOUNTING"));
  return value;
}

bool valid_attempt(const attempt_record_v0 &record) {
  request_owner::internal_request_key_fields_v0 fields = {};
  if (!valid_unit(record.unit) || !valid_stage(record.stage) ||
      record.request_identity == 0 || record.request_generation == 0 ||
      record.operation_seq == 0 || record.chunk_count == 0 ||
      record.chunk_id >= record.chunk_count || record.lane_id >= 32 ||
      request_owner::unpack_internal_request_key(
          record.request_identity, &fields) != request_owner::kStatusOk ||
      fields.request_generation != record.request_generation ||
      fields.lane_id != record.lane_id) {
    return false;
  }
  if (record.outcome == kOutcomeProgress) {
    return record.reason == kReasonNone &&
           progress_action_matches_stage(record.action, record.stage);
  }
  if (record.outcome == kOutcomeStall) {
    return record.action == kActionNone &&
           stall_reason_matches_stage(record.reason, record.stage);
  }
  return false;
}

void emit_attempt(const attempt_record_v0 &record) {
  if (!enabled()) return;
  std::printf(
      "GPGPU-Sim RTCORE_V04_EXCLUSIVE_ATTEMPT_ACCOUNTING "
      "schema=3 owner_hw_sid=%u service_cycle=%llu unit=%s stage=%s "
      "arbitration_slot=%u request_identity=%u request_generation=%u "
      "operation_seq=%u chunk_id=%u chunk_count=%u lane_id=%u "
      "outcome=%s action=%s reason=%s "
      "record_valid=%u\n",
      record.owner_hw_sid,
      static_cast<unsigned long long>(record.service_cycle),
      unit_name(static_cast<unit_kind>(record.unit)),
      stage_name(static_cast<stage_kind>(record.stage)),
      record.arbitration_slot, record.request_identity,
      record.request_generation, record.operation_seq, record.chunk_id,
      record.chunk_count, record.lane_id,
      outcome_name(static_cast<outcome_kind>(record.outcome)),
      action_name(static_cast<action_kind>(record.action)),
      reason_name(static_cast<reason_kind>(record.reason)),
      valid_attempt(record) ? 1u : 0u);
  std::fflush(stdout);
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
    case kUnitMemory:
      return "memory";
    case kUnitShortStack:
      return "short_stack";
    case kUnitInvalid:
      break;
  }
  return "invalid";
}

const char *stage_name(stage_kind stage) {
  switch (stage) {
    case kStageIssue:
      return "issue";
    case kStageCapture:
      return "capture";
    case kStageCommit:
      return "commit";
    case kStageInputPrepare:
      return "input_prepare";
    case kStageFrontend:
      return "frontend";
    case kStageBackend:
      return "backend";
    case kStageAdmission:
      return "admission";
    case kStageInvalid:
      break;
  }
  return "invalid";
}

const char *outcome_name(outcome_kind outcome) {
  switch (outcome) {
    case kOutcomeProgress:
      return "progress";
    case kOutcomeStall:
      return "stall";
    case kOutcomeInvalid:
      break;
  }
  return "invalid";
}

const char *action_name(action_kind action) {
  switch (action) {
    case kActionNone:
      return "none";
    case kActionIssue:
      return "issue";
    case kActionCapture:
      return "capture";
    case kActionCommit:
      return "commit";
    case kActionInputPrepareAccept:
      return "input_prepare_accept";
    case kActionFrontendAccept:
      return "frontend_accept";
    case kActionMemoryAccept:
      return "memory_accept";
    case kActionAdmissionAccept:
      return "admission_accept";
  }
  return "invalid";
}

const char *reason_name(reason_kind reason) {
  switch (reason) {
    case kReasonNone:
      return "none";
    case kReasonResultSinkCapacity:
      return "result_sink_capacity";
    case kReasonResultCommitCapacity:
      return "result_commit_capacity";
    case kReasonPipelineCapacity:
      return "pipeline_capacity";
    case kReasonUnitBusy:
      return "unit_busy";
    case kReasonInputProvider:
      return "input_provider";
    case kReasonIssueBudget:
      return "issue_budget";
    case kReasonFrontendBudget:
      return "frontend_budget";
    case kReasonL1dReservation:
      return "l1d_reservation";
    case kReasonReservationBudget:
      return "reservation_budget";
    case kReasonQueueCapacity:
      return "queue_capacity";
    case kReasonInvalidOffer:
      return "invalid_offer";
    case kReasonL1dDataPort:
      return "l1d_data_port";
    case kReasonL1dRequestArbiter:
      return "l1d_request_arbiter";
  }
  return "invalid";
}

}  // namespace stall_attribution
}  // namespace v04
}  // namespace rtcore
