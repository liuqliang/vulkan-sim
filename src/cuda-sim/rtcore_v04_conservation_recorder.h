#ifndef RTCORE_V04_CONSERVATION_RECORDER_H
#define RTCORE_V04_CONSERVATION_RECORDER_H

#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"

namespace rtcore {
namespace v04 {
namespace conservation {

enum event_kind : uint8_t {
  kEventInvalid = 0,
  kEventTargetReserved,
  kEventMemoryRequest,
  kEventMemoryZero,
  kEventMemoryResponse,
  kEventTargetReady,
  kEventTypedOperation,
  kEventResultCommitBegin,
  kEventResultCommitAck,
  kEventResultCommitZero,
  kEventPublicationArm,
  kEventPublicationAck,
  kEventLaneCompletion,
  kEventWarpAdmit,
  kEventWarpCompletion,
  kEventCohortLaunch,
  kEventCohortComplete,
  kEventResubmit,
  kEventTerminalRelease,
  kEventRetire,
};

struct lane_event_v0 {
  private_frontier::owner_binding_v0 owner;
  uint64_t cycle;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint32_t transport_id;
  uint16_t item_id;
  uint16_t item_count;
  uint8_t event;
  uint8_t detail_kind;
  uint8_t reserved_zero[2];
};

struct warp_event_v0 {
  uint64_t cycle;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t resident_warp_slot;
  uint32_t resident_generation;
  uint32_t completion_transaction_generation;
  uint32_t active_mask;
  uint32_t lane_mask;
  uint32_t cohort_seq;
  uint8_t event;
  uint8_t reserved_zero[3];
};

bool enabled();
bool emit_lane_event(const lane_event_v0 &record);
bool emit_warp_event(const warp_event_v0 &record);

const char *event_name(event_kind event);

}  // namespace conservation
}  // namespace v04
}  // namespace rtcore

#endif
