#include "rtcore_v04_conservation_recorder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

#include "rtcore_v04_request_owner_binding.h"

namespace rtcore {
namespace v04 {
namespace conservation {
namespace {

struct lane_operation_key {
  uint32_t owner_hw_sid;
  uint32_t resident_warp_slot;
  uint32_t request_identity;
  uint32_t request_generation;
  uint32_t private_slot_id;
  uint32_t lane_id;
  uint32_t operation_seq;

  bool operator<(const lane_operation_key &other) const {
    if (owner_hw_sid != other.owner_hw_sid)
      return owner_hw_sid < other.owner_hw_sid;
    if (resident_warp_slot != other.resident_warp_slot)
      return resident_warp_slot < other.resident_warp_slot;
    if (request_identity != other.request_identity)
      return request_identity < other.request_identity;
    if (request_generation != other.request_generation)
      return request_generation < other.request_generation;
    if (private_slot_id != other.private_slot_id)
      return private_slot_id < other.private_slot_id;
    if (lane_id != other.lane_id) return lane_id < other.lane_id;
    return operation_seq < other.operation_seq;
  }
};

std::map<lane_operation_key, uint8_t> g_target_reservations;
std::set<lane_operation_key> g_target_ready;

lane_operation_key make_lane_operation_key(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq) {
  lane_operation_key key = {};
  key.owner_hw_sid = owner.owner_hw_sid;
  key.resident_warp_slot = owner.resident_warp_id;
  key.request_identity = owner.request_identity;
  key.request_generation = owner.generation;
  key.private_slot_id = owner.private_slot_id;
  key.lane_id = owner.lane_id;
  key.operation_seq = operation_seq;
  return key;
}

bool env_value_is_true(const char *value) {
  return value != NULL &&
         (std::strcmp(value, "1") == 0 ||
          std::strcmp(value, "true") == 0 ||
          std::strcmp(value, "on") == 0 ||
          std::strcmp(value, "yes") == 0);
}

bool bytes_are_zero(const uint8_t *bytes, size_t byte_count) {
  for (size_t index = 0; index < byte_count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

bool lane_event_kind_valid(uint8_t event) {
  return event >= kEventTargetReserved &&
         event <= kEventLaneCompletion;
}

bool warp_event_kind_valid(uint8_t event) {
  return event >= kEventWarpAdmit && event <= kEventRetire;
}

bool item_shape_valid(const lane_event_v0 &record) {
  const bool itemized =
      record.event == kEventMemoryRequest ||
      record.event == kEventMemoryResponse ||
      record.event == kEventResultCommitBegin ||
      record.event == kEventResultCommitAck ||
      record.event == kEventPublicationArm ||
      record.event == kEventPublicationAck;
  if (record.event == kEventPublicationArm &&
      record.item_count == 0) {
    return record.item_id == 0;
  }
  if (!itemized) {
    return record.item_id == 0 && record.item_count == 0;
  }
  return record.item_count != 0 && record.item_id < record.item_count;
}

}  // namespace

bool enabled() {
  static const bool value = env_value_is_true(
      std::getenv("VULKAN_SIM_RTCORE_ABI_V04_CONSERVATION_RECORDS"));
  return value;
}

bool emit_lane_event(const lane_event_v0 &record) {
  if (!enabled()) return true;
  if (!lane_event_kind_valid(record.event) ||
      !request_owner::validate_private_frontier_owner_identity(
          record.owner) ||
      record.operation_seq == 0 ||
      !bytes_are_zero(record.reserved_zero,
                      sizeof(record.reserved_zero)) ||
      !item_shape_valid(record)) {
    return false;
  }

  const lane_operation_key key =
      make_lane_operation_key(record.owner, record.operation_seq);
  if (record.event == kEventTargetReserved) {
    if (!g_target_reservations
             .insert(std::make_pair(key, record.detail_kind))
             .second) {
      return false;
    }
  } else if (record.event == kEventTargetReady) {
    if (g_target_reservations.find(key) ==
            g_target_reservations.end() ||
        !g_target_ready.insert(key).second) {
      return false;
    }
  }

  std::printf(
      "GPGPU-Sim RTCORE_V04_CONSERVATION "
      "schema=1 scope=lane event=%s owner_hw_sid=%u "
      "resident_warp_slot=%u request_identity=%u "
      "request_generation=%u private_slot_id=%u lane_id=%u "
      "operation_seq=%u commit_epoch=%u transport_id=%u "
      "item_id=%u item_count=%u detail_kind=%u cycle=%llu\n",
      event_name(static_cast<event_kind>(record.event)),
      record.owner.owner_hw_sid, record.owner.resident_warp_id,
      record.owner.request_identity, record.owner.generation,
      record.owner.private_slot_id, record.owner.lane_id,
      record.operation_seq, record.commit_epoch, record.transport_id,
      record.item_id, record.item_count, record.detail_kind,
      static_cast<unsigned long long>(record.cycle));
  std::fflush(stdout);
  return true;
}

bool ensure_target_ready_before_operation(
    const private_frontier::owner_binding_v0 &owner,
    uint32_t operation_seq) {
  if (!enabled()) return true;
  if (!request_owner::validate_private_frontier_owner_identity(owner) ||
      operation_seq == 0) {
    return false;
  }
  const lane_operation_key key =
      make_lane_operation_key(owner, operation_seq);
  const std::map<lane_operation_key, uint8_t>::const_iterator
      reservation = g_target_reservations.find(key);
  if (reservation == g_target_reservations.end() ||
      g_target_ready.find(key) != g_target_ready.end()) {
    return true;
  }
  lane_event_v0 ready = {};
  ready.owner = owner;
  ready.operation_seq = operation_seq;
  ready.event = kEventTargetReady;
  ready.detail_kind = reservation->second;
  return emit_lane_event(ready);
}

bool emit_warp_event(const warp_event_v0 &record) {
  if (!enabled()) return true;
  const bool cohort_event =
      record.event == kEventCohortLaunch ||
      record.event == kEventCohortComplete;
  if (!warp_event_kind_valid(record.event) ||
      record.resident_generation == 0 ||
      record.completion_transaction_generation == 0 ||
      record.active_mask == 0 ||
      (record.lane_mask & ~record.active_mask) != 0 ||
      (cohort_event != (record.cohort_seq != 0)) ||
      !bytes_are_zero(record.reserved_zero,
                      sizeof(record.reserved_zero))) {
    return false;
  }

  std::printf(
      "GPGPU-Sim RTCORE_V04_CONSERVATION "
      "schema=1 scope=warp event=%s owner_hw_sid=%u warp_uid=%u "
      "warp_id=%u resident_warp_slot=%u resident_generation=%u "
      "completion_transaction_generation=%u active_mask=0x%08x "
      "lane_mask=0x%08x cohort_seq=%u cycle=%llu\n",
      event_name(static_cast<event_kind>(record.event)),
      record.owner_hw_sid, record.warp_uid, record.warp_id,
      record.resident_warp_slot, record.resident_generation,
      record.completion_transaction_generation, record.active_mask,
      record.lane_mask, record.cohort_seq,
      static_cast<unsigned long long>(record.cycle));
  std::fflush(stdout);
  return true;
}

const char *event_name(event_kind event) {
  switch (event) {
    case kEventTargetReserved:
      return "target_reserved";
    case kEventMemoryRequest:
      return "memory_request";
    case kEventMemoryZero:
      return "memory_zero";
    case kEventMemoryResponse:
      return "memory_response";
    case kEventTargetReady:
      return "target_ready";
    case kEventTypedOperation:
      return "typed_operation";
    case kEventResultCommitBegin:
      return "result_commit_begin";
    case kEventResultCommitAck:
      return "result_commit_ack";
    case kEventResultCommitZero:
      return "result_commit_zero";
    case kEventPublicationArm:
      return "publication_arm";
    case kEventPublicationAck:
      return "publication_ack";
    case kEventLaneCompletion:
      return "lane_completion";
    case kEventWarpAdmit:
      return "warp_admit";
    case kEventWarpCompletion:
      return "warp_completion";
    case kEventCohortLaunch:
      return "cohort_launch";
    case kEventCohortComplete:
      return "cohort_complete";
    case kEventResubmit:
      return "resubmit";
    case kEventTerminalRelease:
      return "terminal_release";
    case kEventRetire:
      return "retire";
    case kEventInvalid:
      break;
  }
  return "invalid";
}

}  // namespace conservation
}  // namespace v04
}  // namespace rtcore
