#include "rtcore_v04_allocation_identity_authority.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace allocation_identity {
namespace {

static bool same_owner(const owner_v0 &lhs, const owner_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.dynamic_warp_id == rhs.dynamic_warp_id &&
         lhs.warp_id == rhs.warp_id;
}

static bool same_slot(const allocation_slot_v0 &lhs,
                      const allocation_slot_v0 &rhs) {
  return lhs.allocation_domain_id == rhs.allocation_domain_id &&
         lhs.allocation_slot_id == rhs.allocation_slot_id;
}

static bool same_ranges(const allocation_ranges_v0 &lhs,
                        const allocation_ranges_v0 &rhs) {
  return lhs.context_base == rhs.context_base &&
         lhs.context_byte_count == rhs.context_byte_count &&
         lhs.handoff_base == rhs.handoff_base &&
         lhs.handoff_byte_count == rhs.handoff_byte_count;
}

static bool checked_end(uint64_t base, uint64_t byte_count, uint64_t *end) {
  if (end == NULL || byte_count == 0 ||
      base > std::numeric_limits<uint64_t>::max() - byte_count) {
    return false;
  }
  *end = base + byte_count;
  return true;
}

static bool ranges_overlap(uint64_t lhs_base, uint64_t lhs_bytes,
                           uint64_t rhs_base, uint64_t rhs_bytes) {
  uint64_t lhs_end = 0;
  uint64_t rhs_end = 0;
  return checked_end(lhs_base, lhs_bytes, &lhs_end) &&
         checked_end(rhs_base, rhs_bytes, &rhs_end) &&
         lhs_base < rhs_end && rhs_base < lhs_end;
}

static status_kind validate_ranges(const allocation_ranges_v0 &ranges) {
  uint64_t context_end = 0;
  uint64_t handoff_end = 0;
  if (ranges.context_byte_count == 0 || ranges.handoff_byte_count == 0 ||
      (ranges.context_base % kAddressAlignment) != 0 ||
      (ranges.context_byte_count % kAddressAlignment) != 0 ||
      (ranges.handoff_base % kAddressAlignment) != 0 ||
      (ranges.handoff_byte_count % kAddressAlignment) != 0) {
    return kStatusInvalidRange;
  }
  if (!checked_end(ranges.context_base, ranges.context_byte_count,
                   &context_end) ||
      !checked_end(ranges.handoff_base, ranges.handoff_byte_count,
                   &handoff_end)) {
    return kStatusAddressOverflow;
  }
  if (ranges_overlap(ranges.context_base, ranges.context_byte_count,
                     ranges.handoff_base, ranges.handoff_byte_count)) {
    return kStatusRangeOverlap;
  }
  return kStatusOk;
}

static uint32_t lane_mask(uint8_t lane_id) {
  return uint32_t{1} << lane_id;
}

static bool same_identity(const allocation_identity_v0 &lhs,
                          const allocation_identity_v0 &rhs) {
  return lhs.record_id == rhs.record_id &&
         same_slot(lhs.slot, rhs.slot) &&
         same_owner(lhs.owner, rhs.owner) &&
         same_ranges(lhs.ranges, rhs.ranges) &&
         lhs.active_mask == rhs.active_mask &&
         lhs.launch_allocation_generation ==
             rhs.launch_allocation_generation &&
         lhs.window_generation == rhs.window_generation;
}

static bool next_generation(uint32_t previous, uint32_t first,
                            uint32_t *next) {
  static const uint32_t kGenerationStride = 2;
  if (next == NULL || first == 0 || first > kGenerationStride) {
    return false;
  }
  if (previous == 0) {
    *next = first;
    return true;
  }
  if (previous > std::numeric_limits<uint32_t>::max() -
                     kGenerationStride) {
    return false;
  }
  *next = previous + kGenerationStride;
  return *next != 0;
}

}  // namespace

bool authority_v0::allocation_key_v0::operator<(
    const allocation_key_v0 &other) const {
  if (allocation_domain_id != other.allocation_domain_id) {
    return allocation_domain_id < other.allocation_domain_id;
  }
  return allocation_slot_id < other.allocation_slot_id;
}

authority_v0::authority_v0() : next_record_id_(1), histories_(), records_() {}

void authority_v0::reset() { records_.clear(); }

status_kind authority_v0::observe_publication(
    const publication_request_v0 &request,
    publication_observation_v0 *observation) {
  if (observation == NULL || request.active_mask == 0 ||
      request.lane_id >= kLaneCapacity ||
      request.slot.allocation_domain_id == 0) {
    return kStatusInvalidArgument;
  }
  std::memset(observation, 0, sizeof(*observation));
  if ((request.active_mask & lane_mask(request.lane_id)) == 0) {
    return kStatusLaneNotActive;
  }
  const status_kind range_status = validate_ranges(request.ranges);
  if (range_status != kStatusOk) return range_status;

  allocation_key_v0 key = {};
  key.allocation_domain_id = request.slot.allocation_domain_id;
  key.allocation_slot_id = request.slot.allocation_slot_id;

  std::map<allocation_key_v0, active_record_v0>::iterator found =
      records_.find(key);
  if (found != records_.end()) {
    active_record_v0 &record = found->second;
    if (record.phase != kPhaseAllocated) return kStatusWrongPhase;
    if (!same_ranges(record.identity.ranges, request.ranges)) {
      return kStatusIdentityMismatch;
    }
    if (!same_owner(record.identity.owner, request.owner)) {
      return kStatusOwnerMismatch;
    }
    if (record.identity.active_mask != request.active_mask) {
      return kStatusActiveMaskMismatch;
    }
    const uint32_t bit = lane_mask(request.lane_id);
    observation->identity = record.identity;
    observation->lane_was_new =
        (record.published_lane_mask & bit) == 0 ? 1 : 0;
    record.published_lane_mask |= bit;
    observation->published_lane_mask = record.published_lane_mask;
    observation->publication_complete =
        record.published_lane_mask == record.identity.active_mask ? 1 : 0;
    return kStatusOk;
  }

  for (std::map<allocation_key_v0, active_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const allocation_ranges_v0 &active = it->second.identity.ranges;
    if (ranges_overlap(request.ranges.context_base,
                       request.ranges.context_byte_count,
                       active.context_base, active.context_byte_count) ||
        ranges_overlap(request.ranges.context_base,
                       request.ranges.context_byte_count,
                       active.handoff_base, active.handoff_byte_count) ||
        ranges_overlap(request.ranges.handoff_base,
                       request.ranges.handoff_byte_count,
                       active.context_base, active.context_byte_count) ||
        ranges_overlap(request.ranges.handoff_base,
                       request.ranges.handoff_byte_count,
                       active.handoff_base, active.handoff_byte_count)) {
      return kStatusAllocationBusy;
    }
  }

  std::map<allocation_key_v0, generation_history_v0>::iterator history_found =
      histories_.find(key);
  generation_history_v0 history = {};
  if (history_found != histories_.end()) history = history_found->second;

  uint32_t launch_generation = 0;
  uint32_t window_generation = 0;
  if (!next_generation(history.last_launch_allocation_generation, 1,
                       &launch_generation) ||
      !next_generation(history.last_window_generation, 2,
                       &window_generation)) {
    return kStatusGenerationExhausted;
  }
  if (next_record_id_ == 0 ||
      next_record_id_ == std::numeric_limits<uint64_t>::max()) {
    return kStatusRecordIdExhausted;
  }

  active_record_v0 record = {};
  record.phase = kPhaseAllocated;
  record.identity.record_id = next_record_id_++;
  record.identity.slot = request.slot;
  record.identity.owner = request.owner;
  record.identity.ranges = request.ranges;
  record.identity.active_mask = request.active_mask;
  record.identity.launch_allocation_generation = launch_generation;
  record.identity.window_generation = window_generation;
  record.published_lane_mask = lane_mask(request.lane_id);

  history.last_launch_allocation_generation = launch_generation;
  history.last_window_generation = window_generation;
  histories_[key] = history;
  records_[key] = record;

  observation->identity = record.identity;
  observation->published_lane_mask = record.published_lane_mask;
  observation->newly_allocated = 1;
  observation->lane_was_new = 1;
  observation->publication_complete =
      record.published_lane_mask == record.identity.active_mask ? 1 : 0;
  return kStatusOk;
}

status_kind authority_v0::lookup_for_submit(
    const allocation_slot_v0 &slot, const owner_v0 &owner,
    uint32_t active_mask,
    const allocation_ranges_v0 &ranges,
    allocation_identity_v0 *identity) const {
  if (identity == NULL || active_mask == 0 ||
      slot.allocation_domain_id == 0) {
    return kStatusInvalidArgument;
  }
  const status_kind range_status = validate_ranges(ranges);
  if (range_status != kStatusOk) return range_status;

  allocation_key_v0 key = {};
  key.allocation_domain_id = slot.allocation_domain_id;
  key.allocation_slot_id = slot.allocation_slot_id;
  std::map<allocation_key_v0, active_record_v0>::const_iterator found =
      records_.find(key);
  if (found == records_.end()) return kStatusRecordNotFound;
  const active_record_v0 &record = found->second;
  if (record.phase != kPhaseAllocated) return kStatusWrongPhase;
  if (!same_ranges(record.identity.ranges, ranges)) {
    return kStatusIdentityMismatch;
  }
  if (!same_owner(record.identity.owner, owner)) return kStatusOwnerMismatch;
  if (record.identity.active_mask != active_mask) {
    return kStatusActiveMaskMismatch;
  }
  if (record.published_lane_mask != record.identity.active_mask) {
    return kStatusPublicationIncomplete;
  }
  *identity = record.identity;
  return kStatusOk;
}

status_kind authority_v0::begin_release(
    const allocation_identity_v0 &identity) {
  const status_kind range_status = validate_ranges(identity.ranges);
  if (range_status != kStatusOk) return range_status;
  allocation_key_v0 key = {};
  key.allocation_domain_id = identity.slot.allocation_domain_id;
  key.allocation_slot_id = identity.slot.allocation_slot_id;
  std::map<allocation_key_v0, active_record_v0>::iterator found =
      records_.find(key);
  if (found == records_.end()) return kStatusRecordNotFound;
  if (!same_identity(found->second.identity, identity)) {
    return kStatusIdentityMismatch;
  }
  if (found->second.phase != kPhaseAllocated) return kStatusWrongPhase;
  found->second.phase = kPhaseReleasePending;
  return kStatusOk;
}

status_kind authority_v0::commit_release(
    const allocation_identity_v0 &identity) {
  const status_kind range_status = validate_ranges(identity.ranges);
  if (range_status != kStatusOk) return range_status;
  allocation_key_v0 key = {};
  key.allocation_domain_id = identity.slot.allocation_domain_id;
  key.allocation_slot_id = identity.slot.allocation_slot_id;
  std::map<allocation_key_v0, active_record_v0>::iterator found =
      records_.find(key);
  if (found == records_.end()) return kStatusRecordNotFound;
  if (!same_identity(found->second.identity, identity)) {
    return kStatusIdentityMismatch;
  }
  if (found->second.phase != kPhaseReleasePending) return kStatusWrongPhase;
  records_.erase(found);
  return kStatusOk;
}

bool authority_v0::ranges_available(
    const allocation_ranges_v0 &ranges) const {
  if (validate_ranges(ranges) != kStatusOk) return false;
  for (std::map<allocation_key_v0, active_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const allocation_ranges_v0 &active = it->second.identity.ranges;
    if (ranges_overlap(ranges.context_base, ranges.context_byte_count,
                       active.context_base, active.context_byte_count) ||
        ranges_overlap(ranges.context_base, ranges.context_byte_count,
                       active.handoff_base, active.handoff_byte_count) ||
        ranges_overlap(ranges.handoff_base, ranges.handoff_byte_count,
                       active.context_base, active.context_byte_count) ||
        ranges_overlap(ranges.handoff_base, ranges.handoff_byte_count,
                       active.handoff_base, active.handoff_byte_count)) {
      return false;
    }
  }
  return true;
}

authority_snapshot_v0 authority_v0::snapshot() const {
  authority_snapshot_v0 result = {};
  result.active_record_count = records_.size();
  result.generation_history_count = histories_.size();
  for (std::map<allocation_key_v0, active_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    if (it->second.phase == kPhaseAllocated) {
      ++result.allocated_count;
    } else if (it->second.phase == kPhaseReleasePending) {
      ++result.release_pending_count;
    }
  }
  return result;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidRange:
      return "invalid_range";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusRangeOverlap:
      return "range_overlap";
    case kStatusAllocationBusy:
      return "allocation_busy";
    case kStatusRecordNotFound:
      return "record_not_found";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusActiveMaskMismatch:
      return "active_mask_mismatch";
    case kStatusLaneNotActive:
      return "lane_not_active";
    case kStatusPublicationIncomplete:
      return "publication_incomplete";
    case kStatusIdentityMismatch:
      return "identity_mismatch";
    case kStatusWrongPhase:
      return "wrong_phase";
    case kStatusGenerationExhausted:
      return "generation_exhausted";
    case kStatusRecordIdExhausted:
      return "record_id_exhausted";
  }
  return "unknown";
}

const char *phase_name(phase_kind phase) {
  switch (phase) {
    case kPhaseAllocated:
      return "allocated";
    case kPhaseReleasePending:
      return "release_pending";
    case kPhaseInvalid:
      return "invalid";
  }
  return "unknown";
}

}  // namespace allocation_identity
}  // namespace v04
}  // namespace rtcore
