#include "rtcore_v04_address_range_registry.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace address_range_registry {
namespace {

uint32_t lane_bit(uint32_t lane) {
  return lane < kLaneCapacity ? uint32_t{1} << lane : 0;
}

uint32_t count_lanes(uint32_t mask) {
  uint32_t count = 0;
  while (mask != 0) {
    count += mask & 1u;
    mask >>= 1;
  }
  return count;
}

bool checked_range_end(uint64_t base, uint64_t byte_count,
                       uint64_t *end) {
  if (end == NULL || byte_count == 0 ||
      byte_count > std::numeric_limits<uint64_t>::max() - base) {
    return false;
  }
  *end = base + byte_count;
  return true;
}

bool ranges_overlap(uint64_t lhs_base, uint64_t lhs_end,
                    uint64_t rhs_base, uint64_t rhs_end) {
  return lhs_base < rhs_end && rhs_base < lhs_end;
}

bool same_provisional_owner(const provisional_owner_v0 &lhs,
                            const provisional_owner_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.warp_uid == rhs.warp_uid &&
         lhs.dynamic_warp_id == rhs.dynamic_warp_id &&
         lhs.launch_allocation_generation ==
             rhs.launch_allocation_generation &&
         lhs.window_generation == rhs.window_generation;
}

bool same_live_owner(const live_owner_v0 &lhs,
                     const live_owner_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.resident_warp_generation ==
             rhs.resident_warp_generation &&
         lhs.window_generation == rhs.window_generation;
}

bool valid_provisional_owner(const provisional_owner_v0 &owner) {
  return owner.launch_allocation_generation != 0 &&
         owner.window_generation != 0;
}

bool valid_live_owner(const live_owner_v0 &owner) {
  return owner.resident_warp_generation != 0 &&
         owner.window_generation != 0;
}

bool valid_object(object_kind object) {
  return object == kObjectHandoff || object == kObjectContext;
}

bool access_matches_object(access_kind access, object_kind object) {
  if (object == kObjectHandoff) {
    return access >= kAccessHandoffShaderTraceInputPublish &&
           access <= kAccessHandoffShaderReturn;
  }
  if (object == kObjectContext) {
    return access == kAccessContextShaderRead ||
           access == kAccessContextShaderWrite;
  }
  return false;
}

bool valid_range_spec(const range_spec_v0 &range) {
  uint64_t end = 0;
  if (!valid_object(range.object) ||
      range.base % kAddressChunkBytes != 0 ||
      range.byte_count % kAddressChunkBytes != 0 ||
      !checked_range_end(range.base, range.byte_count, &end) ||
      range.lane_mask == 0 ||
      (range.lane_mask & (range.lane_mask - 1u)) != 0) {
    return false;
  }
  const size_t chunk_count =
      static_cast<size_t>(range.byte_count / kAddressChunkBytes);
  if (range.allowed_publication_mask_count != chunk_count ||
      (chunk_count != 0 &&
       range.allowed_publication_masks == NULL)) {
    return false;
  }
  bool saw_allowed_handoff_byte = false;
  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    if (range.object == kObjectContext &&
        range.allowed_publication_masks[chunk] != 0) {
      return false;
    }
    saw_allowed_handoff_byte |=
        range.allowed_publication_masks[chunk] != 0;
  }
  return range.object != kObjectHandoff ||
         saw_allowed_handoff_byte;
}

bool token_is_zero(const transaction_token_v0 &token) {
  const uint8_t *bytes =
      reinterpret_cast<const uint8_t *>(&token);
  for (size_t index = 0; index < sizeof(token); ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

}  // namespace

registry_v0::registry_v0()
    : next_record_id_(1), next_transaction_id_(1) {}

bool registry_v0::same_range_spec(
    const range_record_v0 &record,
    const range_spec_v0 &range) {
  if (record.base != range.base ||
      record.byte_count != range.byte_count ||
      record.lane_mask != range.lane_mask ||
      record.object != range.object ||
      record.allowed_publication_masks.size() !=
          range.allowed_publication_mask_count) {
    return false;
  }
  for (size_t chunk = 0;
       chunk < record.allowed_publication_masks.size(); ++chunk) {
    if (record.allowed_publication_masks[chunk] !=
        range.allowed_publication_masks[chunk]) {
      return false;
    }
  }
  return true;
}

void registry_v0::reset() {
  records_.clear();
  transactions_.clear();
}

status_kind registry_v0::register_provisional_group(
    const provisional_owner_v0 &owner, uint32_t active_mask,
    const range_spec_v0 *ranges, size_t range_count) {
  if (!valid_provisional_owner(owner) || active_mask == 0 ||
      ranges == NULL ||
      range_count != 2u * count_lanes(active_mask)) {
    return kStatusInvalidArgument;
  }
  if (next_record_id_ == 0 ||
      range_count >
          std::numeric_limits<uint64_t>::max() -
              next_record_id_ + 1u) {
    return kStatusRecordIdExhausted;
  }

  bool saw_handoff[kLaneCapacity] = {};
  bool saw_context[kLaneCapacity] = {};
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    if (same_provisional_owner(it->second.provisional_owner,
                               owner)) {
      return kStatusDuplicateOwner;
    }
  }

  for (size_t range_index = 0; range_index < range_count;
       ++range_index) {
    const range_spec_v0 &range = ranges[range_index];
    if (!valid_range_spec(range) ||
        (range.lane_mask & active_mask) != range.lane_mask) {
      return kStatusInvalidRange;
    }
    uint32_t lane = 0;
    while (lane < kLaneCapacity &&
           range.lane_mask != lane_bit(lane)) {
      ++lane;
    }
    if (lane == kLaneCapacity) return kStatusLaneMismatch;
    bool &seen = range.object == kObjectHandoff
                     ? saw_handoff[lane]
                     : saw_context[lane];
    if (seen) return kStatusDuplicateOwner;
    seen = true;

    uint64_t range_end = 0;
    if (!checked_range_end(range.base, range.byte_count,
                           &range_end)) {
      return kStatusAddressOverflow;
    }
    for (size_t prior = 0; prior < range_index; ++prior) {
      uint64_t prior_end = 0;
      if (!checked_range_end(ranges[prior].base,
                             ranges[prior].byte_count,
                             &prior_end)) {
        return kStatusAddressOverflow;
      }
      if (ranges_overlap(range.base, range_end,
                         ranges[prior].base, prior_end)) {
        return kStatusRangeOverlap;
      }
    }
    for (std::map<uint64_t, range_record_v0>::const_iterator it =
             records_.begin();
         it != records_.end(); ++it) {
      uint64_t existing_end = 0;
      if (!checked_range_end(it->second.base,
                             it->second.byte_count,
                             &existing_end)) {
        return kStatusAddressOverflow;
      }
      if (ranges_overlap(range.base, range_end,
                         it->second.base, existing_end)) {
        return kStatusRangeOverlap;
      }
    }
  }
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    const bool active = (active_mask & lane_bit(lane)) != 0;
    if (saw_handoff[lane] != active ||
        saw_context[lane] != active) {
      return kStatusInvalidRange;
    }
  }

  std::vector<range_record_v0> prepared;
  prepared.reserve(range_count);
  for (size_t range_index = 0; range_index < range_count;
       ++range_index) {
    range_record_v0 record = {};
    record.record_id = next_record_id_ + range_index;
    record.phase = kPhaseProvisional;
    record.provisional_owner = owner;
    record.base = ranges[range_index].base;
    record.byte_count = ranges[range_index].byte_count;
    record.lane_mask = ranges[range_index].lane_mask;
    record.object = ranges[range_index].object;
    record.active_mask = active_mask;
    record.allowed_publication_masks.assign(
        ranges[range_index].allowed_publication_masks,
        ranges[range_index].allowed_publication_masks +
            ranges[range_index].allowed_publication_mask_count);
    prepared.push_back(record);
  }
  for (size_t index = 0; index < prepared.size(); ++index) {
    records_[prepared[index].record_id] = prepared[index];
  }
  next_record_id_ += range_count;
  return kStatusOk;
}

status_kind registry_v0::accept_provisional_publication_store(
    const provisional_store_v0 &store,
    transaction_token_v0 *token) {
  if (token == NULL) return kStatusInvalidArgument;
  std::memset(token, 0, sizeof(*token));
  const status_kind validation =
      validate_provisional_publication_store(store);
  if (validation != kStatusOk) return validation;

  range_record_v0 *match = NULL;
  for (std::map<uint64_t, range_record_v0>::iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    range_record_v0 &record = it->second;
    uint64_t end = 0;
    if (!checked_range_end(record.base, record.byte_count, &end)) {
      return kStatusAddressOverflow;
    }
    if (store.aligned_32b_address >= record.base &&
        store.aligned_32b_address < end) {
      match = &record;
      break;
    }
  }
  if (match == NULL) return kStatusRecordNotFound;
  if (match->outstanding_transactions ==
      std::numeric_limits<uint32_t>::max()) {
    return kStatusOutstandingTransactions;
  }
  if (next_transaction_id_ == 0 ||
      next_transaction_id_ ==
          std::numeric_limits<uint64_t>::max()) {
    return kStatusTransactionIdExhausted;
  }

  transaction_token_v0 prepared = {};
  prepared.transaction_id = next_transaction_id_++;
  prepared.record_id = match->record_id;
  prepared.owner_hw_sid = store.owner.owner_hw_sid;
  prepared.owner_generation =
      store.owner.launch_allocation_generation;
  prepared.window_generation = store.owner.window_generation;
  prepared.lane_id = store.lane_id;
  prepared.object = store.object;
  prepared.access = kAccessHandoffShaderTraceInputPublish;
  transaction_record_v0 transaction = {};
  transaction.token = prepared;
  transactions_[prepared.transaction_id] = transaction;
  ++match->outstanding_transactions;
  *token = prepared;
  return kStatusOk;
}

status_kind registry_v0::validate_provisional_publication_store(
    const provisional_store_v0 &store) const {
  if (!valid_provisional_owner(store.owner) ||
      store.lane_id >= kLaneCapacity ||
      store.object != kObjectHandoff ||
      store.aligned_32b_address % kAddressChunkBytes != 0 ||
      store.byte_mask == 0) {
    return kStatusInvalidArgument;
  }

  const range_record_v0 *match = NULL;
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const range_record_v0 &record = it->second;
    uint64_t end = 0;
    if (!checked_range_end(record.base, record.byte_count, &end)) {
      return kStatusAddressOverflow;
    }
    if (store.aligned_32b_address < record.base ||
        store.aligned_32b_address >= end) {
      continue;
    }
    if (!same_provisional_owner(record.provisional_owner,
                                store.owner)) {
      return kStatusOwnerMismatch;
    }
    if ((record.lane_mask & lane_bit(store.lane_id)) == 0) {
      return kStatusLaneMismatch;
    }
    if (record.object != store.object) {
      return kStatusObjectKindMismatch;
    }
    if (record.phase != kPhaseProvisional) {
      return kStatusWrongPhase;
    }
    const size_t chunk = static_cast<size_t>(
        (store.aligned_32b_address - record.base) /
        kAddressChunkBytes);
    if (chunk >= record.allowed_publication_masks.size() ||
        (store.byte_mask &
         ~record.allowed_publication_masks[chunk]) != 0) {
      return kStatusByteMaskMismatch;
    }
    match = &record;
    break;
  }
  if (match == NULL) return kStatusRecordNotFound;
  return kStatusOk;
}

status_kind registry_v0::observe_provisional_range(
    uint64_t aligned_32b_address,
    provisional_range_observation_v0 *observation) const {
  if (observation == NULL ||
      aligned_32b_address % kAddressChunkBytes != 0) {
    return kStatusInvalidArgument;
  }
  std::memset(observation, 0, sizeof(*observation));
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const range_record_v0 &record = it->second;
    uint64_t end = 0;
    if (!checked_range_end(record.base, record.byte_count, &end)) {
      return kStatusAddressOverflow;
    }
    if (aligned_32b_address < record.base ||
        aligned_32b_address >= end) {
      continue;
    }
    uint8_t lane = 0;
    while (lane < kLaneCapacity &&
           record.lane_mask != lane_bit(lane)) {
      ++lane;
    }
    if (lane == kLaneCapacity) return kStatusLaneMismatch;
    const size_t chunk = static_cast<size_t>(
        (aligned_32b_address - record.base) /
        kAddressChunkBytes);
    if (chunk >= record.allowed_publication_masks.size()) {
      return kStatusInvalidRange;
    }
    observation->owner = record.provisional_owner;
    observation->live_owner = record.live_owner;
    observation->range_base = record.base;
    observation->range_byte_count = record.byte_count;
    observation->lane_id = lane;
    observation->object = record.object;
    observation->phase = record.phase;
    observation->allowed_publication_mask =
        record.allowed_publication_masks[chunk];
    return kStatusOk;
  }
  return kStatusRecordNotFound;
}

status_kind registry_v0::provisional_group_outstanding(
    const provisional_owner_v0 &owner,
    uint64_t *outstanding_transactions) const {
  if (!valid_provisional_owner(owner) ||
      outstanding_transactions == NULL) {
    return kStatusInvalidArgument;
  }
  *outstanding_transactions = 0;
  bool found = false;
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const range_record_v0 &record = it->second;
    if (!same_provisional_owner(record.provisional_owner, owner)) {
      continue;
    }
    if (record.phase != kPhaseProvisional &&
        record.phase != kPhasePendingBind &&
        record.phase != kPhaseCancelPending) {
      return kStatusWrongPhase;
    }
    if (record.outstanding_transactions >
        std::numeric_limits<uint64_t>::max() -
            *outstanding_transactions) {
      return kStatusOutstandingTransactions;
    }
    *outstanding_transactions +=
        record.outstanding_transactions;
    found = true;
  }
  return found ? kStatusOk : kStatusRecordNotFound;
}

status_kind registry_v0::begin_live_bind(
    const provisional_owner_v0 &owner, uint32_t active_mask,
    const range_spec_v0 *expected_ranges,
    size_t expected_range_count) {
  if (!valid_provisional_owner(owner) || active_mask == 0 ||
      expected_ranges == NULL ||
      expected_range_count != 2u * count_lanes(active_mask)) {
    return kStatusInvalidArgument;
  }
  for (size_t index = 0; index < expected_range_count; ++index) {
    if (!valid_range_spec(expected_ranges[index]) ||
        (expected_ranges[index].lane_mask & active_mask) !=
            expected_ranges[index].lane_mask) {
      return kStatusInvalidRange;
    }
  }
  std::vector<range_record_v0 *> matching;
  for (std::map<uint64_t, range_record_v0>::iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    range_record_v0 &record = it->second;
    if (!same_provisional_owner(record.provisional_owner,
                                owner)) {
      continue;
    }
    if (record.phase != kPhaseProvisional) {
      return kStatusWrongPhase;
    }
    matching.push_back(&record);
  }
  if (matching.size() != expected_range_count) {
    return kStatusRecordNotFound;
  }
  std::vector<bool> used(expected_range_count, false);
  uint64_t outstanding = 0;
  for (size_t record_index = 0;
       record_index < matching.size(); ++record_index) {
    range_record_v0 &record = *matching[record_index];
    if (record.active_mask != active_mask) {
      return kStatusOwnerMismatch;
    }
    bool found = false;
    for (size_t expected_index = 0;
         expected_index < expected_range_count; ++expected_index) {
      if (!used[expected_index] &&
          same_range_spec(record,
                          expected_ranges[expected_index])) {
        used[expected_index] = true;
        found = true;
        break;
      }
    }
    if (!found) return kStatusRecordNotFound;
    outstanding += record.outstanding_transactions;
  }
  for (size_t index = 0; index < matching.size(); ++index) {
    matching[index]->phase = kPhasePendingBind;
  }
  return outstanding == 0 ? kStatusOk
                          : kStatusOutstandingTransactions;
}

status_kind registry_v0::commit_live_bind(
    const provisional_owner_v0 &owner,
    uint32_t resident_warp_generation) {
  if (!valid_provisional_owner(owner) ||
      resident_warp_generation == 0) {
    return kStatusInvalidResidentGeneration;
  }
  std::vector<range_record_v0 *> matching;
  for (std::map<uint64_t, range_record_v0>::iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    range_record_v0 &record = it->second;
    if (!same_provisional_owner(record.provisional_owner,
                                owner)) {
      continue;
    }
    if (record.phase != kPhasePendingBind) {
      return kStatusWrongPhase;
    }
    if (record.outstanding_transactions != 0) {
      return kStatusOutstandingTransactions;
    }
    matching.push_back(&record);
  }
  if (matching.empty()) return kStatusRecordNotFound;

  live_owner_v0 live = {};
  live.owner_hw_sid = owner.owner_hw_sid;
  live.resident_warp_generation = resident_warp_generation;
  live.window_generation = owner.window_generation;
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    if (valid_live_owner(it->second.live_owner) &&
        same_live_owner(it->second.live_owner, live) &&
        !same_provisional_owner(
            it->second.provisional_owner, owner)) {
      return kStatusDuplicateOwner;
    }
  }
  for (size_t index = 0; index < matching.size(); ++index) {
    matching[index]->live_owner = live;
    matching[index]->phase = kPhaseLive;
  }
  return kStatusOk;
}

status_kind registry_v0::validate_live_group(
    const live_owner_v0 &owner, uint32_t active_mask,
    const range_spec_v0 *expected_ranges,
    size_t expected_range_count) const {
  if (!valid_live_owner(owner) || active_mask == 0 ||
      expected_ranges == NULL ||
      expected_range_count != 2u * count_lanes(active_mask)) {
    return kStatusInvalidArgument;
  }
  for (size_t index = 0; index < expected_range_count; ++index) {
    if (!valid_range_spec(expected_ranges[index]) ||
        (expected_ranges[index].lane_mask & active_mask) !=
            expected_ranges[index].lane_mask) {
      return kStatusInvalidRange;
    }
  }
  std::vector<const range_record_v0 *> matching;
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const range_record_v0 &record = it->second;
    if (!same_live_owner(record.live_owner, owner)) continue;
    if (record.phase != kPhaseLive) return kStatusWrongPhase;
    matching.push_back(&record);
  }
  if (matching.size() != expected_range_count) {
    return kStatusRecordNotFound;
  }
  std::vector<bool> used(expected_range_count, false);
  for (size_t record_index = 0;
       record_index < matching.size(); ++record_index) {
    const range_record_v0 &record = *matching[record_index];
    if (record.active_mask != active_mask) {
      return kStatusOwnerMismatch;
    }
    bool found = false;
    for (size_t expected_index = 0;
         expected_index < expected_range_count; ++expected_index) {
      if (!used[expected_index] &&
          same_range_spec(record,
                          expected_ranges[expected_index])) {
        used[expected_index] = true;
        found = true;
        break;
      }
    }
    if (!found) return kStatusRecordNotFound;
  }
  return kStatusOk;
}

status_kind registry_v0::validate_live_access(
    const live_access_v0 &access) const {
  const bool shader_return =
      access.access == kAccessHandoffShaderReturn;
  if (!valid_live_owner(access.owner) ||
      access.lane_id >= kLaneCapacity ||
      !access_matches_object(access.access, access.object) ||
      access.access == kAccessHandoffShaderTraceInputPublish ||
      access.reserved_zero != 0 ||
      (shader_return !=
       (access.completion_transaction_generation != 0)) ||
      (shader_return && access.cohort_index >= kLaneCapacity) ||
      (!shader_return && access.cohort_index != 0) ||
      access.aligned_32b_address % kAddressChunkBytes != 0 ||
      access.byte_mask == 0) {
    return kStatusInvalidArgument;
  }

  const range_record_v0 *match = NULL;
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const range_record_v0 &record = it->second;
    uint64_t end = 0;
    if (!checked_range_end(record.base, record.byte_count, &end)) {
      return kStatusAddressOverflow;
    }
    if (access.aligned_32b_address < record.base ||
        access.aligned_32b_address >= end) {
      continue;
    }
    if (!same_live_owner(record.live_owner, access.owner)) {
      return kStatusOwnerMismatch;
    }
    if ((record.lane_mask & lane_bit(access.lane_id)) == 0) {
      return kStatusLaneMismatch;
    }
    if (record.object != access.object) {
      return kStatusObjectKindMismatch;
    }
    if (record.phase != kPhaseLive) return kStatusWrongPhase;
    match = &record;
    break;
  }
  if (match == NULL) return kStatusRecordNotFound;
  return kStatusOk;
}

status_kind registry_v0::accept_live_access(
    const live_access_v0 &access,
    transaction_token_v0 *token) {
  if (token == NULL) return kStatusInvalidArgument;
  std::memset(token, 0, sizeof(*token));
  const status_kind validation = validate_live_access(access);
  if (validation != kStatusOk) return validation;

  range_record_v0 *match = NULL;
  for (std::map<uint64_t, range_record_v0>::iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    range_record_v0 &record = it->second;
    uint64_t end = 0;
    if (!checked_range_end(record.base, record.byte_count, &end)) {
      return kStatusAddressOverflow;
    }
    if (access.aligned_32b_address >= record.base &&
        access.aligned_32b_address < end &&
        same_live_owner(record.live_owner, access.owner) &&
        (record.lane_mask & lane_bit(access.lane_id)) != 0 &&
        record.object == access.object && record.phase == kPhaseLive) {
      match = &record;
      break;
    }
  }
  if (match == NULL) return kStatusRecordNotFound;
  if (match->outstanding_transactions ==
      std::numeric_limits<uint32_t>::max()) {
    return kStatusOutstandingTransactions;
  }
  if (next_transaction_id_ == 0 ||
      next_transaction_id_ ==
          std::numeric_limits<uint64_t>::max()) {
    return kStatusTransactionIdExhausted;
  }

  transaction_token_v0 prepared = {};
  prepared.transaction_id = next_transaction_id_++;
  prepared.record_id = match->record_id;
  prepared.owner_hw_sid = access.owner.owner_hw_sid;
  prepared.owner_generation =
      access.owner.resident_warp_generation;
  prepared.window_generation = access.owner.window_generation;
  prepared.lane_id = access.lane_id;
  prepared.object = access.object;
  prepared.access = access.access;
  prepared.completion_transaction_generation =
      access.completion_transaction_generation;
  prepared.cohort_index = access.cohort_index;
  transaction_record_v0 transaction = {};
  transaction.token = prepared;
  transactions_[prepared.transaction_id] = transaction;
  ++match->outstanding_transactions;
  *token = prepared;
  return kStatusOk;
}

status_kind registry_v0::live_group_outstanding(
    const live_owner_v0 &owner,
    uint64_t *outstanding_transactions) const {
  if (!valid_live_owner(owner) || outstanding_transactions == NULL) {
    return kStatusInvalidArgument;
  }
  *outstanding_transactions = 0;
  bool found = false;
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const range_record_v0 &record = it->second;
    if (!same_live_owner(record.live_owner, owner)) continue;
    if (record.phase != kPhaseLive) return kStatusWrongPhase;
    found = true;
    if (record.outstanding_transactions >
        std::numeric_limits<uint64_t>::max() -
            *outstanding_transactions) {
      return kStatusOutstandingTransactions;
    }
    *outstanding_transactions += record.outstanding_transactions;
  }
  return found ? kStatusOk : kStatusRecordNotFound;
}

status_kind registry_v0::complete_transaction(
    const transaction_token_v0 &token) {
  if (token_is_zero(token) || token.reserved_zero != 0) {
    return kStatusInvalidArgument;
  }
  std::map<uint64_t, transaction_record_v0>::iterator
      transaction_it = transactions_.find(token.transaction_id);
  if (transaction_it == transactions_.end()) {
    return kStatusTransactionNotFound;
  }
  if (std::memcmp(&transaction_it->second.token, &token,
                  sizeof(token)) != 0) {
    return kStatusStaleTransaction;
  }
  std::map<uint64_t, range_record_v0>::iterator record_it =
      records_.find(token.record_id);
  if (record_it == records_.end() ||
      record_it->second.outstanding_transactions == 0) {
    return kStatusStaleTransaction;
  }
  range_record_v0 &record = record_it->second;
  if (record.phase == kPhaseProvisional ||
      record.phase == kPhasePendingBind ||
      record.phase == kPhaseCancelPending) {
    if (token.access !=
            kAccessHandoffShaderTraceInputPublish ||
        token.owner_hw_sid !=
            record.provisional_owner.owner_hw_sid ||
        token.owner_generation !=
            record.provisional_owner
                .launch_allocation_generation ||
        token.window_generation !=
            record.provisional_owner.window_generation) {
      return kStatusStaleTransaction;
    }
  } else if (record.phase == kPhaseLive ||
             record.phase == kPhaseReleasePending) {
    if (token.access ==
            kAccessHandoffShaderTraceInputPublish ||
        token.owner_hw_sid != record.live_owner.owner_hw_sid ||
        token.owner_generation !=
            record.live_owner.resident_warp_generation ||
        token.window_generation !=
            record.live_owner.window_generation) {
      return kStatusStaleTransaction;
    }
  } else {
    return kStatusWrongPhase;
  }

  transactions_.erase(transaction_it);
  --record.outstanding_transactions;

  const phase_kind terminal_phase = record.phase;
  if (terminal_phase == kPhaseCancelPending ||
      terminal_phase == kPhaseReleasePending) {
    bool group_drained = true;
    if (terminal_phase == kPhaseCancelPending) {
      const provisional_owner_v0 owner =
          record.provisional_owner;
      for (std::map<uint64_t, range_record_v0>::const_iterator it =
               records_.begin();
           it != records_.end(); ++it) {
        if (same_provisional_owner(
                it->second.provisional_owner, owner) &&
            it->second.outstanding_transactions != 0) {
          group_drained = false;
          break;
        }
      }
      if (group_drained) {
        for (std::map<uint64_t, range_record_v0>::iterator it =
                 records_.begin();
             it != records_.end();) {
          if (same_provisional_owner(
                  it->second.provisional_owner, owner)) {
            records_.erase(it++);
          } else {
            ++it;
          }
        }
      }
    } else {
      const live_owner_v0 owner = record.live_owner;
      for (std::map<uint64_t, range_record_v0>::const_iterator it =
               records_.begin();
           it != records_.end(); ++it) {
        if (same_live_owner(it->second.live_owner, owner) &&
            it->second.outstanding_transactions != 0) {
          group_drained = false;
          break;
        }
      }
      if (group_drained) {
        for (std::map<uint64_t, range_record_v0>::iterator it =
                 records_.begin();
             it != records_.end();) {
          if (same_live_owner(it->second.live_owner, owner)) {
            records_.erase(it++);
          } else {
            ++it;
          }
        }
      }
    }
  }
  return kStatusOk;
}

status_kind registry_v0::cancel_provisional_group(
    const provisional_owner_v0 &owner) {
  if (!valid_provisional_owner(owner)) {
    return kStatusInvalidOwner;
  }
  std::vector<range_record_v0 *> matching;
  uint64_t outstanding = 0;
  for (std::map<uint64_t, range_record_v0>::iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    range_record_v0 &record = it->second;
    if (!same_provisional_owner(record.provisional_owner,
                                owner)) {
      continue;
    }
    if (record.phase != kPhaseProvisional &&
        record.phase != kPhasePendingBind) {
      return kStatusWrongPhase;
    }
    outstanding += record.outstanding_transactions;
    matching.push_back(&record);
  }
  if (matching.empty()) return kStatusRecordNotFound;
  if (outstanding == 0) {
    for (std::map<uint64_t, range_record_v0>::iterator it =
             records_.begin();
         it != records_.end();) {
      if (same_provisional_owner(it->second.provisional_owner,
                                 owner)) {
        records_.erase(it++);
      } else {
        ++it;
      }
    }
    return kStatusOk;
  }
  for (size_t index = 0; index < matching.size(); ++index) {
    matching[index]->phase = kPhaseCancelPending;
  }
  return kStatusOutstandingTransactions;
}

status_kind registry_v0::release_live_group(
    const live_owner_v0 &owner) {
  if (!valid_live_owner(owner)) return kStatusInvalidOwner;
  std::vector<range_record_v0 *> matching;
  uint64_t outstanding = 0;
  for (std::map<uint64_t, range_record_v0>::iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    range_record_v0 &record = it->second;
    if (!same_live_owner(record.live_owner, owner)) continue;
    if (record.phase != kPhaseLive) return kStatusWrongPhase;
    outstanding += record.outstanding_transactions;
    matching.push_back(&record);
  }
  if (matching.empty()) return kStatusRecordNotFound;
  if (outstanding == 0) {
    for (std::map<uint64_t, range_record_v0>::iterator it =
             records_.begin();
         it != records_.end();) {
      if (same_live_owner(it->second.live_owner, owner)) {
        records_.erase(it++);
      } else {
        ++it;
      }
    }
    return kStatusOk;
  }
  for (size_t index = 0; index < matching.size(); ++index) {
    matching[index]->phase = kPhaseReleasePending;
  }
  return kStatusOutstandingTransactions;
}

status_kind registry_v0::poll_live_group_release(
    const live_owner_v0 &owner,
    live_release_observation_v0 *observation) const {
  if (!valid_live_owner(owner) || observation == NULL) {
    return kStatusInvalidArgument;
  }
  std::memset(observation, 0, sizeof(*observation));
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    const range_record_v0 &record = it->second;
    if (!same_live_owner(record.live_owner, owner)) continue;
    if (record.phase != kPhaseReleasePending) return kStatusWrongPhase;
    observation->records_present = 1;
    observation->release_pending = 1;
    observation->outstanding_transactions +=
        record.outstanding_transactions;
  }
  observation->released = observation->records_present == 0;
  return kStatusOk;
}

bool registry_v0::range_available(uint64_t base,
                                  uint64_t byte_count) const {
  uint64_t end = 0;
  if (base % kAddressChunkBytes != 0 ||
      byte_count % kAddressChunkBytes != 0 ||
      !checked_range_end(base, byte_count, &end)) {
    return false;
  }
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    uint64_t existing_end = 0;
    if (!checked_range_end(it->second.base,
                           it->second.byte_count,
                           &existing_end) ||
        ranges_overlap(base, end, it->second.base,
                       existing_end)) {
      return false;
    }
  }
  return true;
}

registry_snapshot_v0 registry_v0::snapshot() const {
  registry_snapshot_v0 result = {};
  result.record_count = records_.size();
  result.transaction_count = transactions_.size();
  for (std::map<uint64_t, range_record_v0>::const_iterator it =
           records_.begin();
       it != records_.end(); ++it) {
    switch (it->second.phase) {
      case kPhaseProvisional:
        ++result.provisional_count;
        break;
      case kPhasePendingBind:
        ++result.pending_bind_count;
        break;
      case kPhaseLive:
        ++result.live_count;
        break;
      case kPhaseCancelPending:
        ++result.cancel_pending_count;
        break;
      case kPhaseReleasePending:
        ++result.release_pending_count;
        break;
      default:
        break;
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
    case kStatusInvalidOwner:
      return "invalid_owner";
    case kStatusInvalidRange:
      return "invalid_range";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusRangeOverlap:
      return "range_overlap";
    case kStatusDuplicateOwner:
      return "duplicate_owner";
    case kStatusRecordNotFound:
      return "record_not_found";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusLaneMismatch:
      return "lane_mismatch";
    case kStatusObjectKindMismatch:
      return "object_kind_mismatch";
    case kStatusByteMaskMismatch:
      return "byte_mask_mismatch";
    case kStatusWrongPhase:
      return "wrong_phase";
    case kStatusOutstandingTransactions:
      return "outstanding_transactions";
    case kStatusInvalidResidentGeneration:
      return "invalid_resident_generation";
    case kStatusTransactionIdExhausted:
      return "transaction_id_exhausted";
    case kStatusRecordIdExhausted:
      return "record_id_exhausted";
    case kStatusTransactionNotFound:
      return "transaction_not_found";
    case kStatusStaleTransaction:
      return "stale_transaction";
  }
  return "unknown";
}

const char *phase_name(phase_kind phase) {
  switch (phase) {
    case kPhaseInvalid:
      return "invalid";
    case kPhaseProvisional:
      return "provisional";
    case kPhasePendingBind:
      return "pending_bind";
    case kPhaseLive:
      return "live";
    case kPhaseCancelPending:
      return "cancel_pending";
    case kPhaseReleasePending:
      return "release_pending";
  }
  return "unknown";
}

const char *access_name(access_kind access) {
  switch (access) {
    case kAccessInvalid:
      return "invalid";
    case kAccessHandoffShaderTraceInputPublish:
      return "handoff.shader_trace_input_publish";
    case kAccessHandoffRtcoreAcquire:
      return "handoff.rtcore_acquire";
    case kAccessHandoffRtcorePublish:
      return "handoff.rtcore_publish";
    case kAccessHandoffShaderDispatchRead:
      return "handoff.shader_dispatch_read";
    case kAccessHandoffShaderBuiltinRead:
      return "handoff.shader_builtin_read";
    case kAccessHandoffShaderReturn:
      return "handoff.shader_return";
    case kAccessContextShaderRead:
      return "context.shader_read";
    case kAccessContextShaderWrite:
      return "context.shader_write";
  }
  return "unknown";
}

}  // namespace address_range_registry
}  // namespace v04
}  // namespace rtcore
