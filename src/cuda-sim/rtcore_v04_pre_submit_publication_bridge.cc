#include "rtcore_v04_pre_submit_publication_bridge.h"

#include <cstring>
#include <limits>
#include <vector>

namespace rtcore {
namespace v04 {
namespace pre_submit_publication {
namespace {

static uint32_t capacity_mask(uint32_t capacity) {
  if (capacity == 0) return 0;
  if (capacity >= allocation_identity::kLaneCapacity) return 0xffffffffu;
  return (uint32_t{1} << capacity) - 1u;
}

static bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t *product) {
  if (product == NULL ||
      (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *product = lhs * rhs;
  return true;
}

static bool checked_lane_address(uint64_t base, uint32_t stride,
                                 uint32_t lane, uint64_t *address) {
  uint64_t offset = 0;
  if (!checked_multiply(stride, lane, &offset) ||
      base > std::numeric_limits<uint64_t>::max() - offset) {
    return false;
  }
  *address = base + offset;
  return true;
}

static uint32_t count_lanes(uint32_t mask) {
  uint32_t count = 0;
  while (mask != 0) {
    count += mask & 1u;
    mask >>= 1;
  }
  return count;
}

static status_kind validate_geometry(
    const lane_publication_request_v0 &request) {
  if (request.publication_warp_uid == 0 || request.active_mask == 0 ||
      request.capacity_lane_slots == 0 ||
      request.capacity_lane_slots > allocation_identity::kLaneCapacity ||
      request.lane_id >= request.capacity_lane_slots ||
      request.context_lane_stride_bytes == 0 ||
      request.handoff_lane_stride_bytes == 0 ||
      request.context_lane_stride_bytes %
              address_range_registry::kAddressChunkBytes !=
          0 ||
      request.handoff_lane_stride_bytes %
              address_range_registry::kAddressChunkBytes !=
          0 ||
      request.handoff_allowed_publication_masks == NULL ||
      request.handoff_allowed_publication_mask_count !=
          request.handoff_lane_stride_bytes /
              address_range_registry::kAddressChunkBytes ||
      (request.active_mask & ~capacity_mask(request.capacity_lane_slots)) !=
          0) {
    return kStatusInvalidGeometry;
  }
  uint64_t expected_context_bytes = 0;
  uint64_t expected_handoff_bytes = 0;
  if (!checked_multiply(request.capacity_lane_slots,
                        request.context_lane_stride_bytes,
                        &expected_context_bytes) ||
      !checked_multiply(request.capacity_lane_slots,
                        request.handoff_lane_stride_bytes,
                        &expected_handoff_bytes)) {
    return kStatusAddressOverflow;
  }
  if (request.allocation_ranges.context_byte_count !=
          expected_context_bytes ||
      request.allocation_ranges.handoff_byte_count !=
          expected_handoff_bytes) {
    return kStatusInvalidGeometry;
  }
  bool has_allowed_handoff_byte = false;
  for (size_t index = 0;
       index < request.handoff_allowed_publication_mask_count; ++index) {
    has_allowed_handoff_byte |=
        request.handoff_allowed_publication_masks[index] != 0;
  }
  return has_allowed_handoff_byte ? kStatusOk : kStatusInvalidGeometry;
}

static status_kind make_range_specs(
    const lane_publication_request_v0 &request,
    std::vector<std::vector<uint32_t> > *owned_masks,
    std::vector<address_range_registry::range_spec_v0> *ranges) {
  if (owned_masks == NULL || ranges == NULL) return kStatusInvalidArgument;
  const size_t range_count =
      static_cast<size_t>(2u * count_lanes(request.active_mask));
  owned_masks->clear();
  ranges->clear();
  owned_masks->resize(range_count);
  ranges->resize(range_count);

  const size_t context_chunks =
      request.context_lane_stride_bytes /
      address_range_registry::kAddressChunkBytes;
  const size_t handoff_chunks =
      request.handoff_lane_stride_bytes /
      address_range_registry::kAddressChunkBytes;
  size_t index = 0;
  for (uint32_t lane = 0; lane < request.capacity_lane_slots; ++lane) {
    const uint32_t lane_mask = uint32_t{1} << lane;
    if ((request.active_mask & lane_mask) == 0) continue;
    uint64_t handoff_address = 0;
    uint64_t context_address = 0;
    if (!checked_lane_address(request.allocation_ranges.handoff_base,
                              request.handoff_lane_stride_bytes, lane,
                              &handoff_address) ||
        !checked_lane_address(request.allocation_ranges.context_base,
                              request.context_lane_stride_bytes, lane,
                              &context_address)) {
      return kStatusAddressOverflow;
    }

    (*owned_masks)[index].assign(
        request.handoff_allowed_publication_masks,
        request.handoff_allowed_publication_masks + handoff_chunks);
    address_range_registry::range_spec_v0 handoff = {};
    handoff.base = handoff_address;
    handoff.byte_count = request.handoff_lane_stride_bytes;
    handoff.lane_mask = lane_mask;
    handoff.object = address_range_registry::kObjectHandoff;
    handoff.allowed_publication_masks = (*owned_masks)[index].data();
    handoff.allowed_publication_mask_count = handoff_chunks;
    (*ranges)[index] = handoff;
    ++index;

    (*owned_masks)[index].assign(context_chunks, 0);
    address_range_registry::range_spec_v0 context = {};
    context.base = context_address;
    context.byte_count = request.context_lane_stride_bytes;
    context.lane_mask = lane_mask;
    context.object = address_range_registry::kObjectContext;
    context.allowed_publication_masks = (*owned_masks)[index].data();
    context.allowed_publication_mask_count = context_chunks;
    (*ranges)[index] = context;
    ++index;
  }
  return index == range_count ? kStatusOk : kStatusInvalidGeometry;
}

}  // namespace

bridge_v0::bridge_v0()
    : authority_(),
      registry_(),
      publication_identity_observations_(),
      registered_groups_() {}

void bridge_v0::reset() {
  authority_.reset();
  registry_.reset();
  publication_identity_observations_.clear();
  registered_groups_.clear();
}

status_kind bridge_v0::observe_initial_publication(
    const lane_publication_request_v0 &request,
    lane_publication_result_v0 *result) {
  if (result == NULL) return kStatusInvalidArgument;
  std::memset(result, 0, sizeof(*result));
  result->authority_status = allocation_identity::kStatusInvalidArgument;
  result->registry_status = address_range_registry::kStatusInvalidArgument;

  const status_kind geometry_status = validate_geometry(request);
  if (geometry_status != kStatusOk) return geometry_status;

  const std::pair<uint64_t, uint32_t> publication_key(
      request.slot.allocation_domain_id, request.slot.allocation_slot_id);
  std::map<std::pair<uint64_t, uint32_t>,
           publication_identity_observation_v0>::iterator
      publication_identity =
          publication_identity_observations_.find(publication_key);
  if (publication_identity != publication_identity_observations_.end() &&
      publication_identity->second.publication_warp_uid !=
          request.publication_warp_uid) {
    return kStatusGroupConflict;
  }

  allocation_identity::publication_request_v0 authority_request = {};
  authority_request.slot = request.slot;
  authority_request.owner = request.owner;
  authority_request.ranges = request.allocation_ranges;
  authority_request.active_mask = request.active_mask;
  authority_request.lane_id = request.lane_id;
  allocation_identity::publication_observation_v0 observation = {};
  result->authority_status =
      authority_.observe_publication(authority_request, &observation);
  if (result->authority_status != allocation_identity::kStatusOk) {
    return kStatusAuthorityRejected;
  }

  result->identity = observation.identity;
  result->published_lane_mask = observation.published_lane_mask;
  result->newly_allocated = observation.newly_allocated;
  result->lane_was_new = observation.lane_was_new;
  result->publication_complete = observation.publication_complete;
  result->registry_status = address_range_registry::kStatusOk;

  if (observation.newly_allocated) {
    if (publication_identity != publication_identity_observations_.end()) {
      return kStatusGroupConflict;
    }
    publication_identity_observation_v0 identity_observation = {};
    identity_observation.record_id = observation.identity.record_id;
    identity_observation.publication_warp_uid =
        request.publication_warp_uid;
    publication_identity_observations_[publication_key] =
        identity_observation;
  } else if (publication_identity ==
                 publication_identity_observations_.end() ||
             publication_identity->second.record_id !=
                 observation.identity.record_id) {
    return kStatusGroupConflict;
  }
  if (!observation.publication_complete || !observation.lane_was_new) {
    return kStatusOk;
  }

  if (registered_groups_.find(observation.identity.record_id) !=
      registered_groups_.end()) {
    return kStatusGroupConflict;
  }

  std::vector<std::vector<uint32_t> > owned_masks;
  std::vector<address_range_registry::range_spec_v0> ranges;
  const status_kind range_status =
      make_range_specs(request, &owned_masks, &ranges);
  if (range_status != kStatusOk) return range_status;

  address_range_registry::provisional_owner_v0 provisional_owner = {};
  provisional_owner.owner_hw_sid = request.owner.owner_hw_sid;
  provisional_owner.warp_uid = request.publication_warp_uid;
  provisional_owner.dynamic_warp_id = request.owner.dynamic_warp_id;
  provisional_owner.launch_allocation_generation =
      observation.identity.launch_allocation_generation;
  provisional_owner.window_generation =
      observation.identity.window_generation;
  result->registry_status = registry_.register_provisional_group(
      provisional_owner, request.active_mask, ranges.data(), ranges.size());
  if (result->registry_status != address_range_registry::kStatusOk) {
    result->authority_status =
        authority_.begin_release(observation.identity);
    if (result->authority_status != allocation_identity::kStatusOk) {
      return kStatusAuthorityRollbackFailed;
    }
    result->authority_status =
        authority_.commit_release(observation.identity);
    if (result->authority_status != allocation_identity::kStatusOk) {
      return kStatusAuthorityRollbackFailed;
    }
    publication_identity_observations_.erase(publication_key);
    return kStatusRegistryRejected;
  }

  registered_group_v0 group = {};
  group.provisional_owner = provisional_owner;
  registered_groups_[observation.identity.record_id] = group;
  result->provisional_group_registered = 1;
  return kStatusOk;
}

status_kind bridge_v0::lookup_registered_group_for_submit(
    const allocation_identity::allocation_slot_v0 &slot,
    const allocation_identity::owner_v0 &owner, uint32_t active_mask,
    const allocation_identity::allocation_ranges_v0 &ranges,
    allocation_identity::allocation_identity_v0 *identity,
    address_range_registry::provisional_owner_v0
        *provisional_owner) const {
  if (identity == NULL || provisional_owner == NULL) {
    return kStatusInvalidArgument;
  }
  const allocation_identity::status_kind lookup_status =
      authority_.lookup_for_submit(slot, owner, active_mask, ranges, identity);
  if (lookup_status != allocation_identity::kStatusOk) {
    return kStatusAuthorityRejected;
  }
  std::map<uint64_t, registered_group_v0>::const_iterator found =
      registered_groups_.find(identity->record_id);
  if (found == registered_groups_.end()) return kStatusGroupNotRegistered;
  *provisional_owner = found->second.provisional_owner;
  return kStatusOk;
}

address_range_registry::status_kind bridge_v0::accept_publication_store(
    const address_range_registry::provisional_store_v0 &store,
    address_range_registry::transaction_token_v0 *token) {
  return registry_.accept_provisional_publication_store(store, token);
}

address_range_registry::status_kind bridge_v0::complete_publication_store(
    const address_range_registry::transaction_token_v0 &token) {
  return registry_.complete_transaction(token);
}

bridge_snapshot_v0 bridge_v0::snapshot() const {
  bridge_snapshot_v0 result = {};
  result.authority = authority_.snapshot();
  result.registry = registry_.snapshot();
  result.registered_group_count = registered_groups_.size();
  return result;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidGeometry:
      return "invalid_geometry";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusAuthorityRejected:
      return "authority_rejected";
    case kStatusRegistryRejected:
      return "registry_rejected";
    case kStatusAuthorityRollbackFailed:
      return "authority_rollback_failed";
    case kStatusGroupNotRegistered:
      return "group_not_registered";
    case kStatusGroupConflict:
      return "group_conflict";
  }
  return "unknown";
}

}  // namespace pre_submit_publication
}  // namespace v04
}  // namespace rtcore
