#include "rtcore_v04_pre_submit_publication_bridge.h"

#include <algorithm>
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

static bool same_provisional_owner(
    const address_range_registry::provisional_owner_v0 &lhs,
    const address_range_registry::provisional_owner_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.warp_uid == rhs.warp_uid &&
         lhs.dynamic_warp_id == rhs.dynamic_warp_id &&
         lhs.launch_allocation_generation ==
             rhs.launch_allocation_generation &&
         lhs.window_generation == rhs.window_generation;
}

static bool same_live_owner(
    const address_range_registry::live_owner_v0 &lhs,
    const address_range_registry::live_owner_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.resident_warp_generation ==
             rhs.resident_warp_generation &&
         lhs.window_generation == rhs.window_generation;
}

static bool same_execution_owner(
    const allocation_identity::owner_v0 &lhs,
    const allocation_identity::owner_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.dynamic_warp_id == rhs.dynamic_warp_id &&
         lhs.warp_id == rhs.warp_id;
}

static bool same_allocation_slot(
    const allocation_identity::allocation_slot_v0 &lhs,
    const allocation_identity::allocation_slot_v0 &rhs) {
  return lhs.allocation_domain_id == rhs.allocation_domain_id &&
         lhs.allocation_slot_id == rhs.allocation_slot_id;
}

static bool same_allocation_ranges(
    const allocation_identity::allocation_ranges_v0 &lhs,
    const allocation_identity::allocation_ranges_v0 &rhs) {
  return lhs.context_base == rhs.context_base &&
         lhs.context_byte_count == rhs.context_byte_count &&
         lhs.handoff_base == rhs.handoff_base &&
         lhs.handoff_byte_count == rhs.handoff_byte_count;
}

static uint8_t first_active_lane(uint32_t active_mask) {
  for (uint8_t lane = 0; lane < allocation_identity::kLaneCapacity; ++lane) {
    if ((active_mask & (uint32_t{1} << lane)) != 0) return lane;
  }
  return allocation_identity::kLaneCapacity;
}

static void fill_bind_ticket(
    const allocation_identity::allocation_identity_v0 &identity,
    first_submit_bind_ticket_v0 *ticket) {
  std::memset(ticket, 0, sizeof(*ticket));
  ticket->allocation_record_id = identity.record_id;
  ticket->owner_hw_sid = identity.owner.owner_hw_sid;
  ticket->dynamic_warp_id = identity.owner.dynamic_warp_id;
  ticket->warp_id = identity.owner.warp_id;
  ticket->active_mask = identity.active_mask;
  ticket->launch_allocation_generation =
      identity.launch_allocation_generation;
  ticket->window_generation = identity.window_generation;
  ticket->valid = 1;
}

static bool ticket_matches_group(
    const first_submit_bind_ticket_v0 &ticket,
    uint64_t record_id,
    const allocation_identity::owner_v0 &owner,
    uint32_t active_mask,
    const address_range_registry::provisional_owner_v0
        &provisional_owner) {
  return ticket.valid && ticket.allocation_record_id == record_id &&
         ticket.owner_hw_sid == owner.owner_hw_sid &&
         ticket.dynamic_warp_id == owner.dynamic_warp_id &&
         ticket.warp_id == owner.warp_id &&
         ticket.active_mask == active_mask &&
         ticket.launch_allocation_generation ==
             provisional_owner.launch_allocation_generation &&
         ticket.window_generation == provisional_owner.window_generation;
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
      registered_groups_(),
      accepted_publication_stores_() {}

void bridge_v0::reset() {
  authority_.reset();
  registry_.reset();
  publication_identity_observations_.clear();
  registered_groups_.clear();
  accepted_publication_stores_.clear();
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
  group.execution_owner = request.owner;
  group.slot = request.slot;
  group.allocation_ranges = request.allocation_ranges;
  group.active_mask = request.active_mask;
  group.capacity_lane_slots = request.capacity_lane_slots;
  group.context_lane_stride_bytes = request.context_lane_stride_bytes;
  group.handoff_lane_stride_bytes = request.handoff_lane_stride_bytes;
  group.handoff_allowed_publication_masks.assign(
      request.handoff_allowed_publication_masks,
      request.handoff_allowed_publication_masks +
          request.handoff_allowed_publication_mask_count);
  group.completed_publication_masks.assign(
      request.capacity_lane_slots,
      std::vector<uint32_t>(
          request.handoff_allowed_publication_mask_count, 0));
  group.fence_armed = 1;
  registered_groups_[observation.identity.record_id] = group;
  result->provisional_group_registered = 1;
  return kStatusOk;
}

status_kind bridge_v0::make_registered_group_ranges(
    const registered_group_v0 &group,
    std::vector<std::vector<uint32_t> > *owned_masks,
    std::vector<address_range_registry::range_spec_v0> *ranges) const {
  lane_publication_request_v0 request = {};
  request.slot = group.slot;
  request.owner = group.execution_owner;
  request.allocation_ranges = group.allocation_ranges;
  request.publication_warp_uid = group.provisional_owner.warp_uid;
  request.active_mask = group.active_mask;
  request.capacity_lane_slots = group.capacity_lane_slots;
  request.context_lane_stride_bytes = group.context_lane_stride_bytes;
  request.handoff_lane_stride_bytes = group.handoff_lane_stride_bytes;
  request.handoff_allowed_publication_masks =
      group.handoff_allowed_publication_masks.data();
  request.handoff_allowed_publication_mask_count =
      group.handoff_allowed_publication_masks.size();
  request.lane_id = first_active_lane(group.active_mask);
  const status_kind geometry = validate_geometry(request);
  return geometry == kStatusOk
             ? make_range_specs(request, owned_masks, ranges)
             : geometry;
}

bool bridge_v0::publication_coverage_complete(
    const registered_group_v0 &group) const {
  if (group.completed_publication_masks.size() !=
      group.capacity_lane_slots) {
    return false;
  }
  for (uint32_t lane = 0; lane < group.capacity_lane_slots; ++lane) {
    if ((group.active_mask & (uint32_t{1} << lane)) == 0) continue;
    if (group.completed_publication_masks[lane].size() !=
        group.handoff_allowed_publication_masks.size()) {
      return false;
    }
    for (size_t chunk = 0;
         chunk < group.handoff_allowed_publication_masks.size(); ++chunk) {
      const uint32_t required =
          group.handoff_allowed_publication_masks[chunk];
      if ((group.completed_publication_masks[lane][chunk] & required) !=
          required) {
        return false;
      }
    }
  }
  return true;
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
  const address_range_registry::status_kind status =
      registry_.accept_provisional_publication_store(store, token);
  if (status != address_range_registry::kStatusOk) return status;
  if (!accepted_publication_stores_
           .insert(std::make_pair(token->transaction_id, store))
           .second) {
    registry_.complete_transaction(*token);
    return address_range_registry::kStatusStaleTransaction;
  }
  return address_range_registry::kStatusOk;
}

address_range_registry::status_kind bridge_v0::complete_publication_store(
    const address_range_registry::transaction_token_v0 &token) {
  if (token.access != address_range_registry::
                          kAccessHandoffShaderTraceInputPublish) {
    return registry_.complete_transaction(token);
  }
  std::map<uint64_t, publication_store_v0>::iterator pending =
      accepted_publication_stores_.find(token.transaction_id);
  if (pending == accepted_publication_stores_.end()) {
    return address_range_registry::kStatusTransactionNotFound;
  }
  const publication_store_v0 store = pending->second;
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    if (!same_provisional_owner(it->second.provisional_owner,
                                store.owner)) {
      continue;
    }
    if (group != NULL) {
      return address_range_registry::kStatusDuplicateOwner;
    }
    group = &it->second;
  }
  if (group == NULL) return address_range_registry::kStatusRecordNotFound;
  if (store.lane_id >= group->capacity_lane_slots ||
      store.lane_id >= group->completed_publication_masks.size()) {
    return address_range_registry::kStatusLaneMismatch;
  }
  uint64_t lane_base = 0;
  if (!checked_lane_address(group->allocation_ranges.handoff_base,
                            group->handoff_lane_stride_bytes,
                            store.lane_id, &lane_base) ||
      store.aligned_32b_address < lane_base) {
    return address_range_registry::kStatusInvalidRange;
  }
  const uint64_t offset = store.aligned_32b_address - lane_base;
  if (offset >= group->handoff_lane_stride_bytes ||
      offset % address_range_registry::kAddressChunkBytes != 0) {
    return address_range_registry::kStatusInvalidRange;
  }
  const size_t chunk = static_cast<size_t>(
      offset / address_range_registry::kAddressChunkBytes);
  if (chunk >= group->handoff_allowed_publication_masks.size() ||
      chunk >= group->completed_publication_masks[store.lane_id].size() ||
      (store.byte_mask &
       ~group->handoff_allowed_publication_masks[chunk]) != 0) {
    return address_range_registry::kStatusByteMaskMismatch;
  }
  const address_range_registry::status_kind status =
      registry_.complete_transaction(token);
  if (status != address_range_registry::kStatusOk) return status;
  group->completed_publication_masks[store.lane_id][chunk] |=
      store.byte_mask;
  accepted_publication_stores_.erase(pending);
  return address_range_registry::kStatusOk;
}

status_kind bridge_v0::begin_publication_store_preaccept(
    const address_range_registry::provisional_store_v0 &store) {
  const address_range_registry::status_kind validation =
      registry_.validate_provisional_publication_store(store);
  if (validation != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    if (!same_provisional_owner(it->second.provisional_owner,
                                store.owner)) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusGroupNotRegistered;
  if (group->preaccept_pending ==
      std::numeric_limits<uint64_t>::max()) {
    return kStatusRegistryRejected;
  }
  ++group->preaccept_pending;
  return kStatusOk;
}

status_kind bridge_v0::cancel_publication_store_preaccept(
    const address_range_registry::provisional_store_v0 &store) {
  const address_range_registry::status_kind validation =
      registry_.validate_provisional_publication_store(store);
  if (validation != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    if (!same_provisional_owner(it->second.provisional_owner,
                                store.owner)) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusGroupNotRegistered;
  if (group->preaccept_pending == 0) return kStatusGroupConflict;
  --group->preaccept_pending;
  return kStatusOk;
}

status_kind bridge_v0::accept_preaccepted_publication_store(
    const address_range_registry::provisional_store_v0 &store,
    address_range_registry::transaction_token_v0 *token) {
  if (token == NULL) return kStatusInvalidArgument;
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    if (!same_provisional_owner(it->second.provisional_owner,
                                store.owner)) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusGroupNotRegistered;
  if (group->preaccept_pending == 0) {
    return kStatusGroupConflict;
  }
  const address_range_registry::status_kind status =
      registry_.accept_provisional_publication_store(store, token);
  if (status != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  if (!accepted_publication_stores_
           .insert(std::make_pair(token->transaction_id, store))
           .second) {
    registry_.complete_transaction(*token);
    return kStatusRegistryRejected;
  }
  --group->preaccept_pending;
  return kStatusOk;
}

address_range_registry::status_kind bridge_v0::accept_live_access(
    const address_range_registry::live_access_v0 &access,
    address_range_registry::transaction_token_v0 *token) {
  return registry_.accept_live_access(access, token);
}

status_kind bridge_v0::resolve_live_handoff_chunk(
    uint32_t owner_hw_sid, uint8_t lane_id,
    uint64_t aligned_32b_address, uint64_t *lane_slot_base,
    uint32_t *resident_warp_generation, uint32_t *window_generation,
    uint8_t *lane_slot_chunk) const {
  if (lane_id >= allocation_identity::kLaneCapacity ||
      aligned_32b_address == 0 ||
      aligned_32b_address % address_range_registry::kAddressChunkBytes != 0 ||
      lane_slot_base == NULL || resident_warp_generation == NULL ||
      window_generation == NULL || lane_slot_chunk == NULL) {
    return kStatusInvalidArgument;
  }
  address_range_registry::provisional_range_observation_v0 observation = {};
  const address_range_registry::status_kind observe_status =
      registry_.observe_provisional_range(aligned_32b_address, &observation);
  if (observe_status != address_range_registry::kStatusOk ||
      observation.phase != address_range_registry::kPhaseLive ||
      observation.live_owner.owner_hw_sid != owner_hw_sid ||
      observation.lane_id != lane_id ||
      observation.object != address_range_registry::kObjectHandoff) {
    return kStatusRegistryRejected;
  }
  const uint64_t range_offset = aligned_32b_address - observation.range_base;
  if (observation.range_byte_count !=
          4u * address_range_registry::kAddressChunkBytes ||
      range_offset % address_range_registry::kAddressChunkBytes != 0 ||
      range_offset / address_range_registry::kAddressChunkBytes >= 4u) {
    return kStatusInvalidGeometry;
  }
  *lane_slot_base = observation.range_base;
  *resident_warp_generation =
      observation.live_owner.resident_warp_generation;
  *window_generation = observation.live_owner.window_generation;
  *lane_slot_chunk = static_cast<uint8_t>(
      range_offset / address_range_registry::kAddressChunkBytes);
  return kStatusOk;
}

status_kind bridge_v0::preflight_live_handoff_access(
    uint32_t owner_hw_sid, uint32_t resident_warp_generation,
    uint8_t lane_id, address_range_registry::access_kind access_kind,
    uint64_t aligned_32b_address, uint32_t byte_mask,
    address_range_registry::live_access_v0 *access,
    uint8_t *lane_slot_chunk) const {
  const bool rtcore_acquire =
      access_kind ==
      address_range_registry::kAccessHandoffRtcoreAcquire;
  const bool shader_dispatch_read =
      access_kind ==
      address_range_registry::kAccessHandoffShaderDispatchRead;
  if (access == NULL || resident_warp_generation == 0 ||
      lane_id >= allocation_identity::kLaneCapacity ||
      (!rtcore_acquire && !shader_dispatch_read) || byte_mask == 0 ||
      (rtcore_acquire &&
       byte_mask != std::numeric_limits<uint32_t>::max())) {
    return kStatusInvalidArgument;
  }
  std::memset(access, 0, sizeof(*access));
  uint64_t lane_slot_base = 0;
  uint32_t observed_resident_warp_generation = 0;
  uint32_t observed_window_generation = 0;
  uint8_t observed_lane_slot_chunk = 0xffu;
  const status_kind resolve_status = resolve_live_handoff_chunk(
      owner_hw_sid, lane_id, aligned_32b_address, &lane_slot_base,
      &observed_resident_warp_generation, &observed_window_generation,
      &observed_lane_slot_chunk);
  if (resolve_status != kStatusOk ||
      observed_resident_warp_generation != resident_warp_generation) {
    return resolve_status == kStatusOk ? kStatusRegistryRejected
                                      : resolve_status;
  }
  if (lane_slot_base +
          static_cast<uint64_t>(observed_lane_slot_chunk) *
              address_range_registry::kAddressChunkBytes !=
      aligned_32b_address) {
    return kStatusInvalidGeometry;
  }
  if (lane_slot_chunk != NULL) {
    *lane_slot_chunk = observed_lane_slot_chunk;
  }
  access->owner.owner_hw_sid = owner_hw_sid;
  access->owner.resident_warp_generation =
      observed_resident_warp_generation;
  access->owner.window_generation = observed_window_generation;
  access->lane_id = lane_id;
  access->object = address_range_registry::kObjectHandoff;
  access->access = access_kind;
  access->aligned_32b_address = aligned_32b_address;
  access->byte_mask = byte_mask;
  return registry_.validate_live_access(*access) ==
                 address_range_registry::kStatusOk
             ? kStatusOk
             : kStatusRegistryRejected;
}

status_kind bridge_v0::begin_live_access_preaccept(
    const address_range_registry::live_access_v0 &access) {
  const address_range_registry::status_kind validation =
      registry_.validate_live_access(access);
  if (validation != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    address_range_registry::live_owner_v0 owner = {};
    owner.owner_hw_sid = it->second.execution_owner.owner_hw_sid;
    owner.resident_warp_generation =
        it->second.resident_warp_generation;
    owner.window_generation =
        it->second.provisional_owner.window_generation;
    if (!it->second.live_bound || it->second.release_started ||
        !same_live_owner(owner, access.owner)) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusGroupNotRegistered;
  if (group->preaccept_pending ==
      std::numeric_limits<uint64_t>::max()) {
    return kStatusRegistryRejected;
  }
  ++group->preaccept_pending;
  return kStatusOk;
}

status_kind bridge_v0::cancel_live_access_preaccept(
    const address_range_registry::live_access_v0 &access) {
  const address_range_registry::status_kind validation =
      registry_.validate_live_access(access);
  if (validation != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    address_range_registry::live_owner_v0 owner = {};
    owner.owner_hw_sid = it->second.execution_owner.owner_hw_sid;
    owner.resident_warp_generation =
        it->second.resident_warp_generation;
    owner.window_generation =
        it->second.provisional_owner.window_generation;
    if (!it->second.live_bound || it->second.release_started ||
        !same_live_owner(owner, access.owner)) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusGroupNotRegistered;
  if (group->preaccept_pending == 0) return kStatusGroupConflict;
  --group->preaccept_pending;
  return kStatusOk;
}

status_kind bridge_v0::accept_preaccepted_live_access(
    const address_range_registry::live_access_v0 &access,
    address_range_registry::transaction_token_v0 *token) {
  if (token == NULL) return kStatusInvalidArgument;
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    address_range_registry::live_owner_v0 owner = {};
    owner.owner_hw_sid = it->second.execution_owner.owner_hw_sid;
    owner.resident_warp_generation =
        it->second.resident_warp_generation;
    owner.window_generation =
        it->second.provisional_owner.window_generation;
    if (!it->second.live_bound || it->second.release_started ||
        !same_live_owner(owner, access.owner)) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusGroupNotRegistered;
  if (group->preaccept_pending == 0) return kStatusGroupConflict;
  const address_range_registry::status_kind status =
      registry_.accept_live_access(access, token);
  if (status != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  --group->preaccept_pending;
  return kStatusOk;
}

address_range_registry::status_kind bridge_v0::complete_live_access(
    const address_range_registry::transaction_token_v0 &token) {
  return registry_.complete_transaction(token);
}

status_kind bridge_v0::preflight_ordinary_publication_store(
    const ordinary_store_request_v0 &request,
    ordinary_store_preflight_v0 *preflight) const {
  if (preflight == NULL) return kStatusInvalidArgument;
  std::memset(preflight, 0, sizeof(*preflight));
  preflight->registry_status =
      address_range_registry::kStatusOk;
  if (request.is_global_write && request.is_global_read) {
    return kStatusInvalidArgument;
  }
  if (!request.is_global_write && !request.is_global_read) {
    return kStatusOk;
  }
  if (request.aligned_32b_address %
          address_range_registry::kAddressChunkBytes !=
      0) {
    return kStatusInvalidGeometry;
  }

  address_range_registry::provisional_range_observation_v0
      range = {};
  preflight->registry_status = registry_.observe_provisional_range(
      request.aligned_32b_address, &range);
  if (preflight->registry_status ==
      address_range_registry::kStatusRecordNotFound) {
    preflight->registry_status =
        address_range_registry::kStatusOk;
    return kStatusOk;
  }
  if (preflight->registry_status !=
      address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }

  const registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::const_iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    if (!same_provisional_owner(it->second.provisional_owner,
                                range.owner)) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusGroupNotRegistered;

  allocation_identity::owner_v0 execution_owner = {};
  execution_owner.owner_hw_sid = request.owner_hw_sid;
  execution_owner.dynamic_warp_id = request.dynamic_warp_id;
  execution_owner.warp_id = request.warp_id;
  if (!same_execution_owner(group->execution_owner,
                            execution_owner)) {
    preflight->registry_status =
        address_range_registry::kStatusOwnerMismatch;
    return kStatusRegistryRejected;
  }
  if (request.data_size_bytes !=
          address_range_registry::kAddressChunkBytes ||
      request.sector_count != 1 ||
      !request.byte_mask_single_chunk ||
      !request.sector_mask_matches_address) {
    preflight->registry_status =
        address_range_registry::kStatusInvalidRange;
    return kStatusRegistryRejected;
  }
  if (request.active_lane_mask == 0 ||
      (request.active_lane_mask &
       (request.active_lane_mask - 1u)) != 0 ||
      request.active_lane_mask !=
          (uint32_t{1} << range.lane_id)) {
    preflight->registry_status =
        address_range_registry::kStatusLaneMismatch;
    return kStatusRegistryRejected;
  }
  if (range.object != address_range_registry::kObjectHandoff) {
    preflight->registry_status =
        address_range_registry::kStatusObjectKindMismatch;
    return kStatusRegistryRejected;
  }
  if (range.phase != address_range_registry::kPhaseProvisional &&
      range.phase != address_range_registry::kPhaseLive) {
    preflight->registry_status =
        address_range_registry::kStatusWrongPhase;
    return kStatusRegistryRejected;
  }
  const uint64_t range_offset =
      request.aligned_32b_address - range.range_base;
  if (range.range_byte_count !=
          4u * address_range_registry::kAddressChunkBytes ||
      range_offset % address_range_registry::kAddressChunkBytes != 0 ||
      range_offset / address_range_registry::kAddressChunkBytes >= 4u) {
    preflight->registry_status =
        address_range_registry::kStatusInvalidRange;
    return kStatusInvalidGeometry;
  }
  preflight->handoff_lane_base = range.range_base;
  preflight->lane_slot_chunk = static_cast<uint8_t>(
      range_offset / address_range_registry::kAddressChunkBytes);

  if (range.phase == address_range_registry::kPhaseLive) {
    const uint32_t shader_return_mask = 0xffff0fffu;
    const bool valid_shader_return =
        request.is_global_write &&
        range_offset == 3u * address_range_registry::kAddressChunkBytes &&
        (request.byte_mask & ~shader_return_mask) == 0;
    const bool valid_shader_builtin_read =
        request.is_global_read &&
        range_offset < 4u * address_range_registry::kAddressChunkBytes;
    if (!group->live_bound || group->release_started ||
        (!valid_shader_return && !valid_shader_builtin_read)) {
      preflight->registry_status =
          address_range_registry::kStatusByteMaskMismatch;
      return kStatusRegistryRejected;
    }
    preflight->live_access.owner = range.live_owner;
    preflight->live_access.lane_id = range.lane_id;
    preflight->live_access.object = range.object;
    preflight->live_access.access =
        valid_shader_builtin_read
            ? address_range_registry::kAccessHandoffShaderBuiltinRead
            : address_range_registry::kAccessHandoffShaderReturn;
    preflight->live_access.aligned_32b_address =
        request.aligned_32b_address;
    preflight->live_access.byte_mask = request.byte_mask;
    preflight->registry_status =
        registry_.validate_live_access(preflight->live_access);
    if (preflight->registry_status !=
        address_range_registry::kStatusOk) {
      return kStatusRegistryRejected;
    }
    preflight->candidate = 1;
    preflight->live_candidate = 1;
    return kStatusOk;
  }

  if (request.is_global_read) {
    preflight->registry_status =
        address_range_registry::kStatusWrongPhase;
    return kStatusRegistryRejected;
  }

  preflight->store.owner = range.owner;
  preflight->store.lane_id = range.lane_id;
  preflight->store.object = range.object;
  preflight->store.aligned_32b_address =
      request.aligned_32b_address;
  preflight->store.byte_mask = request.byte_mask;
  preflight->registry_status =
      registry_.validate_provisional_publication_store(
          preflight->store);
  if (preflight->registry_status !=
      address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  preflight->candidate = 1;
  return kStatusOk;
}

status_kind bridge_v0::service_provisional_publication_fence(
    uint32_t owner_hw_sid, uint32_t dynamic_warp_id,
    uint32_t warp_id, uint8_t release_ready,
    provisional_group_drain_v0 *drain) {
  if (drain == NULL) return kStatusInvalidArgument;
  std::memset(drain, 0, sizeof(*drain));
  drain->registry_status =
      address_range_registry::kStatusOk;
  registered_group_v0 *group = NULL;
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    const allocation_identity::owner_v0 &owner =
        it->second.execution_owner;
    if (owner.owner_hw_sid != owner_hw_sid ||
        owner.dynamic_warp_id != dynamic_warp_id ||
        owner.warp_id != warp_id) {
      continue;
    }
    if (group != NULL) return kStatusGroupConflict;
    group = &it->second;
  }
  if (group == NULL) return kStatusOk;
  drain->registered = 1;
  drain->fence_armed = group->fence_armed;
  drain->preaccept_pending = group->preaccept_pending;
  drain->publication_coverage_complete =
      publication_coverage_complete(*group);
  if (!group->fence_armed) return kStatusOk;
  drain->registry_status =
      registry_.provisional_group_outstanding(
          group->provisional_owner,
          &drain->outstanding_transactions);
  if (drain->registry_status !=
      address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  drain->wait_required =
      drain->preaccept_pending != 0 ||
      drain->outstanding_transactions != 0 ||
      !drain->publication_coverage_complete;
  if (!drain->wait_required && release_ready) {
    group->fence_armed = 0;
    drain->fence_consumed = 1;
    drain->release_kind = kFenceReleaseAckRetain;
  }
  return kStatusOk;
}

status_kind bridge_v0::begin_or_poll_first_submit_live_bind(
    const first_submit_bind_request_v0 &request,
    first_submit_bind_result_v0 *result) {
  if (result == NULL) return kStatusInvalidArgument;
  std::memset(result, 0, sizeof(*result));
  result->authority_status = allocation_identity::kStatusInvalidArgument;
  result->registry_status = address_range_registry::kStatusInvalidArgument;

  lane_publication_request_v0 geometry = {};
  geometry.slot = request.slot;
  geometry.owner = request.owner;
  geometry.allocation_ranges = request.allocation_ranges;
  geometry.publication_warp_uid = 1;
  geometry.active_mask = request.active_mask;
  geometry.capacity_lane_slots = request.capacity_lane_slots;
  geometry.context_lane_stride_bytes = request.context_lane_stride_bytes;
  geometry.handoff_lane_stride_bytes = request.handoff_lane_stride_bytes;
  geometry.handoff_allowed_publication_masks =
      request.handoff_allowed_publication_masks;
  geometry.handoff_allowed_publication_mask_count =
      request.handoff_allowed_publication_mask_count;
  geometry.lane_id = first_active_lane(request.active_mask);
  const status_kind geometry_status = validate_geometry(geometry);
  if (geometry_status != kStatusOk) return geometry_status;

  result->authority_status = authority_.lookup_for_submit(
      request.slot, request.owner, request.active_mask,
      request.allocation_ranges, &result->identity);
  if (result->authority_status != allocation_identity::kStatusOk) {
    return kStatusAuthorityRejected;
  }
  std::map<uint64_t, registered_group_v0>::iterator found =
      registered_groups_.find(result->identity.record_id);
  if (found == registered_groups_.end()) return kStatusGroupNotRegistered;
  registered_group_v0 &group = found->second;
  if (!same_execution_owner(group.execution_owner, request.owner) ||
      !same_allocation_slot(group.slot, request.slot) ||
      !same_allocation_ranges(group.allocation_ranges,
                              request.allocation_ranges) ||
      group.active_mask != request.active_mask ||
      group.capacity_lane_slots != request.capacity_lane_slots ||
      group.context_lane_stride_bytes !=
          request.context_lane_stride_bytes ||
      group.handoff_lane_stride_bytes !=
          request.handoff_lane_stride_bytes ||
      group.handoff_allowed_publication_masks.size() !=
          request.handoff_allowed_publication_mask_count ||
      !std::equal(group.handoff_allowed_publication_masks.begin(),
                  group.handoff_allowed_publication_masks.end(),
                  request.handoff_allowed_publication_masks)) {
    return kStatusGroupConflict;
  }
  fill_bind_ticket(result->identity, &result->ticket);
  result->preaccept_pending = group.preaccept_pending;
  result->bind_started = group.bind_started;
  result->live_bound = group.live_bound;
  result->resident_warp_generation = group.resident_warp_generation;
  result->publication_coverage_complete =
      publication_coverage_complete(group);

  std::vector<std::vector<uint32_t> > owned_masks;
  std::vector<address_range_registry::range_spec_v0> ranges;
  const status_kind range_status =
      make_registered_group_ranges(group, &owned_masks, &ranges);
  if (range_status != kStatusOk) return range_status;
  if (group.live_bound) {
    address_range_registry::live_owner_v0 live = {};
    live.owner_hw_sid = group.provisional_owner.owner_hw_sid;
    live.resident_warp_generation = group.resident_warp_generation;
    live.window_generation = group.provisional_owner.window_generation;
    result->registry_status = registry_.validate_live_group(
        live, group.active_mask, ranges.data(), ranges.size());
    return result->registry_status == address_range_registry::kStatusOk
               ? kStatusOk
               : kStatusRegistryRejected;
  }

  if (!result->publication_coverage_complete) {
    result->registry_status = address_range_registry::kStatusOk;
    result->wait_required = 1;
    return kStatusOk;
  }

  if (group.preaccept_pending != 0) {
    result->registry_status = registry_.provisional_group_outstanding(
        group.provisional_owner, &result->outstanding_transactions);
    if (result->registry_status != address_range_registry::kStatusOk) {
      return kStatusRegistryRejected;
    }
    result->wait_required = 1;
    return kStatusOk;
  }

  if (!group.bind_started) {
    result->registry_status = registry_.begin_live_bind(
        group.provisional_owner, group.active_mask,
        ranges.data(), ranges.size());
    if (result->registry_status != address_range_registry::kStatusOk &&
        result->registry_status !=
            address_range_registry::kStatusOutstandingTransactions) {
      return kStatusRegistryRejected;
    }
    group.bind_started = 1;
    result->bind_started = 1;
  }
  result->registry_status = registry_.provisional_group_outstanding(
      group.provisional_owner, &result->outstanding_transactions);
  if (result->registry_status != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  result->wait_required = result->outstanding_transactions != 0;
  return kStatusOk;
}

status_kind bridge_v0::commit_first_submit_live_bind(
    const first_submit_bind_ticket_v0 &ticket,
    uint32_t resident_warp_generation,
    first_submit_bind_result_v0 *result) {
  if (result == NULL || !ticket.valid || resident_warp_generation == 0) {
    return kStatusInvalidArgument;
  }
  std::memset(result, 0, sizeof(*result));
  result->authority_status = allocation_identity::kStatusOk;
  result->registry_status = address_range_registry::kStatusInvalidArgument;
  std::map<uint64_t, registered_group_v0>::iterator found =
      registered_groups_.find(ticket.allocation_record_id);
  if (found == registered_groups_.end()) return kStatusGroupNotRegistered;
  registered_group_v0 &group = found->second;
  if (!ticket_matches_group(ticket, found->first, group.execution_owner,
                            group.active_mask, group.provisional_owner)) {
    return kStatusGroupConflict;
  }
  result->ticket = ticket;
  result->bind_started = group.bind_started;
  result->preaccept_pending = group.preaccept_pending;
  result->resident_warp_generation = group.resident_warp_generation;
  result->live_bound = group.live_bound;
  result->publication_coverage_complete =
      publication_coverage_complete(group);
  if (!result->publication_coverage_complete) {
    result->registry_status = address_range_registry::kStatusOk;
    result->wait_required = 1;
    return kStatusOk;
  }
  if (!group.bind_started || group.preaccept_pending != 0) {
    result->wait_required = group.preaccept_pending != 0;
    return group.bind_started ? kStatusOk : kStatusGroupConflict;
  }

  std::vector<std::vector<uint32_t> > owned_masks;
  std::vector<address_range_registry::range_spec_v0> ranges;
  const status_kind range_status =
      make_registered_group_ranges(group, &owned_masks, &ranges);
  if (range_status != kStatusOk) return range_status;
  if (group.live_bound) {
    if (group.resident_warp_generation != resident_warp_generation) {
      return kStatusGroupConflict;
    }
  } else {
    result->registry_status = registry_.provisional_group_outstanding(
        group.provisional_owner, &result->outstanding_transactions);
    if (result->registry_status != address_range_registry::kStatusOk) {
      return kStatusRegistryRejected;
    }
    if (result->outstanding_transactions != 0) {
      result->wait_required = 1;
      return kStatusOk;
    }
    result->registry_status = registry_.commit_live_bind(
        group.provisional_owner, resident_warp_generation);
    if (result->registry_status != address_range_registry::kStatusOk) {
      return kStatusRegistryRejected;
    }
    group.resident_warp_generation = resident_warp_generation;
    group.live_bound = 1;
  }

  address_range_registry::live_owner_v0 live = {};
  live.owner_hw_sid = group.provisional_owner.owner_hw_sid;
  live.resident_warp_generation = resident_warp_generation;
  live.window_generation = group.provisional_owner.window_generation;
  result->registry_status = registry_.validate_live_group(
      live, group.active_mask, ranges.data(), ranges.size());
  if (result->registry_status != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  result->resident_warp_generation = resident_warp_generation;
  result->live_bound = 1;
  result->wait_required = 0;
  return kStatusOk;
}

status_kind bridge_v0::validate_first_submit_live_bind(
    const first_submit_bind_request_v0 &request,
    uint32_t resident_warp_generation,
    first_submit_bind_result_v0 *result) const {
  if (result == NULL || resident_warp_generation == 0 ||
      request.handoff_allowed_publication_masks == NULL ||
      request.handoff_allowed_publication_mask_count == 0) {
    return kStatusInvalidArgument;
  }
  std::memset(result, 0, sizeof(*result));
  result->authority_status = authority_.lookup_for_submit(
      request.slot, request.owner, request.active_mask,
      request.allocation_ranges, &result->identity);
  if (result->authority_status != allocation_identity::kStatusOk) {
    return kStatusAuthorityRejected;
  }
  std::map<uint64_t, registered_group_v0>::const_iterator found =
      registered_groups_.find(result->identity.record_id);
  if (found == registered_groups_.end()) return kStatusGroupNotRegistered;
  const registered_group_v0 &group = found->second;
  if (!group.live_bound ||
      group.resident_warp_generation != resident_warp_generation ||
      !same_execution_owner(group.execution_owner, request.owner) ||
      !same_allocation_slot(group.slot, request.slot) ||
      !same_allocation_ranges(group.allocation_ranges,
                              request.allocation_ranges) ||
      group.active_mask != request.active_mask ||
      group.capacity_lane_slots != request.capacity_lane_slots ||
      group.context_lane_stride_bytes !=
          request.context_lane_stride_bytes ||
      group.handoff_lane_stride_bytes !=
          request.handoff_lane_stride_bytes ||
      group.handoff_allowed_publication_masks.size() !=
          request.handoff_allowed_publication_mask_count ||
      !std::equal(group.handoff_allowed_publication_masks.begin(),
                  group.handoff_allowed_publication_masks.end(),
                  request.handoff_allowed_publication_masks)) {
    return kStatusGroupConflict;
  }
  std::vector<std::vector<uint32_t> > owned_masks;
  std::vector<address_range_registry::range_spec_v0> ranges;
  const status_kind range_status =
      make_registered_group_ranges(group, &owned_masks, &ranges);
  if (range_status != kStatusOk) return range_status;
  address_range_registry::live_owner_v0 live = {};
  live.owner_hw_sid = group.provisional_owner.owner_hw_sid;
  live.resident_warp_generation = resident_warp_generation;
  live.window_generation = group.provisional_owner.window_generation;
  result->registry_status = registry_.validate_live_group(
      live, group.active_mask, ranges.data(), ranges.size());
  if (result->registry_status != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  fill_bind_ticket(result->identity, &result->ticket);
  result->bind_started = 1;
  result->live_bound = 1;
  result->resident_warp_generation = resident_warp_generation;
  return kStatusOk;
}

status_kind bridge_v0::validate_resubmit_live_subset(
    const first_submit_bind_request_v0 &selected_request,
    uint32_t previous_active_mask, uint32_t resident_warp_generation,
    resubmit_live_validation_result_v0 *result) const {
  if (result == NULL) return kStatusInvalidArgument;
  std::memset(result, 0, sizeof(*result));
  result->authority_status = allocation_identity::kStatusInvalidArgument;
  result->registry_status = address_range_registry::kStatusInvalidArgument;
  if (previous_active_mask == 0 || resident_warp_generation == 0 ||
      selected_request.active_mask == 0 ||
      (selected_request.active_mask & ~previous_active_mask) != 0) {
    return kStatusInvalidArgument;
  }
  result->previous_active_mask = previous_active_mask;
  result->selected_active_mask = selected_request.active_mask;
  result->resident_warp_generation = resident_warp_generation;

  lane_publication_request_v0 geometry = {};
  geometry.slot = selected_request.slot;
  geometry.owner = selected_request.owner;
  geometry.allocation_ranges = selected_request.allocation_ranges;
  geometry.publication_warp_uid = 1;
  geometry.active_mask = selected_request.active_mask;
  geometry.capacity_lane_slots = selected_request.capacity_lane_slots;
  geometry.context_lane_stride_bytes =
      selected_request.context_lane_stride_bytes;
  geometry.handoff_lane_stride_bytes =
      selected_request.handoff_lane_stride_bytes;
  geometry.handoff_allowed_publication_masks =
      selected_request.handoff_allowed_publication_masks;
  geometry.handoff_allowed_publication_mask_count =
      selected_request.handoff_allowed_publication_mask_count;
  geometry.lane_id = first_active_lane(selected_request.active_mask);
  const status_kind geometry_status = validate_geometry(geometry);
  if (geometry_status != kStatusOk) return geometry_status;

  std::map<uint64_t, registered_group_v0>::const_iterator found =
      registered_groups_.end();
  for (std::map<uint64_t, registered_group_v0>::const_iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    const registered_group_v0 &candidate = it->second;
    if (!same_execution_owner(candidate.execution_owner,
                              selected_request.owner) ||
        !same_allocation_slot(candidate.slot, selected_request.slot) ||
        !same_allocation_ranges(candidate.allocation_ranges,
                                selected_request.allocation_ranges) ||
        candidate.capacity_lane_slots !=
            selected_request.capacity_lane_slots ||
        candidate.context_lane_stride_bytes !=
            selected_request.context_lane_stride_bytes ||
        candidate.handoff_lane_stride_bytes !=
            selected_request.handoff_lane_stride_bytes ||
        candidate.handoff_allowed_publication_masks.size() !=
            selected_request.handoff_allowed_publication_mask_count ||
        !std::equal(candidate.handoff_allowed_publication_masks.begin(),
                    candidate.handoff_allowed_publication_masks.end(),
                    selected_request.handoff_allowed_publication_masks)) {
      continue;
    }
    if (found != registered_groups_.end()) return kStatusGroupConflict;
    found = it;
  }
  if (found == registered_groups_.end()) return kStatusGroupNotRegistered;
  const registered_group_v0 &group = found->second;
  if (!group.live_bound || group.release_started ||
      group.preaccept_pending != 0 ||
      group.resident_warp_generation != resident_warp_generation ||
      (previous_active_mask & ~group.active_mask) != 0 ||
      (selected_request.active_mask & ~group.active_mask) != 0) {
    return kStatusGroupConflict;
  }

  result->authority_status = authority_.lookup_for_submit(
      group.slot, group.execution_owner, group.active_mask,
      group.allocation_ranges, &result->identity);
  if (result->authority_status != allocation_identity::kStatusOk ||
      result->identity.record_id != found->first ||
      !same_allocation_slot(result->identity.slot, group.slot) ||
      !same_execution_owner(result->identity.owner,
                            group.execution_owner) ||
      !same_allocation_ranges(result->identity.ranges,
                              group.allocation_ranges) ||
      result->identity.active_mask != group.active_mask ||
      result->identity.launch_allocation_generation !=
          group.provisional_owner.launch_allocation_generation ||
      result->identity.window_generation !=
          group.provisional_owner.window_generation) {
    return kStatusAuthorityRejected;
  }

  const std::pair<uint64_t, uint32_t> publication_key(
      group.slot.allocation_domain_id, group.slot.allocation_slot_id);
  std::map<std::pair<uint64_t, uint32_t>,
           publication_identity_observation_v0>::const_iterator publication =
      publication_identity_observations_.find(publication_key);
  if (publication == publication_identity_observations_.end() ||
      publication->second.record_id != found->first) {
    return kStatusGroupConflict;
  }

  std::vector<std::vector<uint32_t> > owned_masks;
  std::vector<address_range_registry::range_spec_v0> ranges;
  const status_kind range_status =
      make_registered_group_ranges(group, &owned_masks, &ranges);
  if (range_status != kStatusOk) return range_status;
  address_range_registry::live_owner_v0 live = {};
  live.owner_hw_sid = group.provisional_owner.owner_hw_sid;
  live.resident_warp_generation = resident_warp_generation;
  live.window_generation = group.provisional_owner.window_generation;
  result->registry_status = registry_.validate_live_group(
      live, group.active_mask, ranges.data(), ranges.size());
  if (result->registry_status != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }

  result->public_active_mask = group.active_mask;
  result->live_bound = 1;
  return kStatusOk;
}

status_kind bridge_v0::begin_or_poll_retire_live_release(
    const first_submit_bind_request_v0 &request,
    uint32_t resident_warp_generation,
    retire_live_release_result_v0 *result) {
  if (result == NULL || resident_warp_generation == 0) {
    return kStatusInvalidArgument;
  }
  std::memset(result, 0, sizeof(*result));
  result->authority_status = allocation_identity::kStatusInvalidArgument;
  result->registry_status = address_range_registry::kStatusInvalidArgument;

  lane_publication_request_v0 geometry = {};
  geometry.slot = request.slot;
  geometry.owner = request.owner;
  geometry.allocation_ranges = request.allocation_ranges;
  geometry.publication_warp_uid = 1;
  geometry.active_mask = request.active_mask;
  geometry.capacity_lane_slots = request.capacity_lane_slots;
  geometry.context_lane_stride_bytes = request.context_lane_stride_bytes;
  geometry.handoff_lane_stride_bytes = request.handoff_lane_stride_bytes;
  geometry.handoff_allowed_publication_masks =
      request.handoff_allowed_publication_masks;
  geometry.handoff_allowed_publication_mask_count =
      request.handoff_allowed_publication_mask_count;
  geometry.lane_id = first_active_lane(request.active_mask);
  const status_kind geometry_status = validate_geometry(geometry);
  if (geometry_status != kStatusOk) return geometry_status;

  std::map<uint64_t, registered_group_v0>::iterator found =
      registered_groups_.end();
  for (std::map<uint64_t, registered_group_v0>::iterator it =
           registered_groups_.begin();
       it != registered_groups_.end(); ++it) {
    registered_group_v0 &candidate = it->second;
    if (!same_execution_owner(candidate.execution_owner, request.owner) ||
        !same_allocation_slot(candidate.slot, request.slot) ||
        !same_allocation_ranges(candidate.allocation_ranges,
                                request.allocation_ranges) ||
        candidate.active_mask != request.active_mask) {
      continue;
    }
    if (found != registered_groups_.end()) return kStatusGroupConflict;
    found = it;
  }
  if (found == registered_groups_.end()) return kStatusGroupNotRegistered;
  registered_group_v0 &group = found->second;
  if (!group.live_bound ||
      group.resident_warp_generation != resident_warp_generation ||
      group.capacity_lane_slots != request.capacity_lane_slots ||
      group.context_lane_stride_bytes != request.context_lane_stride_bytes ||
      group.handoff_lane_stride_bytes != request.handoff_lane_stride_bytes ||
      group.handoff_allowed_publication_masks.size() !=
          request.handoff_allowed_publication_mask_count ||
      !std::equal(group.handoff_allowed_publication_masks.begin(),
                  group.handoff_allowed_publication_masks.end(),
                  request.handoff_allowed_publication_masks)) {
    return kStatusGroupConflict;
  }

  result->identity.record_id = found->first;
  result->identity.slot = group.slot;
  result->identity.owner = group.execution_owner;
  result->identity.ranges = group.allocation_ranges;
  result->identity.active_mask = group.active_mask;
  result->identity.launch_allocation_generation =
      group.provisional_owner.launch_allocation_generation;
  result->identity.window_generation =
      group.provisional_owner.window_generation;
  result->resident_warp_generation = resident_warp_generation;
  result->release_started = group.release_started;

  const std::pair<uint64_t, uint32_t> publication_key(
      group.slot.allocation_domain_id, group.slot.allocation_slot_id);
  std::map<std::pair<uint64_t, uint32_t>,
           publication_identity_observation_v0>::iterator publication =
      publication_identity_observations_.find(publication_key);
  if (publication == publication_identity_observations_.end() ||
      publication->second.record_id != found->first) {
    return kStatusGroupConflict;
  }

  std::vector<std::vector<uint32_t> > owned_masks;
  std::vector<address_range_registry::range_spec_v0> ranges;
  const status_kind range_status =
      make_registered_group_ranges(group, &owned_masks, &ranges);
  if (range_status != kStatusOk) return range_status;
  address_range_registry::live_owner_v0 live = {};
  live.owner_hw_sid = group.provisional_owner.owner_hw_sid;
  live.resident_warp_generation = resident_warp_generation;
  live.window_generation = group.provisional_owner.window_generation;

  if (!group.release_started) {
    allocation_identity::allocation_identity_v0 authoritative = {};
    result->authority_status = authority_.lookup_for_submit(
        request.slot, request.owner, request.active_mask,
        request.allocation_ranges, &authoritative);
    if (result->authority_status != allocation_identity::kStatusOk ||
        authoritative.record_id != result->identity.record_id ||
        !same_allocation_slot(authoritative.slot, result->identity.slot) ||
        !same_execution_owner(authoritative.owner, result->identity.owner) ||
        !same_allocation_ranges(authoritative.ranges,
                                result->identity.ranges) ||
        authoritative.active_mask != result->identity.active_mask ||
        authoritative.launch_allocation_generation !=
            result->identity.launch_allocation_generation ||
        authoritative.window_generation !=
            result->identity.window_generation) {
      return kStatusAuthorityRejected;
    }
    result->registry_status = registry_.validate_live_group(
        live, group.active_mask, ranges.data(), ranges.size());
    if (result->registry_status != address_range_registry::kStatusOk) {
      return kStatusRegistryRejected;
    }
    if (group.preaccept_pending != 0) {
      result->wait_required = 1;
      return kStatusOk;
    }
    result->authority_status = authority_.begin_release(result->identity);
    if (result->authority_status != allocation_identity::kStatusOk) {
      return kStatusAuthorityRejected;
    }
    result->registry_status = registry_.release_live_group(live);
    if (result->registry_status != address_range_registry::kStatusOk &&
        result->registry_status !=
            address_range_registry::kStatusOutstandingTransactions) {
      return kStatusRegistryRejected;
    }
    group.release_started = 1;
    result->release_started = 1;
  } else {
    result->authority_status = allocation_identity::kStatusOk;
  }

  address_range_registry::live_release_observation_v0 release = {};
  result->registry_status = registry_.poll_live_group_release(live, &release);
  if (result->registry_status != address_range_registry::kStatusOk) {
    return kStatusRegistryRejected;
  }
  result->outstanding_transactions = release.outstanding_transactions;
  if (!release.released) {
    result->wait_required = 1;
    return kStatusOk;
  }

  result->authority_status = authority_.commit_release(result->identity);
  if (result->authority_status != allocation_identity::kStatusOk) {
    return kStatusAuthorityRejected;
  }
  publication_identity_observations_.erase(publication);
  registered_groups_.erase(found);
  result->released = 1;
  return kStatusOk;
}

bridge_snapshot_v0 bridge_v0::snapshot() const {
  bridge_snapshot_v0 result = {};
  result.authority = authority_.snapshot();
  result.registry = registry_.snapshot();
  result.registered_group_count = registered_groups_.size();
  return result;
}

bridge_v0 &shared_bridge() {
  static bridge_v0 bridge;
  return bridge;
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
