#include "rtcore_v04_private_shared_backing_internal.h"

#include <cstring>
#include <limits>
#include <set>

namespace rtcore {
namespace v04 {
namespace private_shared {
namespace {

static uint32_t lane_bit(uint32_t lane) {
  return lane < kLaneCapacity ? uint32_t{1} << lane : 0;
}

static uint32_t count_lanes(uint32_t mask) {
  uint32_t count = 0;
  while (mask != 0) {
    mask &= mask - 1;
    ++count;
  }
  return count;
}

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static bool owner_is_valid(const private_frontier::owner_binding_v0 &owner,
                           uint32_t owner_hw_sid, uint8_t resident_warp_slot,
                           uint8_t lane) {
  return owner.owner_hw_sid == owner_hw_sid &&
         owner.resident_warp_id == resident_warp_slot &&
         owner.request_identity != 0 && owner.generation != 0 &&
         owner.private_slot_id < 256 && owner.lane_id == lane &&
         bytes_are_zero(owner.reserved_zero, sizeof(owner.reserved_zero));
}

static bool write_envelope_is_valid(const shared_write_v0 &operation) {
  return operation.valid && operation.address_space == kAddressSpaceShared &&
         operation.address_mode == kAddressModePrivateField &&
         operation.access_operation == kAccessOperationWrite &&
         operation.destination == kDestinationPrivateCommitAck &&
         bytes_are_zero(operation.reserved_zero,
                        sizeof(operation.reserved_zero)) &&
         operation.reserved_zero1 == 0;
}

static bool is_init_transaction(const shared_write_v0 &operation) {
  return operation.operation_seq == 0 && operation.commit_epoch == 0;
}

static bool is_runtime_transaction(const shared_write_v0 &operation) {
  return operation.operation_seq != 0 && operation.commit_epoch != 0;
}

static bool same_operation(const shared_write_v0 &lhs,
                           const shared_write_v0 &rhs) {
  return lhs.valid == rhs.valid && lhs.address_space == rhs.address_space &&
         lhs.address_mode == rhs.address_mode &&
         lhs.access_operation == rhs.access_operation &&
         lhs.destination == rhs.destination &&
         std::memcmp(lhs.reserved_zero, rhs.reserved_zero,
                     sizeof(lhs.reserved_zero)) == 0 &&
         private_frontier::owners_equal(lhs.owner, rhs.owner) &&
         lhs.operation_seq == rhs.operation_seq &&
         lhs.commit_epoch == rhs.commit_epoch &&
         lhs.memory_op_seq == rhs.memory_op_seq &&
         lhs.chunk_id == rhs.chunk_id && lhs.chunk_count == rhs.chunk_count &&
         lhs.field_kind == rhs.field_kind &&
         lhs.reserved_zero1 == rhs.reserved_zero1 &&
         lhs.aligned_32b_address == rhs.aligned_32b_address &&
         lhs.byte_mask == rhs.byte_mask &&
         std::memcmp(lhs.payload, rhs.payload, sizeof(lhs.payload)) == 0 &&
         lhs.enqueue_cycle == rhs.enqueue_cycle &&
         lhs.accepted_cycle == rhs.accepted_cycle &&
         lhs.ack_cycle == rhs.ack_cycle;
}

static unsigned count_mask_bits(uint32_t mask) {
  unsigned count = 0;
  while (mask != 0) {
    mask &= mask - 1;
    ++count;
  }
  return count;
}

static uint64_t private_region_base(uint32_t owner_hw_sid) {
  return UINT64_C(0xff00000000000000) +
         static_cast<uint64_t>(owner_hw_sid) * UINT64_C(0x1000000);
}

static uint64_t private_slot_base(
    const private_frontier::owner_binding_v0 &owner) {
  return private_region_base(owner.owner_hw_sid) +
         static_cast<uint64_t>(owner.private_slot_id) *
             private_frontier::kPrivateDataSlotBytes;
}

static bool field_contains_offset(uint8_t field_kind, uint32_t offset) {
  switch (field_kind) {
    case private_frontier::kFieldFrontierMetadata:
      return offset >= private_frontier::kFrontierMetadataOffset &&
             offset < private_frontier::kFrontierMetadataOffset +
                          private_frontier::kFrontierMetadataBytes;
    case private_frontier::kFieldFrontierEntry:
      return offset >= private_frontier::kFrontierEntriesOffset &&
             offset < private_frontier::kFrontierEntriesEnd;
    case private_frontier::kFieldTransitionSpill:
      return offset >= private_frontier::kTransitionSpillOffset &&
             offset < private_frontier::kTransitionSpillEnd;
    case private_frontier::kFieldMutableRayState:
      return offset >= private_frontier::kMutableRayStateOffset &&
             offset < private_frontier::kMutableRayStateOffset +
                          private_frontier::kMutableRayStateBytes;
    case private_frontier::kFieldAsDecodeContext:
      return offset >= private_frontier::kAsDecodeContextOffset &&
             offset < private_frontier::kAsDecodeContextOffset +
                          private_frontier::kAsDecodeContextBytes;
    case private_frontier::kFieldCommittedHit:
      return offset >= private_frontier::kCommittedHitOffset &&
             offset < private_frontier::kCommittedHitOffset +
                          private_frontier::kCommittedHitBytes;
    default:
      return false;
  }
}

static bool runtime_operation_structure_is_valid(
    const shared_write_v0 &operation) {
  if (!write_envelope_is_valid(operation) ||
      !is_runtime_transaction(operation) || operation.memory_op_seq == 0 ||
      operation.chunk_count == 0 || operation.chunk_count > 16 ||
      operation.memory_op_seq > operation.chunk_count ||
      operation.chunk_id != operation.memory_op_seq - 1 ||
      (operation.aligned_32b_address %
       private_frontier::kSharedAccessChunkBytes) != 0 ||
      operation.byte_mask == 0) {
    return false;
  }
  const uint64_t slot_base = private_slot_base(operation.owner);
  if (operation.aligned_32b_address < slot_base ||
      operation.aligned_32b_address >
          slot_base + private_frontier::kPrivateDataSlotBytes -
              private_frontier::kSharedAccessChunkBytes) {
    return false;
  }
  const uint32_t chunk_offset =
      static_cast<uint32_t>(operation.aligned_32b_address - slot_base);
  for (unsigned byte = 0; byte < private_frontier::kSharedAccessChunkBytes;
       ++byte) {
    const bool selected = (operation.byte_mask & (uint32_t{1} << byte)) != 0;
    if (selected &&
        !field_contains_offset(operation.field_kind, chunk_offset + byte)) {
      return false;
    }
    if (!selected && operation.payload[byte] != 0) return false;
  }
  return count_mask_bits(operation.byte_mask) != 0;
}

static bool same_runtime_identity(const shared_write_v0 &lhs,
                                  const shared_write_v0 &rhs) {
  return write_envelope_is_valid(lhs) && write_envelope_is_valid(rhs) &&
         is_runtime_transaction(lhs) && is_runtime_transaction(rhs) &&
         private_frontier::owners_equal(lhs.owner, rhs.owner) &&
         lhs.operation_seq == rhs.operation_seq &&
         lhs.commit_epoch == rhs.commit_epoch &&
         lhs.memory_op_seq == rhs.memory_op_seq;
}

static bool has_runtime_identity(const backing_state_v0 &state,
                                 const shared_write_v0 &operation) {
  if (state.offer_valid && same_runtime_identity(state.offered, operation)) {
    return true;
  }
  for (std::deque<shared_write_v0>::const_iterator it =
           state.shared_queue.begin();
       it != state.shared_queue.end(); ++it) {
    if (same_runtime_identity(*it, operation)) return true;
  }
  for (std::deque<shared_write_v0>::const_iterator it =
           state.outstanding.begin();
       it != state.outstanding.end(); ++it) {
    if (same_runtime_identity(*it, operation)) return true;
  }
  return false;
}

static bool operation_belongs_to_owner(
    const shared_write_v0 &operation,
    const private_frontier::owner_binding_v0 &owner) {
  return operation.valid &&
         private_frontier::owners_equal(operation.owner, owner);
}

static bool has_in_flight_owner(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner) {
  if (state.offer_valid && operation_belongs_to_owner(state.offered, owner)) {
    return true;
  }
  for (std::deque<shared_write_v0>::const_iterator it =
           state.shared_queue.begin();
       it != state.shared_queue.end(); ++it) {
    if (operation_belongs_to_owner(*it, owner)) return true;
  }
  for (std::deque<shared_write_v0>::const_iterator it =
           state.outstanding.begin();
       it != state.outstanding.end(); ++it) {
    if (operation_belongs_to_owner(*it, owner)) return true;
  }
  return false;
}

static bool warp_matches(const resident_warp_state_v0 &warp,
                         uint32_t owner_hw_sid, uint32_t warp_uid,
                         uint32_t warp_id) {
  return warp.live && warp.owner_hw_sid == owner_hw_sid &&
         warp.warp_uid == warp_uid && warp.warp_id == warp_id;
}

static bool all_init_chunks_acknowledged(const lane_slot_state_v0 &lane) {
  if (!lane.live || lane.init_plan.access_count == 0 ||
      lane.init_plan.access_count >= 32) {
    return false;
  }
  const uint32_t expected = (uint32_t{1} << lane.init_plan.access_count) - 1;
  return lane.acknowledged_chunk_mask == expected;
}

static void materialize_operation(const lane_slot_state_v0 &lane,
                                  uint8_t chunk_id, uint64_t enqueue_cycle,
                                  shared_write_v0 *operation) {
  std::memset(operation, 0, sizeof(*operation));
  const private_frontier::shared_chunk_access_v0 &access =
      lane.init_plan.accesses[chunk_id];
  operation->valid = true;
  operation->address_space = kAddressSpaceShared;
  operation->address_mode = kAddressModePrivateField;
  operation->access_operation = kAccessOperationWrite;
  operation->destination = kDestinationPrivateCommitAck;
  operation->owner = lane.owner;
  operation->memory_op_seq = kInitMemoryOpSequence;
  operation->chunk_id = chunk_id;
  operation->chunk_count = lane.init_plan.access_count;
  operation->field_kind = access.field_kind;
  operation->aligned_32b_address = access.aligned_32b_address;
  operation->byte_mask = access.byte_mask;
  operation->enqueue_cycle = enqueue_cycle;

  const uint32_t first_byte =
      access.slot_byte_offset % private_frontier::kSharedAccessChunkBytes;
  const uint32_t chunk_slot_base = access.slot_byte_offset - first_byte;
  for (uint32_t byte = 0; byte < private_frontier::kSharedAccessChunkBytes;
       ++byte) {
    if ((access.byte_mask & (uint32_t{1} << byte)) == 0) continue;
    operation->payload[byte] = lane.staging_slot.bytes[chunk_slot_base + byte];
  }
}

static bool validate_operation_against_lane(const shared_write_v0 &operation,
                                            const lane_slot_state_v0 &lane) {
  if (!write_envelope_is_valid(operation) || !is_init_transaction(operation) ||
      operation.memory_op_seq != kInitMemoryOpSequence ||
      operation.chunk_count != lane.init_plan.access_count ||
      operation.chunk_id >= lane.init_plan.access_count ||
      operation.field_kind !=
          lane.init_plan.accesses[operation.chunk_id].field_kind) {
    return false;
  }
  shared_write_v0 expected = {};
  materialize_operation(lane, operation.chunk_id, operation.enqueue_cycle,
                        &expected);
  expected.accepted_cycle = operation.accepted_cycle;
  expected.ack_cycle = operation.ack_cycle;
  return same_operation(operation, expected);
}

static bool validate_runtime_operation_against_lane(
    const shared_write_v0 &operation, const lane_slot_state_v0 &lane) {
  return lane.live &&
         private_frontier::owners_equal(operation.owner, lane.owner) &&
         runtime_operation_structure_is_valid(operation);
}

static status_kind validate_release_lanes(const backing_state_v0 &state,
                                          const resident_warp_state_v0 &warp,
                                          uint32_t release_mask) {
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((release_mask & lane_bit(lane)) == 0) continue;
    if (!warp.lanes[lane].live ||
        has_in_flight_owner(state, warp.lanes[lane].owner)) {
      return kStatusInFlightRelease;
    }
  }
  return kStatusOk;
}

}  // namespace

void initialize(backing_state_v0 *state, uint32_t owner_hw_sid) {
  if (state == NULL) return;
  *state = backing_state_v0();
  state->initialized = true;
  state->owner_hw_sid = owner_hw_sid;
  state->mutation_epoch = 1;
}

static status_kind prepare_new_warp_impl(
    const backing_state_v0 &state, uint32_t warp_uid, uint32_t warp_id,
    uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    const private_frontier::root_private_operands_v0
        root_operands[kLaneCapacity],
    new_warp_plan_v0 *plan) {
  if (plan == NULL || owners == NULL || !state.initialized ||
      active_mask == 0) {
    return kStatusInvalidArgument;
  }
  std::memset(plan, 0, sizeof(*plan));

  uint8_t resident_slot = kResidentWarpCapacity;
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((active_mask & lane_bit(lane)) == 0) continue;
    if (owners[lane].resident_warp_id >= kResidentWarpCapacity) {
      return kStatusInvalidOwner;
    }
    if (resident_slot == kResidentWarpCapacity) {
      resident_slot = static_cast<uint8_t>(owners[lane].resident_warp_id);
    } else if (resident_slot != owners[lane].resident_warp_id) {
      return kStatusInvalidOwner;
    }
  }
  if (resident_slot >= kResidentWarpCapacity) {
    return kStatusInvalidOwner;
  }
  if (state.resident_warps[resident_slot].live) {
    return kStatusDuplicateWarp;
  }
  for (uint32_t slot = 0; slot < kResidentWarpCapacity; ++slot) {
    if (warp_matches(state.resident_warps[slot], state.owner_hw_sid, warp_uid,
                     warp_id)) {
      return kStatusDuplicateWarp;
    }
  }

  const uint32_t lane_count = count_lanes(active_mask);
  const uint32_t charge = lane_count * kResidentChargeBytesPerLane;
  if (charge > kResidentSharedBytes - state.charged_bytes) {
    return kStatusCapacityExceeded;
  }

  std::set<uint32_t> request_identities;
  std::set<uint32_t> private_slots;
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((active_mask & lane_bit(lane)) == 0) continue;
    const private_frontier::owner_binding_v0 &owner = owners[lane];
    if (!owner_is_valid(owner, state.owner_hw_sid, resident_slot,
                        static_cast<uint8_t>(lane))) {
      return kStatusInvalidOwner;
    }
    if (!request_identities.insert(owner.request_identity).second ||
        !private_slots.insert(owner.private_slot_id).second) {
      return kStatusOwnerAlias;
    }
    for (uint32_t slot = 0; slot < kResidentWarpCapacity; ++slot) {
      const resident_warp_state_v0 &live_warp = state.resident_warps[slot];
      if (!live_warp.live) continue;
      for (uint32_t live_lane = 0; live_lane < kLaneCapacity; ++live_lane) {
        if (!live_warp.lanes[live_lane].live) continue;
        const private_frontier::owner_binding_v0 &live_owner =
            live_warp.lanes[live_lane].owner;
        if (live_owner.request_identity == owner.request_identity ||
            live_owner.private_slot_id == owner.private_slot_id) {
          return kStatusOwnerAlias;
        }
      }
    }

    private_frontier::region_binding_v0 region = {};
    region.profile_id = private_frontier::kLayoutProfileId;
    region.slot_count = 256;
    region.private_region_base =
        UINT64_C(0xff00000000000000) +
        static_cast<uint64_t>(state.owner_hw_sid) * UINT64_C(0x1000000);
    private_frontier::frontier_metadata_image_v0 metadata = {};
    metadata.frontier_capacity = private_frontier::kFrontierEntryCapacity;
    metadata.max_level_depth = 1;
    const private_frontier::status_kind planner_status =
        root_operands == NULL
            ? private_frontier::initialize_shadow_slot(
                  &plan->staging_slots[lane], owner, region, metadata,
                  &plan->init_plans[lane])
            : private_frontier::initialize_root_shadow_slot(
                  &plan->staging_slots[lane], owner, region, metadata,
                  root_operands[lane], &plan->init_plans[lane]);
    if (planner_status != private_frontier::kStatusOk ||
        plan->init_plans[lane].access_count !=
            (root_operands == NULL ? 2 : 9)) {
      return kStatusPlannerFailure;
    }
  }

  plan->valid = true;
  plan->root_operands_initialized = root_operands != NULL;
  plan->resident_warp_slot = resident_slot;
  plan->owner_hw_sid = state.owner_hw_sid;
  plan->warp_uid = warp_uid;
  plan->warp_id = warp_id;
  plan->active_mask = active_mask;
  plan->charged_bytes = charge;
  plan->expected_mutation_epoch = state.mutation_epoch;
  return kStatusOk;
}

status_kind prepare_new_warp(
    const backing_state_v0 &state, uint32_t warp_uid, uint32_t warp_id,
    uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    new_warp_plan_v0 *plan) {
  return prepare_new_warp_impl(state, warp_uid, warp_id, active_mask, owners,
                               NULL, plan);
}

status_kind prepare_new_warp_with_root_operands(
    const backing_state_v0 &state, uint32_t warp_uid, uint32_t warp_id,
    uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    const private_frontier::root_private_operands_v0
        root_operands[kLaneCapacity],
    new_warp_plan_v0 *plan) {
  if (root_operands == NULL) return kStatusInvalidArgument;
  return prepare_new_warp_impl(state, warp_uid, warp_id, active_mask, owners,
                               root_operands, plan);
}

status_kind commit_new_warp(backing_state_v0 *state,
                            const new_warp_plan_v0 &plan) {
  if (state == NULL || !state->initialized || !plan.valid ||
      plan.resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidArgument;
  }
  if (state->mutation_epoch != plan.expected_mutation_epoch) {
    return kStatusStalePlan;
  }
  private_frontier::owner_binding_v0 owners[kLaneCapacity] = {};
  private_frontier::root_private_operands_v0
      root_operands[kLaneCapacity] = {};
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.active_mask & lane_bit(lane)) == 0) continue;
    owners[lane] = plan.staging_slots[lane].owner;
    if (plan.root_operands_initialized &&
        private_frontier::decode_root_private_operands(
            plan.staging_slots[lane], owners[lane],
            &root_operands[lane]) != private_frontier::kStatusOk) {
      return kStatusPlannerFailure;
    }
  }
  new_warp_plan_v0 revalidated = {};
  const status_kind revalidation_status =
      plan.root_operands_initialized
          ? prepare_new_warp_with_root_operands(
                *state, plan.warp_uid, plan.warp_id, plan.active_mask, owners,
                root_operands, &revalidated)
          : prepare_new_warp(*state, plan.warp_uid, plan.warp_id,
                             plan.active_mask, owners, &revalidated);
  if (revalidation_status != kStatusOk) {
    return revalidation_status;
  }
  if (std::memcmp(&revalidated, &plan, sizeof(plan)) != 0) {
    return kStatusOwnerMismatch;
  }
  if (state->owner_hw_sid != plan.owner_hw_sid ||
      state->resident_warps[plan.resident_warp_slot].live ||
      plan.active_mask == 0 ||
      plan.charged_bytes !=
          count_lanes(plan.active_mask) * kResidentChargeBytesPerLane ||
      plan.charged_bytes > kResidentSharedBytes - state->charged_bytes) {
    return kStatusOwnerMismatch;
  }

  resident_warp_state_v0 committed = {};
  committed.live = true;
  committed.resident_warp_slot = plan.resident_warp_slot;
  committed.owner_hw_sid = plan.owner_hw_sid;
  committed.warp_uid = plan.warp_uid;
  committed.warp_id = plan.warp_id;
  committed.active_mask = plan.active_mask;
  committed.charged_bytes = plan.charged_bytes;
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.active_mask & lane_bit(lane)) == 0) continue;
    lane_slot_state_v0 &slot = committed.lanes[lane];
    slot.live = true;
    slot.owner = plan.staging_slots[lane].owner;
    slot.canonical_slot.owner = slot.owner;
    slot.staging_slot = plan.staging_slots[lane];
    slot.init_plan = plan.init_plans[lane];
  }

  state->resident_warps[plan.resident_warp_slot] = committed;
  state->charged_bytes += plan.charged_bytes;
  ++state->mutation_epoch;
  return kStatusOk;
}

uint32_t service_init_enqueue(backing_state_v0 *state, uint64_t service_cycle,
                              uint32_t enqueue_budget) {
  if (state == NULL || !state->initialized || enqueue_budget == 0) {
    return 0;
  }
  uint32_t enqueued = 0;
  for (uint32_t warp_slot = 0;
       warp_slot < kResidentWarpCapacity && enqueued < enqueue_budget;
       ++warp_slot) {
    resident_warp_state_v0 &warp = state->resident_warps[warp_slot];
    if (!warp.live || warp.scheduler_ready) continue;
    for (uint32_t lane = 0; lane < kLaneCapacity && enqueued < enqueue_budget;
         ++lane) {
      lane_slot_state_v0 &slot = warp.lanes[lane];
      while (slot.live &&
             slot.next_chunk_to_enqueue < slot.init_plan.access_count &&
             enqueued < enqueue_budget) {
        if (state->shared_queue.size() >= kSharedQueueCapacity) {
          ++state->shared_queue_capacity_blocked_count;
          return enqueued;
        }
        shared_write_v0 operation = {};
        materialize_operation(slot, slot.next_chunk_to_enqueue, service_cycle,
                              &operation);
        state->shared_queue.push_back(operation);
        ++state->enqueue_count;
        ++warp.enqueued_chunk_count;
        if (state->shared_queue.size() > state->max_shared_queue_depth) {
          state->max_shared_queue_depth =
              static_cast<uint32_t>(state->shared_queue.size());
        }
        slot.enqueued_chunk_mask |= uint32_t{1} << slot.next_chunk_to_enqueue;
        ++slot.next_chunk_to_enqueue;
        ++enqueued;
      }
    }
  }
  return enqueued;
}

bool pop_shared_offer(backing_state_v0 *state, uint64_t service_cycle,
                      shared_write_v0 *operation) {
  if (operation != NULL) *operation = shared_write_v0();
  if (state == NULL || !state->initialized || state->offer_valid ||
      state->shared_queue.empty() ||
      state->shared_queue.front().enqueue_cycle >= service_cycle) {
    return false;
  }
  if (state->outstanding.size() >= kOutstandingCapacity) {
    ++state->outstanding_capacity_blocked_count;
    return false;
  }
  state->offered = state->shared_queue.front();
  state->shared_queue.pop_front();
  state->offer_valid = true;
  ++state->offer_count;
  if (operation != NULL) *operation = state->offered;
  return true;
}

status_kind validate_runtime_write(const backing_state_v0 &state,
                                   const shared_write_v0 &operation) {
  if (!state.initialized || !runtime_operation_structure_is_valid(operation) ||
      operation.accepted_cycle != 0 || operation.ack_cycle != 0 ||
      operation.owner.owner_hw_sid != state.owner_hw_sid ||
      operation.owner.resident_warp_id >= kResidentWarpCapacity ||
      operation.owner.lane_id >= kLaneCapacity) {
    return kStatusInvalidArgument;
  }
  const resident_warp_state_v0 &warp =
      state.resident_warps[operation.owner.resident_warp_id];
  const lane_slot_state_v0 &lane = warp.lanes[operation.owner.lane_id];
  if (!warp.live || !warp.scheduler_ready ||
      !validate_runtime_operation_against_lane(operation, lane)) {
    return kStatusOwnerMismatch;
  }
  if (has_runtime_identity(state, operation)) {
    return kStatusDuplicateOperation;
  }
  if (state.shared_queue.size() >= kSharedQueueCapacity) {
    return kStatusQueueFull;
  }
  return kStatusOk;
}

status_kind enqueue_runtime_write(backing_state_v0 *state,
                                  const shared_write_v0 &operation) {
  if (state == NULL) return kStatusInvalidArgument;
  const status_kind validation = validate_runtime_write(*state, operation);
  if (validation != kStatusOk) return validation;
  state->shared_queue.push_back(operation);
  ++state->enqueue_count;
  resident_warp_state_v0 &warp =
      state->resident_warps[operation.owner.resident_warp_id];
  ++warp.enqueued_chunk_count;
  if (state->shared_queue.size() > state->max_shared_queue_depth) {
    state->max_shared_queue_depth =
        static_cast<uint32_t>(state->shared_queue.size());
  }
  return kStatusOk;
}

status_kind accept_shared_offer(backing_state_v0 *state,
                                const shared_write_v0 &operation,
                                uint64_t accepted_cycle) {
  if (state == NULL || !state->initialized || !state->offer_valid ||
      !operation.valid) {
    return kStatusInvalidArgument;
  }
  if (!same_operation(state->offered, operation)) {
    return kStatusOfferMismatch;
  }
  if (operation.owner.resident_warp_id >= kResidentWarpCapacity ||
      operation.owner.lane_id >= kLaneCapacity) {
    return kStatusOwnerMismatch;
  }
  const lane_slot_state_v0 &lane =
      state->resident_warps[operation.owner.resident_warp_id]
          .lanes[operation.owner.lane_id];
  const bool valid_operation =
      is_init_transaction(operation)
          ? validate_operation_against_lane(operation, lane)
          : validate_runtime_operation_against_lane(operation, lane);
  if (!valid_operation) return kStatusOfferMismatch;
  if (state->outstanding.size() >= kOutstandingCapacity) {
    return kStatusOutstandingFull;
  }
  shared_write_v0 accepted = state->offered;
  accepted.accepted_cycle = accepted_cycle;
  accepted.ack_cycle = accepted_cycle + 1;
  state->outstanding.push_back(accepted);
  ++state->accept_count;
  resident_warp_state_v0 &warp =
      state->resident_warps[accepted.owner.resident_warp_id];
  ++warp.accepted_chunk_count;
  if (state->outstanding.size() > state->max_outstanding_depth) {
    state->max_outstanding_depth =
        static_cast<uint32_t>(state->outstanding.size());
  }
  state->offered = shared_write_v0();
  state->offer_valid = false;
  return kStatusOk;
}

static void make_runtime_ack(const shared_write_v0 &operation,
                             runtime_write_ack_v0 *ack) {
  std::memset(ack, 0, sizeof(*ack));
  ack->valid = true;
  ack->field_kind = operation.field_kind;
  ack->memory_operation_seq = static_cast<uint16_t>(operation.memory_op_seq);
  ack->owner = operation.owner;
  ack->operation_seq = operation.operation_seq;
  ack->commit_epoch = operation.commit_epoch;
}

static bool runtime_acks_equal(const runtime_write_ack_v0 &lhs,
                               const runtime_write_ack_v0 &rhs) {
  return lhs.valid == rhs.valid && lhs.field_kind == rhs.field_kind &&
         lhs.memory_operation_seq == rhs.memory_operation_seq &&
         private_frontier::owners_equal(lhs.owner, rhs.owner) &&
         lhs.operation_seq == rhs.operation_seq &&
         lhs.commit_epoch == rhs.commit_epoch &&
         bytes_are_zero(lhs.reserved_zero, sizeof(lhs.reserved_zero)) &&
         bytes_are_zero(rhs.reserved_zero, sizeof(rhs.reserved_zero));
}

status_kind detail::peek_runtime_write_ack(const backing_state_v0 &state,
                                           uint64_t service_cycle,
                                           runtime_write_ack_v0 *ack) {
  if (ack == NULL || !state.initialized) return kStatusInvalidArgument;
  std::memset(ack, 0, sizeof(*ack));
  if (state.outstanding.empty() ||
      state.outstanding.front().ack_cycle > service_cycle ||
      !is_runtime_transaction(state.outstanding.front())) {
    return kStatusNoAckReady;
  }
  const shared_write_v0 &operation = state.outstanding.front();
  if (operation.owner.resident_warp_id >= kResidentWarpCapacity ||
      operation.owner.lane_id >= kLaneCapacity) {
    return kStatusAckMismatch;
  }
  const resident_warp_state_v0 &warp =
      state.resident_warps[operation.owner.resident_warp_id];
  const lane_slot_state_v0 &lane = warp.lanes[operation.owner.lane_id];
  if (!warp.live || !validate_runtime_operation_against_lane(operation, lane)) {
    return kStatusAckMismatch;
  }
  make_runtime_ack(operation, ack);
  return kStatusOk;
}

status_kind detail::commit_runtime_write_ack(backing_state_v0 *state,
                                             uint64_t service_cycle,
                                             const runtime_write_ack_v0 &ack) {
  if (state == NULL) return kStatusInvalidArgument;
  runtime_write_ack_v0 expected = {};
  const status_kind peek_status =
      detail::peek_runtime_write_ack(*state, service_cycle, &expected);
  if (peek_status != kStatusOk) return peek_status;
  if (!runtime_acks_equal(ack, expected)) return kStatusAckMismatch;

  const shared_write_v0 operation = state->outstanding.front();
  resident_warp_state_v0 &warp =
      state->resident_warps[operation.owner.resident_warp_id];
  lane_slot_state_v0 &lane = warp.lanes[operation.owner.lane_id];
  const uint64_t slot_base = private_slot_base(operation.owner);
  const uint32_t chunk_offset =
      static_cast<uint32_t>(operation.aligned_32b_address - slot_base);
  for (unsigned byte = 0; byte < private_frontier::kSharedAccessChunkBytes;
       ++byte) {
    if ((operation.byte_mask & (uint32_t{1} << byte)) != 0) {
      lane.canonical_slot.bytes[chunk_offset + byte] = operation.payload[byte];
    }
  }
  state->outstanding.pop_front();
  ++state->acknowledgment_count;
  ++warp.acknowledged_chunk_count;
  return kStatusOk;
}

uint32_t service_write_acks(backing_state_v0 *state, uint64_t service_cycle,
                            uint32_t response_budget) {
  if (state == NULL || !state->initialized || response_budget == 0) {
    return 0;
  }
  uint32_t acknowledged = 0;
  while (!state->outstanding.empty() && acknowledged < response_budget &&
         state->outstanding.front().ack_cycle <= service_cycle) {
    const shared_write_v0 operation = state->outstanding.front();
    if (!is_init_transaction(operation)) break;
    if (operation.owner.resident_warp_id >= kResidentWarpCapacity ||
        operation.owner.lane_id >= kLaneCapacity) {
      state->fault_status = kStatusAckMismatch;
      break;
    }
    resident_warp_state_v0 &warp =
        state->resident_warps[operation.owner.resident_warp_id];
    lane_slot_state_v0 &lane = warp.lanes[operation.owner.lane_id];
    if (!warp.live || !lane.live ||
        !private_frontier::owners_equal(lane.owner, operation.owner) ||
        !validate_operation_against_lane(operation, lane)) {
      state->fault_status = kStatusAckMismatch;
      break;
    }
    if ((lane.acknowledged_chunk_mask & (uint32_t{1} << operation.chunk_id)) !=
        0) {
      state->fault_status = kStatusDuplicateAck;
      break;
    }

    for (uint32_t byte = 0; byte < private_frontier::kSharedAccessChunkBytes;
         ++byte) {
      if ((operation.byte_mask & (uint32_t{1} << byte)) == 0) continue;
      const uint32_t first =
          lane.init_plan.accesses[operation.chunk_id].slot_byte_offset %
          private_frontier::kSharedAccessChunkBytes;
      const uint32_t chunk_slot_base =
          lane.init_plan.accesses[operation.chunk_id].slot_byte_offset - first;
      lane.canonical_slot.bytes[chunk_slot_base + byte] =
          operation.payload[byte];
    }
    lane.acknowledged_chunk_mask |= uint32_t{1} << operation.chunk_id;
    state->outstanding.pop_front();
    ++state->acknowledgment_count;
    ++warp.acknowledged_chunk_count;
    ++acknowledged;

    if (!warp.scheduler_ready) {
      bool complete = true;
      for (uint32_t warp_lane = 0; warp_lane < kLaneCapacity; ++warp_lane) {
        if ((warp.active_mask & lane_bit(warp_lane)) == 0) continue;
        complete =
            complete && all_init_chunks_acknowledged(warp.lanes[warp_lane]);
      }
      if (complete) {
        warp.scheduler_ready = true;
        state->ready_commit_slots.push_back(
            static_cast<uint8_t>(operation.owner.resident_warp_id));
      }
    }
  }
  return acknowledged;
}

bool pop_ready_commit(backing_state_v0 *state, ready_commit_v0 *commit) {
  if (commit != NULL) *commit = ready_commit_v0();
  if (state == NULL || !state->initialized ||
      state->ready_commit_slots.empty()) {
    return false;
  }
  const uint8_t slot = state->ready_commit_slots.front();
  state->ready_commit_slots.pop_front();
  if (slot >= kResidentWarpCapacity || !state->resident_warps[slot].live ||
      !state->resident_warps[slot].scheduler_ready) {
    return false;
  }
  const resident_warp_state_v0 &warp = state->resident_warps[slot];
  if (commit != NULL) {
    commit->valid = true;
    commit->resident_warp_slot = slot;
    commit->owner_hw_sid = warp.owner_hw_sid;
    commit->warp_uid = warp.warp_uid;
    commit->warp_id = warp.warp_id;
    commit->active_mask = warp.active_mask;
  }
  return true;
}

status_kind prepare_mask_shrink(const backing_state_v0 &state,
                                uint8_t resident_warp_slot,
                                uint32_t previous_warp_uid,
                                uint32_t next_warp_uid, uint32_t warp_id,
                                uint32_t next_active_mask,
                                mask_shrink_plan_v0 *plan) {
  if (plan == NULL || !state.initialized ||
      resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidArgument;
  }
  std::memset(plan, 0, sizeof(*plan));
  const resident_warp_state_v0 &warp = state.resident_warps[resident_warp_slot];
  if (!warp_matches(warp, state.owner_hw_sid, previous_warp_uid, warp_id) ||
      !warp.scheduler_ready || next_warp_uid == previous_warp_uid ||
      next_active_mask == 0 || (next_active_mask & ~warp.active_mask) != 0) {
    return kStatusOwnerMismatch;
  }
  const uint32_t release_mask = warp.active_mask & ~next_active_mask;
  const status_kind release_status =
      validate_release_lanes(state, warp, release_mask);
  if (release_status != kStatusOk) return release_status;

  plan->valid = true;
  plan->resident_warp_slot = resident_warp_slot;
  plan->owner_hw_sid = state.owner_hw_sid;
  plan->previous_warp_uid = previous_warp_uid;
  plan->next_warp_uid = next_warp_uid;
  plan->warp_id = warp_id;
  plan->previous_active_mask = warp.active_mask;
  plan->next_active_mask = next_active_mask;
  plan->release_mask = release_mask;
  plan->released_bytes =
      count_lanes(release_mask) * kResidentChargeBytesPerLane;
  plan->expected_mutation_epoch = state.mutation_epoch;
  return kStatusOk;
}

status_kind commit_mask_shrink(backing_state_v0 *state,
                               const mask_shrink_plan_v0 &plan) {
  if (state == NULL || !state->initialized || !plan.valid ||
      plan.resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidArgument;
  }
  if (state->mutation_epoch != plan.expected_mutation_epoch) {
    return kStatusStalePlan;
  }
  mask_shrink_plan_v0 revalidated = {};
  const status_kind revalidation_status = prepare_mask_shrink(
      *state, plan.resident_warp_slot, plan.previous_warp_uid,
      plan.next_warp_uid, plan.warp_id, plan.next_active_mask, &revalidated);
  if (revalidation_status != kStatusOk) {
    return revalidation_status;
  }
  if (std::memcmp(&revalidated, &plan, sizeof(plan)) != 0) {
    return kStatusOwnerMismatch;
  }
  resident_warp_state_v0 &warp = state->resident_warps[plan.resident_warp_slot];
  if (!warp_matches(warp, plan.owner_hw_sid, plan.previous_warp_uid,
                    plan.warp_id) ||
      warp.active_mask != plan.previous_active_mask ||
      plan.release_mask !=
          (plan.previous_active_mask & ~plan.next_active_mask) ||
      plan.released_bytes > warp.charged_bytes ||
      plan.released_bytes > state->charged_bytes) {
    return kStatusOwnerMismatch;
  }
  const status_kind release_status =
      validate_release_lanes(*state, warp, plan.release_mask);
  if (release_status != kStatusOk) return release_status;

  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.release_mask & lane_bit(lane)) == 0) continue;
    warp.lanes[lane] = lane_slot_state_v0();
  }
  warp.warp_uid = plan.next_warp_uid;
  warp.active_mask = plan.next_active_mask;
  warp.charged_bytes -= plan.released_bytes;
  state->charged_bytes -= plan.released_bytes;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind prepare_release_warp(const backing_state_v0 &state,
                                 uint8_t resident_warp_slot, uint32_t warp_uid,
                                 uint32_t warp_id, release_warp_plan_v0 *plan) {
  if (plan == NULL || !state.initialized ||
      resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidArgument;
  }
  std::memset(plan, 0, sizeof(*plan));
  const resident_warp_state_v0 &warp = state.resident_warps[resident_warp_slot];
  if (!warp_matches(warp, state.owner_hw_sid, warp_uid, warp_id) ||
      !warp.scheduler_ready) {
    return kStatusOwnerMismatch;
  }
  const status_kind release_status =
      validate_release_lanes(state, warp, warp.active_mask);
  if (release_status != kStatusOk) return release_status;
  plan->valid = true;
  plan->resident_warp_slot = resident_warp_slot;
  plan->owner_hw_sid = state.owner_hw_sid;
  plan->warp_uid = warp_uid;
  plan->warp_id = warp_id;
  plan->active_mask = warp.active_mask;
  plan->released_bytes = warp.charged_bytes;
  plan->expected_mutation_epoch = state.mutation_epoch;
  return kStatusOk;
}

status_kind commit_release_warp(backing_state_v0 *state,
                                const release_warp_plan_v0 &plan) {
  if (state == NULL || !state->initialized || !plan.valid ||
      plan.resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidArgument;
  }
  if (state->mutation_epoch != plan.expected_mutation_epoch) {
    return kStatusStalePlan;
  }
  release_warp_plan_v0 revalidated = {};
  const status_kind revalidation_status =
      prepare_release_warp(*state, plan.resident_warp_slot, plan.warp_uid,
                           plan.warp_id, &revalidated);
  if (revalidation_status != kStatusOk) {
    return revalidation_status;
  }
  if (std::memcmp(&revalidated, &plan, sizeof(plan)) != 0) {
    return kStatusOwnerMismatch;
  }
  resident_warp_state_v0 &warp = state->resident_warps[plan.resident_warp_slot];
  if (!warp_matches(warp, plan.owner_hw_sid, plan.warp_uid, plan.warp_id) ||
      warp.active_mask != plan.active_mask ||
      warp.charged_bytes != plan.released_bytes ||
      plan.released_bytes > state->charged_bytes) {
    return kStatusOwnerMismatch;
  }
  const status_kind release_status =
      validate_release_lanes(*state, warp, warp.active_mask);
  if (release_status != kStatusOk) return release_status;
  state->charged_bytes -= plan.released_bytes;
  warp = resident_warp_state_v0();
  ++state->mutation_epoch;
  return kStatusOk;
}

const lane_slot_state_v0 *find_live_lane(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner) {
  if (!state.initialized || owner.resident_warp_id >= kResidentWarpCapacity ||
      owner.lane_id >= kLaneCapacity) {
    return NULL;
  }
  const lane_slot_state_v0 &lane =
      state.resident_warps[owner.resident_warp_id].lanes[owner.lane_id];
  return lane.live && private_frontier::owners_equal(lane.owner, owner) ? &lane
                                                                        : NULL;
}

status_kind prepare_root_operand_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan) {
  if (!state.initialized || read_plan == NULL) {
    return kStatusInvalidArgument;
  }
  const lane_slot_state_v0 *lane = find_live_lane(state, owner);
  if (lane == NULL) return kStatusInvalidOwner;
  const resident_warp_state_v0 &warp =
      state.resident_warps[owner.resident_warp_id];
  if (!warp.live || !warp.scheduler_ready) {
    return kStatusNoAckReady;
  }
  private_frontier::region_binding_v0 region = {};
  region.profile_id = private_frontier::kLayoutProfileId;
  region.slot_count = 256;
  region.private_region_base = private_region_base(state.owner_hw_sid);
  const private_frontier::status_kind status =
      private_frontier::build_root_operand_read_plan(
          lane->canonical_slot, owner, region, read_plan);
  return status == private_frontier::kStatusOk ? kStatusOk
                                               : kStatusPlannerFailure;
}

status_kind prepare_stack_selected_fetch_spill_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan) {
  if (!state.initialized || read_plan == NULL) {
    return kStatusInvalidArgument;
  }
  const lane_slot_state_v0 *lane = find_live_lane(state, owner);
  if (lane == NULL) return kStatusInvalidOwner;
  const resident_warp_state_v0 &warp =
      state.resident_warps[owner.resident_warp_id];
  if (!warp.live || !warp.scheduler_ready) {
    return kStatusNoAckReady;
  }
  private_frontier::region_binding_v0 region = {};
  region.profile_id = private_frontier::kLayoutProfileId;
  region.slot_count = 256;
  region.private_region_base = private_region_base(state.owner_hw_sid);
  const private_frontier::status_kind status =
      private_frontier::build_stack_selected_fetch_spill_read_plan(
          lane->canonical_slot, owner, region, read_plan);
  return status == private_frontier::kStatusOk ? kStatusOk
                                               : kStatusPlannerFailure;
}

status_kind prepare_frontier_metadata_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan) {
  if (!state.initialized || read_plan == NULL) {
    return kStatusInvalidArgument;
  }
  const lane_slot_state_v0 *lane = find_live_lane(state, owner);
  if (lane == NULL) return kStatusInvalidOwner;
  const resident_warp_state_v0 &warp =
      state.resident_warps[owner.resident_warp_id];
  if (!warp.live || !warp.scheduler_ready) {
    return kStatusNoAckReady;
  }
  private_frontier::region_binding_v0 region = {};
  region.profile_id = private_frontier::kLayoutProfileId;
  region.slot_count = 256;
  region.private_region_base = private_region_base(state.owner_hw_sid);
  const private_frontier::status_kind status =
      private_frontier::build_frontier_metadata_read_plan(
          lane->canonical_slot, owner, region, read_plan);
  return status == private_frontier::kStatusOk ? kStatusOk
                                               : kStatusPlannerFailure;
}

status_kind read_canonical_chunk(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::shared_chunk_access_v0 &access,
    uint8_t payload[private_frontier::kSharedAccessChunkBytes]) {
  if (!state.initialized || payload == NULL ||
      access.access_kind != private_frontier::kAccessRead ||
      access.byte_mask == 0 ||
      (access.aligned_32b_address %
       private_frontier::kSharedAccessChunkBytes) != 0) {
    return kStatusInvalidArgument;
  }
  const lane_slot_state_v0 *lane = find_live_lane(state, owner);
  if (lane == NULL) return kStatusInvalidOwner;
  const resident_warp_state_v0 &warp =
      state.resident_warps[owner.resident_warp_id];
  if (!warp.live || !warp.scheduler_ready) return kStatusNoAckReady;
  const uint64_t base = private_slot_base(owner);
  if (access.aligned_32b_address < base ||
      access.aligned_32b_address >
          base + private_frontier::kPrivateDataSlotBytes -
                     private_frontier::kSharedAccessChunkBytes) {
    return kStatusOfferMismatch;
  }
  const uint32_t chunk_offset =
      static_cast<uint32_t>(access.aligned_32b_address - base);
  std::memset(payload, 0, private_frontier::kSharedAccessChunkBytes);
  for (unsigned byte = 0; byte < private_frontier::kSharedAccessChunkBytes;
       ++byte) {
    const bool selected = (access.byte_mask & (uint32_t{1} << byte)) != 0;
    if (selected &&
        !field_contains_offset(access.field_kind, chunk_offset + byte)) {
      return kStatusOfferMismatch;
    }
    if (selected) {
      payload[byte] = lane->canonical_slot.bytes[chunk_offset + byte];
    }
  }
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidOwner:
      return "invalid_owner";
    case kStatusDuplicateWarp:
      return "duplicate_warp";
    case kStatusCapacityExceeded:
      return "capacity_exceeded";
    case kStatusOwnerAlias:
      return "owner_alias";
    case kStatusStalePlan:
      return "stale_plan";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusQueueFull:
      return "queue_full";
    case kStatusOutstandingFull:
      return "outstanding_full";
    case kStatusOfferMismatch:
      return "offer_mismatch";
    case kStatusAckMismatch:
      return "ack_mismatch";
    case kStatusDuplicateAck:
      return "duplicate_ack";
    case kStatusInFlightRelease:
      return "in_flight_release";
    case kStatusPlannerFailure:
      return "planner_failure";
    case kStatusDuplicateOperation:
      return "duplicate_operation";
    case kStatusNoAckReady:
      return "no_ack_ready";
  }
  return "unknown";
}

}  // namespace private_shared
}  // namespace v04
}  // namespace rtcore
