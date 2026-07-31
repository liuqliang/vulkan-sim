#include "rtcore_v04_private_state_384_backing.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace backing {
namespace {

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

uint8_t count_lanes(uint32_t mask) {
  uint8_t count = 0;
  while (mask != 0) {
    count = static_cast<uint8_t>(count + (mask & 1u));
    mask >>= 1;
  }
  return count;
}

bool owner_valid(const private_frontier::owner_binding_v0 &owner,
                 uint32_t owner_hw_sid, uint8_t resident_warp_slot,
                 uint8_t lane_id) {
  return owner.owner_hw_sid == owner_hw_sid &&
         owner.resident_warp_id == resident_warp_slot &&
         owner.request_identity != 0 && owner.generation != 0 &&
         owner.private_slot_id < kPrivateSlotCapacity &&
         owner.lane_id == lane_id &&
         bytes_are_zero(owner.reserved_zero,
                        sizeof(owner.reserved_zero));
}

bool owner_equal(const private_frontier::owner_binding_v0 &lhs,
                 const private_frontier::owner_binding_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.resident_warp_id == rhs.resident_warp_id &&
         lhs.request_identity == rhs.request_identity &&
         lhs.generation == rhs.generation &&
         lhs.private_slot_id == rhs.private_slot_id &&
         lhs.lane_id == rhs.lane_id &&
         bytes_are_zero(lhs.reserved_zero,
                        sizeof(lhs.reserved_zero)) &&
         bytes_are_zero(rhs.reserved_zero,
                        sizeof(rhs.reserved_zero));
}

bool operation_identity_matches(
    const private_frontier::owner_binding_v0 &owner,
    const operand_materializer::operation_identity_v1 &identity) {
  return identity.owner_hw_sid == owner.owner_hw_sid &&
         identity.resident_warp_id == owner.resident_warp_id &&
         identity.request_identity == owner.request_identity &&
         identity.request_generation == owner.generation &&
         identity.private_slot_id == owner.private_slot_id &&
         identity.lane_id == owner.lane_id &&
         identity.operation_sequence != 0 &&
         bytes_are_zero(identity.reserved_zero,
                        sizeof(identity.reserved_zero));
}

bool canonical_read_plan(const operand_plan::read_plan_v1 &plan) {
  operand_plan::read_request_v1 request = {};
  request.private_layout_profile_id =
      plan.private_layout_profile_id;
  request.consumer = plan.consumer;
  request.operation = plan.operation;
  request.completion_reason = plan.completion_reason;
  operand_plan::read_plan_v1 expected = {};
  return operand_plan::make_read_plan(request, &expected) ==
             operand_plan::kStatusOk &&
         std::memcmp(&plan, &expected, sizeof(plan)) == 0;
}

bool admission_plan_shape_valid(const new_warp_plan_v1 &plan) {
  if (plan.valid != 1 ||
      plan.resident_warp_slot >= kResidentWarpCapacity ||
      plan.active_mask == 0 ||
      plan.active_lane_count != count_lanes(plan.active_mask) ||
      plan.reserved_zero != 0) {
    return false;
  }
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    const admission_lane_v1 &entry = plan.lanes[lane];
    const bool active = (plan.active_mask & (uint32_t{1} << lane)) != 0;
    if (!active) {
      if (!bytes_are_zero(
              reinterpret_cast<const uint8_t *>(&entry),
              sizeof(entry))) {
        return false;
      }
      continue;
    }
    if (entry.valid != 1 || entry.lane_id != lane ||
        entry.private_slot_id != entry.owner.private_slot_id ||
        !owner_valid(entry.owner, plan.owner_hw_sid,
                     plan.resident_warp_slot, lane)) {
      return false;
    }
    for (uint8_t chunk = 0; chunk < kChunkCount; ++chunk) {
      const uint32_t expected =
          chunk < kLaunchWriteCount ? kFullChunkByteMask : 0;
      if (entry.valid_byte_masks[chunk] != expected) {
        return false;
      }
    }
  }
  return true;
}

bool resident_matches(const resident_warp_v1 &resident,
                      uint8_t resident_warp_slot,
                      uint32_t warp_uid, uint32_t warp_id) {
  return resident.live == 1 &&
         resident.resident_warp_slot == resident_warp_slot &&
         resident.warp_uid == warp_uid &&
         resident.warp_id == warp_id &&
         resident.active_mask != 0 &&
         resident.reserved_zero == 0;
}

}  // namespace

void initialize(state_v1 *state, uint32_t owner_hw_sid) {
  if (state == NULL) return;
  *state = state_v1();
  state->initialized = 1;
  state->owner_hw_sid = owner_hw_sid;
}

status_kind prepare_new_warp(
    const state_v1 &state,
    const private_storage::admission_candidate_plan_v0 &candidate,
    uint32_t warp_uid, uint32_t warp_id, new_warp_plan_v1 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = new_warp_plan_v1();
  if (state.initialized != 1 ||
      !bytes_are_zero(state.reserved_zero,
                      sizeof(state.reserved_zero))) {
    return kStatusInvalidState;
  }
  if (!candidate.valid || !candidate.compressed_candidate_valid ||
      candidate.profile !=
          private_storage::kProfileCompressedShared384 ||
      candidate.active_mask == 0 ||
      candidate.active_lane_count !=
          count_lanes(candidate.active_mask) ||
      !candidate.legacy_live_plan.valid ||
      !candidate.legacy_live_plan.root_operands_initialized ||
      !candidate.legacy_live_plan.short_stack_initialized ||
      candidate.legacy_live_plan.resident_warp_slot >=
          kResidentWarpCapacity ||
      candidate.legacy_live_plan.owner_hw_sid !=
          state.owner_hw_sid ||
      candidate.legacy_live_plan.warp_uid != warp_uid ||
      candidate.legacy_live_plan.warp_id != warp_id ||
      candidate.legacy_live_plan.active_mask !=
          candidate.active_mask ||
      candidate.legacy_live_plan.charged_bytes !=
          candidate.active_lane_count *
              private_shared::kResidentChargeBytesPerLane) {
    return kStatusInvalidAdmission;
  }
  const uint8_t resident_warp_slot =
      candidate.legacy_live_plan.resident_warp_slot;
  if (state.resident_warps[resident_warp_slot].live != 0) {
    return kStatusDuplicateWarp;
  }

  new_warp_plan_v1 prepared = {};
  prepared.valid = 1;
  prepared.resident_warp_slot = resident_warp_slot;
  prepared.active_lane_count = candidate.active_lane_count;
  prepared.owner_hw_sid = state.owner_hw_sid;
  prepared.warp_uid = warp_uid;
  prepared.warp_id = warp_id;
  prepared.active_mask = candidate.active_mask;
  prepared.expected_mutation_epoch = state.mutation_epoch;
  bool planned_slots[kPrivateSlotCapacity] = {};
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    const uint32_t lane_mask = uint32_t{1} << lane;
    if ((candidate.active_mask & lane_mask) == 0) {
      if (!bytes_are_zero(
              reinterpret_cast<const uint8_t *>(
                  &candidate.lanes[lane]),
              sizeof(candidate.lanes[lane]))) {
        return kStatusInvalidAdmission;
      }
      continue;
    }
    const private_storage::lane_launch_candidate_v0 &source =
        candidate.lanes[lane];
    if (!source.valid ||
        !bytes_are_zero(source.reserved_zero,
                        sizeof(source.reserved_zero)) ||
        !owner_valid(source.owner, state.owner_hw_sid,
                     resident_warp_slot, lane) ||
        source.valid_chunk_mask !=
            static_cast<uint8_t>(
                (uint8_t{1} << kLaunchWriteCount) - 1u)) {
      return kStatusInvalidAdmission;
    }
    const uint16_t private_slot_id =
        static_cast<uint16_t>(source.owner.private_slot_id);
    if (planned_slots[private_slot_id] ||
        state.slots[private_slot_id].live != 0) {
      return kStatusOwnerAlias;
    }
    planned_slots[private_slot_id] = true;

    admission_lane_v1 &destination = prepared.lanes[lane];
    destination.valid = 1;
    destination.lane_id = lane;
    destination.private_slot_id = private_slot_id;
    destination.owner = source.owner;
    if (source.sparse_writes.write_count != kLaunchWriteCount ||
        !bytes_are_zero(source.sparse_writes.reserved_zero,
                        sizeof(source.sparse_writes.reserved_zero))) {
      return kStatusInvalidAdmission;
    }
    for (uint8_t write_index = 0;
         write_index < source.sparse_writes.write_count;
         ++write_index) {
      const chunk_write_v1 &write =
          source.sparse_writes.writes[write_index];
      const uint8_t chunk =
          static_cast<uint8_t>(write.slot_byte_offset / kChunkBytes);
      if (chunk != write_index ||
          write.slot_byte_offset !=
              static_cast<uint16_t>(chunk * kChunkBytes) ||
          write.byte_count != kChunkBytes ||
          write.byte_mask != kFullChunkByteMask) {
        return kStatusInvalidAdmission;
      }
      std::memcpy(destination.image.bytes +
                      write.slot_byte_offset,
                  write.payload, kChunkBytes);
      destination.valid_byte_masks[chunk] =
          write.byte_mask;
    }
  }
  if (!admission_plan_shape_valid(prepared)) {
    return kStatusInvalidAdmission;
  }
  *plan = prepared;
  return kStatusOk;
}

status_kind commit_new_warp(state_v1 *state,
                            const new_warp_plan_v1 &plan) {
  if (state == NULL) return kStatusInvalidArgument;
  if (state->initialized != 1 ||
      state->owner_hw_sid != plan.owner_hw_sid ||
      state->mutation_epoch != plan.expected_mutation_epoch) {
    return kStatusStalePlan;
  }
  if (!admission_plan_shape_valid(plan)) {
    return kStatusInvalidAdmission;
  }
  if (state->resident_warps[plan.resident_warp_slot].live != 0) {
    return kStatusDuplicateWarp;
  }
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.active_mask & (uint32_t{1} << lane)) == 0) continue;
    const admission_lane_v1 &entry = plan.lanes[lane];
    if (state->slots[entry.private_slot_id].live != 0) {
      return kStatusOwnerAlias;
    }
  }

  resident_warp_v1 &resident =
      state->resident_warps[plan.resident_warp_slot];
  resident.live = 1;
  resident.resident_warp_slot = plan.resident_warp_slot;
  resident.warp_uid = plan.warp_uid;
  resident.warp_id = plan.warp_id;
  resident.active_mask = plan.active_mask;
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.active_mask & (uint32_t{1} << lane)) == 0) continue;
    const admission_lane_v1 &entry = plan.lanes[lane];
    lane_slot_v1 &slot = state->slots[entry.private_slot_id];
    slot.live = 1;
    slot.owner = entry.owner;
    std::memcpy(slot.valid_byte_masks, entry.valid_byte_masks,
                sizeof(slot.valid_byte_masks));
    for (uint8_t chunk = 0; chunk < kChunkCount; ++chunk) {
      const uint32_t byte_mask = entry.valid_byte_masks[chunk];
      const uint16_t byte_offset =
          static_cast<uint16_t>(chunk * kChunkBytes);
      for (uint8_t byte = 0; byte < kChunkBytes; ++byte) {
        if ((byte_mask & (uint32_t{1} << byte)) != 0) {
          slot.image.bytes[byte_offset + byte] =
              entry.image.bytes[byte_offset + byte];
        }
      }
    }
  }
  ++state->mutation_epoch;
  return kStatusOk;
}

const lane_slot_v1 *find_live_lane(
    const state_v1 &state,
    const private_frontier::owner_binding_v0 &owner) {
  if (state.initialized != 1 ||
      owner.private_slot_id >= kPrivateSlotCapacity) {
    return NULL;
  }
  const lane_slot_v1 &slot = state.slots[owner.private_slot_id];
  return slot.live == 1 && owner_equal(slot.owner, owner)
             ? &slot
             : NULL;
}

status_kind prepare_read_responses(
    const state_v1 &state,
    const operand_plan::read_plan_v1 &read_plan,
    const operand_materializer::operation_identity_v1 &identity,
    operand_materializer::chunk_response_v1
        responses[operand_materializer::kMaxOperationReadChunks],
    uint8_t *response_count) {
  if (responses == NULL || response_count == NULL) {
    return kStatusInvalidArgument;
  }
  *response_count = 0;
  std::memset(
      responses, 0,
      sizeof(*responses) *
          operand_materializer::kMaxOperationReadChunks);
  if (state.initialized != 1 ||
      !bytes_are_zero(state.reserved_zero,
                      sizeof(state.reserved_zero)) ||
      state.owner_hw_sid != identity.owner_hw_sid ||
      !canonical_read_plan(read_plan) ||
      read_plan.read_count >
          operand_materializer::kMaxOperationReadChunks) {
    return kStatusInvalidReadPlan;
  }
  if (identity.private_slot_id >= kPrivateSlotCapacity) {
    return kStatusOwnerMismatch;
  }
  const lane_slot_v1 &slot =
      state.slots[identity.private_slot_id];
  if (slot.live != 1 ||
      !operation_identity_matches(slot.owner, identity)) {
    return kStatusOwnerMismatch;
  }
  for (uint8_t index = 0; index < read_plan.read_count; ++index) {
    const operand_plan::chunk_read_v1 &read =
        read_plan.reads[index];
    if (read.chunk_index >= kChunkCount ||
        slot.valid_byte_masks[read.chunk_index] !=
            kFullChunkByteMask) {
      return kStatusUninitializedChunk;
    }
    operand_materializer::chunk_response_v1 &response =
        responses[index];
    response.identity = identity;
    response.private_layout_profile_id =
        read_plan.private_layout_profile_id;
    response.consumer = read_plan.consumer;
    response.operation = read_plan.operation;
    response.completion_reason =
        read_plan.completion_reason;
    response.chunk_index = read.chunk_index;
    response.slot_byte_offset = read.slot_byte_offset;
    response.byte_count = read.byte_count;
    std::memcpy(response.payload,
                slot.image.bytes + read.slot_byte_offset,
                read.byte_count);
  }
  *response_count = read_plan.read_count;
  return kStatusOk;
}

status_kind apply_sparse_deltas(
    state_v1 *state,
    const operand_materializer::operation_identity_v1 &identity,
    const operand_plan::write_request_v1 &request,
    const operand_plan::chunk_delta_v1 *deltas, size_t delta_count) {
  if (state == NULL) return kStatusInvalidArgument;
  if (state->initialized != 1 ||
      !bytes_are_zero(state->reserved_zero,
                      sizeof(state->reserved_zero)) ||
      state->owner_hw_sid != identity.owner_hw_sid) {
    return kStatusInvalidState;
  }
  if (deltas == NULL || delta_count == 0) {
    return kStatusInvalidWritePlan;
  }
  if (identity.private_slot_id >= kPrivateSlotCapacity) {
    return kStatusOwnerMismatch;
  }
  lane_slot_v1 &slot = state->slots[identity.private_slot_id];
  if (slot.live != 1 ||
      !operation_identity_matches(slot.owner, identity)) {
    return kStatusOwnerMismatch;
  }
  if (identity.operation_sequence <=
      slot.last_committed_operation_sequence) {
    return kStatusOperationSequenceStale;
  }
  operand_plan::unit_sparse_write_plan_v1 write_plan = {};
  if (operand_plan::merge_sparse_writes(
          request, deltas, delta_count, &write_plan) !=
      operand_plan::kStatusOk) {
    return kStatusInvalidWritePlan;
  }
  for (uint8_t index = 0; index < write_plan.write_count; ++index) {
    const chunk_write_v1 &write = write_plan.writes[index];
    const uint8_t chunk =
        static_cast<uint8_t>(write.slot_byte_offset / kChunkBytes);
    if (chunk >= kChunkCount ||
        write.slot_byte_offset !=
            static_cast<uint16_t>(chunk * kChunkBytes) ||
        write.byte_count != kChunkBytes ||
        write.byte_mask == 0) {
      return kStatusInvalidWritePlan;
    }
    if ((slot.valid_byte_masks[chunk] | write.byte_mask) !=
        kFullChunkByteMask) {
      return kStatusIncompleteFirstWrite;
    }
  }
  for (uint8_t index = 0; index < write_plan.write_count; ++index) {
    const chunk_write_v1 &write = write_plan.writes[index];
    const uint8_t chunk =
        static_cast<uint8_t>(write.slot_byte_offset / kChunkBytes);
    for (uint8_t byte = 0; byte < kChunkBytes; ++byte) {
      const uint32_t bit = uint32_t{1} << byte;
      if ((write.byte_mask & bit) != 0) {
        slot.image.bytes[write.slot_byte_offset + byte] =
            write.payload[byte];
      }
    }
    slot.valid_byte_masks[chunk] |= write.byte_mask;
  }
  slot.last_committed_operation_sequence =
      identity.operation_sequence;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind prepare_mask_shrink(
    const state_v1 &state, uint8_t resident_warp_slot,
    uint32_t previous_warp_uid, uint32_t next_warp_uid,
    uint32_t warp_id, uint32_t next_active_mask,
    mask_shrink_plan_v1 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = mask_shrink_plan_v1();
  if (state.initialized != 1 ||
      resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidState;
  }
  const resident_warp_v1 &resident =
      state.resident_warps[resident_warp_slot];
  if (!resident_matches(resident, resident_warp_slot,
                        previous_warp_uid, warp_id) ||
      next_warp_uid == 0 || next_active_mask == 0 ||
      (next_active_mask & ~resident.active_mask) != 0) {
    return kStatusInvalidMaskShrink;
  }
  mask_shrink_plan_v1 prepared = {};
  prepared.valid = 1;
  prepared.resident_warp_slot = resident_warp_slot;
  prepared.owner_hw_sid = state.owner_hw_sid;
  prepared.previous_warp_uid = previous_warp_uid;
  prepared.next_warp_uid = next_warp_uid;
  prepared.warp_id = warp_id;
  prepared.previous_active_mask = resident.active_mask;
  prepared.next_active_mask = next_active_mask;
  prepared.release_mask =
      resident.active_mask & ~next_active_mask;
  prepared.expected_mutation_epoch = state.mutation_epoch;
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((prepared.release_mask & (uint32_t{1} << lane)) == 0) continue;
    bool found = false;
    for (uint16_t slot_index = 0;
         slot_index < kPrivateSlotCapacity; ++slot_index) {
      const lane_slot_v1 &slot = state.slots[slot_index];
      if (slot.live == 1 &&
          slot.owner.resident_warp_id == resident_warp_slot &&
          slot.owner.lane_id == lane) {
        if (found) return kStatusOwnerAlias;
        prepared.released_owners[lane] = slot.owner;
        found = true;
      }
    }
    if (!found) return kStatusOwnerMismatch;
  }
  *plan = prepared;
  return kStatusOk;
}

status_kind commit_mask_shrink(state_v1 *state,
                               const mask_shrink_plan_v1 &plan) {
  if (state == NULL || plan.valid != 1) {
    return kStatusInvalidArgument;
  }
  if (state->initialized != 1 ||
      state->owner_hw_sid != plan.owner_hw_sid ||
      state->mutation_epoch != plan.expected_mutation_epoch ||
      plan.resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusStalePlan;
  }
  resident_warp_v1 &resident =
      state->resident_warps[plan.resident_warp_slot];
  if (!resident_matches(
          resident, plan.resident_warp_slot,
          plan.previous_warp_uid, plan.warp_id) ||
      resident.active_mask != plan.previous_active_mask ||
      plan.release_mask !=
          (plan.previous_active_mask & ~plan.next_active_mask)) {
    return kStatusInvalidMaskShrink;
  }
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.release_mask & (uint32_t{1} << lane)) == 0) continue;
    const private_frontier::owner_binding_v0 &owner =
        plan.released_owners[lane];
    if (owner.private_slot_id >= kPrivateSlotCapacity ||
        !owner_equal(state->slots[owner.private_slot_id].owner,
                     owner) ||
        state->slots[owner.private_slot_id].live != 1) {
      return kStatusOwnerMismatch;
    }
  }
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.release_mask & (uint32_t{1} << lane)) == 0) continue;
    lane_slot_v1 &slot =
        state->slots[plan.released_owners[lane].private_slot_id];
    slot.live = 0;
    std::memset(&slot.owner, 0, sizeof(slot.owner));
    std::memset(slot.valid_byte_masks, 0,
                sizeof(slot.valid_byte_masks));
    slot.last_committed_operation_sequence = 0;
  }
  resident.warp_uid = plan.next_warp_uid;
  resident.active_mask = plan.next_active_mask;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind prepare_release_warp(
    const state_v1 &state, uint8_t resident_warp_slot,
    uint32_t warp_uid, uint32_t warp_id,
    release_warp_plan_v1 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  *plan = release_warp_plan_v1();
  if (state.initialized != 1 ||
      resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusInvalidState;
  }
  const resident_warp_v1 &resident =
      state.resident_warps[resident_warp_slot];
  if (!resident_matches(resident, resident_warp_slot,
                        warp_uid, warp_id)) {
    return kStatusInvalidRelease;
  }
  release_warp_plan_v1 prepared = {};
  prepared.valid = 1;
  prepared.resident_warp_slot = resident_warp_slot;
  prepared.owner_hw_sid = state.owner_hw_sid;
  prepared.warp_uid = warp_uid;
  prepared.warp_id = warp_id;
  prepared.active_mask = resident.active_mask;
  prepared.expected_mutation_epoch = state.mutation_epoch;
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((resident.active_mask & (uint32_t{1} << lane)) == 0) continue;
    bool found = false;
    for (uint16_t slot_index = 0;
         slot_index < kPrivateSlotCapacity; ++slot_index) {
      const lane_slot_v1 &slot = state.slots[slot_index];
      if (slot.live == 1 &&
          slot.owner.resident_warp_id == resident_warp_slot &&
          slot.owner.lane_id == lane) {
        if (found) return kStatusOwnerAlias;
        prepared.released_owners[lane] = slot.owner;
        found = true;
      }
    }
    if (!found) return kStatusOwnerMismatch;
  }
  *plan = prepared;
  return kStatusOk;
}

status_kind commit_release_warp(state_v1 *state,
                                const release_warp_plan_v1 &plan) {
  if (state == NULL || plan.valid != 1) {
    return kStatusInvalidArgument;
  }
  if (state->initialized != 1 ||
      state->owner_hw_sid != plan.owner_hw_sid ||
      state->mutation_epoch != plan.expected_mutation_epoch ||
      plan.resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusStalePlan;
  }
  resident_warp_v1 &resident =
      state->resident_warps[plan.resident_warp_slot];
  if (!resident_matches(resident, plan.resident_warp_slot,
                        plan.warp_uid, plan.warp_id) ||
      resident.active_mask != plan.active_mask) {
    return kStatusInvalidRelease;
  }
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.active_mask & (uint32_t{1} << lane)) == 0) continue;
    const private_frontier::owner_binding_v0 &owner =
        plan.released_owners[lane];
    if (owner.private_slot_id >= kPrivateSlotCapacity ||
        state->slots[owner.private_slot_id].live != 1 ||
        !owner_equal(state->slots[owner.private_slot_id].owner,
                     owner)) {
      return kStatusOwnerMismatch;
    }
  }
  for (uint8_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.active_mask & (uint32_t{1} << lane)) == 0) continue;
    lane_slot_v1 &slot =
        state->slots[plan.released_owners[lane].private_slot_id];
    slot.live = 0;
    std::memset(&slot.owner, 0, sizeof(slot.owner));
    std::memset(slot.valid_byte_masks, 0,
                sizeof(slot.valid_byte_masks));
    slot.last_committed_operation_sequence = 0;
  }
  resident = resident_warp_v1();
  ++state->mutation_epoch;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidState:
      return "invalid_state";
    case kStatusInvalidAdmission:
      return "invalid_admission";
    case kStatusCapacityExceeded:
      return "capacity_exceeded";
    case kStatusDuplicateWarp:
      return "duplicate_warp";
    case kStatusOwnerAlias:
      return "owner_alias";
    case kStatusStalePlan:
      return "stale_plan";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusInvalidReadPlan:
      return "invalid_read_plan";
    case kStatusUninitializedChunk:
      return "uninitialized_chunk";
    case kStatusInvalidWritePlan:
      return "invalid_write_plan";
    case kStatusOperationSequenceStale:
      return "operation_sequence_stale";
    case kStatusIncompleteFirstWrite:
      return "incomplete_first_write";
    case kStatusInvalidMaskShrink:
      return "invalid_mask_shrink";
    case kStatusInvalidRelease:
      return "invalid_release";
  }
  return "unknown";
}

}  // namespace backing
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore
