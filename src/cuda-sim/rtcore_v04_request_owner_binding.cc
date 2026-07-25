#include "rtcore_v04_request_owner_binding.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace request_owner {
namespace {

static uint32_t lane_bit(uint32_t lane) {
  return lane < kLaneCapacity ? uint32_t{1} << lane : 0;
}

static bool valid_identity(const warp_identity_v0 &identity) {
  return identity.active_mask != 0;
}

static bool same_warp(const warp_identity_v0 &lhs,
                      const warp_identity_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.warp_uid == rhs.warp_uid && lhs.warp_id == rhs.warp_id &&
         lhs.active_mask == rhs.active_mask;
}

static bool find_live_resident(const allocator_state_v0 &state,
                               uint32_t owner_hw_sid, uint32_t warp_id) {
  for (uint32_t slot = 0; slot < kResidentWarpCapacity; ++slot) {
    const resident_slot_state_v0 &resident = state.resident_slots[slot];
    if (resident.live && resident.identity.owner_hw_sid == owner_hw_sid &&
        resident.identity.warp_id == warp_id) {
      return true;
    }
  }
  return false;
}

static bool binding_matches_slot(const allocator_state_v0 &state,
                                 const lane_binding_v0 &binding) {
  if (binding.resident_warp_slot >= kResidentWarpCapacity ||
      binding.request_control_slot >= kRequestControlCapacity ||
      binding.private_slot_id != binding.request_control_slot ||
      binding.lane_id >= kLaneCapacity || binding.request_generation == 0) {
    return false;
  }
  const resident_slot_state_v0 &resident =
      state.resident_slots[binding.resident_warp_slot];
  const request_slot_state_v0 &request =
      state.request_slots[binding.request_control_slot];
  if (!resident.live || !request.live ||
      resident.identity.owner_hw_sid != binding.owner_hw_sid ||
      request.resident_warp_slot != binding.resident_warp_slot ||
      request.lane_id != binding.lane_id ||
      request.last_generation != binding.request_generation ||
      resident.request_control_slots[binding.lane_id] !=
          binding.request_control_slot ||
      resident.request_generations[binding.lane_id] !=
          binding.request_generation) {
    return false;
  }
  internal_request_key_fields_v0 fields = {};
  return unpack_internal_request_key(binding.packed_request_key, &fields) ==
             kStatusOk &&
         fields.resident_warp_slot == binding.resident_warp_slot &&
         fields.request_control_slot == binding.request_control_slot &&
         fields.lane_id == binding.lane_id &&
         fields.request_generation == binding.request_generation;
}

static lane_binding_v0 binding_from_state(const allocator_state_v0 &state,
                                          uint8_t resident_slot,
                                          uint8_t lane) {
  lane_binding_v0 binding = {};
  const resident_slot_state_v0 &resident =
      state.resident_slots[resident_slot];
  binding.owner_hw_sid = resident.identity.owner_hw_sid;
  binding.resident_warp_slot = resident_slot;
  binding.lane_id = lane;
  binding.request_control_slot = resident.request_control_slots[lane];
  binding.private_slot_id = binding.request_control_slot;
  binding.request_generation = resident.request_generations[lane];
  internal_request_key_fields_v0 fields = {};
  fields.resident_warp_slot = binding.resident_warp_slot;
  fields.request_control_slot = binding.request_control_slot;
  fields.lane_id = binding.lane_id;
  fields.request_generation = binding.request_generation;
  const status_kind status =
      pack_internal_request_key(fields, &binding.packed_request_key);
  if (status != kStatusOk) {
    std::memset(&binding, 0, sizeof(binding));
  }
  return binding;
}

}  // namespace

void initialize_allocator(allocator_state_v0 *state) {
  if (state == NULL) return;
  std::memset(state, 0, sizeof(*state));
  state->initialized = true;
  state->mutation_epoch = 1;
}

status_kind pack_internal_request_key(
    const internal_request_key_fields_v0 &fields, uint32_t *packed_key) {
  if (packed_key == NULL) return kStatusInvalidArgument;
  if (fields.resident_warp_slot >= kResidentWarpCapacity ||
      fields.request_control_slot >= kRequestControlCapacity ||
      fields.lane_id >= kLaneCapacity || fields.request_generation == 0 ||
      fields.reserved_zero != 0) {
    return kStatusInvalidKey;
  }
  *packed_key =
      static_cast<uint32_t>(fields.resident_warp_slot) |
      (static_cast<uint32_t>(fields.request_control_slot) << 3) |
      (static_cast<uint32_t>(fields.lane_id) << 11) |
      (static_cast<uint32_t>(fields.request_generation) << 16);
  return kStatusOk;
}

status_kind unpack_internal_request_key(
    uint32_t packed_key, internal_request_key_fields_v0 *fields) {
  if (fields == NULL) return kStatusInvalidArgument;
  std::memset(fields, 0, sizeof(*fields));
  fields->resident_warp_slot = packed_key & 0x7u;
  fields->request_control_slot = (packed_key >> 3) & 0xffu;
  fields->lane_id = (packed_key >> 11) & 0x1fu;
  fields->request_generation = (packed_key >> 16) & 0xffffu;
  return fields->request_generation == 0 ? kStatusInvalidKey : kStatusOk;
}

bool validate_private_frontier_owner_identity(
    const private_frontier::owner_binding_v0 &owner) {
  if (owner.request_identity == 0 || owner.generation == 0 ||
      owner.generation > kRequestGenerationMax ||
      owner.resident_warp_id >= kResidentWarpCapacity ||
      owner.private_slot_id >= kRequestControlCapacity ||
      owner.lane_id >= kLaneCapacity ||
      owner.reserved_zero[0] != 0 || owner.reserved_zero[1] != 0 ||
      owner.reserved_zero[2] != 0) {
    return false;
  }
  internal_request_key_fields_v0 fields = {};
  return unpack_internal_request_key(
             owner.request_identity, &fields) == kStatusOk &&
         fields.resident_warp_slot == owner.resident_warp_id &&
         fields.request_control_slot == owner.private_slot_id &&
         fields.lane_id == owner.lane_id &&
         fields.request_generation == owner.generation;
}

status_kind prepare_new_warp(const allocator_state_v0 &state,
                             const warp_identity_v0 &identity,
                             new_warp_plan_v0 *plan) {
  if (plan == NULL || !state.initialized) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  if (!valid_identity(identity)) return kStatusInvalidActiveMask;
  if (find_live_resident(state, identity.owner_hw_sid, identity.warp_id)) {
    return kStatusDuplicateWarp;
  }

  uint32_t resident_slot = kResidentWarpCapacity;
  for (uint32_t offset = 0; offset < kResidentWarpCapacity; ++offset) {
    const uint32_t candidate =
        (state.next_resident_slot + offset) % kResidentWarpCapacity;
    if (!state.resident_slots[candidate].live) {
      resident_slot = candidate;
      break;
    }
  }
  if (resident_slot == kResidentWarpCapacity) {
    return kStatusResidentCapacityExceeded;
  }

  bool selected[kRequestControlCapacity] = {};
  bool saw_wrapped_free_slot = false;
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((identity.active_mask & lane_bit(lane)) == 0) continue;
    uint32_t request_slot = kRequestControlCapacity;
    for (uint32_t offset = 0; offset < kRequestControlCapacity; ++offset) {
      const uint32_t candidate =
          (state.next_request_slot + offset) % kRequestControlCapacity;
      const request_slot_state_v0 &request = state.request_slots[candidate];
      if (request.live || selected[candidate]) continue;
      if (request.generation_exhausted ||
          request.last_generation == kRequestGenerationMax) {
        saw_wrapped_free_slot = true;
        continue;
      }
      request_slot = candidate;
      break;
    }
    if (request_slot == kRequestControlCapacity) {
      return saw_wrapped_free_slot ? kStatusGenerationExhausted
                                   : kStatusRequestCapacityExceeded;
    }
    selected[request_slot] = true;
    lane_binding_v0 &binding = plan->lane_bindings[lane];
    binding.owner_hw_sid = identity.owner_hw_sid;
    binding.resident_warp_slot = resident_slot;
    binding.lane_id = lane;
    binding.request_control_slot = request_slot;
    binding.private_slot_id = request_slot;
    binding.request_generation =
        static_cast<uint16_t>(state.request_slots[request_slot]
                                  .last_generation + 1u);
    internal_request_key_fields_v0 fields = {};
    fields.resident_warp_slot = binding.resident_warp_slot;
    fields.request_control_slot = binding.request_control_slot;
    fields.lane_id = binding.lane_id;
    fields.request_generation = binding.request_generation;
    const status_kind key_status =
        pack_internal_request_key(fields, &binding.packed_request_key);
    if (key_status != kStatusOk) return key_status;
  }

  plan->valid = true;
  plan->expected_mutation_epoch = state.mutation_epoch;
  plan->identity = identity;
  plan->resident_warp_slot = resident_slot;
  return kStatusOk;
}

status_kind commit_new_warp(allocator_state_v0 *state,
                            const new_warp_plan_v0 &plan) {
  if (state == NULL || !state->initialized || !plan.valid) {
    return kStatusInvalidArgument;
  }
  if (state->mutation_epoch != plan.expected_mutation_epoch) {
    return kStatusStalePlan;
  }
  if (!valid_identity(plan.identity) ||
      plan.resident_warp_slot >= kResidentWarpCapacity ||
      state->resident_slots[plan.resident_warp_slot].live ||
      find_live_resident(*state, plan.identity.owner_hw_sid,
                         plan.identity.warp_id)) {
    return kStatusOwnerMismatch;
  }
  bool selected[kRequestControlCapacity] = {};
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.identity.active_mask & lane_bit(lane)) == 0) continue;
    const lane_binding_v0 &binding = plan.lane_bindings[lane];
    if (binding.owner_hw_sid != plan.identity.owner_hw_sid ||
        binding.resident_warp_slot != plan.resident_warp_slot ||
        binding.lane_id != lane ||
        binding.request_control_slot >= kRequestControlCapacity ||
        binding.private_slot_id != binding.request_control_slot ||
        selected[binding.request_control_slot]) {
      return kStatusOwnerMismatch;
    }
    selected[binding.request_control_slot] = true;
    const request_slot_state_v0 &request =
        state->request_slots[binding.request_control_slot];
    if (request.live || request.generation_exhausted ||
        request.last_generation == kRequestGenerationMax ||
        binding.request_generation != request.last_generation + 1u) {
      return kStatusStalePlan;
    }
    internal_request_key_fields_v0 fields = {};
    if (unpack_internal_request_key(binding.packed_request_key, &fields) !=
            kStatusOk ||
        fields.resident_warp_slot != binding.resident_warp_slot ||
        fields.request_control_slot != binding.request_control_slot ||
        fields.lane_id != binding.lane_id ||
        fields.request_generation != binding.request_generation) {
      return kStatusInvalidKey;
    }
  }

  resident_slot_state_v0 &resident =
      state->resident_slots[plan.resident_warp_slot];
  std::memset(&resident, 0, sizeof(resident));
  resident.live = true;
  resident.identity = plan.identity;
  uint32_t last_request_slot = state->next_request_slot;
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.identity.active_mask & lane_bit(lane)) == 0) continue;
    const lane_binding_v0 &binding = plan.lane_bindings[lane];
    request_slot_state_v0 &request =
        state->request_slots[binding.request_control_slot];
    request.live = true;
    request.generation_exhausted = false;
    request.resident_warp_slot = binding.resident_warp_slot;
    request.lane_id = binding.lane_id;
    request.last_generation = binding.request_generation;
    resident.request_control_slots[lane] = binding.request_control_slot;
    resident.request_generations[lane] = binding.request_generation;
    last_request_slot = binding.request_control_slot;
  }
  state->next_resident_slot =
      (plan.resident_warp_slot + 1u) % kResidentWarpCapacity;
  state->next_request_slot =
      (last_request_slot + 1u) % kRequestControlCapacity;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind prepare_mask_shrink(
    const allocator_state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t previous_warp_uid,
    uint32_t next_warp_uid, uint32_t warp_id, uint32_t next_active_mask,
    mask_shrink_plan_v0 *plan) {
  if (plan == NULL || !state.initialized) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  if (resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusOwnerMismatch;
  }
  const resident_slot_state_v0 &resident =
      state.resident_slots[resident_warp_slot];
  if (!resident.live || resident.identity.owner_hw_sid != owner_hw_sid ||
      resident.identity.warp_uid != previous_warp_uid ||
      resident.identity.warp_id != warp_id) {
    return kStatusOwnerMismatch;
  }
  if (next_warp_uid == previous_warp_uid || next_active_mask == 0 ||
      (next_active_mask & ~resident.identity.active_mask) != 0) {
    return kStatusInvalidMaskShrink;
  }
  plan->valid = true;
  plan->expected_mutation_epoch = state.mutation_epoch;
  plan->owner_hw_sid = owner_hw_sid;
  plan->previous_warp_uid = previous_warp_uid;
  plan->next_warp_uid = next_warp_uid;
  plan->warp_id = warp_id;
  plan->previous_active_mask = resident.identity.active_mask;
  plan->next_active_mask = next_active_mask;
  plan->release_mask = resident.identity.active_mask & ~next_active_mask;
  plan->resident_warp_slot = resident_warp_slot;
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((resident.identity.active_mask & lane_bit(lane)) == 0) continue;
    plan->lane_bindings[lane] =
        binding_from_state(state, resident_warp_slot, lane);
    if (!binding_matches_slot(state, plan->lane_bindings[lane])) {
      std::memset(plan, 0, sizeof(*plan));
      return kStatusOwnerMismatch;
    }
  }
  return kStatusOk;
}

status_kind commit_mask_shrink(allocator_state_v0 *state,
                               const mask_shrink_plan_v0 &plan) {
  if (state == NULL || !state->initialized || !plan.valid) {
    return kStatusInvalidArgument;
  }
  if (state->mutation_epoch != plan.expected_mutation_epoch) {
    return kStatusStalePlan;
  }
  if (plan.resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusOwnerMismatch;
  }
  if (plan.next_warp_uid == plan.previous_warp_uid ||
      plan.next_active_mask == 0 ||
      (plan.next_active_mask & ~plan.previous_active_mask) != 0 ||
      plan.release_mask !=
          (plan.previous_active_mask & ~plan.next_active_mask)) {
    return kStatusInvalidMaskShrink;
  }
  resident_slot_state_v0 &resident =
      state->resident_slots[plan.resident_warp_slot];
  if (!resident.live || resident.identity.owner_hw_sid != plan.owner_hw_sid ||
      resident.identity.warp_uid != plan.previous_warp_uid ||
      resident.identity.warp_id != plan.warp_id ||
      resident.identity.active_mask != plan.previous_active_mask) {
    return kStatusOwnerMismatch;
  }
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.previous_active_mask & lane_bit(lane)) == 0) continue;
    if (plan.lane_bindings[lane].lane_id != lane ||
        !binding_matches_slot(*state, plan.lane_bindings[lane])) {
      return kStatusOwnerMismatch;
    }
  }
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.release_mask & lane_bit(lane)) == 0) continue;
    const lane_binding_v0 &binding = plan.lane_bindings[lane];
    request_slot_state_v0 &request =
        state->request_slots[binding.request_control_slot];
    request.live = false;
    if (request.last_generation == kRequestGenerationMax) {
      request.generation_exhausted = true;
    }
    resident.request_control_slots[lane] = 0;
    resident.request_generations[lane] = 0;
  }
  resident.identity.warp_uid = plan.next_warp_uid;
  resident.identity.active_mask = plan.next_active_mask;
  ++state->mutation_epoch;
  return kStatusOk;
}

status_kind prepare_release_warp(
    const allocator_state_v0 &state, uint8_t resident_warp_slot,
    uint32_t owner_hw_sid, uint32_t current_warp_uid, uint32_t warp_id,
    release_warp_plan_v0 *plan) {
  if (plan == NULL || !state.initialized) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  if (resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusOwnerMismatch;
  }
  const resident_slot_state_v0 &resident =
      state.resident_slots[resident_warp_slot];
  if (!resident.live || resident.identity.owner_hw_sid != owner_hw_sid ||
      resident.identity.warp_uid != current_warp_uid ||
      resident.identity.warp_id != warp_id) {
    return kStatusOwnerMismatch;
  }
  plan->valid = true;
  plan->expected_mutation_epoch = state.mutation_epoch;
  plan->identity = resident.identity;
  plan->resident_warp_slot = resident_warp_slot;
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((resident.identity.active_mask & lane_bit(lane)) == 0) continue;
    plan->lane_bindings[lane] =
        binding_from_state(state, resident_warp_slot, lane);
    if (!binding_matches_slot(state, plan->lane_bindings[lane])) {
      std::memset(plan, 0, sizeof(*plan));
      return kStatusOwnerMismatch;
    }
  }
  return kStatusOk;
}

status_kind commit_release_warp(allocator_state_v0 *state,
                                const release_warp_plan_v0 &plan) {
  if (state == NULL || !state->initialized || !plan.valid) {
    return kStatusInvalidArgument;
  }
  if (state->mutation_epoch != plan.expected_mutation_epoch) {
    return kStatusStalePlan;
  }
  if (plan.resident_warp_slot >= kResidentWarpCapacity) {
    return kStatusOwnerMismatch;
  }
  resident_slot_state_v0 &resident =
      state->resident_slots[plan.resident_warp_slot];
  if (!resident.live || !same_warp(resident.identity, plan.identity)) {
    return kStatusOwnerMismatch;
  }
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.identity.active_mask & lane_bit(lane)) == 0) continue;
    if (plan.lane_bindings[lane].lane_id != lane ||
        !binding_matches_slot(*state, plan.lane_bindings[lane])) {
      return kStatusOwnerMismatch;
    }
  }
  for (uint32_t lane = 0; lane < kLaneCapacity; ++lane) {
    if ((plan.identity.active_mask & lane_bit(lane)) == 0) continue;
    const lane_binding_v0 &binding = plan.lane_bindings[lane];
    request_slot_state_v0 &request =
        state->request_slots[binding.request_control_slot];
    request.live = false;
    if (request.last_generation == kRequestGenerationMax) {
      request.generation_exhausted = true;
    }
  }
  std::memset(&resident, 0, sizeof(resident));
  ++state->mutation_epoch;
  return kStatusOk;
}

bool validate_live_binding(const allocator_state_v0 &state,
                           const lane_binding_v0 &binding,
                           uint32_t current_warp_uid,
                           uint32_t current_active_mask) {
  if (!state.initialized ||
      !binding_matches_slot(state, binding) ||
      (current_active_mask & lane_bit(binding.lane_id)) == 0) {
    return false;
  }
  const resident_slot_state_v0 &resident =
      state.resident_slots[binding.resident_warp_slot];
  return resident.identity.warp_uid == current_warp_uid &&
         resident.identity.active_mask == current_active_mask;
}

private_frontier::owner_binding_v0 make_private_frontier_owner(
    const lane_binding_v0 &binding) {
  private_frontier::owner_binding_v0 owner = {};
  owner.owner_hw_sid = binding.owner_hw_sid;
  owner.resident_warp_id = binding.resident_warp_slot;
  owner.request_identity = binding.packed_request_key;
  owner.generation = binding.request_generation;
  owner.private_slot_id = binding.private_slot_id;
  owner.lane_id = binding.lane_id;
  return owner;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk: return "ok";
    case kStatusInvalidArgument: return "invalid_argument";
    case kStatusInvalidActiveMask: return "invalid_active_mask";
    case kStatusDuplicateWarp: return "duplicate_warp";
    case kStatusResidentCapacityExceeded: return "resident_capacity_exceeded";
    case kStatusRequestCapacityExceeded: return "request_capacity_exceeded";
    case kStatusGenerationExhausted: return "generation_exhausted";
    case kStatusStalePlan: return "stale_plan";
    case kStatusOwnerMismatch: return "owner_mismatch";
    case kStatusInvalidKey: return "invalid_key";
    case kStatusInvalidMaskShrink: return "invalid_mask_shrink";
  }
  return "unknown";
}

}  // namespace request_owner
}  // namespace v04
}  // namespace rtcore
