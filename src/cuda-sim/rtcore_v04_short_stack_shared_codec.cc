#include "rtcore_v04_short_stack_shared_codec.h"

#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace short_stack_shared {
namespace {

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static bool valid_owner(const private_frontier::owner_binding_v0 &owner) {
  return owner.request_identity != 0 && owner.generation != 0 &&
         owner.lane_id < 32 &&
         bytes_are_zero(owner.reserved_zero, sizeof(owner.reserved_zero));
}

static bool local_owners_equal(const private_frontier::owner_binding_v0 &lhs,
                               const private_frontier::owner_binding_v0 &rhs) {
  return lhs.owner_hw_sid == rhs.owner_hw_sid &&
         lhs.resident_warp_id == rhs.resident_warp_id &&
         lhs.request_identity == rhs.request_identity &&
         lhs.generation == rhs.generation &&
         lhs.private_slot_id == rhs.private_slot_id &&
         lhs.lane_id == rhs.lane_id &&
         bytes_are_zero(lhs.reserved_zero, sizeof(lhs.reserved_zero)) &&
         bytes_are_zero(rhs.reserved_zero, sizeof(rhs.reserved_zero));
}

static bool checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (result == NULL || lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

static bool checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (result == NULL ||
      (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

static status_kind slot_base_address(
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::region_binding_v0 &region, uint64_t *slot_base) {
  if (slot_base == NULL) return kStatusInvalidArgument;
  if (region.profile_id != private_frontier::kLayoutProfileId ||
      region.slot_count == 0 ||
      (region.private_region_base %
       private_frontier::kPrivateDataSlotAlignment) != 0 ||
      owner.private_slot_id >= region.slot_count) {
    return kStatusInvalidRegion;
  }

  uint64_t region_bytes = 0;
  uint64_t region_end = 0;
  uint64_t slot_delta = 0;
  uint64_t candidate = 0;
  uint64_t slot_end = 0;
  if (!checked_mul_u64(region.slot_count,
                       private_frontier::kPrivateDataSlotBytes,
                       &region_bytes) ||
      !checked_add_u64(region.private_region_base, region_bytes, &region_end) ||
      !checked_mul_u64(owner.private_slot_id,
                       private_frontier::kPrivateDataSlotBytes, &slot_delta) ||
      !checked_add_u64(region.private_region_base, slot_delta, &candidate) ||
      !checked_add_u64(candidate, private_frontier::kPrivateDataSlotBytes,
                       &slot_end) ||
      slot_end > region_end) {
    return kStatusAddressOverflow;
  }
  *slot_base = candidate;
  return kStatusOk;
}

static void encode_u16_le(uint8_t *destination, uint16_t value) {
  destination[0] = static_cast<uint8_t>(value);
  destination[1] = static_cast<uint8_t>(value >> 8);
}

static void encode_u32_le(uint8_t *destination, uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte) {
    destination[byte] = static_cast<uint8_t>(value >> (byte * 8));
  }
}

static void encode_u64_le(uint8_t *destination, uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte) {
    destination[byte] = static_cast<uint8_t>(value >> (byte * 8));
  }
}

static uint16_t decode_u16_le(const uint8_t *source) {
  return static_cast<uint16_t>(source[0]) | static_cast<uint16_t>(source[1])
                                                << 8;
}

static uint32_t decode_u32_le(const uint8_t *source) {
  uint32_t value = 0;
  for (unsigned byte = 0; byte < 4; ++byte) {
    value |= static_cast<uint32_t>(source[byte]) << (byte * 8);
  }
  return value;
}

static uint64_t decode_u64_le(const uint8_t *source) {
  uint64_t value = 0;
  for (unsigned byte = 0; byte < 8; ++byte) {
    value |= static_cast<uint64_t>(source[byte]) << (byte * 8);
  }
  return value;
}

static metadata_image_v0 metadata_from_state(const persistent_state_v0 &state) {
  metadata_image_v0 metadata = {};
  metadata.format_tag = kMetadataTag;
  metadata.tlas_build_generation = state.tlas_build_generation;
  metadata.blas_build_generation = state.blas_build_generation;
  metadata.stack_count = state.stack.stack_count;
  metadata.stack_top_ptr = state.stack.stack_top_ptr;
  metadata.cross_as = state.stack.cross_as;
  metadata.lost = state.stack.lost;
  metadata.active_domain = state.stack.active_domain;
  return metadata;
}

static void encode_metadata(uint8_t *destination,
                            const metadata_image_v0 &metadata) {
  encode_u32_le(destination + 0, metadata.format_tag);
  encode_u32_le(destination + 4, metadata.tlas_build_generation);
  encode_u32_le(destination + 8, metadata.blas_build_generation);
  destination[12] = metadata.stack_count;
  destination[13] = metadata.stack_top_ptr;
  destination[14] = metadata.cross_as;
  destination[15] = metadata.lost;
  destination[16] = metadata.active_domain;
  std::memcpy(destination + 17, metadata.reserved_zero,
              sizeof(metadata.reserved_zero));
}

static metadata_image_v0 decode_metadata(const uint8_t *source) {
  metadata_image_v0 metadata = {};
  metadata.format_tag = decode_u32_le(source + 0);
  metadata.tlas_build_generation = decode_u32_le(source + 4);
  metadata.blas_build_generation = decode_u32_le(source + 8);
  metadata.stack_count = source[12];
  metadata.stack_top_ptr = source[13];
  metadata.cross_as = source[14];
  metadata.lost = source[15];
  metadata.active_domain = source[16];
  std::memcpy(metadata.reserved_zero, source + 17,
              sizeof(metadata.reserved_zero));
  return metadata;
}

static bool valid_metadata(const metadata_image_v0 &metadata) {
  const bool blas_active = metadata.active_domain == short_stack::kDomainBlas;
  return metadata.format_tag == kMetadataTag &&
         metadata.tlas_build_generation != 0 &&
         (blas_active ? metadata.blas_build_generation != 0
                      : metadata.blas_build_generation == 0) &&
         metadata.stack_count <= short_stack::kLogicalCapacity &&
         metadata.stack_top_ptr < short_stack::kLogicalCapacity &&
         metadata.cross_as <= 1 && metadata.lost <= 1 &&
         metadata.active_domain <= short_stack::kDomainBlas &&
         metadata.cross_as == (blas_active ? 1 : 0) &&
         bytes_are_zero(metadata.reserved_zero, sizeof(metadata.reserved_zero));
}

static void encode_entry(uint8_t *destination,
                         const short_stack::entry_v0 &entry) {
  encode_u64_le(destination + 0, entry.payload_offset);
  encode_u32_le(destination + 8, entry.near_t_bits);
  encode_u16_le(destination + 12, entry.payload_byte_count);
  destination[14] = entry.payload_kind;
  destination[15] = entry.control;
}

static short_stack::entry_v0 decode_entry(const uint8_t *source) {
  short_stack::entry_v0 entry = {};
  entry.payload_offset = decode_u64_le(source + 0);
  entry.near_t_bits = decode_u32_le(source + 8);
  entry.payload_byte_count = decode_u16_le(source + 12);
  entry.payload_kind = source[14];
  entry.control = source[15];
  return entry;
}

static status_kind append_range(
    private_frontier::access_plan_v0 *plan,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::region_binding_v0 &region,
    private_frontier::field_kind field, private_frontier::access_kind access,
    uint32_t slot_offset, uint32_t byte_count) {
  if (plan == NULL || byte_count == 0 ||
      (access != private_frontier::kAccessRead &&
       access != private_frontier::kAccessWrite)) {
    return kStatusInvalidArgument;
  }
  if (slot_offset >= private_frontier::kPrivateDataSlotBytes ||
      byte_count > private_frontier::kPrivateDataSlotBytes - slot_offset) {
    return kStatusInvalidRegion;
  }

  uint64_t slot_base = 0;
  status_kind status = slot_base_address(owner, region, &slot_base);
  if (status != kStatusOk) return status;

  const uint32_t range_end = slot_offset + byte_count;
  uint32_t chunk_offset =
      slot_offset - (slot_offset % private_frontier::kSharedAccessChunkBytes);
  while (chunk_offset < range_end) {
    if (plan->access_count >= private_frontier::kMaxAccessChunks) {
      return kStatusPlanCapacityExceeded;
    }
    const uint32_t first =
        slot_offset > chunk_offset ? slot_offset - chunk_offset : 0;
    const uint32_t chunk_end =
        chunk_offset + private_frontier::kSharedAccessChunkBytes;
    const uint32_t covered_end = range_end < chunk_end ? range_end : chunk_end;
    const uint32_t count = covered_end - (chunk_offset + first);
    const uint64_t low_bits = count == 32
                                  ? uint64_t{0xffffffff}
                                  : (uint64_t{1} << count) - uint64_t{1};
    uint64_t address = 0;
    if (!checked_add_u64(slot_base, chunk_offset, &address)) {
      return kStatusAddressOverflow;
    }
    private_frontier::shared_chunk_access_v0 &chunk =
        plan->accesses[plan->access_count++];
    chunk.aligned_32b_address = address;
    chunk.byte_mask = static_cast<uint32_t>(low_bits << first);
    chunk.slot_byte_offset = static_cast<uint16_t>(chunk_offset + first);
    chunk.byte_count = static_cast<uint8_t>(count);
    chunk.field_kind = static_cast<uint8_t>(field);
    chunk.access_kind = static_cast<uint8_t>(access);
    chunk_offset += private_frontier::kSharedAccessChunkBytes;
  }
  return kStatusOk;
}

static status_kind build_plan(const private_frontier::owner_binding_v0 &owner,
                              const private_frontier::region_binding_v0 &region,
                              private_frontier::access_kind access,
                              private_frontier::access_plan_v0 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  plan->owner = owner;
  status_kind status = append_range(
      plan, owner, region, private_frontier::kFieldFrontierMetadata, access,
      private_frontier::kFrontierMetadataOffset,
      private_frontier::kFrontierMetadataBytes);
  if (status != kStatusOk) return status;
  status = append_range(
      plan, owner, region, private_frontier::kFieldFrontierEntry, access,
      private_frontier::kFrontierEntriesOffset, kEncodedEntriesBytes);
  if (status != kStatusOk) return status;
  return plan->access_count == kStateAccessChunkCount
             ? kStatusOk
             : kStatusPlanCapacityExceeded;
}

static status_kind validate_slot_owner(
    const private_frontier::shadow_slot_v0 &slot,
    const private_frontier::owner_binding_v0 &owner) {
  if (!valid_owner(owner)) return kStatusInvalidOwner;
  return local_owners_equal(slot.owner, owner) ? kStatusOk
                                               : kStatusOwnerMismatch;
}

}  // namespace

bool validate_persistent_state(const persistent_state_v0 &state) {
  if (!short_stack::validate_state(state.stack) ||
      state.tlas_build_generation == 0) {
    return false;
  }
  return state.stack.active_domain == short_stack::kDomainBlas
             ? state.blas_build_generation != 0
             : state.blas_build_generation == 0;
}

status_kind apply_persistent_state(
    private_frontier::shadow_slot_v0 *slot,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::region_binding_v0 &region,
    const persistent_state_v0 &state,
    private_frontier::access_plan_v0 *write_plan) {
  if (slot == NULL || write_plan == NULL) {
    return kStatusInvalidArgument;
  }
  status_kind status = validate_slot_owner(*slot, owner);
  if (status != kStatusOk) return status;
  if (!validate_persistent_state(state)) return kStatusInvalidState;

  private_frontier::access_plan_v0 plan = {};
  status = build_plan(owner, region, private_frontier::kAccessWrite, &plan);
  if (status != kStatusOk) return status;

  private_frontier::shadow_slot_v0 updated = *slot;
  const metadata_image_v0 metadata = metadata_from_state(state);
  encode_metadata(updated.bytes + private_frontier::kFrontierMetadataOffset,
                  metadata);
  for (uint32_t index = 0; index < kEncodedEntryCount; ++index) {
    encode_entry(updated.bytes + private_frontier::kFrontierEntriesOffset +
                     index * private_frontier::kFrontierEntryBytes,
                 state.stack.entries[index]);
  }
  *slot = updated;
  *write_plan = plan;
  return kStatusOk;
}

status_kind decode_persistent_state(
    const private_frontier::shadow_slot_v0 &slot,
    const private_frontier::owner_binding_v0 &owner,
    persistent_state_v0 *state) {
  if (state == NULL) return kStatusInvalidArgument;
  status_kind status = validate_slot_owner(slot, owner);
  if (status != kStatusOk) return status;

  const metadata_image_v0 metadata =
      decode_metadata(slot.bytes + private_frontier::kFrontierMetadataOffset);
  if (!valid_metadata(metadata)) return kStatusInvalidMetadata;

  persistent_state_v0 decoded = {};
  decoded.tlas_build_generation = metadata.tlas_build_generation;
  decoded.blas_build_generation = metadata.blas_build_generation;
  decoded.stack.stack_count = metadata.stack_count;
  decoded.stack.stack_top_ptr = metadata.stack_top_ptr;
  decoded.stack.cross_as = metadata.cross_as;
  decoded.stack.lost = metadata.lost;
  decoded.stack.active_domain = metadata.active_domain;
  for (uint32_t index = 0; index < kEncodedEntryCount; ++index) {
    decoded.stack.entries[index] =
        decode_entry(slot.bytes + private_frontier::kFrontierEntriesOffset +
                     index * private_frontier::kFrontierEntryBytes);
  }
  if (!validate_persistent_state(decoded)) return kStatusInvalidState;
  *state = decoded;
  return kStatusOk;
}

status_kind build_persistent_state_read_plan(
    const private_frontier::shadow_slot_v0 &slot,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::region_binding_v0 &region,
    private_frontier::access_plan_v0 *read_plan) {
  if (read_plan == NULL) return kStatusInvalidArgument;
  persistent_state_v0 ignored = {};
  status_kind status = decode_persistent_state(slot, owner, &ignored);
  if (status != kStatusOk) return status;
  return build_plan(owner, region, private_frontier::kAccessRead, read_plan);
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidOwner:
      return "invalid_owner";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusInvalidRegion:
      return "invalid_region";
    case kStatusAddressOverflow:
      return "address_overflow";
    case kStatusInvalidMetadata:
      return "invalid_metadata";
    case kStatusInvalidState:
      return "invalid_state";
    case kStatusPlanCapacityExceeded:
      return "plan_capacity_exceeded";
  }
  return "unknown";
}

}  // namespace short_stack_shared
}  // namespace v04
}  // namespace rtcore
