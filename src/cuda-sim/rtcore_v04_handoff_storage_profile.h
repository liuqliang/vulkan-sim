#ifndef RTCORE_V04_HANDOFF_STORAGE_PROFILE_H
#define RTCORE_V04_HANDOFF_STORAGE_PROFILE_H

#include <cstdint>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace handoff_storage {

static const char kSelectorEnvironmentName[] =
    "VULKAN_SIM_RTCORE_REPLAY_V04_HANDOFF_STORAGE_PROFILE";
static const uint32_t kChunkBytes = 32u;
static const uint32_t kChunkCount = 4u;
static const uint32_t kLaneBytes = kChunkBytes * kChunkCount;
static const uint32_t kSelectiveH3Mask = uint32_t{1} << 3u;
static const uint32_t kFullSharedMask = (uint32_t{1} << kChunkCount) - 1u;
static const uint32_t kHandoffSemanticTagMask = (uint32_t{1} << 6u) - 1u;
static const uint32_t kSharedBankCount = 32u;
static const uint32_t kSharedBankWordBytes = 4u;

enum profile_kind : uint8_t {
  kProfileGlobal128 = 0,
  kProfileSelectiveSharedH3 = 1,
  kProfileFullShared128 = 2,
};

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidProfile,
  kStatusInvalidChunk,
  kStatusInvalidAddress,
  kStatusInvalidResourceState,
  kStatusCapacityExceeded,
  kStatusOverflow,
};

inline bool exact_handoff_tag_set(uint32_t tag_set) {
  return tag_set != 0u && (tag_set & ~kHandoffSemanticTagMask) == 0u;
}

inline status_kind parse_profile(const char *value, profile_kind *profile) {
  if (profile == NULL) return kStatusInvalidArgument;
  if (value == NULL || value[0] == '\0' ||
      std::strcmp(value, "global128") == 0) {
    *profile = kProfileGlobal128;
    return kStatusOk;
  }
  if (std::strcmp(value, "selective_shared_h3") == 0) {
    *profile = kProfileSelectiveSharedH3;
    return kStatusOk;
  }
  if (std::strcmp(value, "full_shared128") == 0) {
    *profile = kProfileFullShared128;
    return kStatusOk;
  }
  return kStatusInvalidProfile;
}

inline const char *profile_name(profile_kind profile) {
  switch (profile) {
    case kProfileGlobal128:
      return "global128";
    case kProfileSelectiveSharedH3:
      return "selective_shared_h3";
    case kProfileFullShared128:
      return "full_shared128";
  }
  return "invalid";
}

inline status_kind shared_chunk_mask(profile_kind profile, uint32_t *mask) {
  if (mask == NULL) return kStatusInvalidArgument;
  switch (profile) {
    case kProfileGlobal128:
      *mask = 0u;
      return kStatusOk;
    case kProfileSelectiveSharedH3:
      *mask = kSelectiveH3Mask;
      return kStatusOk;
    case kProfileFullShared128:
      *mask = kFullSharedMask;
      return kStatusOk;
  }
  return kStatusInvalidProfile;
}

inline bool routes_chunk_to_shared(profile_kind profile, uint32_t tag_set,
                                   uint32_t chunk) {
  if (!exact_handoff_tag_set(tag_set) || chunk >= kChunkCount) return false;
  uint32_t mask = 0;
  return shared_chunk_mask(profile, &mask) == kStatusOk &&
         (mask & (uint32_t{1} << chunk)) != 0u;
}

inline status_kind shared_bytes_per_lane(profile_kind profile,
                                         uint32_t *bytes) {
  if (bytes == NULL) return kStatusInvalidArgument;
  uint32_t mask = 0;
  const status_kind status = shared_chunk_mask(profile, &mask);
  if (status != kStatusOk) return status;
  uint32_t count = 0;
  while (mask != 0u) {
    count += mask & 1u;
    mask >>= 1u;
  }
  *bytes = count * kChunkBytes;
  return kStatusOk;
}

inline uint32_t active_lane_count(uint32_t active_mask) {
  uint32_t count = 0;
  while (active_mask != 0u) {
    count += active_mask & 1u;
    active_mask >>= 1u;
  }
  return count;
}

inline status_kind resident_charge_bytes(profile_kind profile,
                                         uint32_t active_mask,
                                         uint32_t *bytes) {
  if (bytes == NULL || active_mask == 0u) return kStatusInvalidArgument;
  uint32_t per_lane = 0;
  const status_kind status = shared_bytes_per_lane(profile, &per_lane);
  if (status != kStatusOk) return status;
  const uint32_t lanes = active_lane_count(active_mask);
  if (per_lane != 0u &&
      lanes > std::numeric_limits<uint32_t>::max() / per_lane) {
    return kStatusOverflow;
  }
  *bytes = lanes * per_lane;
  return kStatusOk;
}

inline status_kind resource_available(uint32_t capacity_bytes,
                                      uint32_t cta_occupied_bytes,
                                      uint32_t rt_occupied_bytes,
                                      uint32_t demand_bytes,
                                      bool *available) {
  if (available == NULL) return kStatusInvalidArgument;
  if (cta_occupied_bytes > capacity_bytes ||
      rt_occupied_bytes > capacity_bytes - cta_occupied_bytes) {
    return kStatusInvalidResourceState;
  }
  const uint32_t remaining =
      capacity_bytes - cta_occupied_bytes - rt_occupied_bytes;
  *available = demand_bytes <= remaining;
  return *available ? kStatusOk : kStatusCapacityExceeded;
}

inline status_kind bank_mask_for_32b(uint64_t aligned_address,
                                    uint32_t bank_count,
                                    uint32_t bank_word_bytes,
                                    uint32_t *bank_mask) {
  if (bank_mask == NULL || bank_count == 0u || bank_count > 32u ||
      bank_word_bytes == 0u ||
      aligned_address % kChunkBytes != 0u ||
      kChunkBytes % bank_word_bytes != 0u) {
    return kStatusInvalidAddress;
  }
  const uint64_t first_word = aligned_address / bank_word_bytes;
  const uint32_t word_count = kChunkBytes / bank_word_bytes;
  uint32_t mask = 0;
  for (uint32_t word = 0; word < word_count; ++word) {
    const uint32_t bank =
        static_cast<uint32_t>((first_word + word) % bank_count);
    mask |= uint32_t{1} << bank;
  }
  *bank_mask = mask;
  return kStatusOk;
}

inline bool bank_masks_conflict(uint32_t lhs, uint32_t rhs) {
  return (lhs & rhs) != 0u;
}

inline const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidProfile:
      return "invalid_profile";
    case kStatusInvalidChunk:
      return "invalid_chunk";
    case kStatusInvalidAddress:
      return "invalid_address";
    case kStatusInvalidResourceState:
      return "invalid_resource_state";
    case kStatusCapacityExceeded:
      return "capacity_exceeded";
    case kStatusOverflow:
      return "overflow";
  }
  return "unknown";
}

}  // namespace handoff_storage
}  // namespace v04
}  // namespace rtcore

#endif
