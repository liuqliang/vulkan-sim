#ifndef RTCORE_V04_PRIVATE_PLACEMENT_PROFILE_H
#define RTCORE_V04_PRIVATE_PLACEMENT_PROFILE_H

#include <cstdint>
#include <cstring>
#include <limits>

namespace rtcore {
namespace v04 {
namespace private_placement {

static const char kSelectorEnvironmentName[] =
    "VULKAN_SIM_RTCORE_REPLAY_V04_PRIVATE_PLACEMENT_PROFILE";
static const uint32_t kChunkBytes = 32u;
static const uint32_t kChunkCount = 12u;
static const uint32_t kSelectiveC4Mask = uint32_t{1} << 4u;
static const uint32_t kSelectiveC5C7Mask =
    (uint32_t{1} << 5u) | (uint32_t{1} << 6u) |
    (uint32_t{1} << 7u);
static const uint32_t kFullSharedMask =
    (uint32_t{1} << kChunkCount) - 1u;
// Semantic tags 7 through 14 map to tag-set bits 6 through 13.
static const uint32_t kPrivateSemanticTagMask =
    ((uint32_t{1} << 14u) - 1u) & ~((uint32_t{1} << 6u) - 1u);

enum profile_kind : uint8_t {
  kProfileGlobal384 = 0,
  kProfileSelectiveSharedC4 = 1,
  kProfileSelectiveSharedC5C7 = 2,
  kProfileFullShared384 = 3,
};

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidProfile,
  kStatusStorageAuthorityRequired,
  kStatusOverflow,
};

inline bool exact_private_tag_set(uint32_t tag_set) {
  return tag_set != 0u && (tag_set & ~kPrivateSemanticTagMask) == 0u;
}

inline status_kind parse_profile(const char *value, profile_kind *profile) {
  if (profile == NULL) return kStatusInvalidArgument;
  if (value == NULL || value[0] == '\0' ||
      std::strcmp(value, "global384") == 0) {
    *profile = kProfileGlobal384;
    return kStatusOk;
  }
  if (std::strcmp(value, "selective_shared_c4") == 0) {
    *profile = kProfileSelectiveSharedC4;
    return kStatusOk;
  }
  if (std::strcmp(value, "selective_shared_c5_c7") == 0) {
    *profile = kProfileSelectiveSharedC5C7;
    return kStatusOk;
  }
  if (std::strcmp(value, "full_shared384") == 0) {
    *profile = kProfileFullShared384;
    return kStatusOk;
  }
  return kStatusInvalidProfile;
}

inline const char *profile_name(profile_kind profile) {
  switch (profile) {
    case kProfileGlobal384:
      return "global384";
    case kProfileSelectiveSharedC4:
      return "selective_shared_c4";
    case kProfileSelectiveSharedC5C7:
      return "selective_shared_c5_c7";
    case kProfileFullShared384:
      return "full_shared384";
  }
  return "invalid";
}

inline status_kind validate_storage_authority(profile_kind profile,
                                              bool global384_authority) {
  if (profile == kProfileGlobal384 || global384_authority) return kStatusOk;
  return kStatusStorageAuthorityRequired;
}

inline status_kind shared_chunk_mask(profile_kind profile, uint32_t *mask) {
  if (mask == NULL) return kStatusInvalidArgument;
  switch (profile) {
    case kProfileGlobal384:
      *mask = 0u;
      return kStatusOk;
    case kProfileSelectiveSharedC4:
      *mask = kSelectiveC4Mask;
      return kStatusOk;
    case kProfileSelectiveSharedC5C7:
      *mask = kSelectiveC5C7Mask;
      return kStatusOk;
    case kProfileFullShared384:
      *mask = kFullSharedMask;
      return kStatusOk;
  }
  return kStatusInvalidProfile;
}

inline bool routes_chunk_to_shared(profile_kind profile, uint32_t tag_set,
                                   uint32_t chunk) {
  if (!exact_private_tag_set(tag_set) || chunk >= kChunkCount) return false;
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

inline status_kind resident_charge_bytes(profile_kind profile,
                                         uint32_t active_mask,
                                         uint32_t *bytes) {
  if (bytes == NULL || active_mask == 0u) return kStatusInvalidArgument;
  uint32_t per_lane = 0;
  const status_kind status = shared_bytes_per_lane(profile, &per_lane);
  if (status != kStatusOk) return status;
  const uint32_t lanes = static_cast<uint32_t>(__builtin_popcount(active_mask));
  if (per_lane != 0u &&
      lanes > std::numeric_limits<uint32_t>::max() / per_lane) {
    return kStatusOverflow;
  }
  *bytes = lanes * per_lane;
  return kStatusOk;
}

inline const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidProfile:
      return "invalid_profile";
    case kStatusStorageAuthorityRequired:
      return "storage_authority_required";
    case kStatusOverflow:
      return "overflow";
  }
  return "unknown";
}

}  // namespace private_placement
}  // namespace v04
}  // namespace rtcore

#endif
