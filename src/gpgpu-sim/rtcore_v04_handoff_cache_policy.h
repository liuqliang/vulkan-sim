#ifndef RTCORE_V04_HANDOFF_CACHE_POLICY_H
#define RTCORE_V04_HANDOFF_CACHE_POLICY_H

#include <cstdint>
#include <cstring>

namespace rtcore {
namespace v04 {
namespace handoff_cache_policy {

enum profile_kind {
  kProfileBaseline = 0,
  kProfileWriteAllocateRetain = 1,
};

enum status_kind {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidProfile,
};

static const uint32_t kHandoffSemanticTagMask = (uint32_t{1} << 6u) - 1u;

inline bool exact_handoff_tag_set(uint32_t tag_set) {
  return tag_set != 0 && (tag_set & ~kHandoffSemanticTagMask) == 0;
}

inline bool request_eligible(uint32_t tag_set, unsigned chunk,
                             bool is_write) {
  return is_write && exact_handoff_tag_set(tag_set) && chunk < 4u;
}

inline status_kind parse_profile(const char *value, profile_kind *profile) {
  if (profile == NULL) return kStatusInvalidArgument;
  if (value == NULL || value[0] == '\0' || std::strcmp(value, "baseline") == 0) {
    *profile = kProfileBaseline;
    return kStatusOk;
  }
  if (std::strcmp(value, "write_allocate_retain") == 0) {
    *profile = kProfileWriteAllocateRetain;
    return kStatusOk;
  }
  return kStatusInvalidProfile;
}

inline const char *profile_name(profile_kind profile) {
  switch (profile) {
    case kProfileBaseline:
      return "baseline";
    case kProfileWriteAllocateRetain:
      return "write_allocate_retain";
  }
  return "invalid";
}

}  // namespace handoff_cache_policy
}  // namespace v04
}  // namespace rtcore

#endif
