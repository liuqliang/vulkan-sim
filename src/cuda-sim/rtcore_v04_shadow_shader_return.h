#ifndef RTCORE_V04_SHADOW_SHADER_RETURN_H
#define RTCORE_V04_SHADOW_SHADER_RETURN_H

#include <array>
#include <cstdint>

#include "rtcore_v04_shadow_boundary.h"

namespace rtcore {
namespace abi_v04 {
namespace shadow {

enum shader_return_error {
  kShaderReturnErrorNone = 0,
  kShaderReturnErrorInvalidReason,
  kShaderReturnErrorReservedBits,
  kShaderReturnErrorIllegalEffect,
  kShaderReturnErrorTerminateUnsupported,
  kShaderReturnErrorInvalidReportedHitKind,
  kShaderReturnErrorReportedAttributesUnsupported,
  kShaderReturnErrorSemanticMismatch,
};

inline const char *shader_return_error_name(shader_return_error error) {
  switch (error) {
    case kShaderReturnErrorNone:
      return "none";
    case kShaderReturnErrorInvalidReason:
      return "invalid_reason";
    case kShaderReturnErrorReservedBits:
      return "reserved_bits";
    case kShaderReturnErrorIllegalEffect:
      return "illegal_effect";
    case kShaderReturnErrorTerminateUnsupported:
      return "terminate_unsupported";
    case kShaderReturnErrorInvalidReportedHitKind:
      return "invalid_reported_hit_kind";
    case kShaderReturnErrorReportedAttributesUnsupported:
      return "reported_attributes_unsupported";
    case kShaderReturnErrorSemanticMismatch:
      return "semantic_mismatch";
  }
  return "unknown";
}

struct shader_return_observation {
  shader_return_observation()
      : error(kShaderReturnErrorNone), traversal_effect(0) {}

  bool valid() const { return error == kShaderReturnErrorNone; }

  shader_return_error error;
  uint32_t traversal_effect;
  boundary_return_update update;
};

inline uint32_t shader_return_unpublished_sentinel() {
  return kReservedMasks[kCommitRetainedCandidate.word];
}

inline bool shader_return_reason_requires_publication(uint32_t reason) {
  return reason == kReasonAnyHitRequired ||
         reason == kReasonIntersectionRequired;
}

inline bool boundary_return_updates_equal(
    const boundary_return_update &lhs, const boundary_return_update &rhs) {
  if (lhs.action != rhs.action) return false;
  if (lhs.action != kBoundaryReturnCommitIntersection) return true;
  return lhs.reported_t_fp32 == rhs.reported_t_fp32 &&
         lhs.reported_hit_kind == rhs.reported_hit_kind &&
         lhs.reported_attribute_word_count ==
             rhs.reported_attribute_word_count &&
         lhs.reported_attribute_format == rhs.reported_attribute_format &&
         lhs.reported_attribute_words == rhs.reported_attribute_words;
}

inline shader_return_observation decode_shader_return_words(
    const std::array<uint32_t, kWordCount> &words, uint32_t reason) {
  shader_return_observation observation;
  if (reason != kReasonAnyHitRequired &&
      reason != kReasonIntersectionRequired) {
    observation.error = kShaderReturnErrorInvalidReason;
    return observation;
  }

  if ((words[kCommitRetainedCandidate.word] &
       kReservedMasks[kCommitRetainedCandidate.word]) != 0u ||
      (words[27] & kReservedMasks[27]) != 0u) {
    observation.error = kShaderReturnErrorReservedBits;
    return observation;
  }

  const uint32_t commit_retained =
      extract_field(words, kCommitRetainedCandidate);
  const uint32_t accepted_report =
      extract_field(words, kAcceptedReportedHitValid);
  const uint32_t terminate = extract_field(words, kTerminateSearch);
  observation.traversal_effect = commit_retained | (accepted_report << 1) |
                                 (terminate << 2);
  if (!is_legal_traversal_effect(reason, observation.traversal_effect)) {
    observation.error = kShaderReturnErrorIllegalEffect;
    return observation;
  }
  if (reason == kReasonAnyHitRequired) {
    observation.update.action = commit_retained != 0u
                                    ? kBoundaryReturnCommitAnyHit
                                    : kBoundaryReturnKeepExisting;
    return observation;
  }

  if (accepted_report == 0u) {
    observation.update.action = kBoundaryReturnKeepExisting;
    return observation;
  }

  if ((words[kReportedHitKind.word] &
       kReservedMasks[kReportedHitKind.word]) != 0u) {
    observation.error = kShaderReturnErrorReservedBits;
    return observation;
  }

  const uint32_t reported_hit_kind =
      extract_field(words, kReportedHitKind);
  const uint32_t reported_attribute_word_count =
      extract_field(words, kReportedAttributeWordCount);
  const uint32_t reported_attribute_format =
      extract_field(words, kReportedAttributeFormat);
  if (reported_hit_kind > 0x7fu) {
    observation.error = kShaderReturnErrorInvalidReportedHitKind;
    return observation;
  }
  const bool no_attributes =
      reported_attribute_word_count == 0u &&
      reported_attribute_format == 0u;
  const bool inline_attributes =
      reported_attribute_word_count > 0u &&
      reported_attribute_word_count <=
          observation.update.reported_attribute_words.size() &&
      reported_attribute_format == 0x02u;
  if (!no_attributes && !inline_attributes) {
    observation.error = kShaderReturnErrorReportedAttributesUnsupported;
    return observation;
  }

  observation.update.action = kBoundaryReturnCommitIntersection;
  observation.update.reported_t_fp32 =
      extract_field(words, kReportedTFp32);
  observation.update.reported_hit_kind = reported_hit_kind;
  observation.update.reported_attribute_word_count =
      reported_attribute_word_count;
  observation.update.reported_attribute_format =
      reported_attribute_format;
  const field_spec reported_attributes[] = {
      kInlineAttributeWord0,
      kInlineAttributeWord1,
      kInlineAttributeWord2,
      kInlineAttributeWord3,
  };
  for (uint32_t word = 0; word < reported_attribute_word_count; ++word) {
    observation.update.reported_attribute_words[word] =
        extract_field(words, reported_attributes[word]);
  }
  return observation;
}

inline shader_return_observation compare_shader_return_words(
    const std::array<uint32_t, kWordCount> &words, uint32_t reason,
    const boundary_return_update &expected) {
  shader_return_observation observation =
      decode_shader_return_words(words, reason);
  if (observation.valid() &&
      !boundary_return_updates_equal(observation.update, expected)) {
    observation.error = kShaderReturnErrorSemanticMismatch;
  }
  return observation;
}

}  // namespace shadow
}  // namespace abi_v04
}  // namespace rtcore

#endif  // RTCORE_V04_SHADOW_SHADER_RETURN_H
