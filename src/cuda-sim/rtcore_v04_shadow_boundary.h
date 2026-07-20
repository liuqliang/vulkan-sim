#ifndef RTCORE_V04_SHADOW_BOUNDARY_H
#define RTCORE_V04_SHADOW_BOUNDARY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rtcore_v04_shadow_trace_input.h"

namespace rtcore {
namespace abi_v04 {
namespace shadow {

enum boundary_geometry_type {
  kBoundaryGeometryNone = 0,
  kBoundaryGeometryTriangle = 1,
  kBoundaryGeometryProcedural = 2,
};

enum boundary_error {
  kBoundaryErrorNone = 0,
  kBoundaryErrorInvalidReason,
  kBoundaryErrorMissingCandidate,
  kBoundaryErrorInvalidInstanceMetadataReference,
  kBoundaryErrorInvalidInstanceSbtContribution,
  kBoundaryErrorInvalidGeometryType,
  kBoundaryErrorInvalidHitKind,
  kBoundaryErrorInvalidAttributeContract,
  kBoundaryErrorEncoding,
  kBoundaryErrorReservedBits,
};

inline const char *boundary_error_name(boundary_error error) {
  switch (error) {
    case kBoundaryErrorNone:
      return "none";
    case kBoundaryErrorInvalidReason:
      return "invalid_reason";
    case kBoundaryErrorMissingCandidate:
      return "missing_candidate";
    case kBoundaryErrorInvalidInstanceMetadataReference:
      return "invalid_instance_metadata_reference";
    case kBoundaryErrorInvalidInstanceSbtContribution:
      return "invalid_instance_sbt_contribution";
    case kBoundaryErrorInvalidGeometryType:
      return "invalid_geometry_type";
    case kBoundaryErrorInvalidHitKind:
      return "invalid_hit_kind";
    case kBoundaryErrorInvalidAttributeContract:
      return "invalid_attribute_contract";
    case kBoundaryErrorEncoding:
      return "encoding";
    case kBoundaryErrorReservedBits:
      return "reserved_bits";
  }
  return "unknown";
}

struct boundary_values {
  boundary_values()
      : candidate_valid(false),
        instance_metadata_reference(0),
        instance_sbt_contribution(0),
        geometry_index(0),
        boundary_ray_tmax_fp32(0),
        primitive_index(0),
        instance_index(0),
        instance_custom_index(0),
        geometry_type(kBoundaryGeometryNone),
        hit_kind(0),
        procedural_any_hit_eligible(false),
        input_attribute_word_count(0),
        input_attribute_location(0),
        input_attribute_format(0) {
    inline_attribute_words.fill(0);
  }

  bool candidate_valid;
  uint64_t instance_metadata_reference;
  uint32_t instance_sbt_contribution;
  uint32_t geometry_index;
  uint32_t boundary_ray_tmax_fp32;
  uint32_t primitive_index;
  uint32_t instance_index;
  uint32_t instance_custom_index;
  uint32_t geometry_type;
  uint32_t hit_kind;
  bool procedural_any_hit_eligible;
  uint32_t input_attribute_word_count;
  uint32_t input_attribute_location;
  uint32_t input_attribute_format;
  std::array<uint32_t, 4> inline_attribute_words;
};

enum boundary_return_action {
  kBoundaryReturnKeepExisting = 0,
  kBoundaryReturnCommitAnyHit,
  kBoundaryReturnCommitIntersection,
};

struct boundary_return_update {
  boundary_return_update()
      : action(kBoundaryReturnKeepExisting),
        reported_t_fp32(0),
        reported_hit_kind(0),
        reported_attribute_word_count(0),
        reported_attribute_format(0) {
    reported_attribute_words.fill(0);
  }

  boundary_return_action action;
  uint32_t reported_t_fp32;
  uint32_t reported_hit_kind;
  uint32_t reported_attribute_word_count;
  uint32_t reported_attribute_format;
  std::array<uint32_t, 4> reported_attribute_words;
};

inline bool triangle_candidate_is_strictly_closer(
    const boundary_values &current_terminal,
    const boundary_values &current_candidate) {
  if (!current_candidate.candidate_valid ||
      current_candidate.geometry_type != kBoundaryGeometryTriangle) {
    return false;
  }
  if (!current_terminal.candidate_valid) return true;
  float candidate_t = 0.0f;
  float committed_t = 0.0f;
  std::memcpy(&candidate_t, &current_candidate.boundary_ray_tmax_fp32,
              sizeof(candidate_t));
  std::memcpy(&committed_t, &current_terminal.boundary_ray_tmax_fp32,
              sizeof(committed_t));
  return candidate_t < committed_t;
}

inline bool commit_triangle_candidate_if_strictly_closer(
    const boundary_values &current_terminal,
    const boundary_values &current_candidate,
    boundary_values *next_terminal) {
  if (next_terminal == NULL || !current_candidate.candidate_valid ||
      current_candidate.geometry_type != kBoundaryGeometryTriangle) {
    return false;
  }
  if (!triangle_candidate_is_strictly_closer(current_terminal,
                                              current_candidate)) {
    *next_terminal = current_terminal;
    return true;
  }
  *next_terminal = current_candidate;
  return true;
}

inline bool apply_boundary_return_update(
    const boundary_values &current_terminal,
    const boundary_values &current_candidate,
    const boundary_return_update &update, boundary_values *next_terminal) {
  if (next_terminal == NULL) return false;
  if (update.action == kBoundaryReturnKeepExisting) {
    *next_terminal = current_terminal;
    return true;
  }
  if (!current_candidate.candidate_valid) return false;

  if (update.action == kBoundaryReturnCommitAnyHit) {
    return commit_triangle_candidate_if_strictly_closer(
        current_terminal, current_candidate, next_terminal);
  }
  if (update.action != kBoundaryReturnCommitIntersection ||
      current_candidate.geometry_type != kBoundaryGeometryProcedural ||
      update.reported_hit_kind > 0x7fu) {
    return false;
  }

  const bool no_attributes =
      update.reported_attribute_word_count == 0u &&
      update.reported_attribute_format == 0u;
  const bool inline_attributes =
      update.reported_attribute_word_count > 0u &&
      update.reported_attribute_word_count <=
          update.reported_attribute_words.size() &&
      update.reported_attribute_format == 0x02u;
  if (!no_attributes && !inline_attributes) return false;

  if (current_terminal.candidate_valid) {
    float reported_t = 0.0f;
    float committed_t = 0.0f;
    std::memcpy(&reported_t, &update.reported_t_fp32,
                sizeof(reported_t));
    std::memcpy(&committed_t, &current_terminal.boundary_ray_tmax_fp32,
                sizeof(committed_t));
    if (!(reported_t < committed_t)) {
      *next_terminal = current_terminal;
      return true;
    }
  }

  *next_terminal = current_candidate;
  next_terminal->boundary_ray_tmax_fp32 = update.reported_t_fp32;
  next_terminal->hit_kind = update.reported_hit_kind;
  next_terminal->procedural_any_hit_eligible = false;
  next_terminal->input_attribute_word_count =
      update.reported_attribute_word_count;
  next_terminal->input_attribute_location =
      update.reported_attribute_word_count == 0u ? 0u : 0x01u;
  next_terminal->input_attribute_format =
      update.reported_attribute_format;
  next_terminal->inline_attribute_words = update.reported_attribute_words;
  return true;
}

inline void tighten_intersection_boundary(
    const boundary_values &current_terminal, boundary_values *candidate) {
  if (candidate == NULL || !candidate->candidate_valid ||
      candidate->geometry_type != kBoundaryGeometryProcedural ||
      !current_terminal.candidate_valid) {
    return;
  }
  float candidate_bound = 0.0f;
  float committed_bound = 0.0f;
  std::memcpy(&candidate_bound, &candidate->boundary_ray_tmax_fp32,
              sizeof(candidate_bound));
  std::memcpy(&committed_bound,
              &current_terminal.boundary_ray_tmax_fp32,
              sizeof(committed_bound));
  if (committed_bound < candidate_bound) {
    candidate->boundary_ray_tmax_fp32 =
        current_terminal.boundary_ray_tmax_fp32;
  }
}

struct boundary_publication {
  boundary_publication()
      : reason(kReasonNoneOrInvalid),
        error(kBoundaryErrorNone),
        written_word_mask(0) {
    words.fill(0);
  }

  bool valid() const { return error == kBoundaryErrorNone; }

  uint32_t reason;
  boundary_error error;
  uint32_t written_word_mask;
  std::array<uint32_t, kWordCount> words;
};

inline uint32_t boundary_word_bit(field_spec field) {
  return uint32_t{1} << field.word;
}

inline bool boundary_insert(std::array<uint32_t, kWordCount> *words,
                            field_spec field, uint32_t value,
                            uint32_t *word_mask) {
  if (words == NULL || word_mask == NULL ||
      !insert_field(*words, field, value)) {
    return false;
  }
  *word_mask |= boundary_word_bit(field);
  return true;
}

inline void seed_boundary_image_from_trace_input(
    const std::array<uint32_t, kWordCount> &trace_input_words,
    std::array<uint32_t, kWordCount> *boundary_words) {
  boundary_words->fill(0);
  const uint32_t immutable_mask = trace_input_owned_word_mask();
  for (std::size_t word = 0; word < kWordCount; ++word) {
    if ((immutable_mask & (uint32_t{1} << word)) != 0) {
      (*boundary_words)[word] = trace_input_words[word];
    }
  }
}

inline bool boundary_reason_has_candidate(uint32_t reason) {
  return reason == kReasonClosestHitReady ||
         reason == kReasonAnyHitRequired ||
         reason == kReasonIntersectionRequired;
}

inline boundary_publication build_boundary_publication(
    const std::array<uint32_t, kWordCount> &trace_input_words,
    uint32_t reason, const boundary_values &values) {
  boundary_publication publication;
  publication.reason = reason;
  seed_boundary_image_from_trace_input(trace_input_words,
                                       &publication.words);

  if (reason == kReasonMiss || reason == kReasonTraceDoneNoShader) {
    if (!validate_reserved_zero(publication.words)) {
      publication.error = kBoundaryErrorReservedBits;
    }
    return publication;
  }
  if (!boundary_reason_has_candidate(reason)) {
    publication.error = kBoundaryErrorInvalidReason;
    return publication;
  }
  if (!values.candidate_valid) {
    publication.error = kBoundaryErrorMissingCandidate;
    return publication;
  }
  if (values.instance_metadata_reference == 0 ||
      (values.instance_metadata_reference & 0x3fu) != 0) {
    publication.error = kBoundaryErrorInvalidInstanceMetadataReference;
    return publication;
  }
  if (values.instance_sbt_contribution > 0x00ffffffu) {
    publication.error = kBoundaryErrorInvalidInstanceSbtContribution;
    return publication;
  }
  if (values.geometry_type != kBoundaryGeometryTriangle &&
      values.geometry_type != kBoundaryGeometryProcedural) {
    publication.error = kBoundaryErrorInvalidGeometryType;
    return publication;
  }

  const bool triangle =
      values.geometry_type == kBoundaryGeometryTriangle;
  const bool procedural =
      values.geometry_type == kBoundaryGeometryProcedural;
  if (reason == kReasonAnyHitRequired && !triangle) {
    publication.error = kBoundaryErrorInvalidGeometryType;
    return publication;
  }
  if (reason == kReasonIntersectionRequired && !procedural) {
    publication.error = kBoundaryErrorInvalidGeometryType;
    return publication;
  }
  if (reason != kReasonIntersectionRequired &&
      ((triangle && values.hit_kind != 0xfeu && values.hit_kind != 0xffu) ||
       (procedural && values.hit_kind > 0x7fu))) {
    publication.error = kBoundaryErrorInvalidHitKind;
    return publication;
  }
  if (reason == kReasonIntersectionRequired && values.hit_kind != 0) {
    publication.error = kBoundaryErrorInvalidHitKind;
    return publication;
  }

  if (!boundary_insert(&publication.words, kInstanceMetadataReferenceLow32,
                       static_cast<uint32_t>(
                           values.instance_metadata_reference),
                       &publication.written_word_mask) ||
      !boundary_insert(&publication.words, kInstanceMetadataReferenceHigh32,
                       static_cast<uint32_t>(
                           values.instance_metadata_reference >> 32),
                       &publication.written_word_mask) ||
      !boundary_insert(&publication.words, kInstanceSbtContribution,
                       values.instance_sbt_contribution,
                       &publication.written_word_mask) ||
      !boundary_insert(&publication.words, kGeometryIndex,
                       values.geometry_index,
                       &publication.written_word_mask) ||
      !boundary_insert(&publication.words, kBoundaryRayTmaxFp32,
                       values.boundary_ray_tmax_fp32,
                       &publication.written_word_mask) ||
      !boundary_insert(&publication.words, kPrimitiveIndex,
                       values.primitive_index,
                       &publication.written_word_mask) ||
      !boundary_insert(&publication.words, kInstanceIndex,
                       values.instance_index,
                       &publication.written_word_mask) ||
      !boundary_insert(&publication.words, kInstanceCustomIndex,
                       values.instance_custom_index,
                       &publication.written_word_mask)) {
    publication.error = kBoundaryErrorEncoding;
    return publication;
  }

  if (reason == kReasonIntersectionRequired) {
    if (values.input_attribute_word_count != 0 ||
        values.input_attribute_location != 0 ||
        values.input_attribute_format != 0 ||
        !boundary_insert(&publication.words, kProceduralAnyHitEligible,
                         values.procedural_any_hit_eligible ? 1u : 0u,
                         &publication.written_word_mask)) {
      publication.error = kBoundaryErrorInvalidAttributeContract;
      return publication;
    }
  } else {
    if (values.procedural_any_hit_eligible) {
      publication.error = kBoundaryErrorInvalidAttributeContract;
      return publication;
    }
    const bool triangle_attributes =
        triangle && values.input_attribute_word_count == 2u &&
        values.input_attribute_location == 0x01u &&
        values.input_attribute_format == 0x01u;
    const bool procedural_no_attributes =
        procedural && values.input_attribute_word_count == 0u &&
        values.input_attribute_location == 0u &&
        values.input_attribute_format == 0u;
    const bool procedural_inline_attributes =
        procedural && values.input_attribute_word_count > 0u &&
        values.input_attribute_word_count <=
            values.inline_attribute_words.size() &&
        values.input_attribute_location == 0x01u &&
        values.input_attribute_format == 0x02u;
    if (!triangle_attributes && !procedural_no_attributes &&
        !procedural_inline_attributes) {
      publication.error = kBoundaryErrorInvalidAttributeContract;
      return publication;
    }
    if (!boundary_insert(&publication.words, kHitKind, values.hit_kind,
                         &publication.written_word_mask) ||
        !boundary_insert(&publication.words, kProceduralAnyHitEligible, 0,
                         &publication.written_word_mask) ||
        !boundary_insert(&publication.words, kInputAttributeWordCount,
                         values.input_attribute_word_count,
                         &publication.written_word_mask) ||
        !boundary_insert(&publication.words, kInputAttributeLocation,
                         values.input_attribute_location,
                         &publication.written_word_mask) ||
        !boundary_insert(&publication.words, kInputAttributeFormat,
                         values.input_attribute_format,
                         &publication.written_word_mask)) {
      publication.error = kBoundaryErrorEncoding;
      return publication;
    }
    for (uint32_t word = 0; word < values.input_attribute_word_count; ++word) {
      const field_spec attributes[] = {
          kInlineAttributeWord0, kInlineAttributeWord1,
          kInlineAttributeWord2, kInlineAttributeWord3};
      if (!boundary_insert(&publication.words, attributes[word],
                           values.inline_attribute_words[word],
                           &publication.written_word_mask)) {
        publication.error = kBoundaryErrorEncoding;
        return publication;
      }
    }
  }

  if (!validate_reserved_zero(publication.words)) {
    publication.error = kBoundaryErrorReservedBits;
  }
  return publication;
}

struct boundary_validation {
  boundary_validation()
      : expected_valid(false), compared_word_mask(0), mismatch_word_mask(0),
        error(kBoundaryErrorNone) {}

  bool matches() const {
    return expected_valid && error == kBoundaryErrorNone &&
           mismatch_word_mask == 0;
  }

  bool expected_valid;
  uint32_t compared_word_mask;
  uint32_t mismatch_word_mask;
  boundary_error error;
};

inline boundary_validation validate_boundary_publication(
    const std::array<uint32_t, kWordCount> &actual,
    const std::array<uint32_t, kWordCount> &trace_input_words,
    uint32_t reason, const boundary_values &values) {
  boundary_validation validation;
  const boundary_publication expected =
      build_boundary_publication(trace_input_words, reason, values);
  validation.expected_valid = expected.valid();
  validation.error = expected.error;
  validation.compared_word_mask =
      trace_input_owned_word_mask() | expected.written_word_mask;
  if (!expected.valid()) {
    validation.mismatch_word_mask = validation.compared_word_mask;
    return validation;
  }
  for (std::size_t word = 0; word < kWordCount; ++word) {
    const uint32_t word_bit = uint32_t{1} << word;
    if ((validation.compared_word_mask & word_bit) != 0 &&
        actual[word] != expected.words[word]) {
      validation.mismatch_word_mask |= word_bit;
    }
  }
  if (!validate_reserved_zero(actual)) {
    validation.error = kBoundaryErrorReservedBits;
  }
  return validation;
}

}  // namespace shadow
}  // namespace abi_v04
}  // namespace rtcore

#endif  // RTCORE_V04_SHADOW_BOUNDARY_H
