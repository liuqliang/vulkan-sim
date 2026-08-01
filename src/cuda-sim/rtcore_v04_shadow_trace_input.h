#ifndef RTCORE_V04_SHADOW_TRACE_INPUT_H
#define RTCORE_V04_SHADOW_TRACE_INPUT_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "rtcore_abi_v04_generated.h"

namespace rtcore {
namespace abi_v04 {
namespace shadow {

struct trace_input_values {
  uint64_t context_address;
  uint64_t traversable_reference;
  uint32_t world_ray_origin_fp32[3];
  uint32_t ray_tmin_fp32;
  uint32_t world_ray_direction_fp32[3];
  uint32_t launch_ray_tmax_fp32;
  uint32_t ray_flags;
  uint32_t cull_mask;
  uint32_t sbt_record_offset;
  uint32_t sbt_record_stride;
  uint32_t miss_index;
};

inline bool pack_trace_input_words(
    const trace_input_values &values,
    std::array<uint32_t, kWordCount> *words) {
  if (words == NULL) return false;
  words->fill(0);
  return
      insert_field(*words, kContextAddressLow32,
                   static_cast<uint32_t>(values.context_address)) &&
      insert_field(*words, kContextAddressHigh32,
                   static_cast<uint32_t>(values.context_address >> 32)) &&
      insert_field(*words, kTraversableReferenceLow32,
                   static_cast<uint32_t>(values.traversable_reference)) &&
      insert_field(*words, kTraversableReferenceHigh32,
                   static_cast<uint32_t>(values.traversable_reference >> 32)) &&
      insert_field(*words, kWorldRayOriginXFp32,
                   values.world_ray_origin_fp32[0]) &&
      insert_field(*words, kWorldRayOriginYFp32,
                   values.world_ray_origin_fp32[1]) &&
      insert_field(*words, kWorldRayOriginZFp32,
                   values.world_ray_origin_fp32[2]) &&
      insert_field(*words, kRayTminFp32, values.ray_tmin_fp32) &&
      insert_field(*words, kWorldRayDirectionXFp32,
                   values.world_ray_direction_fp32[0]) &&
      insert_field(*words, kWorldRayDirectionYFp32,
                   values.world_ray_direction_fp32[1]) &&
      insert_field(*words, kWorldRayDirectionZFp32,
                   values.world_ray_direction_fp32[2]) &&
      insert_field(*words, kLaunchRayTmaxFp32,
                   values.launch_ray_tmax_fp32) &&
      insert_field(*words, kRayFlags, values.ray_flags) &&
      insert_field(*words, kCullMask, values.cull_mask) &&
      insert_field(*words, kSbtRecordOffset, values.sbt_record_offset) &&
      insert_field(*words, kSbtRecordStride, values.sbt_record_stride) &&
      insert_field(*words, kMissIndex, values.miss_index);
}

inline bool unpack_trace_input_words(
    const std::array<uint32_t, kWordCount> &words,
    trace_input_values *values) {
  if (values == NULL || !validate_reserved_zero(words)) return false;
  trace_input_values decoded = {};
  decoded.context_address =
      static_cast<uint64_t>(extract_field(words, kContextAddressLow32)) |
      (static_cast<uint64_t>(extract_field(words, kContextAddressHigh32))
       << 32);
  decoded.traversable_reference =
      static_cast<uint64_t>(
          extract_field(words, kTraversableReferenceLow32)) |
      (static_cast<uint64_t>(
           extract_field(words, kTraversableReferenceHigh32))
       << 32);
  decoded.world_ray_origin_fp32[0] =
      extract_field(words, kWorldRayOriginXFp32);
  decoded.world_ray_origin_fp32[1] =
      extract_field(words, kWorldRayOriginYFp32);
  decoded.world_ray_origin_fp32[2] =
      extract_field(words, kWorldRayOriginZFp32);
  decoded.ray_tmin_fp32 = extract_field(words, kRayTminFp32);
  decoded.world_ray_direction_fp32[0] =
      extract_field(words, kWorldRayDirectionXFp32);
  decoded.world_ray_direction_fp32[1] =
      extract_field(words, kWorldRayDirectionYFp32);
  decoded.world_ray_direction_fp32[2] =
      extract_field(words, kWorldRayDirectionZFp32);
  decoded.launch_ray_tmax_fp32 = extract_field(words, kLaunchRayTmaxFp32);
  decoded.ray_flags = extract_field(words, kRayFlags);
  decoded.cull_mask = extract_field(words, kCullMask);
  decoded.sbt_record_offset = extract_field(words, kSbtRecordOffset);
  decoded.sbt_record_stride = extract_field(words, kSbtRecordStride);
  decoded.miss_index = extract_field(words, kMissIndex);
  *values = decoded;
  return true;
}

inline uint32_t trace_input_owned_word_mask() {
  static_assert(kWordCount <= 32, "shadow mismatch mask is 32-bit");
  const field_spec fields[] = {
      kContextAddressLow32,
      kContextAddressHigh32,
      kTraversableReferenceLow32,
      kTraversableReferenceHigh32,
      kWorldRayOriginXFp32,
      kWorldRayOriginYFp32,
      kWorldRayOriginZFp32,
      kRayTminFp32,
      kWorldRayDirectionXFp32,
      kWorldRayDirectionYFp32,
      kWorldRayDirectionZFp32,
      kLaunchRayTmaxFp32,
      kRayFlags,
      kCullMask,
      kSbtRecordOffset,
      kSbtRecordStride,
      kMissIndex,
  };
  uint32_t mask = 0;
  for (std::size_t index = 0; index < sizeof(fields) / sizeof(fields[0]);
       ++index) {
    mask |= uint32_t{1} << fields[index].word;
  }
  for (std::size_t word = 0; word < kWordCount; ++word) {
    if (kReservedMasks[word] == 0xffffffffu) {
      mask |= uint32_t{1} << word;
    }
  }
  return mask;
}

struct trace_input_validation {
  trace_input_validation()
      : expected_encoding_valid(false),
        owned_word_mask(trace_input_owned_word_mask()),
        mismatch_word_mask(0) {}

  bool matches() const {
    return expected_encoding_valid && mismatch_word_mask == 0;
  }

  bool expected_encoding_valid;
  uint32_t owned_word_mask;
  uint32_t mismatch_word_mask;
};

inline trace_input_validation validate_trace_input_words(
    const std::array<uint32_t, kWordCount> &actual,
    const trace_input_values &expected_values) {
  trace_input_validation validation;
  std::array<uint32_t, kWordCount> expected = {};
  validation.expected_encoding_valid =
      pack_trace_input_words(expected_values, &expected);
  if (!validation.expected_encoding_valid) {
    validation.mismatch_word_mask = validation.owned_word_mask;
    return validation;
  }
  for (std::size_t word = 0; word < kWordCount; ++word) {
    const uint32_t word_bit = uint32_t{1} << word;
    if ((validation.owned_word_mask & word_bit) != 0 &&
        actual[word] != expected[word]) {
      validation.mismatch_word_mask |= word_bit;
    }
  }
  return validation;
}

}  // namespace shadow
}  // namespace abi_v04
}  // namespace rtcore

#endif  // RTCORE_V04_SHADOW_TRACE_INPUT_H
