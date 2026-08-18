#ifndef RTCORE_V04_SHADER_INPUT_H
#define RTCORE_V04_SHADER_INPUT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "rtcore_abi_v04_generated.h"

namespace rtcore {
namespace abi_v04 {
namespace shader_input {

enum error {
  kErrorNone = 0,
  kErrorInvalidAddress,
  kErrorInvalidLane,
  kErrorAddressOverflow,
  kErrorInvalidReason,
  kErrorReservedBits,
  kErrorInvalidHitKind,
  kErrorInvalidAnyHitCapability,
  kErrorInvalidAttributeContract,
  kErrorUnsupportedContextAttribute,
};

inline const char *error_name(error value) {
  switch (value) {
    case kErrorNone:
      return "none";
    case kErrorInvalidAddress:
      return "invalid_address";
    case kErrorInvalidLane:
      return "invalid_lane";
    case kErrorAddressOverflow:
      return "address_overflow";
    case kErrorInvalidReason:
      return "invalid_reason";
    case kErrorReservedBits:
      return "reserved_bits";
    case kErrorInvalidHitKind:
      return "invalid_hit_kind";
    case kErrorInvalidAnyHitCapability:
      return "invalid_any_hit_capability";
    case kErrorInvalidAttributeContract:
      return "invalid_attribute_contract";
    case kErrorUnsupportedContextAttribute:
      return "unsupported_context_attribute";
  }
  return "unknown";
}

struct attribute_plan {
  attribute_plan()
      : status(kErrorNone),
        hit_kind(0),
        word_count(0),
        location(0),
        format(0) {
    words.fill(0);
  }

  bool valid() const { return status == kErrorNone; }
  bool materializes_attributes() const {
    return valid() && word_count != 0;
  }

  error status;
  uint32_t hit_kind;
  uint32_t word_count;
  uint32_t location;
  uint32_t format;
  std::array<uint32_t, 4> words;
};

static constexpr uint8_t kLaneSlotChunkCount = 4;
static constexpr uint8_t kMaxDispatcherReadCount = 2;

struct dispatcher_chunk_read {
  dispatcher_chunk_read() : chunk(0), byte_mask(0) {}

  uint8_t chunk;
  uint32_t byte_mask;
};

struct dispatcher_read_plan {
  dispatcher_read_plan() : status(kErrorNone), read_count(0) {}

  bool valid() const { return status == kErrorNone; }

  error status;
  uint8_t read_count;
  std::array<dispatcher_chunk_read, kMaxDispatcherReadCount> reads;
};

inline attribute_plan decode(
    const std::array<uint32_t, kWordCount> &words, uint32_t reason);

inline dispatcher_read_plan dispatch_selection_read_plan(uint32_t reason) {
  dispatcher_read_plan plan;
  if (reason != kReasonMiss && reason != kReasonClosestHitReady &&
      reason != kReasonAnyHitRequired &&
      reason != kReasonIntersectionRequired) {
    plan.status = kErrorInvalidReason;
    return plan;
  }

  // w13 selects the miss or hit-group record for every supported reason.
  plan.reads[plan.read_count].chunk = 1;
  plan.reads[plan.read_count++].byte_mask = 0x00f00000u;
  if (reason == kReasonMiss) return plan;

  // w16/w17 complete hit-group selection. w22 validates the hit kind;
  // closest-hit/any-hit also need the w23 attribute contract.
  plan.reads[plan.read_count].chunk = 2;
  plan.reads[plan.read_count++].byte_mask =
      reason == kReasonIntersectionRequired ? 0x0f0000ffu
                                            : 0xff0000ffu;
  return plan;
}

inline dispatcher_read_plan dispatch_attribute_read_plan(
    const std::array<uint32_t, kWordCount> &words, uint32_t reason) {
  dispatcher_read_plan plan;
  const attribute_plan attributes = decode(words, reason);
  if (!attributes.valid()) {
    plan.status = attributes.status;
    return plan;
  }
  if (!attributes.materializes_attributes()) return plan;
  if (attributes.location != 0x01u || attributes.word_count == 0 ||
      attributes.word_count > 4) {
    plan.status = kErrorInvalidAttributeContract;
    return plan;
  }
  plan.reads[0].chunk = 3;
  static constexpr uint32_t kWordsPerChunk = 32u / sizeof(uint32_t);
  static constexpr uint32_t kAttributeByteOffset =
      (kInlineAttributeWord0.word % kWordsPerChunk) * sizeof(uint32_t);
  const uint32_t attribute_byte_count =
      attributes.word_count * sizeof(uint32_t);
  const uint32_t attribute_byte_mask =
      (uint32_t{1} << attribute_byte_count) - 1u;
  plan.reads[0].byte_mask =
      attribute_byte_mask << kAttributeByteOffset;
  plan.read_count = 1;
  return plan;
}

inline bool lane_slot_address(uint64_t base, uint32_t lane,
                              uint64_t *address, error *failure = NULL) {
  if (failure != NULL) *failure = kErrorNone;
  if (address == NULL || base == 0) {
    if (failure != NULL) *failure = kErrorInvalidAddress;
    return false;
  }
  if (lane >= 32) {
    if (failure != NULL) *failure = kErrorInvalidLane;
    return false;
  }
  const uint64_t offset =
      static_cast<uint64_t>(lane) * static_cast<uint64_t>(kLaneSlotBytes);
  if (base > std::numeric_limits<uint64_t>::max() - offset) {
    if (failure != NULL) *failure = kErrorAddressOverflow;
    return false;
  }
  *address = base + offset;
  return true;
}

inline attribute_plan decode(
    const std::array<uint32_t, kWordCount> &words, uint32_t reason) {
  attribute_plan plan;
  if (reason == kReasonMiss) return plan;
  if (reason == kReasonIntersectionRequired) {
    if ((words[kHitKind.word] & kReservedMasks[kHitKind.word]) != 0) {
      plan.status = kErrorReservedBits;
      return plan;
    }
    if (extract_field(words, kHitKind) != 0) {
      plan.status = kErrorInvalidHitKind;
    }
    return plan;
  }
  if (reason != kReasonClosestHitReady &&
      reason != kReasonAnyHitRequired) {
    plan.status = kErrorInvalidReason;
    return plan;
  }
  if ((words[kHitKind.word] & kReservedMasks[kHitKind.word]) != 0 ||
      (words[kInputAttributeWordCount.word] &
       kReservedMasks[kInputAttributeWordCount.word]) != 0) {
    plan.status = kErrorReservedBits;
    return plan;
  }

  plan.hit_kind = extract_field(words, kHitKind);
  plan.word_count = extract_field(words, kInputAttributeWordCount);
  plan.location = extract_field(words, kInputAttributeLocation);
  plan.format = extract_field(words, kInputAttributeFormat);
  const bool triangle =
      plan.hit_kind == 0xfeu || plan.hit_kind == 0xffu;
  const bool procedural = plan.hit_kind <= 0x7fu;
  if (!triangle && !procedural) {
    plan.status = kErrorInvalidHitKind;
    return plan;
  }
  if (reason == kReasonAnyHitRequired && !triangle) {
    plan.status = kErrorInvalidAnyHitCapability;
    return plan;
  }

  const bool triangle_inline =
      triangle && plan.word_count == 2u && plan.location == 0x01u &&
      plan.format == 0x01u;
  const bool procedural_none =
      procedural && plan.word_count == 0u && plan.location == 0u &&
      plan.format == 0u;
  const bool procedural_inline =
      procedural && plan.word_count > 0u && plan.word_count <= 4u &&
      plan.location == 0x01u && plan.format == 0x02u;
  const bool procedural_context =
      procedural && plan.word_count > 0u && plan.word_count <= 22u &&
      plan.location == 0x02u && plan.format == 0x02u;
  if (procedural_context) {
    plan.status = kErrorUnsupportedContextAttribute;
    return plan;
  }
  if (!triangle_inline && !procedural_none && !procedural_inline) {
    plan.status = kErrorInvalidAttributeContract;
    return plan;
  }

  const field_spec attribute_fields[] = {
      kInlineAttributeWord0,
      kInlineAttributeWord1,
      kInlineAttributeWord2,
      kInlineAttributeWord3,
  };
  for (uint32_t word = 0; word < plan.word_count; ++word) {
    plan.words[word] = extract_field(words, attribute_fields[word]);
  }
  return plan;
}

inline bool encode_words_little_endian(
    const std::array<uint32_t, 4> &words, uint32_t word_count,
    unsigned char *destination, std::size_t destination_size) {
  if (word_count == 0 || word_count > words.size() ||
      destination == NULL ||
      destination_size < word_count * sizeof(uint32_t)) {
    return false;
  }
  for (std::size_t byte = 0; byte < destination_size; ++byte) {
    destination[byte] = 0;
  }
  for (uint32_t word = 0; word < word_count; ++word) {
    for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte) {
      destination[word * sizeof(uint32_t) + byte] =
          static_cast<unsigned char>(words[word] >> (byte * 8));
    }
  }
  return true;
}

}  // namespace shader_input
}  // namespace abi_v04
}  // namespace rtcore

#endif
