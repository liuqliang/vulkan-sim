#include "rtcore_v04_typed_node_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "rtcore_v04_canonical_ray.h"

namespace rtcore {
namespace v04 {
namespace typed_node {
namespace {

static uint32_t read_le_u32(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

static uint64_t read_le_u64(const uint8_t *bytes) {
  return static_cast<uint64_t>(read_le_u32(bytes)) |
         (static_cast<uint64_t>(read_le_u32(bytes + 4)) << 32);
}

static int32_t read_le_i32(const uint8_t *bytes) {
  const uint32_t bits = read_le_u32(bytes);
  int32_t value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static float read_le_fp32(const uint8_t *bytes) {
  const uint32_t bits = read_le_u32(bytes);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t fp32_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static bool supported_child_kind(uint8_t level, uint8_t child_kind) {
  if (level == kLevelTlas) {
    return child_kind == kInternalPayloadKind ||
           child_kind == kInstancePayloadKind;
  }
  if (level == kLevelBlas) {
    return child_kind == kInternalPayloadKind ||
           child_kind == kProceduralPayloadKind ||
           child_kind == kQuadPayloadKind;
  }
  return false;
}

static bool valid_decode_context(
    const typed_blas::as_decode_context_v0 &context, uint8_t level) {
  const uint8_t expected_as_type =
      level == kLevelTlas ? uint8_t{1} : typed_blas::kAsTypeBlas;
  return context.bvh_format_profile_id == kGenRtDerivedProfileId &&
         context.reserved_zero == 0 && context.as_object.object_id != 0 &&
         context.as_object.generation != 0 &&
         context.as_object.as_type == expected_as_type &&
         bytes_are_zero(context.as_object.reserved_zero,
                        sizeof(context.as_object.reserved_zero)) &&
         context.device_base != 0 && context.device_range_bytes >= 64 &&
         context.device_range_bytes <=
             std::numeric_limits<uint64_t>::max() - context.device_base;
}

static bool checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (result == NULL || lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

static bool checked_add_signed_blocks(uint64_t base, int32_t blocks,
                                      uint64_t *result) {
  const int64_t byte_delta = static_cast<int64_t>(blocks) * int64_t{64};
  if (byte_delta >= 0) {
    return checked_add_u64(base, static_cast<uint64_t>(byte_delta), result);
  }
  const uint64_t magnitude = static_cast<uint64_t>(-byte_delta);
  if (result == NULL || base < magnitude) return false;
  *result = base - magnitude;
  return true;
}

static bool expected_child_layout(uint8_t kind, uint8_t *blocks,
                                  uint16_t *bytes) {
  if (blocks == NULL || bytes == NULL) return false;
  switch (kind) {
    case kInternalPayloadKind:
    case kProceduralPayloadKind:
    case kQuadPayloadKind:
      *blocks = 1;
      *bytes = 64;
      return true;
    case kInstancePayloadKind:
      *blocks = 2;
      *bytes = 128;
      return true;
  }
  return false;
}

static bool payload_in_range(uint64_t offset, uint16_t bytes,
                             uint64_t range_bytes) {
  return offset <= range_bytes &&
         static_cast<uint64_t>(bytes) <= range_bytes - offset;
}

static bool finite_ray(const ray_state_v0 &ray, float committed_t) {
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(ray.origin[axis]) ||
        !std::isfinite(ray.direction[axis]) ||
        !canonical_ray::inverse_direction_matches(
            ray.direction[axis], ray.inverse_direction[axis])) {
      return false;
    }
  }
  return std::isfinite(ray.t_min) && std::isfinite(ray.t_max) &&
         std::isfinite(committed_t) && ray.t_min <= ray.t_max;
}

struct interval_result {
  bool valid;
  bool hit;
  float near_t;
};

static interval_result intersect_aabb_strict(
    const ray_state_v0 &ray, const float low[3], const float high[3],
    float committed_t) {
  interval_result result = {true, false, 0.0f};
  float near_t = ray.t_min;
  float far_t = std::min(ray.t_max, committed_t);

  if (!std::isfinite(far_t)) {
    result.valid = false;
    return result;
  }

  for (unsigned axis = 0; axis < 3; ++axis) {
    const float origin = ray.origin[axis];
    const float direction = ray.direction[axis];
    if (direction == 0.0f) {
      if (origin < low[axis] || origin > high[axis]) {
        result.near_t = near_t;
        return result;
      }
      continue;
    }

    const float inverse_direction = ray.inverse_direction[axis];

    const float low_delta = low[axis] - origin;
    const float high_delta = high[axis] - origin;
    const float t0 = low_delta * inverse_direction;
    const float t1 = high_delta * inverse_direction;
    if (!std::isfinite(t0) || !std::isfinite(t1)) {
      result.valid = false;
      return result;
    }

    const float axis_near = std::min(t0, t1);
    const float axis_far = std::max(t0, t1);
    near_t = std::max(near_t, axis_near);
    far_t = std::min(far_t, axis_far);
    if (near_t > far_t) {
      result.near_t = near_t;
      return result;
    }
  }

  result.near_t = near_t;
  result.hit = near_t <= far_t && near_t < committed_t;
  return result;
}

static float decode_bound(float origin, uint8_t quantized, int exponent) {
  const float scaled = ::ldexpf(static_cast<float>(quantized), exponent - 8);
  return origin + scaled;
}

static bool apply_replay_cursor(const replay_cursor_v0 &cursor,
                                candidate_result_v0 *candidates) {
  if (candidates == NULL || cursor.anchor_valid > 1 ||
      cursor.inclusive > 1 ||
      !bytes_are_zero(cursor.reserved_zero,
                      sizeof(cursor.reserved_zero)) ||
      (cursor.anchor_valid == 0 &&
       (cursor.child_anchor != 0 || cursor.inclusive != 0)) ||
      (cursor.anchor_valid != 0 &&
       (cursor.child_anchor >= kMaxChildren ||
        (candidates->evaluated_child_mask &
         (1u << cursor.child_anchor)) == 0))) {
    return false;
  }
  if (cursor.anchor_valid == 0) return true;

  float anchor_near = 0.0f;
  std::memcpy(&anchor_near,
              &candidates->near_t_bits[cursor.child_anchor],
              sizeof(anchor_near));
  uint8_t filtered_slots[kMaxChildren] = {};
  uint8_t filtered_mask = 0;
  uint8_t filtered_count = 0;
  for (uint8_t index = 0; index < candidates->candidate_count;
       ++index) {
    const uint8_t slot = candidates->ordered_child_slots[index];
    float near_t = 0.0f;
    std::memcpy(&near_t, &candidates->near_t_bits[slot],
                sizeof(near_t));
    const bool after =
        near_t > anchor_near ||
        (near_t == anchor_near && slot > cursor.child_anchor);
    const bool at_anchor =
        near_t == anchor_near && slot == cursor.child_anchor;
    if (!after && !(cursor.inclusive != 0 && at_anchor)) continue;
    filtered_slots[filtered_count++] = slot;
    filtered_mask |= static_cast<uint8_t>(1u << slot);
  }
  std::memset(candidates->ordered_child_slots, 0,
              sizeof(candidates->ordered_child_slots));
  std::memcpy(candidates->ordered_child_slots, filtered_slots,
              sizeof(filtered_slots));
  candidates->candidate_count = filtered_count;
  candidates->hit_child_mask = filtered_mask;
  return true;
}

}  // namespace

bool make_raw_node_payload(const void *raw_node_bytes,
                           raw_node_payload_v0 *payload) {
  if (raw_node_bytes == NULL || payload == NULL) return false;
  std::memset(payload, 0, sizeof(*payload));
  payload->header.expected_payload_kind = kInternalPayloadKind;
  payload->header.expected_chunk_count = 2;
  payload->header.payload_byte_count = 64;
  payload->header.received_chunk_mask = 0x03;
  std::memcpy(payload->raw_bytes, raw_node_bytes,
              sizeof(payload->raw_bytes));
  return true;
}

candidate_result_v0 execute(const candidate_input_v0 &input) {
  candidate_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (input.profile_id != kGenRtDerivedProfileId) {
    result.status = kStatusUnsupportedProfile;
    return result;
  }
  if ((input.level != kLevelTlas && input.level != kLevelBlas) ||
      !bytes_are_zero(input.reserved_zero0,
                      sizeof(input.reserved_zero0)) ||
      input.reserved_zero1 != 0 ||
      !bytes_are_zero(input.policy.reserved_zero,
                      sizeof(input.policy.reserved_zero))) {
    return result;
  }

  const raw_payload_header_v0 &header = input.raw_node.header;
  if (header.expected_payload_kind != kInternalPayloadKind ||
      header.expected_chunk_count != 2 || header.payload_byte_count != 64 ||
      header.received_chunk_mask != 0x03 ||
      !bytes_are_zero(header.reserved_zero, sizeof(header.reserved_zero))) {
    result.status = kStatusMalformedEnvelope;
    return result;
  }
  if (!finite_ray(input.ray, input.committed_t)) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }

  const uint8_t *raw = input.raw_node.raw_bytes;
  const float node_origin[3] = {read_le_fp32(raw + 0),
                                read_le_fp32(raw + 4),
                                read_le_fp32(raw + 8)};
  if (!std::isfinite(node_origin[0]) || !std::isfinite(node_origin[1]) ||
      !std::isfinite(node_origin[2])) {
    result.status = kStatusInvalidNumericInput;
    return result;
  }

  result.child_offset_blocks = read_le_i32(raw + 12);
  const uint8_t node_type = raw[16];
  const int exponent[3] = {static_cast<int8_t>(raw[18]),
                           static_cast<int8_t>(raw[19]),
                           static_cast<int8_t>(raw[20])};
  result.node_ray_mask = raw[21];
  if (node_type != kInternalPayloadKind || raw[17] != 0) {
    result.status = kStatusMalformedNode;
    return result;
  }

  bool prefix_ended = false;
  for (unsigned child = 0; child < kMaxChildren; ++child) {
    const uint8_t child_info = raw[22 + child];
    if ((child_info & 0xc0u) != 0) {
      result.status = kStatusMalformedNode;
      return result;
    }
    const uint8_t child_size = child_info & 0x03u;
    const uint8_t child_kind = (child_info >> 2) & 0x0fu;
    result.child_size[child] = child_size;
    result.child_kind[child] = child_kind;
    if (child_size == 0) {
      prefix_ended = true;
      continue;
    }
    if (prefix_ended || !supported_child_kind(input.level, child_kind)) {
      result.status = kStatusMalformedNode;
      return result;
    }
    result.evaluated_child_mask |= static_cast<uint8_t>(1u << child);

    const uint8_t low_quantized[3] = {raw[28 + child], raw[40 + child],
                                      raw[52 + child]};
    const uint8_t high_quantized[3] = {raw[34 + child], raw[46 + child],
                                       raw[58 + child]};
    float low[3] = {};
    float high[3] = {};
    for (unsigned axis = 0; axis < 3; ++axis) {
      low[axis] =
          decode_bound(node_origin[axis], low_quantized[axis], exponent[axis]);
      high[axis] = decode_bound(node_origin[axis], high_quantized[axis],
                                exponent[axis]);
      if (!std::isfinite(low[axis]) || !std::isfinite(high[axis]) ||
          low[axis] > high[axis]) {
        result.status = kStatusInvalidNumericInput;
        return result;
      }
    }

    const interval_result interval =
        intersect_aabb_strict(input.ray, low, high, input.committed_t);
    if (!interval.valid || !std::isfinite(interval.near_t)) {
      result.status = kStatusInvalidNumericInput;
      return result;
    }
    result.near_t_bits[child] = fp32_bits(interval.near_t);
    if (interval.hit &&
        (result.node_ray_mask & input.policy.cull_mask) != 0) {
      result.hit_child_mask |= static_cast<uint8_t>(1u << child);
      result.ordered_child_slots[result.candidate_count++] =
          static_cast<uint8_t>(child);
    }
  }

  for (unsigned lhs = 1; lhs < result.candidate_count; ++lhs) {
    const uint8_t slot = result.ordered_child_slots[lhs];
    float slot_near = 0.0f;
    std::memcpy(&slot_near, &result.near_t_bits[slot], sizeof(slot_near));
    unsigned rhs = lhs;
    while (rhs > 0) {
      const uint8_t previous_slot = result.ordered_child_slots[rhs - 1];
      float previous_near = 0.0f;
      std::memcpy(&previous_near, &result.near_t_bits[previous_slot],
                  sizeof(previous_near));
      if (previous_near < slot_near ||
          (previous_near == slot_near && previous_slot < slot)) {
        break;
      }
      result.ordered_child_slots[rhs] = previous_slot;
      --rhs;
    }
    result.ordered_child_slots[rhs] = slot;
  }

  result.status = kStatusOk;
  return result;
}

root_reference_seed_result_v0 execute_root_reference_seed(
    const root_reference_seed_input_v0 &input) {
  root_reference_seed_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  if (!valid_decode_context(input.decode_context, kLevelTlas)) {
    result.status = kStatusInvalidDecodeContext;
    return result;
  }
  const typed_blas::raw_header_envelope_v0 &envelope =
      input.raw_header.envelope;
  if (envelope.expected_chunk_count != 2 ||
      envelope.received_chunk_mask != 0x03 ||
      envelope.payload_byte_count != 64 || envelope.reserved_zero != 0) {
    result.status = kStatusMalformedRootHeader;
    return result;
  }

  result.root_payload_offset = read_le_u64(input.raw_header.raw_bytes);
  if (result.root_payload_offset < 64 ||
      (result.root_payload_offset & uint64_t{0x3f}) != 0 ||
      !payload_in_range(result.root_payload_offset, 64,
                        input.decode_context.device_range_bytes)) {
    result.root_payload_offset = 0;
    result.status = kStatusMalformedRootHeader;
    return result;
  }

  result.status = kStatusOk;
  return result;
}

route_result_v0 execute_route(const route_input_v0 &input) {
  return execute_route(input,
                       static_cast<candidate_result_v0 *>(NULL));
}

route_result_v0 execute_route(const route_input_v0 &input,
                              candidate_result_v0 *candidate_result) {
  replay_cursor_v0 cursor = {};
  return execute_route(input, cursor, candidate_result);
}

route_result_v0 execute_route(
    const route_input_v0 &input, const replay_cursor_v0 &cursor,
    candidate_result_v0 *candidate_result) {
  route_result_v0 result = {};
  result.status = kStatusInvalidArgument;

  candidate_result_v0 candidates = execute(input.candidate);
  if (candidates.status != kStatusOk) {
    if (candidate_result != NULL) *candidate_result = candidates;
    result.status = candidates.status;
    return result;
  }
  if (!apply_replay_cursor(cursor, &candidates)) {
    if (candidate_result != NULL) *candidate_result = candidates;
    result.status = kStatusInvalidReplayCursor;
    return result;
  }
  if (candidate_result != NULL) *candidate_result = candidates;
  if (!valid_decode_context(input.decode_context, input.candidate.level)) {
    result.status = kStatusInvalidDecodeContext;
    return result;
  }
  if ((input.current_payload_offset & uint64_t{0x3f}) != 0 ||
      !payload_in_range(input.current_payload_offset, 64,
                        input.decode_context.device_range_bytes)) {
    result.status = kStatusInvalidCurrentReference;
    return result;
  }

  uint64_t first_child_offset = 0;
  if (!checked_add_signed_blocks(input.current_payload_offset,
                                 candidates.child_offset_blocks,
                                 &first_child_offset)) {
    result.status = kStatusChildReferenceOutOfRange;
    return result;
  }

  compact_child_work_item_v0 children[kMaxChildren] = {};
  uint64_t child_offset = first_child_offset;
  for (unsigned child = 0; child < kMaxChildren; ++child) {
    if ((candidates.evaluated_child_mask & (1u << child)) == 0) continue;

    uint8_t expected_blocks = 0;
    uint16_t expected_bytes = 0;
    if (!expected_child_layout(candidates.child_kind[child],
                               &expected_blocks, &expected_bytes) ||
        candidates.child_size[child] != expected_blocks) {
      result.status = kStatusInvalidChildLayout;
      return result;
    }
    if (!payload_in_range(child_offset, expected_bytes,
                          input.decode_context.device_range_bytes)) {
      result.status = kStatusChildReferenceOutOfRange;
      return result;
    }

    children[child].payload_offset = child_offset;
    children[child].near_t_bits = candidates.near_t_bits[child];
    children[child].payload_byte_count = expected_bytes;
    children[child].payload_kind = candidates.child_kind[child];
    children[child].child_slot = static_cast<uint8_t>(child);

    const uint64_t child_bytes =
        static_cast<uint64_t>(candidates.child_size[child]) * uint64_t{64};
    if (!checked_add_u64(child_offset, child_bytes, &child_offset)) {
      result.status = kStatusChildReferenceOutOfRange;
      return result;
    }
  }

  result.status = kStatusOk;
  if (candidates.candidate_count == 0) {
    result.result_kind = kRouteResultMiss;
    return result;
  }

  const uint8_t selected_slot = candidates.ordered_child_slots[0];
  result.result_kind = kRouteResultSelected;
  result.output_valid_mask = kSelectedFetchValid;
  result.selected_fetch.child = children[selected_slot];
  result.selected_fetch.decode_context = input.decode_context;
  result.frontier_count = candidates.candidate_count - 1;
  for (unsigned index = 0; index < result.frontier_count; ++index) {
    const uint8_t slot = candidates.ordered_child_slots[index + 1];
    result.frontier[index] = children[slot];
  }
  if (result.frontier_count != 0) {
    result.output_valid_mask |= kFrontierItemsValid;
  }
  return result;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusUnsupportedProfile:
      return "unsupported_profile";
    case kStatusMalformedEnvelope:
      return "malformed_envelope";
    case kStatusMalformedNode:
      return "malformed_node";
    case kStatusInvalidNumericInput:
      return "invalid_numeric_input";
    case kStatusInvalidDecodeContext:
      return "invalid_decode_context";
    case kStatusInvalidCurrentReference:
      return "invalid_current_reference";
    case kStatusInvalidChildLayout:
      return "invalid_child_layout";
    case kStatusChildReferenceOutOfRange:
      return "child_reference_out_of_range";
    case kStatusMalformedRootHeader:
      return "malformed_root_header";
    case kStatusInvalidReplayCursor:
      return "invalid_replay_cursor";
  }
  return "unknown";
}

}  // namespace typed_node
}  // namespace v04
}  // namespace rtcore
