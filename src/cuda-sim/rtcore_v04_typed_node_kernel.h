#ifndef RTCORE_V04_TYPED_NODE_KERNEL_H
#define RTCORE_V04_TYPED_NODE_KERNEL_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_typed_blas_decode_context.h"

namespace rtcore {
namespace v04 {
namespace typed_node {

static const uint32_t kGenRtDerivedProfileId = 0x00010001u;
static const uint8_t kInternalPayloadKind = 0u;
static const uint8_t kInstancePayloadKind = 1u;
static const uint8_t kProceduralPayloadKind = 3u;
static const uint8_t kQuadPayloadKind = 4u;
static const unsigned kMaxChildren = 6u;

enum level_kind : uint8_t {
  kLevelInvalid = 0,
  kLevelTlas = 1,
  kLevelBlas = 2,
};

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedProfile,
  kStatusMalformedEnvelope,
  kStatusMalformedNode,
  kStatusInvalidNumericInput,
  kStatusInvalidDecodeContext,
  kStatusInvalidCurrentReference,
  kStatusInvalidChildLayout,
  kStatusChildReferenceOutOfRange,
  kStatusMalformedRootHeader,
};

enum route_result_kind : uint8_t {
  kRouteResultInvalid = 0,
  kRouteResultMiss = 1,
  kRouteResultSelected = 2,
};

enum route_output_valid_bit : uint8_t {
  kSelectedFetchValid = 1u << 0,
  kFrontierItemsValid = 1u << 1,
};

struct raw_payload_header_v0 {
  uint8_t expected_payload_kind;
  uint8_t expected_chunk_count;
  uint16_t payload_byte_count;
  uint8_t received_chunk_mask;
  uint8_t reserved_zero[3];
};

struct raw_node_payload_v0 {
  raw_payload_header_v0 header;
  uint8_t raw_bytes[64];
};

struct ray_state_v0 {
  float origin[3];
  float direction[3];
  float inverse_direction[3];
  float t_min;
  float t_max;
};

struct ray_policy_v0 {
  uint32_t ray_flags;
  uint8_t cull_mask;
  uint8_t reserved_zero[3];
};

struct alignas(16) candidate_input_v0 {
  uint32_t profile_id;
  uint8_t level;
  uint8_t reserved_zero0[3];
  ray_state_v0 ray;
  ray_policy_v0 policy;
  float committed_t;
  uint32_t reserved_zero1;
  raw_node_payload_v0 raw_node;
};

struct candidate_result_v0 {
  uint8_t status;
  uint8_t evaluated_child_mask;
  uint8_t hit_child_mask;
  uint8_t candidate_count;
  int32_t child_offset_blocks;
  uint8_t node_ray_mask;
  uint8_t child_size[kMaxChildren];
  uint8_t child_kind[kMaxChildren];
  uint8_t ordered_child_slots[kMaxChildren];
  uint8_t reserved_zero[3];
  uint32_t near_t_bits[kMaxChildren];
};

struct compact_child_work_item_v0 {
  uint64_t payload_offset;
  uint32_t near_t_bits;
  uint16_t payload_byte_count;
  uint8_t payload_kind;
  uint8_t child_slot;
};

struct selected_child_fetch_work_item_v0 {
  compact_child_work_item_v0 child;
  typed_blas::as_decode_context_v0 decode_context;
};

struct alignas(16) root_reference_seed_input_v0 {
  typed_blas::as_decode_context_v0 decode_context;
  typed_blas::raw_bvh_header_v0 raw_header;
};

struct alignas(16) root_reference_seed_result_v0 {
  uint8_t status;
  uint8_t reserved_zero[7];
  uint64_t root_payload_offset;
};

struct alignas(16) route_input_v0 {
  candidate_input_v0 candidate;
  typed_blas::as_decode_context_v0 decode_context;
  uint64_t current_payload_offset;
};

struct alignas(16) route_result_v0 {
  uint8_t status;
  uint8_t result_kind;
  uint8_t frontier_count;
  uint8_t output_valid_mask;
  uint8_t reserved_zero[12];
  selected_child_fetch_work_item_v0 selected_fetch;
  compact_child_work_item_v0 frontier[kMaxChildren - 1];
  uint8_t reserved_zero_tail[8];
};

static_assert(sizeof(raw_payload_header_v0) == 8,
              "raw payload header must remain 8 bytes");
static_assert(sizeof(raw_node_payload_v0) == 72,
              "GEN_RT raw node envelope must remain 72 bytes");
static_assert(sizeof(ray_state_v0) == 44,
              "canonical typed ray input must remain 44 bytes");
static_assert(sizeof(ray_policy_v0) == 8,
              "ray policy projection must remain 8 bytes");
static_assert(sizeof(candidate_input_v0) == 144,
              "typed node candidate input must remain 144 bytes");
static_assert(sizeof(candidate_result_v0) == 56,
              "typed node candidate result must remain 56 bytes");
static_assert(sizeof(compact_child_work_item_v0) == 16,
              "compact child work item must remain 16 bytes");
static_assert(sizeof(selected_child_fetch_work_item_v0) == 56,
              "selected child fetch item must remain 56 bytes");
static_assert(sizeof(root_reference_seed_input_v0) == 112,
              "typed root reference seed input must remain 112 bytes");
static_assert(alignof(root_reference_seed_input_v0) == 16,
              "typed root reference seed input must remain aligned");
static_assert(offsetof(root_reference_seed_input_v0, raw_header) == 40,
              "typed root reference raw header offset changed");
static_assert(sizeof(root_reference_seed_result_v0) == 16,
              "typed root reference seed result must remain 16 bytes");
static_assert(sizeof(route_input_v0) == 192,
              "typed node route input must remain 192 bytes");
static_assert(alignof(route_input_v0) == 16,
              "typed node route input must remain aligned");
static_assert(offsetof(route_input_v0, decode_context) == 144,
              "typed node route context offset changed");
static_assert(offsetof(route_input_v0, current_payload_offset) == 184,
              "typed node current reference offset changed");
static_assert(sizeof(route_result_v0) == 160,
              "typed node route result must remain 160 bytes");
static_assert(alignof(route_result_v0) == 16,
              "typed node route result must remain aligned");
static_assert(offsetof(route_result_v0, selected_fetch) == 16,
              "typed node selected fetch offset changed");
static_assert(offsetof(route_result_v0, frontier) == 72,
              "typed node frontier offset changed");
static_assert(offsetof(route_result_v0, reserved_zero_tail) == 152,
              "typed node route tail offset changed");

bool make_raw_node_payload(const void *raw_node_bytes,
                           raw_node_payload_v0 *payload);

candidate_result_v0 execute(const candidate_input_v0 &input);

root_reference_seed_result_v0 execute_root_reference_seed(
    const root_reference_seed_input_v0 &input);

route_result_v0 execute_route(const route_input_v0 &input);
route_result_v0 execute_route(const route_input_v0 &input,
                              candidate_result_v0 *candidate_result);

const char *status_name(status_kind status);

}  // namespace typed_node
}  // namespace v04
}  // namespace rtcore

#endif
