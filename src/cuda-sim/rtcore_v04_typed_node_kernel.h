#ifndef RTCORE_V04_TYPED_NODE_KERNEL_H
#define RTCORE_V04_TYPED_NODE_KERNEL_H

#include <cstddef>
#include <cstdint>

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

static_assert(sizeof(raw_payload_header_v0) == 8,
              "raw payload header must remain 8 bytes");
static_assert(sizeof(raw_node_payload_v0) == 72,
              "GEN_RT raw node envelope must remain 72 bytes");
static_assert(sizeof(ray_state_v0) == 32,
              "local typed ray input must remain 32 bytes");
static_assert(sizeof(ray_policy_v0) == 8,
              "ray policy projection must remain 8 bytes");
static_assert(sizeof(candidate_input_v0) == 128,
              "typed node candidate input must remain 128 bytes");
static_assert(sizeof(candidate_result_v0) == 56,
              "typed node candidate result must remain 56 bytes");

bool make_raw_node_payload(const void *raw_node_bytes,
                           raw_node_payload_v0 *payload);

candidate_result_v0 execute(const candidate_input_v0 &input);

const char *status_name(status_kind status);

}  // namespace typed_node
}  // namespace v04
}  // namespace rtcore

#endif
