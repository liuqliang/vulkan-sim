#ifndef RTCORE_REPLAY_INTERFACE_H
#define RTCORE_REPLAY_INTERFACE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rtcore_v04_shadow_boundary.h"

class ptx_instruction;
class ptx_thread_info;

enum rtcore_candidate_gate_state {
  RTCORE_CANDIDATE_GATE_DISABLED = 0,
  RTCORE_CANDIDATE_GATE_ENABLED,
  RTCORE_CANDIDATE_GATE_INVALID,
};

inline char rtcore_candidate_gate_ascii_lower(char value) {
  return value >= 'A' && value <= 'Z' ? value - 'A' + 'a' : value;
}

inline bool rtcore_candidate_gate_value_is(const char *value,
                                           const char *expected) {
  if (value == NULL || expected == NULL) return value == expected;
  while (*value != '\0' && *expected != '\0') {
    if (rtcore_candidate_gate_ascii_lower(*value) !=
        rtcore_candidate_gate_ascii_lower(*expected)) {
      return false;
    }
    ++value;
    ++expected;
  }
  return *value == '\0' && *expected == '\0';
}

inline rtcore_candidate_gate_state rtcore_candidate_gate_state_for(
    const char *name) {
  const char *value = getenv(name);
  if (value == NULL || *value == '\0' || strcmp(value, "0") == 0 ||
      rtcore_candidate_gate_value_is(value, "false") ||
      rtcore_candidate_gate_value_is(value, "off") ||
      rtcore_candidate_gate_value_is(value, "no") ||
      rtcore_candidate_gate_value_is(value, "disabled")) {
    return RTCORE_CANDIDATE_GATE_DISABLED;
  }
  if (strcmp(value, "1") == 0 ||
      rtcore_candidate_gate_value_is(value, "true") ||
      rtcore_candidate_gate_value_is(value, "on") ||
      rtcore_candidate_gate_value_is(value, "yes") ||
      rtcore_candidate_gate_value_is(value, "enabled")) {
    return RTCORE_CANDIDATE_GATE_ENABLED;
  }
  return RTCORE_CANDIDATE_GATE_INVALID;
}

// Snapshot types crossing the cuda-sim/gpgpu-sim translation-unit boundary
// live here so producer and consumer cannot silently drift in size or layout.
struct rtcore_replay_service_cycle_identity_snapshot {
  bool valid;
  bool memory_progressed;
  bool ready_progressed;
  unsigned owner_hw_sid;
  unsigned thread_uid;
  unsigned lane_id;
  bool has_warp_metadata;
  unsigned warp_uid;
  unsigned warp_id;
  unsigned active_mask;
  unsigned static_inst_uid;
};

struct rtcore_boundary_candidate_snapshot {
  rtcore_boundary_candidate_snapshot()
      : valid(0),
        event_seq(0),
        shader_counter(0),
        hit_data_ref(0),
        hit_group_index(0),
        geometry_type(0),
        geometry_index(0),
        primitive_index(0),
        instance_index(0),
        hit_kind(0),
        v04_boundary_values() {}

  unsigned valid;
  unsigned event_seq;
  unsigned shader_counter;
  uint64_t hit_data_ref;
  unsigned hit_group_index;
  unsigned geometry_type;
  unsigned geometry_index;
  unsigned primitive_index;
  unsigned instance_index;
  unsigned hit_kind;
  rtcore::abi_v04::shadow::boundary_values v04_boundary_values;
};

struct rtcore_replay_warp_completion_entry_snapshot {
  bool enabled;
  bool found;
  bool all_active_lanes_complete;
  unsigned owner_hw_sid;
  unsigned warp_uid;
  unsigned warp_id;
  unsigned active_mask;
  unsigned admitted_lane_mask;
  unsigned completed_lane_mask;
  unsigned result_valid_mask;
  unsigned completed_lane_count;
  unsigned result_reg_base;
  unsigned result_data_slot[32];
  unsigned lane_status[32];
  unsigned packet_schema_version;
  unsigned context_profile_valid_mask;
  unsigned reported_attribute_metadata_valid_mask;
  unsigned inline_payload_location_valid_mask;
  unsigned inline_payload_base_word;
  unsigned max_inline_attribute_words;
  unsigned lane_completion_valid_mask;
  unsigned terminal_lane_mask;
  unsigned continuation_lane_mask;
  unsigned unsupported_reason_mask;
  unsigned handoff_selector_valid_mask;
  unsigned handoff_candidate_valid_mask;
  unsigned handoff_software_return_valid_mask;
  unsigned lane_completion_reason[32];
  unsigned lane_continuation_depth[32];
  unsigned context_layout_version[32];
  unsigned context_valid_flags[32];
  unsigned pipeline_profile_id[32];
  unsigned bvh_format_profile_id[32];
  rtcore_boundary_candidate_snapshot boundary_candidates[32];
  unsigned handoff_words[32][32];
  unsigned v04_shadow_boundary_image_valid_mask;
  unsigned v04_shadow_handoff_words[32][32];
  bool scoreboard_handoff_ready;
  bool scoreboard_handoff_delivered;
  unsigned long long scoreboard_handoff_cycle;
};

struct rtcore_v04_target_raw_read_transport_snapshot {
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint64_t raw_payload_base_address;
  uint32_t target_operation_seq;
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint32_t target_slot_generation;
  uint16_t raw_payload_bytes;
  uint16_t slot_chunk_offset;
  uint8_t target_kind;
  uint8_t target_slot_index;
  uint8_t producer_commit_required;
  uint8_t transfer_bytes;
  uint8_t operand_kind;
  uint8_t field_kind;
  uint8_t private_chunk_count;
  uint8_t valid;
  uint8_t reserved_zero[2];
};

struct rtcore_v04_stack_private_read_transport_snapshot {
  uint64_t reservation_id;
  uint64_t reservation_age;
  uint32_t target_operation_seq;
  union {
    uint32_t producer_operation_seq;
    uint32_t source_node_operation_seq;
  };
  uint32_t target_slot_generation;
  uint16_t slot_chunk_offset;
  uint8_t target_slot_index;
  uint8_t field_kind;
  uint8_t operation_kind;
  uint8_t read_phase;
  uint8_t valid;
  uint8_t reserved_zero[5];
};

struct rtcore_v04_private_write_transport_snapshot {
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint8_t field_kind;
  uint8_t valid;
  uint8_t reserved_zero[6];
};

struct rtcore_memory_unit_request_snapshot {
  bool valid;
  unsigned address_space;
  unsigned operation;
  unsigned destination;
  unsigned response_target;
  unsigned owner_hw_sid;
  unsigned rt_request_id;
  unsigned lane_id;
  unsigned resident_warp_id;
  unsigned request_generation;
  unsigned private_slot_id;
  unsigned memory_op_seq;
  unsigned chunk_id;
  unsigned chunk_count;
  unsigned access_kind;
  unsigned long long aligned_32b_addr;
  unsigned byte_mask;
  unsigned char payload[32];
  bool is_write;
  unsigned long long issue_cycle;
  rtcore_v04_target_raw_read_transport_snapshot v04_target_raw_read;
  rtcore_v04_stack_private_read_transport_snapshot v04_stack_private_read;
  rtcore_v04_private_write_transport_snapshot v04_private_write;
};

static const unsigned RTCORE_MEMORY_ADDRESS_SPACE_GLOBAL = 0u;
static const unsigned RTCORE_MEMORY_ADDRESS_SPACE_SHARED = 1u;
static const unsigned RTCORE_MEMORY_OPERATION_READ = 0u;
static const unsigned RTCORE_MEMORY_OPERATION_WRITE = 1u;
static const unsigned RTCORE_MEMORY_DESTINATION_LEGACY = 0u;
static const unsigned RTCORE_MEMORY_DESTINATION_PRIVATE_COMMIT_ACK = 1u;
static const unsigned RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL = 2u;
static const unsigned RTCORE_MEMORY_DESTINATION_STACK_QUEUE_FILL = 3u;
static const unsigned RTCORE_MEMORY_ACCESS_PRIVATE_FRONTIER_INIT = 8u;
static const unsigned RTCORE_MEMORY_ACCESS_TARGET_RAW_READ = 9u;
static const unsigned RTCORE_MEMORY_ACCESS_TARGET_PRIVATE_READ = 10u;
static const unsigned RTCORE_MEMORY_ACCESS_STACK_PRIVATE_READ = 11u;
static const unsigned RTCORE_MEMORY_ACCESS_PRIVATE_RUNTIME_WRITE = 12u;
static const unsigned RTCORE_MEMORY_ACCESS_STACK_SPILL_RECOVERY_READ = 13u;
static const unsigned RTCORE_MEMORY_ACCESS_HANDOFF_RAY_POLICY_READ = 14u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_RAW_GLOBAL = 1u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_PRIVATE_SHARED = 2u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_STACK_SPILL = 3u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_HANDOFF_RAY_POLICY = 4u;

extern "C" bool
rtcore_v04_private_frontier_live_init_memory_issue_profile_active();
extern "C" bool rtcore_v04_live_stack_pop_next_loop_gate_active();

static const unsigned RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE = 0xfffffff0u;
static const unsigned RTCORE_V04_LIVE_PUBLICATION_CHUNK_COUNT = 4u;

inline bool rtcore_v04_live_publication_memory_op_seq(unsigned memory_op_seq) {
  return memory_op_seq >= RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE &&
         memory_op_seq < RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE +
                             RTCORE_V04_LIVE_PUBLICATION_CHUNK_COUNT;
}

extern "C" {

bool rtcore_service_replay_cycle_for_sm_with_identity(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *service_enabled, bool *memory_progressed, bool *ready_progressed,
    rtcore_replay_service_cycle_identity_snapshot *identity_snapshot);

bool rtcore_service_replay_cycle_for_sm_with_identity_and_memory_unit(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *service_enabled, bool *memory_progressed, bool *ready_progressed,
    rtcore_replay_service_cycle_identity_snapshot *identity_snapshot,
    rtcore_memory_unit_request_snapshot *sideband_snapshot);

bool rtcore_pop_memory_unit_request_for_sm(
    unsigned owner_hw_sid,
    rtcore_memory_unit_request_snapshot *sideband_snapshot);

bool rtcore_push_front_memory_unit_request_for_sm(
    unsigned owner_hw_sid,
    const rtcore_memory_unit_request_snapshot *sideband_snapshot);

bool rtcore_push_back_memory_unit_request_for_sm(
    unsigned owner_hw_sid,
    const rtcore_memory_unit_request_snapshot *sideband_snapshot);

bool rtcore_accept_v04_target_raw_read_response(
    const rtcore_memory_unit_request_snapshot *request,
    const unsigned char *response_bytes, unsigned response_byte_count,
    unsigned long long response_cycle);

bool rtcore_accept_v04_target_private_shared_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_accept_v04_stack_spill_recovery_shared_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_accept_v04_handoff_ray_policy_shared_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_accept_v04_handoff_ray_policy_read_response(
    const rtcore_memory_unit_request_snapshot *request,
    const unsigned char *response_bytes, unsigned response_byte_count,
    unsigned long long response_cycle);

bool rtcore_accept_v04_stack_private_shared_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_query_replay_warp_completion_entry(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask,
    rtcore_replay_warp_completion_entry_snapshot *snapshot);

bool rtcore_commit_v04_functional_shader_visible_resubmit_admission(
    const ptx_instruction *pI, unsigned owner_hw_sid,
    unsigned new_warp_uid, unsigned warp_id,
    unsigned new_static_inst_uid, unsigned next_active_mask,
    unsigned expected_previous_warp_uid,
    unsigned expected_resident_generation,
    unsigned expected_previous_active_mask,
    unsigned long long handoff_window_base,
    ptx_thread_info *const *lane_threads,
    unsigned long long service_cycle, unsigned *previous_active_mask,
    unsigned *released_lane_mask, unsigned *reactivated_lane_mask,
    unsigned *resident_occupancy_before, unsigned *resident_occupancy_after,
    const char **failure_reason);

}

#endif
