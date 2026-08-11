#ifndef RTCORE_REPLAY_INTERFACE_H
#define RTCORE_REPLAY_INTERFACE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rtcore_v04_shadow_boundary.h"

class ptx_instruction;
class ptx_thread_info;
class memory_space;

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
  bool v04_native_resident_completion;
  bool v04_functional_only_completion;
  bool all_active_lanes_complete;
  unsigned v04_native_resident_generation;
  unsigned v04_native_completion_transaction_generation;
  unsigned v04_native_identity_valid_mask;
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
  unsigned long long v04_native_context_ptr[32];
  unsigned long long v04_native_handoff_window_base[32];
  unsigned v04_native_token_id[32];
  unsigned v04_native_token_allocator_generation[32];
  unsigned v04_native_window_generation[32];
  unsigned v04_native_packed_request_key[32];
  unsigned v04_native_request_generation[32];
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
  uint32_t private_layout_profile_id;
  uint32_t bvh_format_profile_id;
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
  uint8_t operation_kind;
  uint8_t private_storage_profile;
  uint8_t reserved_zero[1];
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

struct rtcore_v04_private_state_384_read_transport_snapshot {
  uint64_t private_slot_base_address;
  uint32_t operation_sequence;
  uint32_t private_layout_profile_id;
  uint32_t bvh_format_profile_id;
  uint32_t reservation_generation;
  uint8_t consumer;
  uint8_t operation_kind;
  uint8_t completion_reason;
  uint8_t read_index;
  uint8_t read_count;
  uint8_t storage_profile;
  uint8_t valid;
  uint8_t memory_op_seq_base;
};

struct rtcore_v04_handoff_publication_transport_snapshot {
  uint32_t producer_operation_seq;
  uint32_t producer_commit_epoch;
  uint8_t publication_chunk;
  uint8_t valid;
  uint8_t reserved_zero[2];
};

struct rtcore_v04_live_handoff_acquire_transport_snapshot {
  uint32_t resident_warp_generation;
  uint32_t window_generation;
  uint32_t dynamic_warp_id;
  uint32_t submit_warp_uid;
  uint32_t acquire_active_mask;
  uint8_t valid;
  uint8_t preaccepted;
  uint8_t transaction_kind;
  uint8_t reserved_zero[1];
};

struct rtcore_v04_dispatch_handoff_read_transport_snapshot {
  uint32_t resident_warp_generation;
  uint32_t window_generation;
  uint32_t dynamic_warp_id;
  uint32_t warp_uid;
  uint32_t cohort_lane_mask;
  uint32_t dispatch_generation;
  uint16_t cohort_index;
  uint8_t lane_slot_chunk;
  uint8_t lane_read_count;
  uint8_t phase;
  uint8_t valid;
  uint8_t preaccepted;
  uint8_t reserved_zero[1];
};

struct rtcore_v04_global384_private_init_transport_snapshot {
  uint32_t resident_warp_generation;
  uint32_t dynamic_warp_id;
  uint32_t submit_warp_uid;
  uint32_t submit_active_mask;
  uint32_t warp_id;
  uint8_t resident_warp_slot;
  uint8_t field_kind;
  uint8_t valid;
  uint8_t accepted;
};

struct rtcore_v04_live_transaction_transport_snapshot {
  uint64_t transaction_id;
  uint64_t record_id;
  uint32_t owner_hw_sid;
  uint32_t owner_generation;
  uint32_t window_generation;
  uint8_t lane_id;
  uint8_t object;
  uint8_t access;
  uint8_t valid;
};

enum rtcore_v04_semantic_memory_tag {
  RTCORE_V04_SEMANTIC_TAG_INVALID = 0,
  RTCORE_V04_SEMANTIC_TAG_HANDOFF_RTCORE_ACQUIRE = 1,
  RTCORE_V04_SEMANTIC_TAG_HANDOFF_RTCORE_PUBLISH = 2,
  RTCORE_V04_SEMANTIC_TAG_HANDOFF_SHADER_TRACE_INPUT_PUBLISH = 3,
  RTCORE_V04_SEMANTIC_TAG_HANDOFF_SHADER_DISPATCH_READ = 4,
  RTCORE_V04_SEMANTIC_TAG_HANDOFF_SHADER_BUILTIN_READ = 5,
  RTCORE_V04_SEMANTIC_TAG_HANDOFF_SHADER_RETURN = 6,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_RAY = 7,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_AS_CONTEXT = 8,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_COMMITTED_HIT = 9,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_INSTANCE_POLICY = 10,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_SHORT_STACK_METADATA = 11,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_SHORT_STACK_ENTRY = 12,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_BOUNDARY_UNION = 13,
  RTCORE_V04_SEMANTIC_TAG_PRIVATE_PARENT_RESTORE = 14,
  RTCORE_V04_SEMANTIC_TAG_CONTEXT_SHADER_READ = 15,
  RTCORE_V04_SEMANTIC_TAG_CONTEXT_SHADER_WRITE = 16,
  RTCORE_V04_SEMANTIC_TAG_CONTEXT_FUNCTIONAL_SETUP = 17,
  RTCORE_V04_SEMANTIC_TAG_COUNT = 18,
};

static const unsigned RTCORE_V04_SEMANTIC_MAX_SLICES = 4u;

struct rtcore_v04_semantic_memory_slice_snapshot {
  uint32_t useful_byte_mask;
  uint8_t tag;
  uint8_t reserved_zero[3];
};

struct rtcore_v04_semantic_memory_snapshot {
  uint32_t tag_set;
  uint32_t useful_byte_mask;
  uint32_t physical_request_uid;
  uint8_t slice_count;
  uint8_t valid;
  uint8_t canonical_private_chunk;
  uint8_t canonical_private_chunk_valid;
  rtcore_v04_semantic_memory_slice_snapshot
      slices[RTCORE_V04_SEMANTIC_MAX_SLICES];
};

inline bool rtcore_v04_add_semantic_memory_slice(
    rtcore_v04_semantic_memory_snapshot *semantic, uint8_t tag,
    uint32_t useful_byte_mask) {
  if (semantic == NULL ||
      tag <= RTCORE_V04_SEMANTIC_TAG_INVALID ||
      tag >= RTCORE_V04_SEMANTIC_TAG_COUNT || useful_byte_mask == 0) {
    return false;
  }
  const uint32_t tag_bit = uint32_t{1} << (tag - 1u);
  for (unsigned index = 0; index < semantic->slice_count; ++index) {
    if (semantic->slices[index].tag == tag) {
      semantic->slices[index].useful_byte_mask |= useful_byte_mask;
      semantic->tag_set |= tag_bit;
      semantic->useful_byte_mask |= useful_byte_mask;
      semantic->valid = 1;
      return true;
    }
  }
  if (semantic->slice_count >= RTCORE_V04_SEMANTIC_MAX_SLICES) {
    return false;
  }
  rtcore_v04_semantic_memory_slice_snapshot &slice =
      semantic->slices[semantic->slice_count++];
  slice.tag = tag;
  slice.useful_byte_mask = useful_byte_mask;
  semantic->tag_set |= tag_bit;
  semantic->useful_byte_mask |= useful_byte_mask;
  semantic->valid = 1;
  return true;
}

inline bool rtcore_v04_semantic_memory_snapshot_valid(
    const rtcore_v04_semantic_memory_snapshot &semantic,
    uint32_t request_byte_mask) {
  const uint32_t private_tag_mask =
      ((uint32_t{1} << 14u) - 1u) & ~((uint32_t{1} << 6u) - 1u);
  const bool private_only =
      semantic.tag_set != 0 &&
      (semantic.tag_set & ~private_tag_mask) == 0;
  if (semantic.valid != 1 || semantic.slice_count == 0 ||
      semantic.slice_count > RTCORE_V04_SEMANTIC_MAX_SLICES ||
      semantic.tag_set == 0 || semantic.useful_byte_mask == 0 ||
      (semantic.useful_byte_mask & ~request_byte_mask) != 0 ||
      semantic.canonical_private_chunk_valid > 1 ||
      (semantic.canonical_private_chunk_valid == 1) != private_only ||
      (private_only && semantic.canonical_private_chunk >= 12u) ||
      (!private_only && semantic.canonical_private_chunk != 0u)) {
    return false;
  }
  uint32_t rebuilt_tag_set = 0;
  uint32_t rebuilt_useful_mask = 0;
  for (unsigned index = 0; index < semantic.slice_count; ++index) {
    const rtcore_v04_semantic_memory_slice_snapshot &slice =
        semantic.slices[index];
    if (slice.tag <= RTCORE_V04_SEMANTIC_TAG_INVALID ||
        slice.tag >= RTCORE_V04_SEMANTIC_TAG_COUNT ||
        slice.useful_byte_mask == 0 ||
        (slice.useful_byte_mask & ~request_byte_mask) != 0 ||
        slice.reserved_zero[0] != 0 || slice.reserved_zero[1] != 0 ||
        slice.reserved_zero[2] != 0) {
      return false;
    }
    const uint32_t tag_bit = uint32_t{1} << (slice.tag - 1u);
    if ((rebuilt_tag_set & tag_bit) != 0) return false;
    rebuilt_tag_set |= tag_bit;
    rebuilt_useful_mask |= slice.useful_byte_mask;
  }
  return rebuilt_tag_set == semantic.tag_set &&
         rebuilt_useful_mask == semantic.useful_byte_mask;
}

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
  rtcore_v04_private_state_384_read_transport_snapshot
      v04_private_state_384_read;
  rtcore_v04_handoff_publication_transport_snapshot
      v04_handoff_publication;
  rtcore_v04_live_handoff_acquire_transport_snapshot
      v04_live_handoff_acquire;
  rtcore_v04_dispatch_handoff_read_transport_snapshot
      v04_dispatch_handoff_read;
  rtcore_v04_global384_private_init_transport_snapshot
      v04_global384_private_init;
  rtcore_v04_live_transaction_transport_snapshot v04_live_transaction;
  rtcore_v04_semantic_memory_snapshot v04_semantic_memory;
};

extern "C" void rtcore_record_v04_global384_context_functional_setup(
    unsigned owner_hw_sid, unsigned lane_id, unsigned byte_count);
extern "C" unsigned long long
rtcore_v04_global384_context_functional_setup_access_count();
extern "C" unsigned long long
rtcore_v04_global384_context_functional_setup_byte_count();
extern "C" void rtcore_record_v04_global384_semantic_last_arrival(
    const rtcore_memory_unit_request_snapshot *snapshot,
    unsigned long long completion_cycle);

extern "C" bool rtcore_record_v04_memory_conservation_event(
    const rtcore_memory_unit_request_snapshot *snapshot,
    bool response, unsigned long long cycle);

extern "C" bool
rtcore_validate_v04_global384_resubmit_handoff_acquire_chunk(
    const rtcore_memory_unit_request_snapshot *snapshot,
    const unsigned char *response_bytes, unsigned response_byte_count);

extern "C" bool
rtcore_complete_v04_global384_resubmit_handoff_acquire_chunk(
    const rtcore_memory_unit_request_snapshot *snapshot,
    const unsigned char *response_bytes, unsigned response_byte_count,
    unsigned long long completion_cycle);

extern "C" bool
rtcore_validate_v04_global384_initial_handoff_acquire_chunk(
    const rtcore_memory_unit_request_snapshot *snapshot,
    const unsigned char *response_bytes, unsigned response_byte_count);

extern "C" bool
rtcore_complete_v04_global384_initial_handoff_acquire_chunk(
    const rtcore_memory_unit_request_snapshot *snapshot,
    const unsigned char *response_bytes, unsigned response_byte_count,
    unsigned long long completion_cycle);

extern "C" bool rtcore_accept_v04_global384_private_init_write(
    rtcore_memory_unit_request_snapshot *snapshot,
    unsigned long long accept_cycle);

extern "C" bool rtcore_complete_v04_global384_private_init_write(
    const rtcore_memory_unit_request_snapshot *snapshot,
    memory_space *global_memory, unsigned long long response_address,
    unsigned long long completion_cycle);

extern "C" bool rtcore_complete_v04_global384_private_runtime_write(
    const rtcore_memory_unit_request_snapshot *snapshot,
    memory_space *global_memory, unsigned long long response_address,
    unsigned long long completion_cycle);

static const unsigned RTCORE_MEMORY_ADDRESS_SPACE_GLOBAL = 0u;
static const unsigned RTCORE_MEMORY_ADDRESS_SPACE_SHARED = 1u;
static const unsigned RTCORE_MEMORY_OPERATION_READ = 0u;
static const unsigned RTCORE_MEMORY_OPERATION_WRITE = 1u;
static const unsigned RTCORE_MEMORY_DESTINATION_LEGACY = 0u;
static const unsigned RTCORE_MEMORY_DESTINATION_PRIVATE_COMMIT_ACK = 1u;
static const unsigned RTCORE_MEMORY_DESTINATION_TARGET_QUEUE_FILL = 2u;
static const unsigned RTCORE_MEMORY_DESTINATION_STACK_QUEUE_FILL = 3u;
static const unsigned RTCORE_MEMORY_DESTINATION_HANDOFF_PUBLICATION_ACK = 4u;
static const unsigned RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL = 5u;
static const unsigned RTCORE_MEMORY_DESTINATION_PRIVATE_BOUNDARY_FILL = 6u;
static const unsigned RTCORE_MEMORY_ACCESS_PRIVATE_FRONTIER_INIT = 8u;
static const unsigned RTCORE_MEMORY_ACCESS_TARGET_RAW_READ = 9u;
static const unsigned RTCORE_MEMORY_ACCESS_TARGET_PRIVATE_READ = 10u;
static const unsigned RTCORE_MEMORY_ACCESS_STACK_PRIVATE_READ = 11u;
static const unsigned RTCORE_MEMORY_ACCESS_PRIVATE_RUNTIME_WRITE = 12u;
static const unsigned RTCORE_MEMORY_ACCESS_STACK_SPILL_RECOVERY_READ = 13u;
static const unsigned RTCORE_MEMORY_ACCESS_HANDOFF_RAY_POLICY_READ = 14u;
static const unsigned RTCORE_MEMORY_ACCESS_HANDOFF_PUBLICATION_WRITE = 15u;
static const unsigned RTCORE_MEMORY_ACCESS_SHORT_STACK_STATE_READ = 16u;
static const unsigned RTCORE_MEMORY_ACCESS_SHORT_STACK_RETURN_INSTANCE_READ =
    17u;
static const unsigned RTCORE_MEMORY_ACCESS_PRIVATE_STATE_384_READ = 18u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_RAW_GLOBAL = 1u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_PRIVATE_SHARED = 2u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_STACK_SPILL = 3u;
static const unsigned RTCORE_MEMORY_TARGET_OPERAND_HANDOFF_RAY_POLICY = 4u;
static const unsigned RTCORE_V04_HANDOFF_ACQUIRE_TRANSACTION_NONE = 0u;
static const unsigned RTCORE_V04_HANDOFF_ACQUIRE_TRANSACTION_INITIAL = 1u;
static const unsigned RTCORE_V04_HANDOFF_ACQUIRE_TRANSACTION_RESUBMIT = 2u;

extern "C" bool
rtcore_v04_private_frontier_live_init_memory_issue_profile_active();
extern "C" bool rtcore_v04_live_stack_pop_next_loop_gate_active();
extern "C" bool rtcore_v04_live_instance_restore_parent_gate_active();
extern "C" bool rtcore_v04_live_instance_enter_transition_gate_active();
extern "C" bool rtcore_v04_live_primitive_timing_route_gate_active();
extern "C" bool rtcore_v04_native_boundary_completion_gate_active();
extern "C" bool rtcore_v04_continuation_lifecycle_gate_active();
extern "C" bool
rtcore_v04_live_stack_terminal_publication_gate_active();

static const unsigned RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE = 0xfffffff0u;
static const unsigned RTCORE_V04_LIVE_PUBLICATION_CHUNK_COUNT = 4u;
static const unsigned RTCORE_V04_NATIVE_PUBLICATION_OP_SEQ_BASE = 0xffffffe0u;
static const unsigned RTCORE_V04_NATIVE_PUBLICATION_CHUNK_COUNT = 4u;

inline bool rtcore_v04_live_publication_memory_op_seq(unsigned memory_op_seq) {
  return memory_op_seq >= RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE &&
         memory_op_seq < RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE +
                             RTCORE_V04_LIVE_PUBLICATION_CHUNK_COUNT;
}

inline bool rtcore_v04_native_publication_memory_op_seq(
    unsigned memory_op_seq) {
  return memory_op_seq >= RTCORE_V04_NATIVE_PUBLICATION_OP_SEQ_BASE &&
         memory_op_seq <
             RTCORE_V04_NATIVE_PUBLICATION_OP_SEQ_BASE +
                 RTCORE_V04_NATIVE_PUBLICATION_CHUNK_COUNT;
}

inline bool rtcore_v04_native_publication_requires_exact_address(
    const rtcore_memory_unit_request_snapshot &request) {
  return request.destination ==
             RTCORE_MEMORY_DESTINATION_HANDOFF_PUBLICATION_ACK ||
         request.access_kind ==
             RTCORE_MEMORY_ACCESS_HANDOFF_PUBLICATION_WRITE ||
         request.v04_live_handoff_acquire.valid == 1 ||
         request.v04_global384_private_init.valid == 1;
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

bool rtcore_peek_memory_unit_request_for_sm(
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

bool rtcore_accept_v04_target_private_state_384_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_accept_v04_target_private_state_384_read_response(
    const rtcore_memory_unit_request_snapshot *request,
    const unsigned char *response_bytes, unsigned response_byte_count,
    unsigned long long response_cycle);

bool rtcore_accept_v04_stack_spill_recovery_shared_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_accept_v04_stack_spill_recovery_read_response(
    const rtcore_memory_unit_request_snapshot *request,
    const unsigned char *response_bytes, unsigned response_byte_count,
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

bool rtcore_accept_v04_short_stack_private_shared_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_accept_v04_short_stack_private_state_384_read_response(
    const rtcore_memory_unit_request_snapshot *request,
    const unsigned char *response_bytes, unsigned response_byte_count,
    unsigned long long response_cycle);

bool rtcore_accept_v04_private_boundary_state_384_read(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_accept_v04_private_boundary_state_384_read_response(
    const rtcore_memory_unit_request_snapshot *request,
    const unsigned char *response_bytes, unsigned response_byte_count,
    unsigned long long response_cycle);

bool rtcore_accept_v04_short_stack_return_instance_read(
    const rtcore_memory_unit_request_snapshot *request,
    const uint8_t *response_payload,
    unsigned response_payload_bytes,
    unsigned long long response_cycle);

bool rtcore_accept_v04_handoff_publication_response(
    const rtcore_memory_unit_request_snapshot *request,
    unsigned long long response_cycle);

bool rtcore_query_replay_warp_completion_entry(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask,
    rtcore_replay_warp_completion_entry_snapshot *snapshot);

bool rtcore_query_resident_rt_warp_active_lane_state(
    unsigned owner_hw_sid, unsigned warp_id,
    unsigned *resident_active_lane_count, unsigned *warp_active_mask,
    bool *warp_resident);

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

bool rtcore_validate_v04_functional_only_compatibility_resubmit_lane(
    unsigned owner_hw_sid, unsigned new_warp_uid, unsigned warp_id,
    unsigned next_active_mask, unsigned lane_id, unsigned thread_uid,
    unsigned long long context_ptr, unsigned long long handoff_window_base,
    unsigned token_id, unsigned token_allocator_generation,
    unsigned window_generation, const char **failure_reason);

bool rtcore_commit_v04_functional_only_compatibility_resubmit(
    unsigned owner_hw_sid, unsigned new_warp_uid, unsigned warp_id,
    unsigned new_static_inst_uid, unsigned next_active_mask,
    unsigned expected_previous_warp_uid,
    unsigned expected_resident_generation,
    unsigned long long service_cycle, unsigned *previous_active_mask,
    unsigned *released_lane_mask, unsigned *reactivated_lane_mask,
    unsigned *resident_occupancy_before, unsigned *resident_occupancy_after,
    const char **failure_reason);

bool rtcore_stage_v04_native_continuation_resubmit(
    unsigned owner_hw_sid, unsigned dynamic_warp_id,
    unsigned new_warp_uid, unsigned warp_id, unsigned new_static_inst_uid,
    unsigned next_active_mask,
    unsigned expected_previous_warp_uid,
    unsigned expected_resident_generation,
    unsigned long long handoff_window_base,
    unsigned long long service_cycle, unsigned *previous_active_mask,
    unsigned *released_lane_mask, unsigned *reactivated_lane_mask,
    unsigned *resident_occupancy_before, unsigned *resident_occupancy_after,
    const char **failure_reason);

bool rtcore_validate_v04_native_continuation_lane_authority(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned resident_generation,
    unsigned completion_transaction_generation, unsigned lane_id,
    unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned token_id,
    unsigned token_allocator_generation, unsigned window_generation,
    unsigned packed_request_key, unsigned request_generation);

bool rtcore_mark_v04_native_continuation_dispatch_complete(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned resident_generation,
    unsigned completion_transaction_generation,
    unsigned completed_lane_mask);

bool rtcore_mark_v04_native_shader_terminal_publication(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned resident_generation,
    unsigned completion_transaction_generation,
    unsigned terminal_lane_mask);

bool rtcore_query_v04_functional_only_completion_entry(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask,
    rtcore_replay_warp_completion_entry_snapshot *snapshot);

bool rtcore_consume_v04_functional_only_completion_entry(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask);
bool rtcore_mark_v04_functional_only_dispatch_cohort_complete(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned resident_generation,
    unsigned completion_transaction_generation,
    unsigned completed_lane_mask);

bool rtcore_validate_v04_functional_only_continuation_lane_authority(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned resident_generation,
    unsigned completion_transaction_generation, unsigned lane_id,
    unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned token_id,
    unsigned token_allocator_generation, unsigned window_generation,
    unsigned packed_request_key, unsigned request_generation);

bool rtcore_apply_v04_functional_only_terminal_shader_return(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned lane_id,
    const ptx_instruction *source_inst, ptx_thread_info *thread);

}

#endif
