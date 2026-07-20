#ifndef RTCORE_REPLAY_INTERFACE_H
#define RTCORE_REPLAY_INTERFACE_H

#include <stdint.h>

#include "rtcore_v04_shadow_boundary.h"

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

struct rtcore_memory_unit_request_snapshot {
  bool valid;
  unsigned response_target;
  unsigned owner_hw_sid;
  unsigned rt_request_id;
  unsigned lane_id;
  unsigned memory_op_seq;
  unsigned chunk_id;
  unsigned chunk_count;
  unsigned access_kind;
  unsigned long long aligned_32b_addr;
  bool is_write;
  unsigned long long issue_cycle;
};

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

bool rtcore_query_replay_warp_completion_entry(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask,
    rtcore_replay_warp_completion_entry_snapshot *snapshot);

}

#endif
