#ifndef RTCORE_V04_ROOT_NODE_PACKET_H
#define RTCORE_V04_ROOT_NODE_PACKET_H

#include <array>
#include <cstdint>

#include "rtcore_abi_v04_generated.h"
#include "rtcore_v04_fetch_target_queue.h"

class ptx_instruction;
class ptx_thread_info;

namespace rtcore {
namespace v04 {
namespace root_node_packet {

static const uint8_t kLaneCapacity = 32;

struct lane_input_v0 {
  bool valid;
  uint8_t lane_id;
  uint8_t reserved_zero0[2];
  uint32_t thread_uid;
  uint64_t context_ptr;
  uint64_t handoff_window_base;
  uint32_t sbt_record_offset;
  uint32_t sbt_record_stride;
  uint32_t miss_index;
  uint32_t reserved_zero1;
  bool v04_trace_input_valid;
  uint8_t reserved_zero2[7];
  std::array<uint32_t, abi_v04::kWordCount> v04_trace_input_words;
  fetch_target::target_reference_v0 target_reference;
  private_frontier::root_private_operands_v0 private_operands;
  typed_node::ray_policy_v0 ray_policy;
  uint64_t raw_payload_base_address;
  uint32_t root_build_generation;
  uint32_t reserved_zero3;
};

struct warp_input_v0 {
  bool valid;
  uint8_t reserved_zero0[3];
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t static_inst_uid;
  uint32_t active_mask;
  lane_input_v0 lanes[kLaneCapacity];
};

enum shared_handoff_window_status_kind : uint8_t {
  kSharedHandoffWindowOk = 0,
  kSharedHandoffWindowInvalidInput,
  kSharedHandoffWindowMisaligned,
  kSharedHandoffWindowDivergent,
  kSharedHandoffWindowAddressOverflow,
};

inline shared_handoff_window_status_kind validate_shared_handoff_window(
    const warp_input_v0 &input, uint64_t *shared_window_base) {
  if (shared_window_base == NULL || !input.valid ||
      input.active_mask == 0) {
    return kSharedHandoffWindowInvalidInput;
  }
  uint64_t common_base = 0;
  for (unsigned lane = 0; lane < kLaneCapacity; ++lane) {
    if ((input.active_mask & (uint32_t{1} << lane)) == 0) continue;
    const lane_input_v0 &lane_input = input.lanes[lane];
    if (!lane_input.valid || lane_input.lane_id != lane ||
        lane_input.handoff_window_base == 0) {
      return kSharedHandoffWindowInvalidInput;
    }
    if ((lane_input.handoff_window_base &
         (abi_v04::kLaneSlotBytes - 1)) != 0) {
      return kSharedHandoffWindowMisaligned;
    }
    if (common_base == 0) {
      common_base = lane_input.handoff_window_base;
    } else if (common_base != lane_input.handoff_window_base) {
      return kSharedHandoffWindowDivergent;
    }
    const uint64_t lane_offset =
        static_cast<uint64_t>(lane) * abi_v04::kLaneSlotBytes;
    if (lane_input.handoff_window_base > UINT64_MAX - lane_offset) {
      return kSharedHandoffWindowAddressOverflow;
    }
  }
  *shared_window_base = common_base;
  return kSharedHandoffWindowOk;
}

}  // namespace root_node_packet
}  // namespace v04
}  // namespace rtcore

extern "C" bool rtcore_v04_root_node_ready_packet_gate_active();
extern "C" bool rtcore_v04_functional_node_driver_gate_active();
extern "C" bool rtcore_v04_functional_only_engine_gate_active();
extern "C" bool rtcore_v04_live_node_timing_gate_active();
extern "C" bool rtcore_v04_root_node_input_gate_active();
extern "C" bool rtcore_v04_functional_node_driver_configuration_valid();

enum rtcore_v04_first_submit_live_bind_preissue_status {
  RTCORE_V04_FIRST_SUBMIT_LIVE_BIND_NOT_APPLICABLE = 0,
  RTCORE_V04_FIRST_SUBMIT_LIVE_BIND_READY = 1,
  RTCORE_V04_FIRST_SUBMIT_LIVE_BIND_WAIT = 2,
  RTCORE_V04_FIRST_SUBMIT_LIVE_BIND_FAULT = 3,
};

extern "C" rtcore_v04_first_submit_live_bind_preissue_status
rtcore_service_v04_global384_first_submit_live_bind_before_issue(
    const ptx_instruction *instruction,
    ptx_thread_info *const *lane_threads, unsigned owner_hw_sid,
    unsigned dynamic_warp_id, unsigned expected_warp_uid,
    unsigned warp_id, unsigned active_mask,
    unsigned long long issue_cycle);

extern "C" bool
rtcore_validate_v04_global384_first_submit_live_bind_before_functional(
    const ptx_instruction *instruction,
    ptx_thread_info *const *lane_threads, unsigned owner_hw_sid,
    unsigned warp_uid, unsigned warp_id, unsigned active_mask);

extern "C" bool rtcore_prepare_v04_global384_resident_warp_shell(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned static_inst_uid,
    unsigned *resident_generation);

extern "C" bool rtcore_validate_v04_global384_resident_warp_shell(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned static_inst_uid,
    unsigned resident_generation);

extern "C" bool rtcore_admit_v04_root_node_packet(
    const rtcore::v04::root_node_packet::warp_input_v0 *input,
    unsigned long long issue_cycle, const char **failure_reason);

extern "C" bool rtcore_prepare_v04_root_node_packet_before_functional(
    const ptx_instruction *instruction, ptx_thread_info *const *lane_threads,
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned long long issue_cycle);

extern "C" bool rtcore_finalize_v04_root_node_packet_after_functional(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask);

#endif
