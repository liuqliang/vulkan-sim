#ifndef RTCORE_V04_ROOT_NODE_PACKET_H
#define RTCORE_V04_ROOT_NODE_PACKET_H

#include <cstdint>

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
  fetch_target::target_reference_v0 target_reference;
  private_frontier::root_private_operands_v0 private_operands;
  typed_node::ray_policy_v0 ray_policy;
  uint64_t raw_payload_base_address;
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

}  // namespace root_node_packet
}  // namespace v04
}  // namespace rtcore

extern "C" bool rtcore_v04_root_node_ready_packet_gate_active();
extern "C" bool rtcore_v04_functional_node_driver_gate_active();
extern "C" bool rtcore_v04_live_node_timing_gate_active();
extern "C" bool rtcore_v04_root_node_input_gate_active();
extern "C" bool rtcore_v04_functional_node_driver_configuration_valid();

extern "C" bool rtcore_admit_v04_root_node_packet(
    const rtcore::v04::root_node_packet::warp_input_v0 *input,
    unsigned long long issue_cycle, const char **failure_reason);

extern "C" bool rtcore_prepare_v04_root_node_packet_before_functional(
    const ptx_instruction *instruction, ptx_thread_info *const *lane_threads,
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned long long issue_cycle);

#endif
