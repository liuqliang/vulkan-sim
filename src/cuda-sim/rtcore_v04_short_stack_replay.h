#ifndef RTCORE_V04_SHORT_STACK_REPLAY_H
#define RTCORE_V04_SHORT_STACK_REPLAY_H

#include <cstdint>

#include "rtcore_v04_typed_node_kernel.h"

namespace rtcore {
namespace v04 {
namespace short_stack {

static const uint8_t kLogicalCapacity = 6;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidState,
  kStatusInvalidEntry,
  kStatusInvalidRoute,
  kStatusInvalidParentEdge,
  kStatusCrossAsInvariant,
};

enum entry_kind : uint8_t {
  kEntryDirectTarget = 0,
  kEntrySameNodeReplay = 1,
  kEntryParentResume = 2,
  kEntryCrossAsReturn = 3,
};

enum domain_kind : uint8_t {
  kDomainTlas = 0,
  kDomainBlas = 1,
};

struct entry_v0 {
  uint64_t payload_offset;
  uint32_t near_t_bits;
  uint16_t payload_byte_count;
  uint8_t payload_kind;
  uint8_t control;
};

struct state_v0 {
  entry_v0 entries[kLogicalCapacity];
  uint8_t stack_count;
  uint8_t stack_top_ptr;
  uint8_t cross_as;
  uint8_t lost;
  uint8_t active_domain;
  uint8_t reserved_zero[3];
};

struct parent_edge_v0 {
  uint64_t parent_payload_offset;
  uint8_t parent_child_slot;
  uint8_t root;
  uint8_t reserved_zero[6];
};

struct route_push_input_v0 {
  state_v0 state;
  typed_node::route_result_v0 route;
  uint64_t current_node_payload_offset;
  entry_v0 parent_resume;
  uint8_t active_domain;
  uint8_t parent_resume_valid;
  uint8_t reserved_zero[6];
};

struct route_push_result_v0 {
  uint8_t status;
  uint8_t selected_valid;
  uint8_t compressed_to_replay;
  uint8_t overflowed_bottom;
  uint8_t parent_resume_appended;
  uint8_t parent_resume_deferred;
  uint8_t reserved_zero[2];
  state_v0 state;
  typed_node::selected_child_fetch_work_item_v0 selected;
};

struct pop_result_v0 {
  uint8_t status;
  uint8_t entry_valid;
  uint8_t reserved_zero[6];
  state_v0 state;
  entry_v0 entry;
};

struct cross_as_input_v0 {
  state_v0 state;
  entry_v0 tlas_instance;
  entry_v0 blas_root;
};

struct cross_as_result_v0 {
  uint8_t status;
  uint8_t reserved_zero[7];
  state_v0 state;
};

struct parent_bailout_result_v0 {
  uint8_t status;
  uint8_t parent_resume_valid;
  uint8_t root_replay;
  uint8_t reserved_zero[5];
  state_v0 state;
  entry_v0 parent_resume;
};

uint8_t make_control(entry_kind kind, domain_kind domain,
                     uint8_t child_anchor, bool anchor_valid,
                     bool inclusive);
entry_kind control_kind(uint8_t control);
domain_kind control_domain(uint8_t control);
uint8_t control_child_anchor(uint8_t control);
bool control_anchor_valid(uint8_t control);
bool control_inclusive(uint8_t control);

bool validate_state(const state_v0 &state);
bool validate_drained_recovery_state(const state_v0 &state);
bool validate_entry(const entry_v0 &entry);
bool read_logical_entry(const state_v0 &state, uint8_t logical_index,
                        entry_v0 *entry);

route_push_result_v0 push_node_route(const route_push_input_v0 &input);
pop_result_v0 pop_top(const state_v0 &state);

bool parent_bailout_required(const state_v0 &state);
parent_bailout_result_v0 prepare_parent_bailout(
    const pop_result_v0 &popped, const parent_edge_v0 &parent);

cross_as_result_v0 enter_blas(const cross_as_input_v0 &input);
status_kind return_to_tlas(state_v0 *state,
                           const parent_edge_v0 &instance_parent);

bool terminal_allowed(const state_v0 &state, bool pending_candidate,
                      bool pending_memory, bool pending_commit);

const char *status_name(status_kind status);

}  // namespace short_stack
}  // namespace v04
}  // namespace rtcore

#endif
