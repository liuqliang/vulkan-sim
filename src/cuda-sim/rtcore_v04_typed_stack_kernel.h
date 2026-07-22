#ifndef RTCORE_V04_TYPED_STACK_KERNEL_H
#define RTCORE_V04_TYPED_STACK_KERNEL_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_typed_node_kernel.h"

namespace rtcore {
namespace v04 {
namespace typed_stack {

static const uint32_t kGenRtDerivedProfileId =
    typed_node::kGenRtDerivedProfileId;
static const unsigned kMaxRemainderChildren =
    typed_node::kMaxChildren - 1;

enum operation_kind : uint8_t {
  kOperationInvalid = 0,
  kPushRemainderAndForwardSelected = 1,
};

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedProfile,
  kStatusInvalidFrontierMetadata,
  kStatusInvalidRoutePacket,
  kStatusInvalidTraversalBound,
  kStatusInvalidWorkItem,
  kStatusInvalidRemainderOrder,
  kStatusFrontierCapacityExceeded,
};

enum result_kind : uint8_t {
  kResultInvalid = 0,
  kStackPushedAndSelected = 1,
};

enum output_valid_bit : uint8_t {
  kFrontierDeltaValid = 1u << 0,
  kSelectedFetchValid = 1u << 1,
};

enum frontier_action : uint8_t {
  kFrontierActionNone = 0,
  kFrontierActionAppendChildren = 1,
};

struct frontier_metadata_v0 {
  uint32_t frontier_top;
  uint32_t frontier_count;
  uint32_t frontier_capacity;
};

struct alignas(16) push_input_v0 {
  uint32_t profile_id;
  uint8_t operation_kind;
  uint8_t reserved_zero0[3];
  frontier_metadata_v0 frontier;
  uint32_t current_traversal_bound_bits;
  uint8_t reserved_zero1[8];
  typed_node::route_result_v0 node_route;
};

struct frontier_append_delta_v0 {
  uint8_t action;
  uint8_t write_count;
  uint16_t reserved_zero;
  uint32_t append_base_index;
  uint32_t new_frontier_top;
  uint32_t new_frontier_count;
  typed_node::compact_child_work_item_v0
      written_items[kMaxRemainderChildren];
};

struct alignas(16) push_result_v0 {
  uint8_t status;
  uint8_t result_kind;
  uint8_t output_valid_mask;
  uint8_t pruned_count;
  uint8_t reserved_zero0[12];
  frontier_append_delta_v0 frontier_delta;
  typed_node::selected_child_fetch_work_item_v0 selected_fetch;
  uint8_t reserved_zero_tail[8];
};

static_assert(sizeof(frontier_metadata_v0) == 12,
              "frontier metadata must remain 12 bytes");
static_assert(sizeof(push_input_v0) == 192,
              "typed Stack push input must remain 192 bytes");
static_assert(alignof(push_input_v0) == 16,
              "typed Stack push input must remain aligned");
static_assert(offsetof(push_input_v0, node_route) == 32,
              "typed Stack Node route offset changed");
static_assert(sizeof(frontier_append_delta_v0) == 96,
              "frontier append delta must remain 96 bytes");
static_assert(offsetof(frontier_append_delta_v0, written_items) == 16,
              "frontier append item offset changed");
static_assert(sizeof(push_result_v0) == 176,
              "typed Stack push result must remain 176 bytes");
static_assert(alignof(push_result_v0) == 16,
              "typed Stack push result must remain aligned");
static_assert(offsetof(push_result_v0, frontier_delta) == 16,
              "typed Stack delta offset changed");
static_assert(offsetof(push_result_v0, selected_fetch) == 112,
              "typed Stack selected fetch offset changed");
static_assert(offsetof(push_result_v0, reserved_zero_tail) == 168,
              "typed Stack result tail offset changed");

push_result_v0 execute_push(const push_input_v0 &input);

const char *status_name(status_kind status);

}  // namespace typed_stack
}  // namespace v04
}  // namespace rtcore

#endif
