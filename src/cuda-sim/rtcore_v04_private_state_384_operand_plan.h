#ifndef RTCORE_V04_PRIVATE_STATE_384_OPERAND_PLAN_H
#define RTCORE_V04_PRIVATE_STATE_384_OPERAND_PLAN_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_state_384_codec.h"

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace operand_plan {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedPrivateLayout,
  kStatusInvalidConsumer,
  kStatusInvalidOperation,
  kStatusInvalidCompletionReason,
  kStatusInvalidOperationReasonCombination,
  kStatusInvalidProducer,
  kStatusInvalidReservedBits,
  kStatusInvalidDelta,
  kStatusUnauthorizedChunk,
  kStatusConflictingOverlap,
};

enum consumer_kind : uint8_t {
  kConsumerInvalid = 0,
  kConsumerNode,
  kConsumerPrimitive,
  kConsumerInstance,
  kConsumerStack,
  kConsumerCompletionPublisher,
  kConsumerResubmitApply,
};

enum operation_kind : uint8_t {
  kOperationDefault = 0,
  kOperationPrimitiveResume,
  kOperationStackTerminal,
  kOperationStackCrossAsReturn,
  kOperationStackEntries,
};

// Values match the V0.4 compact-result reason encoding.
enum completion_reason_kind : uint8_t {
  kCompletionReasonNone = 0x00,
  kCompletionReasonMiss = 0x01,
  kCompletionReasonClosestHitReady = 0x02,
  kCompletionReasonAnyHitRequired = 0x03,
  kCompletionReasonIntersectionRequired = 0x04,
  kCompletionReasonTraceDoneNoShader = 0x05,
};

enum producer_kind : uint8_t {
  kProducerInvalid = 0,
  kProducerNode,
  kProducerPrimitive,
  kProducerInstanceEnter,
  kProducerStack,
  kProducerStackCrossAsReturn,
  kProducerResubmitApply,
  kProducerCompletionPublisher,
  kProducerStackTransitionSpill,
};

struct read_request_v1 {
  uint32_t private_layout_profile_id;
  uint8_t consumer;
  uint8_t operation;
  uint8_t completion_reason;
  uint8_t reserved_zero;
};

struct chunk_read_v1 {
  uint8_t chunk_index;
  uint8_t reserved_zero0;
  uint16_t slot_byte_offset;
  uint16_t byte_count;
  uint16_t reserved_zero1;
};

struct read_plan_v1 {
  uint32_t private_layout_profile_id;
  uint8_t consumer;
  uint8_t operation;
  uint8_t completion_reason;
  uint8_t read_count;
  chunk_read_v1 reads[kChunkCount];
};

struct write_request_v1 {
  uint32_t private_layout_profile_id;
  uint8_t producer;
  uint8_t reserved_zero[3];
};

struct chunk_delta_v1 {
  uint8_t chunk_index;
  uint8_t reserved_zero[3];
  uint32_t byte_mask;
  uint8_t payload[kChunkBytes];
};

struct unit_sparse_write_plan_v1 {
  uint32_t private_layout_profile_id;
  uint8_t producer;
  uint8_t write_count;
  uint8_t reserved_zero[2];
  chunk_write_v1 writes[kChunkCount];
};

status_kind make_read_plan(const read_request_v1 &request,
                           read_plan_v1 *plan);

status_kind merge_sparse_writes(const write_request_v1 &request,
                                const chunk_delta_v1 *deltas,
                                size_t delta_count,
                                unit_sparse_write_plan_v1 *plan);

const char *status_name(status_kind status);

static_assert(sizeof(read_request_v1) == 8,
              "384B operand-plan read request changed");
static_assert(sizeof(chunk_read_v1) == 8,
              "384B operand-plan chunk read changed");
static_assert(sizeof(write_request_v1) == 8,
              "384B operand-plan write request changed");
static_assert(sizeof(chunk_delta_v1) == 40,
              "384B operand-plan chunk delta changed");

}  // namespace operand_plan
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore

#endif
