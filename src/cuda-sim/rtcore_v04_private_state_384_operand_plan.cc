#include "rtcore_v04_private_state_384_operand_plan.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace private_state_384 {
namespace operand_plan {
namespace {

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static bool is_software_boundary(uint8_t reason) {
  return reason == kCompletionReasonAnyHitRequired ||
         reason == kCompletionReasonIntersectionRequired;
}

static bool is_final_completion(uint8_t reason) {
  return reason == kCompletionReasonMiss ||
         reason == kCompletionReasonClosestHitReady ||
         reason == kCompletionReasonTraceDoneNoShader;
}

static bool is_valid_completion_reason(uint8_t reason) {
  return reason <= kCompletionReasonTraceDoneNoShader;
}

static status_kind validate_read_request(
    const read_request_v1 &request) {
  if (request.private_layout_profile_id != kPrivateLayoutProfileId) {
    return kStatusUnsupportedPrivateLayout;
  }
  if (request.reserved_zero != 0) {
    return kStatusInvalidReservedBits;
  }
  if (request.consumer == kConsumerInvalid ||
      request.consumer > kConsumerResubmitApply) {
    return kStatusInvalidConsumer;
  }
  if (request.operation > kOperationStackCrossAsReturn) {
    return kStatusInvalidOperation;
  }
  if (!is_valid_completion_reason(request.completion_reason)) {
    return kStatusInvalidCompletionReason;
  }

  switch (request.consumer) {
    case kConsumerNode:
    case kConsumerInstance:
      return request.operation == kOperationDefault &&
                     request.completion_reason ==
                         kCompletionReasonNone
                 ? kStatusOk
                 : kStatusInvalidOperationReasonCombination;
    case kConsumerPrimitive:
      if (request.operation == kOperationDefault &&
          request.completion_reason == kCompletionReasonNone) {
        return kStatusOk;
      }
      return request.operation == kOperationPrimitiveResume &&
                     is_software_boundary(request.completion_reason)
                 ? kStatusOk
                 : kStatusInvalidOperationReasonCombination;
    case kConsumerStack:
      if (request.completion_reason != kCompletionReasonNone) {
        return kStatusInvalidOperationReasonCombination;
      }
      return request.operation == kOperationDefault ||
                     request.operation == kOperationStackTerminal ||
                     request.operation == kOperationStackCrossAsReturn
                 ? kStatusOk
                 : kStatusInvalidOperationReasonCombination;
    case kConsumerCompletionPublisher:
      return request.operation == kOperationDefault &&
                     (is_final_completion(request.completion_reason) ||
                      is_software_boundary(request.completion_reason))
                 ? kStatusOk
                 : kStatusInvalidOperationReasonCombination;
    case kConsumerResubmitApply:
      return request.operation == kOperationDefault &&
                     is_software_boundary(request.completion_reason)
                 ? kStatusOk
                 : kStatusInvalidOperationReasonCombination;
    case kConsumerInvalid:
      break;
  }
  return kStatusInvalidConsumer;
}

static void append_read(uint8_t chunk, read_plan_v1 *plan) {
  chunk_read_v1 &read = plan->reads[plan->read_count++];
  read.chunk_index = chunk;
  read.slot_byte_offset =
      static_cast<uint16_t>(chunk * kChunkBytes);
  read.byte_count = kChunkBytes;
}

static bool producer_is_valid(uint8_t producer) {
  return producer > kProducerInvalid &&
         producer <= kProducerCompletionPublisher;
}

static uint32_t producer_allowed_byte_mask(uint8_t producer,
                                           uint8_t chunk) {
  static const uint32_t kFull = 0xffffffffu;
  static const uint32_t kCommittedHitTail = 0x00ffffffu;
  static const uint32_t kCurrentInstanceRef = 0xff000000u;
  static const uint32_t kCurrentInstanceHead = 0x00000fffu;
  static const uint32_t kBlasBuildGeneration = 0x00f00000u;
  static const uint32_t kStackMetadata = 0x0f000000u;
  switch (producer) {
    case kProducerNode:
    case kProducerCompletionPublisher:
      return 0;
    case kProducerPrimitive:
    case kProducerResubmitApply:
      if (chunk == 2 || chunk == 8 || chunk == 9) return kFull;
      return chunk == 3 ? kCommittedHitTail : 0;
    case kProducerInstanceEnter:
      if (chunk == 0 || chunk == 1 ||
          (chunk >= 5 && chunk <= 7) ||
          chunk == 10 || chunk == 11) {
        return kFull;
      }
      if (chunk == 3) return kCurrentInstanceRef;
      if (chunk == 4) {
        return kCurrentInstanceHead | kBlasBuildGeneration |
               kStackMetadata;
      }
      return 0;
    case kProducerStack:
      if (chunk == 4) return kStackMetadata;
      return chunk >= 5 && chunk <= 7 ? kFull : 0;
    case kProducerStackCrossAsReturn:
      if (chunk == 0 || chunk == 1 ||
          (chunk >= 5 && chunk <= 7)) {
        return kFull;
      }
      if (chunk == 3) return kCurrentInstanceRef;
      if (chunk == 4) {
        return kCurrentInstanceHead | kStackMetadata;
      }
      return 0;
    case kProducerInvalid:
      break;
  }
  return 0;
}

}  // namespace

status_kind make_read_plan(const read_request_v1 &request,
                           read_plan_v1 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  const status_kind status = validate_read_request(request);
  if (status != kStatusOk) return status;

  read_plan_v1 result = {};
  result.private_layout_profile_id =
      request.private_layout_profile_id;
  result.consumer = request.consumer;
  result.operation = request.operation;
  result.completion_reason = request.completion_reason;

  switch (request.consumer) {
    case kConsumerNode:
      append_read(0, &result);
      append_read(1, &result);
      append_read(2, &result);
      append_read(4, &result);
      break;
    case kConsumerPrimitive:
      for (uint8_t chunk = 0; chunk <= 4; ++chunk) {
        append_read(chunk, &result);
      }
      if (request.operation == kOperationPrimitiveResume) {
        append_read(8, &result);
        append_read(9, &result);
      }
      break;
    case kConsumerInstance:
      append_read(0, &result);
      append_read(1, &result);
      append_read(4, &result);
      break;
    case kConsumerStack:
      append_read(0, &result);
      append_read(1, &result);
      append_read(2, &result);
      if (request.operation == kOperationStackTerminal) {
        append_read(3, &result);
      }
      append_read(4, &result);
      append_read(5, &result);
      append_read(6, &result);
      append_read(7, &result);
      if (request.operation == kOperationStackCrossAsReturn) {
        append_read(10, &result);
        append_read(11, &result);
      }
      break;
    case kConsumerCompletionPublisher:
      if (is_final_completion(request.completion_reason)) {
        append_read(2, &result);
        append_read(3, &result);
      } else {
        append_read(8, &result);
        append_read(9, &result);
      }
      break;
    case kConsumerResubmitApply:
      append_read(8, &result);
      append_read(9, &result);
      break;
    case kConsumerInvalid:
      return kStatusInvalidConsumer;
  }

  *plan = result;
  return kStatusOk;
}

status_kind merge_sparse_writes(const write_request_v1 &request,
                                const chunk_delta_v1 *deltas,
                                size_t delta_count,
                                unit_sparse_write_plan_v1 *plan) {
  if (plan == NULL) return kStatusInvalidArgument;
  std::memset(plan, 0, sizeof(*plan));
  if (delta_count != 0 && deltas == NULL) {
    return kStatusInvalidArgument;
  }
  if (request.private_layout_profile_id != kPrivateLayoutProfileId) {
    return kStatusUnsupportedPrivateLayout;
  }
  if (!producer_is_valid(request.producer)) {
    return kStatusInvalidProducer;
  }
  if (!bytes_are_zero(request.reserved_zero,
                      sizeof(request.reserved_zero))) {
    return kStatusInvalidReservedBits;
  }

  uint32_t masks[kChunkCount] = {};
  uint8_t payloads[kChunkCount][kChunkBytes] = {};
  for (size_t delta_index = 0; delta_index < delta_count;
       ++delta_index) {
    const chunk_delta_v1 &delta = deltas[delta_index];
    if (!bytes_are_zero(delta.reserved_zero,
                        sizeof(delta.reserved_zero))) {
      return kStatusInvalidReservedBits;
    }
    if (delta.chunk_index >= kChunkCount ||
        delta.byte_mask == 0) {
      return kStatusInvalidDelta;
    }
    const uint32_t allowed_mask =
        producer_allowed_byte_mask(request.producer,
                                   delta.chunk_index);
    if ((delta.byte_mask & ~allowed_mask) != 0) {
      return kStatusUnauthorizedChunk;
    }
    const uint32_t overlap =
        masks[delta.chunk_index] & delta.byte_mask;
    for (uint8_t byte = 0; byte < kChunkBytes; ++byte) {
      const uint32_t bit = uint32_t{1} << byte;
      if ((delta.byte_mask & bit) == 0) continue;
      if ((overlap & bit) != 0 &&
          payloads[delta.chunk_index][byte] !=
              delta.payload[byte]) {
        return kStatusConflictingOverlap;
      }
      payloads[delta.chunk_index][byte] = delta.payload[byte];
    }
    masks[delta.chunk_index] |= delta.byte_mask;
  }

  unit_sparse_write_plan_v1 result = {};
  result.private_layout_profile_id =
      request.private_layout_profile_id;
  result.producer = request.producer;
  for (uint8_t chunk = 0; chunk < kChunkCount; ++chunk) {
    if (masks[chunk] == 0) continue;
    chunk_write_v1 &write = result.writes[result.write_count++];
    write.slot_byte_offset =
        static_cast<uint16_t>(chunk * kChunkBytes);
    write.byte_count = kChunkBytes;
    write.byte_mask = masks[chunk];
    for (uint8_t byte = 0; byte < kChunkBytes; ++byte) {
      if ((masks[chunk] & (uint32_t{1} << byte)) != 0) {
        write.payload[byte] = payloads[chunk][byte];
      }
    }
  }
  *plan = result;
  return kStatusOk;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusUnsupportedPrivateLayout:
      return "unsupported_private_layout";
    case kStatusInvalidConsumer:
      return "invalid_consumer";
    case kStatusInvalidOperation:
      return "invalid_operation";
    case kStatusInvalidCompletionReason:
      return "invalid_completion_reason";
    case kStatusInvalidOperationReasonCombination:
      return "invalid_operation_reason_combination";
    case kStatusInvalidProducer:
      return "invalid_producer";
    case kStatusInvalidReservedBits:
      return "invalid_reserved_bits";
    case kStatusInvalidDelta:
      return "invalid_delta";
    case kStatusUnauthorizedChunk:
      return "unauthorized_chunk";
    case kStatusConflictingOverlap:
      return "conflicting_overlap";
  }
  return "unknown";
}

}  // namespace operand_plan
}  // namespace private_state_384
}  // namespace v04
}  // namespace rtcore
