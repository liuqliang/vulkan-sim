#include "rtcore_v04_typed_stack_result_stream.h"

#include <cstring>

namespace rtcore {
namespace v04 {
namespace typed_stack_stream {
namespace {

static bool bytes_are_zero(const uint8_t *bytes, size_t size) {
  for (size_t index = 0; index < size; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static bool owner_is_zero(
    const private_frontier::owner_binding_v0 &owner) {
  return owner.owner_hw_sid == 0 && owner.resident_warp_id == 0 &&
         owner.request_identity == 0 && owner.generation == 0 &&
         owner.private_slot_id == 0 && owner.lane_id == 0 &&
         bytes_are_zero(owner.reserved_zero, sizeof(owner.reserved_zero));
}

static bool owner_is_valid(
    const private_frontier::owner_binding_v0 &owner) {
  return owner.request_identity != 0 && owner.generation != 0 &&
         owner.lane_id < 32 && owner.private_slot_id < 256 &&
         bytes_are_zero(owner.reserved_zero,
                        sizeof(owner.reserved_zero));
}

static bool result_is_valid(
    uint64_t producer_node_reference,
    const typed_stack::push_result_v0 &result) {
  const typed_blas::as_decode_context_v0 &context =
      result.selected_fetch.decode_context;
  return typed_stack::validate_push_result(result) &&
         (producer_node_reference & uint64_t{0x3f}) == 0 &&
         producer_node_reference <= context.device_range_bytes &&
         uint64_t{64} <=
             context.device_range_bytes - producer_node_reference;
}

static status_kind validate_record(
    const push_result_record_v0 &record, uint32_t expected_sequence,
    bool expect_bound,
    const private_frontier::owner_binding_v0 *expected_owner) {
  if (!record.valid) return kStatusInvalidArgument;
  if (record.reserved_zero != 0 ||
      record.operation_seq != expected_sequence ||
      record.operation_seq == 0) {
    return kStatusInvalidSequence;
  }
  if (record.operation_kind !=
          typed_stack::kPushRemainderAndForwardSelected ||
      record.result_kind != typed_stack::kStackPushedAndSelected ||
      record.payload_bytes != kPushResultPayloadBytes ||
      record.payload.result_kind != record.result_kind ||
      !result_is_valid(record.producer_node_reference, record.payload)) {
    return kStatusInvalidResult;
  }
  if (record.owner_bound != expect_bound) {
    return expect_bound ? kStatusInvalidOwner : kStatusAlreadyBound;
  }
  if (!expect_bound) {
    return owner_is_zero(record.owner) ? kStatusOk
                                       : kStatusOwnerMismatch;
  }
  if (expected_owner == NULL || !owner_is_valid(record.owner) ||
      !private_frontier::owners_equal(record.owner, *expected_owner)) {
    return kStatusOwnerMismatch;
  }
  return kStatusOk;
}

}  // namespace

status_kind append_push_result(
    std::vector<push_result_record_v0> *records,
    uint64_t producer_node_reference,
    const typed_stack::push_result_v0 &result) {
  if (records == NULL) return kStatusInvalidArgument;
  if (records->size() >= kMaxPushResultsPerLane) {
    return kStatusCapacityExceeded;
  }
  if (!result_is_valid(producer_node_reference, result)) {
    return kStatusInvalidResult;
  }

  push_result_record_v0 record = {};
  record.valid = true;
  record.operation_seq = static_cast<uint32_t>(records->size()) + 1;
  record.producer_node_reference = producer_node_reference;
  record.operation_kind =
      typed_stack::kPushRemainderAndForwardSelected;
  record.result_kind = typed_stack::kStackPushedAndSelected;
  record.payload_bytes = kPushResultPayloadBytes;
  record.payload = result;
  records->push_back(record);
  return kStatusOk;
}

status_kind validate_unbound(
    const std::vector<push_result_record_v0> &records) {
  if (records.size() > kMaxPushResultsPerLane) {
    return kStatusCapacityExceeded;
  }
  for (size_t index = 0; index < records.size(); ++index) {
    const status_kind status =
        validate_record(records[index],
                        static_cast<uint32_t>(index) + 1,
                        false, NULL);
    if (status != kStatusOk) return status;
  }
  return kStatusOk;
}

status_kind validate_bound(
    const std::vector<push_result_record_v0> &records,
    const private_frontier::owner_binding_v0 &owner) {
  if (!owner_is_valid(owner)) return kStatusInvalidOwner;
  if (records.size() > kMaxPushResultsPerLane) {
    return kStatusCapacityExceeded;
  }
  for (size_t index = 0; index < records.size(); ++index) {
    const status_kind status =
        validate_record(records[index],
                        static_cast<uint32_t>(index) + 1,
                        true, &owner);
    if (status != kStatusOk) return status;
  }
  return kStatusOk;
}

status_kind bind_owner(
    std::vector<push_result_record_v0> *records,
    const private_frontier::owner_binding_v0 &owner) {
  if (records == NULL || !owner_is_valid(owner)) {
    return kStatusInvalidOwner;
  }
  const status_kind validation = validate_unbound(*records);
  if (validation != kStatusOk) return validation;

  for (size_t index = 0; index < records->size(); ++index) {
    (*records)[index].owner = owner;
    (*records)[index].owner_bound = true;
  }
  return validate_bound(*records, owner);
}

bool streams_equal(
    const std::vector<push_result_record_v0> &lhs,
    const std::vector<push_result_record_v0> &rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index].valid != rhs[index].valid ||
        lhs[index].owner_bound != rhs[index].owner_bound ||
        lhs[index].reserved_zero != rhs[index].reserved_zero ||
        lhs[index].operation_seq != rhs[index].operation_seq ||
        lhs[index].producer_node_reference !=
            rhs[index].producer_node_reference ||
        lhs[index].operation_kind != rhs[index].operation_kind ||
        lhs[index].result_kind != rhs[index].result_kind ||
        lhs[index].payload_bytes != rhs[index].payload_bytes ||
        !private_frontier::owners_equal(lhs[index].owner,
                                        rhs[index].owner) ||
        std::memcmp(&lhs[index].payload, &rhs[index].payload,
                    sizeof(lhs[index].payload)) != 0) {
      return false;
    }
  }
  return true;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusCapacityExceeded:
      return "capacity_exceeded";
    case kStatusInvalidSequence:
      return "invalid_sequence";
    case kStatusInvalidResult:
      return "invalid_result";
    case kStatusInvalidOwner:
      return "invalid_owner";
    case kStatusAlreadyBound:
      return "already_bound";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
  }
  return "unknown";
}

}  // namespace typed_stack_stream
}  // namespace v04
}  // namespace rtcore
