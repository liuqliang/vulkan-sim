#ifndef RTCORE_V04_ADDRESS_RANGE_REGISTRY_H
#define RTCORE_V04_ADDRESS_RANGE_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace rtcore {
namespace v04 {
namespace address_range_registry {

static const uint32_t kLaneCapacity = 32;
static const uint32_t kAddressChunkBytes = 32;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOwner,
  kStatusInvalidRange,
  kStatusAddressOverflow,
  kStatusRangeOverlap,
  kStatusDuplicateOwner,
  kStatusRecordNotFound,
  kStatusOwnerMismatch,
  kStatusLaneMismatch,
  kStatusObjectKindMismatch,
  kStatusByteMaskMismatch,
  kStatusWrongPhase,
  kStatusOutstandingTransactions,
  kStatusInvalidResidentGeneration,
  kStatusTransactionIdExhausted,
  kStatusRecordIdExhausted,
  kStatusTransactionNotFound,
  kStatusStaleTransaction,
};

enum object_kind : uint8_t {
  kObjectInvalid = 0,
  kObjectHandoff = 1,
  kObjectContext = 2,
};

enum phase_kind : uint8_t {
  kPhaseInvalid = 0,
  kPhaseProvisional = 1,
  kPhasePendingBind = 2,
  kPhaseLive = 3,
  kPhaseCancelPending = 4,
  kPhaseReleasePending = 5,
};

enum access_kind : uint8_t {
  kAccessInvalid = 0,
  kAccessHandoffShaderTraceInputPublish = 1,
  kAccessHandoffRtcoreAcquire = 2,
  kAccessHandoffRtcorePublish = 3,
  kAccessHandoffShaderDispatchRead = 4,
  kAccessHandoffShaderBuiltinRead = 5,
  kAccessHandoffShaderReturn = 6,
  kAccessContextShaderRead = 7,
  kAccessContextShaderWrite = 8,
};

struct provisional_owner_v0 {
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t dynamic_warp_id;
  uint32_t launch_allocation_generation;
  uint32_t window_generation;
};

struct live_owner_v0 {
  uint32_t owner_hw_sid;
  uint32_t resident_warp_generation;
  uint32_t window_generation;
};

struct range_spec_v0 {
  uint64_t base;
  uint64_t byte_count;
  uint32_t lane_mask;
  object_kind object;
  const uint32_t *allowed_publication_masks;
  size_t allowed_publication_mask_count;
};

struct provisional_store_v0 {
  provisional_owner_v0 owner;
  uint8_t lane_id;
  object_kind object;
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
};

struct provisional_range_observation_v0 {
  provisional_owner_v0 owner;
  live_owner_v0 live_owner;
  uint64_t range_base;
  uint64_t range_byte_count;
  uint8_t lane_id;
  object_kind object;
  phase_kind phase;
  uint8_t reserved_zero;
  uint32_t allowed_publication_mask;
};

struct live_access_v0 {
  live_owner_v0 owner;
  uint8_t lane_id;
  object_kind object;
  access_kind access;
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
};

struct transaction_token_v0 {
  uint64_t transaction_id;
  uint64_t record_id;
  uint32_t owner_hw_sid;
  uint32_t owner_generation;
  uint32_t window_generation;
  uint8_t lane_id;
  object_kind object;
  access_kind access;
  uint8_t reserved_zero;
};

struct registry_snapshot_v0 {
  size_t record_count;
  size_t transaction_count;
  size_t provisional_count;
  size_t pending_bind_count;
  size_t live_count;
  size_t cancel_pending_count;
  size_t release_pending_count;
};

struct live_release_observation_v0 {
  uint64_t outstanding_transactions;
  uint8_t records_present;
  uint8_t release_pending;
  uint8_t released;
  uint8_t reserved_zero[5];
};

class registry_v0 {
 public:
  registry_v0();

  void reset();

  status_kind register_provisional_group(
      const provisional_owner_v0 &owner, uint32_t active_mask,
      const range_spec_v0 *ranges, size_t range_count);

  status_kind accept_provisional_publication_store(
      const provisional_store_v0 &store,
      transaction_token_v0 *token);

  status_kind validate_provisional_publication_store(
      const provisional_store_v0 &store) const;

  status_kind observe_provisional_range(
      uint64_t aligned_32b_address,
      provisional_range_observation_v0 *observation) const;

  status_kind provisional_group_outstanding(
      const provisional_owner_v0 &owner,
      uint64_t *outstanding_transactions) const;

  status_kind begin_live_bind(
      const provisional_owner_v0 &owner, uint32_t active_mask,
      const range_spec_v0 *expected_ranges,
      size_t expected_range_count);

  status_kind commit_live_bind(
      const provisional_owner_v0 &owner,
      uint32_t resident_warp_generation);

  status_kind validate_live_group(
      const live_owner_v0 &owner, uint32_t active_mask,
      const range_spec_v0 *expected_ranges,
      size_t expected_range_count) const;

  status_kind validate_live_access(const live_access_v0 &access) const;

  status_kind accept_live_access(
      const live_access_v0 &access,
      transaction_token_v0 *token);

  status_kind complete_transaction(
      const transaction_token_v0 &token);

  status_kind cancel_provisional_group(
      const provisional_owner_v0 &owner);

  status_kind release_live_group(const live_owner_v0 &owner);

  status_kind poll_live_group_release(
      const live_owner_v0 &owner,
      live_release_observation_v0 *observation) const;

  bool range_available(uint64_t base, uint64_t byte_count) const;

  registry_snapshot_v0 snapshot() const;

 private:
  struct range_record_v0 {
    uint64_t record_id;
    phase_kind phase;
    provisional_owner_v0 provisional_owner;
    live_owner_v0 live_owner;
    uint64_t base;
    uint64_t byte_count;
    uint32_t lane_mask;
    object_kind object;
    uint32_t active_mask;
    uint32_t outstanding_transactions;
    std::vector<uint32_t> allowed_publication_masks;
  };

  struct transaction_record_v0 {
    transaction_token_v0 token;
  };

  static bool same_range_spec(
      const range_record_v0 &record,
      const range_spec_v0 &range);

  uint64_t next_record_id_;
  uint64_t next_transaction_id_;
  std::map<uint64_t, range_record_v0> records_;
  std::map<uint64_t, transaction_record_v0> transactions_;

  registry_v0(const registry_v0 &);
  registry_v0 &operator=(const registry_v0 &);
};

const char *status_name(status_kind status);
const char *phase_name(phase_kind phase);
const char *access_name(access_kind access);

}  // namespace address_range_registry
}  // namespace v04
}  // namespace rtcore

#endif
