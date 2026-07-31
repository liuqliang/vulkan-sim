#ifndef RTCORE_V04_ALLOCATION_IDENTITY_AUTHORITY_H
#define RTCORE_V04_ALLOCATION_IDENTITY_AUTHORITY_H

#include <cstddef>
#include <cstdint>
#include <map>

namespace rtcore {
namespace v04 {
namespace allocation_identity {

static const uint32_t kLaneCapacity = 32;
static const uint64_t kAddressAlignment = 32;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidRange,
  kStatusAddressOverflow,
  kStatusRangeOverlap,
  kStatusAllocationBusy,
  kStatusRecordNotFound,
  kStatusOwnerMismatch,
  kStatusActiveMaskMismatch,
  kStatusLaneNotActive,
  kStatusPublicationIncomplete,
  kStatusIdentityMismatch,
  kStatusWrongPhase,
  kStatusGenerationExhausted,
  kStatusRecordIdExhausted,
};

enum phase_kind : uint8_t {
  kPhaseInvalid = 0,
  kPhaseAllocated = 1,
  kPhaseReleasePending = 2,
};

struct owner_v0 {
  uint32_t owner_hw_sid;
  uint32_t dynamic_warp_id;
  uint32_t warp_id;
};

struct allocation_slot_v0 {
  uint64_t allocation_domain_id;
  uint32_t allocation_slot_id;
  uint32_t reserved_zero;
};

struct allocation_ranges_v0 {
  uint64_t context_base;
  uint64_t context_byte_count;
  uint64_t handoff_base;
  uint64_t handoff_byte_count;
};

struct publication_request_v0 {
  allocation_slot_v0 slot;
  owner_v0 owner;
  allocation_ranges_v0 ranges;
  uint32_t active_mask;
  uint8_t lane_id;
  uint8_t reserved_zero[3];
};

struct allocation_identity_v0 {
  uint64_t record_id;
  allocation_slot_v0 slot;
  owner_v0 owner;
  allocation_ranges_v0 ranges;
  uint32_t active_mask;
  uint32_t launch_allocation_generation;
  uint32_t window_generation;
};

struct publication_observation_v0 {
  allocation_identity_v0 identity;
  uint32_t published_lane_mask;
  uint8_t newly_allocated;
  uint8_t lane_was_new;
  uint8_t publication_complete;
  uint8_t reserved_zero;
};

struct authority_snapshot_v0 {
  size_t active_record_count;
  size_t generation_history_count;
  size_t allocated_count;
  size_t release_pending_count;
};

#ifdef RTCORE_V04_ALLOCATION_IDENTITY_AUTHORITY_TESTING
struct authority_test_access_v0;
#endif

class authority_v0 {
 public:
  authority_v0();

  // Simulator-teardown only: all request and memory activity must be stopped.
  // Drops live allocations while preserving generation and record histories.
  void reset();

  status_kind observe_publication(
      const publication_request_v0 &request,
      publication_observation_v0 *observation);

  status_kind lookup_for_submit(
      const allocation_slot_v0 &slot, const owner_v0 &owner,
      uint32_t active_mask,
      const allocation_ranges_v0 &ranges,
      allocation_identity_v0 *identity) const;

  status_kind begin_release(
      const allocation_identity_v0 &identity);

  status_kind commit_release(
      const allocation_identity_v0 &identity);

  bool ranges_available(const allocation_ranges_v0 &ranges) const;

  authority_snapshot_v0 snapshot() const;

 private:
  struct allocation_key_v0 {
    uint64_t allocation_domain_id;
    uint32_t allocation_slot_id;

    bool operator<(const allocation_key_v0 &other) const;
  };

  struct generation_history_v0 {
    uint32_t last_launch_allocation_generation;
    uint32_t last_window_generation;
  };

  struct active_record_v0 {
    phase_kind phase;
    allocation_identity_v0 identity;
    uint32_t published_lane_mask;
  };

  uint64_t next_record_id_;
  std::map<allocation_key_v0, generation_history_v0> histories_;
  std::map<allocation_key_v0, active_record_v0> records_;

  authority_v0(const authority_v0 &);
  authority_v0 &operator=(const authority_v0 &);

#ifdef RTCORE_V04_ALLOCATION_IDENTITY_AUTHORITY_TESTING
  friend struct authority_test_access_v0;
#endif
};

const char *status_name(status_kind status);
const char *phase_name(phase_kind phase);

}  // namespace allocation_identity
}  // namespace v04
}  // namespace rtcore

#endif
