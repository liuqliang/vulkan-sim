#ifndef RTCORE_V04_PRE_SUBMIT_PUBLICATION_BRIDGE_H
#define RTCORE_V04_PRE_SUBMIT_PUBLICATION_BRIDGE_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

#include "rtcore_v04_address_range_registry.h"
#include "rtcore_v04_allocation_identity_authority.h"

namespace rtcore {
namespace v04 {
namespace pre_submit_publication {

typedef address_range_registry::provisional_store_v0
    publication_store_v0;
typedef address_range_registry::transaction_token_v0
    publication_ticket_v0;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidGeometry,
  kStatusAddressOverflow,
  kStatusAuthorityRejected,
  kStatusRegistryRejected,
  kStatusAuthorityRollbackFailed,
  kStatusGroupNotRegistered,
  kStatusGroupConflict,
};

struct lane_publication_request_v0 {
  allocation_identity::allocation_slot_v0 slot;
  allocation_identity::owner_v0 owner;
  allocation_identity::allocation_ranges_v0 allocation_ranges;
  uint32_t publication_warp_uid;
  uint32_t active_mask;
  uint32_t capacity_lane_slots;
  uint32_t context_lane_stride_bytes;
  uint32_t handoff_lane_stride_bytes;
  const uint32_t *handoff_allowed_publication_masks;
  size_t handoff_allowed_publication_mask_count;
  uint8_t lane_id;
  uint8_t reserved_zero[7];
};

struct lane_publication_result_v0 {
  allocation_identity::status_kind authority_status;
  address_range_registry::status_kind registry_status;
  allocation_identity::allocation_identity_v0 identity;
  uint32_t published_lane_mask;
  uint8_t newly_allocated;
  uint8_t lane_was_new;
  uint8_t publication_complete;
  uint8_t provisional_group_registered;
};

struct bridge_snapshot_v0 {
  allocation_identity::authority_snapshot_v0 authority;
  address_range_registry::registry_snapshot_v0 registry;
  size_t registered_group_count;
};

struct ordinary_store_request_v0 {
  uint32_t owner_hw_sid;
  uint32_t dynamic_warp_id;
  uint32_t warp_id;
  uint32_t active_lane_mask;
  uint32_t data_size_bytes;
  uint32_t sector_count;
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
  uint8_t is_global_write;
  uint8_t byte_mask_single_chunk;
  uint8_t sector_mask_matches_address;
  uint8_t reserved_zero[5];
};

struct ordinary_store_preflight_v0 {
  address_range_registry::status_kind registry_status;
  address_range_registry::provisional_store_v0 store;
  uint8_t candidate;
  uint8_t reserved_zero[7];
};

struct provisional_group_drain_v0 {
  address_range_registry::status_kind registry_status;
  uint64_t preaccept_pending;
  uint64_t outstanding_transactions;
  uint8_t registered;
  uint8_t fence_armed;
  uint8_t wait_required;
  uint8_t fence_consumed;
  uint8_t reserved_zero[4];
};

class bridge_v0 {
 public:
  bridge_v0();

  // Simulator-teardown only: all publication and memory activity must stop.
  void reset();

  status_kind observe_initial_publication(
      const lane_publication_request_v0 &request,
      lane_publication_result_v0 *result);

  status_kind lookup_registered_group_for_submit(
      const allocation_identity::allocation_slot_v0 &slot,
      const allocation_identity::owner_v0 &owner, uint32_t active_mask,
      const allocation_identity::allocation_ranges_v0 &ranges,
      allocation_identity::allocation_identity_v0 *identity,
      address_range_registry::provisional_owner_v0
          *provisional_owner) const;

  address_range_registry::status_kind accept_publication_store(
      const address_range_registry::provisional_store_v0 &store,
      address_range_registry::transaction_token_v0 *token);

  address_range_registry::status_kind complete_publication_store(
      const address_range_registry::transaction_token_v0 &token);

  status_kind begin_publication_store_preaccept(
      const address_range_registry::provisional_store_v0 &store);

  status_kind accept_preaccepted_publication_store(
      const address_range_registry::provisional_store_v0 &store,
      address_range_registry::transaction_token_v0 *token);

  status_kind preflight_ordinary_publication_store(
      const ordinary_store_request_v0 &request,
      ordinary_store_preflight_v0 *preflight) const;

  status_kind service_provisional_publication_fence(
      uint32_t owner_hw_sid, uint32_t dynamic_warp_id,
      uint32_t warp_id, provisional_group_drain_v0 *drain);

  bridge_snapshot_v0 snapshot() const;

 private:
  struct registered_group_v0 {
    address_range_registry::provisional_owner_v0 provisional_owner;
    allocation_identity::owner_v0 execution_owner;
    uint64_t preaccept_pending;
    uint8_t fence_armed;
  };

  struct publication_identity_observation_v0 {
    uint64_t record_id;
    uint32_t publication_warp_uid;
  };

  allocation_identity::authority_v0 authority_;
  address_range_registry::registry_v0 registry_;
  std::map<std::pair<uint64_t, uint32_t>,
           publication_identity_observation_v0>
      publication_identity_observations_;
  std::map<uint64_t, registered_group_v0> registered_groups_;

  bridge_v0(const bridge_v0 &);
  bridge_v0 &operator=(const bridge_v0 &);
};

const char *status_name(status_kind status);

bridge_v0 &shared_bridge();

}  // namespace pre_submit_publication
}  // namespace v04
}  // namespace rtcore

#endif
