#ifndef RTCORE_V04_PRIVATE_SHARED_BACKING_H
#define RTCORE_V04_PRIVATE_SHARED_BACKING_H

#include <cstdint>
#include <deque>

#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_private_state_384_codec.h"

namespace rtcore {
namespace v04 {
namespace private_shared {

static const uint32_t kResidentSharedBytes = 96 * 1024;
static const uint32_t kPrivateBytesPerLane =
    private_frontier::kPrivateDataSlotBytes;
static const uint32_t kHandoffChargeBytesPerLane = 128;
static const uint32_t kLegacyResidentChargeBytesPerLane =
    kPrivateBytesPerLane + kHandoffChargeBytesPerLane;
static const uint32_t kCompressedResidentChargeBytesPerLane = 384;
static const uint32_t kResidentChargeBytesPerLane =
    kLegacyResidentChargeBytesPerLane;
static const uint32_t kResidentWarpCapacity = 8;
static const uint32_t kLaneCapacity = 32;
static const uint32_t kSharedQueueCapacity = 32;
static const uint32_t kOutstandingCapacity = 32;
static const uint32_t kPrivateWriteEnqueueWidth = 4;
static const uint32_t kResponseFillWidth = 4;
static const uint32_t kInitMemoryOpSequence = 1;

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidOwner,
  kStatusDuplicateWarp,
  kStatusCapacityExceeded,
  kStatusOwnerAlias,
  kStatusStalePlan,
  kStatusOwnerMismatch,
  kStatusQueueFull,
  kStatusOutstandingFull,
  kStatusOfferMismatch,
  kStatusAckMismatch,
  kStatusDuplicateAck,
  kStatusInFlightRelease,
  kStatusPlannerFailure,
  kStatusDuplicateOperation,
  kStatusNoAckReady,
};

enum resident_charge_profile_kind : uint8_t {
  kResidentChargeProfileInvalid = 0,
  kResidentChargeProfileLegacyShared832 = 1,
  kResidentChargeProfileCompressedShared384 = 2,
};

enum address_space_kind : uint8_t {
  kAddressSpaceInvalid = 0,
  kAddressSpaceShared = 1,
  kAddressSpaceGlobal = 2,
};

enum address_mode_kind : uint8_t {
  kAddressModeInvalid = 0,
  kAddressModePrivateField = 1,
};

enum access_operation_kind : uint8_t {
  kAccessOperationInvalid = 0,
  kAccessOperationWrite = 1,
};

enum destination_kind : uint8_t {
  kDestinationInvalid = 0,
  kDestinationPrivateCommitAck = 1,
};

struct shared_write_v0 {
  bool valid;
  uint8_t address_space;
  uint8_t address_mode;
  uint8_t access_operation;
  uint8_t destination;
  uint8_t reserved_zero[3];
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint32_t memory_op_seq;
  uint8_t chunk_id;
  uint8_t chunk_count;
  uint8_t field_kind;
  uint8_t reserved_zero1;
  uint64_t aligned_32b_address;
  uint32_t byte_mask;
  uint8_t payload[private_frontier::kSharedAccessChunkBytes];
  uint64_t enqueue_cycle;
  uint64_t accepted_cycle;
  uint64_t ack_cycle;
};

struct runtime_write_ack_v0 {
  bool valid;
  uint8_t field_kind;
  uint16_t memory_operation_seq;
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint32_t commit_epoch;
  uint8_t reserved_zero[8];
};

struct lane_slot_state_v0 {
  bool live;
  private_frontier::owner_binding_v0 owner;
  private_frontier::shadow_slot_v0 canonical_slot;
  private_frontier::shadow_slot_v0 staging_slot;
  private_frontier::access_plan_v0 init_plan;
  uint32_t enqueued_chunk_mask;
  uint32_t acknowledged_chunk_mask;
  uint8_t next_chunk_to_enqueue;
  uint8_t reserved_zero[3];
};

struct resident_warp_state_v0 {
  bool live;
  bool scheduler_ready;
  uint8_t resident_warp_slot;
  uint8_t resident_charge_profile;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
  uint32_t charge_bytes_per_lane;
  uint32_t charged_bytes;
  uint32_t enqueued_chunk_count;
  uint32_t accepted_chunk_count;
  uint32_t acknowledged_chunk_count;
  lane_slot_state_v0 lanes[kLaneCapacity];
};

struct backing_state_v0 {
  bool initialized;
  bool offer_valid;
  uint8_t fault_status;
  uint8_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t charged_bytes;
  uint64_t mutation_epoch;
  uint64_t enqueue_count;
  uint64_t offer_count;
  uint64_t accept_count;
  uint64_t acknowledgment_count;
  uint64_t shared_queue_capacity_blocked_count;
  uint64_t outstanding_capacity_blocked_count;
  uint32_t max_shared_queue_depth;
  uint32_t max_outstanding_depth;
  resident_warp_state_v0 resident_warps[kResidentWarpCapacity];
  std::deque<shared_write_v0> shared_queue;
  std::deque<shared_write_v0> outstanding;
  std::deque<uint8_t> ready_commit_slots;
  shared_write_v0 offered;
};

struct new_warp_plan_v0 {
  bool valid;
  bool root_operands_initialized;
  bool short_stack_initialized;
  bool compressed_launch_initialized;
  uint8_t resident_warp_slot;
  uint8_t resident_charge_profile;
  uint8_t reserved_zero[2];
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
  uint32_t charge_bytes_per_lane;
  uint32_t charged_bytes;
  uint64_t expected_mutation_epoch;
  private_frontier::shadow_slot_v0 staging_slots[kLaneCapacity];
  private_frontier::access_plan_v0 init_plans[kLaneCapacity];
};

struct mask_shrink_plan_v0 {
  bool valid;
  uint8_t resident_warp_slot;
  uint16_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t previous_warp_uid;
  uint32_t next_warp_uid;
  uint32_t warp_id;
  uint32_t previous_active_mask;
  uint32_t next_active_mask;
  uint32_t release_mask;
  uint32_t released_bytes;
  uint64_t expected_mutation_epoch;
};

struct release_warp_plan_v0 {
  bool valid;
  uint8_t resident_warp_slot;
  uint16_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
  uint32_t released_bytes;
  uint64_t expected_mutation_epoch;
};

struct ready_commit_v0 {
  bool valid;
  uint8_t resident_warp_slot;
  uint16_t reserved_zero;
  uint32_t owner_hw_sid;
  uint32_t warp_uid;
  uint32_t warp_id;
  uint32_t active_mask;
};

void initialize(backing_state_v0 *state, uint32_t owner_hw_sid);

status_kind prepare_new_warp(
    const backing_state_v0 &state, uint32_t warp_uid, uint32_t warp_id,
    uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    new_warp_plan_v0 *plan);

status_kind prepare_new_warp_with_root_operands(
    const backing_state_v0 &state, uint32_t warp_uid, uint32_t warp_id,
    uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    const private_frontier::root_private_operands_v0
        root_operands[kLaneCapacity],
    new_warp_plan_v0 *plan);

status_kind prepare_new_warp_with_root_operands_and_short_stack(
    const backing_state_v0 &state, uint32_t warp_uid, uint32_t warp_id,
    uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    const private_frontier::root_private_operands_v0
        root_operands[kLaneCapacity],
    const uint32_t root_build_generations[kLaneCapacity],
    new_warp_plan_v0 *plan);

status_kind prepare_new_warp_with_compressed_launch(
    const backing_state_v0 &state, uint32_t warp_uid, uint32_t warp_id,
    uint32_t active_mask,
    const private_frontier::owner_binding_v0 owners[kLaneCapacity],
    const private_state_384::sparse_write_plan_v1
        launch_plans[kLaneCapacity],
    new_warp_plan_v0 *plan);

status_kind commit_new_warp(backing_state_v0 *state,
                            const new_warp_plan_v0 &plan);

uint32_t service_init_enqueue(backing_state_v0 *state, uint64_t service_cycle,
                              uint32_t enqueue_budget);

bool pop_shared_offer(backing_state_v0 *state, uint64_t service_cycle,
                      shared_write_v0 *operation);

status_kind accept_shared_offer(backing_state_v0 *state,
                                const shared_write_v0 &operation,
                                uint64_t accepted_cycle);

status_kind validate_runtime_write(const backing_state_v0 &state,
                                   const shared_write_v0 &operation);

status_kind enqueue_runtime_write(backing_state_v0 *state,
                                  const shared_write_v0 &operation);

uint32_t service_write_acks(backing_state_v0 *state, uint64_t service_cycle,
                            uint32_t response_budget);

bool pop_ready_commit(backing_state_v0 *state, ready_commit_v0 *commit);

status_kind prepare_mask_shrink(const backing_state_v0 &state,
                                uint8_t resident_warp_slot,
                                uint32_t previous_warp_uid,
                                uint32_t next_warp_uid, uint32_t warp_id,
                                uint32_t next_active_mask,
                                mask_shrink_plan_v0 *plan);

status_kind commit_mask_shrink(backing_state_v0 *state,
                               const mask_shrink_plan_v0 &plan);

status_kind prepare_release_warp(const backing_state_v0 &state,
                                 uint8_t resident_warp_slot, uint32_t warp_uid,
                                 uint32_t warp_id, release_warp_plan_v0 *plan);

status_kind commit_release_warp(backing_state_v0 *state,
                                const release_warp_plan_v0 &plan);

const lane_slot_state_v0 *find_live_lane(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner);

status_kind prepare_root_operand_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan);

status_kind prepare_primitive_operand_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan);

status_kind prepare_stack_selected_fetch_spill_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan);

status_kind prepare_frontier_metadata_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan);

status_kind prepare_nonempty_pop_operand_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::frontier_metadata_image_v0 &returned_metadata,
    private_frontier::access_plan_v0 *read_plan);

status_kind prepare_empty_pop_operand_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::frontier_metadata_image_v0 &returned_metadata,
    private_frontier::access_plan_v0 *read_plan);

status_kind prepare_parent_frame_read_plan(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    private_frontier::access_plan_v0 *read_plan);

status_kind read_canonical_chunk(
    const backing_state_v0 &state,
    const private_frontier::owner_binding_v0 &owner,
    const private_frontier::shared_chunk_access_v0 &access,
    uint8_t payload[private_frontier::kSharedAccessChunkBytes]);

const char *status_name(status_kind status);

}  // namespace private_shared
}  // namespace v04
}  // namespace rtcore

#endif
