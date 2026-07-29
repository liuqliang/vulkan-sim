#include "rtcore_v04_short_stack_timing_service.h"

#include <cstring>
#include <limits>

#include "rtcore_v04_private_shared_backing_internal.h"

namespace rtcore {
namespace v04 {
namespace short_stack_timing {
namespace {

static const uint8_t kAllReadChunks =
    static_cast<uint8_t>((1u << kReadChunkCount) - 1u);
static const uint8_t kAllWriteChunks =
    static_cast<uint8_t>((1u << kWriteChunkCount) - 1u);
static const uint8_t kReadPhaseShortStackState = 4;
static const uint8_t kOperationNodeTransition = 1;
static const unsigned kResponseTargetRtcore = 1;

bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

uint64_t private_slot_base(
    const private_frontier::owner_binding_v0 &owner) {
  return UINT64_C(0xff00000000000000) +
         static_cast<uint64_t>(owner.owner_hw_sid) * UINT64_C(0x1000000) +
         static_cast<uint64_t>(owner.private_slot_id) *
             private_frontier::kPrivateDataSlotBytes;
}

bool make_request_owner(
    const private_frontier::owner_binding_v0 &private_owner,
    request_owner::lane_binding_v0 *request_binding) {
  if (request_binding == NULL || private_owner.request_identity == 0 ||
      private_owner.generation == 0 ||
      private_owner.generation > request_owner::kRequestGenerationMax ||
      private_owner.resident_warp_id >=
          request_owner::kResidentWarpCapacity ||
      private_owner.private_slot_id >
          std::numeric_limits<uint16_t>::max() ||
      private_owner.lane_id >= request_owner::kLaneCapacity ||
      !bytes_are_zero(private_owner.reserved_zero,
                      sizeof(private_owner.reserved_zero))) {
    return false;
  }
  request_owner::internal_request_key_fields_v0 fields = {};
  if (request_owner::unpack_internal_request_key(
          private_owner.request_identity, &fields) !=
          request_owner::kStatusOk ||
      fields.resident_warp_slot != private_owner.resident_warp_id ||
      fields.lane_id != private_owner.lane_id ||
      fields.request_generation != private_owner.generation) {
    return false;
  }
  *request_binding = request_owner::lane_binding_v0();
  request_binding->packed_request_key =
      private_owner.request_identity;
  request_binding->owner_hw_sid = private_owner.owner_hw_sid;
  request_binding->request_control_slot =
      fields.request_control_slot;
  request_binding->request_generation =
      fields.request_generation;
  request_binding->private_slot_id =
      static_cast<uint16_t>(private_owner.private_slot_id);
  request_binding->resident_warp_slot =
      fields.resident_warp_slot;
  request_binding->lane_id = fields.lane_id;
  return private_frontier::owners_equal(
      request_owner::make_private_frontier_owner(*request_binding),
      private_owner);
}

bool valid_config(const config_v0 &config) {
  return config.capacity != 0 && config.capacity <= kMaxSlots &&
         config.reservation_width != 0 &&
         config.reservation_width <= config.capacity &&
         config.unit_count != 0 && config.unit_count <= kMaxUnits &&
         config.stack_latency != 0 &&
         config.initiation_interval != 0 &&
         config.issue_width != 0 &&
         config.issue_width <= config.unit_count &&
         config.parent_lookup_latency != 0 &&
         config.reserved_zero == 0;
}

bool request_matches_entry(
    const rtcore_memory_unit_request_snapshot &request,
    const operation_entry_v0 &entry) {
  const rtcore_v04_stack_private_read_transport_snapshot &transport =
      request.v04_stack_private_read;
  return entry.valid != 0 && entry.phase == kPhaseReading &&
         request.valid &&
         request.address_space == RTCORE_MEMORY_ADDRESS_SPACE_SHARED &&
         request.operation == RTCORE_MEMORY_OPERATION_READ &&
         request.destination ==
             RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL &&
         request.response_target == kResponseTargetRtcore &&
         request.access_kind ==
             RTCORE_MEMORY_ACCESS_SHORT_STACK_STATE_READ &&
         !request.is_write &&
         request.owner_hw_sid == entry.input.owner.owner_hw_sid &&
         request.rt_request_id == entry.input.owner.request_identity &&
         request.resident_warp_id ==
             entry.input.owner.resident_warp_id &&
         request.request_generation == entry.input.owner.generation &&
         request.private_slot_id == entry.input.owner.private_slot_id &&
         request.lane_id == entry.input.owner.lane_id &&
         request.chunk_count == kReadChunkCount &&
         request.chunk_id < request.chunk_count &&
         request.memory_op_seq == request.chunk_id + 1 &&
         request.byte_mask != 0 &&
         transport.valid == 1 &&
         transport.reservation_id ==
             entry.reservation.reservation_id &&
         transport.reservation_age ==
             entry.reservation.reservation_age &&
         transport.target_operation_seq ==
             entry.reservation.operation_seq &&
         transport.producer_operation_seq ==
             entry.reservation.producer_operation_seq &&
         transport.target_slot_generation ==
             entry.reservation.slot_generation &&
         transport.target_slot_index ==
             entry.reservation.slot_index &&
         transport.operation_kind == kOperationNodeTransition &&
         transport.read_phase == kReadPhaseShortStackState &&
         bytes_are_zero(transport.reserved_zero,
                        sizeof(transport.reserved_zero));
}

int find_free_slot(const engine_state_v0 &state) {
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    if (state.slots[index].valid == 0) return static_cast<int>(index);
  }
  return -1;
}

int find_oldest_phase(const engine_state_v0 &state, phase_kind phase,
                      uint64_t service_cycle, bool require_mature) {
  int selected = -1;
  uint64_t oldest = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    const operation_entry_v0 &entry = state.slots[index];
    if (entry.valid == 0 || entry.phase != phase) continue;
    const uint64_t ready_cycle =
        phase == kPhaseParentLookup ? entry.parent_lookup_ready_cycle
                                    : entry.result_ready_cycle;
    if (require_mature && ready_cycle > service_cycle) continue;
    if (entry.issue_age < oldest) {
      selected = static_cast<int>(index);
      oldest = entry.issue_age;
    }
  }
  return selected;
}

status_kind begin_transition_commit(
    operation_entry_v0 *entry,
    timing_driver::state_v0 *timing_state) {
  if (entry == NULL || timing_state == NULL ||
      entry->transition.write_plan.access_count != kWriteChunkCount) {
    return kStatusInvalidArgument;
  }

  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry->input.owner, &request_binding)) {
    return kStatusOwnerMismatch;
  }
  timing_driver::state_v0 staged_timing = *timing_state;
  uint32_t commit_epoch = 0;
  if (timing_driver::begin_result_commit(
          &staged_timing, request_binding,
          entry->reservation.operation_seq, &commit_epoch) !=
      timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  entry->commit_epoch = commit_epoch;
  entry->phase = kPhaseWriting;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind enqueue_transition_write(
    operation_entry_v0 *entry, uint8_t chunk_index,
    timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle) {
  if (entry == NULL || timing_state == NULL || private_backing == NULL ||
      entry->phase != kPhaseWriting ||
      chunk_index >= kWriteChunkCount ||
      (entry->enqueued_write_mask & (1u << chunk_index)) != 0) {
    return kStatusInvalidArgument;
  }
  if (private_backing->shared_queue.size() >=
      private_shared::kSharedQueueCapacity) {
    return kStatusSharedQueueBackpressure;
  }

  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry->input.owner, &request_binding)) {
    return kStatusOwnerMismatch;
  }
  timing_driver::state_v0 staged_timing = *timing_state;
  private_shared::backing_state_v0 staged_backing = *private_backing;
  const uint64_t slot_base = private_slot_base(entry->input.owner);
  const private_frontier::shared_chunk_access_v0 &access =
      entry->transition.write_plan.accesses[chunk_index];
  if (access.access_kind != private_frontier::kAccessWrite ||
      access.aligned_32b_address < slot_base ||
      access.aligned_32b_address - slot_base >
          private_frontier::kPrivateDataSlotBytes -
              private_frontier::kSharedAccessChunkBytes ||
      access.byte_mask == 0) {
    return kStatusSharedPlanRejected;
  }
  const uint32_t chunk_offset = static_cast<uint32_t>(
      access.aligned_32b_address - slot_base);
  private_shared::shared_write_v0 write = {};
  write.valid = true;
  write.address_space = private_shared::kAddressSpaceShared;
  write.address_mode = private_shared::kAddressModePrivateField;
  write.access_operation = private_shared::kAccessOperationWrite;
  write.destination = private_shared::kDestinationPrivateCommitAck;
  write.owner = entry->input.owner;
  write.operation_seq = entry->reservation.operation_seq;
  write.commit_epoch = entry->commit_epoch;
  write.memory_op_seq = chunk_index + 1;
  write.chunk_id = chunk_index;
  write.chunk_count = kWriteChunkCount;
  write.field_kind = access.field_kind;
  write.aligned_32b_address = access.aligned_32b_address;
  write.byte_mask = access.byte_mask;
  std::memcpy(
      write.payload,
      entry->transition.updated_slot.bytes + chunk_offset,
      sizeof(write.payload));
  write.enqueue_cycle = service_cycle;
  if (timing_driver::begin_memory_transaction(
          &staged_timing, request_binding,
          entry->reservation.operation_seq) !=
          timing_driver::kStatusOk ||
      private_shared::enqueue_runtime_write(
          &staged_backing, write) != private_shared::kStatusOk) {
    return kStatusSharedWriteRejected;
  }
  entry->enqueued_write_mask = static_cast<uint8_t>(
      entry->enqueued_write_mask | (1u << chunk_index));
  *timing_state = staged_timing;
  *private_backing = staged_backing;
  return kStatusOk;
}

int find_oldest_pending_write(const engine_state_v0 &state) {
  int selected = -1;
  uint64_t oldest = std::numeric_limits<uint64_t>::max();
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    const operation_entry_v0 &entry = state.slots[index];
    if (entry.valid == 0 || entry.phase != kPhaseWriting ||
        entry.enqueued_write_mask == kAllWriteChunks) {
      continue;
    }
    if (entry.issue_age < oldest) {
      selected = static_cast<int>(index);
      oldest = entry.issue_age;
    }
  }
  return selected;
}

uint8_t first_missing_write_chunk(const operation_entry_v0 &entry) {
  for (uint8_t index = 0; index < kWriteChunkCount; ++index) {
    if ((entry.enqueued_write_mask & (1u << index)) == 0) {
      return index;
    }
  }
  return kWriteChunkCount;
}

}  // namespace

config_v0 candidate_profile_config() {
  config_v0 config = {};
  config.capacity = 16;
  config.reservation_width = 4;
  config.unit_count = 1;
  config.stack_latency = 2;
  config.initiation_interval = 1;
  config.issue_width = 1;
  config.parent_lookup_latency = 2;
  return config;
}

status_kind initialize(engine_state_v0 *state, const config_v0 &config) {
  if (state == NULL || !valid_config(config)) {
    return kStatusInvalidConfig;
  }
  *state = engine_state_v0();
  state->initialized = 1;
  state->config = config;
  state->next_reservation_id = 1;
  state->next_age = 1;
  return kStatusOk;
}

status_kind reserve(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const reservation_input_v0 &input, uint64_t reservation_cycle,
    reservation_receipt_v0 *reservation, request_plan_v0 *requests) {
  if (state == NULL || timing_state == NULL || reservation == NULL ||
      requests == NULL || state->initialized != 1 ||
      !valid_config(state->config) ||
      input.producer_operation_seq == 0 ||
      input.pending_parent_resume_valid > 1 ||
      !bytes_are_zero(input.reserved_zero,
                      sizeof(input.reserved_zero))) {
    return kStatusInvalidArgument;
  }
  *reservation = reservation_receipt_v0();
  *requests = request_plan_v0();
  if (state->reservation_cycle != reservation_cycle) {
    state->reservation_cycle = reservation_cycle;
    state->reservations_this_cycle = 0;
  }
  if (state->reservations_this_cycle >=
      state->config.reservation_width) {
    return kStatusReservationBudgetBackpressure;
  }
  const int slot_index = find_free_slot(*state);
  if (slot_index < 0) return kStatusCapacityBackpressure;

  request_owner::lane_binding_v0 request_binding = {};
  const private_shared::lane_slot_state_v0 *lane =
      private_shared::find_live_lane(private_backing, input.owner);
  if (!make_request_owner(input.owner, &request_binding) ||
      lane == NULL) {
    return kStatusOwnerMismatch;
  }
  private_frontier::access_plan_v0 read_plan = {};
  if (short_stack_shared::build_persistent_state_read_plan(
          lane->canonical_slot, input.owner, input.private_region,
          &read_plan) != short_stack_shared::kStatusOk ||
      read_plan.access_count != kReadChunkCount) {
    return kStatusSharedPlanRejected;
  }

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  uint32_t operation_seq = 0;
  if (timing_driver::allocate_target_operation(
          &staged_timing, request_binding, &operation_seq) !=
      timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }

  operation_entry_v0 &entry = staged_state.slots[slot_index];
  const uint32_t next_generation =
      entry.reservation.slot_generation == UINT32_MAX
          ? 1
          : entry.reservation.slot_generation + 1;
  entry = operation_entry_v0();
  entry.input = input;
  entry.read_slot.owner = input.owner;
  entry.read_plan = read_plan;
  entry.issue_age = staged_state.next_age++;
  entry.phase = kPhaseReading;
  entry.valid = 1;
  entry.reservation.owner = input.owner;
  entry.reservation.reservation_id =
      staged_state.next_reservation_id++;
  entry.reservation.reservation_age = entry.issue_age;
  entry.reservation.operation_seq = operation_seq;
  entry.reservation.producer_operation_seq =
      input.producer_operation_seq;
  entry.reservation.slot_generation = next_generation;
  entry.reservation.slot_index = static_cast<uint8_t>(slot_index);
  entry.reservation.read_chunk_count = kReadChunkCount;
  entry.reservation.valid = 1;

  const uint64_t slot_base = private_slot_base(input.owner);
  for (unsigned index = 0; index < kReadChunkCount; ++index) {
    const private_frontier::shared_chunk_access_v0 &access =
        read_plan.accesses[index];
    if (access.access_kind != private_frontier::kAccessRead ||
        access.aligned_32b_address < slot_base ||
        access.aligned_32b_address - slot_base > UINT16_MAX ||
        access.byte_mask == 0) {
      return kStatusSharedPlanRejected;
    }
    rtcore_memory_unit_request_snapshot &request =
        requests->requests[index];
    request.valid = true;
    request.address_space = RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
    request.operation = RTCORE_MEMORY_OPERATION_READ;
    request.destination =
        RTCORE_MEMORY_DESTINATION_SHORT_STACK_QUEUE_FILL;
    request.response_target = kResponseTargetRtcore;
    request.owner_hw_sid = input.owner.owner_hw_sid;
    request.rt_request_id = input.owner.request_identity;
    request.lane_id = input.owner.lane_id;
    request.resident_warp_id = input.owner.resident_warp_id;
    request.request_generation = input.owner.generation;
    request.private_slot_id = input.owner.private_slot_id;
    request.memory_op_seq = index + 1;
    request.chunk_id = index;
    request.chunk_count = kReadChunkCount;
    request.access_kind =
        RTCORE_MEMORY_ACCESS_SHORT_STACK_STATE_READ;
    request.aligned_32b_addr = access.aligned_32b_address;
    request.byte_mask = access.byte_mask;
    request.is_write = false;
    request.issue_cycle = reservation_cycle;
    rtcore_v04_stack_private_read_transport_snapshot &transport =
        request.v04_stack_private_read;
    transport.reservation_id = entry.reservation.reservation_id;
    transport.reservation_age = entry.reservation.reservation_age;
    transport.target_operation_seq = operation_seq;
    transport.producer_operation_seq =
        input.producer_operation_seq;
    transport.target_slot_generation =
        entry.reservation.slot_generation;
    transport.slot_chunk_offset = static_cast<uint16_t>(
        access.aligned_32b_address - slot_base);
    transport.target_slot_index =
        entry.reservation.slot_index;
    transport.field_kind = access.field_kind;
    transport.operation_kind = kOperationNodeTransition;
    transport.read_phase = kReadPhaseShortStackState;
    transport.valid = 1;
    if (!request_matches_entry(request, entry) ||
        timing_driver::begin_memory_transaction(
            &staged_timing, request_binding, operation_seq) !=
            timing_driver::kStatusOk) {
      return kStatusTimingControlRejected;
    }
  }
  requests->request_count = kReadChunkCount;
  requests->valid = 1;
  ++staged_state.reservations_this_cycle;
  *reservation = entry.reservation;
  *state = staged_state;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind accept_read_response(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    const private_shared::backing_state_v0 &private_backing,
    const rtcore_memory_unit_request_snapshot &request,
    uint64_t response_cycle) {
  if (state == NULL || timing_state == NULL ||
      state->initialized != 1 ||
      request.v04_stack_private_read.target_slot_index >=
          state->config.capacity) {
    return kStatusInvalidArgument;
  }
  const uint8_t slot_index =
      request.v04_stack_private_read.target_slot_index;
  const operation_entry_v0 &current = state->slots[slot_index];
  if (!request_matches_entry(request, current)) {
    return kStatusMalformedTransport;
  }
  const uint8_t chunk_bit =
      static_cast<uint8_t>(1u << request.chunk_id);
  if ((current.received_read_mask & chunk_bit) != 0) {
    return kStatusDuplicateResponse;
  }

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  private_frontier::shared_chunk_access_v0 access = {};
  access.aligned_32b_address = request.aligned_32b_addr;
  access.byte_mask = request.byte_mask;
  access.field_kind =
      request.v04_stack_private_read.field_kind;
  access.access_kind = private_frontier::kAccessRead;
  uint8_t payload[private_frontier::kSharedAccessChunkBytes] = {};
  if (private_shared::read_canonical_chunk(
          private_backing, entry.input.owner, access, payload) !=
      private_shared::kStatusOk) {
    return kStatusSharedPlanRejected;
  }
  const uint32_t chunk_offset =
      request.v04_stack_private_read.slot_chunk_offset;
  if (chunk_offset >
      private_frontier::kPrivateDataSlotBytes -
          private_frontier::kSharedAccessChunkBytes) {
    return kStatusMalformedTransport;
  }
  for (unsigned byte = 0;
       byte < private_frontier::kSharedAccessChunkBytes; ++byte) {
    if ((request.byte_mask & (uint32_t{1} << byte)) != 0) {
      entry.read_slot.bytes[chunk_offset + byte] = payload[byte];
    }
  }
  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry.input.owner, &request_binding) ||
      timing_driver::complete_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk) {
    return kStatusTimingControlRejected;
  }
  entry.received_read_mask =
      static_cast<uint8_t>(entry.received_read_mask | chunk_bit);
  if (entry.received_read_mask == kAllReadChunks) {
    entry.phase = kPhaseReadyToIssue;
    entry.result_ready_cycle = response_cycle;
  }
  *state = staged_state;
  *timing_state = staged_timing;
  return kStatusOk;
}

status_kind service_cycle(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    const parent_resolver_v0 &parent_resolver, uint64_t service_cycle,
    cycle_result_v0 *result) {
  if (state == NULL || timing_state == NULL ||
      private_backing == NULL || result == NULL ||
      state->initialized != 1 || parent_resolver.resolve == NULL) {
    return kStatusInvalidArgument;
  }
  *result = cycle_result_v0();

  const int lookup_index = find_oldest_phase(
      *state, kPhaseParentLookup, service_cycle, true);
  if (lookup_index >= 0) {
    operation_entry_v0 &entry = state->slots[lookup_index];
    short_stack::parent_edge_v0 parent = {};
    if (!parent_resolver.resolve(
            parent_resolver.context,
            entry.transition.parent_lookup_decode_context,
            entry.transition.parent_lookup_build_generation,
            entry.transition.parent_lookup_payload_offset,
            &parent)) {
      return kStatusParentResolveRejected;
    }
    entry.parent_edge = parent;
    entry.parent_edge_valid = 1;
    entry.phase = kPhaseExecuting;
    entry.result_ready_cycle = service_cycle;
    ++result->parent_lookup_completed;
  }

  uint8_t issued_unit_mask = 0;
  for (unsigned issued = 0; issued < state->config.issue_width; ++issued) {
    const int ready_index = find_oldest_phase(
        *state, kPhaseReadyToIssue, service_cycle, false);
    if (ready_index < 0) break;
    int unit_index = -1;
    for (unsigned unit = 0; unit < state->config.unit_count; ++unit) {
      if ((issued_unit_mask & (1u << unit)) == 0 &&
          state->units[unit].next_issue_cycle <= service_cycle) {
        unit_index = static_cast<int>(unit);
        break;
      }
    }
    if (unit_index < 0) break;
    operation_entry_v0 &entry = state->slots[ready_index];
    entry.phase = kPhaseExecuting;
    entry.issue_cycle = service_cycle;
    entry.result_ready_cycle =
        service_cycle + state->config.stack_latency;
    state->units[unit_index].next_issue_cycle =
        service_cycle + state->config.initiation_interval;
    issued_unit_mask = static_cast<uint8_t>(
        issued_unit_mask | (1u << unit_index));
    ++result->issued;
  }

  const int execute_index = find_oldest_phase(
      *state, kPhaseExecuting, service_cycle, true);
  if (execute_index >= 0) {
    operation_entry_v0 &entry = state->slots[execute_index];
    short_stack_transition::node_input_v0 transition_input = {};
    transition_input.owner = entry.input.owner;
    transition_input.region = entry.input.private_region;
    transition_input.canonical_slot = entry.read_slot;
    transition_input.node_route = entry.input.node_route;
    transition_input.current_target = entry.input.current_target;
    transition_input.current_decode_context =
        entry.input.current_decode_context;
    transition_input.pending_parent_resume =
        entry.input.pending_parent_resume;
    transition_input.pending_parent_resume_valid =
        entry.input.pending_parent_resume_valid;
    transition_input.parent_edge = entry.parent_edge;
    transition_input.parent_edge_valid = entry.parent_edge_valid;
    short_stack_transition::result_v0 transition = {};
    const short_stack_transition::status_kind transition_status =
        short_stack_transition::prepare_node_transition(
            transition_input, &transition);
    if (transition_status ==
        short_stack_transition::kStatusParentLookupRequired) {
      entry.transition = transition;
      entry.phase = kPhaseParentLookup;
      entry.parent_lookup_ready_cycle =
          service_cycle + state->config.parent_lookup_latency;
    } else if (transition_status !=
               short_stack_transition::kStatusOk) {
      return kStatusTransitionRejected;
    } else {
      entry.transition = transition;
      const status_kind commit_status =
          begin_transition_commit(&entry, timing_state);
      if (commit_status != kStatusOk) return commit_status;
      ++result->transition_committed;
    }
  }

  result->active_operations = active_operation_count(*state);
  for (unsigned index = 0; index < state->config.capacity; ++index) {
    result->ready_results +=
        state->slots[index].valid != 0 &&
        state->slots[index].phase == kPhaseResultReady;
  }
  return kStatusOk;
}

status_kind service_write_enqueue(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle, uint8_t write_enqueue_budget,
    uint8_t *writes_enqueued, bool *shared_queue_blocked) {
  if (state == NULL || timing_state == NULL ||
      private_backing == NULL || writes_enqueued == NULL ||
      shared_queue_blocked == NULL || state->initialized != 1) {
    return kStatusInvalidArgument;
  }
  *writes_enqueued = 0;
  *shared_queue_blocked = false;
  while (*writes_enqueued < write_enqueue_budget) {
    const int write_index = find_oldest_pending_write(*state);
    if (write_index < 0) break;
    operation_entry_v0 &entry = state->slots[write_index];
    const uint8_t chunk_index = first_missing_write_chunk(entry);
    const status_kind write_status = enqueue_transition_write(
        &entry, chunk_index, timing_state, private_backing,
        service_cycle);
    if (write_status == kStatusSharedQueueBackpressure) {
      *shared_queue_blocked = true;
      break;
    }
    if (write_status != kStatusOk) return write_status;
    ++*writes_enqueued;
  }
  return kStatusOk;
}

bool owns_ack(const engine_state_v0 &state,
              const private_shared::runtime_write_ack_v0 &ack) {
  if (state.initialized != 1 || !ack.valid ||
      ack.memory_operation_seq == 0 ||
      ack.memory_operation_seq > kWriteChunkCount) {
    return false;
  }
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    const operation_entry_v0 &entry = state.slots[index];
    if (entry.valid != 0 && entry.phase == kPhaseWriting &&
        entry.reservation.operation_seq == ack.operation_seq &&
        entry.commit_epoch == ack.commit_epoch &&
        private_frontier::owners_equal(entry.input.owner, ack.owner) &&
        (entry.enqueued_write_mask &
         (1u << (ack.memory_operation_seq - 1))) != 0 &&
        (entry.acknowledged_write_mask &
         (1u << (ack.memory_operation_seq - 1))) == 0) {
      return true;
    }
  }
  return false;
}

status_kind accept_write_ack(
    engine_state_v0 *state, timing_driver::state_v0 *timing_state,
    private_shared::backing_state_v0 *private_backing,
    uint64_t service_cycle,
    const private_shared::runtime_write_ack_v0 &ack) {
  if (state == NULL || timing_state == NULL ||
      private_backing == NULL || !owns_ack(*state, ack)) {
    return kStatusNoAckOwned;
  }
  int slot_index = -1;
  for (unsigned index = 0; index < state->config.capacity; ++index) {
    const operation_entry_v0 &entry = state->slots[index];
    if (entry.valid != 0 && entry.phase == kPhaseWriting &&
        entry.reservation.operation_seq == ack.operation_seq &&
        entry.commit_epoch == ack.commit_epoch &&
        private_frontier::owners_equal(entry.input.owner, ack.owner)) {
      slot_index = static_cast<int>(index);
      break;
    }
  }
  if (slot_index < 0) return kStatusNoAckOwned;

  engine_state_v0 staged_state = *state;
  timing_driver::state_v0 staged_timing = *timing_state;
  private_shared::backing_state_v0 staged_backing = *private_backing;
  operation_entry_v0 &entry = staged_state.slots[slot_index];
  request_owner::lane_binding_v0 request_binding = {};
  if (!make_request_owner(entry.input.owner, &request_binding) ||
      timing_driver::complete_memory_transaction(
          &staged_timing, request_binding,
          entry.reservation.operation_seq) !=
          timing_driver::kStatusOk ||
      private_shared::detail::commit_runtime_write_ack(
          &staged_backing, service_cycle, ack) !=
          private_shared::kStatusOk) {
    return kStatusAckRejected;
  }
  entry.acknowledged_write_mask = static_cast<uint8_t>(
      entry.acknowledged_write_mask |
      (1u << (ack.memory_operation_seq - 1)));
  if (entry.acknowledged_write_mask == kAllWriteChunks) {
    entry.phase = kPhaseResultReady;
  }
  *state = staged_state;
  *timing_state = staged_timing;
  *private_backing = staged_backing;
  return kStatusOk;
}

status_kind peek_ready_result(const engine_state_v0 &state,
                              ready_result_v0 *result) {
  if (state.initialized != 1 || result == NULL) {
    return kStatusInvalidArgument;
  }
  *result = ready_result_v0();
  const int index = find_oldest_phase(
      state, kPhaseResultReady, 0, false);
  if (index < 0) return kStatusNoReadyResult;
  const operation_entry_v0 &entry = state.slots[index];
  result->owner = entry.input.owner;
  result->ray_policy = entry.input.ray_policy;
  result->transition = entry.transition;
  result->reservation_id = entry.reservation.reservation_id;
  result->operation_seq = entry.reservation.operation_seq;
  result->producer_operation_seq =
      entry.reservation.producer_operation_seq;
  result->commit_epoch = entry.commit_epoch;
  result->slot_index = entry.reservation.slot_index;
  result->slot_generation =
      entry.reservation.slot_generation;
  result->valid = 1;
  return kStatusOk;
}

status_kind consume_ready_result(engine_state_v0 *state,
                                 const ready_result_v0 &result) {
  if (state == NULL || state->initialized != 1 || !result.valid ||
      result.slot_index >= state->config.capacity) {
    return kStatusInvalidArgument;
  }
  operation_entry_v0 &entry = state->slots[result.slot_index];
  if (entry.valid == 0 || entry.phase != kPhaseResultReady ||
      entry.reservation.reservation_id != result.reservation_id ||
      entry.reservation.operation_seq != result.operation_seq ||
      entry.commit_epoch != result.commit_epoch ||
      !private_frontier::owners_equal(entry.input.owner,
                                      result.owner)) {
    return kStatusNoReadyResult;
  }
  const uint32_t generation = entry.reservation.slot_generation;
  entry = operation_entry_v0();
  entry.reservation.slot_generation = generation;
  return kStatusOk;
}

uint8_t active_operation_count(const engine_state_v0 &state) {
  uint8_t count = 0;
  if (state.initialized != 1) return 0;
  for (unsigned index = 0; index < state.config.capacity; ++index) {
    count += state.slots[index].valid != 0;
  }
  return count;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfig:
      return "invalid_config";
    case kStatusCapacityBackpressure:
      return "capacity_backpressure";
    case kStatusReservationBudgetBackpressure:
      return "reservation_budget_backpressure";
    case kStatusOwnerMismatch:
      return "owner_mismatch";
    case kStatusTimingControlRejected:
      return "timing_control_rejected";
    case kStatusSharedPlanRejected:
      return "shared_plan_rejected";
    case kStatusMalformedTransport:
      return "malformed_transport";
    case kStatusDuplicateResponse:
      return "duplicate_response";
    case kStatusTransitionRejected:
      return "transition_rejected";
    case kStatusParentResolveRejected:
      return "parent_resolve_rejected";
    case kStatusSharedQueueBackpressure:
      return "shared_queue_backpressure";
    case kStatusSharedWriteRejected:
      return "shared_write_rejected";
    case kStatusNoAckOwned:
      return "no_ack_owned";
    case kStatusAckRejected:
      return "ack_rejected";
    case kStatusNoReadyResult:
      return "no_ready_result";
  }
  return "unknown";
}

}  // namespace short_stack_timing
}  // namespace v04
}  // namespace rtcore
