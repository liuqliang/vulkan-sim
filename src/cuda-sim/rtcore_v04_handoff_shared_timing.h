#ifndef RTCORE_V04_HANDOFF_SHARED_TIMING_H
#define RTCORE_V04_HANDOFF_SHARED_TIMING_H

#include <cstdint>
#include <deque>

#include "rtcore_v04_handoff_storage_profile.h"
#include "rtcore_v04_private_placement_profile.h"

namespace rtcore {
namespace v04 {
namespace handoff_shared_timing {

enum client_kind : uint8_t {
  kClientInvalid = 0,
  kClientRtMemoryUnit = 1,
  kClientOrdinaryLsu = 2,
};

enum object_kind : uint8_t {
  kObjectInvalid = 0,
  kObjectHandoff = 1,
  kObjectPrivate = 2,
};

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusInvalidConfiguration,
  kStatusQueueFull,
  kStatusOutstandingFull,
  kStatusResponseFull,
  kStatusDuplicateToken,
  kStatusNoReadyResponse,
};

struct config_v0 {
  uint32_t issue_budget;
  uint32_t response_budget;
  uint32_t base_latency;
  uint32_t ingress_capacity;
  uint32_t outstanding_capacity;
  uint32_t response_capacity;
  uint32_t bank_count;
  uint32_t bank_word_bytes;
};

struct request_v0 {
  bool valid;
  uint8_t client;
  uint8_t object;
  uint8_t chunk;
  uint8_t is_write;
  uint32_t owner_hw_sid;
  uint64_t token;
  uint64_t aligned_32b_address;
  uint32_t bank_mask;
  uint64_t enqueue_cycle;
  uint64_t accept_cycle;
  uint64_t ready_cycle;
  uint64_t order;
};

struct stats_v0 {
  uint64_t enqueue_count;
  uint64_t enqueue_full_count;
  uint64_t issue_count;
  uint64_t bank_conflict_blocked_count;
  uint64_t ordinary_shared_blocked_count;
  uint64_t outstanding_full_count;
  uint64_t response_full_count;
  uint64_t response_ready_count;
  uint64_t response_pop_count;
  uint64_t response_hol_bypass_count;
  uint64_t issue_notification_pop_count;
  uint64_t rt_client_count;
  uint64_t lsu_client_count;
  uint64_t handoff_object_count;
  uint64_t private_object_count;
  uint32_t max_ingress_depth;
  uint32_t max_outstanding_depth;
  uint32_t max_response_depth;
};

struct state_v0 {
  bool initialized;
  uint8_t reserved_zero[7];
  uint32_t owner_hw_sid;
  config_v0 config;
  uint64_t next_order;
  stats_v0 stats;
  std::deque<request_v0> ingress;
  std::deque<request_v0> outstanding;
  std::deque<request_v0> issued;
  std::deque<request_v0> responses;
};

inline config_v0 default_config() {
  config_v0 config = {};
  config.issue_budget = 4u;
  config.response_budget = 4u;
  config.base_latency = 1u;
  config.ingress_capacity = 32u;
  config.outstanding_capacity = 32u;
  config.response_capacity = 32u;
  config.bank_count = handoff_storage::kSharedBankCount;
  config.bank_word_bytes = handoff_storage::kSharedBankWordBytes;
  return config;
}

inline bool config_valid(const config_v0 &config) {
  return config.issue_budget != 0u && config.issue_budget <= 32u &&
         config.response_budget != 0u && config.response_budget <= 32u &&
         config.base_latency != 0u && config.ingress_capacity != 0u &&
         config.outstanding_capacity != 0u &&
         config.response_capacity != 0u && config.bank_count != 0u &&
         config.bank_count <= 32u && config.bank_word_bytes != 0u &&
         handoff_storage::kChunkBytes % config.bank_word_bytes == 0u;
}

inline status_kind initialize(state_v0 *state, uint32_t owner_hw_sid,
                              const config_v0 &config) {
  if (state == NULL) return kStatusInvalidArgument;
  if (!config_valid(config)) return kStatusInvalidConfiguration;
  *state = state_v0();
  state->initialized = true;
  state->owner_hw_sid = owner_hw_sid;
  state->config = config;
  state->next_order = 1u;
  return kStatusOk;
}

inline bool token_live(const state_v0 &state, uint64_t token) {
  if (token == 0u) return false;
  const std::deque<request_v0> *queues[] = {
      &state.ingress, &state.outstanding, &state.responses};
  for (unsigned queue = 0; queue < 3u; ++queue) {
    for (std::deque<request_v0>::const_iterator it = queues[queue]->begin();
         it != queues[queue]->end(); ++it) {
      if (it->token == token) return true;
    }
  }
  return false;
}

inline status_kind enqueue(state_v0 *state, const request_v0 &request,
                           uint64_t enqueue_cycle) {
  const bool valid_chunk =
      (request.object == kObjectHandoff &&
       request.chunk < handoff_storage::kChunkCount) ||
      (request.object == kObjectPrivate &&
       request.chunk < private_placement::kChunkCount);
  if (state == NULL || !state->initialized || !request.valid ||
      request.owner_hw_sid != state->owner_hw_sid || request.token == 0u ||
      !valid_chunk ||
      (request.client != kClientRtMemoryUnit &&
       request.client != kClientOrdinaryLsu) ||
      (request.object == kObjectPrivate &&
       request.client != kClientRtMemoryUnit) ||
      request.aligned_32b_address % handoff_storage::kChunkBytes != 0u) {
    return kStatusInvalidArgument;
  }
  uint32_t expected_bank_mask = 0;
  if (handoff_storage::bank_mask_for_32b(
          request.aligned_32b_address, state->config.bank_count,
          state->config.bank_word_bytes, &expected_bank_mask) !=
          handoff_storage::kStatusOk ||
      request.bank_mask != expected_bank_mask) {
    return kStatusInvalidArgument;
  }
  if (token_live(*state, request.token)) return kStatusDuplicateToken;
  if (state->ingress.size() >= state->config.ingress_capacity) {
    ++state->stats.enqueue_full_count;
    return kStatusQueueFull;
  }
  request_v0 queued = request;
  queued.enqueue_cycle = enqueue_cycle;
  queued.accept_cycle = 0u;
  queued.ready_cycle = 0u;
  queued.order = state->next_order++;
  state->ingress.push_back(queued);
  ++state->stats.enqueue_count;
  if (queued.client == kClientRtMemoryUnit) {
    ++state->stats.rt_client_count;
  } else {
    ++state->stats.lsu_client_count;
  }
  if (queued.object == kObjectHandoff) {
    ++state->stats.handoff_object_count;
  } else {
    ++state->stats.private_object_count;
  }
  if (state->ingress.size() > state->stats.max_ingress_depth) {
    state->stats.max_ingress_depth = state->ingress.size();
  }
  return kStatusOk;
}

inline uint32_t service_responses(state_v0 *state, uint64_t cycle) {
  if (state == NULL || !state->initialized) return 0u;
  uint32_t progressed = 0;
  for (std::deque<request_v0>::iterator it = state->outstanding.begin();
       it != state->outstanding.end() &&
       progressed < state->config.response_budget;) {
    if (it->ready_cycle > cycle) {
      ++it;
      continue;
    }
    if (state->responses.size() >= state->config.response_capacity) {
      ++state->stats.response_full_count;
      break;
    }
    state->responses.push_back(*it);
    it = state->outstanding.erase(it);
    ++state->stats.response_ready_count;
    ++progressed;
  }
  if (state->responses.size() > state->stats.max_response_depth) {
    state->stats.max_response_depth = state->responses.size();
  }
  return progressed;
}

inline uint32_t service_issues(state_v0 *state, uint64_t cycle,
                               uint32_t ordinary_shared_bank_mask) {
  if (state == NULL || !state->initialized) return 0u;
  uint32_t used_bank_mask = ordinary_shared_bank_mask;
  uint32_t progressed = 0;
  for (std::deque<request_v0>::iterator it = state->ingress.begin();
       it != state->ingress.end() &&
       progressed < state->config.issue_budget;) {
    if (state->outstanding.size() >= state->config.outstanding_capacity) {
      ++state->stats.outstanding_full_count;
      break;
    }
    if (handoff_storage::bank_masks_conflict(it->bank_mask,
                                             used_bank_mask)) {
      if (ordinary_shared_bank_mask != 0u &&
          handoff_storage::bank_masks_conflict(
              it->bank_mask, ordinary_shared_bank_mask)) {
        ++state->stats.ordinary_shared_blocked_count;
      } else {
        ++state->stats.bank_conflict_blocked_count;
      }
      ++it;
      continue;
    }
    request_v0 accepted = *it;
    it = state->ingress.erase(it);
    accepted.accept_cycle = cycle;
    accepted.ready_cycle = cycle + state->config.base_latency;
    state->outstanding.push_back(accepted);
    state->issued.push_back(accepted);
    used_bank_mask |= accepted.bank_mask;
    ++state->stats.issue_count;
    ++progressed;
  }
  if (state->outstanding.size() > state->stats.max_outstanding_depth) {
    state->stats.max_outstanding_depth = state->outstanding.size();
  }
  return progressed;
}

inline status_kind pop_issued(state_v0 *state, request_v0 *issued) {
  if (state == NULL || issued == NULL || !state->initialized) {
    return kStatusInvalidArgument;
  }
  if (state->issued.empty()) return kStatusNoReadyResponse;
  *issued = state->issued.front();
  state->issued.pop_front();
  ++state->stats.issue_notification_pop_count;
  return kStatusOk;
}

inline status_kind pop_response(state_v0 *state, request_v0 *response) {
  if (state == NULL || response == NULL || !state->initialized) {
    return kStatusInvalidArgument;
  }
  if (state->responses.empty()) return kStatusNoReadyResponse;
  *response = state->responses.front();
  state->responses.pop_front();
  ++state->stats.response_pop_count;
  return kStatusOk;
}

inline status_kind pop_response_for_client(state_v0 *state,
                                           client_kind client,
                                           request_v0 *response) {
  if (state == NULL || response == NULL || !state->initialized ||
      (client != kClientRtMemoryUnit && client != kClientOrdinaryLsu)) {
    return kStatusInvalidArgument;
  }
  for (std::deque<request_v0>::iterator it = state->responses.begin();
       it != state->responses.end(); ++it) {
    if (it->client != client) continue;
    if (it != state->responses.begin()) {
      ++state->stats.response_hol_bypass_count;
    }
    *response = *it;
    state->responses.erase(it);
    ++state->stats.response_pop_count;
    return kStatusOk;
  }
  return kStatusNoReadyResponse;
}

inline status_kind peek_response(const state_v0 &state,
                                 request_v0 *response) {
  if (response == NULL || !state.initialized) {
    return kStatusInvalidArgument;
  }
  if (state.responses.empty()) return kStatusNoReadyResponse;
  *response = state.responses.front();
  return kStatusOk;
}

inline bool drained(const state_v0 &state) {
  return state.ingress.empty() && state.outstanding.empty() &&
         state.issued.empty() && state.responses.empty();
}

inline const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusInvalidConfiguration:
      return "invalid_configuration";
    case kStatusQueueFull:
      return "queue_full";
    case kStatusOutstandingFull:
      return "outstanding_full";
    case kStatusResponseFull:
      return "response_full";
    case kStatusDuplicateToken:
      return "duplicate_token";
    case kStatusNoReadyResponse:
      return "no_ready_response";
  }
  return "unknown";
}

}  // namespace handoff_shared_timing
}  // namespace v04
}  // namespace rtcore

#endif
