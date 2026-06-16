// Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "vulkan_ray_tracing.h"
#include "vulkan_rt_thread_data.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <deque>
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

static bool rt_progress_logging_enabled() {
    static int enabled = []() {
        const char *value = getenv("VULKAN_SIM_PROGRESS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled;
}

struct rtcore_pixel_trace_filter {
    bool enabled = false;
    unsigned x = 0;
    unsigned y = 0;
};

static const rtcore_pixel_trace_filter &rtcore_pixel_trace_filter_value()
{
    static rtcore_pixel_trace_filter filter = []() {
        rtcore_pixel_trace_filter parsed;
        const char *value = getenv("VULKAN_SIM_RTCORE_PIXEL_TRACE");
        if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
            return parsed;
        }

        unsigned x = 0;
        unsigned y = 0;
        if (sscanf(value, "%u,%u", &x, &y) == 2) {
            parsed.enabled = true;
            parsed.x = x;
            parsed.y = y;
        } else {
            printf("GPGPU-Sim RTCORE_PIXEL_TRACE ignored invalid filter '%s' "
                   "(expected x,y)\n",
                   value);
        }
        return parsed;
    }();
    return filter;
}

static bool rtcore_pixel_trace_matches(unsigned x, unsigned y)
{
    const rtcore_pixel_trace_filter &filter = rtcore_pixel_trace_filter_value();
    return filter.enabled && filter.x == x && filter.y == y;
}

#define __CUDA_RUNTIME_API_H__
// clang-format off
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "../../libcuda/cuda_api.h"
#include "cudaProfiler.h"
// clang-format on
#if (CUDART_VERSION < 8000)
#include "__cudaFatFormat.h"
#endif

#include "../../libcuda/gpgpu_context.h"
#include "../../libcuda/cuda_api_object.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../cuda-sim/ptx_loader.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_parser.h"
#include "../gpgpusim_entrypoint.h"
#include "../stream_manager.h"
#include "../abstract_hardware_model.h"
#include "vulkan_acceleration_structure_util.h"
#include "../gpgpu-sim/vector-math.h"

#if defined(MESA_USE_LVPIPE_DRIVER)
#include "lvp_private.h"
#endif 
//#include "intel_image_util.h"
#include "astc_decomp.h"

// #define HAVE_PTHREAD
// #define UTIL_ARCH_LITTLE_ENDIAN 1
// #define UTIL_ARCH_BIG_ENDIAN 0
// #define signbit signbit

// #define UINT_MAX 65535
// #define GLuint MESA_GLuint
// // #include "isl/isl.h"
// // #include "isl/isl_tiled_memcpy.c"
// #include "vulkan/anv_private.h"
// #undef GLuint

// #undef HAVE_PTHREAD
// #undef UTIL_ARCH_LITTLE_ENDIAN
// #undef UTIL_ARCH_BIG_ENDIAN
// #undef signbit

// #include "vulkan/anv_public.h"

#if defined(MESA_USE_INTEL_DRIVER)
#include "intel_image.h"
#elif defined(MESA_USE_LVPIPE_DRIVER)
// #include "lvp_image.h"
#endif

// #include "anv_include.h"

VkRayTracingPipelineCreateInfoKHR* VulkanRayTracing::pCreateInfos = NULL;
VkAccelerationStructureGeometryKHR* VulkanRayTracing::pGeometries = NULL;
uint32_t VulkanRayTracing::geometryCount = 0;
VkAccelerationStructureKHR VulkanRayTracing::topLevelAS = NULL;
std::vector<std::vector<Descriptor> > VulkanRayTracing::descriptors;
std::ofstream VulkanRayTracing::imageFile;
std::map<std::string, std::string> outputImages;
bool VulkanRayTracing::firstTime = true;
std::vector<shader_stage_info> VulkanRayTracing::shaders;
// RayDebugGPUData VulkanRayTracing::rayDebugGPUData[2000][2000] = {0};
struct DESCRIPTOR_SET_STRUCT* VulkanRayTracing::descriptorSet = NULL;
void* VulkanRayTracing::launcher_descriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS] = {NULL};
void* VulkanRayTracing::launcher_deviceDescriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS] = {NULL};
std::vector<void*> VulkanRayTracing::child_addrs_from_driver;
std::map<void*, void*> VulkanRayTracing::blas_addr_map;
void* VulkanRayTracing::tlas_addr;

bool VulkanRayTracing::dumped = false;

bool use_external_launcher = false;
const bool dump_trace = false;

bool VulkanRayTracing::_init_ = false;
warp_intersection_table *** VulkanRayTracing::intersection_table;
warp_intersection_table *** VulkanRayTracing::anyhit_table;
IntersectionTableType VulkanRayTracing::intersectionTableType = IntersectionTableType::Baseline;

static unsigned rtcore_launch_id_x_for_thread(ptx_thread_info *thread);
static unsigned rtcore_launch_id_y_for_thread(ptx_thread_info *thread);
static bool rtcore_pixel_trace_matches_thread(ptx_thread_info *thread);

static const char *RTCORE_TRACE_REPLAY_MODEL_NAME =
    "RTCORE_SM_LOCAL_BOUNDED_TRACE_REPLAY_V0_1";
static const unsigned RTCORE_COMPACT_TRACE_DEFAULT_EVENTS_PER_LANE = 64;
static const unsigned RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR = 256;
static const unsigned RTCORE_COMPACT_TRACE_EVENT_TARGET_BYTES = 16;

enum rtcore_compact_trace_event_type {
    RTCORE_TRACE_NODE_FETCH = 0,
    RTCORE_TRACE_NODE_TEST,
    RTCORE_TRACE_STACK_PUSH,
    RTCORE_TRACE_STACK_POP,
    RTCORE_TRACE_PRIMITIVE_FETCH,
    RTCORE_TRACE_PRIMITIVE_TEST,
    RTCORE_TRACE_MEMORY_WAIT,
    RTCORE_TRACE_HIT_UPDATE,
    RTCORE_TRACE_COMPLETION,
    RTCORE_TRACE_OVERFLOW_SUMMARY,
};

enum rtcore_compact_trace_resource_class {
    RTCORE_TRACE_RESOURCE_NODE = 0,
    RTCORE_TRACE_RESOURCE_PRIMITIVE,
    RTCORE_TRACE_RESOURCE_MEMORY,
    RTCORE_TRACE_RESOURCE_STACK,
    RTCORE_TRACE_RESOURCE_COMPLETION,
    RTCORE_TRACE_RESOURCE_SUMMARY,
};

enum rtcore_compact_trace_node_kind {
    RTCORE_TRACE_NODE_KIND_BVH_HEADER = 0,
    RTCORE_TRACE_NODE_KIND_INTERNAL,
    RTCORE_TRACE_NODE_KIND_INSTANCE_LEAF,
};

enum rtcore_compact_trace_primitive_kind {
    RTCORE_TRACE_PRIMITIVE_KIND_LEAF_DESCRIPTOR = 0,
    RTCORE_TRACE_PRIMITIVE_KIND_QUAD_LEAF,
    RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_LEAF,
    RTCORE_TRACE_PRIMITIVE_KIND_TRIANGLE_TEST,
    RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_DEFERRED,
};

enum rtcore_compact_trace_hit_update_kind {
    RTCORE_TRACE_HIT_UPDATE_KIND_ANY_HIT = 0,
    RTCORE_TRACE_HIT_UPDATE_KIND_CLOSEST_HIT,
};

enum rtcore_trace_timing_precision_class {
    RTCORE_TRACE_TIMING_PRECISION_EXACT = 0,
    RTCORE_TRACE_TIMING_PRECISION_BOUNDED_OVERFLOW_SUMMARY,
};

struct rtcore_compact_trace_overflow_summary {
    unsigned overflow_node_fetch_count;
    unsigned overflow_node_test_count;
    unsigned overflow_primitive_fetch_count;
    unsigned overflow_primitive_test_count;
    unsigned overflow_stack_push_count;
    unsigned overflow_stack_pop_count;
    unsigned overflow_memory_wait_count;
    unsigned overflow_memory_bytes;
    unsigned overflow_completion_count;
};

enum rtcore_replay_lane_request_state {
    RTCORE_REPLAY_QUEUED = 0,
    RTCORE_REPLAY_READY_NODE,
    RTCORE_REPLAY_READY_PRIMITIVE,
    RTCORE_REPLAY_READY_STACK,
    RTCORE_REPLAY_WAITING_MEMORY,
    RTCORE_REPLAY_READY_COMPLETION,
    RTCORE_REPLAY_DONE,
};

enum rtcore_replay_ready_selection_policy {
    RTCORE_REPLAY_SELECT_OLDEST_READY_FIRST = 0,
    RTCORE_REPLAY_SELECT_ROUND_ROBIN_READY_ORDER,
};

struct rtcore_compact_trace_event {
    uint64_t address_or_ref;
    uint32_t packed_fields;
    uint16_t event_seq;
    uint16_t packed_count_bytes;
};

static_assert(sizeof(rtcore_compact_trace_event) <=
                  RTCORE_COMPACT_TRACE_EVENT_TARGET_BYTES,
              "rtcore_compact_trace_event must stay within the compact target");

struct rtcore_compact_trace_export_record {
    bool valid;
    const char *model_name;
    unsigned thread_uid;
    unsigned owner_hw_sid;
    unsigned lane_id;
    bool has_warp_metadata;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;
    unsigned static_inst_uid;
    unsigned event_count;
    unsigned max_trace_events_per_lane;
    bool timing_trace_overflowed;
    rtcore_trace_timing_precision_class timing_precision_class;
    unsigned overflow_summary_events;
    rtcore_compact_trace_overflow_summary overflow_summary;
    std::vector<rtcore_compact_trace_event> events;
};

static std::map<unsigned, rtcore_compact_trace_export_record>
    g_rtcore_compact_trace_exports;

struct rtcore_replay_lane_request {
    bool valid;
    unsigned thread_uid;
    unsigned owner_hw_sid;
    unsigned lane_id;
    bool has_warp_metadata;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;
    unsigned static_inst_uid;
    unsigned next_event_index;
    unsigned event_count;
    unsigned node_event_count;
    unsigned primitive_event_count;
    unsigned stack_event_count;
    unsigned memory_event_count;
    unsigned completion_event_count;
    unsigned ready_order;
    bool timing_trace_overflowed;
    rtcore_replay_lane_request_state state;
    std::vector<rtcore_compact_trace_event> events;
};

static std::map<unsigned, rtcore_replay_lane_request>
    g_rtcore_replay_lane_requests;

struct rtcore_replay_warp_completion_shadow_key {
    unsigned owner_hw_sid;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;

    bool operator<(const rtcore_replay_warp_completion_shadow_key &other) const
    {
        if (owner_hw_sid != other.owner_hw_sid) {
            return owner_hw_sid < other.owner_hw_sid;
        }
        if (warp_uid != other.warp_uid) {
            return warp_uid < other.warp_uid;
        }
        if (warp_id != other.warp_id) {
            return warp_id < other.warp_id;
        }
        return active_mask < other.active_mask;
    }
};

struct rtcore_replay_warp_completion_shadow_state {
    bool valid;
    rtcore_replay_warp_completion_shadow_key key;
    unsigned admitted_lane_mask;
    unsigned completed_lane_mask;
    unsigned completed_lane_count;
    bool all_active_lanes_complete;
    bool all_active_lanes_complete_logged;
};

struct rtcore_replay_warp_completion_shadow_snapshot {
    bool enabled;
    bool found;
    bool all_active_lanes_complete;
    unsigned owner_hw_sid;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;
    unsigned admitted_lane_mask;
    unsigned completed_lane_mask;
    unsigned completed_lane_count;
};

static std::map<rtcore_replay_warp_completion_shadow_key,
                rtcore_replay_warp_completion_shadow_state>
    g_rtcore_replay_warp_completion_shadow;

struct rtcore_replay_ready_queues {
    std::deque<unsigned> queued_queue;
    std::deque<unsigned> ready_node_queue;
    std::deque<unsigned> ready_primitive_queue;
    std::deque<unsigned> ready_stack_queue;
    std::deque<unsigned> waiting_memory_queue;
    std::deque<unsigned> ready_completion_queue;
    std::deque<unsigned> done_queue;
};

struct rtcore_replay_issue_budget {
    unsigned node_issue_budget;
    unsigned primitive_issue_budget;
    unsigned stack_issue_budget;
    unsigned completion_issue_budget;
};

struct rtcore_replay_service_cycle_identity_snapshot {
    bool valid;
    bool memory_progressed;
    bool ready_progressed;
    unsigned owner_hw_sid;
    unsigned thread_uid;
    unsigned lane_id;
    bool has_warp_metadata;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;
    unsigned static_inst_uid;
};

struct rtcore_replay_service_tick_result {
    bool progressed;
    bool memory_progressed;
    bool ready_progressed;
    rtcore_replay_service_cycle_identity_snapshot last_progress_identity;
};

struct rtcore_replay_service_tick_stats {
    unsigned tick_attempts;
    unsigned ticks_progressed;
    unsigned memory_ticks_progressed;
    unsigned ready_ticks_progressed;
};

struct rtcore_replay_unit_arbitration_stats {
    unsigned node_unit_issue_attempts;
    unsigned node_unit_issued;
    unsigned node_unit_budget_exhausted;
    unsigned primitive_unit_issue_attempts;
    unsigned primitive_unit_issued;
    unsigned primitive_unit_budget_exhausted;
    unsigned stack_unit_issue_attempts;
    unsigned stack_unit_issued;
    unsigned stack_unit_budget_exhausted;
    unsigned completion_unit_issue_attempts;
    unsigned completion_unit_issued;
    unsigned completion_unit_budget_exhausted;
};

struct rtcore_replay_service_tick_stats_snapshot {
    bool valid;
    unsigned tick_attempts;
    unsigned ticks_progressed;
    unsigned memory_ticks_progressed;
    unsigned ready_ticks_progressed;
};

typedef rtcore_replay_service_tick_stats_snapshot
    rtcore_service_tick_stats_snapshot;

struct rtcore_replay_service_cycle_result {
    bool service_enabled;
    unsigned owner_hw_sid;
    unsigned long long service_cycle;
    rtcore_replay_service_tick_result tick_result;
    rtcore_service_tick_stats_snapshot stats_snapshot;
};

static rtcore_replay_ready_queues g_rtcore_replay_ready_queues;
static rtcore_replay_service_tick_stats g_rtcore_replay_service_tick_stats;
static rtcore_replay_unit_arbitration_stats
    g_rtcore_replay_unit_arbitration_stats;
static rtcore_service_tick_stats_snapshot
    g_rtcore_replay_service_tick_stats_snapshot;
static unsigned g_rtcore_replay_service_tick_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_service_tick_stats_progress_logs_emitted = 0;
static unsigned g_rtcore_replay_service_tick_stats_last_logged_ticks_progressed =
    0;
static unsigned g_rtcore_replay_unit_arbitration_stats_logs_emitted = 0;
static unsigned
    g_rtcore_replay_unit_arbitration_stats_progress_logs_emitted = 0;
static unsigned
    g_rtcore_replay_unit_arbitration_stats_last_logged_total_issued = 0;
static unsigned
    g_rtcore_replay_unit_arbitration_stats_last_logged_total_budget_exhausted =
        0;
static unsigned g_rtcore_compact_trace_overflow_stats_logs_emitted = 0;
static unsigned g_rtcore_next_replay_ready_order = 0;
static unsigned g_rtcore_replay_round_robin_cursor = 0;

static bool rtcore_bounded_trace_collection_enabled()
{
    static int enabled = []() {
        const char *value = getenv("VULKAN_SIM_RTCORE_BOUNDED_TRACE_COLLECTION");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static unsigned rtcore_compact_trace_events_per_lane_config()
{
    static unsigned events_per_lane = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_COMPACT_TRACE_EVENTS_PER_LANE");
        if (!value || value[0] == '\0') {
            return RTCORE_COMPACT_TRACE_DEFAULT_EVENTS_PER_LANE;
        }

        char *end = NULL;
        unsigned long parsed = strtoul(value, &end, 10);
        if (end == value || parsed == 0) {
            return RTCORE_COMPACT_TRACE_DEFAULT_EVENTS_PER_LANE;
        }
        if (parsed > RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR) {
            return RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR;
        }
        return static_cast<unsigned>(parsed);
    }();
    return events_per_lane;
}

static bool rtcore_replay_admission_enabled()
{
    static int enabled = []() {
        const char *value = getenv("VULKAN_SIM_RTCORE_REPLAY_ADMISSION");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_compact_trace_overflow_stats_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_COMPACT_TRACE_OVERFLOW_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_service_tick_enabled()
{
    static int enabled = []() {
        const char *value = getenv("VULKAN_SIM_RTCORE_REPLAY_SERVICE_TICK");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_service_tick_stats_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_SERVICE_TICK_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_unit_arbitration_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_UNIT_ARBITRATION");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_unit_arbitration_stats_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_UNIT_ARBITRATION_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_warp_completion_aggregation_shadow_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_WARP_COMPLETION_AGGREGATION_SHADOW");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static unsigned rtcore_replay_service_tick_stats_log_limit_from_env(
    const char *name, unsigned default_limit)
{
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return default_limit;
    }

    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value) {
        return default_limit;
    }
    if (parsed > 1024) {
        return 1024;
    }
    return static_cast<unsigned>(parsed);
}

static unsigned rtcore_replay_service_tick_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_SERVICE_TICK_STATS_LOG_LIMIT", 8);
    return limit;
}

static unsigned rtcore_replay_service_tick_stats_progress_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_SERVICE_TICK_STATS_PROGRESS_LOG_LIMIT", 8);
    return limit;
}

static unsigned rtcore_replay_unit_arbitration_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_UNIT_ARBITRATION_STATS_LOG_LIMIT", 8);
    return limit;
}

static unsigned rtcore_replay_unit_arbitration_stats_progress_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_UNIT_ARBITRATION_STATS_PROGRESS_LOG_LIMIT",
        8);
    return limit;
}

static unsigned rtcore_compact_trace_overflow_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_COMPACT_TRACE_OVERFLOW_STATS_LOG_LIMIT", 16);
    return limit;
}

static rtcore_replay_ready_selection_policy
rtcore_replay_ready_selection_policy()
{
    static int policy = []() {
        const char *value = getenv("VULKAN_SIM_RTCORE_REPLAY_READY_SELECTION");
        if (value && strcmp(value, "round_robin_ready_order") == 0) {
            return static_cast<int>(
                RTCORE_REPLAY_SELECT_ROUND_ROBIN_READY_ORDER);
        }
        if (value && strcmp(value, "oldest_ready_first") == 0) {
            return static_cast<int>(RTCORE_REPLAY_SELECT_OLDEST_READY_FIRST);
        }
        return static_cast<int>(RTCORE_REPLAY_SELECT_OLDEST_READY_FIRST);
    }();
    return static_cast<enum rtcore_replay_ready_selection_policy>(policy);
}

static unsigned rtcore_replay_issue_budget_from_env(const char *name,
                                                    unsigned default_budget)
{
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return default_budget;
    }

    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value) {
        return default_budget;
    }
    if (parsed > 32) {
        return 32;
    }
    return static_cast<unsigned>(parsed);
}

static rtcore_replay_issue_budget rtcore_replay_issue_budget_config()
{
    static rtcore_replay_issue_budget budget = []() {
        rtcore_replay_issue_budget parsed = {};
        parsed.node_issue_budget = rtcore_replay_issue_budget_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_NODE_ISSUE_BUDGET", 1);
        parsed.primitive_issue_budget = rtcore_replay_issue_budget_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_PRIMITIVE_ISSUE_BUDGET", 1);
        parsed.stack_issue_budget = rtcore_replay_issue_budget_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_STACK_ISSUE_BUDGET", 1);
        parsed.completion_issue_budget = rtcore_replay_issue_budget_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_COMPLETION_ISSUE_BUDGET", 1);
        return parsed;
    }();
    return budget;
}

static unsigned rtcore_replay_memory_wake_budget_config()
{
    static unsigned budget = rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_WAKE_BUDGET", 1);
    return budget;
}

static bool rtcore_replay_issue_budget_available(
    const rtcore_replay_issue_budget &budget)
{
    return budget.node_issue_budget || budget.primitive_issue_budget ||
           budget.stack_issue_budget || budget.completion_issue_budget;
}

static uint32_t rtcore_pack_compact_trace_fields(
    unsigned lane_id, rtcore_compact_trace_event_type event_type,
    rtcore_compact_trace_resource_class resource_class, unsigned flags)
{
    return ((lane_id & 0xffu) << 24) |
           ((static_cast<unsigned>(event_type) & 0xffu) << 16) |
           ((static_cast<unsigned>(resource_class) & 0xffu) << 8) |
           (flags & 0xffu);
}

static uint16_t rtcore_pack_compact_trace_count_bytes(unsigned count,
                                                      unsigned bytes)
{
    unsigned compact_count = count > 255 ? 255 : count;
    unsigned compact_bytes = bytes > 255 ? 255 : bytes;
    return static_cast<uint16_t>((compact_bytes << 8) | compact_count);
}

static uint16_t rtcore_saturate_u16(unsigned value)
{
    return static_cast<uint16_t>(value > 0xffffu ? 0xffffu : value);
}

static rtcore_compact_trace_event_type rtcore_unpack_compact_trace_event_type(
    const rtcore_compact_trace_event &event)
{
    return static_cast<rtcore_compact_trace_event_type>(
        (event.packed_fields >> 16) & 0xffu);
}

static rtcore_compact_trace_resource_class
rtcore_unpack_compact_trace_resource_class(
    const rtcore_compact_trace_event &event)
{
    return static_cast<rtcore_compact_trace_resource_class>(
        (event.packed_fields >> 8) & 0xffu);
}

static unsigned rtcore_unpack_compact_trace_count(
    const rtcore_compact_trace_event &event)
{
    return event.packed_count_bytes & 0xffu;
}

static unsigned rtcore_unpack_compact_trace_bytes(
    const rtcore_compact_trace_event &event)
{
    return (event.packed_count_bytes >> 8) & 0xffu;
}

static const char *rtcore_trace_timing_precision_class_name(
    rtcore_trace_timing_precision_class precision_class)
{
    switch (precision_class) {
    case RTCORE_TRACE_TIMING_PRECISION_EXACT:
        return "exact";
    case RTCORE_TRACE_TIMING_PRECISION_BOUNDED_OVERFLOW_SUMMARY:
        return "bounded_overflow_summary";
    default:
        return "unknown";
    }
}

static uint64_t rtcore_pack_overflow_summary_address(
    const rtcore_compact_trace_overflow_summary &summary)
{
    return static_cast<uint64_t>(
               rtcore_saturate_u16(summary.overflow_node_fetch_count)) |
           (static_cast<uint64_t>(
                rtcore_saturate_u16(summary.overflow_node_test_count))
            << 16) |
           (static_cast<uint64_t>(
                rtcore_saturate_u16(summary.overflow_primitive_fetch_count))
            << 32) |
           (static_cast<uint64_t>(
                rtcore_saturate_u16(summary.overflow_primitive_test_count))
            << 48);
}

static unsigned rtcore_overflow_summary_total_count(
    const rtcore_compact_trace_overflow_summary &summary)
{
    return summary.overflow_node_fetch_count +
           summary.overflow_node_test_count +
           summary.overflow_primitive_fetch_count +
           summary.overflow_primitive_test_count +
           summary.overflow_stack_push_count + summary.overflow_stack_pop_count +
           summary.overflow_memory_wait_count +
           summary.overflow_completion_count;
}

static rtcore_replay_lane_request_state rtcore_classify_replay_state(
    rtcore_compact_trace_event_type event_type)
{
    switch (event_type) {
    case RTCORE_TRACE_NODE_FETCH:
    case RTCORE_TRACE_NODE_TEST:
        return RTCORE_REPLAY_READY_NODE;
    case RTCORE_TRACE_PRIMITIVE_FETCH:
    case RTCORE_TRACE_PRIMITIVE_TEST:
    case RTCORE_TRACE_HIT_UPDATE:
        return RTCORE_REPLAY_READY_PRIMITIVE;
    case RTCORE_TRACE_STACK_PUSH:
    case RTCORE_TRACE_STACK_POP:
        return RTCORE_REPLAY_READY_STACK;
    case RTCORE_TRACE_MEMORY_WAIT:
        return RTCORE_REPLAY_WAITING_MEMORY;
    case RTCORE_TRACE_COMPLETION:
    case RTCORE_TRACE_OVERFLOW_SUMMARY:
        return RTCORE_REPLAY_READY_COMPLETION;
    default:
        return RTCORE_REPLAY_DONE;
    }
}

static unsigned rtcore_trace_node_fetch_flags(
    bool top_level, rtcore_compact_trace_node_kind node_kind)
{
    unsigned level_flag = top_level ? 0x1u : 0x2u;
    return level_flag | ((static_cast<unsigned>(node_kind) & 0x7u) << 2);
}

static unsigned rtcore_trace_node_test_flags(unsigned child_index, bool hit,
                                             bool top_level)
{
    return (hit ? 0x1u : 0u) | (top_level ? 0x2u : 0u) |
           ((child_index & 0x7u) << 2);
}

static unsigned rtcore_trace_stack_flags(bool top_level, bool leaf,
                                         bool clear)
{
    return (top_level ? 0x1u : 0x2u) | (leaf ? 0x4u : 0u) |
           (clear ? 0x8u : 0u);
}

static unsigned rtcore_trace_primitive_flags(
    rtcore_compact_trace_primitive_kind primitive_kind, bool hit,
    bool deferred)
{
    return (static_cast<unsigned>(primitive_kind) & 0x0fu) |
           (hit ? 0x10u : 0u) | (deferred ? 0x20u : 0u);
}

static unsigned rtcore_trace_hit_update_flags(
    rtcore_compact_trace_hit_update_kind hit_update_kind)
{
    return static_cast<unsigned>(hit_update_kind) & 0xffu;
}

struct rtcore_bounded_trace_collector {
    bool enabled;
    unsigned lane_id;
    unsigned max_trace_events_per_lane;
    unsigned next_event_seq;
    bool timing_trace_overflowed;
    rtcore_trace_timing_precision_class timing_precision_class;
    unsigned overflow_summary_events;
    rtcore_compact_trace_overflow_summary overflow_summary;
    bool has_overflow_summary_event;
    unsigned overflow_summary_event_index;
    std::vector<rtcore_compact_trace_event> events;

    explicit rtcore_bounded_trace_collector(ptx_thread_info *thread)
        : enabled(rtcore_bounded_trace_collection_enabled()),
          lane_id(thread ? (thread->get_tid().x & 31u) : 0),
          max_trace_events_per_lane(
              rtcore_compact_trace_events_per_lane_config()),
          next_event_seq(0), timing_trace_overflowed(false),
          timing_precision_class(RTCORE_TRACE_TIMING_PRECISION_EXACT),
          overflow_summary_events(0), overflow_summary(),
          has_overflow_summary_event(false), overflow_summary_event_index(0)
    {
        if (enabled) {
            if (max_trace_events_per_lane >
                RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR) {
                max_trace_events_per_lane =
                    RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR;
            }
            events.reserve(max_trace_events_per_lane);
        }
    }

    void record_overflow_event(rtcore_compact_trace_event_type event_type,
                               rtcore_compact_trace_resource_class resource_class,
                               unsigned bytes, unsigned count)
    {
        const unsigned effective_count = count == 0 ? 1 : count;
        switch (event_type) {
        case RTCORE_TRACE_NODE_FETCH:
            overflow_summary.overflow_node_fetch_count += effective_count;
            break;
        case RTCORE_TRACE_NODE_TEST:
            overflow_summary.overflow_node_test_count += effective_count;
            break;
        case RTCORE_TRACE_PRIMITIVE_FETCH:
            overflow_summary.overflow_primitive_fetch_count += effective_count;
            break;
        case RTCORE_TRACE_PRIMITIVE_TEST:
            overflow_summary.overflow_primitive_test_count += effective_count;
            break;
        case RTCORE_TRACE_STACK_PUSH:
            overflow_summary.overflow_stack_push_count += effective_count;
            break;
        case RTCORE_TRACE_STACK_POP:
            overflow_summary.overflow_stack_pop_count += effective_count;
            break;
        case RTCORE_TRACE_MEMORY_WAIT:
            overflow_summary.overflow_memory_wait_count += effective_count;
            break;
        case RTCORE_TRACE_HIT_UPDATE:
        case RTCORE_TRACE_COMPLETION:
            overflow_summary.overflow_completion_count += effective_count;
            break;
        default:
            break;
        }
        if (resource_class == RTCORE_TRACE_RESOURCE_MEMORY ||
            event_type == RTCORE_TRACE_NODE_FETCH ||
            event_type == RTCORE_TRACE_PRIMITIVE_FETCH) {
            overflow_summary.overflow_memory_bytes += bytes * effective_count;
        }
    }

    void record_overflow_event(const rtcore_compact_trace_event &event)
    {
        record_overflow_event(
            rtcore_unpack_compact_trace_event_type(event),
            rtcore_unpack_compact_trace_resource_class(event),
            rtcore_unpack_compact_trace_bytes(event),
            rtcore_unpack_compact_trace_count(event));
    }

    rtcore_compact_trace_event make_overflow_summary_event(
        uint16_t event_seq) const
    {
        rtcore_compact_trace_event event = {};
        event.address_or_ref =
            rtcore_pack_overflow_summary_address(overflow_summary);
        event.packed_fields = rtcore_pack_compact_trace_fields(
            lane_id, RTCORE_TRACE_OVERFLOW_SUMMARY,
            RTCORE_TRACE_RESOURCE_SUMMARY, 0);
        event.event_seq = event_seq;
        event.packed_count_bytes = rtcore_pack_compact_trace_count_bytes(
            rtcore_overflow_summary_total_count(overflow_summary),
            overflow_summary.overflow_memory_bytes);
        return event;
    }

    void append_or_update_overflow_summary()
    {
        timing_trace_overflowed = true;
        timing_precision_class =
            RTCORE_TRACE_TIMING_PRECISION_BOUNDED_OVERFLOW_SUMMARY;
        overflow_summary_events = 1;

        if (has_overflow_summary_event &&
            overflow_summary_event_index < events.size()) {
            const uint16_t event_seq =
                events[overflow_summary_event_index].event_seq;
            events[overflow_summary_event_index] =
                make_overflow_summary_event(event_seq);
            return;
        }

        if (events.size() < max_trace_events_per_lane) {
            overflow_summary_event_index = events.size();
            has_overflow_summary_event = true;
            events.push_back(make_overflow_summary_event(
                static_cast<uint16_t>(next_event_seq++)));
            return;
        }

        if (!events.empty()) {
            overflow_summary_event_index = events.size() - 1;
            record_overflow_event(events[overflow_summary_event_index]);
            has_overflow_summary_event = true;
            const uint16_t event_seq =
                events[overflow_summary_event_index].event_seq;
            events[overflow_summary_event_index] =
                make_overflow_summary_event(event_seq);
        }
    }

    void append(rtcore_compact_trace_event_type event_type,
                rtcore_compact_trace_resource_class resource_class,
                uint64_t address_or_ref, unsigned bytes, unsigned count,
                unsigned flags)
    {
        if (!enabled) {
            return;
        }
        if (events.size() >= max_trace_events_per_lane) {
            record_overflow_event(event_type, resource_class, bytes, count);
            append_or_update_overflow_summary();
            return;
        }

        rtcore_compact_trace_event event = {};
        event.address_or_ref = address_or_ref;
        event.packed_fields =
            rtcore_pack_compact_trace_fields(lane_id, event_type,
                                             resource_class, flags);
        event.event_seq = static_cast<uint16_t>(next_event_seq++);
        event.packed_count_bytes =
            rtcore_pack_compact_trace_count_bytes(count, bytes);
        events.push_back(event);
    }

    void append_node_fetch(uint64_t address, unsigned bytes, unsigned flags)
    {
        append(RTCORE_TRACE_NODE_FETCH, RTCORE_TRACE_RESOURCE_NODE, address,
               bytes, 1, flags);
    }

    void append_node_test(uint64_t parent_node_address, unsigned child_index,
                          bool hit, bool top_level)
    {
        append(RTCORE_TRACE_NODE_TEST, RTCORE_TRACE_RESOURCE_NODE,
               parent_node_address, 0, 1,
               rtcore_trace_node_test_flags(child_index, hit, top_level));
    }

    void append_stack_push(uint64_t entry_address, bool top_level, bool leaf)
    {
        append(RTCORE_TRACE_STACK_PUSH, RTCORE_TRACE_RESOURCE_STACK,
               entry_address, 0, 1,
               rtcore_trace_stack_flags(top_level, leaf, false));
    }

    void append_stack_pop(uint64_t entry_address, bool top_level, bool leaf)
    {
        append(RTCORE_TRACE_STACK_POP, RTCORE_TRACE_RESOURCE_STACK,
               entry_address, 0, 1,
               rtcore_trace_stack_flags(top_level, leaf, false));
    }

    void append_stack_clear(unsigned entries)
    {
        append(RTCORE_TRACE_STACK_POP, RTCORE_TRACE_RESOURCE_STACK, 0, 0,
               entries, rtcore_trace_stack_flags(false, false, true));
    }

    void append_primitive_fetch(uint64_t address, unsigned bytes,
                                unsigned flags)
    {
        append(RTCORE_TRACE_PRIMITIVE_FETCH,
               RTCORE_TRACE_RESOURCE_PRIMITIVE, address, bytes, 1, flags);
    }

    void append_primitive_test(uint64_t address_or_ref, bool hit,
                               unsigned flags)
    {
        append(RTCORE_TRACE_PRIMITIVE_TEST,
               RTCORE_TRACE_RESOURCE_PRIMITIVE, address_or_ref, 0, 1,
               flags | (hit ? 0x10u : 0u));
    }

    void append_hit_update(uint64_t address_or_ref, unsigned hit_count,
                           unsigned flags)
    {
        append(RTCORE_TRACE_HIT_UPDATE, RTCORE_TRACE_RESOURCE_COMPLETION,
               address_or_ref, 0, hit_count, flags);
    }

    void append_completion_summary(unsigned node_events,
                                   unsigned primitive_events)
    {
        append(RTCORE_TRACE_COMPLETION, RTCORE_TRACE_RESOURCE_COMPLETION, 0, 0,
               node_events + primitive_events, 0);
    }

    const char *model_name() const { return RTCORE_TRACE_REPLAY_MODEL_NAME; }

    rtcore_compact_trace_export_record export_record() const
    {
        rtcore_compact_trace_export_record record = {};
        record.valid = enabled;
        record.model_name = RTCORE_TRACE_REPLAY_MODEL_NAME;
        record.thread_uid = 0;
        record.owner_hw_sid = 0;
        record.lane_id = lane_id;
        record.has_warp_metadata = false;
        record.warp_uid = 0;
        record.warp_id = 0;
        record.active_mask = 0;
        record.static_inst_uid = 0;
        record.event_count = events.size();
        record.max_trace_events_per_lane = max_trace_events_per_lane;
        record.timing_trace_overflowed = timing_trace_overflowed;
        record.timing_precision_class = timing_precision_class;
        record.overflow_summary_events = overflow_summary_events;
        record.overflow_summary = overflow_summary;
        if (enabled) {
            record.events = events;
        }
        return record;
    }
};

static void rtcore_maybe_log_compact_trace_overflow_summary(
    const rtcore_compact_trace_export_record &record)
{
    if (!rtcore_compact_trace_overflow_stats_log_enabled() ||
        !record.timing_trace_overflowed) {
        return;
    }
    if (g_rtcore_compact_trace_overflow_stats_logs_emitted >=
        rtcore_compact_trace_overflow_stats_log_limit()) {
        return;
    }
    g_rtcore_compact_trace_overflow_stats_logs_emitted++;

    printf("GPGPU-Sim RTCORE_COMPACT_TRACE_OVERFLOW_SUMMARY "
           "thread_uid=%u owner_hw_sid=%u lane_id=%u "
           "event_count=%u max_trace_events_per_lane=%u "
           "overflow_summary_events=%u "
           "overflow_node_fetch_count=%u overflow_node_test_count=%u "
           "overflow_primitive_fetch_count=%u "
           "overflow_primitive_test_count=%u "
           "overflow_stack_push_count=%u overflow_stack_pop_count=%u "
           "overflow_memory_wait_count=%u overflow_memory_bytes=%u "
           "overflow_completion_count=%u timing_precision_class=%s\n",
           record.thread_uid, record.owner_hw_sid, record.lane_id,
           record.event_count, record.max_trace_events_per_lane,
           record.overflow_summary_events,
           record.overflow_summary.overflow_node_fetch_count,
           record.overflow_summary.overflow_node_test_count,
           record.overflow_summary.overflow_primitive_fetch_count,
           record.overflow_summary.overflow_primitive_test_count,
           record.overflow_summary.overflow_stack_push_count,
           record.overflow_summary.overflow_stack_pop_count,
           record.overflow_summary.overflow_memory_wait_count,
           record.overflow_summary.overflow_memory_bytes,
           record.overflow_summary.overflow_completion_count,
           rtcore_trace_timing_precision_class_name(
               record.timing_precision_class));
    fflush(stdout);
}

static void rtcore_publish_compact_trace_export(
    ptx_thread_info *thread, const rtcore_compact_trace_export_record &record)
{
    if (!record.valid) {
        return;
    }
    rtcore_compact_trace_export_record stored = record;
    stored.thread_uid = thread ? thread->get_uid() : 0;
    stored.owner_hw_sid = thread ? thread->get_hw_sid() : 0;
    ptx_thread_info::rtcore_current_warp_metadata warp_metadata;
    if (thread && thread->get_rtcore_current_warp_metadata(&warp_metadata)) {
        stored.has_warp_metadata = true;
        stored.warp_uid = warp_metadata.warp_uid;
        stored.warp_id = warp_metadata.warp_id;
        stored.active_mask = warp_metadata.active_mask;
        stored.static_inst_uid = warp_metadata.static_inst_uid;
    }
    g_rtcore_compact_trace_exports[stored.thread_uid] = stored;
    rtcore_maybe_log_compact_trace_overflow_summary(stored);
}

static bool rtcore_get_compact_trace_export(
    unsigned thread_uid, rtcore_compact_trace_export_record *record)
{
    std::map<unsigned, rtcore_compact_trace_export_record>::const_iterator it =
        g_rtcore_compact_trace_exports.find(thread_uid);
    if (it == g_rtcore_compact_trace_exports.end()) {
        return false;
    }
    if (record) {
        *record = it->second;
    }
    return true;
}

static rtcore_replay_lane_request rtcore_build_replay_lane_request(
    const rtcore_compact_trace_export_record &record)
{
    rtcore_replay_lane_request request = {};
    request.valid = record.valid;
    request.thread_uid = record.thread_uid;
    request.owner_hw_sid = record.owner_hw_sid;
    request.lane_id = record.lane_id;
    request.has_warp_metadata = record.has_warp_metadata;
    request.warp_uid = record.warp_uid;
    request.warp_id = record.warp_id;
    request.active_mask = record.active_mask;
    request.static_inst_uid = record.static_inst_uid;
    request.next_event_index = 0;
    request.event_count = record.event_count;
    request.timing_trace_overflowed = record.timing_trace_overflowed;
    request.state = RTCORE_REPLAY_DONE;
    request.events = record.events;

    for (unsigned i = 0; i < request.events.size(); ++i) {
        rtcore_compact_trace_event_type event_type =
            rtcore_unpack_compact_trace_event_type(request.events[i]);
        switch (rtcore_classify_replay_state(event_type)) {
        case RTCORE_REPLAY_READY_NODE:
            request.node_event_count++;
            break;
        case RTCORE_REPLAY_READY_PRIMITIVE:
            request.primitive_event_count++;
            break;
        case RTCORE_REPLAY_READY_STACK:
            request.stack_event_count++;
            break;
        case RTCORE_REPLAY_WAITING_MEMORY:
            request.memory_event_count++;
            break;
        case RTCORE_REPLAY_READY_COMPLETION:
            request.completion_event_count++;
            break;
        default:
            break;
        }
    }

    if (!request.events.empty()) {
        request.state = rtcore_classify_replay_state(
            rtcore_unpack_compact_trace_event_type(request.events[0]));
    } else if (request.valid) {
        request.state = RTCORE_REPLAY_QUEUED;
    }
    return request;
}

static void rtcore_enqueue_replay_request_by_state(
    const rtcore_replay_lane_request &request)
{
    if (!request.valid) {
        return;
    }

    switch (request.state) {
    case RTCORE_REPLAY_QUEUED:
        g_rtcore_replay_ready_queues.queued_queue.push_back(request.thread_uid);
        break;
    case RTCORE_REPLAY_READY_NODE:
        g_rtcore_replay_ready_queues.ready_node_queue.push_back(
            request.thread_uid);
        break;
    case RTCORE_REPLAY_READY_PRIMITIVE:
        g_rtcore_replay_ready_queues.ready_primitive_queue.push_back(
            request.thread_uid);
        break;
    case RTCORE_REPLAY_READY_STACK:
        g_rtcore_replay_ready_queues.ready_stack_queue.push_back(
            request.thread_uid);
        break;
    case RTCORE_REPLAY_WAITING_MEMORY:
        g_rtcore_replay_ready_queues.waiting_memory_queue.push_back(
            request.thread_uid);
        break;
    case RTCORE_REPLAY_READY_COMPLETION:
        g_rtcore_replay_ready_queues.ready_completion_queue.push_back(
            request.thread_uid);
        break;
    case RTCORE_REPLAY_DONE:
        g_rtcore_replay_ready_queues.done_queue.push_back(request.thread_uid);
        break;
    }
}

static bool rtcore_route_admitted_replay_request(unsigned thread_uid)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    rtcore_enqueue_replay_request_by_state(it->second);
    return true;
}

static bool rtcore_get_replay_request_ready_order(unsigned thread_uid,
                                                  unsigned *ready_order)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    if (ready_order) {
        *ready_order = it->second.ready_order;
    }
    return true;
}

static bool rtcore_replay_request_owned_by_sm(unsigned thread_uid,
                                              unsigned owner_hw_sid)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }

    const rtcore_replay_lane_request &request = it->second;
    return request.owner_hw_sid == owner_hw_sid;
}

static rtcore_replay_service_cycle_identity_snapshot
rtcore_make_replay_service_progress_identity(unsigned thread_uid,
                                             bool memory_progressed,
                                             bool ready_progressed)
{
    rtcore_replay_service_cycle_identity_snapshot snapshot = {};
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return snapshot;
    }

    const rtcore_replay_lane_request &request = it->second;
    snapshot.valid = true;
    snapshot.memory_progressed = memory_progressed;
    snapshot.ready_progressed = ready_progressed;
    snapshot.owner_hw_sid = request.owner_hw_sid;
    snapshot.thread_uid = request.thread_uid;
    snapshot.lane_id = request.lane_id;
    snapshot.has_warp_metadata = request.has_warp_metadata;
    snapshot.warp_uid = request.warp_uid;
    snapshot.warp_id = request.warp_id;
    snapshot.active_mask = request.active_mask;
    snapshot.static_inst_uid = request.static_inst_uid;
    return snapshot;
}

static unsigned rtcore_count_replay_warp_completion_shadow_lanes(unsigned mask)
{
    unsigned count = 0;
    while (mask != 0) {
        count += mask & 1u;
        mask >>= 1;
    }
    return count;
}

static bool rtcore_replay_request_has_warp_completion_shadow_metadata(
    const rtcore_replay_lane_request &request)
{
    return request.valid && request.has_warp_metadata && request.lane_id < 32 &&
           request.active_mask != 0;
}

static rtcore_replay_warp_completion_shadow_key
rtcore_make_replay_warp_completion_shadow_key(
    const rtcore_replay_lane_request &request)
{
    rtcore_replay_warp_completion_shadow_key key = {};
    key.owner_hw_sid = request.owner_hw_sid;
    key.warp_uid = request.warp_uid;
    key.warp_id = request.warp_id;
    key.active_mask = request.active_mask;
    return key;
}

static void rtcore_update_replay_warp_completion_shadow_state(
    rtcore_replay_warp_completion_shadow_state *state)
{
    if (!state || !state->valid) {
        return;
    }

    const unsigned active_completed_lanes =
        state->completed_lane_mask & state->key.active_mask;
    state->completed_lane_count =
        rtcore_count_replay_warp_completion_shadow_lanes(
            active_completed_lanes);
    state->all_active_lanes_complete =
        state->key.active_mask != 0 &&
        active_completed_lanes == state->key.active_mask;
}

static void rtcore_log_replay_warp_completion_aggregation_shadow(
    const rtcore_replay_warp_completion_shadow_state &state)
{
    printf("GPGPU-Sim RTCORE_REPLAY_WARP_COMPLETION_AGGREGATION_SHADOW "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
           "admitted_lane_mask=0x%08x completed_lane_mask=0x%08x "
           "completed_lane_count=%u all_active_lanes_complete=%u\n",
           state.key.owner_hw_sid, state.key.warp_uid, state.key.warp_id,
           state.key.active_mask, state.admitted_lane_mask,
           state.completed_lane_mask, state.completed_lane_count,
           state.all_active_lanes_complete ? 1 : 0);
    fflush(stdout);
}

static void rtcore_record_replay_lane_admission_shadow(
    const rtcore_replay_lane_request &request)
{
    if (!rtcore_replay_warp_completion_aggregation_shadow_enabled() ||
        !rtcore_replay_request_has_warp_completion_shadow_metadata(request)) {
        return;
    }

    const rtcore_replay_warp_completion_shadow_key key =
        rtcore_make_replay_warp_completion_shadow_key(request);
    rtcore_replay_warp_completion_shadow_state &state =
        g_rtcore_replay_warp_completion_shadow[key];
    if (!state.valid) {
        state.valid = true;
        state.key = key;
    }
    state.admitted_lane_mask |= 1u << request.lane_id;
    rtcore_update_replay_warp_completion_shadow_state(&state);
}

static void rtcore_record_replay_lane_completion_shadow(
    const rtcore_replay_lane_request &request)
{
    if (!rtcore_replay_warp_completion_aggregation_shadow_enabled() ||
        !rtcore_replay_request_has_warp_completion_shadow_metadata(request)) {
        return;
    }

    const rtcore_replay_warp_completion_shadow_key key =
        rtcore_make_replay_warp_completion_shadow_key(request);
    rtcore_replay_warp_completion_shadow_state &state =
        g_rtcore_replay_warp_completion_shadow[key];
    if (!state.valid) {
        state.valid = true;
        state.key = key;
    }
    state.admitted_lane_mask |= 1u << request.lane_id;
    state.completed_lane_mask |= 1u << request.lane_id;
    rtcore_update_replay_warp_completion_shadow_state(&state);
    if (state.all_active_lanes_complete &&
        !state.all_active_lanes_complete_logged) {
        state.all_active_lanes_complete_logged = true;
        rtcore_log_replay_warp_completion_aggregation_shadow(state);
    }
}

static bool rtcore_record_replay_lane_completion_shadow(unsigned thread_uid)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }

    rtcore_record_replay_lane_completion_shadow(it->second);
    return true;
}

static bool rtcore_consider_oldest_ready_queue_front(
    const std::deque<unsigned> &queue, bool *has_selection,
    unsigned *selected_thread_uid, unsigned *selected_order)
{
    if (queue.empty()) {
        return false;
    }

    unsigned candidate_order = 0;
    if (!rtcore_get_replay_request_ready_order(queue.front(),
                                               &candidate_order)) {
        return false;
    }
    if (!has_selection || !selected_thread_uid || !selected_order ||
        !*has_selection || candidate_order < *selected_order) {
        if (has_selection) {
            *has_selection = true;
        }
        if (selected_thread_uid) {
            *selected_thread_uid = queue.front();
        }
        if (selected_order) {
            *selected_order = candidate_order;
        }
        return true;
    }
    return false;
}

static bool rtcore_consider_oldest_ready_queue_for_owner(
    const std::deque<unsigned> &queue, unsigned owner_hw_sid,
    bool *has_selection, unsigned *selected_thread_uid,
    unsigned *selected_order)
{
    if (queue.empty()) {
        return false;
    }

    bool considered = false;
    for (std::deque<unsigned>::const_iterator it = queue.begin();
         it != queue.end(); ++it) {
        unsigned candidate_thread_uid = *it;
        if (!rtcore_replay_request_owned_by_sm(candidate_thread_uid,
                                               owner_hw_sid)) {
            continue;
        }

        unsigned candidate_order = 0;
        if (!rtcore_get_replay_request_ready_order(candidate_thread_uid,
                                                   &candidate_order)) {
            continue;
        }
        if (!has_selection || !selected_thread_uid || !selected_order ||
            !*has_selection || candidate_order < *selected_order) {
            if (has_selection) {
                *has_selection = true;
            }
            if (selected_thread_uid) {
                *selected_thread_uid = candidate_thread_uid;
            }
            if (selected_order) {
                *selected_order = candidate_order;
            }
            considered = true;
        }
    }
    return considered;
}

static bool rtcore_select_oldest_ready_request(unsigned *thread_uid)
{
    bool has_selection = false;
    unsigned selected_thread_uid = 0;
    unsigned selected_order = 0;

    rtcore_consider_oldest_ready_queue_front(
        g_rtcore_replay_ready_queues.ready_node_queue, &has_selection,
        &selected_thread_uid, &selected_order);
    rtcore_consider_oldest_ready_queue_front(
        g_rtcore_replay_ready_queues.ready_primitive_queue,
        &has_selection, &selected_thread_uid, &selected_order);
    rtcore_consider_oldest_ready_queue_front(
        g_rtcore_replay_ready_queues.ready_stack_queue, &has_selection,
        &selected_thread_uid, &selected_order);
    rtcore_consider_oldest_ready_queue_front(
        g_rtcore_replay_ready_queues.ready_completion_queue,
        &has_selection, &selected_thread_uid, &selected_order);

    if (!has_selection) {
        return false;
    }
    if (thread_uid) {
        *thread_uid = selected_thread_uid;
    }
    return true;
}

static bool rtcore_select_oldest_ready_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid)
{
    bool has_selection = false;
    unsigned selected_thread_uid = 0;
    unsigned selected_order = 0;

    rtcore_consider_oldest_ready_queue_for_owner(
        g_rtcore_replay_ready_queues.ready_node_queue, owner_hw_sid,
        &has_selection, &selected_thread_uid, &selected_order);
    rtcore_consider_oldest_ready_queue_for_owner(
        g_rtcore_replay_ready_queues.ready_primitive_queue, owner_hw_sid,
        &has_selection, &selected_thread_uid, &selected_order);
    rtcore_consider_oldest_ready_queue_for_owner(
        g_rtcore_replay_ready_queues.ready_stack_queue, owner_hw_sid,
        &has_selection, &selected_thread_uid, &selected_order);
    rtcore_consider_oldest_ready_queue_for_owner(
        g_rtcore_replay_ready_queues.ready_completion_queue, owner_hw_sid,
        &has_selection, &selected_thread_uid, &selected_order);

    if (!has_selection) {
        return false;
    }
    if (thread_uid) {
        *thread_uid = selected_thread_uid;
    }
    return true;
}

static bool rtcore_ready_queue_front_by_round_robin_slot(unsigned slot,
                                                        unsigned *thread_uid)
{
    const std::deque<unsigned> *queue = NULL;
    switch (slot) {
    case 0:
        queue = &g_rtcore_replay_ready_queues.ready_node_queue;
        break;
    case 1:
        queue = &g_rtcore_replay_ready_queues.ready_primitive_queue;
        break;
    case 2:
        queue = &g_rtcore_replay_ready_queues.ready_stack_queue;
        break;
    case 3:
        queue = &g_rtcore_replay_ready_queues.ready_completion_queue;
        break;
    default:
        return false;
    }

    if (!queue || queue->empty()) {
        return false;
    }
    if (thread_uid) {
        *thread_uid = queue->front();
    }
    return true;
}

static bool rtcore_ready_queue_request_by_round_robin_slot_for_owner(
    unsigned slot, unsigned owner_hw_sid, unsigned *thread_uid)
{
    const std::deque<unsigned> *queue = NULL;
    switch (slot) {
    case 0:
        queue = &g_rtcore_replay_ready_queues.ready_node_queue;
        break;
    case 1:
        queue = &g_rtcore_replay_ready_queues.ready_primitive_queue;
        break;
    case 2:
        queue = &g_rtcore_replay_ready_queues.ready_stack_queue;
        break;
    case 3:
        queue = &g_rtcore_replay_ready_queues.ready_completion_queue;
        break;
    default:
        return false;
    }

    if (!queue || queue->empty()) {
        return false;
    }

    for (std::deque<unsigned>::const_iterator it = queue->begin();
         it != queue->end(); ++it) {
        if (!rtcore_replay_request_owned_by_sm(*it, owner_hw_sid)) {
            continue;
        }
        if (thread_uid) {
            *thread_uid = *it;
        }
        return true;
    }
    return false;
}

static bool rtcore_select_round_robin_ready_request(unsigned *thread_uid)
{
    for (unsigned offset = 0; offset < 4; ++offset) {
        unsigned slot = (g_rtcore_replay_round_robin_cursor + offset) % 4;
        if (rtcore_ready_queue_front_by_round_robin_slot(slot, thread_uid)) {
            g_rtcore_replay_round_robin_cursor = (slot + 1) % 4;
            return true;
        }
    }
    return false;
}

static bool rtcore_select_round_robin_ready_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid)
{
    for (unsigned offset = 0; offset < 4; ++offset) {
        unsigned slot = (g_rtcore_replay_round_robin_cursor + offset) % 4;
        if (rtcore_ready_queue_request_by_round_robin_slot_for_owner(
                slot, owner_hw_sid, thread_uid)) {
            g_rtcore_replay_round_robin_cursor = (slot + 1) % 4;
            return true;
        }
    }
    return false;
}

static bool rtcore_select_ready_replay_request(unsigned *thread_uid)
{
    switch (rtcore_replay_ready_selection_policy()) {
    case RTCORE_REPLAY_SELECT_ROUND_ROBIN_READY_ORDER:
        return rtcore_select_round_robin_ready_request(thread_uid);
    case RTCORE_REPLAY_SELECT_OLDEST_READY_FIRST:
    default:
        return rtcore_select_oldest_ready_request(thread_uid);
    }
}

static bool rtcore_select_ready_replay_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid)
{
    switch (rtcore_replay_ready_selection_policy()) {
    case RTCORE_REPLAY_SELECT_ROUND_ROBIN_READY_ORDER:
        return rtcore_select_round_robin_ready_request_for_owner(owner_hw_sid,
                                                                 thread_uid);
    case RTCORE_REPLAY_SELECT_OLDEST_READY_FIRST:
    default:
        return rtcore_select_oldest_ready_request_for_owner(owner_hw_sid,
                                                            thread_uid);
    }
}

static bool rtcore_remove_replay_request_from_queue(
    std::deque<unsigned> &queue, unsigned thread_uid)
{
    if (queue.empty()) {
        return false;
    }

    if (queue.front() == thread_uid) {
        queue.pop_front();
        return true;
    }

    for (std::deque<unsigned>::iterator it = queue.begin(); it != queue.end();
         ++it) {
        if (*it == thread_uid) {
            queue.erase(it);
            return true;
        }
    }
    return false;
}

static bool rtcore_step_admitted_replay_request(unsigned thread_uid);

static bool rtcore_select_ready_request_from_queue_for_owner(
    const std::deque<unsigned> &queue, unsigned owner_hw_sid,
    unsigned *thread_uid)
{
    if (queue.empty()) {
        return false;
    }

    for (std::deque<unsigned>::const_iterator it = queue.begin();
         it != queue.end(); ++it) {
        if (!rtcore_replay_request_owned_by_sm(*it, owner_hw_sid)) {
            continue;
        }
        if (thread_uid) {
            *thread_uid = *it;
        }
        return true;
    }
    return false;
}

static bool rtcore_service_replay_ready_queue_with_unit_budget_for_owner(
    std::deque<unsigned> &queue, unsigned owner_hw_sid, unsigned *issue_budget,
    unsigned *issue_attempts, unsigned *issued, unsigned *budget_exhausted,
    rtcore_replay_service_cycle_identity_snapshot *last_identity)
{
    bool progressed = false;
    while (true) {
        unsigned thread_uid = 0;
        if (!rtcore_select_ready_request_from_queue_for_owner(
                queue, owner_hw_sid, &thread_uid)) {
            break;
        }
        if (issue_attempts) {
            (*issue_attempts)++;
        }
        if (!issue_budget || *issue_budget == 0) {
            if (budget_exhausted) {
                (*budget_exhausted)++;
            }
            break;
        }

        (*issue_budget)--;
        if (!rtcore_remove_replay_request_from_queue(queue, thread_uid)) {
            break;
        }
        if (!rtcore_step_admitted_replay_request(thread_uid)) {
            rtcore_route_admitted_replay_request(thread_uid);
            break;
        }

        if (issued) {
            (*issued)++;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, false, true);
        }
        progressed = true;
        if (!rtcore_route_admitted_replay_request(thread_uid)) {
            break;
        }
    }
    return progressed;
}

static bool rtcore_dequeue_selected_ready_request(unsigned thread_uid)
{
    if (rtcore_remove_replay_request_from_queue(
            g_rtcore_replay_ready_queues.ready_node_queue, thread_uid)) {
        return true;
    }
    if (rtcore_remove_replay_request_from_queue(
            g_rtcore_replay_ready_queues.ready_primitive_queue, thread_uid)) {
        return true;
    }
    if (rtcore_remove_replay_request_from_queue(
            g_rtcore_replay_ready_queues.ready_stack_queue, thread_uid)) {
        return true;
    }
    if (rtcore_remove_replay_request_from_queue(
            g_rtcore_replay_ready_queues.ready_completion_queue, thread_uid)) {
        return true;
    }
    return false;
}

static void rtcore_try_service_replay_after_admission(unsigned owner_hw_sid);

static void rtcore_admit_compact_trace_for_replay(ptx_thread_info *thread)
{
    if (!rtcore_replay_admission_enabled() || !thread) {
        return;
    }

    rtcore_compact_trace_export_record record = {};
    if (!rtcore_get_compact_trace_export(thread->get_uid(), &record)) {
        return;
    }
    if (!record.valid) {
        return;
    }

    rtcore_replay_lane_request request =
        rtcore_build_replay_lane_request(record);
    request.ready_order = g_rtcore_next_replay_ready_order++;
    g_rtcore_replay_lane_requests[request.thread_uid] = request;
    rtcore_record_replay_lane_admission_shadow(request);
    if (request.state == RTCORE_REPLAY_DONE) {
        rtcore_record_replay_lane_completion_shadow(request);
    }
    rtcore_route_admitted_replay_request(request.thread_uid);
    rtcore_try_service_replay_after_admission(thread->get_hw_sid());
}

static bool rtcore_replay_request_done(
    const rtcore_replay_lane_request &request)
{
    return !request.valid || request.state == RTCORE_REPLAY_DONE ||
           request.next_event_index >= request.events.size();
}

static bool rtcore_replay_advance_lane_request(
    rtcore_replay_lane_request *request)
{
    if (!request || !request->valid) {
        return false;
    }
    if (request->events.empty()) {
        request->state = RTCORE_REPLAY_DONE;
        return false;
    }
    if (request->next_event_index >= request->events.size()) {
        request->state = RTCORE_REPLAY_DONE;
        return false;
    }

    request->next_event_index++;
    if (request->next_event_index >= request->events.size()) {
        request->state = RTCORE_REPLAY_DONE;
        return true;
    }

    request->state = rtcore_classify_replay_state(
        rtcore_unpack_compact_trace_event_type(
            request->events[request->next_event_index]));
    return true;
}

static bool rtcore_step_admitted_replay_request(unsigned thread_uid)
{
    std::map<unsigned, rtcore_replay_lane_request>::iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    if (rtcore_replay_request_done(it->second)) {
        return false;
    }
    const bool advanced = rtcore_replay_advance_lane_request(&it->second);
    if (advanced && rtcore_replay_request_done(it->second)) {
        rtcore_record_replay_lane_completion_shadow(it->second.thread_uid);
    }
    return advanced;
}

static bool rtcore_consume_replay_issue_budget_for_state(
    rtcore_replay_lane_request_state state, rtcore_replay_issue_budget *budget)
{
    if (!budget) {
        return false;
    }

    switch (state) {
    case RTCORE_REPLAY_READY_NODE:
        if (budget->node_issue_budget == 0) {
            return false;
        }
        budget->node_issue_budget--;
        return true;
    case RTCORE_REPLAY_READY_PRIMITIVE:
        if (budget->primitive_issue_budget == 0) {
            return false;
        }
        budget->primitive_issue_budget--;
        return true;
    case RTCORE_REPLAY_READY_STACK:
        if (budget->stack_issue_budget == 0) {
            return false;
        }
        budget->stack_issue_budget--;
        return true;
    case RTCORE_REPLAY_READY_COMPLETION:
        if (budget->completion_issue_budget == 0) {
            return false;
        }
        budget->completion_issue_budget--;
        return true;
    default:
        return false;
    }
}

static bool rtcore_consume_replay_issue_budget(unsigned thread_uid,
                                               rtcore_replay_issue_budget *budget)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    return rtcore_consume_replay_issue_budget_for_state(it->second.state,
                                                        budget);
}

static bool rtcore_step_selected_ready_replay_request_with_budget(
    rtcore_replay_issue_budget *budget, unsigned *serviced_thread_uid = NULL)
{
    unsigned thread_uid = 0;
    if (!rtcore_select_ready_replay_request(&thread_uid)) {
        return false;
    }
    if (!rtcore_consume_replay_issue_budget(thread_uid, budget)) {
        return false;
    }
    if (!rtcore_dequeue_selected_ready_request(thread_uid)) {
        return false;
    }
    if (!rtcore_step_admitted_replay_request(thread_uid)) {
        return false;
    }
    if (serviced_thread_uid) {
        *serviced_thread_uid = thread_uid;
    }
    return rtcore_route_admitted_replay_request(thread_uid);
}

static bool rtcore_step_selected_ready_replay_request_with_budget_for_owner(
    unsigned owner_hw_sid, rtcore_replay_issue_budget *budget,
    unsigned *serviced_thread_uid = NULL)
{
    unsigned thread_uid = 0;
    if (!rtcore_select_ready_replay_request_for_owner(owner_hw_sid, &thread_uid)) {
        return false;
    }
    if (!rtcore_consume_replay_issue_budget(thread_uid, budget)) {
        return false;
    }
    if (!rtcore_dequeue_selected_ready_request(thread_uid)) {
        return false;
    }
    if (!rtcore_step_admitted_replay_request(thread_uid)) {
        return false;
    }
    if (serviced_thread_uid) {
        *serviced_thread_uid = thread_uid;
    }
    return rtcore_route_admitted_replay_request(thread_uid);
}

static bool rtcore_step_selected_ready_replay_request()
{
    rtcore_replay_issue_budget budget = rtcore_replay_issue_budget_config();
    return rtcore_step_selected_ready_replay_request_with_budget(&budget);
}

static bool rtcore_service_replay_ready_requests_with_budget(
    rtcore_replay_issue_budget budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL)
{
    bool progressed = false;
    while (rtcore_replay_issue_budget_available(budget)) {
        unsigned serviced_thread_uid = 0;
        if (!rtcore_step_selected_ready_replay_request_with_budget(
                &budget, &serviced_thread_uid)) {
            break;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                serviced_thread_uid, false, true);
        }
        progressed = true;
    }
    return progressed;
}

static bool rtcore_service_replay_ready_requests_with_budget_for_owner(
    unsigned owner_hw_sid, rtcore_replay_issue_budget budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL)
{
    bool progressed = false;
    while (rtcore_replay_issue_budget_available(budget)) {
        unsigned serviced_thread_uid = 0;
        if (!rtcore_step_selected_ready_replay_request_with_budget_for_owner(
                owner_hw_sid, &budget, &serviced_thread_uid)) {
            break;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                serviced_thread_uid, false, true);
        }
        progressed = true;
    }
    return progressed;
}

static bool
rtcore_service_replay_ready_requests_with_unit_arbitration_for_owner(
    unsigned owner_hw_sid, rtcore_replay_issue_budget budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL)
{
    bool progressed = false;
    progressed |= rtcore_service_replay_ready_queue_with_unit_budget_for_owner(
        g_rtcore_replay_ready_queues.ready_node_queue, owner_hw_sid,
        &budget.node_issue_budget,
        &g_rtcore_replay_unit_arbitration_stats.node_unit_issue_attempts,
        &g_rtcore_replay_unit_arbitration_stats.node_unit_issued,
        &g_rtcore_replay_unit_arbitration_stats.node_unit_budget_exhausted,
        last_identity);
    progressed |= rtcore_service_replay_ready_queue_with_unit_budget_for_owner(
        g_rtcore_replay_ready_queues.ready_primitive_queue, owner_hw_sid,
        &budget.primitive_issue_budget,
        &g_rtcore_replay_unit_arbitration_stats.primitive_unit_issue_attempts,
        &g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued,
        &g_rtcore_replay_unit_arbitration_stats
             .primitive_unit_budget_exhausted,
        last_identity);
    progressed |= rtcore_service_replay_ready_queue_with_unit_budget_for_owner(
        g_rtcore_replay_ready_queues.ready_stack_queue, owner_hw_sid,
        &budget.stack_issue_budget,
        &g_rtcore_replay_unit_arbitration_stats.stack_unit_issue_attempts,
        &g_rtcore_replay_unit_arbitration_stats.stack_unit_issued,
        &g_rtcore_replay_unit_arbitration_stats.stack_unit_budget_exhausted,
        last_identity);
    progressed |= rtcore_service_replay_ready_queue_with_unit_budget_for_owner(
        g_rtcore_replay_ready_queues.ready_completion_queue, owner_hw_sid,
        &budget.completion_issue_budget,
        &g_rtcore_replay_unit_arbitration_stats
             .completion_unit_issue_attempts,
        &g_rtcore_replay_unit_arbitration_stats.completion_unit_issued,
        &g_rtcore_replay_unit_arbitration_stats
             .completion_unit_budget_exhausted,
        last_identity);
    return progressed;
}

static bool rtcore_dequeue_waiting_memory_request(unsigned *thread_uid)
{
    if (g_rtcore_replay_ready_queues.waiting_memory_queue.empty()) {
        return false;
    }

    if (thread_uid) {
        *thread_uid = g_rtcore_replay_ready_queues.waiting_memory_queue.front();
    }
    g_rtcore_replay_ready_queues.waiting_memory_queue.pop_front();
    return true;
}

static bool rtcore_dequeue_waiting_memory_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid)
{
    std::deque<unsigned> &queue =
        g_rtcore_replay_ready_queues.waiting_memory_queue;
    if (queue.empty()) {
        return false;
    }

    for (std::deque<unsigned>::iterator it = queue.begin();
         it != queue.end(); ++it) {
        if (!rtcore_replay_request_owned_by_sm(*it, owner_hw_sid)) {
            continue;
        }
        if (thread_uid) {
            *thread_uid = *it;
        }
        queue.erase(it);
        return true;
    }
    return false;
}

static bool rtcore_wake_waiting_memory_replay_request(unsigned thread_uid)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    if (it->second.state != RTCORE_REPLAY_WAITING_MEMORY) {
        return false;
    }
    if (!rtcore_step_admitted_replay_request(thread_uid)) {
        return false;
    }
    return rtcore_route_admitted_replay_request(thread_uid);
}

static bool rtcore_service_waiting_memory_replay_requests(
    unsigned wake_budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL)
{
    bool progressed = false;
    while (wake_budget > 0) {
        unsigned thread_uid = 0;
        if (!rtcore_dequeue_waiting_memory_request(&thread_uid)) {
            break;
        }
        if (!rtcore_wake_waiting_memory_replay_request(thread_uid)) {
            rtcore_route_admitted_replay_request(thread_uid);
            break;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, true, false);
        }
        wake_budget--;
        progressed = true;
    }
    return progressed;
}

static bool rtcore_service_waiting_memory_replay_requests_for_owner(
    unsigned owner_hw_sid, unsigned wake_budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL)
{
    bool progressed = false;
    while (wake_budget > 0) {
        unsigned thread_uid = 0;
        if (!rtcore_dequeue_waiting_memory_request_for_owner(owner_hw_sid,
                                                             &thread_uid)) {
            break;
        }
        if (!rtcore_wake_waiting_memory_replay_request(thread_uid)) {
            rtcore_route_admitted_replay_request(thread_uid);
            break;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, true, false);
        }
        wake_budget--;
        progressed = true;
    }
    return progressed;
}

static bool rtcore_service_waiting_memory_replay_requests()
{
    return rtcore_service_waiting_memory_replay_requests(
        rtcore_replay_memory_wake_budget_config());
}

static bool rtcore_service_waiting_memory_replay_requests_for_owner(
    unsigned owner_hw_sid)
{
    return rtcore_service_waiting_memory_replay_requests_for_owner(
        owner_hw_sid, rtcore_replay_memory_wake_budget_config());
}

static rtcore_replay_service_tick_result rtcore_service_replay_tick()
{
    rtcore_replay_service_tick_result result = {};
    rtcore_replay_service_cycle_identity_snapshot memory_identity = {};
    rtcore_replay_service_cycle_identity_snapshot ready_identity = {};
    result.memory_progressed =
        rtcore_service_waiting_memory_replay_requests(
            rtcore_replay_memory_wake_budget_config(), &memory_identity);
    result.ready_progressed = rtcore_service_replay_ready_requests_with_budget(
        rtcore_replay_issue_budget_config(), &ready_identity);
    result.progressed = result.memory_progressed || result.ready_progressed;
    if (result.ready_progressed) {
        result.last_progress_identity = ready_identity;
    } else if (result.memory_progressed) {
        result.last_progress_identity = memory_identity;
    }
    return result;
}

static rtcore_replay_service_tick_result
rtcore_service_replay_tick_for_owner(unsigned owner_hw_sid)
{
    rtcore_replay_service_tick_result result = {};
    rtcore_replay_service_cycle_identity_snapshot memory_identity = {};
    rtcore_replay_service_cycle_identity_snapshot ready_identity = {};
    result.memory_progressed =
        rtcore_service_waiting_memory_replay_requests_for_owner(
            owner_hw_sid, rtcore_replay_memory_wake_budget_config(),
            &memory_identity);
    result.ready_progressed =
        rtcore_replay_unit_arbitration_enabled()
            ? rtcore_service_replay_ready_requests_with_unit_arbitration_for_owner(
                  owner_hw_sid, rtcore_replay_issue_budget_config(),
                  &ready_identity)
            : rtcore_service_replay_ready_requests_with_budget_for_owner(
                  owner_hw_sid, rtcore_replay_issue_budget_config(),
                  &ready_identity);
    result.progressed = result.memory_progressed || result.ready_progressed;
    if (result.ready_progressed) {
        result.last_progress_identity = ready_identity;
    } else if (result.memory_progressed) {
        result.last_progress_identity = memory_identity;
    }
    return result;
}

static rtcore_replay_service_tick_result rtcore_maybe_service_replay_tick()
{
    rtcore_replay_service_tick_result result = {};
    if (!rtcore_replay_service_tick_enabled()) {
        return result;
    }
    return rtcore_service_replay_tick();
}

static rtcore_replay_service_tick_result
rtcore_maybe_service_replay_tick(unsigned owner_hw_sid)
{
    rtcore_replay_service_tick_result result = {};
    if (!rtcore_replay_service_tick_enabled()) {
        return result;
    }
    return rtcore_service_replay_tick_for_owner(owner_hw_sid);
}

static void rtcore_record_replay_service_tick_result(
    const rtcore_replay_service_tick_result &result)
{
    g_rtcore_replay_service_tick_stats.tick_attempts++;
    if (result.progressed) {
        g_rtcore_replay_service_tick_stats.ticks_progressed++;
    }
    if (result.memory_progressed) {
        g_rtcore_replay_service_tick_stats.memory_ticks_progressed++;
    }
    if (result.ready_progressed) {
        g_rtcore_replay_service_tick_stats.ready_ticks_progressed++;
    }
}

static rtcore_service_tick_stats_snapshot
rtcore_get_replay_service_tick_stats_snapshot()
{
    rtcore_service_tick_stats_snapshot snapshot = {};
    snapshot.valid = true;
    snapshot.tick_attempts =
        g_rtcore_replay_service_tick_stats.tick_attempts;
    snapshot.ticks_progressed =
        g_rtcore_replay_service_tick_stats.ticks_progressed;
    snapshot.memory_ticks_progressed =
        g_rtcore_replay_service_tick_stats.memory_ticks_progressed;
    snapshot.ready_ticks_progressed =
        g_rtcore_replay_service_tick_stats.ready_ticks_progressed;
    return snapshot;
}

static bool rtcore_should_log_replay_service_tick_stats_snapshot(
    const rtcore_service_tick_stats_snapshot &snapshot)
{
    if (!snapshot.valid) {
        return false;
    }

    if (g_rtcore_replay_service_tick_stats_logs_emitted <
        rtcore_replay_service_tick_stats_log_limit()) {
        g_rtcore_replay_service_tick_stats_logs_emitted++;
        if (snapshot.ticks_progressed >
            g_rtcore_replay_service_tick_stats_last_logged_ticks_progressed) {
            g_rtcore_replay_service_tick_stats_last_logged_ticks_progressed =
                snapshot.ticks_progressed;
        }
        return true;
    }

    if (snapshot.ticks_progressed <=
        g_rtcore_replay_service_tick_stats_last_logged_ticks_progressed) {
        return false;
    }
    if (g_rtcore_replay_service_tick_stats_progress_logs_emitted >=
        rtcore_replay_service_tick_stats_progress_log_limit()) {
        return false;
    }

    g_rtcore_replay_service_tick_stats_logs_emitted++;
    g_rtcore_replay_service_tick_stats_progress_logs_emitted++;
    g_rtcore_replay_service_tick_stats_last_logged_ticks_progressed =
        snapshot.ticks_progressed;
    return true;
}

static void rtcore_log_replay_service_tick_stats_snapshot(
    const rtcore_service_tick_stats_snapshot &snapshot)
{
    if (!rtcore_should_log_replay_service_tick_stats_snapshot(snapshot)) {
        return;
    }

    printf("GPGPU-Sim RTCORE_REPLAY_SERVICE_TICK_STATS "
           "tick_attempts=%u ticks_progressed=%u "
           "memory_ticks_progressed=%u ready_ticks_progressed=%u\n",
           snapshot.tick_attempts, snapshot.ticks_progressed,
           snapshot.memory_ticks_progressed, snapshot.ready_ticks_progressed);
}

static unsigned rtcore_replay_unit_arbitration_total_issued()
{
    return g_rtcore_replay_unit_arbitration_stats.node_unit_issued +
           g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued +
           g_rtcore_replay_unit_arbitration_stats.stack_unit_issued +
           g_rtcore_replay_unit_arbitration_stats.completion_unit_issued;
}

static unsigned rtcore_replay_unit_arbitration_total_budget_exhausted()
{
    return g_rtcore_replay_unit_arbitration_stats.node_unit_budget_exhausted +
           g_rtcore_replay_unit_arbitration_stats
               .primitive_unit_budget_exhausted +
           g_rtcore_replay_unit_arbitration_stats.stack_unit_budget_exhausted +
           g_rtcore_replay_unit_arbitration_stats
               .completion_unit_budget_exhausted;
}

static bool rtcore_should_log_replay_unit_arbitration_stats()
{
    const unsigned total_issued = rtcore_replay_unit_arbitration_total_issued();
    const unsigned total_budget_exhausted =
        rtcore_replay_unit_arbitration_total_budget_exhausted();

    if (g_rtcore_replay_unit_arbitration_stats_logs_emitted <
        rtcore_replay_unit_arbitration_stats_log_limit()) {
        g_rtcore_replay_unit_arbitration_stats_logs_emitted++;
        g_rtcore_replay_unit_arbitration_stats_last_logged_total_issued =
            total_issued;
        g_rtcore_replay_unit_arbitration_stats_last_logged_total_budget_exhausted =
            total_budget_exhausted;
        return true;
    }

    if (total_issued <=
            g_rtcore_replay_unit_arbitration_stats_last_logged_total_issued &&
        total_budget_exhausted <=
            g_rtcore_replay_unit_arbitration_stats_last_logged_total_budget_exhausted) {
        return false;
    }
    if (g_rtcore_replay_unit_arbitration_stats_progress_logs_emitted >=
        rtcore_replay_unit_arbitration_stats_progress_log_limit()) {
        return false;
    }

    g_rtcore_replay_unit_arbitration_stats_logs_emitted++;
    g_rtcore_replay_unit_arbitration_stats_progress_logs_emitted++;
    g_rtcore_replay_unit_arbitration_stats_last_logged_total_issued =
        total_issued;
    g_rtcore_replay_unit_arbitration_stats_last_logged_total_budget_exhausted =
        total_budget_exhausted;
    return true;
}

static void rtcore_maybe_log_replay_unit_arbitration_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_unit_arbitration_stats_log_enabled()) {
        return;
    }
    if (!rtcore_should_log_replay_unit_arbitration_stats()) {
        return;
    }

    printf("GPGPU-Sim RTCORE_REPLAY_UNIT_ARBITRATION_STATS "
           "owner_hw_sid=%u "
           "node_unit_issue_attempts=%u node_unit_issued=%u "
           "node_unit_budget_exhausted=%u "
           "primitive_unit_issue_attempts=%u primitive_unit_issued=%u "
           "primitive_unit_budget_exhausted=%u "
           "stack_unit_issue_attempts=%u stack_unit_issued=%u "
           "stack_unit_budget_exhausted=%u "
           "completion_unit_issue_attempts=%u completion_unit_issued=%u "
           "completion_unit_budget_exhausted=%u\n",
           owner_hw_sid,
           g_rtcore_replay_unit_arbitration_stats.node_unit_issue_attempts,
           g_rtcore_replay_unit_arbitration_stats.node_unit_issued,
           g_rtcore_replay_unit_arbitration_stats.node_unit_budget_exhausted,
           g_rtcore_replay_unit_arbitration_stats
               .primitive_unit_issue_attempts,
           g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued,
           g_rtcore_replay_unit_arbitration_stats
               .primitive_unit_budget_exhausted,
           g_rtcore_replay_unit_arbitration_stats.stack_unit_issue_attempts,
           g_rtcore_replay_unit_arbitration_stats.stack_unit_issued,
           g_rtcore_replay_unit_arbitration_stats.stack_unit_budget_exhausted,
           g_rtcore_replay_unit_arbitration_stats
               .completion_unit_issue_attempts,
           g_rtcore_replay_unit_arbitration_stats.completion_unit_issued,
           g_rtcore_replay_unit_arbitration_stats
               .completion_unit_budget_exhausted);
    fflush(stdout);
}

static void rtcore_publish_replay_service_tick_stats_snapshot()
{
    g_rtcore_replay_service_tick_stats_snapshot =
        rtcore_get_replay_service_tick_stats_snapshot();
    if (rtcore_replay_service_tick_stats_log_enabled()) {
        rtcore_log_replay_service_tick_stats_snapshot(
            g_rtcore_replay_service_tick_stats_snapshot);
    }
}

static rtcore_replay_service_cycle_result
rtcore_service_replay_cycle(unsigned owner_hw_sid, unsigned long long service_cycle)
{
    rtcore_replay_service_cycle_result result = {};
    result.owner_hw_sid = owner_hw_sid;
    result.service_cycle = service_cycle;
    result.service_enabled = rtcore_replay_service_tick_enabled();
    if (!result.service_enabled) {
        return result;
    }

    result.tick_result = rtcore_maybe_service_replay_tick(owner_hw_sid);
    rtcore_record_replay_service_tick_result(result.tick_result);
    rtcore_publish_replay_service_tick_stats_snapshot();
    rtcore_maybe_log_replay_unit_arbitration_stats(owner_hw_sid);
    result.stats_snapshot = g_rtcore_replay_service_tick_stats_snapshot;
    return result;
}

extern "C" bool rtcore_service_replay_cycle_for_sm_with_identity(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *service_enabled, bool *memory_progressed, bool *ready_progressed,
    rtcore_replay_service_cycle_identity_snapshot *identity_snapshot)
{
    rtcore_replay_service_cycle_result result = rtcore_service_replay_cycle(owner_hw_sid, service_cycle);
    if (service_enabled) {
        *service_enabled = result.service_enabled;
    }
    if (memory_progressed) {
        *memory_progressed = result.tick_result.memory_progressed;
    }
    if (ready_progressed) {
        *ready_progressed = result.tick_result.ready_progressed;
    }
    if (identity_snapshot) {
        *identity_snapshot = result.tick_result.last_progress_identity;
    }
    return result.tick_result.progressed;
}

extern "C" bool rtcore_service_replay_cycle_for_sm(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *service_enabled, bool *memory_progressed, bool *ready_progressed)
{
    return rtcore_service_replay_cycle_for_sm_with_identity(
        owner_hw_sid, service_cycle, service_enabled, memory_progressed,
        ready_progressed, NULL);
}

extern "C" bool rtcore_query_replay_warp_completion_shadow(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask,
    rtcore_replay_warp_completion_shadow_snapshot *snapshot)
{
    rtcore_replay_warp_completion_shadow_snapshot local_snapshot = {};
    local_snapshot.enabled =
        rtcore_replay_warp_completion_aggregation_shadow_enabled();
    local_snapshot.owner_hw_sid = owner_hw_sid;
    local_snapshot.warp_uid = warp_uid;
    local_snapshot.warp_id = warp_id;
    local_snapshot.active_mask = active_mask;

    if (local_snapshot.enabled) {
        rtcore_replay_warp_completion_shadow_key key = {};
        key.owner_hw_sid = owner_hw_sid;
        key.warp_uid = warp_uid;
        key.warp_id = warp_id;
        key.active_mask = active_mask;
        std::map<rtcore_replay_warp_completion_shadow_key,
                 rtcore_replay_warp_completion_shadow_state>::const_iterator it =
            g_rtcore_replay_warp_completion_shadow.find(key);
        if (it != g_rtcore_replay_warp_completion_shadow.end()) {
            local_snapshot.found = true;
            local_snapshot.all_active_lanes_complete =
                it->second.all_active_lanes_complete;
            local_snapshot.admitted_lane_mask = it->second.admitted_lane_mask;
            local_snapshot.completed_lane_mask = it->second.completed_lane_mask;
            local_snapshot.completed_lane_count = it->second.completed_lane_count;
        }
    }

    if (snapshot) {
        *snapshot = local_snapshot;
    }
    return local_snapshot.enabled && local_snapshot.found &&
           local_snapshot.all_active_lanes_complete;
}

static void rtcore_try_service_replay_after_admission(unsigned owner_hw_sid)
{
    (void)rtcore_service_replay_cycle(owner_hw_sid, 0);
}

float get_norm(float4 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}
float get_norm(float3 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float4 normalized(float4 v)
{
    float norm = get_norm(v);
    return {v.x / norm, v.y / norm, v.z / norm, v.w / norm};
}
float3 normalized(float3 v)
{
    float norm = get_norm(v);
    return {v.x / norm, v.y / norm, v.z / norm};
}

Ray make_transformed_ray(Ray &ray, float4x4 matrix, float *worldToObject_tMultiplier)
{
    Ray transformedRay;
    float4 transformedOrigin4 = matrix * float4({ray.get_origin().x, ray.get_origin().y, ray.get_origin().z, 1});
    float4 transformedDirection4 = matrix * float4({ray.get_direction().x, ray.get_direction().y, ray.get_direction().z, 0});

    float3 transformedOrigin = {transformedOrigin4.x / transformedOrigin4.w, transformedOrigin4.y / transformedOrigin4.w, transformedOrigin4.z / transformedOrigin4.w};
    float3 transformedDirection = {transformedDirection4.x, transformedDirection4.y, transformedDirection4.z};
    *worldToObject_tMultiplier = get_norm(transformedDirection);
    transformedDirection = normalized(transformedDirection);

    transformedRay.make_ray(transformedOrigin, transformedDirection, ray.get_tmin() * (*worldToObject_tMultiplier), ray.get_tmax() * (*worldToObject_tMultiplier));
    return transformedRay;
}

float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MIN_MAX(a0, a1, d);
	float t2 = MIN_MAX(b0, b1, t1);
	float t3 = MIN_MAX(c0, c1, t2);
	return t3;
}

float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MAX_MIN(a0, a1, d);
	float t2 = MAX_MIN(b0, b1, t1);
	float t3 = MAX_MIN(c0, c1, t2);
	return t3;
}

float3 get_t_bound(float3 box, float3 origin, float3 idirection)
{
    // // Avoid div by zero, returns 1/2^80, an extremely small number
    // const float ooeps = exp2f(-80.0f);

    // // Calculate inverse direction
    // float3 idir;
    // idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    // idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    // idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    // Calculate bounds
    float3 result;
    result.x = (box.x - origin.x) * idirection.x;
    result.y = (box.y - origin.y) * idirection.y;
    result.z = (box.z - origin.z) * idirection.z;

    // Return
    return result;
}

float3 calculate_idir(float3 direction) {
    // Avoid div by zero, returns 1/2^80, an extremely small number
    const float ooeps = exp2f(-80.0f);

    // Calculate inverse direction
    float3 idir;
    // TODO: is this wrong?
    idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    // idir.x = fabsf(direction.x) > ooeps ? 1.0f / direction.x : copysignf(ooeps, direction.x);
    // idir.y = fabsf(direction.y) > ooeps ? 1.0f / direction.y : copysignf(ooeps, direction.y);
    // idir.z = fabsf(direction.z) > ooeps ? 1.0f / direction.z : copysignf(ooeps, direction.z);
    return idir;
}

bool ray_box_test(float3 low, float3 high, float3 idirection, float3 origin, float tmin, float tmax, float& thit)
{
	// const float3 lo = Low * InvDir - Ood;
	// const float3 hi = High * InvDir - Ood;
    float3 lo = get_t_bound(low, origin, idirection);
    float3 hi = get_t_bound(high, origin, idirection);

    // QUESTION: max value does not match rtao benchmark, rtao benchmark converts float to int with __float_as_int
    // i.e. __float_as_int: -110.704826 => -1025677090, -24.690834 => -1044019502

	// const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	// const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);
    float min = magic_max7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
    float max = magic_min7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);

	// OutIntersectionDist = slabMin;
    thit = min;

	// return slabMin <= slabMax;
    return (min <= max);
}

typedef struct StackEntry {
    uint8_t* addr;
    bool topLevel;
    bool leaf;
    StackEntry(uint8_t* addr, bool topLevel, bool leaf): addr(addr), topLevel(topLevel), leaf(leaf) {}
} StackEntry;


std::ofstream print_tree;
void traverse_tree(volatile uint8_t* address, bool isTopLevel = true, bool isLeaf = false, bool isRoot = true)
{
    if(isRoot)
    {
        GEN_RT_BVH topBVH;
        GEN_RT_BVH_unpack(&topBVH, (uint8_t*)address);

        uint8_t* topRootAddr = (uint8_t*)address + topBVH.RootNodeOffset;

        if (print_tree.is_open())
        {
            print_tree << "traversing bvh , isTopLevel = " << isTopLevel << (void *)(address) << ", RootNodeOffset = (" << topBVH.RootNodeOffset << std::endl;
        }

        traverse_tree(topRootAddr, isTopLevel, false, false);
    }
    
    else if(!isLeaf) // internal nodes
    {
        struct GEN_RT_BVH_INTERNAL_NODE node;
        GEN_RT_BVH_INTERNAL_NODE_unpack(&node, address);

        if (print_tree.is_open())
        {
            uint8_t *child_addrs[6];
            child_addrs[0] = address + (node.ChildOffset * 64);
            for(int i = 0; i < 5; i++)
                child_addrs[i + 1] = child_addrs[i] + node.ChildSize[i] * 64;
            
            print_tree << "traversing internal node " << (void *)address;
            print_tree << ", isTopLevel = " << isTopLevel << ", child offset = " << node.ChildOffset << ", node type = " << node.NodeType;
            print_tree << ", child size = (" << node.ChildSize[0] << ", " << node.ChildSize[1] << ", " << node.ChildSize[2] << ", " << node.ChildSize[3] << ", " << node.ChildSize[4] << ", " << node.ChildSize[5] << ")";
            print_tree << ", child type = (" << node.ChildType[0] << ", " << node.ChildType[1] << ", " << node.ChildType[2] << ", " << node.ChildType[3] << ", " << node.ChildType[4] << ", " << node.ChildType[5] << ")";
            print_tree << ", child addresses = (" << (void*)(child_addrs[0]) << ", " << (void*)(child_addrs[1]) << ", " << (void*)(child_addrs[2]) << ", " << (void*)(child_addrs[3]) << ", " << (void*)(child_addrs[4]) << ", " << (void*)(child_addrs[5]) << ")";
            print_tree << std::endl;
        }

        uint8_t *child_addr = address + (node.ChildOffset * 64);
        for(int i = 0; i < 6; i++)
        {
            if(node.ChildSize[i] > 0)
            {
                if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                    isLeaf = true;
                else
                    isLeaf = false;

                traverse_tree(child_addr, isTopLevel, isLeaf, false);
            }

            child_addr += node.ChildSize[i] * 64;
        }
    }

    else // leaf nodes
    {
        if(isTopLevel)
        {
            GEN_RT_BVH_INSTANCE_LEAF instanceLeaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&instanceLeaf, address);

            float4x4 worldToObjectMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.WorldToObjectm00);
            float4x4 objectToWorldMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.ObjectToWorldm00);

            assert(instanceLeaf.BVHAddress != NULL);

            if (print_tree.is_open())
            {
                print_tree << "traversing top level leaf node " << (void *)address << ", instanceID = " << instanceLeaf.InstanceID << ", BVHAddress = " << instanceLeaf.BVHAddress << ", ShaderIndex = " << instanceLeaf.ShaderIndex << std::endl;
            }

            traverse_tree(address + instanceLeaf.BVHAddress, false, false, true);
        }
        else
        {
            struct GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR leaf_descriptor;
            GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_unpack(&leaf_descriptor, address);
            
            if (leaf_descriptor.LeafType == TYPE_QUAD)
            {
                struct GEN_RT_BVH_QUAD_LEAF leaf;
                GEN_RT_BVH_QUAD_LEAF_unpack(&leaf, address);

                float3 p[3];
                for(int i = 0; i < 3; i++)
                {
                    p[i].x = leaf.QuadVertex[i].X;
                    p[i].y = leaf.QuadVertex[i].Y;
                    p[i].z = leaf.QuadVertex[i].Z;
                }

                assert(leaf.PrimitiveIndex1Delta == 0);

                if (print_tree.is_open())
                {
                    print_tree << "quad node " << (void*)address << " ";
                    print_tree << "primitiveID = " << leaf.PrimitiveIndex0 << "\n";

                    print_tree << "p[0] = (" << p[0].x << ", " << p[0].y << ", " << p[0].z << ") ";
                    print_tree << "p[1] = (" << p[1].x << ", " << p[1].y << ", " << p[1].z << ") ";
                    print_tree << "p[2] = (" << p[2].x << ", " << p[2].y << ", " << p[2].z << ") ";
                    print_tree << "p[3] = (" << p[3].x << ", " << p[3].y << ", " << p[3].z << ")" << std::endl;
                }
            }
            else
            {
                struct GEN_RT_BVH_PROCEDURAL_LEAF leaf;
                GEN_RT_BVH_PROCEDURAL_LEAF_unpack(&leaf, address);

                if (print_tree.is_open())
                {
                    print_tree << "PROCEDURAL node " << (void*)address << " ";
                    print_tree << "NumPrimitives = " << leaf.NumPrimitives << ", LastPrimitive = " << leaf.LastPrimitive << ", PrimitiveIndex[0]" << leaf.PrimitiveIndex[0] << "\n";
                }
            }
        }
    }
}

void VulkanRayTracing::init(uint32_t launch_width, uint32_t launch_height)
{
    if(_init_)
        return;
    _init_ = true;

    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    uint32_t width = (launch_width + 31) / 32;
    uint32_t height = launch_height;

    if(ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->m_rt_intersection_table_type == 0)
        intersectionTableType = IntersectionTableType::Baseline;
    else if(ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->m_rt_intersection_table_type == 1)
        intersectionTableType = IntersectionTableType::Function_Call_Coalescing;
    else
        assert(0);

    if(intersectionTableType == IntersectionTableType::Baseline)
    {
        intersection_table = new Baseline_warp_intersection_table**[width];
        for(int i = 0; i < width; i++)
        {
            intersection_table[i] = new Baseline_warp_intersection_table*[height];
            for(int j = 0; j < height; j++)
                intersection_table[i][j] = new Baseline_warp_intersection_table();
        }
    }
    else
    {
        intersection_table = new Coalescing_warp_intersection_table**[width];
        for(int i = 0; i < width; i++)
        {
            intersection_table[i] = new Coalescing_warp_intersection_table*[height];
            for(int j = 0; j < height; j++)
                intersection_table[i][j] = new Coalescing_warp_intersection_table();
        }

    }
    anyhit_table = new Baseline_warp_intersection_table**[width];
    for(int i = 0; i < width; i++)
    {
        anyhit_table[i] = new Baseline_warp_intersection_table*[height];
        for(int j = 0; j < height; j++)
            anyhit_table[i][j] = new Baseline_warp_intersection_table();
    }
}


bool debugTraversal = false;
bool found_AS = false;
VkAccelerationStructureKHR topLevelAS_first = NULL;

static uint64_t g_rtcore_next_proxy_id = 1;
static std::map<std::pair<uint64_t, uint64_t>, uint64_t>
    g_rtcore_traversable_proxy_ids;
static std::map<std::pair<uint64_t, uint64_t>, uint64_t>
    g_rtcore_root_proxy_ids;

static uint64_t rtcore_get_or_create_proxy_id(
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> *registry,
    uint64_t key0, uint64_t key1)
{
    std::pair<uint64_t, uint64_t> key = std::make_pair(key0, key1);
    std::map<std::pair<uint64_t, uint64_t>, uint64_t>::iterator it =
        registry->find(key);
    if (it != registry->end())
        return it->second;

    uint64_t proxy_id = g_rtcore_next_proxy_id++;
    (*registry)[key] = proxy_id;
    return proxy_id;
}

static uint64_t rtcore_get_or_create_traversable_proxy_id(
    VkAccelerationStructureKHR top_level_as)
{
    return rtcore_get_or_create_proxy_id(&g_rtcore_traversable_proxy_ids,
                                         (uint64_t)top_level_as, 0);
}

static uint64_t rtcore_get_or_create_root_proxy_id(
    VkAccelerationStructureKHR top_level_as, uint64_t root_node_offset)
{
    return rtcore_get_or_create_proxy_id(&g_rtcore_root_proxy_ids,
                                         (uint64_t)top_level_as,
                                         root_node_offset);
}

void VulkanRayTracing::traceRay(VkAccelerationStructureKHR _topLevelAS,
				   uint rayFlags,
                   uint cullMask,
                   uint sbtRecordOffset,
                   uint sbtRecordStride,
                   uint missIndex,
                   float3 origin,
                   float Tmin,
                   float3 direction,
                   float Tmax,
                   int payload,
                   const ptx_instruction *pI,
                   ptx_thread_info *thread)
{
    // printf("## calling trceRay function. rayFlags = %d, cullMask = %d, sbtRecordOffset = %d, sbtRecordStride = %d, missIndex = %d, origin = (%f, %f, %f), Tmin = %f, direction = (%f, %f, %f), Tmax = %f, payload = %d\n",
    //         rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex, origin.x, origin.y, origin.z, Tmin, direction.x, direction.y, direction.z, Tmax, payload);

    VkAccelerationStructureKHR rtcore_trace_input_top_level_as = _topLevelAS;

    if (dump_trace && !dumped) 
    {
        dump_AS(VulkanRayTracing::descriptorSet, _topLevelAS);
        std::cout << "Trace dumped" << std::endl;
        dumped = true;
    }

    // Convert device address back to host address for func sim. This will break if the device address was modified then passed to traceRay. Should be fixable if I also record the size when I malloc then I can check the bounds of the device address.
    uint8_t* deviceAddress = nullptr;
    int64_t device_offset = (uint64_t)tlas_addr - (uint64_t)_topLevelAS;
    if (use_external_launcher)
    {
        deviceAddress = (uint8_t*)_topLevelAS;
        bool addressFound = false;
        for (int i = 0; i < MAX_DESCRIPTOR_SETS; i++)
        {
            for (int j = 0; j < MAX_DESCRIPTOR_SET_BINDINGS; j++)
            {
                if (launcher_deviceDescriptorSets[i][j] == (void*)_topLevelAS)
                {
                    _topLevelAS = launcher_descriptorSets[i][j];
                    addressFound = true;
                    break;
                }
            }
            if (addressFound)
                break;
        }
        if (!addressFound)
            abort();
    
        // Calculate offset between host and device for memory transactions
        device_offset = (uint64_t)deviceAddress - (uint64_t)_topLevelAS;
    }

    // if(!found_AS)
    // {
    //     found_AS = true;
    //     topLevelAS_first = _topLevelAS;
    //     print_tree.open("bvh_tree.txt");
    //     traverse_tree((uint8_t*)_topLevelAS);
    //     print_tree.close();
    // }
    // else
    // {
    //     assert(topLevelAS_first != NULL);
    //     assert(topLevelAS_first == _topLevelAS);
    // }

    Traversal_data traversal_data;

    traversal_data.n_all_hits = 0;
    traversal_data.ray_world_direction = direction;
    traversal_data.ray_world_origin = origin;
    traversal_data.sbtRecordOffset = sbtRecordOffset;
    traversal_data.sbtRecordStride = sbtRecordStride;
    traversal_data.missIndex = missIndex;
    traversal_data.rayFlags = rayFlags;
    traversal_data.cullMask = cullMask;
    traversal_data.rtcore_trace_input_top_level_as = (uint64_t)rtcore_trace_input_top_level_as;
    traversal_data.rtcore_trace_input_has_top_level_as =
        rtcore_trace_input_top_level_as != NULL ? 1 : 0;
    traversal_data.Tmin = Tmin;
    traversal_data.Tmax = Tmax;
    traversal_data.rtcore_traversable_proxy_id = 0;
    traversal_data.rtcore_root_proxy_id = 0;
    traversal_data.rtcore_node_visits = 0;
    traversal_data.rtcore_primitive_tests = 0;

    rtcore_bounded_trace_collector rtcore_compact_trace(thread);

    const bool pixel_trace_enabled = rtcore_pixel_trace_matches_thread(thread);
    if (pixel_trace_enabled) {
        printf("GPGPU-Sim RTCORE_PIXEL_TRACE trace-ray-begin "
               "launch=(%u,%u), thread_uid=%u, tid=(%u,%u,%u), "
               "ctaid=(%u,%u,%u), origin=(%.9g,%.9g,%.9g), "
               "direction=(%.9g,%.9g,%.9g), tmin=%.9g, tmax=%.9g, "
               "rayFlags=%u, cullMask=%u, sbtOffset=%u, sbtStride=%u, "
               "missIndex=%u\n",
               rtcore_launch_id_x_for_thread(thread),
               rtcore_launch_id_y_for_thread(thread), thread->get_uid(),
               thread->get_tid().x, thread->get_tid().y, thread->get_tid().z,
               thread->get_ctaid().x, thread->get_ctaid().y,
               thread->get_ctaid().z, origin.x, origin.y, origin.z,
               direction.x, direction.y, direction.z, Tmin, Tmax, rayFlags,
               cullMask, sbtRecordOffset, sbtRecordStride, missIndex);
    }

    bool hit_procedural = false;

    std::ofstream traversalFile;

    if (debugTraversal)
    {
        traversalFile.open("traversal.txt");
        traversalFile << "starting traversal\n";
        traversalFile << "origin = (" << origin.x << ", " << origin.y << ", " << origin.z << "), ";
        traversalFile << "direction = (" << direction.x << ", " << direction.y << ", " << direction.z << "), ";
        traversalFile << "tmin = " << Tmin << ", tmax = " << Tmax << std::endl << std::endl;
    }


    bool terminateOnFirstHit = rayFlags & SpvRayFlagsTerminateOnFirstHitKHRMask;
    bool skipClosestHitShader = rayFlags & SpvRayFlagsSkipClosestHitShaderKHRMask;
    bool skipAnyHitShader = rayFlags & SpvRayFlagsOpaqueKHRMask;

    std::vector<MemoryTransactionRecord> transactions;
    std::vector<MemoryStoreTransactionRecord> store_transactions;

    gpgpu_context *ctx = GPGPU_Context();

    if (terminateOnFirstHit) ctx->func_sim->g_n_anyhit_rays++;
    else ctx->func_sim->g_n_closesthit_rays++;

    unsigned total_nodes_accessed = 0;
    unsigned total_primitive_tests = 0;
    std::map<uint8_t*, unsigned> tree_level_map;
    
	// Create ray
	Ray ray;
	ray.make_ray(origin, direction, Tmin, Tmax);
    thread->add_ray_properties(ray);

	// Set thit to max
    float min_thit = ray.dir_tmax.w;
    struct GEN_RT_BVH_QUAD_LEAF closest_leaf;
    struct GEN_RT_BVH_INSTANCE_LEAF closest_instanceLeaf;    
    float4x4 closest_worldToObject, closest_objectToWorld;
    Ray closest_objectRay;
    float min_thit_object;

	// Get bottom-level AS
    //uint8_t* topLevelASAddr = get_anv_accel_address((VkAccelerationStructureKHR)_topLevelAS);
    GEN_RT_BVH topBVH; //TODO: test hit with world before traversal
    GEN_RT_BVH_unpack(&topBVH, (uint8_t*)_topLevelAS);
    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)_topLevelAS + device_offset), GEN_RT_BVH_length * 4, TransactionType::BVH_STRUCTURE));
    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;
    rtcore_compact_trace.append_node_fetch(
        (uint64_t)_topLevelAS + device_offset, GEN_RT_BVH_length * 4,
        rtcore_trace_node_fetch_flags(
            true, RTCORE_TRACE_NODE_KIND_BVH_HEADER));

    uint8_t* topRootAddr = (uint8_t*)_topLevelAS + topBVH.RootNodeOffset;
    traversal_data.rtcore_traversable_proxy_id =
        rtcore_get_or_create_traversable_proxy_id(_topLevelAS);
    traversal_data.rtcore_root_proxy_id =
        rtcore_get_or_create_root_proxy_id(_topLevelAS,
                                           topBVH.RootNodeOffset);

    // Get min/max
    if (!ctx->func_sim->g_rt_world_set) {
        struct GEN_RT_BVH_INTERNAL_NODE node;
        GEN_RT_BVH_INTERNAL_NODE_unpack(&node, topRootAddr);
        for(int i = 0; i < 6; i++) {
            if (node.ChildSize[i] > 0) {
                float3 idir = calculate_idir(ray.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                float3 lo, hi;
                set_child_bounds(&node, i, &lo, &hi);
                ctx->func_sim->g_rt_world_min = min(ctx->func_sim->g_rt_world_min, lo);
                ctx->func_sim->g_rt_world_max = min(ctx->func_sim->g_rt_world_max, hi);
            }
        }
        ctx->func_sim->g_rt_world_set = true;
    }

    std::list<StackEntry> stack;
    tree_level_map[topRootAddr] = 1;
    
    {
        float3 lo, hi;
        lo.x = topBVH.BoundsMin.X;
        lo.y = topBVH.BoundsMin.Y;
        lo.z = topBVH.BoundsMin.Z;
        hi.x = topBVH.BoundsMax.X;
        hi.y = topBVH.BoundsMax.Y;
        hi.z = topBVH.BoundsMax.Z;

        float thit;
        bool root_hit = ray_box_test(lo, hi, calculate_idir(ray.get_direction()), ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit);
        rtcore_compact_trace.append_node_test(
            (uint64_t)topRootAddr + device_offset, 0, root_hit, true);
        if(root_hit) {
            stack.push_back(StackEntry(topRootAddr, true, false));
            rtcore_compact_trace.append_stack_push(
                (uint64_t)topRootAddr + device_offset, true, false);
        }
    }

    while (!stack.empty())
    {
        uint8_t *node_addr = NULL;
        uint8_t *next_node_addr = NULL;

        // traverse top level internal nodes
        assert(stack.back().topLevel);
        
        if(!stack.back().leaf)
        {
            next_node_addr = stack.back().addr;
            uint64_t tlas_stack_device_offset =
                (uint64_t)tlas_addr - (uint64_t)_topLevelAS;
            rtcore_compact_trace.append_stack_pop(
                (uint64_t)stack.back().addr + tlas_stack_device_offset,
                stack.back().topLevel, stack.back().leaf);
            stack.pop_back();
        }

        while (next_node_addr != NULL)
        {
            // TLAS offset
            device_offset = (uint64_t)tlas_addr - (uint64_t)_topLevelAS;

            node_addr = next_node_addr;
            next_node_addr = NULL;
            struct GEN_RT_BVH_INTERNAL_NODE node;
            GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);
            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)node_addr + device_offset), GEN_RT_BVH_INTERNAL_NODE_length * 4, TransactionType::BVH_INTERNAL_NODE));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
            total_nodes_accessed++;
            rtcore_compact_trace.append_node_fetch(
                (uint64_t)node_addr + device_offset,
                GEN_RT_BVH_INTERNAL_NODE_length * 4,
                rtcore_trace_node_fetch_flags(
                    true, RTCORE_TRACE_NODE_KIND_INTERNAL));

            if (debugTraversal)
            {
                traversalFile << "traversing top level internal node " << (void *)node_addr;
                traversalFile << ", child offset = " << node.ChildOffset << ", node type = " << node.NodeType;
                traversalFile << ", child size = (" << node.ChildSize[0] << ", " << node.ChildSize[1] << ", " << node.ChildSize[2] << ", " << node.ChildSize[3] << ", " << node.ChildSize[4] << ", " << node.ChildSize[5] << ")";
                traversalFile << ", child type = (" << node.ChildType[0] << ", " << node.ChildType[1] << ", " << node.ChildType[2] << ", " << node.ChildType[3] << ", " << node.ChildType[4] << ", " << node.ChildType[5] << ")";
                traversalFile << std::endl;
            }

            bool child_hit[6];
            float thit[6];
            for(int i = 0; i < 6; i++)
            {
                if (node.ChildSize[i] > 0)
                {
                    float3 idir = calculate_idir(ray.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                    float3 lo, hi;
                    set_child_bounds(&node, i, &lo, &hi);

                    child_hit[i] = ray_box_test(lo, hi, idir, ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit[i]);
                    if(child_hit[i] && thit[i] >= min_thit)
                        child_hit[i] = false;
                    rtcore_compact_trace.append_node_test(
                        (uint64_t)node_addr + device_offset, i,
                        child_hit[i], true);

                    
                    if (debugTraversal)
                    {
                        if(child_hit[i])
                            traversalFile << "hit child number " << i << ", ";
                        else
                            traversalFile << "missed child number " << i << ", ";
                        traversalFile << "lo = (" << lo.x << ", " << lo.y << ", " << lo.z << "), ";
                        traversalFile << "hi = (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
                    }
                }
                else
                    child_hit[i] = false;
            }

            uint8_t *child_addr = node_addr + (node.ChildOffset * 64);
            for(int i = 0; i < 6; i++)
            {
                if(child_hit[i])
                {
                    if (debugTraversal)
                    {
                        traversalFile << "add child node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << ", to stack" << std::endl;
                    }
                    if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                    {
                        assert(node.ChildType[i] == NODE_TYPE_INSTANCE);
                        stack.push_back(StackEntry(child_addr, true, true));
                        rtcore_compact_trace.append_stack_push(
                            (uint64_t)child_addr + device_offset, true, true);
                        assert(tree_level_map.find(node_addr) != tree_level_map.end());
                        tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                    }
                    else
                    {
                        if(next_node_addr == NULL) {
                            next_node_addr = child_addr; // TODO: sort by thit
                            assert(tree_level_map.find(node_addr) != tree_level_map.end());
                            tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                        }
                        else {
                            stack.push_back(StackEntry(child_addr, true, false));
                            rtcore_compact_trace.append_stack_push(
                                (uint64_t)child_addr + device_offset, true,
                                false);
                            assert(tree_level_map.find(node_addr) != tree_level_map.end());
                            tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                        }
                    }
                }
                else
                {
                    if (debugTraversal)
                    {
                        traversalFile << "ignoring missed node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << std::endl;
                    }
                }
                child_addr += node.ChildSize[i] * 64;
            }

            if (debugTraversal)
            {
                traversalFile << std::endl;
            }
        }

        // traverse top level leaf nodes
        while (!stack.empty() && stack.back().leaf)
        {
            // TLAS offset
            device_offset = (uint64_t)tlas_addr - (uint64_t)_topLevelAS;

            assert(stack.back().topLevel);

            uint8_t* leaf_addr = stack.back().addr;
            rtcore_compact_trace.append_stack_pop(
                (uint64_t)leaf_addr + device_offset, stack.back().topLevel,
                stack.back().leaf);
            stack.pop_back();

            GEN_RT_BVH_INSTANCE_LEAF instanceLeaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&instanceLeaf, leaf_addr);
            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_INSTANCE_LEAF_length * 4, TransactionType::BVH_INSTANCE_LEAF));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INSTANCE_LEAF)]++;
            total_nodes_accessed++;
            rtcore_compact_trace.append_node_fetch(
                (uint64_t)leaf_addr + device_offset,
                GEN_RT_BVH_INSTANCE_LEAF_length * 4,
                rtcore_trace_node_fetch_flags(
                    true, RTCORE_TRACE_NODE_KIND_INSTANCE_LEAF));


            if (debugTraversal)
            {
                traversalFile << "traversing top level leaf node " << (void *)leaf_addr << ", instanceID = " << instanceLeaf.InstanceID << ", BVHAddress = " << instanceLeaf.BVHAddress << ", ShaderIndex = " << instanceLeaf.ShaderIndex << std::endl;
            }


            float4x4 worldToObjectMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.WorldToObjectm00);
            float4x4 objectToWorldMatrix = instance_leaf_matrix_to_float4x4(&instanceLeaf.ObjectToWorldm00);

            assert(instanceLeaf.BVHAddress != 0);
            GEN_RT_BVH botLevelASAddr;
            GEN_RT_BVH_unpack(&botLevelASAddr, (uint8_t *)(leaf_addr + instanceLeaf.BVHAddress));

            // BLAS offset
            uint8_t * botLevelRootAddr = (uint8_t *)(leaf_addr + instanceLeaf.BVHAddress);
            RT_DPRINTF("Traversing BLAS %p -> %p\n", (void*)botLevelRootAddr, blas_addr_map[(void*)botLevelRootAddr]);
            assert(blas_addr_map.find((void*)botLevelRootAddr) != blas_addr_map.end());
            device_offset = (uint64_t)blas_addr_map[(void*)botLevelRootAddr] - (uint64_t)botLevelRootAddr;

            transactions.push_back(MemoryTransactionRecord((uint8_t*)(botLevelRootAddr + device_offset), GEN_RT_BVH_length * 4, TransactionType::BVH_STRUCTURE));
            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_STRUCTURE)]++;
            rtcore_compact_trace.append_node_fetch(
                (uint64_t)botLevelRootAddr + device_offset,
                GEN_RT_BVH_length * 4,
                rtcore_trace_node_fetch_flags(
                    false, RTCORE_TRACE_NODE_KIND_BVH_HEADER));

            if (debugTraversal)
            {
                traversalFile << "bot level bvh " << (void *)(leaf_addr + instanceLeaf.BVHAddress) << ", RootNodeOffset = (" << botLevelASAddr.RootNodeOffset << std::endl;
            }

            // std::ofstream offsetfile;
            // offsetfile.open("offsets.txt", std::ios::app);
            // offsetfile << (int64_t)instanceLeaf.BVHAddress << std::endl;

            // std::ofstream leaf_addr_file;
            // leaf_addr_file.open("leaf.txt", std::ios::app);
            // leaf_addr_file << (int64_t)((uint64_t)leaf_addr - (uint64_t)_topLevelAS) << std::endl;

            float worldToObject_tMultiplier;
            Ray objectRay = make_transformed_ray(ray, worldToObjectMatrix, &worldToObject_tMultiplier);
            
            botLevelRootAddr = ((uint8_t *)((uint64_t)leaf_addr + instanceLeaf.BVHAddress)) + botLevelASAddr.RootNodeOffset;
            stack.push_back(StackEntry(botLevelRootAddr, false, false));
            rtcore_compact_trace.append_stack_push(
                (uint64_t)botLevelRootAddr + device_offset, false, false);
            assert(tree_level_map.find(leaf_addr) != tree_level_map.end());
            tree_level_map[botLevelRootAddr] = tree_level_map[leaf_addr];

            if (debugTraversal)
            {
                traversalFile << "bot level root address = " << (void*)botLevelRootAddr << std::endl;
                traversalFile << "warped ray to object coordinates, origin = (" << objectRay.get_origin().x << ", " << objectRay.get_origin().y << ", " << objectRay.get_origin().z << "), ";
                traversalFile << "direction = (" << objectRay.get_direction().x << ", " << objectRay.get_direction().y << ", " << objectRay.get_direction().z << "), ";
                traversalFile << "tmin = " << objectRay.get_tmin() << ", tmax = " << objectRay.get_tmax() << std::endl << std::endl;
            }

            // traverse bottom level tree
            while (!stack.empty() && !stack.back().topLevel)
            {
                uint8_t* node_addr = NULL;
                uint8_t* next_node_addr = stack.back().addr;
                rtcore_compact_trace.append_stack_pop(
                    (uint64_t)stack.back().addr + device_offset,
                    stack.back().topLevel, stack.back().leaf);
                stack.pop_back();
                

                // traverse bottom level internal nodes
                while (next_node_addr != NULL)
                {
                    node_addr = next_node_addr;
                    next_node_addr = NULL;

                    // if(node_addr == *(++path.rbegin()))
                    //     printf("this is where things go wrong\n");

                    struct GEN_RT_BVH_INTERNAL_NODE node;
                    GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);
                    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)node_addr + device_offset), GEN_RT_BVH_INTERNAL_NODE_length * 4, TransactionType::BVH_INTERNAL_NODE));
                    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_INTERNAL_NODE)]++;
                    total_nodes_accessed++;
                    rtcore_compact_trace.append_node_fetch(
                        (uint64_t)node_addr + device_offset,
                        GEN_RT_BVH_INTERNAL_NODE_length * 4,
                        rtcore_trace_node_fetch_flags(
                            false, RTCORE_TRACE_NODE_KIND_INTERNAL));

                    if (debugTraversal)
                    {
                        traversalFile << "traversing bot level internal node " << (void *)node_addr;
                        traversalFile << ", child offset = " << node.ChildOffset << ", node type = " << node.NodeType;
                        traversalFile << ", child size = (" << node.ChildSize[0] << ", " << node.ChildSize[1] << ", " << node.ChildSize[2] << ", " << node.ChildSize[3] << ", " << node.ChildSize[4] << ", " << node.ChildSize[5] << ")";
                        traversalFile << ", child type = (" << node.ChildType[0] << ", " << node.ChildType[1] << ", " << node.ChildType[2] << ", " << node.ChildType[3] << ", " << node.ChildType[4] << ", " << node.ChildType[5] << ")";
                        traversalFile << std::endl;
                    }

                    bool child_hit[6];
                    float thit[6];
                    for(int i = 0; i < 6; i++)
                    {
                        if (node.ChildSize[i] > 0)
                        {
                            float3 idir = calculate_idir(objectRay.get_direction()); //TODO: this works wierd if one of ray dimensions is 0
                            float3 lo, hi;
                            set_child_bounds(&node, i, &lo, &hi);

                            child_hit[i] = ray_box_test(lo, hi, idir, objectRay.get_origin(), objectRay.get_tmin(), objectRay.get_tmax(), thit[i]);
                            if(child_hit[i] && thit[i] >= min_thit * worldToObject_tMultiplier)
                                child_hit[i] = false;
                            rtcore_compact_trace.append_node_test(
                                (uint64_t)node_addr + device_offset, i,
                                child_hit[i], false);

                            if (debugTraversal)
                            {
                                if(child_hit[i])
                                    traversalFile << "hit child number " << i << ", ";
                                else
                                    traversalFile << "missed child number " << i << ", ";
                                traversalFile << "lo = (" << lo.x << ", " << lo.y << ", " << lo.z << "), ";
                                traversalFile << "hi = (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
                            }
                        }
                        else
                            child_hit[i] = false;
                    }

                    uint8_t *child_addr = node_addr + (node.ChildOffset * 64);
                    for(int i = 0; i < 6; i++)
                    {
                        if(child_hit[i])
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "add child node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << ", to stack" << std::endl;
                            }

                            if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                            {
                                stack.push_back(StackEntry(child_addr, false, true));
                                rtcore_compact_trace.append_stack_push(
                                    (uint64_t)child_addr + device_offset,
                                    false, true);
                                assert(tree_level_map.find(node_addr) != tree_level_map.end());
                                tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                            }
                            else
                            {
                                if(next_node_addr == 0) {
                                    next_node_addr = child_addr; // TODO: sort by thit
                                    assert(tree_level_map.find(node_addr) != tree_level_map.end());
                                    tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                                }
                                else {
                                    stack.push_back(StackEntry(child_addr, false, false));
                                    rtcore_compact_trace.append_stack_push(
                                        (uint64_t)child_addr + device_offset,
                                        false, false);
                                    assert(tree_level_map.find(node_addr) != tree_level_map.end());
                                    tree_level_map[child_addr] = tree_level_map[node_addr] + 1;
                                }
                            }
                        }
                        else
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "ignoring missed node " << (void *)child_addr << ", child number " << i << ", type " << node.ChildType[i] << std::endl;
                            }
                        }
                        child_addr += node.ChildSize[i] * 64;
                    }

                    if (debugTraversal)
                    {
                        traversalFile << std::endl;
                    }
                }

                // traverse bottom level leaf nodes
                while(!stack.empty() && !stack.back().topLevel && stack.back().leaf)
                {
                    uint8_t* leaf_addr = stack.back().addr;
                    rtcore_compact_trace.append_stack_pop(
                        (uint64_t)leaf_addr + device_offset,
                        stack.back().topLevel, stack.back().leaf);
                    stack.pop_back();
                    struct GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR leaf_descriptor;
                    GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_unpack(&leaf_descriptor, leaf_addr);
                    transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_length * 4, TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR));
                    ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PRIMITIVE_LEAF_DESCRIPTOR)]++;
                    rtcore_compact_trace.append_primitive_fetch(
                        (uint64_t)leaf_addr + device_offset,
                        GEN_RT_BVH_PRIMITIVE_LEAF_DESCRIPTOR_length * 4,
                        rtcore_trace_primitive_flags(
                            RTCORE_TRACE_PRIMITIVE_KIND_LEAF_DESCRIPTOR,
                            false, false));

                    if (leaf_descriptor.LeafType == TYPE_QUAD)
                    {
                        struct GEN_RT_BVH_QUAD_LEAF leaf;
                        GEN_RT_BVH_QUAD_LEAF_unpack(&leaf, leaf_addr);
                        rtcore_compact_trace.append_primitive_fetch(
                            (uint64_t)leaf_addr + device_offset,
                            GEN_RT_BVH_QUAD_LEAF_length * 4,
                            rtcore_trace_primitive_flags(
                                RTCORE_TRACE_PRIMITIVE_KIND_QUAD_LEAF, false,
                                false));

                        // if(leaf.PrimitiveIndex0 == 9600)
                        // {
                        //     leaf.QuadVertex[2].Z = -0.001213;
                        // }

                        float3 p[3];
                        for(int i = 0; i < 3; i++)
                        {
                            p[i].x = leaf.QuadVertex[i].X;
                            p[i].y = leaf.QuadVertex[i].Y;
                            p[i].z = leaf.QuadVertex[i].Z;
                        }

                        // Triangle intersection algorithm
                        float thit;
                        total_primitive_tests++;
                        bool hit = VulkanRayTracing::mt_ray_triangle_test(p[0], p[1], p[2], objectRay, &thit);
                        rtcore_compact_trace.append_primitive_test(
                            (uint64_t)leaf_addr + device_offset, hit,
                            rtcore_trace_primitive_flags(
                                RTCORE_TRACE_PRIMITIVE_KIND_TRIANGLE_TEST,
                                hit, false));

                        assert(leaf.PrimitiveIndex1Delta == 0);

                        if (debugTraversal)
                        {
                            if(hit)
                                traversalFile << "hit quad node " << (void *)leaf_addr << " with thit " << thit << " ";
                            else
                                traversalFile << "miss quad node " << leaf_addr << " ";
                            traversalFile << "primitiveID = " << leaf.PrimitiveIndex0 << ", InstanceID = " << instanceLeaf.InstanceID << "\n";

                            traversalFile << "p[0] = (" << p[0].x << ", " << p[0].y << ", " << p[0].z << ") ";
                            traversalFile << "p[1] = (" << p[1].x << ", " << p[1].y << ", " << p[1].z << ") ";
                            traversalFile << "p[2] = (" << p[2].x << ", " << p[2].y << ", " << p[2].z << ") ";
                            traversalFile << "p[3] = (" << p[3].x << ", " << p[3].y << ", " << p[3].z << ")" << std::endl;
                        }

                        float world_thit = thit / worldToObject_tMultiplier;

                        //TODO: why the Tmin Tmax consition wasn't handled in the object coordinates?
                        if(hit && Tmin <= world_thit && world_thit <= Tmax)
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "quad node " << (void *)leaf_addr << ", primitiveID " << leaf.PrimitiveIndex0 << " is the closest hit. world_thit " << thit / worldToObject_tMultiplier;
                            }

                            if (skipAnyHitShader && world_thit < min_thit) {
                                min_thit = thit / worldToObject_tMultiplier;
                            }
                            min_thit_object = thit;
                            closest_leaf = leaf;
                            closest_instanceLeaf = instanceLeaf;
                            closest_worldToObject = worldToObjectMatrix;
                            closest_objectToWorld = objectToWorldMatrix;
                            closest_objectRay = objectRay;
                            min_thit_object = thit;
                            thread->add_ray_intersect();
                            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_QUAD_LEAF_length * 4, TransactionType::BVH_QUAD_LEAF_HIT));
                            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF_HIT)]++;
                            total_nodes_accessed++;

                            if (!skipAnyHitShader) {
                                VSIM_DPRINTF("gpgpusim: Adding triangle intersection to anyhit shader table\n");
                                warp_intersection_table* table = anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
                                
                                uint32_t hit_group_index = instanceLeaf.InstanceContributionToHitGroupIndex;
                                auto intersectionTransactions = table->add_intersection(hit_group_index, thread->get_tid().x, leaf.PrimitiveIndex0, instanceLeaf.InstanceID, pI, thread); // TODO: switch these to device addresses

                                for(auto & newTransaction : intersectionTransactions.first)
                                {
                                    bool found = false;
                                    for(auto & transaction : transactions)
                                        if(transaction.address == newTransaction.address)
                                        {
                                            found = true;
                                            break;
                                        }
                                    if(!found)
                                        transactions.push_back(newTransaction);

                                }
                                store_transactions.insert(store_transactions.end(), intersectionTransactions.second.begin(), intersectionTransactions.second.end());

                                VSIM_DPRINTF("gpgpusim: Storing triangle intersection HitAttributes for anyhit shader\n");

                                ctx->func_sim->g_rt_num_any_hits++;

                                Hit_data anyhit_hit_attributes;
                                anyhit_hit_attributes.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                                anyhit_hit_attributes.geometry_index = leaf.LeafDescriptor.GeometryIndex;
                                anyhit_hit_attributes.primitive_index = leaf.PrimitiveIndex0;
                                anyhit_hit_attributes.instance_index = instanceLeaf.InstanceID;

                                float anyhit_thit = thit / worldToObject_tMultiplier;
                                float3 intersection_point = ray.get_origin() + make_float3(ray.get_direction().x * anyhit_thit, ray.get_direction().y * anyhit_thit, ray.get_direction().z * anyhit_thit);
                                float3 rayatinter = ray.at(anyhit_thit);

                                anyhit_hit_attributes.intersection_point = intersection_point;
                                anyhit_hit_attributes.worldToObjectMatrix = worldToObjectMatrix;
                                anyhit_hit_attributes.objectToWorldMatrix = objectToWorldMatrix;
                                anyhit_hit_attributes.world_min_thit = anyhit_thit;
                                
                                float3 p[3];
                                for(int i = 0; i < 3; i++)
                                {
                                    p[i].x = leaf.QuadVertex[i].X;
                                    p[i].y = leaf.QuadVertex[i].Y;
                                    p[i].z = leaf.QuadVertex[i].Z;
                                }
                                float3 object_intersection_point = objectRay.get_origin() + make_float3(objectRay.get_direction().x * thit, objectRay.get_direction().y * thit, objectRay.get_direction().z * thit);
                                float3 barycentric = Barycentric(object_intersection_point, p[0], p[1], p[2]);
                                anyhit_hit_attributes.barycentric_coordinates = barycentric;

                                VSIM_DPRINTF("gpgpusim: Ray hit geomID %d primID %d at (%5.3f, %5.3f, %5.3f) with t = %5.3f\n", anyhit_hit_attributes.geometry_index, anyhit_hit_attributes.primitive_index, barycentric.x, barycentric.y, barycentric.z, thit);

                                // Allocate memory to store hit attributes
                                memory_space *mem = thread->get_global_memory();
                                Hit_data* device_hit_attributes = (Hit_data*) VulkanRayTracing::gpgpusim_alloc(sizeof(Hit_data));
                                mem->write(device_hit_attributes, sizeof(Hit_data), &anyhit_hit_attributes, thread, pI);
                                thread->RT_thread_data->all_hit_data.push_back(device_hit_attributes);

                                traversal_data.n_all_hits++;
                                rtcore_compact_trace.append_hit_update(
                                    (uint64_t)device_hit_attributes,
                                    traversal_data.n_all_hits,
                                    rtcore_trace_hit_update_flags(
                                        RTCORE_TRACE_HIT_UPDATE_KIND_ANY_HIT));
                            }

                            if(terminateOnFirstHit)
                            {
                                rtcore_compact_trace.append_stack_clear(stack.size());
                                stack.clear();
                            }
                        }
                        else {
                            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_QUAD_LEAF_length * 4, TransactionType::BVH_QUAD_LEAF));
                            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF)]++;
                            total_nodes_accessed++;
                        }
                        if (debugTraversal)
                        {
                            traversalFile << std::endl;
                        }
                    }
                    else
                    {
                        hit_procedural = true;
                        struct GEN_RT_BVH_PROCEDURAL_LEAF leaf;
                        GEN_RT_BVH_PROCEDURAL_LEAF_unpack(&leaf, leaf_addr);
                        transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_PROCEDURAL_LEAF_length * 4, TransactionType::BVH_PROCEDURAL_LEAF));
                        ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_PROCEDURAL_LEAF)]++;
                        total_nodes_accessed++;
                        rtcore_compact_trace.append_primitive_fetch(
                            (uint64_t)leaf_addr + device_offset,
                            GEN_RT_BVH_PROCEDURAL_LEAF_length * 4,
                            rtcore_trace_primitive_flags(
                                RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_LEAF,
                                false, false));

                        uint32_t hit_group_index = instanceLeaf.InstanceContributionToHitGroupIndex;

                        warp_intersection_table* table = intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
                        auto intersectionTransactions = table->add_intersection(hit_group_index, thread->get_tid().x, leaf.PrimitiveIndex[0], instanceLeaf.InstanceID, pI, thread); // TODO: switch these to device addresses
                        rtcore_compact_trace.append_primitive_test(
                            (uint64_t)leaf_addr + device_offset, true,
                            rtcore_trace_primitive_flags(
                                RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_DEFERRED,
                                true, true));
                        
                        // transactions.insert(transactions.end(), intersectionTransactions.first.begin(), intersectionTransactions.first.end());
                        for(auto & newTransaction : intersectionTransactions.first)
                        {
                            bool found = false;
                            for(auto & transaction : transactions)
                                if(transaction.address == newTransaction.address)
                                {
                                    found = true;
                                    break;
                                }
                            if(!found)
                                transactions.push_back(newTransaction);

                        }
                        store_transactions.insert(store_transactions.end(), intersectionTransactions.second.begin(), intersectionTransactions.second.end());
                    }
                }
            }
        }
    }

    if (min_thit < ray.dir_tmax.w)
    {
        traversal_data.hit_geometry = true;
        ctx->func_sim->g_rt_num_hits++;
        traversal_data.closest_hit.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        traversal_data.closest_hit.geometry_index = closest_leaf.LeafDescriptor.GeometryIndex;
        traversal_data.closest_hit.primitive_index = closest_leaf.PrimitiveIndex0;
        traversal_data.closest_hit.instance_index = closest_instanceLeaf.InstanceID;
        float3 intersection_point = ray.get_origin() + make_float3(ray.get_direction().x * min_thit, ray.get_direction().y * min_thit, ray.get_direction().z * min_thit);
        float3 rayatinter = ray.at(min_thit);
        // assert(intersection_point.x == ray.at(min_thit).x && intersection_point.y == ray.at(min_thit).y && intersection_point.z == ray.at(min_thit).z);
        traversal_data.closest_hit.intersection_point = intersection_point;
        traversal_data.closest_hit.worldToObjectMatrix = closest_worldToObject;
        traversal_data.closest_hit.objectToWorldMatrix = closest_objectToWorld;
        traversal_data.closest_hit.world_min_thit = min_thit;

        VSIM_DPRINTF("gpgpusim: Ray hit geomID %d primID %d\n", traversal_data.closest_hit.geometry_index, traversal_data.closest_hit.primitive_index);
        VSIM_DPRINTF("gpgpusim: Ray [%d] awaiting %d anyhit shader calls\n", thread->get_uid(), traversal_data.n_all_hits);
        assert(thread->RT_thread_data->all_hit_data.size() == traversal_data.n_all_hits);
        float3 p[3];
        for(int i = 0; i < 3; i++)
        {
            p[i].x = closest_leaf.QuadVertex[i].X;
            p[i].y = closest_leaf.QuadVertex[i].Y;
            p[i].z = closest_leaf.QuadVertex[i].Z;
        }
        float3 object_intersection_point = closest_objectRay.get_origin() + make_float3(closest_objectRay.get_direction().x * min_thit_object, closest_objectRay.get_direction().y * min_thit_object, closest_objectRay.get_direction().z * min_thit_object);
        //closest_objectRay.at(min_thit_object);
        float3 barycentric = Barycentric(object_intersection_point, p[0], p[1], p[2]);
        traversal_data.closest_hit.barycentric_coordinates = barycentric;
        rtcore_compact_trace.append_hit_update(
            closest_leaf.PrimitiveIndex0, traversal_data.n_all_hits,
            rtcore_trace_hit_update_flags(
                RTCORE_TRACE_HIT_UPDATE_KIND_CLOSEST_HIT));
        thread->RT_thread_data->set_hitAttribute(barycentric, pI, thread);

        // store_transactions.push_back(MemoryStoreTransactionRecord(&traversal_data, sizeof(traversal_data), StoreTransactionType::Traversal_Results));
    }
    else if (hit_procedural)
    {
        VSIM_DPRINTF("gpgpusim: Ray hit procedural geometry; requires intersection shader.\n");
        traversal_data.hit_geometry = false;
    }
    else
    {
        VSIM_DPRINTF("gpgpusim: Ray [%d] missed.\n", thread->get_uid());
        traversal_data.hit_geometry = false;
    }

    if (pixel_trace_enabled) {
        if (traversal_data.hit_geometry) {
            printf("GPGPU-Sim RTCORE_PIXEL_TRACE trace-ray-result "
                   "launch=(%u,%u), thread_uid=%u, result=closest-hit, "
                   "geometry=%u, primitive=%u, instance=%u, t=%.9g, "
                   "bary=(%.9g,%.9g,%.9g), intersection=(%.9g,%.9g,%.9g), "
                   "nodes=%u, primitive_tests=%u\n",
                   rtcore_launch_id_x_for_thread(thread),
                   rtcore_launch_id_y_for_thread(thread), thread->get_uid(),
                   traversal_data.closest_hit.geometry_index,
                   traversal_data.closest_hit.primitive_index,
                   traversal_data.closest_hit.instance_index,
                   traversal_data.closest_hit.world_min_thit,
                   traversal_data.closest_hit.barycentric_coordinates.x,
                   traversal_data.closest_hit.barycentric_coordinates.y,
                   traversal_data.closest_hit.barycentric_coordinates.z,
                   traversal_data.closest_hit.intersection_point.x,
                   traversal_data.closest_hit.intersection_point.y,
                   traversal_data.closest_hit.intersection_point.z,
                   total_nodes_accessed, total_primitive_tests);
        } else {
            printf("GPGPU-Sim RTCORE_PIXEL_TRACE trace-ray-result "
                   "launch=(%u,%u), thread_uid=%u, result=%s, nodes=%u, "
                   "primitive_tests=%u\n",
                   rtcore_launch_id_x_for_thread(thread),
                   rtcore_launch_id_y_for_thread(thread), thread->get_uid(),
                   hit_procedural ? "procedural-deferred" : "miss",
                   total_nodes_accessed, total_primitive_tests);
        }
    }

    memory_space *mem = thread->get_global_memory();
    Traversal_data* device_traversal_data = (Traversal_data*) VulkanRayTracing::gpgpusim_alloc(sizeof(Traversal_data));
    traversal_data.rtcore_node_visits = total_nodes_accessed;
    traversal_data.rtcore_primitive_tests = total_primitive_tests;
    rtcore_compact_trace.append_completion_summary(total_nodes_accessed,
                                                   total_primitive_tests);
    rtcore_compact_trace_export_record rtcore_trace_export = rtcore_compact_trace.export_record();
    rtcore_publish_compact_trace_export(thread, rtcore_trace_export);
    rtcore_admit_compact_trace_for_replay(thread);
    mem->write(device_traversal_data, sizeof(Traversal_data), &traversal_data, thread, pI);
    thread->RT_thread_data->traversal_data.push_back(device_traversal_data);
    
    thread->set_rt_transactions(transactions);
    thread->set_rt_store_transactions(store_transactions);

    if (debugTraversal)
    {
        traversalFile.close();
    }

    if (total_nodes_accessed > ctx->func_sim->g_max_nodes_per_ray) {
        ctx->func_sim->g_max_nodes_per_ray = total_nodes_accessed;
    }
    ctx->func_sim->g_tot_nodes_per_ray += total_nodes_accessed;

    unsigned level = 0;
    for (auto it=tree_level_map.begin(); it!=tree_level_map.end(); it++) {
        if (it->second > level) {
            level = it->second;
        }
    }
    if (level > ctx->func_sim->g_max_tree_depth) {
        ctx->func_sim->g_max_tree_depth = level;
    }

    RT_DPRINTF("Traversal: \n");
    for (auto t : transactions) {
        RT_DPRINTF("\ttransaction %d, address %p, size %d\n", t.type, t.address, t.size);
    }
}

void VulkanRayTracing::endTraceRay(const ptx_instruction *pI, ptx_thread_info *thread)
{
    assert(thread->RT_thread_data->traversal_data.size() > 0);
    thread->RT_thread_data->traversal_data.pop_back();
    thread->RT_thread_data->all_hit_data.clear();
    warp_intersection_table* itable = intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
    itable->clear(pI, thread);
    warp_intersection_table* atable = anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
    atable->clear(pI, thread);
}

bool VulkanRayTracing::mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit)
{
    // Moller Trumbore algorithm (from scratchapixel.com)
    float3 v0v1 = p1 - p0;
    float3 v0v2 = p2 - p0;
    float3 pvec = cross(ray_properties.get_direction(), v0v2);
    float det = dot(v0v1, pvec);

    float idet = 1 / det;

    float3 tvec = ray_properties.get_origin() - p0;
    float u = dot(tvec, pvec) * idet;

    if (u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray_properties.get_direction(), qvec) * idet;

    if (v < 0 || (u + v) > 1) return false;

    *thit = dot(v0v2, qvec) * idet;
    return true;
}

float3 VulkanRayTracing::Barycentric(float3 p, float3 a, float3 b, float3 c)
{
    //source: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    float3 v0 = b - a;
    float3 v1 = c - a;
    float3 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return {v, w, u};
}

void VulkanRayTracing::load_descriptor(const ptx_instruction *pI, ptx_thread_info *thread)
{

}


void VulkanRayTracing::setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos)
{
    VulkanRayTracing::pCreateInfos = pCreateInfos;
	std::cout << "gpgpusim: set pipeline" << std::endl;
}


void VulkanRayTracing::setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount)
{
    VulkanRayTracing::pGeometries = pGeometries;
    VulkanRayTracing::geometryCount = geometryCount;
	std::cout << "gpgpusim: set geometry" << std::endl;
}

void VulkanRayTracing::setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure)
{
    GEN_RT_BVH topBVH; //TODO: test hit with world before traversal
    GEN_RT_BVH_unpack(&topBVH, (uint8_t *)accelerationStructure);




    std::cout << "gpgpusim: set AS" << std::endl;
    VulkanRayTracing::topLevelAS = accelerationStructure;
}

std::string base_name(std::string & path)
{
  return path.substr(path.find_last_of("/") + 1);
}

void VulkanRayTracing::setDescriptorSet(struct DESCRIPTOR_SET_STRUCT *set)
{
    if (VulkanRayTracing::descriptorSet == NULL) {
        printf("gpgpusim: set descriptor set 0x%x\n", set);
        VulkanRayTracing::descriptorSet = set;
    }
    // TODO: Figure out why it sets the descriptor set twice
    else {
        printf("gpgpusim: descriptor set already set; ignoring update.\n");
    }
}

static bool invoked = false;

void copyHardCodedShaders()
{
    std::ifstream  src;
    std::ofstream  dst;

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_MISS_2.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_MISS_2.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();
    
    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_CLOSEST_HIT_2.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_CLOSEST_HIT_2.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_RAYGEN_0.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_RAYGEN_0.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // src.open("/home/mrs/emerald-ray-tracing/hardcodeShader/MESA_SHADER_INTERSECTION_4.ptx", std::ios::binary);
    // dst.open("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_INTERSECTION_4.ptx", std::ios::binary);
    // dst << src.rdbuf();
    // src.close();
    // dst.close();

    // {
    //     std::ifstream  src("/home/mrs/emerald-ray-tracing/MESA_SHADER_MISS_0.ptx", std::ios::binary);
    //     std::ofstream  dst("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_MISS_1.ptx",   std::ios::binary);
    //     dst << src.rdbuf();
    //     src.close();
    //     dst.close();
    // }
}

uint32_t VulkanRayTracing::registerShaders(char * shaderPath, gl_shader_stage shaderType)
{
    printf("gpgpusim: register shaders\n");
    copyHardCodedShaders();

    VulkanRayTracing::invoke_gpgpusim();
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    // Register all the ptx files in $MESA_ROOT/gpgpusimShaders by looping through them
    // std::vector <std::string> ptx_list;

    // Add ptx file names in gpgpusimShaders folder to ptx_list
    char *mesa_root = getenv("MESA_ROOT");
    char *gpgpusim_root = getenv("GPGPUSIM_ROOT");
    // char *filePath = "gpgpusimShaders/";
    // char fullPath[200];
    // snprintf(fullPath, sizeof(fullPath), "%s%s", mesa_root, filePath);
    // std::string fullPathString(fullPath);

    // for (auto &p : fs::recursive_directory_iterator(fullPathString))
    // {
    //     if (p.path().extension() == ".ptx")
    //     {
    //         //std::cout << p.path().string() << '\n';
    //         ptx_list.push_back(p.path().string());
    //     }
    // }

    std::string fullpath(shaderPath);
    std::string fullfilename = base_name(fullpath);
    std::string filenameNoExt;
    size_t start = fullfilename.find_first_not_of('.', 0);
    size_t end = fullfilename.find('.', start);
    filenameNoExt = fullfilename.substr(start, end - start);
    std::string idInString = filenameNoExt.substr(filenameNoExt.find_last_of("_") + 1);
    // Register each ptx file in ptx_list
    shader_stage_info shader;
    //shader.ID = VulkanRayTracing::shaders.size();
    shader.ID = std::stoi(idInString);
    shader.type = shaderType;
    shader.function_name = (char*)malloc(200 * sizeof(char));

    std::string deviceFunction;

    switch(shaderType) {
        case MESA_SHADER_RAYGEN:
            // shader.function_name = "raygen_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "raygen_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_RAYGEN";
            break;
        case MESA_SHADER_ANY_HIT:
            // shader.function_name = "anyhit_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "anyhit_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_ANY_HIT";
            break;
        case MESA_SHADER_CLOSEST_HIT:
            // shader.function_name = "closesthit_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "closesthit_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_CLOSEST_HIT";
            break;
        case MESA_SHADER_MISS:
            // shader.function_name = "miss_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "miss_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_MISS";
            break;
        case MESA_SHADER_INTERSECTION:
            // shader.function_name = "intersection_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "intersection_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "MESA_SHADER_INTERSECTION";
            break;
        case MESA_SHADER_CALLABLE:
            // shader.function_name = "callable_" + std::to_string(shader.ID);
            strcpy(shader.function_name, "callable_");
            strcat(shader.function_name, std::to_string(shader.ID).c_str());
            deviceFunction = "";
            assert(0);
            break;
    }
    deviceFunction += "_func" + std::to_string(shader.ID) + "_main";
    // deviceFunction += "_main";

    symbol_table *symtab;
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 20;
    unsigned selected_capability = 20;
    bool found = false;
    
    unsigned long long fat_cubin_handle = shader.ID;

    // PTX File
    //std::cout << itr << std::endl;
    symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(shaderPath);
    context->add_binary(symtab, fat_cubin_handle);
    // need to add all the magic registers to ptx.l to special_register, reference ayub ptx.l:225

    // PTX info
    // Run the python script and get ptxinfo
    std::cout << "GPGPUSIM: Generating PTXINFO for" << shaderPath << "info" << std::endl;
    char command[400];
    snprintf(command, sizeof(command), "python3 %s/scripts/generate_rt_ptxinfo.py %s", gpgpusim_root, shaderPath);
    int result = system(command);
    if (result != 0) {
        printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
        printf("               Ensure ptxas is in your path.\n");
        exit(1);
    }
    
    char ptxinfo_filename[400];
    snprintf(ptxinfo_filename, sizeof(ptxinfo_filename), "%sinfo", shaderPath);
    ctx->gpgpu_ptx_info_load_from_external_file(ptxinfo_filename); // TODO: make a version where it just loads my ptxinfo instead of generating a new one

    context->register_function(fat_cubin_handle, shader.function_name, deviceFunction.c_str());

    VulkanRayTracing::shaders.push_back(shader);

    return shader.ID;

    // if (itr.find("RAYGEN") != std::string::npos)
    // {
    //     printf("############### registering %s\n", shaderPath);
    //     context->register_function(fat_cubin_handle, "raygen_shader", "MESA_SHADER_RAYGEN_main");
    // }

    // if (itr.find("MISS") != std::string::npos)
    // {
    //     printf("############### registering %s\n", shaderPath);
    //     context->register_function(fat_cubin_handle, "miss_shader", "MESA_SHADER_MISS_main");
    // }

    // if (itr.find("CLOSEST") != std::string::npos)
    // {
    //     printf("############### registering %s\n", shaderPath);
    //     context->register_function(fat_cubin_handle, "closest_hit_shader", "MESA_SHADER_CLOSEST_HIT_main");
    // }
}


void VulkanRayTracing::invoke_gpgpusim()
{
    printf("gpgpusim: invoking gpgpusim\n");
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    if(!invoked)
    {
        //registerShaders();
        invoked = true;
    }
}

// int CmdTraceRaysKHRID = 0;

const bool writeImageBinary = true;

void VulkanRayTracing::vkCmdTraceRaysKHR(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr) {
    printf("gpgpusim: launching cmd trace ray\n");
    // launch_width = 224;
    // launch_height = 160;
    init(launch_width, launch_height);
    
    // Dump Descriptor Sets
    if (dump_trace) 
    {
        dump_descriptor_sets(VulkanRayTracing::descriptorSet);
        dump_callparams_and_sbt(raygen_sbt, miss_sbt, hit_sbt, callable_sbt, is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
    }

    // CmdTraceRaysKHRID++;
    // if(CmdTraceRaysKHRID != 1)
    //     return;
    // launch_width = 420;
    // launch_height = 320;

    if(writeImageBinary && !imageFile.is_open())
    {
        char* imageFileName;
        char defaultFileName[40] = "image.binary";
        if(getenv("VULKAN_IMAGE_FILE_NAME"))
            imageFileName = getenv("VULKAN_IMAGE_FILE_NAME");
        else
            imageFileName = defaultFileName;
        imageFile.open(imageFileName, std::ios::out | std::ios::binary);
        
        // imageFile.open("image.txt", std::ios::out);
    }
    // memset(((uint8_t*)descriptors[0][1].address), uint8_t(127), launch_height * launch_width * 4);
    // return;

    // {
    //     std::ifstream infile("debug_printf.log");
    //     std::string line;
    //     while (std::getline(infile, line))
    //     {
    //         if(line == "")
    //             continue;

    //         RayDebugGPUData data;
    //         // sscanf(line.c_str(), "LaunchID:(%d,%d), InstanceCustomIndex = %d, primitiveID = %d, v0 = (%f, %f, %f), v1 = (%f, %f, %f), v2 = (%f, %f, %f), hitAttribute = (%f, %f), normalWorld = (%f, %f, %f), objectIntersection = (%f, %f, %f), worldIntersection = (%f, %f, %f), objectNormal = (%f, %f, %f), worldNormal = (%f, %f, %f), NdotL = %f",
    //         //             &data.launchIDx, &data.launchIDy, &data.instanceCustomIndex, &data.primitiveID, &data.v0pos.x, &data.v0pos.y, &data.v0pos.z, &data.v1pos.x, &data.v1pos.y, &data.v1pos.z, &data.v2pos.x, &data.v2pos.y, &data.v2pos.z, &data.attribs.x, &data.attribs.y, &data.N.x, &data.N.y, &data.N.z, &data.P_object.x, &data.P_object.y, &data.P_object.z, &data.P.x, &data.P.y, &data.P.z, &data.N_object.x, &data.N_object.y, &data.N_object.z, &data.N.x, &data.N.y, &data.N.z, &data.NdotL);
    //         sscanf(line.c_str(), "launchID = (%d, %d), hitValue = (%f, %f, %f)",
    //                     &data.launchIDx, &data.launchIDy, &data.hitValue.x, &data.hitValue.y, &data.hitValue.z);
    //         data.valid = true;
    //         assert(data.launchIDx < 2000 && data.launchIDy < 2000);
    //         // printf("#### (%d, %d)\n", data.launchIDx, data.launchIDy);
    //         // fflush(stdout);
    //         rayDebugGPUData[data.launchIDx][data.launchIDy] = data;

    //     }
    // }

    assert(launch_depth == 1);

#if defined(MESA_USE_INTEL_DRIVER)
    struct DESCRIPTOR_STRUCT desc;
    desc.image_view = NULL;
#endif

    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    unsigned long shaderId = *(uint64_t*)raygen_sbt;
    int index = 0;
    for (int i = 0; i < shaders.size(); i++) {
        if (shaders[i].ID == 0){
            index = i;
            break;
        }
    }
    ctx->func_sim->g_total_shaders = shaders.size();

    shader_stage_info raygen_shader = shaders[index];
    function_info *entry = context->get_kernel(raygen_shader.function_name);
    // printf("################ number of args = %d\n", entry->num_args());

    if (entry->is_pdom_set()) {
        printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
            entry->get_name().c_str());
    } else {
        printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
            entry->get_name().c_str());
        /*
        * Some of the instructions like printf() gives the gpgpusim the wrong
        * impression that it is a function call. As printf() doesnt have a body
        * like functions do, doing pdom analysis for printf() causes a crash.
        */
        if (entry->get_function_size() > 0) entry->do_pdom();
        entry->set_pdom();
    }

    // check that number of args and return match function requirements
    //if (pI->has_return() ^ entry->has_return()) {
    //    printf(
    //        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
    //        "between\n"
    //        "               call instruction and function declaration\n");
    //    abort();
    //}
    unsigned n_return = entry->has_return();
    unsigned n_args = entry->num_args();
    //unsigned n_operands = pI->get_num_operands();

    // launch_width = 192;
    // launch_height = 32;

    dim3 blockDim = dim3(1, 1, 1);
    dim3 gridDim = dim3(1, launch_height, launch_depth);
    if(launch_width <= 32) {
        blockDim.x = launch_width;
        gridDim.x = 1;
    }
    else {
        blockDim.x = 32;
        gridDim.x = launch_width / 32;
        if(launch_width % 32 != 0)
            gridDim.x++;
    }
    printf("gpgpusim: launch dimensions %d x %d x %d\n", gridDim.x, gridDim.y, gridDim.z);

    gpgpu_ptx_sim_arg_list_t args;
    // kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
    //   raygen_shader.function_name, args, dim3(4, 128, 1), dim3(32, 1, 1), context);
    kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      raygen_shader.function_name, args, gridDim, blockDim, context);
    grid->vulkan_metadata.raygen_sbt = raygen_sbt;
    grid->vulkan_metadata.miss_sbt = miss_sbt;
    grid->vulkan_metadata.hit_sbt = hit_sbt;
    grid->vulkan_metadata.callable_sbt = callable_sbt;
    grid->vulkan_metadata.launch_width = launch_width;
    grid->vulkan_metadata.launch_height = launch_height;
    grid->vulkan_metadata.launch_depth = launch_depth;
    
    printf("gpgpusim: SBT: raygen %p, miss %p, hit %p, callable %p\n", 
            raygen_sbt, miss_sbt, hit_sbt, callable_sbt);

    printf("gpgpusim: blas address\n");
    for (auto mapping : blas_addr_map) {
        printf("\t[%p] -> %p\n", mapping.first, mapping.second);
    }

    printf("gpgpusim: tlas address %p\n", tlas_addr);
            
    struct CUstream_st *stream = 0;
    stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
    ctx->the_gpgpusim->g_stream_manager->push(op);

    //printf("%d\n", descriptors[0][1].address);

    fflush(stdout);

    while(!op.is_done() && !op.get_kernel()->done()) {
        if (rt_progress_logging_enabled()) {
            printf("gpgpusim: waiting for op to finish (kernel_uid=%u done=%d is_finished=%d)\n",
                   op.get_kernel()->get_uid(), op.get_kernel()->done(), op.get_kernel()->is_finished());
        } else {
            printf("waiting for op to finish\n");
        }
        sleep(1);
        continue;
    }
    // for (unsigned i = 0; i < entry->num_args(); i++) {
    //     std::pair<size_t, unsigned> p = entry->get_param_config(i);
    //     cudaSetupArgumentInternal(args[i], p.first, p.second);
    // }
}

void VulkanRayTracing::callMissShader(const ptx_instruction *pI, ptx_thread_info *thread) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

    bool hit_geometry;
    mem->read(&(traversal_data->hit_geometry), sizeof(bool), &hit_geometry);
    assert(!hit_geometry);

    int32_t current_shader_counter = -1;
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &current_shader_counter, thread, pI);

    int32_t current_shader_type = -1;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    uint32_t missIndex;
    mem->read(&(traversal_data->missIndex), sizeof(traversal_data->missIndex), &missIndex);

    uint32_t shaderID = *((uint32_t *)(thread->get_kernel().vulkan_metadata.miss_sbt) + 8 * missIndex);
    VSIM_DPRINTF("gpgpusim: Calling Miss Shader at ID %d\n", shaderID);

    shader_stage_info miss_shader = shaders[shaderID];

    function_info *entry = context->get_kernel(miss_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callClosestHitShader(const ptx_instruction *pI, ptx_thread_info *thread) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();

    bool hit_geometry;
    mem->read(&(traversal_data->hit_geometry), sizeof(bool), &hit_geometry);
    assert(hit_geometry);

    int32_t current_shader_counter = -1;
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &current_shader_counter, thread, pI);

    int32_t current_shader_type = -1;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    VkGeometryTypeKHR geometryType;
    mem->read(&(traversal_data->closest_hit.geometryType), sizeof(traversal_data->closest_hit.geometryType), &geometryType);

    shader_stage_info closesthit_shader;
    if(geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
        uint32_t shaderID = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt));
        closesthit_shader = shaders[shaderID];
        VSIM_DPRINTF("gpgpusim: Calling Closest Hit Shader at ID %d\n", shaderID);

    }
    else {
        int32_t hitGroupIndex;
        mem->read(&(traversal_data->closest_hit.hitGroupIndex), sizeof(traversal_data->closest_hit.hitGroupIndex), &hitGroupIndex);
        uint32_t shaderID = *((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex);
        closesthit_shader = shaders[shaderID];
        VSIM_DPRINTF("gpgpusim: Calling Closest Hit Shader at ID %d\n", shaderID);
    }

    function_info *entry = context->get_kernel(closesthit_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callIntersectionShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter) {
    VSIM_DPRINTF("gpgpusim: Calling Intersection Shader\n");
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    
    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter, thread, pI);

    int32_t current_shader_type = 1;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    warp_intersection_table* table = VulkanRayTracing::intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
    uint32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);

    shader_stage_info intersection_shader = shaders[*((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex + 1)];
    function_info *entry = context->get_kernel(intersection_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callAnyHitShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter) {
    VSIM_DPRINTF("gpgpusim: Calling Any Hit Shader\n");
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    memory_space *mem = thread->get_global_memory();
    Traversal_data* traversal_data = thread->RT_thread_data->traversal_data.back();
    mem->write(&(traversal_data->current_shader_counter), sizeof(traversal_data->current_shader_counter), &shader_counter, thread, pI);

    int32_t current_shader_type = 2;
    mem->write(&(traversal_data->current_shader_type), sizeof(traversal_data->current_shader_type), &current_shader_type, thread, pI);

    assert(shader_counter < thread->RT_thread_data->all_hit_data.size());
    Hit_data anyhit_hit_attributes;
    mem->read(thread->RT_thread_data->all_hit_data[shader_counter],
              sizeof(anyhit_hit_attributes), &anyhit_hit_attributes);
    thread->RT_thread_data->set_hitAttribute(
        anyhit_hit_attributes.barycentric_coordinates, pI, thread);

    warp_intersection_table* table = VulkanRayTracing::anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
    uint32_t hitGroupIndex = table->get_hitGroupIndex(shader_counter, thread->get_tid().x, pI, thread);

    shader_stage_info anyhit_shader = shaders[*((uint32_t *)(thread->get_kernel().vulkan_metadata.hit_sbt) + 8 * hitGroupIndex + 1)];
    function_info *entry = context->get_kernel(anyhit_shader.function_name);
    callShader(pI, thread, entry);
}

void VulkanRayTracing::callShader(const ptx_instruction *pI, ptx_thread_info *thread, function_info *target_func) {
    static unsigned call_uid_next = 1;

  if (target_func->is_pdom_set()) {
    // printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
    //        target_func->get_name().c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           target_func->get_name().c_str());
    /*
     * Some of the instructions like printf() gives the gpgpusim the wrong
     * impression that it is a function call. As printf() doesnt have a body
     * like functions do, doing pdom analysis for printf() causes a crash.
     */
    if (target_func->get_function_size() > 0) target_func->do_pdom();
    target_func->set_pdom();
  }

  thread->set_npc(target_func->get_start_PC());

  // check that number of args and return match function requirements
  if (pI->has_return() ^ target_func->has_return()) {
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
        "between\n"
        "               call instruction and function declaration\n");
    abort();
  }
  unsigned n_return = target_func->has_return();
  unsigned n_args = target_func->num_args();
  unsigned n_operands = pI->get_num_operands();

  // TODO: why this fails?
//   if (n_operands != (n_return + 1 + n_args)) {
//     printf(
//         "GPGPU-Sim PTX: Execution error - mismatch in number of arguements "
//         "between\n"
//         "               call instruction and function declaration\n");
//     abort();
//   }

  // handle intrinsic functions
//   std::string fname = target_func->get_name();
//   if (fname == "vprintf") {
//     gpgpusim_cuda_vprintf(pI, thread, target_func);
//     return;
//   }
// #if (CUDART_VERSION >= 5000)
//   // Jin: handle device runtime apis for CDP
//   else if (fname == "cudaGetParameterBufferV2") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_getParameterBufferV2(
//         pI, thread, target_func);
//     return;
//   } else if (fname == "cudaLaunchDeviceV2") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_launchDeviceV2(
//         pI, thread, target_func);
//     return;
//   } else if (fname == "cudaStreamCreateWithFlags") {
//     target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_streamCreateWithFlags(
//         pI, thread, target_func);
//     return;
//   }
// #endif

  // read source arguements into register specified in declaration of function
  arg_buffer_list_t arg_values;
  copy_args_into_buffer_list(pI, thread, target_func, arg_values);

  // record local for return value (we only support a single return value)
  const symbol *return_var_src = NULL;
  const symbol *return_var_dst = NULL;
  if (target_func->has_return()) {
    return_var_dst = pI->dst().get_symbol();
    return_var_src = target_func->get_return_var();
  }

  gpgpu_sim *gpu = thread->get_gpu();
  unsigned callee_pc = 0, callee_rpc = 0;
  /*if (gpu->simd_model() == POST_DOMINATOR)*/ { //MRS_TODO: why this fails?
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc);
    assert(callee_pc == thread->get_pc());
  }

  thread->callstack_push(callee_pc + pI->inst_size(), callee_rpc,
                         return_var_src, return_var_dst, call_uid_next++);

  copy_buffer_list_into_frame(thread, arg_values);

  thread->set_npc(target_func);
}

void VulkanRayTracing::setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    printf("gpgpusim: set descriptor\n");
    if(descriptors.size() <= setID)
        descriptors.resize(setID + 1);
    if(descriptors[setID].size() <= descID)
        descriptors[setID].resize(descID + 1);
    
    descriptors[setID][descID].setID = setID;
    descriptors[setID][descID].descID = descID;
    descriptors[setID][descID].address = address;
    descriptors[setID][descID].size = size;
    descriptors[setID][descID].type = type;
}


void VulkanRayTracing::setDescriptorSetFromLauncher(void *address, void *deviceAddress, uint32_t setID, uint32_t descID)
{
    launcher_deviceDescriptorSets[setID][descID] = deviceAddress;
    launcher_descriptorSets[setID][descID] = address;
}

void* VulkanRayTracing::getDescriptorAddress(uint32_t setID, uint32_t binding)
{
#if defined(MESA_USE_INTEL_DRIVER)
    if (use_external_launcher)
    {
        return launcher_deviceDescriptorSets[setID][binding];
        // return launcher_descriptorSets[setID][binding];
    }
    else 
    {
        // assert(setID < descriptors.size());
        // assert(binding < descriptors[setID].size());

        struct anv_descriptor_set* set = VulkanRayTracing::descriptorSet;

        const struct anv_descriptor_set_binding_layout *bind_layout = &set->layout->binding[binding];
        struct anv_descriptor *desc = &set->descriptors[bind_layout->descriptor_index];
        void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

        assert(desc->type == bind_layout->type);

        switch (desc->type)
        {
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            {
                return (void *)(desc);
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            {
                return desc;
            }

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            {
                if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                    desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                {
                    // MRS_TODO: account for desc->offset?
                    return anv_address_map(desc->buffer->address);
                }
                else
                {
                    struct anv_buffer_view *bview = &set->buffer_views[bind_layout->buffer_view_index];
                    return anv_address_map(bview->address);
                }
            }

            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                return (void *)(desc_data->address);
            }

            default:
                assert(0);
                break;
        }

        // return descriptors[setID][binding].address;
    }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    VSIM_DPRINTF("gpgpusim: getDescriptorAddress for binding %d\n", binding);
    struct lvp_descriptor_set* set = VulkanRayTracing::descriptorSet;
    const struct lvp_descriptor_set_binding_layout *bind_layout = &set->layout->binding[binding];
    struct lvp_descriptor *desc = &set->descriptors[bind_layout->descriptor_index];

    // printf("DESCRIPTOR TYPE: %d\n", desc->type);
    switch (desc->type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            VSIM_DPRINTF("gpgpusim: storage image; descriptor address %p\n", desc);
            return (void *) desc;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            VSIM_DPRINTF("gpgpusim: uniform buffer; buffer mem address %p\n", (void *) desc->info.ubo.pmem);
            return (void *) desc->info.ubo.pmem;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            VSIM_DPRINTF("gpgpusim: storage buffer; buffer mem address %p\n", (void *) desc->info.ssbo.pmem);
            return (void *) desc->info.ssbo.pmem;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VSIM_DPRINTF("gpgpusim: accel struct; root address %p\n", (void *)desc->info.ubo.pmem + desc->info.ubo.buffer_offset);
            return (void *)desc->info.ubo.pmem + desc->info.ubo.buffer_offset;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VSIM_DPRINTF("gpgpusim: image sampler; descriptor address %p\n", desc);
            return (void *) desc;
            break;
        default:
            VSIM_DPRINTF("gpgpusim: unimplemented descriptor type\n");
            abort();
    }
#endif
}

void VulkanRayTracing::getTexture(struct DESCRIPTOR_STRUCT *desc, 
                                    float x, float y, float lod, 
                                    float &c0, float &c1, float &c2, float &c3, 
                                    std::vector<ImageMemoryTransactionRecord>& transactions,
                                    uint64_t launcher_offset)
{
#if defined(MESA_USE_INTEL_DRIVER)
    Pixel pixel;

    if (use_external_launcher)
    {
        pixel = get_interpolated_pixel((anv_image_view*) desc, (anv_sampler*) desc, x, y, transactions, launcher_offset); // cast back to metadata later
    }
    else 
    {
        struct anv_image_view *image_view =  desc->image_view;
        struct anv_sampler *sampler = desc->sampler;

        const struct anv_image *image = image_view->image;
        assert(image->n_planes == 1);
        assert(image->samples == 1);
        assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
        assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
        assert(sampler->conversion == NULL);

        pixel = get_interpolated_pixel(image_view, sampler, x, y, transactions);
    }

    TXL_DPRINTF("Setting transaction type to TEXTURE_LOAD\n");
    for(int i = 0; i < transactions.size(); i++)
        transactions[i].type = ImageTransactionType::TEXTURE_LOAD;
    
    c0 = pixel.c0;
    c1 = pixel.c1;
    c2 = pixel.c2;
    c3 = pixel.c3;


    // uint8_t* address = anv_address_map(image->planes[0].address);

    // for(int x = 0; x < image->extent.width; x++)
    // {
    //     for(int y = 0; y < image->extent.height; y++)
    //     {
    //         int blockX = x / 8;
    //         int blockY = y / 8;

    //         uint32_t offset = (blockX + blockY * (image->extent.width / 8)) * (128 / 8);

    //         uint8_t dst_colors[100];
    //         basisu::astc::decompress(dst_colors, address + offset, true, 8, 8);
    //         uint8_t* pixel_color = &dst_colors[0] + (x % 8 + (y % 8) * 8) * 4;

    //         uint32_t bit_map_offset = x + y * image->extent.width;

    //         float data[4];
    //         data[0] = pixel_color[0] / 255.0;
    //         data[1] = pixel_color[1] / 255.0;
    //         data[2] = pixel_color[2] / 255.0;
    //         data[3] = pixel_color[3] / 255.0;
    //         imageFile.write((char*) data, 3 * sizeof(float));
    //         imageFile.write((char*) (&bit_map_offset), sizeof(uint32_t));
    //         imageFile.flush();
    //     }
    // }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    // printf("gpgpusim: getTexture not implemented for lavapipe.\n");
    //
    // printf("GIVEN DESC: %p\n", desc);

    if (x < 0 || x > 1)
        x -= std::floor(x);
    if (y < 0 || y > 1)
        y -= std::floor(y);

    // printf("X: %f, Y: %f\n", x, y);

    struct lvp_descriptor d = *(struct lvp_descriptor*) desc;
    const struct lvp_image *img = d.info.sampler_view->image;
    uint32_t width = img->vk.extent.width;
    uint32_t height = img->vk.extent.height;
    void *i = img->pmem;

    uint32_t x_int = std::floor(x * width);
    uint32_t y_int = std::floor(y * height);
    if(x_int >= width)
        x_int -= width;
    if(y_int >= height)
        y_int -= height;

    void *c = i + (y_int * height + x_int) * 4;

    ImageMemoryTransactionRecord transaction;
    transaction.type = ImageTransactionType::TEXTURE_LOAD;
    transaction.address = c;
    transaction.size = 4;
    transactions.push_back(transaction);

    uint8_t *colors = (uint8_t*) c;
    c0 = colors[0] / 255.0;
    c1 = colors[1] / 255.0;
    c2 = colors[2] / 255.0;
    c3 = colors[3] / 255.0;

    // abort();
#endif
}

#if defined(MESA_USE_LVPIPE_DRIVER)
struct ImageOutputState {
    std::string path;
    std::string contents;
    std::vector<uint8_t> written_pixels;
    std::vector<uint8_t> pixel_nonblack;
    std::vector<unsigned> pixel_max_channels;
    size_t header_offset = 0;
    size_t pixel_count = 0;
    size_t nonblack_pixels = 0;
    unsigned max_channel = 0;
    bool flushed = false;
};

static std::map<std::string, ImageOutputState> outputImageStates;

static const int RTCORE_LVP_ACCUMULATION_IMAGE_BINDING = 1;
static const int RTCORE_LVP_OUTPUT_IMAGE_BINDING = 2;

static unsigned rtcore_clamp_ppm_channel(float value)
{
    if (!std::isfinite(value) || value <= 0.0f) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return static_cast<unsigned>(std::lround(value * 255.0f));
}

static int rtcore_lvp_descriptor_binding(const struct DESCRIPTOR_SET_STRUCT *descriptor_set,
                                         const struct DESCRIPTOR_STRUCT *desc)
{
    if (descriptor_set == NULL || descriptor_set->layout == NULL || desc == NULL) {
        return -1;
    }

    for (uint32_t binding = 0; binding < descriptor_set->layout->binding_count; ++binding) {
        const struct DESCRIPTOR_LAYOUT_STRUCT *bind_layout = &descriptor_set->layout->binding[binding];
        if (!bind_layout->valid || bind_layout->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            continue;
        }
        const struct DESCRIPTOR_STRUCT *base =
            &descriptor_set->descriptors[bind_layout->descriptor_index];
        const struct DESCRIPTOR_STRUCT *end = base + bind_layout->array_size;
        if (desc >= base && desc < end) {
            return binding;
        }
    }

    return -1;
}

static bool rtcore_lvp_should_dump_storage_image(int binding)
{
    if (binding == RTCORE_LVP_ACCUMULATION_IMAGE_BINDING) {
        return false;
    }
    if (binding == RTCORE_LVP_OUTPUT_IMAGE_BINDING) {
        return true;
    }
    return true;
}
#endif

static unsigned rtcore_launch_id_x_for_thread(ptx_thread_info *thread)
{
    const dim3 tid = thread->get_tid();
    const dim3 ctaid = thread->get_ctaid();
    return tid.x + ctaid.x * 32;
}

static unsigned rtcore_launch_id_y_for_thread(ptx_thread_info *thread)
{
    return thread->get_ctaid().y;
}

static bool rtcore_pixel_trace_matches_thread(ptx_thread_info *thread)
{
    return thread != NULL &&
           rtcore_pixel_trace_matches(rtcore_launch_id_x_for_thread(thread),
                                      rtcore_launch_id_y_for_thread(thread));
}

void VulkanRayTracing::image_load(struct DESCRIPTOR_STRUCT *desc, uint32_t x, uint32_t y, float &c0, float &c1, float &c2, float &c3)
{
#if defined(MESA_USE_INTEL_DRIVER)
    ImageMemoryTransactionRecord transaction;

    struct anv_image_view *image_view =  desc->image_view;
    struct anv_sampler *sampler = desc->sampler;

    const struct anv_image *image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
    assert(sampler->conversion == NULL);

    Pixel pixel = load_image_pixel(image, x, y, 0, transaction);

    transaction.type = ImageTransactionType::IMAGE_LOAD;
    
    c0 = pixel.c0;
    c1 = pixel.c1;
    c2 = pixel.c2;
    c3 = pixel.c3;

#elif defined(MESA_USE_LVPIPE_DRIVER)
    VSIM_DPRINTF("gpgpusim: image_load not implemented for lavapipe.\n");
    abort();

#endif
}

void VulkanRayTracing::image_store(struct DESCRIPTOR_STRUCT* desc, uint32_t gl_LaunchIDEXT_X, uint32_t gl_LaunchIDEXT_Y, uint32_t gl_LaunchIDEXT_Z, uint32_t gl_LaunchIDEXT_W, 
              float hitValue_X, float hitValue_Y, float hitValue_Z, float hitValue_W, const ptx_instruction *pI, ptx_thread_info *thread)
{
#if defined(MESA_USE_INTEL_DRIVER)
    ImageMemoryTransactionRecord transaction;
    Pixel pixel = Pixel(hitValue_X, hitValue_Y, hitValue_Z, hitValue_W);

    VkFormat vk_format;
    if (use_external_launcher)
    {
        storage_image_metadata *metadata = (storage_image_metadata*) desc;
        vk_format = metadata->format;
        store_image_pixel((anv_image*) desc, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
    }
    else
    {
        assert(desc->sampler == NULL);

        struct anv_image_view *image_view = desc->image_view;
        assert(image_view != NULL);
        struct anv_image * image = image_view->image;

        vk_format = image->vk_format;

        store_image_pixel(image, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
    }

    
    transaction.type = ImageTransactionType::IMAGE_STORE;

    if(writeImageBinary && vk_format != VK_FORMAT_R32G32B32A32_SFLOAT)
    {
        uint32_t image_width = thread->get_kernel().vulkan_metadata.launch_width;
        uint32_t offset = 0;
        offset += gl_LaunchIDEXT_Y * image_width;
        offset += gl_LaunchIDEXT_X;

        float data[4];
        data[0] = hitValue_X;
        data[1] = hitValue_Y;
        data[2] = hitValue_Z;
        data[3] = hitValue_W;
        imageFile.write((char*) data, 3 * sizeof(float));
        imageFile.write((char*) (&offset), sizeof(uint32_t));
        imageFile.flush();

        // imageFile << "(" << gl_LaunchIDEXT_X << ", " << gl_LaunchIDEXT_Y << ") : (";
        // imageFile << hitValue_X << ", " << hitValue_Y << ", " << hitValue_Z << ", " << hitValue_W << ")\n";
    }

    TXL_DPRINTF("Setting transaction for image_store\n");
    thread->set_txl_transactions(transaction);

    // // if(std::abs(hitValue_X - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.x) > 0.0001 || 
    // //     std::abs(hitValue_Y - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.y) > 0.0001 ||
    // //     std::abs(hitValue_Z - rayDebugGPUData[gl_LaunchIDEXT_X][gl_LaunchIDEXT_Y].hitValue.z) > 0.0001)
    // //     {
    // //         printf("wrong value. (%d, %d): (%f, %f, %f)\n"
    // //                 , gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, hitValue_X, hitValue_Y, hitValue_Z);
    // //     }
    
    // // if (gl_LaunchIDEXT_X == 1070 && gl_LaunchIDEXT_Y == 220)
    // //     printf("this one has wrong value\n");

    // // if(hitValue_X > 1 || hitValue_Y > 1 || hitValue_Z > 1)
    // // {
    // //     printf("this one has wrong value.\n");
    // // }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    assert(desc->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    struct lvp_image *image = (struct lvp_image *)desc->info.image_view.image;
    VkFormat vk_format = image->vk.format;
    assert(image != NULL);
    VSIM_DPRINTF("gpgpusim: image_store to %s at %p\n", image->vk.base.object_name, image->pmem_gpgpusim);

    Pixel pixel = Pixel(hitValue_X, hitValue_Y, hitValue_Z, hitValue_W);

    uint32_t width = image->vk.extent.width;
    uint32_t height = image->vk.extent.height;
    const int storage_image_binding =
        rtcore_lvp_descriptor_binding(VulkanRayTracing::descriptorSet, desc);

    if (rtcore_pixel_trace_matches(gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y)) {
        printf("GPGPU-Sim RTCORE_PIXEL_TRACE image-store "
               "launch=(%u,%u), thread_uid=%u, binding=%d, dump_image=%u, "
               "value=(%.9g,%.9g,%.9g,%.9g)\n",
               gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, thread->get_uid(),
               storage_image_binding,
               rtcore_lvp_should_dump_storage_image(storage_image_binding) ? 1
                                                                           : 0,
               hitValue_X, hitValue_Y, hitValue_Z, hitValue_W);
    }

    if (writeImageBinary && rtcore_lvp_should_dump_storage_image(storage_image_binding)) {
        // TODO: fix the bottom, is NULL
        // assert(image->vk.base.object_name);
        // std::string img_name(image->vk.base.object_name);
        std::string img_name("SCENE");

        if (outputImageStates.find(img_name) == outputImageStates.end()) {
            std::time_t raw_time = std::time(0);
            struct tm *time_info;
            char time_buf[30];

            time_info = localtime(&raw_time);

            strftime(time_buf, sizeof(time_buf), "%d-%m-%Y-%H-%M-%S-", time_info);

            std::string time_offset(time_buf);
            std::string new_img_file_name = time_offset + img_name;

            outputImages[img_name] = new_img_file_name + ".ppm";
            printf("gpgpusim: saving image %s to file %s\n", img_name.c_str(), outputImages[img_name].c_str());

            ImageOutputState state;
            state.path = outputImages[img_name];
            state.contents = "P3\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
            state.header_offset = state.contents.size();
            state.contents.resize(state.header_offset + static_cast<size_t>(width) * height * 12, ' ');
            state.written_pixels.assign(static_cast<size_t>(width) * height, 0);
            state.pixel_nonblack.assign(static_cast<size_t>(width) * height, 0);
            state.pixel_max_channels.assign(static_cast<size_t>(width) * height, 0);
            outputImageStates[img_name] = std::move(state);
        }

        ImageOutputState &state = outputImageStates[img_name];
        const size_t pixel_index = gl_LaunchIDEXT_X + static_cast<size_t>(gl_LaunchIDEXT_Y) * width;
        const size_t value_offset = state.header_offset + pixel_index * 12;
        const unsigned ppm_r = rtcore_clamp_ppm_channel(hitValue_X);
        const unsigned ppm_g = rtcore_clamp_ppm_channel(hitValue_Y);
        const unsigned ppm_b = rtcore_clamp_ppm_channel(hitValue_Z);
        const bool pixel_is_nonblack = ppm_r != 0 || ppm_g != 0 || ppm_b != 0;
        unsigned pixel_max_channel = ppm_r;
        if (ppm_g > pixel_max_channel) {
            pixel_max_channel = ppm_g;
        }
        if (ppm_b > pixel_max_channel) {
            pixel_max_channel = ppm_b;
        }
        char pixel_line[13];
        snprintf(pixel_line, sizeof(pixel_line), "%3u %3u %3u\n", ppm_r,
                 ppm_g, ppm_b);
        state.contents.replace(value_offset, 12, pixel_line, 12);

        if (state.written_pixels[pixel_index]) {
            if (state.pixel_nonblack[pixel_index] && !pixel_is_nonblack) {
                state.nonblack_pixels--;
            } else if (!state.pixel_nonblack[pixel_index] && pixel_is_nonblack) {
                state.nonblack_pixels++;
            }
        } else if (pixel_is_nonblack) {
            state.nonblack_pixels++;
        }
        state.pixel_nonblack[pixel_index] = pixel_is_nonblack ? 1 : 0;
        state.pixel_max_channels[pixel_index] = pixel_max_channel;

        if (!state.written_pixels[pixel_index]) {
            state.written_pixels[pixel_index] = 1;
            state.pixel_count++;
            if (rt_progress_logging_enabled() &&
                (state.pixel_count == 1 || state.pixel_count % 4096 == 0 ||
                 state.pixel_count == state.written_pixels.size())) {
                printf("gpgpusim: image %s progress %zu / %zu pixels\n",
                       img_name.c_str(), state.pixel_count, state.written_pixels.size());
            }
        }

        if (!state.flushed && state.pixel_count == state.written_pixels.size()) {
            FILE *img_bin = fopen(state.path.c_str(), "wb");
            if (img_bin == nullptr) {
                perror("gpgpusim: fopen image output");
                abort();
            }

            const size_t written = fwrite(state.contents.data(), 1, state.contents.size(), img_bin);
            if (written != state.contents.size()) {
                perror("gpgpusim: fwrite image output");
                fclose(img_bin);
                abort();
            }

            fflush(img_bin);
            fclose(img_bin);
            state.max_channel = 0;
            for (size_t i = 0; i < state.pixel_max_channels.size(); ++i) {
                if (state.pixel_max_channels[i] > state.max_channel) {
                    state.max_channel = state.pixel_max_channels[i];
                }
            }
            printf("gpgpusim: image %s stats nonblack_pixels=%zu / %zu, "
                   "max_channel=%u, all_black=%u\n",
                   img_name.c_str(), state.nonblack_pixels,
                   state.written_pixels.size(), state.max_channel,
                   state.nonblack_pixels == 0 ? 1 : 0);
            state.flushed = true;
            printf("gpgpusim: finished image %s (%zu pixels)\n", img_name.c_str(), state.pixel_count);
        }
    }

    // Setup transaction record for timing model
    ImageMemoryTransactionRecord transaction;
    transaction.type = ImageTransactionType::IMAGE_STORE;

    VkImageTiling tiling = image->vk.tiling;
    uint32_t pixelX = gl_LaunchIDEXT_X;
    uint32_t pixelY = gl_LaunchIDEXT_Y;

    // Size of image_store content depends on data type
    switch (vk_format) {
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            transaction.size = 16;
            break; 

        case VK_FORMAT_B8G8R8A8_UNORM:
            transaction.size = 4;
            break;

        default:
            printf("gpgpusim: unsupported image format option %d\n", vk_format);
            abort();
    }

    switch (tiling) {
        // Just an arbitrary tiling (TODO: Find a better tiling option)
        case VK_IMAGE_TILING_OPTIMAL:
        {
            uint32_t tileWidth = 16;
            uint32_t tileHeight = 16;

            uint32_t nTileX = (width + tileWidth - 1) / tileWidth;
            uint32_t tileX = floor(pixelX / tileWidth);
            uint32_t tileY = floor(pixelY / tileHeight);

            uint32_t tileOffset = tileWidth * tileHeight * (tileY * nTileX + tileX);
            uint32_t pixelOffset = (pixelY % tileHeight) * tileWidth + (pixelX % tileWidth);

            transaction.address = image->pmem_gpgpusim + ((tileOffset + pixelOffset) * transaction.size);
            break;
        }
        // Linear
        case VK_IMAGE_TILING_LINEAR:
        {
            uint32_t offset = pixelY * width + pixelX;
            transaction.address = image->pmem_gpgpusim + offset * transaction.size;
            break;
        }
        default:
        {
            printf("gpgpusim: unsupported image tiling option %d\n", tiling);
            abort();
        }
    }

    TXL_DPRINTF("Setting transaction for image_store\n");
    thread->set_txl_transactions(transaction);

    // store_image_pixel(image, gl_LaunchIDEXT_X, gl_LaunchIDEXT_Y, 0, pixel, transaction);
#endif
}

// variable_decleration_entry* VulkanRayTracing::get_variable_decleration_entry(std::string name, ptx_thread_info *thread)
// {
//     std::vector<variable_decleration_entry>& table = thread->RT_thread_data->variable_decleration_table;
//     for (int i = 0; i < table.size(); i++) {
//         if (table[i].name == name) {
//             assert (table[i].address != NULL);
//             return &(table[i]);
//         }
//     }
//     return NULL;
// }

// void VulkanRayTracing::add_variable_decleration_entry(uint64_t type, std::string name, uint64_t address, uint32_t size, ptx_thread_info *thread)
// {
//     variable_decleration_entry entry;

//     entry.type = type;
//     entry.name = name;
//     entry.address = address;
//     entry.size = size;
//     thread->RT_thread_data->variable_decleration_table.push_back(entry);
// }


void VulkanRayTracing::dumpTextures(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type)
{
#if defined(MESA_USE_INTEL_DRIVER)
    DESCRIPTOR_STRUCT *desc_offset = ((DESCRIPTOR_STRUCT*)((void*)desc)); // offset for raytracing_extended
    struct anv_image_view *image_view =  desc_offset->image_view;
    struct anv_sampler *sampler = desc_offset->sampler;

    const struct anv_image *image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);
    assert(image->tiling == VK_IMAGE_TILING_OPTIMAL);
    assert(image->planes[0].surface.isl.tiling == ISL_TILING_Y0);
    assert(sampler->conversion == NULL);

    uint8_t* address = anv_address_map(image->planes[0].address);
    uint32_t image_extent_width = image->extent.width;
    uint32_t image_extent_height = image->extent.height;
    VkFormat format = image->vk_format;
    uint64_t size = image->size;

    VkFilter filter;
    if(sampler->conversion == NULL)
        filter = VK_FILTER_NEAREST;

    // Data to dump
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsettexturedata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            VkDescriptorTypeNum = 0;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VkDescriptorTypeNum = 1;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            VkDescriptorTypeNum = 2;
            break;
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            VkDescriptorTypeNum = 10;
            break;
        default:
            abort(); // should not be here!
    }

    // Texture data
    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vktexturedata", mesa_root, filePath, setID, binding);
    // File name format: setID_descID.vktexturedata

    fp = fopen(fullPath, "wb+");
    fwrite(address, 1, size, fp);
    fclose(fp);

    // Texture metadata
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vktexturemetadata", mesa_root, filePath, setID, binding);
    fp = fopen(fullPath, "w+");
    // File name format: setID_descID.vktexturemetadata

    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", size, 
                                                 image_extent_width, 
                                                 image_extent_height, 
                                                 format, 
                                                 VkDescriptorTypeNum, 
                                                 image->n_planes, 
                                                 image->samples, 
                                                 image->tiling, 
                                                 image->planes[0].surface.isl.tiling,
                                                 image->planes[0].surface.isl.row_pitch_B,
                                                 filter);
    fclose(fp);
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dumpTextures not implemented for lavapipe.\n");
    abort();

#endif

}


void VulkanRayTracing::dumpStorageImage(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type)
{
#if defined(MESA_USE_INTEL_DRIVER)
    assert(type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

    assert(desc->sampler == NULL);

    struct anv_image_view *image_view = desc->image_view;
    assert(image_view != NULL);
    struct anv_image * image = image_view->image;
    assert(image->n_planes == 1);
    assert(image->samples == 1);

    void* mem_address = anv_address_map(image->planes[0].address);

    VkFormat format = image->vk_format;
    VkImageTiling tiling = image->tiling;
    isl_tiling isl_tiling_mode = image->planes[0].surface.isl.tiling;
    uint32_t row_pitch_B  = image->planes[0].surface.isl.row_pitch_B;

    uint32_t width = image->extent.width;
    uint32_t height = image->extent.height;

    // Dump storage image metadata
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum = 3;

    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.vkstorageimagemetadata", mesa_root, filePath, setID, binding);
    fp = fopen(fullPath, "w+");
    // File name format: setID_descID.vktexturemetadata

    fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d",   width, 
                                                height, 
                                                format, 
                                                VkDescriptorTypeNum, 
                                                image->n_planes, 
                                                image->samples, 
                                                tiling, 
                                                isl_tiling_mode,
                                                row_pitch_B);
    fclose(fp);
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dumpStorageImage not implemented for lavapipe.\n");
    abort();

#endif
}


void VulkanRayTracing::dump_descriptor_set_for_AS(uint32_t setID, uint32_t descID, void *address, uint32_t desc_size, VkDescriptorType type, uint32_t backwards_range, uint32_t forward_range, bool split_files, VkAccelerationStructureKHR _topLevelAS)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VkDescriptorTypeNum = 1000150000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            VkDescriptorTypeNum = 1000165000;
            break;
        default:
            abort(); // should not be here!
    }

    char fullPath[200];
    int result;

    int64_t max_backwards; // negative number
    int64_t min_backwards; // negative number
    int64_t min_forwards;
    int64_t max_forwards;
    int64_t back_buffer_amount = 0; //20kB buffer just in case
    int64_t front_buffer_amount = 1024*20; //20kB buffer just in case
    findOffsetBounds(max_backwards, min_backwards, min_forwards, max_forwards, _topLevelAS);

    bool haveBackwards = (max_backwards != 0) && (min_backwards != 0);
    bool haveForwards = (min_forwards != 0) && (max_forwards != 0);
    
    if (split_files) // Used when the AS is too far apart between top tree and BVHAddress and cant just dump the whole thing
    {
        // Main Top Level
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asmain", mesa_root, filePath, setID, descID);
        fp = fopen(fullPath, "wb+");
        result = fwrite(address, 1, desc_size, fp);
        assert(result == desc_size);
        fclose(fp);

        // Bot level whose address is smaller than top level
        if (haveBackwards)
        {
            snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asback", mesa_root, filePath, setID, descID);
            fp = fopen(fullPath, "wb+");
            result = fwrite(address + max_backwards, 1, min_backwards - max_backwards + back_buffer_amount, fp);
            assert(result == min_backwards - max_backwards + back_buffer_amount);
            fclose(fp);
        }

        // Bot level whose address is larger than top level
        if (haveForwards)
        {
            snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asfront", mesa_root, filePath, setID, descID);
            fp = fopen(fullPath, "wb+");
            result = fwrite(address + min_forwards, 1, max_forwards - min_forwards + front_buffer_amount, fp);
            assert(result == max_forwards - min_forwards + front_buffer_amount);
            fclose(fp);
        }

        // AS metadata
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d.asmetadata", mesa_root, filePath, setID, descID);
        fp = fopen(fullPath, "w+");
        fprintf(fp, "%d,%d,%ld,%ld,%ld,%ld,%ld,%ld,%d,%d", desc_size,
                                                            VkDescriptorTypeNum,
                                                            max_backwards,
                                                            min_backwards,
                                                            min_forwards,
                                                            max_forwards,
                                                            back_buffer_amount,
                                                            front_buffer_amount,
                                                            haveBackwards,
                                                            haveForwards);
        fclose(fp);

        
        // uint64_t total_size = (desc_size + backwards_range + forward_range);
        // uint64_t chunk_size = 1024*1024*20; // 20MB chunks
        // int totalFiles =  (total_size + chunk_size) / chunk_size; // rounds up

        // for (int i = 0; i < totalFiles; i++)
        // {
        //     // if split_files is 1, then look at the next number to see what the file part number is
        //     snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d_%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, desc_size, VkDescriptorTypeNum, backwards_range, forward_range, split_files, i, extension);
        //     fp = fopen(fullPath, "wb+");
        //     int result = fwrite(address-(uint64_t)backwards_range + chunk_size * i, 1, chunk_size, fp);
        //     printf("File part %d, %d bytes written, starting address 0x%.12" PRIXPTR "\n", i, result, (uintptr_t)(address-(uint64_t)backwards_range + chunk_size * i));
        //     fclose(fp);
        // }
    }
    else 
    {
        snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, desc_size, VkDescriptorTypeNum, backwards_range, forward_range, extension);
        // File name format: setID_descID_SizeInBytes_VkDescriptorType_desired_range.vkdescrptorsetdata

        fp = fopen(fullPath, "wb+");
        int result = fwrite(address-(uint64_t)backwards_range, 1, desc_size + backwards_range + forward_range, fp);
        fclose(fp);
    }
}


void VulkanRayTracing::dump_descriptor_set(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";
    char *extension = ".vkdescrptorsetdata";

    int VkDescriptorTypeNum;

    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            VkDescriptorTypeNum = 0;
            break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            VkDescriptorTypeNum = 1;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            VkDescriptorTypeNum = 2;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            VkDescriptorTypeNum = 3;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            VkDescriptorTypeNum = 4;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            VkDescriptorTypeNum = 5;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            VkDescriptorTypeNum = 6;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            VkDescriptorTypeNum = 7;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            VkDescriptorTypeNum = 8;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            VkDescriptorTypeNum = 9;
            break;
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            VkDescriptorTypeNum = 10;
            break;
        case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
            VkDescriptorTypeNum = 1000138000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            VkDescriptorTypeNum = 1000150000;
            break;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            VkDescriptorTypeNum = 1000165000;
            break;
        case VK_DESCRIPTOR_TYPE_MUTABLE_VALVE:
            VkDescriptorTypeNum = 1000351000;
            break;
        case VK_DESCRIPTOR_TYPE_MAX_ENUM:
            VkDescriptorTypeNum = 0x7FFFFFF;
            break;
        default:
            abort(); // should not be here!
    }

    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s%d_%d_%d_%d%s", mesa_root, filePath, setID, descID, size, VkDescriptorTypeNum, extension);
    // File name format: setID_descID_SizeInBytes_VkDescriptorType.vkdescrptorsetdata

    fp = fopen(fullPath, "wb+");
    fwrite(address, 1, size, fp);
    fclose(fp);
}


void VulkanRayTracing::dump_descriptor_sets(struct DESCRIPTOR_SET_STRUCT *set)
{
#if defined(MESA_USE_INTEL_DRIVER)
   for(int i = 0; i < set->descriptor_count; i++)
   {
       if(i == 3 || i > 9)
       {
            // for some reason raytracing_extended skipped binding = 3
            // and somehow they have 34 descriptor sets but only 10 are used
            // so we just skip those
            continue;
       }

        struct DESCRIPTOR_SET_STRUCT* set = VulkanRayTracing::descriptorSet;

        const struct DESCRIPTOR_LAYOUT_STRUCT *bind_layout = &set->layout->binding[i];
        struct DESCRIPTOR_STRUCT *desc = &set->descriptors[bind_layout->descriptor_index];
        void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

        assert(desc->type == bind_layout->type);

        switch (desc->type)
        {
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            {
                //return (void *)(desc);
                dumpStorageImage(desc, 0, i, desc->type);
                break;
            }
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            {
                //return desc;
                dumpTextures(desc, 0, i, desc->type);
                break;
            }

            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            {
                if (desc->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                    desc->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                {
                    // MRS_TODO: account for desc->offset?
                    //return anv_address_map(desc->buffer->address);
                    dump_descriptor_set(0, i, anv_address_map(desc->buffer->address), set->descriptors[i].buffer->size, set->descriptors[i].type);
                    break;
                }
                else
                {
                    struct anv_buffer_view *bview = &set->buffer_views[bind_layout->buffer_view_index];
                    //return anv_address_map(bview->address);
                    dump_descriptor_set(0, i, anv_address_map(bview->address), bview->range, set->descriptors[i].type);
                    break;
                }
            }

            case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
                assert(0);
                break;

            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                //return (void *)(desc_data->address);
                //dump_descriptor_set_for_AS(0, i, (void *)(desc_data->address), desc_data->range, set->descriptors[i].type, 1024*1024*10, 1024*1024*10, true);
                break;
            }

            default:
                assert(0);
                break;
        }
   }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dump_descriptor_sets not implemented for lavapipe.\n");
    abort();

#endif
}

void VulkanRayTracing::dump_AS(struct DESCRIPTOR_SET_STRUCT *set, VkAccelerationStructureKHR _topLevelAS)
{
#if defined(MESA_USE_INTEL_DRIVER)
   for(int i = 0; i < set->descriptor_count; i++)
   {
       if(i == 3 || i > 9)
       {
            // for some reason raytracing_extended skipped binding = 3
            // and somehow they have 34 descriptor sets but only 10 are used
            // so we just skip those
            continue;
       }

        struct DESCRIPTOR_SET_STRUCT* set = VulkanRayTracing::descriptorSet;

        const struct DESCRIPTOR_LAYOUT_STRUCT *bind_layout = &set->layout->binding[i];
        struct DESCRIPTOR_STRUCT *desc = &set->descriptors[bind_layout->descriptor_index];
        void *desc_map = set->desc_mem.map + bind_layout->descriptor_offset;

        assert(desc->type == bind_layout->type);

        switch (desc->type)
        {
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV:
            {
                struct anv_address_range_descriptor *desc_data = desc_map;
                //return (void *)(desc_data->address);
                dump_descriptor_set_for_AS(0, i, (void *)(desc_data->address), desc_data->range, set->descriptors[i].type, 1024*1024*10, 1024*1024*10, true, _topLevelAS);
                break;
            }

            default:
                break;
        }
    }
#elif defined(MESA_USE_LVPIPE_DRIVER)
    printf("gpgpusim: dump_AS not implemented for lavapipe.\n");
    abort();

#endif
}

void VulkanRayTracing::dump_callparams_and_sbt(void *raygen_sbt, void *miss_sbt, void *hit_sbt, void *callable_sbt, bool is_indirect, uint32_t launch_width, uint32_t launch_height, uint32_t launch_depth, uint32_t launch_size_addr)
{
    FILE *fp;
    char *mesa_root = getenv("MESA_ROOT");
    char *filePath = "gpgpusimShaders/";

    char call_params_filename [200];
    int trace_rays_call_count = 0; // just a placeholder for now
    snprintf(call_params_filename, sizeof(call_params_filename), "%s%s%d.callparams", mesa_root, filePath, trace_rays_call_count);
    fp = fopen(call_params_filename, "w+");
    fprintf(fp, "%d,%d,%d,%d,%lu", is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
    fclose(fp);

    // TODO: Is the size always 32?
    int sbt_size = 64 *sizeof(uint64_t);
    if (raygen_sbt) {
        char raygen_sbt_filename [200];
        snprintf(raygen_sbt_filename, sizeof(raygen_sbt_filename), "%s%s%d.raygensbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(raygen_sbt_filename, "wb+");
        fwrite(raygen_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (miss_sbt) {
        char miss_sbt_filename [200];
        snprintf(miss_sbt_filename, sizeof(miss_sbt_filename), "%s%s%d.misssbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(miss_sbt_filename, "wb+");
        fwrite(miss_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (hit_sbt) {
        char hit_sbt_filename [200];
        snprintf(hit_sbt_filename, sizeof(hit_sbt_filename), "%s%s%d.hitsbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(hit_sbt_filename, "wb+");
        fwrite(hit_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }

    if (callable_sbt) {
        char callable_sbt_filename [200];
        snprintf(callable_sbt_filename, sizeof(callable_sbt_filename), "%s%s%d.callablesbt", mesa_root, filePath, trace_rays_call_count);
        fp = fopen(callable_sbt_filename, "wb+");
        fwrite(callable_sbt, 1, sbt_size, fp); // max is 32 bytes according to struct anv_rt_shader_group.handle
        fclose(fp);
    }
}

void VulkanRayTracing::setStorageImageFromLauncher(void *address, 
                                                void *deviceAddress, 
                                                uint32_t setID, 
                                                uint32_t descID, 
                                                uint32_t width,
                                                uint32_t height,
                                                VkFormat format,
                                                uint32_t VkDescriptorTypeNum,
                                                uint32_t n_planes,
                                                uint32_t n_samples,
                                                VkImageTiling tiling,
                                                uint32_t isl_tiling_mode, 
                                                uint32_t row_pitch_B)
{
    storage_image_metadata *storage_image = new storage_image_metadata;
    storage_image->address = address;
    storage_image->setID = setID;
    storage_image->descID = descID;
    storage_image->width = width;
    storage_image->height = height;
    storage_image->format = format;
    storage_image->VkDescriptorTypeNum = VkDescriptorTypeNum;
    storage_image->n_planes = n_planes;
    storage_image->n_samples = n_samples;
    storage_image->tiling = tiling;
    storage_image->isl_tiling_mode = isl_tiling_mode; 
    storage_image->row_pitch_B = row_pitch_B;
    storage_image->deviceAddress = deviceAddress;

    launcher_descriptorSets[setID][descID] = (void*) storage_image;
    launcher_deviceDescriptorSets[setID][descID] = (void*) storage_image;
}

void VulkanRayTracing::setTextureFromLauncher(void *address, 
                                            void *deviceAddress, 
                                            uint32_t setID, 
                                            uint32_t descID, 
                                            uint64_t size,
                                            uint32_t width,
                                            uint32_t height,
                                            VkFormat format,
                                            uint32_t VkDescriptorTypeNum,
                                            uint32_t n_planes,
                                            uint32_t n_samples,
                                            VkImageTiling tiling,
                                            uint32_t isl_tiling_mode,
                                            uint32_t row_pitch_B,
                                            uint32_t filter)
{
    texture_metadata *texture = new texture_metadata;
    texture->address = address;
    texture->setID = setID;
    texture->descID = descID;
    texture->size = size;
    texture->width = width;
    texture->height = height;
    texture->format = format;
    texture->VkDescriptorTypeNum = VkDescriptorTypeNum;
    texture->n_planes = n_planes;
    texture->n_samples = n_samples;
    texture->tiling = tiling;
    texture->isl_tiling_mode = isl_tiling_mode;
    texture->row_pitch_B = row_pitch_B;
    texture->filter = filter;
    texture->deviceAddress = deviceAddress;

    launcher_descriptorSets[setID][descID] = (void*) texture;
    launcher_deviceDescriptorSets[setID][descID] = (void*) texture;
}

void VulkanRayTracing::pass_child_addr(void *address)
{
    child_addrs_from_driver.push_back(address);
}

void VulkanRayTracing::allocBLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr) {
    printf("gpgpusim: set BLAS address for 0x%lx at %p to %p\n", bufferSize, rootAddr, gpgpusimAddr);
    blas_addr_map[rootAddr] = gpgpusimAddr;
}

void VulkanRayTracing::allocTLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr) {
    printf("gpgpusim: set TLAS address %p to %p\n", rootAddr, gpgpusimAddr);
    tlas_addr = gpgpusimAddr;
}

void VulkanRayTracing::findOffsetBounds(int64_t &max_backwards, int64_t &min_backwards, int64_t &min_forwards, int64_t &max_forwards, VkAccelerationStructureKHR _topLevelAS)
{
    // uint64_t current_min_backwards = 0;
    // uint64_t current_max_backwards = 0;
    // uint64_t current_min_forwards = 0;
    // uint64_t current_max_forwards = 0;
    int64_t offset;

    std::vector<int64_t> positive_offsets;
    std::vector<int64_t> negative_offsets;

    for (auto addr : child_addrs_from_driver)
    {
        offset = (uint64_t)addr - (uint64_t)_topLevelAS;
        if (offset >= 0)
            positive_offsets.push_back(offset);
        else
            negative_offsets.push_back(offset);
    }

    sort(positive_offsets.begin(), positive_offsets.end());
    sort(negative_offsets.begin(), negative_offsets.end());

    if (negative_offsets.size() > 0)
    {
        max_backwards = negative_offsets.front();
        min_backwards = negative_offsets.back();
    }
    else
    {
        max_backwards = 0;
        min_backwards = 0;
    }

    if (positive_offsets.size() > 0)
    {
        min_forwards = positive_offsets.front();
        max_forwards = positive_offsets.back();
    }
    else
    {
        min_forwards = 0;
        max_forwards = 0;
    }
}


void* VulkanRayTracing::gpgpusim_alloc(uint32_t size)
{
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    void* devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
    if (g_debug_execution >= 3) {
        printf("GPGPU-Sim PTX: gpgpusim_allocing %zu bytes starting at 0x%llx..\n",
            size, (unsigned long long)devPtr);
        ctx->api->g_mallocPtr_Size[(unsigned long long)devPtr] = size;
    }
    assert(devPtr);

    if(!use_external_launcher) {
        void* bufferAddr = malloc(size);
        memory_space *mem = context->get_device()->get_gpgpu()->get_global_memory();
        mem->bind_vulkan_buffer(bufferAddr, size, devPtr);
    }

    return devPtr;
}

void* VulkanRayTracing::allocBuffer(void* bufferAddr, uint64_t bufferSize)
{
    gpgpu_context *ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);
    void* devPtr = context->get_device()->get_gpgpu()->gpu_malloc(bufferSize);
    assert(devPtr);

    memory_space *mem = context->get_device()->get_gpgpu()->get_global_memory();
    
    printf("gpgpusim: binding gpgpusim buffer %p (size %d) to vulkan buffer %p\n", devPtr, bufferSize, bufferAddr);
    mem->bind_vulkan_buffer(bufferAddr, bufferSize, devPtr);
    return devPtr;
}
