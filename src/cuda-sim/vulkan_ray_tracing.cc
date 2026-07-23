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
#include "rtcore_replay_interface.h"
#include "rtcore_procedural_hit_ordering.h"
#include "rtcore_tlas_binding_registry.h"
#include "rtcore_v04_instance_blas_reference_registry.h"
#include "rtcore_v04_private_frontier_layout.h"
#include "rtcore_v04_private_shared_backing.h"
#include "rtcore_v04_request_owner_binding.h"
#include "rtcore_v04_shadow_shader_return.h"
#include "rtcore_v04_typed_blas_decode_context.h"
#include "rtcore_v04_typed_instance_kernel.h"
#include "rtcore_v04_typed_node_kernel.h"
#include "rtcore_v04_typed_primitive_kernel.h"
#include "rtcore_v04_typed_stack_kernel.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <deque>
#include <set>
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

extern "C" bool rtcore_custom_path_mode_enabled();

static bool rt_progress_logging_enabled() {
    static int enabled = []() {
        const char *value = getenv("VULKAN_SIM_PROGRESS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool rtcore_standalone_cuda_memory_enabled() {
    static int enabled = []() {
        const char *value = getenv("VULKAN_SIM_STANDALONE_CUDA_MEMORY");
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

namespace {

static rtcore_tlas_binding_registry<rtcore_tlas_binding_snapshot>
    g_rtcore_tlas_binding_registry;
static rtcore_tlas_binding_registry<rtcore_blas_binding_snapshot>
    g_rtcore_blas_binding_registry;
static rtcore_v04_instance_blas_reference_registry
    g_rtcore_instance_blas_reference_registry;

static void rtcore_fail_tlas_binding(const char *reason,
                                     uint64_t host_root_address,
                                     uint64_t device_base_address,
                                     uint64_t size_bytes,
                                     uint64_t driver_object_key = 0)
{
    fprintf(stderr,
            "GPGPU-Sim RTCORE_TLAS_BINDING_FAULT reason=%s "
            "driver_object_key=0x%llx host_root=0x%llx "
            "device_base=0x%llx size=%llu\n",
            reason != NULL ? reason : "unknown",
            (unsigned long long)driver_object_key,
            (unsigned long long)host_root_address,
            (unsigned long long)device_base_address,
            (unsigned long long)size_bytes);
    fflush(stderr);
    abort();
}

static void rtcore_fail_blas_binding(const char *reason,
                                     uint64_t host_root_address,
                                     uint64_t device_base_address,
                                     uint64_t size_bytes,
                                     uint64_t driver_object_key = 0)
{
    fprintf(stderr,
            "GPGPU-Sim RTCORE_BLAS_BINDING_FAULT reason=%s "
            "driver_object_key=0x%llx host_root=0x%llx "
            "device_base=0x%llx size=%llu\n",
            reason != NULL ? reason : "unknown",
            (unsigned long long)driver_object_key,
            (unsigned long long)host_root_address,
            (unsigned long long)device_base_address,
            (unsigned long long)size_bytes);
    fflush(stderr);
    abort();
}

static void rtcore_fail_instance_blas_reference(
    const char *reason, uint64_t tlas_object_id,
    uint32_t tlas_generation, uint64_t instance_metadata_reference,
    uint64_t blas_object_id = 0, uint32_t blas_generation = 0)
{
    fprintf(stderr,
            "GPGPU-Sim RTCORE_INSTANCE_BLAS_REFERENCE_FAULT reason=%s "
            "tlas_object_id=%llu tlas_generation=%u "
            "instance_metadata_ref=0x%llx blas_object_id=%llu "
            "blas_generation=%u\n",
            reason != NULL ? reason : "unknown",
            (unsigned long long)tlas_object_id, tlas_generation,
            (unsigned long long)instance_metadata_reference,
            (unsigned long long)blas_object_id, blas_generation);
    fflush(stderr);
    abort();
}

}  // namespace

bool VulkanRayTracing::dumped = false;

bool use_external_launcher = rtcore_standalone_cuda_memory_enabled();
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
static const unsigned RTCORE_COMPACT_TRACE_EVENT_SEQ_CAPACITY = 65536;
static const unsigned RTCORE_REPLAY_MEMORY_DEMAND_CACHE_LINE_BYTES = 64;
static const unsigned RTCORE_REPLAY_MEMORY_REQUEST_GRANULE_BYTES = 32;
static const unsigned RTCORE_REPLAY_STACK_LATENCY_CYCLES = 2;
static const unsigned RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES = 32;
static const unsigned RTCORE_V02_LSU_TRANSACTION_IDENTITY_FIELD_COUNT = 10;
static const unsigned long long RTCORE_V02_LSU_SYNTHETIC_STACK_BASE =
    0x50000000ull;

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

enum rtcore_replay_resource_route {
    RTCORE_REPLAY_ROUTE_MEMORY = 0,
    RTCORE_REPLAY_ROUTE_NODE,
    RTCORE_REPLAY_ROUTE_PRIMITIVE,
    RTCORE_REPLAY_ROUTE_STACK,
    RTCORE_REPLAY_ROUTE_COMPLETION,
    RTCORE_REPLAY_ROUTE_INVALID = 255,
};

enum rtcore_v02_lsu_access_kind {
    RTCORE_V02_LSU_ACCESS_HANDOFF_ACQUIRE = 0,
    RTCORE_V02_LSU_ACCESS_NODE_FETCH,
    RTCORE_V02_LSU_ACCESS_PRIMITIVE_FETCH,
    RTCORE_V02_LSU_ACCESS_STACK_LOAD,
    RTCORE_V02_LSU_ACCESS_STACK_STORE,
    RTCORE_V02_LSU_ACCESS_RESULT_STORE,
    RTCORE_V02_LSU_ACCESS_HANDOFF_PUBLICATION_STORE,
    RTCORE_V02_LSU_ACCESS_KIND_COUNT,
};

enum rtcore_v02_lsu_response_target {
    RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE = 1,
};

struct rtcore_v02_lsu_memory_transaction_identity {
    unsigned response_target;
    unsigned owner_hw_sid;
    unsigned rt_request_id;
    unsigned lane_id;
    unsigned memory_op_seq;
    unsigned chunk_id;
    unsigned chunk_count;
    unsigned access_kind;
    unsigned long long aligned_32b_addr;
    bool is_write;
    unsigned long long issue_cycle;
};

struct rtcore_v02_lsu_merge_key {
    unsigned owner_hw_sid;
    unsigned long long issue_cycle;
    unsigned long long aligned_32b_addr;
    bool is_write;

    bool operator<(const rtcore_v02_lsu_merge_key &other) const
    {
        if (owner_hw_sid != other.owner_hw_sid) {
            return owner_hw_sid < other.owner_hw_sid;
        }
        if (issue_cycle != other.issue_cycle) {
            return issue_cycle < other.issue_cycle;
        }
        if (aligned_32b_addr != other.aligned_32b_addr) {
            return aligned_32b_addr < other.aligned_32b_addr;
        }
        return is_write < other.is_write;
    }
};

struct rtcore_replay_memory_unit_request_descriptor_stats {
    unsigned actual_fetch_event_count;
    unsigned node_fetch_event_count;
    unsigned primitive_fetch_event_count;
    unsigned descriptor_count;
    unsigned node_descriptor_count;
    unsigned primitive_descriptor_count;
    unsigned total_chunk_count;
    unsigned max_chunk_count;
    unsigned response_target_rtcore_count;
    unsigned unique_transaction_count;
    unsigned same_cycle_merge_candidate_count;
};

struct rtcore_v02_lsu_response_wait_stats {
    unsigned gate_evaluations;
    unsigned gate_armed_count;
    unsigned gate_blocked_count;
    unsigned gate_woken_count;
    unsigned stack_load_armed_count;
    unsigned stack_store_armed_count;
    unsigned stack_load_woken_count;
    unsigned stack_store_woken_count;
    unsigned completed_event_count;
    unsigned response_chunk_count;
    unsigned duplicate_response_count;
    unsigned stale_response_count;
    unsigned pending_request_count;
    unsigned max_pending_request_count;
    unsigned max_chunk_count;
    unsigned response_target_rtcore_count;
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
    RTCORE_REPLAY_INVALID = 0,
    RTCORE_REPLAY_ADMITTED,
    RTCORE_REPLAY_READY,
    RTCORE_REPLAY_ISSUED_NODE,
    RTCORE_REPLAY_ISSUED_PRIMITIVE,
    RTCORE_REPLAY_ISSUED_STACK,
    RTCORE_REPLAY_ISSUED_MEMORY,
    RTCORE_REPLAY_COMPLETION_PENDING,
    RTCORE_REPLAY_WAITING_SHADER,
    RTCORE_REPLAY_FINAL_WAIT_RETIRE,
    RTCORE_REPLAY_COMPLETED,
};

static const unsigned RTCORE_REPLAY_CONTINUATION_PACKET_SCHEMA_VERSION = 3;

enum rtcore_replay_continuation_packet_reason {
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_NONE = 0,
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS = 1,
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY = 2,
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED = 3,
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED = 4,
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_TRACE_DONE_NO_SHADER = 5,
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_FAULT = 6,
    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_UNSUPPORTED = 7,
};

enum rtcore_replay_unit_latency_gate_unit {
    RTCORE_REPLAY_UNIT_LATENCY_NONE = 0,
    RTCORE_REPLAY_UNIT_LATENCY_NODE = 1,
    RTCORE_REPLAY_UNIT_LATENCY_PRIMITIVE = 2,
    RTCORE_REPLAY_UNIT_LATENCY_STACK = 3,
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

static uint32_t rtcore_v04_fp32_bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static rtcore::abi_v04::shadow::boundary_values
rtcore_make_v04_boundary_values_from_hit(
    const Hit_data &hit, uint64_t instance_metadata_reference,
    uint32_t instance_sbt_contribution, uint32_t instance_index,
    uint32_t instance_custom_index, uint32_t boundary_ray_tmax_fp32,
    bool publish_triangle_attributes)
{
    rtcore::abi_v04::shadow::boundary_values values;
    values.candidate_valid = true;
    values.instance_metadata_reference = instance_metadata_reference;
    values.instance_sbt_contribution = instance_sbt_contribution;
    values.geometry_index = hit.geometry_index;
    values.boundary_ray_tmax_fp32 = boundary_ray_tmax_fp32;
    values.primitive_index = hit.primitive_index;
    values.instance_index = instance_index;
    values.instance_custom_index = instance_custom_index;
    values.geometry_type =
        hit.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR
            ? rtcore::abi_v04::shadow::kBoundaryGeometryTriangle
            : rtcore::abi_v04::shadow::kBoundaryGeometryProcedural;
    values.hit_kind = hit.hit_kind;
    // The current Simulator subset does not yet expose procedural any-hit.
    values.procedural_any_hit_eligible = false;
    if (publish_triangle_attributes &&
        values.geometry_type ==
            rtcore::abi_v04::shadow::kBoundaryGeometryTriangle) {
        values.input_attribute_word_count = 2;
        values.input_attribute_location = 0x01;
        values.input_attribute_format = 0x01;
        values.inline_attribute_words[0] =
            rtcore_v04_fp32_bits(hit.barycentric_coordinates.x);
        values.inline_attribute_words[1] =
            rtcore_v04_fp32_bits(hit.barycentric_coordinates.y);
    }
    return values;
}

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
    bool context_profile_valid;
    unsigned context_layout_version;
    unsigned context_valid_flags;
    unsigned pipeline_profile_id;
    unsigned bvh_format_profile_id;
    unsigned event_count;
    unsigned max_trace_events_per_lane;
    bool timing_trace_overflowed;
    rtcore_trace_timing_precision_class timing_precision_class;
    unsigned overflow_summary_events;
    rtcore_compact_trace_overflow_summary overflow_summary;
    unsigned oracle_anyhit_candidate_count;
    bool oracle_requires_intersection_shader;
    bool ray_sbt_inputs_valid;
    unsigned sbt_record_offset;
    unsigned sbt_record_stride;
    unsigned miss_index;
    unsigned ray_flags;
    unsigned cull_mask;
    bool hit_geometry_summary_valid;
    unsigned closest_hit_kind;
    unsigned closest_hit_geometry_type;
    unsigned closest_hit_geometry_index;
    unsigned closest_hit_primitive_index;
    unsigned closest_hit_instance_index;
    bool instance_sbt_contribution_valid;
    unsigned instance_sbt_contribution;
    bool v04_shadow_boundary_enabled;
    bool v04_tlas_binding_enforcement_enabled;
    rtcore_tlas_binding_snapshot v04_tlas_binding;
    bool v04_shadow_trace_input_valid;
    std::array<uint32_t, rtcore::abi_v04::kWordCount>
        v04_shadow_trace_input_words;
    unsigned long long handoff_window_base;
    memory_space *v04_live_handoff_memory;
    std::vector<rtcore_compact_trace_event> events;
    std::vector<rtcore_boundary_candidate_snapshot> boundary_candidates;
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
    bool context_profile_valid;
    unsigned context_layout_version;
    unsigned context_valid_flags;
    unsigned pipeline_profile_id;
    unsigned bvh_format_profile_id;
    unsigned next_event_index;
    unsigned event_count;
    unsigned node_event_count;
    unsigned primitive_event_count;
    unsigned stack_event_count;
    unsigned memory_event_count;
    unsigned completion_event_count;
    unsigned oracle_anyhit_candidate_count;
    bool oracle_requires_intersection_shader;
    bool ray_sbt_inputs_valid;
    unsigned sbt_record_offset;
    unsigned sbt_record_stride;
    unsigned miss_index;
    unsigned ray_flags;
    unsigned cull_mask;
    bool hit_geometry_summary_valid;
    unsigned closest_hit_kind;
    unsigned closest_hit_geometry_type;
    unsigned closest_hit_geometry_index;
    unsigned closest_hit_primitive_index;
    unsigned closest_hit_instance_index;
    bool instance_sbt_contribution_valid;
    unsigned instance_sbt_contribution;
    bool v04_shadow_boundary_enabled;
    bool v04_tlas_binding_enforcement_enabled;
    rtcore_tlas_binding_snapshot v04_tlas_binding;
    bool v04_shadow_trace_input_valid;
    std::array<uint32_t, rtcore::abi_v04::kWordCount>
        v04_shadow_trace_input_words;
    unsigned long long handoff_window_base;
    memory_space *v04_live_handoff_memory;
    bool v04_request_owner_binding_valid;
    rtcore::v04::request_owner::lane_binding_v0 v04_request_owner_binding;
    rtcore::v04::private_frontier::owner_binding_v0
        v04_private_frontier_owner;
    bool v04_private_frontier_init_pending;
    bool v04_live_publication_armed;
    bool v04_live_publication_committed;
    unsigned v04_live_publication_reason;
    unsigned v04_live_publication_warp_uid;
    uint32_t v04_live_publication_word_mask;
    unsigned v04_live_publication_pending_chunk_mask;
    unsigned v04_live_publication_acked_chunk_mask;
    unsigned long long v04_live_publication_armed_cycle;
    std::array<uint32_t, rtcore::abi_v04::kWordCount>
        v04_live_publication_words;
    std::array<uint32_t, rtcore::abi_v04::kWordCount>
        v04_live_publication_preimage_words;
    rtcore::abi_v04::shadow::boundary_values
        v04_replay_committed_boundary_values;
    rtcore_boundary_candidate_snapshot boundary_candidate;
    bool continuation_boundary_pending;
    unsigned continuation_depth;
    unsigned continuation_segment_event_count;
    unsigned ready_order;
    unsigned request_state_bank_id;
    bool ready_node_bit;
    bool ready_primitive_bit;
    bool ready_stack_bit;
    bool ready_memory_bit;
    bool ready_result_bit;
    unsigned long long admitted_cycle;
    bool timing_trace_overflowed;
    rtcore_trace_timing_precision_class timing_precision_class;
    unsigned overflow_summary_events;
    rtcore_compact_trace_overflow_summary overflow_summary;
    rtcore_replay_lane_request_state state;
    bool memory_address_gen_latency_gate_pending;
    bool memory_address_gen_completed_valid;
    unsigned memory_address_gen_event_index;
    unsigned memory_address_gen_latency_cycles;
    unsigned long long memory_address_gen_armed_cycle;
    unsigned long long memory_address_gen_ready_cycle;
    bool memory_wake_latency_gate_pending;
    unsigned memory_wake_event_index;
    unsigned memory_wake_latency_cycles;
    unsigned long long memory_wake_armed_cycle;
    unsigned long long memory_wake_ready_cycle;
    bool memory_contention_gate_pending;
    unsigned memory_contention_event_index;
    unsigned memory_contention_cache_lines;
    unsigned memory_contention_cycles;
    unsigned memory_contention_queue_delay_cycles;
    unsigned long long memory_contention_armed_cycle;
    unsigned long long memory_contention_start_cycle;
    unsigned long long memory_contention_ready_cycle;
    bool v02_lsu_response_wait_gate_pending;
    unsigned v02_lsu_response_wait_event_index;
    unsigned v02_lsu_response_wait_chunk_count;
    unsigned v02_lsu_response_wait_completed_chunk_count;
    unsigned v02_lsu_response_wait_access_kind;
    unsigned long long v02_lsu_response_wait_armed_cycle;
    std::set<unsigned> v02_lsu_response_wait_completed_chunks;
    bool unit_latency_gate_pending;
    unsigned unit_latency_unit;
    unsigned unit_latency_event_index;
    unsigned unit_latency_cycles;
    unsigned long long unit_latency_armed_cycle;
    unsigned long long unit_latency_ready_cycle;
    std::vector<rtcore_compact_trace_event> events;
    std::vector<rtcore_boundary_candidate_snapshot> boundary_candidates;
};

static std::map<unsigned, rtcore_replay_lane_request>
    g_rtcore_replay_lane_requests;
static std::deque<rtcore_replay_lane_request>
    g_rtcore_replay_lane_request_state_capacity_pending_admissions;
static std::set<unsigned> g_rtcore_retire_lifecycle_busy_owners;

extern "C" bool rtcore_retire_lifecycle_busy_for_owner(
    unsigned owner_hw_sid)
{
    return g_rtcore_retire_lifecycle_busy_owners.count(owner_hw_sid) != 0;
}

extern "C" bool rtcore_reserve_retire_lifecycle_frontend(
    unsigned owner_hw_sid)
{
    return g_rtcore_retire_lifecycle_busy_owners.insert(owner_hw_sid).second;
}

struct rtcore_replay_lane_state_init_bandwidth_owner_cycle {
    bool valid;
    unsigned long long service_cycle;
    unsigned used;
};
static std::map<unsigned, rtcore_replay_lane_state_init_bandwidth_owner_cycle>
    g_rtcore_replay_lane_state_init_bandwidth_by_owner;
static std::map<unsigned long long, unsigned>
    g_rtcore_replay_v03_hw_ready_bank_rr_cursor_by_owner_unit;

struct rtcore_replay_warp_completion_entry_key {
    unsigned owner_hw_sid;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;

    bool operator<(const rtcore_replay_warp_completion_entry_key &other) const
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

enum rtcore_continuation_packet_kind {
    RTCORE_CONTINUATION_PACKET_FINAL = 0,
    RTCORE_CONTINUATION_PACKET_CONTINUATION = 1,
};

struct rtcore_continuation_return_packet {
    rtcore_continuation_return_packet()
        : valid(false), kind(RTCORE_CONTINUATION_PACKET_FINAL),
          owner_hw_sid(0), warp_uid(0), warp_id(0), active_mask(0),
          boundary_reached_mask(0), terminal_mask(0),
          resume_required_mask(0), shader_required_mask(0), miss_mask(0),
          closest_hit_mask(0), fault_mask(0), continuation_depth(0),
          reason_oracle_anyhit_mask(0), reason_oracle_intersection_mask(0),
          reason_synthetic_split_mask(0), reason_final_mask(0),
          reason_unsupported_mask(0), handoff_selector_valid_mask(0),
          handoff_candidate_valid_mask(0),
          handoff_software_return_valid_mask(0), context_profile_valid_mask(0),
          reported_attribute_metadata_valid_mask(0),
          inline_payload_location_valid_mask(0), inline_payload_base_word(16),
          max_inline_attribute_words(4),
          v04_shadow_boundary_image_valid_mask(0)
    {
        for (unsigned lane = 0; lane < 32; ++lane) {
            v_result[lane] = 0;
            context_layout_version[lane] = 0;
            context_valid_flags[lane] = 0;
            pipeline_profile_id[lane] = 0;
            bvh_format_profile_id[lane] = 0;
            boundary_candidates[lane] = rtcore_boundary_candidate_snapshot();
            for (unsigned word = 0; word < 32; ++word) {
                handoff_words[lane][word] = 0;
                v04_shadow_handoff_words[lane][word] = 0;
            }
        }
    }

    bool valid;
    rtcore_continuation_packet_kind kind;
    unsigned owner_hw_sid;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;
    unsigned boundary_reached_mask;
    unsigned terminal_mask;
    unsigned resume_required_mask;
    unsigned shader_required_mask;
    unsigned miss_mask;
    unsigned closest_hit_mask;
    unsigned fault_mask;
    unsigned continuation_depth;
    unsigned reason_oracle_anyhit_mask;
    unsigned reason_oracle_intersection_mask;
    unsigned reason_synthetic_split_mask;
    unsigned reason_final_mask;
    unsigned reason_unsupported_mask;
    unsigned handoff_selector_valid_mask;
    unsigned handoff_candidate_valid_mask;
    unsigned handoff_software_return_valid_mask;
    unsigned context_profile_valid_mask;
    unsigned reported_attribute_metadata_valid_mask;
    unsigned inline_payload_location_valid_mask;
    unsigned inline_payload_base_word;
    unsigned max_inline_attribute_words;
    unsigned v_result[32];
    unsigned context_layout_version[32];
    unsigned context_valid_flags[32];
    unsigned pipeline_profile_id[32];
    unsigned bvh_format_profile_id[32];
    rtcore_boundary_candidate_snapshot boundary_candidates[32];
    unsigned handoff_words[32][32];
    unsigned v04_shadow_boundary_image_valid_mask;
    unsigned v04_shadow_handoff_words[32][32];
};

struct rtcore_continuation_warp_boundary_state {
    rtcore_continuation_warp_boundary_state()
        : valid(false), owner_hw_sid(0), warp_uid(0), warp_id(0),
          active_mask(0), boundary_reached_mask(0), terminal_mask(0),
          resume_required_mask(0), shader_required_mask(0),
          continuation_depth(0), packet_published(false),
          pending_packet_valid(false), live_publication_wait_logged(false),
          reason_oracle_anyhit_mask(0), reason_oracle_intersection_mask(0),
          reason_synthetic_split_mask(0), reason_final_mask(0),
          reason_unsupported_mask(0)
    {
        for (unsigned lane = 0; lane < 32; ++lane) {
            boundary_candidates[lane] = rtcore_boundary_candidate_snapshot();
        }
    }

    bool valid;
    unsigned owner_hw_sid;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;
    unsigned boundary_reached_mask;
    unsigned terminal_mask;
    unsigned resume_required_mask;
    unsigned shader_required_mask;
    unsigned continuation_depth;
    bool packet_published;
    bool pending_packet_valid;
    bool live_publication_wait_logged;
    unsigned reason_oracle_anyhit_mask;
    unsigned reason_oracle_intersection_mask;
    unsigned reason_synthetic_split_mask;
    unsigned reason_final_mask;
    unsigned reason_unsupported_mask;
    rtcore_boundary_candidate_snapshot boundary_candidates[32];
    rtcore_continuation_return_packet pending_packet;
};

struct rtcore_resident_warp_continuation_state {
    rtcore_resident_warp_continuation_state()
        : valid(false), owner_hw_sid(0), warp_uid(0), warp_id(0),
          active_mask(0), resume_required_mask(0), shader_required_mask(0),
          ready_cycle(0), continuation_depth(0)
    {
    }

    bool valid;
    unsigned owner_hw_sid;
    unsigned warp_uid;
    unsigned warp_id;
    unsigned active_mask;
    unsigned resume_required_mask;
    unsigned shader_required_mask;
    unsigned long long ready_cycle;
    unsigned continuation_depth;
};

struct rtcore_resident_rt_warp_record_key {
    unsigned owner_hw_sid;
    unsigned warp_id;

    bool operator<(const rtcore_resident_rt_warp_record_key &other) const
    {
        if (owner_hw_sid != other.owner_hw_sid) {
            return owner_hw_sid < other.owner_hw_sid;
        }
        return warp_id < other.warp_id;
    }
};

struct rtcore_resident_rt_warp_lane_identity {
    rtcore_resident_rt_warp_lane_identity()
        : valid(false), thread_uid(0), context_ptr(0), handoff_window_base(0),
          token_id(0), token_allocator_generation(0), window_generation(0),
          v04_request_owner_binding_valid(false),
          v04_request_owner_binding()
    {
    }

    bool valid;
    unsigned thread_uid;
    unsigned long long context_ptr;
    unsigned long long handoff_window_base;
    unsigned token_id;
    unsigned token_allocator_generation;
    unsigned window_generation;
    bool v04_request_owner_binding_valid;
    rtcore::v04::request_owner::lane_binding_v0
        v04_request_owner_binding;
};

struct rtcore_resident_rt_warp_record {
    rtcore_resident_rt_warp_record()
        : valid(false), owner_hw_sid(0), warp_id(0), current_warp_uid(0),
          current_static_inst_uid(0), active_mask(0), bound_lane_mask(0),
          admitted_lane_mask(0), retired_lane_mask(0), resident_generation(0),
          resubmit_count(0), v04_request_owner_binding_valid(false),
          v04_resident_warp_slot(0), v04_request_owner_active_mask(0),
          v04_private_frontier_live_init_valid(false),
          v04_private_frontier_init_committed(false),
          v04_private_frontier_init_active_mask(0)
    {
    }

    bool valid;
    unsigned owner_hw_sid;
    unsigned warp_id;
    unsigned current_warp_uid;
    unsigned current_static_inst_uid;
    unsigned active_mask;
    unsigned bound_lane_mask;
    unsigned admitted_lane_mask;
    unsigned retired_lane_mask;
    unsigned resident_generation;
    unsigned resubmit_count;
    bool v04_request_owner_binding_valid;
    unsigned v04_resident_warp_slot;
    unsigned v04_request_owner_active_mask;
    bool v04_private_frontier_live_init_valid;
    bool v04_private_frontier_init_committed;
    unsigned v04_private_frontier_init_active_mask;
    rtcore_resident_rt_warp_lane_identity lane_identity[32];
};

struct rtcore_replay_warp_completion_entry_state {
    bool valid;
    rtcore_replay_warp_completion_entry_key key;
    unsigned admitted_lane_mask;
    unsigned completed_lane_mask;
    unsigned result_valid_mask;
    unsigned completed_lane_count;
    unsigned result_reg_base;
    unsigned result_data_slot[32];
    unsigned lane_status[32];
    unsigned packet_schema_version;
    unsigned context_profile_valid_mask;
    unsigned reported_attribute_metadata_valid_mask;
    unsigned inline_payload_location_valid_mask;
    unsigned inline_payload_base_word;
    unsigned max_inline_attribute_words;
    unsigned lane_completion_valid_mask;
    unsigned terminal_lane_mask;
    unsigned continuation_lane_mask;
    unsigned unsupported_reason_mask;
    unsigned handoff_selector_valid_mask;
    unsigned handoff_candidate_valid_mask;
    unsigned handoff_software_return_valid_mask;
    unsigned lane_completion_reason[32];
    unsigned lane_continuation_depth[32];
    unsigned context_layout_version[32];
    unsigned context_valid_flags[32];
    unsigned pipeline_profile_id[32];
    unsigned bvh_format_profile_id[32];
    rtcore_boundary_candidate_snapshot boundary_candidates[32];
    unsigned handoff_words[32][32];
    unsigned v04_shadow_boundary_image_valid_mask;
    unsigned v04_shadow_handoff_words[32][32];
    bool all_active_lanes_complete;
    bool all_active_lanes_complete_logged;
    bool scoreboard_handoff_ready;
    bool scoreboard_handoff_delivered;
    unsigned long long scoreboard_handoff_cycle;
};

static std::map<rtcore_replay_warp_completion_entry_key,
                rtcore_replay_warp_completion_entry_state>
    g_rtcore_replay_warp_completion_entries;
static std::map<rtcore_replay_warp_completion_entry_key,
                rtcore_continuation_warp_boundary_state>
    g_rtcore_continuation_warp_boundary_states;
static std::map<rtcore_replay_warp_completion_entry_key,
                rtcore_resident_warp_continuation_state>
    g_rtcore_resident_warp_continuation_states;
static std::map<rtcore_resident_rt_warp_record_key,
                rtcore_resident_rt_warp_record>
    g_rtcore_resident_rt_warp_records;
static unsigned g_rtcore_next_resident_rt_warp_generation = 1;
static std::map<unsigned, rtcore::v04::request_owner::allocator_state_v0>
    g_rtcore_v04_request_owner_allocators;

static rtcore::v04::request_owner::allocator_state_v0 &
rtcore_v04_request_owner_allocator_for(unsigned owner_hw_sid)
{
    rtcore::v04::request_owner::allocator_state_v0 &allocator =
        g_rtcore_v04_request_owner_allocators[owner_hw_sid];
    if (!allocator.initialized) {
        rtcore::v04::request_owner::initialize_allocator(
            &allocator);
    }
    return allocator;
}

struct rtcore_replay_issue_budget {
    unsigned node_issue_budget;
    unsigned primitive_issue_budget;
    unsigned stack_issue_budget;
    unsigned warp_completion_ingress_budget;
};

struct rtcore_replay_scoreboard_result_handoff_stats {
    unsigned ready_warp_count;
    unsigned delivered_count;
    unsigned blocked_count;
    unsigned max_ready_warp_count;
    unsigned max_blocked_count;
};

struct rtcore_continuation_stats {
    rtcore_continuation_stats()
        : rtcore_continuation_packet_count(0),
          rtcore_continuation_lane_count(0),
          rtcore_continuation_warp_wakeup_count(0),
          rtcore_continuation_wait_cycles(0),
          rtcore_modeled_resubmit_count(0),
          rtcore_modeled_resubmit_lane_count(0),
          rtcore_continuation_synthetic_boundary_count(0),
          rtcore_continuation_oracle_anyhit_boundary_count(0),
          rtcore_continuation_oracle_intersection_boundary_count(0),
          rtcore_continuation_max_depth(0)
    {
    }

    unsigned long long rtcore_continuation_packet_count;
    unsigned long long rtcore_continuation_lane_count;
    unsigned long long rtcore_continuation_warp_wakeup_count;
    unsigned long long rtcore_continuation_wait_cycles;
    unsigned long long rtcore_modeled_resubmit_count;
    unsigned long long rtcore_modeled_resubmit_lane_count;
    unsigned long long rtcore_continuation_synthetic_boundary_count;
    unsigned long long rtcore_continuation_oracle_anyhit_boundary_count;
    unsigned long long rtcore_continuation_oracle_intersection_boundary_count;
    unsigned rtcore_continuation_max_depth;
};

struct rtcore_replay_service_tick_result {
    bool progressed;
    bool memory_progressed;
    bool ready_progressed;
    bool unit_wake_progressed;
    bool ready_issue_progressed;
    bool scoreboard_handoff_progressed;
    rtcore_replay_service_cycle_identity_snapshot last_progress_identity;
};

struct rtcore_replay_service_tick_stats {
    unsigned tick_attempts;
    unsigned ticks_progressed;
    unsigned memory_ticks_progressed;
    unsigned ready_ticks_progressed;
    unsigned service_stage_memory_wake_progressed_count;
    unsigned service_stage_unit_wake_progressed_count;
    unsigned service_stage_ready_issue_progressed_count;
    unsigned service_stage_memory_and_unit_progressed_count;
    unsigned service_stage_memory_and_ready_issue_progressed_count;
    unsigned service_stage_unit_and_ready_issue_progressed_count;
    unsigned service_stage_all_progressed_count;
    unsigned scoreboard_handoff_progressed_count;
};

struct rtcore_replay_overflow_summary_estimate_stats {
    unsigned overflow_summary_requests_consumed;
    unsigned estimated_overflow_node_events;
    unsigned estimated_overflow_primitive_events;
    unsigned estimated_overflow_stack_events;
    unsigned estimated_overflow_memory_wait_events;
    unsigned estimated_overflow_memory_bytes;
    unsigned estimated_overflow_completion_events;
};

struct rtcore_replay_memory_demand_estimate_stats {
    unsigned explicit_memory_wait_events;
    unsigned explicit_fetch_memory_bytes;
    unsigned overflow_memory_wait_events;
    unsigned overflow_memory_bytes;
    unsigned estimated_memory_demand_events;
    unsigned estimated_memory_demand_bytes;
    unsigned estimated_cache_lines_64b;
};

struct rtcore_replay_memory_latency_policy_stats {
    unsigned policy_evaluations;
    unsigned cache_line_latency_cycles;
    unsigned memory_wait_latency_cycles;
    unsigned estimated_cache_latency_cycles;
    unsigned estimated_memory_wait_latency_cycles;
    unsigned estimated_total_memory_latency_cycles;
};

struct rtcore_replay_memory_latency_blocked_cycle_stats {
    unsigned policy_evaluations;
    unsigned blocked_cycle_total;
    unsigned max_delta_blocked_cycles;
    unsigned last_delta_blocked_cycles;
    unsigned current_estimated_total_memory_latency_cycles;
};

struct rtcore_replay_memory_wake_latency_gate_stats {
    unsigned gate_evaluations;
    unsigned gate_armed_count;
    unsigned gate_blocked_count;
    unsigned gate_woken_count;
    unsigned max_latency_cycles;
    unsigned max_blocked_cycles;
};

struct rtcore_replay_memory_contention_gate_stats {
    unsigned gate_evaluations;
    unsigned gate_armed_count;
    unsigned gate_blocked_count;
    unsigned gate_woken_count;
    unsigned capacity_blocked_count;
    unsigned max_cache_lines;
    unsigned max_contention_cycles;
    unsigned max_queue_delay_cycles;
    unsigned max_inflight_reservations;
};

struct rtcore_replay_memory_contention_owner_state {
    unsigned long long next_available_cycle;
    unsigned reservations;
    unsigned inflight_reservations;
};

struct rtcore_replay_unit_latency_gate_stats {
    unsigned gate_evaluations;
    unsigned node_gate_armed_count;
    unsigned primitive_gate_armed_count;
    unsigned stack_gate_armed_count;
    unsigned gate_blocked_count;
    unsigned gate_woken_count;
    unsigned max_latency_cycles;
    unsigned max_blocked_cycles;
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
    unsigned warp_completion_ingress_attempts;
    unsigned warp_completion_ingress_issued;
    unsigned warp_completion_ingress_budget_exhausted;
};

struct rtcore_replay_data_path_access_stats {
    unsigned lane_request_state_identity_reads;
    unsigned lane_request_state_identity_writes;
    unsigned request_state_reads;
    unsigned request_state_writes;
    unsigned max_lane_request_state_entries;
};

struct rtcore_replay_resource_route_stats {
    unsigned lane_requests;
    unsigned total_trace_events;
    unsigned memory_routed_events;
    unsigned node_routed_events;
    unsigned primitive_routed_events;
    unsigned stack_routed_events;
    unsigned completion_routed_events;
    unsigned node_fetch_events;
    unsigned primitive_fetch_events;
    unsigned node_test_events;
    unsigned primitive_test_events;
    unsigned hit_update_events;
    unsigned stack_events;
    unsigned completion_events;
    unsigned fetch_events_routed_to_memory;
    unsigned fetch_events_routed_to_compute;
    unsigned test_events_routed_to_compute;
    unsigned test_events_routed_to_memory;
    unsigned hit_update_events_folded_into_primitive;
};

struct rtcore_replay_data_path_access_snapshot {
    unsigned lane_request_state_identity_reads;
    unsigned lane_request_state_identity_writes;
    unsigned request_state_reads;
    unsigned request_state_writes;
    unsigned lane_request_state_identity_accesses;
    unsigned request_state_accesses;
};

struct rtcore_replay_v03_hw_request_state_scoreboard_stats {
    unsigned evaluations;
    unsigned max_request_state_hot_delta;
    unsigned max_scoreboard_update_count;
};

struct rtcore_replay_v03_hw_unit_state_wake_service_stats {
    unsigned evaluations;
    unsigned wake_attempt_count;
    unsigned wake_progress_count;
    unsigned request_state_unit_wake_attempt_count;
    unsigned request_state_unit_wake_progress_count;
    unsigned max_request_state_executing_count;
    unsigned max_request_state_ready_pending_count;
    unsigned max_wake_progress_count;
    unsigned max_request_state_unit_wake_progress_count;
};

struct rtcore_replay_v03_hw_memory_outstanding_stats {
    unsigned evaluations;
    unsigned memory_address_gen_attempt_count;
    unsigned memory_address_gen_issued_count;
    unsigned memory_address_gen_blocked_count;
    unsigned memory_address_gen_latency_blocked_count;
    unsigned memory_ready_issue_attempt_count;
    unsigned memory_ready_issue_count;
    unsigned memory_outstanding_capacity_blocked_count;
    unsigned memory_wake_attempt_count;
    unsigned memory_wake_progress_count;
    unsigned memory_outstanding_table_register_count;
    unsigned memory_outstanding_table_release_count;
    unsigned max_memory_address_gen_issued_count;
    unsigned max_memory_address_gen_blocked_count;
    unsigned max_memory_address_gen_latency_blocked_count;
    unsigned max_memory_ready_issue_count;
    unsigned max_memory_outstanding_capacity_blocked_count;
    unsigned max_outstanding_entry_count;
    unsigned max_response_fanout_waiter_count;
    unsigned max_outstanding_chunk_count;
    unsigned max_memory_outstanding_table_active_entry_count;
    unsigned max_memory_outstanding_table_response_fanout_waiter_count;
    unsigned max_memory_outstanding_table_register_count;
    unsigned max_memory_outstanding_table_release_count;
    unsigned max_memory_wake_progress_count;
};

enum rtcore_replay_memory_outstanding_kind {
    RTCORE_REPLAY_MEMORY_OUTSTANDING_V02_LSU_RESPONSE_WAIT = 1,
    RTCORE_REPLAY_MEMORY_OUTSTANDING_CONTENTION = 2,
    RTCORE_REPLAY_MEMORY_OUTSTANDING_WAKE_LATENCY = 3,
};

struct rtcore_replay_memory_outstanding_key {
    unsigned owner_hw_sid;
    unsigned thread_uid;
    unsigned event_index;
    unsigned kind;

    bool operator<(const rtcore_replay_memory_outstanding_key &other) const
    {
        if (owner_hw_sid != other.owner_hw_sid) {
            return owner_hw_sid < other.owner_hw_sid;
        }
        if (thread_uid != other.thread_uid) {
            return thread_uid < other.thread_uid;
        }
        if (event_index != other.event_index) {
            return event_index < other.event_index;
        }
        return kind < other.kind;
    }
};

struct rtcore_replay_memory_unit_transaction_key {
    unsigned owner_hw_sid;
    unsigned long long service_cycle;
    unsigned long long aligned_32b_addr;
    bool is_write;

    bool operator<(const rtcore_replay_memory_unit_transaction_key &other) const
    {
        if (owner_hw_sid != other.owner_hw_sid) {
            return owner_hw_sid < other.owner_hw_sid;
        }
        if (service_cycle != other.service_cycle) {
            return service_cycle < other.service_cycle;
        }
        if (aligned_32b_addr != other.aligned_32b_addr) {
            return aligned_32b_addr < other.aligned_32b_addr;
        }
        return is_write < other.is_write;
    }
};

struct rtcore_replay_memory_outstanding_entry {
    bool active;
    rtcore_replay_memory_outstanding_key key;
    unsigned chunk_count;
    unsigned response_fanout_waiter_count;
    unsigned long long issue_cycle;
    bool has_transaction_key;
    rtcore_replay_memory_unit_transaction_key transaction_key;
    std::set<rtcore_replay_memory_unit_transaction_key> transaction_keys;
};

struct rtcore_replay_lane_request_state_capacity_gate_stats {
    unsigned evaluations;
    unsigned admitted_count;
    unsigned blocked_count;
    unsigned released_count;
    unsigned lane_state_init_bandwidth_blocked_count;
    unsigned last_occupancy;
    unsigned last_capacity_blocked;
    unsigned last_init_bandwidth_blocked;
    unsigned last_admitted;
    unsigned last_released;
    unsigned last_pending_admissions;
    unsigned last_lane_state_init_bandwidth_used;
    unsigned max_lane_request_state_occupancy;
    unsigned max_pending_admissions;
    unsigned max_lane_state_init_bandwidth_used_per_cycle;
};

struct rtcore_replay_service_tick_stats_snapshot {
    bool valid;
    unsigned tick_attempts;
    unsigned ticks_progressed;
    unsigned memory_ticks_progressed;
    unsigned ready_ticks_progressed;
};

struct rtcore_replay_model_summary_progress_snapshot {
    bool valid;
    unsigned service_ticks_progressed;
    unsigned admitted_lane_requests;
    unsigned completed_lane_requests;
    unsigned total_unit_issued;
    unsigned total_unit_busy_cycles;
    unsigned memory_blocked_events;
    unsigned memory_wake_blocked_count;
    unsigned memory_wake_max_blocked_cycles;
    unsigned memory_contention_gate_armed_count;
    unsigned memory_contention_gate_blocked_count;
    unsigned memory_contention_gate_woken_count;
    unsigned memory_contention_max_contention_cycles;
    unsigned memory_contention_max_queue_delay_cycles;
    unsigned memory_contention_capacity_blocked_count;
    unsigned lane_request_state_capacity_blocked_count;
    unsigned lane_state_init_bandwidth_blocked_count;
    unsigned lane_request_state_capacity_max_occupancy;
    unsigned lane_request_state_capacity_max_pending_admissions;
    unsigned warp_aggregated_completion_count;
    unsigned scoreboard_handoff_ready_warp_count;
    unsigned scoreboard_handoff_delivered_count;
    unsigned scoreboard_handoff_blocked_count;
    unsigned scoreboard_handoff_max_blocked_count;
    unsigned data_path_lane_request_state_identity_accesses;
    unsigned data_path_request_state_accesses;
    unsigned data_path_max_lane_request_state_entries;
    unsigned long long max_observed_ready_cycle;
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

static rtcore_replay_service_tick_stats g_rtcore_replay_service_tick_stats;
static rtcore_replay_overflow_summary_estimate_stats
    g_rtcore_replay_overflow_summary_estimate_stats;
static rtcore_replay_memory_demand_estimate_stats
    g_rtcore_replay_memory_demand_estimate_stats;
static rtcore_replay_memory_latency_policy_stats
    g_rtcore_replay_memory_latency_policy_stats;
static rtcore_replay_memory_latency_blocked_cycle_stats
    g_rtcore_replay_memory_latency_blocked_cycle_stats;
static rtcore_replay_memory_wake_latency_gate_stats
    g_rtcore_replay_memory_wake_latency_gate_stats;
static rtcore_replay_memory_contention_gate_stats
    g_rtcore_replay_memory_contention_gate_stats;
static std::map<unsigned, rtcore_replay_memory_contention_owner_state>
    g_rtcore_replay_memory_contention_owner_states;
static rtcore_replay_unit_latency_gate_stats
    g_rtcore_replay_unit_latency_gate_stats;
static rtcore_replay_unit_arbitration_stats
    g_rtcore_replay_unit_arbitration_stats;
static rtcore_replay_scoreboard_result_handoff_stats
    g_rtcore_replay_scoreboard_result_handoff_stats;
static rtcore_continuation_stats g_rtcore_continuation_stats;
static bool g_rtcore_continuation_final_summary_registered = false;
static bool g_rtcore_continuation_final_summary_emitted = false;
static rtcore_replay_data_path_access_stats
    g_rtcore_replay_data_path_access_stats;
static rtcore_replay_resource_route_stats g_rtcore_replay_resource_route_stats;
static rtcore_replay_v03_hw_request_state_scoreboard_stats
    g_rtcore_replay_v03_hw_request_state_scoreboard_stats;
static rtcore_replay_v03_hw_unit_state_wake_service_stats
    g_rtcore_replay_v03_hw_unit_state_wake_service_stats;
static rtcore_replay_v03_hw_memory_outstanding_stats
    g_rtcore_replay_v03_hw_memory_outstanding_stats;
static std::map<rtcore_replay_memory_outstanding_key,
                rtcore_replay_memory_outstanding_entry>
    g_rtcore_replay_memory_outstanding_table;
static std::map<rtcore_replay_memory_unit_transaction_key, unsigned>
    g_rtcore_replay_memory_unit_transaction_groups_this_cycle;
static bool
    g_rtcore_replay_memory_unit_transaction_groups_cycle_valid = false;
static unsigned long long
    g_rtcore_replay_memory_unit_transaction_groups_cycle = 0;
static rtcore_replay_lane_request_state_capacity_gate_stats
    g_rtcore_replay_lane_request_state_capacity_gate_stats;
static rtcore_replay_memory_unit_request_descriptor_stats
    g_rtcore_replay_memory_unit_request_descriptor_stats;
static std::map<rtcore_v02_lsu_merge_key, unsigned>
    g_rtcore_replay_memory_unit_request_descriptor_merge_counts;
static rtcore_v02_lsu_response_wait_stats
    g_rtcore_v02_lsu_response_wait_stats;
static std::map<unsigned,
                std::deque<rtcore_memory_unit_request_snapshot> >
    g_rtcore_memory_unit_request_snapshots_by_owner;
static std::map<unsigned, rtcore::v04::private_shared::backing_state_v0>
    g_rtcore_v04_private_shared_backing_by_owner;

static rtcore::v04::private_shared::backing_state_v0 &
rtcore_v04_private_shared_backing_for(unsigned owner_hw_sid)
{
    rtcore::v04::private_shared::backing_state_v0 &state =
        g_rtcore_v04_private_shared_backing_by_owner[owner_hw_sid];
    if (!state.initialized) {
        rtcore::v04::private_shared::initialize(&state, owner_hw_sid);
    }
    return state;
}
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
static unsigned g_rtcore_replay_data_path_access_stats_logs_emitted = 0;
static unsigned
    g_rtcore_replay_v03_hw_request_state_scoreboard_stats_logs_emitted = 0;
static unsigned
    g_rtcore_replay_v03_hw_unit_state_wake_service_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_v03_hw_memory_outstanding_stats_logs_emitted =
    0;
static unsigned
    g_rtcore_replay_lane_request_state_capacity_gate_stats_logs_emitted = 0;
static unsigned g_rtcore_compact_trace_overflow_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_overflow_summary_estimate_stats_logs_emitted =
    0;
static unsigned g_rtcore_replay_memory_demand_estimate_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_memory_latency_policy_stats_logs_emitted = 0;
static unsigned
    g_rtcore_replay_memory_latency_blocked_cycle_stats_logs_emitted = 0;
static unsigned
    g_rtcore_replay_memory_latency_blocked_cycle_last_total_cycles = 0;
static unsigned g_rtcore_replay_memory_wake_latency_gate_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_memory_contention_gate_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_unit_latency_gate_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_model_summary_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_memory_unit_request_descriptor_stats_logs_emitted = 0;
static unsigned g_rtcore_v02_lsu_response_wait_stats_logs_emitted = 0;
static unsigned g_rtcore_replay_resource_route_stats_logs_emitted = 0;
static std::map<unsigned, rtcore_replay_model_summary_progress_snapshot>
    g_rtcore_replay_model_summary_progress_snapshots;
static unsigned g_rtcore_next_replay_ready_order = 0;

static bool rtcore_replay_model_preset_simple_enabled()
{
    static int enabled = []() {
        const char *value = getenv("VULKAN_SIM_RTCORE_REPLAY_MODEL_PRESET");
        if (!value || value[0] == '\0' || strcmp(value, "0") == 0) {
            return 0;
        }
        return (strcmp(value, "simple") == 0 ||
                strcmp(value, "simple_replay_local") == 0 ||
                strcmp(value, "1") == 0)
                   ? 1
                   : 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_env_enabled_or_model_preset(const char *name,
                                                       bool preset_enabled)
{
    const char *value = getenv(name);
    if (value && value[0] != '\0') {
        return strcmp(value, "0") != 0;
    }
    return preset_enabled && rtcore_replay_model_preset_simple_enabled();
}

static unsigned rtcore_replay_uint_config_or_model_preset(
    const char *name, unsigned default_value, unsigned preset_value,
    unsigned max_value, bool allow_zero)
{
    const char *value = getenv(name);
    if (!value || value[0] == '\0') {
        return rtcore_replay_model_preset_simple_enabled() ? preset_value
                                                          : default_value;
    }

    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    if (parsed == 0 && !allow_zero) {
        return default_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return static_cast<unsigned>(parsed);
}

enum rtcore_continuation_model {
    RTCORE_CONTINUATION_MODEL_OFF = 0,
    RTCORE_CONTINUATION_MODEL_SYNTHETIC_SPLIT = 1,
    RTCORE_CONTINUATION_MODEL_ORACLE_SHADER_BOUNDARY = 2,
};

static rtcore_continuation_model rtcore_continuation_model_config()
{
    const char *value = getenv("VULKAN_SIM_RTCORE_CONTINUATION_MODEL");
    if (value == NULL || value[0] == '\0') {
        return rtcore_custom_path_mode_enabled()
                   ? RTCORE_CONTINUATION_MODEL_ORACLE_SHADER_BOUNDARY
                   : RTCORE_CONTINUATION_MODEL_OFF;
    }
    if (strcmp(value, "off") == 0) {
        return RTCORE_CONTINUATION_MODEL_OFF;
    }
    if (strcmp(value, "synthetic_split") == 0) {
        return RTCORE_CONTINUATION_MODEL_SYNTHETIC_SPLIT;
    }
    if (strcmp(value, "oracle_shader_boundary") == 0) {
        return RTCORE_CONTINUATION_MODEL_ORACLE_SHADER_BOUNDARY;
    }
    return RTCORE_CONTINUATION_MODEL_OFF;
}

extern "C" bool rtcore_oracle_shader_boundary_continuation_enabled()
{
    return rtcore_continuation_model_config() ==
           RTCORE_CONTINUATION_MODEL_ORACLE_SHADER_BOUNDARY;
}

extern "C" bool rtcore_custom_submit_continuation_contract_valid()
{
    if (!rtcore_custom_path_mode_enabled()) {
        return false;
    }
    const char *value = getenv("VULKAN_SIM_RTCORE_CONTINUATION_MODEL");
    if (value == NULL || value[0] == '\0') {
        return true;
    }
    return strcmp(value, "oracle_shader_boundary") == 0;
}

static bool rtcore_continuation_model_enabled()
{
    return rtcore_continuation_model_config() != RTCORE_CONTINUATION_MODEL_OFF;
}

static bool rtcore_shader_continuation_resubmit_bridge_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_SHADER_CONTINUATION_RESUBMIT_BRIDGE");
        if (!value || value[0] == '\0') {
            return 0;
        }
        return strcmp(value, "0") != 0 ? 1 : 0;
    }();
    return enabled != 0;
}

static const char *rtcore_continuation_model_name(rtcore_continuation_model model)
{
    switch (model) {
    case RTCORE_CONTINUATION_MODEL_OFF:
        return "off";
    case RTCORE_CONTINUATION_MODEL_SYNTHETIC_SPLIT:
        return "synthetic_split";
    case RTCORE_CONTINUATION_MODEL_ORACLE_SHADER_BOUNDARY:
        return "oracle_shader_boundary";
    }
    return "unknown";
}

static bool rtcore_continuation_stats_nonzero()
{
    return g_rtcore_continuation_stats.rtcore_continuation_packet_count != 0 ||
           g_rtcore_continuation_stats.rtcore_continuation_lane_count != 0 ||
           g_rtcore_continuation_stats.rtcore_continuation_warp_wakeup_count !=
               0 ||
           g_rtcore_continuation_stats.rtcore_continuation_wait_cycles != 0 ||
           g_rtcore_continuation_stats.rtcore_modeled_resubmit_count != 0 ||
           g_rtcore_continuation_stats.rtcore_modeled_resubmit_lane_count != 0 ||
           g_rtcore_continuation_stats
                   .rtcore_continuation_synthetic_boundary_count != 0 ||
           g_rtcore_continuation_stats
                   .rtcore_continuation_oracle_anyhit_boundary_count != 0 ||
           g_rtcore_continuation_stats
                   .rtcore_continuation_oracle_intersection_boundary_count !=
               0 ||
           g_rtcore_continuation_stats.rtcore_continuation_max_depth != 0;
}

static void rtcore_log_continuation_final_summary()
{
    if (g_rtcore_continuation_final_summary_emitted) {
        return;
    }
    g_rtcore_continuation_final_summary_emitted = true;

    const rtcore_continuation_model model = rtcore_continuation_model_config();
    if (model == RTCORE_CONTINUATION_MODEL_OFF &&
        !rtcore_continuation_stats_nonzero()) {
        return;
    }

    printf("GPGPU-Sim RTCORE_CONTINUATION_FINAL_SUMMARY "
           "continuation_model=%s "
           "rtcore_continuation_packet_count=%llu "
           "rtcore_continuation_lane_count=%llu "
           "rtcore_continuation_warp_wakeup_count=%llu "
           "rtcore_continuation_wait_cycles=%llu "
           "rtcore_modeled_resubmit_count=%llu "
           "rtcore_modeled_resubmit_lane_count=%llu "
           "rtcore_continuation_synthetic_boundary_count=%llu "
           "rtcore_continuation_oracle_anyhit_boundary_count=%llu "
           "rtcore_continuation_oracle_intersection_boundary_count=%llu "
           "rtcore_continuation_max_depth=%u\n",
           rtcore_continuation_model_name(model),
           g_rtcore_continuation_stats.rtcore_continuation_packet_count,
           g_rtcore_continuation_stats.rtcore_continuation_lane_count,
           g_rtcore_continuation_stats.rtcore_continuation_warp_wakeup_count,
           g_rtcore_continuation_stats.rtcore_continuation_wait_cycles,
           g_rtcore_continuation_stats.rtcore_modeled_resubmit_count,
           g_rtcore_continuation_stats.rtcore_modeled_resubmit_lane_count,
           g_rtcore_continuation_stats
               .rtcore_continuation_synthetic_boundary_count,
           g_rtcore_continuation_stats
               .rtcore_continuation_oracle_anyhit_boundary_count,
           g_rtcore_continuation_stats
               .rtcore_continuation_oracle_intersection_boundary_count,
           g_rtcore_continuation_stats.rtcore_continuation_max_depth);
    fflush(stdout);
}

static void rtcore_register_continuation_final_summary()
{
    if (g_rtcore_continuation_final_summary_registered) {
        return;
    }
    g_rtcore_continuation_final_summary_registered = true;
    std::atexit(rtcore_log_continuation_final_summary);
}

static unsigned rtcore_continuation_shader_latency_cycles_config()
{
    static unsigned latency = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_CONTINUATION_SHADER_LATENCY_CYCLES", 0, 0,
        1048576, true);
    return latency;
}

static unsigned rtcore_continuation_segment_event_budget_config()
{
    static unsigned budget = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_CONTINUATION_SEGMENT_EVENT_BUDGET", 0, 0,
        1048576, true);
    return budget;
}

static unsigned rtcore_continuation_max_resubmits_per_lane_config()
{
    static unsigned max_resubmits = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_CONTINUATION_MAX_RESUBMITS_PER_LANE", 4, 4,
        1048576, true);
    return max_resubmits;
}

static bool rtcore_bounded_trace_collection_enabled()
{
    static int enabled = []() {
        return rtcore_replay_env_enabled_or_model_preset(
                   "VULKAN_SIM_RTCORE_BOUNDED_TRACE_COLLECTION", true)
                   ? 1
                   : 0;
    }();
    return enabled != 0;
}

static unsigned rtcore_compact_trace_events_per_lane_config()
{
    static unsigned events_per_lane = []() {
        return rtcore_replay_uint_config_or_model_preset(
            "VULKAN_SIM_RTCORE_COMPACT_TRACE_EVENTS_PER_LANE",
            RTCORE_COMPACT_TRACE_DEFAULT_EVENTS_PER_LANE, 64,
            RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR, false);
    }();
    return events_per_lane;
}

static bool rtcore_replay_admission_enabled()
{
    static int enabled = []() {
        return rtcore_replay_env_enabled_or_model_preset(
                   "VULKAN_SIM_RTCORE_REPLAY_ADMISSION", true)
                   ? 1
                   : 0;
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

static bool rtcore_replay_overflow_summary_estimate_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_OVERFLOW_SUMMARY_ESTIMATE_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_demand_estimate_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_DEMAND_ESTIMATE_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_latency_policy_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_LATENCY_POLICY_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_latency_blocked_cycle_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_MEMORY_LATENCY_BLOCKED_CYCLE_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_wake_latency_gate_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_WAKE_LATENCY_GATE");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_wake_latency_gate_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_MEMORY_WAKE_LATENCY_GATE_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_contention_gate_enabled()
{
    static int enabled = []() {
        return rtcore_replay_env_enabled_or_model_preset(
                   "VULKAN_SIM_RTCORE_REPLAY_MEMORY_CONTENTION_GATE", true)
                   ? 1
                   : 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_contention_gate_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_MEMORY_CONTENTION_GATE_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_service_tick_enabled()
{
    static int enabled = []() {
        return rtcore_replay_env_enabled_or_model_preset(
                   "VULKAN_SIM_RTCORE_REPLAY_SERVICE_TICK", true)
                   ? 1
                   : 0;
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

static bool rtcore_replay_v03_hw_banked_ready_selection_enabled()
{
    return true;
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

static bool rtcore_replay_unit_latency_gate_enabled()
{
    static int enabled = []() {
        return rtcore_replay_env_enabled_or_model_preset(
                   "VULKAN_SIM_RTCORE_REPLAY_UNIT_LATENCY_GATE", true)
                   ? 1
                   : 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_unit_latency_gate_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_UNIT_LATENCY_GATE_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_model_summary_stats_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MODEL_SUMMARY_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_unit_request_descriptor_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_UNIT_REQUEST_DESCRIPTOR_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_unit_request_offer_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_UNIT_REQUEST_OFFER");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_v02_lsu_stack_sideband_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_V02_LSU_STACK_SIDEBAND");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_unit_l1d_client_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_UNIT_L1D_CLIENT");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_memory_unit_response_wait_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_UNIT_RESPONSE_WAIT");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_memory_unit_path_active()
{
    return rtcore_replay_memory_unit_request_offer_enabled() &&
           rtcore_replay_memory_unit_l1d_client_enabled() &&
           rtcore_memory_unit_response_wait_enabled();
}

static bool rtcore_v04_live_handoff_publication_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_LIVE_HANDOFF_PUBLICATION") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_shader_return_consumer_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_SHADOW_SHADER_RETURN_CONSUMER") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_functional_shader_return_authority_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_FUNCTIONAL_SHADER_RETURN_AUTHORITY") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_node_candidate_kernel_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_NODE_CANDIDATE_KERNEL") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_node_child_route_kernel_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_NODE_CHILD_ROUTE_KERNEL") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_stack_push_remainder_kernel_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_STACK_PUSH_REMAINDER_KERNEL") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_stack_pop_next_kernel_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_STACK_POP_NEXT_KERNEL") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_private_frontier_owner_layout_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_PRIVATE_FRONTIER_OWNER_LAYOUT") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_request_owner_binding_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_REQUEST_OWNER_BINDING") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_private_frontier_live_init_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_PRIVATE_FRONTIER_LIVE_INIT") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

extern "C" bool
rtcore_v04_private_frontier_live_init_memory_issue_profile_active()
{
    return rtcore_v04_private_frontier_live_init_enabled();
}

static bool rtcore_v04_typed_primitive_candidate_kernel_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_PRIMITIVE_CANDIDATE_KERNEL") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_procedural_boundary_seed_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_PROCEDURAL_BOUNDARY_SEED") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_instance_boundary_seed_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_INSTANCE_BOUNDARY_SEED") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_blas_decode_context_bridge_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_BLAS_DECODE_CONTEXT_BRIDGE") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_producer_backed_blas_root_descriptor_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_PRODUCER_BACKED_BLAS_ROOT_DESCRIPTOR") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_producer_backed_instance_blas_reference_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_PRODUCER_BACKED_INSTANCE_BLAS_REFERENCE_TABLE") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_typed_instance_enter_transition_enabled()
{
    static int enabled = []() {
        return rtcore_candidate_gate_state_for(
                   "VULKAN_SIM_RTCORE_ABI_V04_TYPED_INSTANCE_ENTER_TRANSITION_KERNEL") ==
               RTCORE_CANDIDATE_GATE_ENABLED;
    }();
    return enabled != 0;
}

static bool rtcore_v04_tlas_binding_enforcement_gate_enabled()
{
    return rtcore_candidate_gate_state_for(
               "VULKAN_SIM_RTCORE_ABI_V04_TLAS_BINDING_ENFORCEMENT") ==
           RTCORE_CANDIDATE_GATE_ENABLED;
}

static bool rtcore_v04_instance_blas_reference_prerequisites_enabled()
{
    return rtcore_v04_typed_instance_boundary_seed_enabled() &&
           rtcore_v04_typed_blas_decode_context_bridge_enabled() &&
           rtcore_v04_producer_backed_blas_root_descriptor_enabled() &&
           rtcore_v04_tlas_binding_enforcement_gate_enabled();
}

static bool rtcore_v04_typed_instance_enter_prerequisites_enabled()
{
    return rtcore_v04_producer_backed_instance_blas_reference_enabled() &&
           rtcore_v04_instance_blas_reference_prerequisites_enabled();
}

static bool rtcore_v04_typed_node_child_route_prerequisites_enabled()
{
    return rtcore_v04_typed_node_candidate_kernel_enabled() &&
           rtcore_v04_typed_instance_enter_transition_enabled() &&
           rtcore_v04_typed_instance_enter_prerequisites_enabled();
}

static bool rtcore_v04_typed_stack_push_remainder_prerequisites_enabled()
{
    return rtcore_v04_typed_node_child_route_kernel_enabled() &&
           rtcore_v04_typed_node_child_route_prerequisites_enabled();
}

static bool rtcore_v04_typed_stack_pop_next_prerequisites_enabled()
{
    return rtcore_v04_typed_stack_push_remainder_kernel_enabled() &&
           rtcore_v04_typed_stack_push_remainder_prerequisites_enabled();
}

static bool rtcore_v04_private_frontier_owner_layout_prerequisites_enabled()
{
    return rtcore_v04_typed_stack_pop_next_kernel_enabled() &&
           rtcore_v04_typed_stack_pop_next_prerequisites_enabled();
}

static bool rtcore_v04_request_owner_binding_prerequisites_enabled()
{
    return rtcore_v04_private_frontier_owner_layout_enabled() &&
           rtcore_v04_private_frontier_owner_layout_prerequisites_enabled() &&
           rtcore_replay_admission_enabled() &&
           rtcore_continuation_model_enabled();
}

static bool rtcore_v04_private_frontier_live_init_prerequisites_enabled()
{
    return rtcore_v04_request_owner_binding_enabled() &&
           rtcore_v04_request_owner_binding_prerequisites_enabled() &&
           rtcore_replay_memory_unit_request_offer_enabled();
}

static bool rtcore_memory_unit_response_wait_stats_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_UNIT_RESPONSE_WAIT_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static unsigned rtcore_memory_unit_response_wait_stats_log_limit()
{
    static unsigned limit = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_MEMORY_UNIT_RESPONSE_WAIT_STATS_LOG_LIMIT");
        if (!value || !value[0]) {
            return 256u;
        }
        char *end = NULL;
        unsigned long parsed = strtoul(value, &end, 10);
        if (end == value) {
            return 256u;
        }
        if (parsed > 4096) {
            return 4096u;
        }
        return static_cast<unsigned>(parsed);
    }();
    return limit;
}

static bool rtcore_replay_resource_route_stats_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_RESOURCE_ROUTE_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_data_path_access_stats_log_enabled()
{
    static int enabled = []() {
        const char *value =
            getenv("VULKAN_SIM_RTCORE_REPLAY_DATA_PATH_ACCESS_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_v03_hw_request_state_scoreboard_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_V03_HW_REQUEST_STATE_SCOREBOARD_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_v03_hw_unit_state_wake_service_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_V03_HW_UNIT_STATE_WAKE_SERVICE_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_v03_hw_memory_outstanding_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_V03_HW_MEMORY_OUTSTANDING_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_lane_request_state_capacity_gate_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_LANE_REQUEST_STATE_CAPACITY_GATE");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_lane_request_state_capacity_gate_stats_log_enabled()
{
    static int enabled = []() {
        const char *value = getenv(
            "VULKAN_SIM_RTCORE_REPLAY_LANE_REQUEST_STATE_CAPACITY_GATE_STATS_LOG");
        return value && value[0] && strcmp(value, "0") != 0;
    }();
    return enabled != 0;
}

static bool rtcore_replay_warp_completion_entry_enabled();

static bool rtcore_replay_warp_completion_entry_enabled()
{
    static int enabled = []() {
        return rtcore_replay_env_enabled_or_model_preset(
                   "VULKAN_SIM_RTCORE_REPLAY_WARP_COMPLETION_ENTRY",
                   true)
                   ? 1
                   : 0;
    }();
    return enabled != 0;
}

static const unsigned RTCORE_REPLAY_STATS_LOG_LIMIT_MAX = 4096;

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
    if (parsed > RTCORE_REPLAY_STATS_LOG_LIMIT_MAX) {
        return RTCORE_REPLAY_STATS_LOG_LIMIT_MAX;
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

static unsigned rtcore_replay_overflow_summary_estimate_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_OVERFLOW_SUMMARY_ESTIMATE_LOG_LIMIT", 16);
    return limit;
}

static unsigned rtcore_replay_memory_demand_estimate_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_DEMAND_ESTIMATE_LOG_LIMIT", 16);
    return limit;
}

static unsigned rtcore_replay_memory_latency_policy_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_LATENCY_POLICY_LOG_LIMIT", 16);
    return limit;
}

static unsigned rtcore_replay_memory_latency_blocked_cycle_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_LATENCY_BLOCKED_CYCLE_STATS_LOG_LIMIT",
        16);
    return limit;
}

static unsigned rtcore_replay_memory_wake_latency_gate_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_WAKE_LATENCY_GATE_STATS_LOG_LIMIT",
        64);
    return limit;
}

static unsigned rtcore_replay_memory_contention_gate_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_CONTENTION_GATE_STATS_LOG_LIMIT", 64);
    return limit;
}

static unsigned rtcore_replay_unit_latency_gate_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_UNIT_LATENCY_GATE_STATS_LOG_LIMIT", 64);
    return limit;
}

static unsigned rtcore_replay_model_summary_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MODEL_SUMMARY_STATS_LOG_LIMIT", 16);
    return limit;
}

static unsigned rtcore_replay_resource_route_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_RESOURCE_ROUTE_STATS_LOG_LIMIT", 64);
    return limit;
}

static unsigned rtcore_replay_data_path_access_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_DATA_PATH_ACCESS_STATS_LOG_LIMIT", 16);
    return limit;
}

static unsigned
rtcore_replay_v03_hw_request_state_scoreboard_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_V03_HW_REQUEST_STATE_SCOREBOARD_STATS_LOG_LIMIT",
        64);
    return limit;
}

static unsigned
rtcore_replay_v03_hw_unit_state_wake_service_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_V03_HW_UNIT_STATE_WAKE_SERVICE_STATS_LOG_LIMIT",
        64);
    return limit;
}

static unsigned rtcore_replay_v03_hw_memory_outstanding_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_V03_HW_MEMORY_OUTSTANDING_STATS_LOG_LIMIT",
        64);
    return limit;
}

static unsigned rtcore_replay_lane_request_state_capacity_gate_stats_log_limit()
{
    static unsigned limit = rtcore_replay_service_tick_stats_log_limit_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_LANE_REQUEST_STATE_CAPACITY_GATE_STATS_LOG_LIMIT",
        64);
    return limit;
}

static unsigned rtcore_replay_lane_request_state_capacity_config()
{
    static unsigned capacity = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_LANE_REQUEST_STATE_CAPACITY", 0, 32,
        1048576, true);
    return capacity;
}

static unsigned rtcore_replay_lane_state_init_bandwidth_config()
{
    static unsigned bandwidth = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_LANE_STATE_INIT_BANDWIDTH", 32, 1,
        1024, true);
    return bandwidth == 0 ? 1 : bandwidth;
}

static unsigned rtcore_replay_v03_hw_request_state_bank_count_config()
{
    static unsigned bank_count = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_V03_HW_REQUEST_STATE_BANK_COUNT", 8, 1,
        1024, true);
    return bank_count == 0 ? 1 : bank_count;
}

static unsigned rtcore_replay_warp_completion_entry_capacity_config()
{
    static unsigned capacity = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_WARP_COMPLETION_ENTRY_CAPACITY", 8, 1,
        1024, false);
    return capacity == 0 ? 1 : capacity;
}

static unsigned rtcore_replay_memory_contention_cache_lines_per_cycle_config()
{
    static unsigned lines_per_cycle = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_CONTENTION_CACHE_LINES_PER_CYCLE", 4,
        1, 1024, false);
    return lines_per_cycle == 0 ? 1 : lines_per_cycle;
}

static unsigned rtcore_replay_memory_cache_line_latency_config()
{
    static unsigned latency =
        rtcore_replay_service_tick_stats_log_limit_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_MEMORY_CACHE_LINE_LATENCY", 4);
    return latency;
}

static unsigned rtcore_replay_memory_wait_event_latency_config()
{
    static unsigned latency =
        rtcore_replay_service_tick_stats_log_limit_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_MEMORY_WAIT_EVENT_LATENCY", 32);
    return latency;
}

static unsigned rtcore_replay_node_test_latency_config()
{
    static unsigned latency =
        rtcore_replay_service_tick_stats_log_limit_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_NODE_TEST_LATENCY", 2);
    return latency;
}

static unsigned rtcore_replay_primitive_test_latency_config()
{
    static unsigned latency =
        rtcore_replay_service_tick_stats_log_limit_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_PRIMITIVE_TEST_LATENCY", 4);
    return latency;
}

static unsigned rtcore_replay_stack_latency_cycles_config()
{
    static unsigned latency = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_STACK_LATENCY_CYCLES",
        RTCORE_REPLAY_STACK_LATENCY_CYCLES, 1, 1048576, false);
    return latency == 0 ? 1 : latency;
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

static unsigned rtcore_replay_warp_completion_ingress_budget_config()
{
    return rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_WARP_COMPLETION_INGRESS_BUDGET", 8);
}

static unsigned rtcore_replay_scoreboard_result_handoff_budget_config()
{
    return rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_SCOREBOARD_RESULT_HANDOFF_BUDGET", 1);
}

static rtcore_replay_issue_budget rtcore_replay_issue_budget_config()
{
    static rtcore_replay_issue_budget budget = []() {
        rtcore_replay_issue_budget parsed = {};
        parsed.node_issue_budget = rtcore_replay_issue_budget_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_NODE_ISSUE_BUDGET", 8);
        parsed.primitive_issue_budget = rtcore_replay_issue_budget_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_PRIMITIVE_ISSUE_BUDGET", 4);
        parsed.stack_issue_budget = rtcore_replay_issue_budget_from_env(
            "VULKAN_SIM_RTCORE_REPLAY_STACK_ISSUE_BUDGET", 1);
        parsed.warp_completion_ingress_budget =
            rtcore_replay_warp_completion_ingress_budget_config();
        return parsed;
    }();
    return budget;
}

static unsigned rtcore_replay_memory_wake_budget_config()
{
    static unsigned budget = rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_WAKE_BUDGET", 4);
    return budget;
}

static unsigned rtcore_replay_memory_outstanding_capacity_config()
{
    static unsigned capacity = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_OUTSTANDING_CAPACITY", 32, 1,
        1048576, true);
    return capacity == 0 ? 1 : capacity;
}

static unsigned rtcore_replay_memory_outstanding_alloc_budget_config()
{
    static unsigned budget = rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_OUTSTANDING_ALLOC_BUDGET", 1);
    return budget;
}

static unsigned rtcore_replay_memory_address_gen_budget_config()
{
    static unsigned budget = rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_ADDRESS_GEN_BUDGET", 1);
    return budget;
}

static unsigned rtcore_replay_memory_address_gen_latency_config()
{
    static unsigned latency = rtcore_replay_uint_config_or_model_preset(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_ADDRESS_GEN_LATENCY", 1, 1, 1048576,
        true);
    return latency == 0 ? 1 : latency;
}

static unsigned rtcore_replay_memory_issue_budget_config()
{
    static unsigned budget = rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_MEMORY_ISSUE_BUDGET", 1);
    return budget;
}

static unsigned rtcore_replay_unit_wake_budget_config()
{
    static unsigned budget = rtcore_replay_issue_budget_from_env(
        "VULKAN_SIM_RTCORE_REPLAY_UNIT_WAKE_BUDGET", 1);
    return budget;
}

static bool rtcore_replay_issue_budget_available(
    const rtcore_replay_issue_budget &budget)
{
    return budget.node_issue_budget || budget.primitive_issue_budget ||
           budget.stack_issue_budget ||
           budget.warp_completion_ingress_budget;
}

static void rtcore_update_replay_data_path_max_lane_request_state_entries()
{
    const unsigned entries =
        static_cast<unsigned>(g_rtcore_replay_lane_requests.size());
    if (entries >
        g_rtcore_replay_data_path_access_stats.max_lane_request_state_entries) {
        g_rtcore_replay_data_path_access_stats.max_lane_request_state_entries =
            entries;
    }
}

static void rtcore_record_replay_lane_request_state_identity_read()
{
    g_rtcore_replay_data_path_access_stats.lane_request_state_identity_reads++;
}

static void rtcore_record_replay_lane_request_state_identity_write()
{
    g_rtcore_replay_data_path_access_stats.lane_request_state_identity_writes++;
    rtcore_update_replay_data_path_max_lane_request_state_entries();
}

static void rtcore_record_replay_request_state_read()
{
    g_rtcore_replay_data_path_access_stats.request_state_reads++;
}

static void rtcore_record_replay_request_state_write()
{
    g_rtcore_replay_data_path_access_stats.request_state_writes++;
}

static rtcore_replay_data_path_access_snapshot
rtcore_get_replay_data_path_access_snapshot()
{
    rtcore_replay_data_path_access_snapshot snapshot = {};
    snapshot.lane_request_state_identity_reads =
        g_rtcore_replay_data_path_access_stats.lane_request_state_identity_reads;
    snapshot.lane_request_state_identity_writes =
        g_rtcore_replay_data_path_access_stats.lane_request_state_identity_writes;
    snapshot.request_state_reads =
        g_rtcore_replay_data_path_access_stats.request_state_reads;
    snapshot.request_state_writes =
        g_rtcore_replay_data_path_access_stats.request_state_writes;
    snapshot.lane_request_state_identity_accesses =
        g_rtcore_replay_data_path_access_stats.lane_request_state_identity_reads +
        g_rtcore_replay_data_path_access_stats.lane_request_state_identity_writes;
    snapshot.request_state_accesses =
        g_rtcore_replay_data_path_access_stats.request_state_reads +
        g_rtcore_replay_data_path_access_stats.request_state_writes;
    return snapshot;
}

static unsigned rtcore_replay_data_path_access_delta(unsigned after,
                                                     unsigned before)
{
    return after >= before ? after - before : 0;
}

static const unsigned RTCORE_REPLAY_DATA_PATH_RESOURCE_LANE_REQUEST_STATE_IDENTITY_MASK =
    1u << 0;
static const unsigned RTCORE_REPLAY_DATA_PATH_RESOURCE_REQUEST_STATE_MASK =
    1u << 1;
static const unsigned RTCORE_REPLAY_DATA_PATH_RESOURCE_ALL_MASK =
    RTCORE_REPLAY_DATA_PATH_RESOURCE_LANE_REQUEST_STATE_IDENTITY_MASK |
    RTCORE_REPLAY_DATA_PATH_RESOURCE_REQUEST_STATE_MASK;
static void rtcore_update_replay_unsigned_max(unsigned value,
                                              unsigned *maximum)
{
    if (maximum && value > *maximum) {
        *maximum = value;
    }
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

static unsigned rtcore_unpack_compact_trace_flags(
    const rtcore_compact_trace_event &event)
{
    return event.packed_fields & 0xffu;
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

static rtcore_replay_resource_route rtcore_replay_route_for_event(
    rtcore_compact_trace_event_type event_type)
{
    switch (event_type) {
    case RTCORE_TRACE_NODE_FETCH:
    case RTCORE_TRACE_PRIMITIVE_FETCH:
    case RTCORE_TRACE_MEMORY_WAIT:
        return RTCORE_REPLAY_ROUTE_MEMORY;
    case RTCORE_TRACE_NODE_TEST:
        return RTCORE_REPLAY_ROUTE_NODE;
    case RTCORE_TRACE_PRIMITIVE_TEST:
    case RTCORE_TRACE_HIT_UPDATE:
        return RTCORE_REPLAY_ROUTE_PRIMITIVE;
    case RTCORE_TRACE_STACK_PUSH:
    case RTCORE_TRACE_STACK_POP:
        return RTCORE_REPLAY_ROUTE_STACK;
    case RTCORE_TRACE_COMPLETION:
    case RTCORE_TRACE_OVERFLOW_SUMMARY:
    default:
        return RTCORE_REPLAY_ROUTE_COMPLETION;
    }
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
        return RTCORE_REPLAY_ISSUED_NODE;
    case RTCORE_TRACE_PRIMITIVE_FETCH:
    case RTCORE_TRACE_PRIMITIVE_TEST:
    case RTCORE_TRACE_HIT_UPDATE:
        return RTCORE_REPLAY_ISSUED_PRIMITIVE;
    case RTCORE_TRACE_STACK_PUSH:
    case RTCORE_TRACE_STACK_POP:
        return RTCORE_REPLAY_ISSUED_STACK;
    case RTCORE_TRACE_MEMORY_WAIT:
        return RTCORE_REPLAY_ISSUED_MEMORY;
    case RTCORE_TRACE_COMPLETION:
    case RTCORE_TRACE_OVERFLOW_SUMMARY:
        return RTCORE_REPLAY_COMPLETION_PENDING;
    default:
        return RTCORE_REPLAY_COMPLETED;
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

static const unsigned RTCORE_TRACE_PRIMITIVE_FLAG_OPAQUE_COMMIT = 0x40u;

static unsigned rtcore_trace_primitive_flags(
    rtcore_compact_trace_primitive_kind primitive_kind, bool hit,
    bool deferred, bool opaque_commit = false)
{
    return (static_cast<unsigned>(primitive_kind) & 0x0fu) |
           (hit ? 0x10u : 0u) | (deferred ? 0x20u : 0u) |
           (opaque_commit ? RTCORE_TRACE_PRIMITIVE_FLAG_OPAQUE_COMMIT : 0u);
}

static unsigned rtcore_trace_hit_update_flags(
    rtcore_compact_trace_hit_update_kind hit_update_kind)
{
    return static_cast<unsigned>(hit_update_kind) & 0xffu;
}

static bool rtcore_compact_trace_event_is_semantic_boundary(
    rtcore_compact_trace_event_type event_type, unsigned flags)
{
    if (event_type == RTCORE_TRACE_HIT_UPDATE) {
        return (flags & 0xffu) == RTCORE_TRACE_HIT_UPDATE_KIND_ANY_HIT;
    }
    if (event_type != RTCORE_TRACE_PRIMITIVE_TEST) {
        return false;
    }
    const unsigned primitive_kind = flags & 0x0fu;
    return (primitive_kind ==
                RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_DEFERRED &&
            (flags & 0x20u) != 0) ||
           (primitive_kind == RTCORE_TRACE_PRIMITIVE_KIND_TRIANGLE_TEST &&
            (flags & RTCORE_TRACE_PRIMITIVE_FLAG_OPAQUE_COMMIT) != 0);
}

struct rtcore_bounded_trace_collector {
    bool enabled;
    unsigned lane_id;
    unsigned max_trace_events_per_lane;
    unsigned ordinary_timing_event_count;
    unsigned next_event_seq;
    bool timing_trace_overflowed;
    rtcore_trace_timing_precision_class timing_precision_class;
    unsigned overflow_summary_events;
    rtcore_compact_trace_overflow_summary overflow_summary;
    bool has_overflow_summary_event;
    unsigned overflow_summary_event_index;
    unsigned oracle_anyhit_candidate_count;
    bool oracle_requires_intersection_shader;
    std::vector<rtcore_compact_trace_event> events;
    std::vector<rtcore_boundary_candidate_snapshot> boundary_candidates;

    explicit rtcore_bounded_trace_collector(ptx_thread_info *thread)
        : enabled(rtcore_bounded_trace_collection_enabled()),
          lane_id(thread ? (thread->get_tid().x & 31u) : 0),
          max_trace_events_per_lane(
              rtcore_compact_trace_events_per_lane_config()),
          ordinary_timing_event_count(0), next_event_seq(0),
          timing_trace_overflowed(false),
          timing_precision_class(RTCORE_TRACE_TIMING_PRECISION_EXACT),
          overflow_summary_events(0), overflow_summary(),
          has_overflow_summary_event(false), overflow_summary_event_index(0),
          oracle_anyhit_candidate_count(0),
          oracle_requires_intersection_shader(false)
    {
        if (enabled) {
            if (max_trace_events_per_lane >
                RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR) {
                max_trace_events_per_lane =
                    RTCORE_COMPACT_TRACE_MAX_EVENTS_PER_LANE_WITHOUT_CR;
            }
            events.reserve(max_trace_events_per_lane + 1);
        }
    }

    rtcore_bounded_trace_collector(unsigned test_lane_id,
                                   unsigned test_max_trace_events_per_lane)
        : enabled(true), lane_id(test_lane_id),
          max_trace_events_per_lane(test_max_trace_events_per_lane),
          ordinary_timing_event_count(0), next_event_seq(0),
          timing_trace_overflowed(false),
          timing_precision_class(RTCORE_TRACE_TIMING_PRECISION_EXACT),
          overflow_summary_events(0), overflow_summary(),
          has_overflow_summary_event(false), overflow_summary_event_index(0),
          oracle_anyhit_candidate_count(0),
          oracle_requires_intersection_shader(false)
    {
        events.reserve(max_trace_events_per_lane + 1);
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

    uint16_t allocate_event_seq()
    {
        if (next_event_seq >= RTCORE_COMPACT_TRACE_EVENT_SEQ_CAPACITY) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_COMPACT_TRACE_EVENT_SEQ_EXHAUSTED "
                    "lane_id=%u next_event_seq=%u capacity=%u\n",
                    lane_id, next_event_seq,
                    RTCORE_COMPACT_TRACE_EVENT_SEQ_CAPACITY);
            abort();
        }
        return static_cast<uint16_t>(next_event_seq++);
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

        overflow_summary_event_index = events.size();
        has_overflow_summary_event = true;
        events.push_back(make_overflow_summary_event(allocate_event_seq()));
    }

    unsigned append(rtcore_compact_trace_event_type event_type,
                    rtcore_compact_trace_resource_class resource_class,
                    uint64_t address_or_ref, unsigned bytes, unsigned count,
                    unsigned flags)
    {
        if (!enabled) {
            return UINT_MAX;
        }
        const bool semantic_boundary =
            rtcore_compact_trace_event_is_semantic_boundary(event_type,
                                                            flags);
        if (ordinary_timing_event_count >= max_trace_events_per_lane &&
            !semantic_boundary) {
            record_overflow_event(event_type, resource_class, bytes, count);
            append_or_update_overflow_summary();
            return UINT_MAX;
        }
        const bool semantic_boundary_after_overflow =
            semantic_boundary && has_overflow_summary_event;
        if (semantic_boundary_after_overflow) {
            append_or_update_overflow_summary();
        }

        rtcore_compact_trace_event event = {};
        event.address_or_ref = address_or_ref;
        event.packed_fields =
            rtcore_pack_compact_trace_fields(lane_id, event_type,
                                             resource_class, flags);
        event.packed_count_bytes =
            rtcore_pack_compact_trace_count_bytes(count, bytes);
        if (semantic_boundary_after_overflow &&
            has_overflow_summary_event &&
            overflow_summary_event_index < events.size()) {
            event.event_seq =
                events[overflow_summary_event_index].event_seq;
            events[overflow_summary_event_index].event_seq =
                allocate_event_seq();
            events.insert(events.begin() + overflow_summary_event_index,
                          event);
            overflow_summary_event_index++;
            return event.event_seq;
        }

        event.event_seq = allocate_event_seq();
        events.push_back(event);
        if (!semantic_boundary) {
            ordinary_timing_event_count++;
        }
        return event.event_seq;
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

    unsigned append_primitive_test(uint64_t address_or_ref, bool hit,
                                   unsigned flags)
    {
        return append(RTCORE_TRACE_PRIMITIVE_TEST,
                      RTCORE_TRACE_RESOURCE_PRIMITIVE, address_or_ref, 0, 1,
                      flags | (hit ? 0x10u : 0u));
    }

    unsigned append_hit_update(uint64_t address_or_ref, unsigned hit_count,
                               unsigned flags)
    {
        return append(RTCORE_TRACE_HIT_UPDATE,
                      RTCORE_TRACE_RESOURCE_COMPLETION, address_or_ref, 0,
                      hit_count, flags);
    }

    void record_boundary_candidate(
        unsigned event_seq, unsigned shader_counter, uint64_t hit_data_ref,
        unsigned hit_group_index, unsigned geometry_type,
        unsigned geometry_index, unsigned primitive_index,
        unsigned instance_index, unsigned hit_kind,
        const rtcore::abi_v04::shadow::boundary_values &v04_boundary_values)
    {
        if (event_seq == UINT_MAX) return;
        rtcore_boundary_candidate_snapshot snapshot;
        snapshot.valid = 1;
        snapshot.event_seq = event_seq;
        snapshot.shader_counter = shader_counter;
        snapshot.hit_data_ref = hit_data_ref;
        snapshot.hit_group_index = hit_group_index;
        snapshot.geometry_type = geometry_type;
        snapshot.geometry_index = geometry_index;
        snapshot.primitive_index = primitive_index;
        snapshot.instance_index = instance_index;
        snapshot.hit_kind = hit_kind;
        snapshot.v04_boundary_values = v04_boundary_values;
        boundary_candidates.push_back(snapshot);
    }

    void append_completion_summary(unsigned node_events,
                                   unsigned primitive_events)
    {
        append(RTCORE_TRACE_COMPLETION, RTCORE_TRACE_RESOURCE_COMPLETION, 0, 0,
               node_events + primitive_events, 0);
    }

    void set_oracle_shader_boundary_reason(unsigned anyhit_candidate_count,
                                           bool requires_intersection_shader)
    {
        oracle_anyhit_candidate_count = anyhit_candidate_count;
        oracle_requires_intersection_shader = requires_intersection_shader;
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
        record.oracle_anyhit_candidate_count = oracle_anyhit_candidate_count;
        record.oracle_requires_intersection_shader =
            oracle_requires_intersection_shader;
        if (enabled) {
            record.events = events;
            record.boundary_candidates = boundary_candidates;
        }
        return record;
    }
};

static void rtcore_compact_trace_self_test_require(bool condition,
                                                   const char *reason)
{
    if (condition) return;
    fprintf(stderr,
            "GPGPU-Sim RTCORE_COMPACT_TRACE_BOUNDARY_OVERFLOW_SELF_TEST "
            "failed reason=%s\n",
            reason ? reason : "unknown");
    abort();
}

static void rtcore_compact_trace_boundary_overflow_self_test()
{
    rtcore_bounded_trace_collector collector(7, 2);
    collector.append(RTCORE_TRACE_NODE_TEST, RTCORE_TRACE_RESOURCE_NODE,
                     0x10, 0, 1, 0);
    collector.append(
        RTCORE_TRACE_PRIMITIVE_TEST, RTCORE_TRACE_RESOURCE_PRIMITIVE, 0x20,
        0, 1,
        rtcore_trace_primitive_flags(
            RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_DEFERRED, true, true));
    collector.append(RTCORE_TRACE_STACK_PUSH, RTCORE_TRACE_RESOURCE_STACK,
                     0x30, 0, 1,
                     rtcore_trace_stack_flags(false, false, false));
    collector.append(
        RTCORE_TRACE_HIT_UPDATE, RTCORE_TRACE_RESOURCE_COMPLETION, 0x40, 0,
        1, rtcore_trace_hit_update_flags(
               RTCORE_TRACE_HIT_UPDATE_KIND_ANY_HIT));

    rtcore_compact_trace_self_test_require(
        collector.ordinary_timing_event_count == 2,
        "semantic_boundary_consumed_ordinary_capacity");
    rtcore_compact_trace_self_test_require(
        !collector.timing_trace_overflowed &&
            !collector.has_overflow_summary_event,
        "summary_created_before_ordinary_overflow");

    collector.append(RTCORE_TRACE_PRIMITIVE_FETCH,
                     RTCORE_TRACE_RESOURCE_PRIMITIVE, 0x50, 32, 1, 0);
    collector.append(
        RTCORE_TRACE_PRIMITIVE_TEST, RTCORE_TRACE_RESOURCE_PRIMITIVE, 0x60,
        0, 1,
        rtcore_trace_primitive_flags(
            RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_DEFERRED, true, true));
    collector.append(
        RTCORE_TRACE_HIT_UPDATE, RTCORE_TRACE_RESOURCE_COMPLETION, 0x68, 0,
        1, rtcore_trace_hit_update_flags(
               RTCORE_TRACE_HIT_UPDATE_KIND_ANY_HIT));
    collector.append(
        RTCORE_TRACE_PRIMITIVE_TEST, RTCORE_TRACE_RESOURCE_PRIMITIVE, 0x6c,
        0, 1,
        rtcore_trace_primitive_flags(
            RTCORE_TRACE_PRIMITIVE_KIND_TRIANGLE_TEST, true, false, true));
    collector.append(RTCORE_TRACE_NODE_FETCH, RTCORE_TRACE_RESOURCE_NODE,
                     0x70, 64, 1, 0);

    const rtcore_compact_trace_event_type expected_types[] = {
        RTCORE_TRACE_NODE_TEST,
        RTCORE_TRACE_PRIMITIVE_TEST,
        RTCORE_TRACE_STACK_PUSH,
        RTCORE_TRACE_HIT_UPDATE,
        RTCORE_TRACE_PRIMITIVE_TEST,
        RTCORE_TRACE_HIT_UPDATE,
        RTCORE_TRACE_PRIMITIVE_TEST,
        RTCORE_TRACE_OVERFLOW_SUMMARY,
    };
    const unsigned expected_event_count =
        sizeof(expected_types) / sizeof(expected_types[0]);
    rtcore_compact_trace_self_test_require(
        collector.events.size() == expected_event_count,
        "unexpected_retained_event_count");
    for (unsigned i = 0; i < expected_event_count; ++i) {
        rtcore_compact_trace_self_test_require(
            rtcore_unpack_compact_trace_event_type(collector.events[i]) ==
                expected_types[i],
            "retained_event_order_mismatch");
        rtcore_compact_trace_self_test_require(
            collector.events[i].event_seq == i,
            "event_seq_not_unique_monotonic");
    }
    rtcore_compact_trace_self_test_require(
        collector.timing_trace_overflowed &&
            collector.overflow_summary_events == 1 &&
            collector.overflow_summary_event_index + 1 ==
                collector.events.size(),
        "overflow_summary_not_unique_terminal");
    rtcore_compact_trace_self_test_require(
        collector.overflow_summary.overflow_primitive_fetch_count == 1 &&
            collector.overflow_summary.overflow_node_fetch_count == 1,
        "overflow_counters_mismatch");
    rtcore_compact_trace_self_test_require(
        (rtcore_unpack_compact_trace_flags(
             collector.events[expected_event_count - 2]) &
         RTCORE_TRACE_PRIMITIVE_FLAG_OPAQUE_COMMIT) != 0,
        "opaque_commit_semantic_event_not_retained");

    printf("GPGPU-Sim RTCORE_COMPACT_TRACE_BOUNDARY_OVERFLOW_SELF_TEST ok "
           "ordinary_timing_event_count=%u retained_event_count=%u "
           "overflow_summary_events=%u\n",
           collector.ordinary_timing_event_count,
           (unsigned)collector.events.size(),
           collector.overflow_summary_events);
    fflush(stdout);
}

static void rtcore_maybe_run_compact_trace_boundary_overflow_self_test()
{
    static bool ran = false;
    if (ran) return;
    const char *enabled =
        getenv("VULKAN_SIM_RTCORE_TEST_COMPACT_TRACE_BOUNDARY_OVERFLOW");
    if (!enabled || enabled[0] == '\0' || strcmp(enabled, "0") == 0) return;
    ran = true;
    rtcore_compact_trace_boundary_overflow_self_test();
}

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
    stored.v04_live_handoff_memory =
        thread ? thread->get_global_memory() : NULL;
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
    request.context_profile_valid = record.context_profile_valid;
    request.context_layout_version = record.context_layout_version;
    request.context_valid_flags = record.context_valid_flags;
    request.pipeline_profile_id = record.pipeline_profile_id;
    request.bvh_format_profile_id = record.bvh_format_profile_id;
    request.next_event_index = 0;
    request.event_count = record.event_count;
    request.oracle_anyhit_candidate_count =
        record.oracle_anyhit_candidate_count;
    request.oracle_requires_intersection_shader =
        record.oracle_requires_intersection_shader;
    request.ray_sbt_inputs_valid = record.ray_sbt_inputs_valid;
    request.sbt_record_offset = record.sbt_record_offset;
    request.sbt_record_stride = record.sbt_record_stride;
    request.miss_index = record.miss_index;
    request.ray_flags = record.ray_flags;
    request.cull_mask = record.cull_mask;
    request.hit_geometry_summary_valid = record.hit_geometry_summary_valid;
    request.closest_hit_kind = record.closest_hit_kind;
    request.closest_hit_geometry_type = record.closest_hit_geometry_type;
    request.closest_hit_geometry_index = record.closest_hit_geometry_index;
    request.closest_hit_primitive_index = record.closest_hit_primitive_index;
    request.closest_hit_instance_index = record.closest_hit_instance_index;
    request.instance_sbt_contribution_valid =
        record.instance_sbt_contribution_valid;
    request.instance_sbt_contribution = record.instance_sbt_contribution;
    request.v04_shadow_boundary_enabled =
        record.v04_shadow_boundary_enabled;
    request.v04_tlas_binding_enforcement_enabled =
        record.v04_tlas_binding_enforcement_enabled;
    request.v04_tlas_binding = record.v04_tlas_binding;
    request.v04_shadow_trace_input_valid =
        record.v04_shadow_trace_input_valid;
    request.v04_shadow_trace_input_words =
        record.v04_shadow_trace_input_words;
    request.handoff_window_base = record.handoff_window_base;
    request.v04_live_handoff_memory = record.v04_live_handoff_memory;
    request.v04_request_owner_binding_valid = false;
    request.v04_request_owner_binding =
        rtcore::v04::request_owner::lane_binding_v0();
    request.v04_private_frontier_owner =
        rtcore::v04::private_frontier::owner_binding_v0();
    request.v04_private_frontier_init_pending = false;
    request.v04_replay_committed_boundary_values =
        rtcore::abi_v04::shadow::boundary_values();
    request.continuation_boundary_pending = false;
    request.continuation_depth = 0;
    request.continuation_segment_event_count = 0;
    request.timing_trace_overflowed = record.timing_trace_overflowed;
    request.timing_precision_class = record.timing_precision_class;
    request.overflow_summary_events = record.overflow_summary_events;
    request.overflow_summary = record.overflow_summary;
    request.state = RTCORE_REPLAY_COMPLETED;
    request.events = record.events;
    request.boundary_candidates = record.boundary_candidates;

    for (unsigned i = 0; i < request.events.size(); ++i) {
        rtcore_compact_trace_event_type event_type =
            rtcore_unpack_compact_trace_event_type(request.events[i]);
        switch (rtcore_classify_replay_state(event_type)) {
        case RTCORE_REPLAY_ISSUED_NODE:
            request.node_event_count++;
            break;
        case RTCORE_REPLAY_ISSUED_PRIMITIVE:
            request.primitive_event_count++;
            break;
        case RTCORE_REPLAY_ISSUED_STACK:
            request.stack_event_count++;
            break;
        case RTCORE_REPLAY_ISSUED_MEMORY:
            request.memory_event_count++;
            break;
        case RTCORE_REPLAY_COMPLETION_PENDING:
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
        request.state = RTCORE_REPLAY_ADMITTED;
    }
    return request;
}

static void rtcore_maybe_log_replay_resource_route_stats(unsigned owner_hw_sid)
{
    if (!rtcore_replay_resource_route_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_resource_route_stats_logs_emitted >=
        rtcore_replay_resource_route_stats_log_limit()) {
        return;
    }
    g_rtcore_replay_resource_route_stats_logs_emitted++;

    printf("GPGPU-Sim RTCORE_REPLAY_RESOURCE_ROUTE_STATS "
           "owner_hw_sid=%u lane_requests=%u total_trace_events=%u "
           "memory_routed_events=%u node_routed_events=%u "
           "primitive_routed_events=%u stack_routed_events=%u "
           "completion_routed_events=%u "
           "node_fetch_events=%u primitive_fetch_events=%u "
           "node_test_events=%u primitive_test_events=%u "
           "hit_update_events=%u stack_events=%u completion_events=%u "
           "fetch_events_routed_to_memory=%u "
           "fetch_events_routed_to_compute=%u "
           "test_events_routed_to_compute=%u "
           "test_events_routed_to_memory=%u "
           "hit_update_events_folded_into_primitive=%u\n",
           owner_hw_sid, g_rtcore_replay_resource_route_stats.lane_requests,
           g_rtcore_replay_resource_route_stats.total_trace_events,
           g_rtcore_replay_resource_route_stats.memory_routed_events,
           g_rtcore_replay_resource_route_stats.node_routed_events,
           g_rtcore_replay_resource_route_stats.primitive_routed_events,
           g_rtcore_replay_resource_route_stats.stack_routed_events,
           g_rtcore_replay_resource_route_stats.completion_routed_events,
           g_rtcore_replay_resource_route_stats.node_fetch_events,
           g_rtcore_replay_resource_route_stats.primitive_fetch_events,
           g_rtcore_replay_resource_route_stats.node_test_events,
           g_rtcore_replay_resource_route_stats.primitive_test_events,
           g_rtcore_replay_resource_route_stats.hit_update_events,
           g_rtcore_replay_resource_route_stats.stack_events,
           g_rtcore_replay_resource_route_stats.completion_events,
           g_rtcore_replay_resource_route_stats.fetch_events_routed_to_memory,
           g_rtcore_replay_resource_route_stats.fetch_events_routed_to_compute,
           g_rtcore_replay_resource_route_stats.test_events_routed_to_compute,
           g_rtcore_replay_resource_route_stats.test_events_routed_to_memory,
           g_rtcore_replay_resource_route_stats
               .hit_update_events_folded_into_primitive);
    fflush(stdout);
}

static void rtcore_record_replay_resource_route_stats(
    const rtcore_replay_lane_request &request)
{
    if (!rtcore_replay_resource_route_stats_log_enabled() || !request.valid) {
        return;
    }

    g_rtcore_replay_resource_route_stats.lane_requests++;
    for (unsigned i = 0; i < request.events.size(); ++i) {
        const rtcore_compact_trace_event_type event_type =
            rtcore_unpack_compact_trace_event_type(request.events[i]);
        const rtcore_replay_resource_route route =
            rtcore_replay_route_for_event(event_type);

        g_rtcore_replay_resource_route_stats.total_trace_events++;
        switch (route) {
        case RTCORE_REPLAY_ROUTE_MEMORY:
            g_rtcore_replay_resource_route_stats.memory_routed_events++;
            break;
        case RTCORE_REPLAY_ROUTE_NODE:
            g_rtcore_replay_resource_route_stats.node_routed_events++;
            break;
        case RTCORE_REPLAY_ROUTE_PRIMITIVE:
            g_rtcore_replay_resource_route_stats.primitive_routed_events++;
            break;
        case RTCORE_REPLAY_ROUTE_STACK:
            g_rtcore_replay_resource_route_stats.stack_routed_events++;
            break;
        case RTCORE_REPLAY_ROUTE_COMPLETION:
        default:
            g_rtcore_replay_resource_route_stats.completion_routed_events++;
            break;
        }

        switch (event_type) {
        case RTCORE_TRACE_NODE_FETCH:
            g_rtcore_replay_resource_route_stats.node_fetch_events++;
            if (route == RTCORE_REPLAY_ROUTE_MEMORY) {
                g_rtcore_replay_resource_route_stats
                    .fetch_events_routed_to_memory++;
            } else if (route == RTCORE_REPLAY_ROUTE_NODE ||
                       route == RTCORE_REPLAY_ROUTE_PRIMITIVE) {
                g_rtcore_replay_resource_route_stats
                    .fetch_events_routed_to_compute++;
            }
            break;
        case RTCORE_TRACE_PRIMITIVE_FETCH:
            g_rtcore_replay_resource_route_stats.primitive_fetch_events++;
            if (route == RTCORE_REPLAY_ROUTE_MEMORY) {
                g_rtcore_replay_resource_route_stats
                    .fetch_events_routed_to_memory++;
            } else if (route == RTCORE_REPLAY_ROUTE_NODE ||
                       route == RTCORE_REPLAY_ROUTE_PRIMITIVE) {
                g_rtcore_replay_resource_route_stats
                    .fetch_events_routed_to_compute++;
            }
            break;
        case RTCORE_TRACE_NODE_TEST:
            g_rtcore_replay_resource_route_stats.node_test_events++;
            if (route == RTCORE_REPLAY_ROUTE_NODE ||
                route == RTCORE_REPLAY_ROUTE_PRIMITIVE) {
                g_rtcore_replay_resource_route_stats
                    .test_events_routed_to_compute++;
            } else if (route == RTCORE_REPLAY_ROUTE_MEMORY) {
                g_rtcore_replay_resource_route_stats
                    .test_events_routed_to_memory++;
            }
            break;
        case RTCORE_TRACE_PRIMITIVE_TEST:
            g_rtcore_replay_resource_route_stats.primitive_test_events++;
            if (route == RTCORE_REPLAY_ROUTE_NODE ||
                route == RTCORE_REPLAY_ROUTE_PRIMITIVE) {
                g_rtcore_replay_resource_route_stats
                    .test_events_routed_to_compute++;
            } else if (route == RTCORE_REPLAY_ROUTE_MEMORY) {
                g_rtcore_replay_resource_route_stats
                    .test_events_routed_to_memory++;
            }
            break;
        case RTCORE_TRACE_HIT_UPDATE:
            g_rtcore_replay_resource_route_stats.hit_update_events++;
            if (route == RTCORE_REPLAY_ROUTE_PRIMITIVE) {
                g_rtcore_replay_resource_route_stats
                    .hit_update_events_folded_into_primitive++;
            }
            break;
        case RTCORE_TRACE_STACK_PUSH:
        case RTCORE_TRACE_STACK_POP:
            g_rtcore_replay_resource_route_stats.stack_events++;
            break;
        case RTCORE_TRACE_COMPLETION:
            g_rtcore_replay_resource_route_stats.completion_events++;
            break;
        default:
            break;
        }
    }

    rtcore_maybe_log_replay_resource_route_stats(request.owner_hw_sid);
}

static bool rtcore_replay_lane_request_state_capacity_consumes_entry(
    const rtcore_replay_lane_request &request)
{
    return request.valid && request.state != RTCORE_REPLAY_COMPLETED;
}

static unsigned rtcore_count_replay_lane_request_state_occupied_entries_for_owner(
    unsigned owner_hw_sid)
{
    unsigned occupancy = 0;
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        rtcore_record_replay_lane_request_state_identity_read();
        const rtcore_replay_lane_request &request = it->second;
        if (request.owner_hw_sid == owner_hw_sid &&
            rtcore_replay_lane_request_state_capacity_consumes_entry(request)) {
            occupancy++;
        }
    }
    return occupancy;
}

static void rtcore_update_replay_lane_request_state_capacity_max(
    unsigned occupancy)
{
    if (occupancy >
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_lane_request_state_occupancy) {
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_lane_request_state_occupancy = occupancy;
    }
}

static unsigned
rtcore_count_replay_lane_request_state_capacity_pending_admissions_for_owner(
    unsigned owner_hw_sid)
{
    unsigned pending_admissions = 0;
    for (std::deque<rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_request_state_capacity_pending_admissions.begin();
         it !=
         g_rtcore_replay_lane_request_state_capacity_pending_admissions.end();
         ++it) {
        if (it->owner_hw_sid == owner_hw_sid) {
            pending_admissions++;
        }
    }
    return pending_admissions;
}

static void rtcore_update_replay_lane_request_state_capacity_pending_max(
    unsigned pending_admissions)
{
    if (pending_admissions >
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_pending_admissions) {
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_pending_admissions = pending_admissions;
    }
}

static void rtcore_update_replay_lane_state_init_bandwidth_used_max(
    unsigned used)
{
    if (used >
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_lane_state_init_bandwidth_used_per_cycle) {
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_lane_state_init_bandwidth_used_per_cycle = used;
    }
}

static bool rtcore_replay_lane_state_init_bandwidth_try_consume(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    rtcore_replay_lane_state_init_bandwidth_owner_cycle &owner_state =
        g_rtcore_replay_lane_state_init_bandwidth_by_owner[owner_hw_sid];
    if (!owner_state.valid || owner_state.service_cycle != service_cycle) {
        owner_state.valid = true;
        owner_state.service_cycle = service_cycle;
        owner_state.used = 0;
    }

    const unsigned bandwidth = rtcore_replay_lane_state_init_bandwidth_config();
    if (owner_state.used >= bandwidth) {
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .last_lane_state_init_bandwidth_used = owner_state.used;
        rtcore_update_replay_lane_state_init_bandwidth_used_max(
            owner_state.used);
        return false;
    }

    owner_state.used++;
    g_rtcore_replay_lane_request_state_capacity_gate_stats
        .last_lane_state_init_bandwidth_used = owner_state.used;
    rtcore_update_replay_lane_state_init_bandwidth_used_max(owner_state.used);
    return true;
}

static void rtcore_queue_replay_lane_request_state_capacity_pending_admission(
    const rtcore_replay_lane_request &request)
{
    g_rtcore_replay_lane_request_state_capacity_pending_admissions.push_back(
        request);
    const unsigned pending_admissions =
        rtcore_count_replay_lane_request_state_capacity_pending_admissions_for_owner(
            request.owner_hw_sid);
    g_rtcore_replay_lane_request_state_capacity_gate_stats
        .last_pending_admissions = pending_admissions;
    rtcore_update_replay_lane_request_state_capacity_pending_max(
        pending_admissions);
}

static void rtcore_maybe_log_replay_lane_request_state_capacity_gate_stats(
    const rtcore_replay_lane_request &request, unsigned occupancy,
    bool capacity_blocked, bool admitted, bool released)
{
    if (!rtcore_replay_lane_request_state_capacity_gate_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_lane_request_state_capacity_gate_stats_logs_emitted >=
        rtcore_replay_lane_request_state_capacity_gate_stats_log_limit()) {
        return;
    }
    g_rtcore_replay_lane_request_state_capacity_gate_stats_logs_emitted++;
    const unsigned pending_admissions =
        rtcore_count_replay_lane_request_state_capacity_pending_admissions_for_owner(
            request.owner_hw_sid);
    g_rtcore_replay_lane_request_state_capacity_gate_stats
        .last_pending_admissions = pending_admissions;
    rtcore_update_replay_lane_request_state_capacity_pending_max(
        pending_admissions);

    printf("GPGPU-Sim RTCORE_REPLAY_LANE_REQUEST_STATE_CAPACITY_GATE "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u "
           "has_warp_metadata=%u warp_uid=%u warp_id=%u "
           "active_mask=0x%08x static_inst_uid=%u gate_enabled=%u "
           "capacity=%u lane_state_init_bandwidth=%u occupancy=%u "
           "capacity_blocked=%u init_bandwidth_blocked=%u "
           "admitted=%u released=%u "
           "evaluations=%u admitted_count=%u "
           "blocked_count=%u released_count=%u "
           "lane_state_init_bandwidth_blocked_count=%u "
           "lane_state_init_bandwidth_used=%u "
           "lane_state_init_bandwidth_max_used_per_cycle=%u "
           "max_lane_request_state_occupancy=%u pending_count=%u "
           "max_pending_admissions=%u\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.has_warp_metadata ? 1u : 0u, request.warp_uid,
           request.warp_id, request.active_mask, request.static_inst_uid,
           rtcore_replay_lane_request_state_capacity_gate_enabled() ? 1u : 0u,
           rtcore_replay_lane_request_state_capacity_config(),
           rtcore_replay_lane_state_init_bandwidth_config(), occupancy,
           capacity_blocked ? 1u : 0u,
           g_rtcore_replay_lane_request_state_capacity_gate_stats
               .last_init_bandwidth_blocked,
           admitted ? 1u : 0u,
           released ? 1u : 0u,
           g_rtcore_replay_lane_request_state_capacity_gate_stats.evaluations,
           g_rtcore_replay_lane_request_state_capacity_gate_stats.admitted_count,
           g_rtcore_replay_lane_request_state_capacity_gate_stats.blocked_count,
           g_rtcore_replay_lane_request_state_capacity_gate_stats.released_count,
           g_rtcore_replay_lane_request_state_capacity_gate_stats
               .lane_state_init_bandwidth_blocked_count,
           g_rtcore_replay_lane_request_state_capacity_gate_stats
               .last_lane_state_init_bandwidth_used,
           g_rtcore_replay_lane_request_state_capacity_gate_stats
               .max_lane_state_init_bandwidth_used_per_cycle,
           g_rtcore_replay_lane_request_state_capacity_gate_stats
               .max_lane_request_state_occupancy,
           pending_admissions,
           g_rtcore_replay_lane_request_state_capacity_gate_stats
               .max_pending_admissions);
    fflush(stdout);
}

static void rtcore_record_replay_lane_request_state_capacity_admitted(
    const rtcore_replay_lane_request &request, unsigned occupancy)
{
    g_rtcore_replay_lane_request_state_capacity_gate_stats.evaluations++;
    g_rtcore_replay_lane_request_state_capacity_gate_stats.admitted_count++;
    g_rtcore_replay_lane_request_state_capacity_gate_stats.last_occupancy =
        occupancy;
    g_rtcore_replay_lane_request_state_capacity_gate_stats.last_capacity_blocked =
        0u;
    g_rtcore_replay_lane_request_state_capacity_gate_stats
        .last_init_bandwidth_blocked = 0u;
    g_rtcore_replay_lane_request_state_capacity_gate_stats.last_admitted = 1u;
    g_rtcore_replay_lane_request_state_capacity_gate_stats.last_released = 0u;
    rtcore_update_replay_lane_request_state_capacity_max(occupancy);
    rtcore_maybe_log_replay_lane_request_state_capacity_gate_stats(
        request, occupancy, false, true, false);
}

static void rtcore_refresh_replay_lane_request_ready_bits(
    rtcore_replay_lane_request *request);

static unsigned rtcore_continuation_count_lanes(unsigned mask);

static bool rtcore_mark_resident_warp_continuation_wakeup(
    const rtcore_continuation_return_packet &packet,
    unsigned long long service_cycle);

static bool rtcore_publish_continuation_return_packet(
    rtcore_continuation_warp_boundary_state *state,
    unsigned long long service_cycle);

extern "C" void rtcore_enqueue_memory_unit_handoff_window_request(
    unsigned owner_hw_sid, unsigned rt_request_id, unsigned lane_id,
    unsigned memory_op_seq, unsigned access_kind,
    unsigned long long byte_address, unsigned chunk_count, bool is_write,
    unsigned long long issue_cycle);

static bool rtcore_maybe_block_replay_lane_request_state_capacity_admission(
    const rtcore_replay_lane_request &request,
    unsigned long long service_cycle = 0)
{
    if (!request.valid) {
        return false;
    }

    const bool capacity_gate_enabled =
        rtcore_replay_lane_request_state_capacity_gate_enabled();
    const unsigned capacity = rtcore_replay_lane_request_state_capacity_config();
    const bool consumes_entry =
        rtcore_replay_lane_request_state_capacity_consumes_entry(request);
    const unsigned current_occupancy =
        rtcore_count_replay_lane_request_state_occupied_entries_for_owner(
            request.owner_hw_sid);
    const bool capacity_blocked =
        capacity_gate_enabled && capacity != 0 && consumes_entry &&
        current_occupancy >= capacity;
    const unsigned resulting_occupancy =
        capacity_blocked ? current_occupancy
                         : current_occupancy + (consumes_entry ? 1u : 0u);

    if (capacity_blocked) {
        g_rtcore_replay_lane_request_state_capacity_gate_stats.evaluations++;
        g_rtcore_replay_lane_request_state_capacity_gate_stats.last_occupancy =
            resulting_occupancy;
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .last_capacity_blocked = 1u;
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .last_init_bandwidth_blocked = 0u;
        g_rtcore_replay_lane_request_state_capacity_gate_stats.last_admitted = 0u;
        g_rtcore_replay_lane_request_state_capacity_gate_stats.last_released = 0u;
        rtcore_update_replay_lane_request_state_capacity_max(resulting_occupancy);
        g_rtcore_replay_lane_request_state_capacity_gate_stats.blocked_count++;
        rtcore_queue_replay_lane_request_state_capacity_pending_admission(request);
        rtcore_maybe_log_replay_lane_request_state_capacity_gate_stats(
            request, resulting_occupancy, true, false, false);
        return true;
    }

    if (consumes_entry &&
        !rtcore_replay_lane_state_init_bandwidth_try_consume(
            request.owner_hw_sid, service_cycle)) {
        g_rtcore_replay_lane_request_state_capacity_gate_stats.evaluations++;
        g_rtcore_replay_lane_request_state_capacity_gate_stats.last_occupancy =
            current_occupancy;
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .last_capacity_blocked = 0u;
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .last_init_bandwidth_blocked = 1u;
        g_rtcore_replay_lane_request_state_capacity_gate_stats.last_admitted = 0u;
        g_rtcore_replay_lane_request_state_capacity_gate_stats.last_released = 0u;
        rtcore_update_replay_lane_request_state_capacity_max(current_occupancy);
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .lane_state_init_bandwidth_blocked_count++;
        rtcore_queue_replay_lane_request_state_capacity_pending_admission(request);
        rtcore_maybe_log_replay_lane_request_state_capacity_gate_stats(
            request, current_occupancy, false, false, false);
        return true;
    }

    if (capacity_gate_enabled && capacity != 0) {
        rtcore_record_replay_lane_request_state_capacity_admitted(
            request, resulting_occupancy);
    } else {
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .last_init_bandwidth_blocked = 0u;
    }
    return false;
}

static bool rtcore_try_drain_replay_lane_request_state_capacity_pending_admissions(
    unsigned owner_hw_sid, unsigned long long service_cycle = 0);

static void rtcore_record_replay_lane_request_state_capacity_release(
    const rtcore_replay_lane_request &request, unsigned long long service_cycle = 0)
{
    if (!request.valid) {
        return;
    }

    if (rtcore_replay_lane_request_state_capacity_gate_enabled()) {
        const unsigned capacity = rtcore_replay_lane_request_state_capacity_config();
        if (capacity != 0) {
            const unsigned occupancy =
                rtcore_count_replay_lane_request_state_occupied_entries_for_owner(
                    request.owner_hw_sid);
            g_rtcore_replay_lane_request_state_capacity_gate_stats.evaluations++;
            g_rtcore_replay_lane_request_state_capacity_gate_stats.released_count++;
            g_rtcore_replay_lane_request_state_capacity_gate_stats.last_occupancy =
                occupancy;
            g_rtcore_replay_lane_request_state_capacity_gate_stats
                .last_capacity_blocked = 0u;
            g_rtcore_replay_lane_request_state_capacity_gate_stats
                .last_init_bandwidth_blocked = 0u;
            g_rtcore_replay_lane_request_state_capacity_gate_stats.last_admitted = 0u;
            g_rtcore_replay_lane_request_state_capacity_gate_stats.last_released = 1u;
            rtcore_update_replay_lane_request_state_capacity_max(occupancy);
            rtcore_maybe_log_replay_lane_request_state_capacity_gate_stats(
                request, occupancy, false, false, true);
        }
    }

    (void)rtcore_try_drain_replay_lane_request_state_capacity_pending_admissions(
        request.owner_hw_sid, service_cycle);
}

static bool rtcore_continuation_request_has_warp_metadata(
    const rtcore_replay_lane_request &request)
{
    return request.valid && request.has_warp_metadata &&
           request.active_mask != 0 && request.lane_id < 32;
}

static unsigned rtcore_continuation_lane_mask(unsigned lane_id)
{
    return lane_id < 32 ? (1u << lane_id) : 0u;
}

static rtcore_replay_warp_completion_entry_key
rtcore_make_continuation_warp_key(const rtcore_replay_lane_request &request)
{
    rtcore_replay_warp_completion_entry_key key = {};
    key.owner_hw_sid = request.owner_hw_sid;
    key.warp_uid = request.warp_uid;
    key.warp_id = request.warp_id;
    key.active_mask = request.active_mask;
    return key;
}

static bool rtcore_continuation_request_matches_boundary_key(
    const rtcore_replay_lane_request &request,
    const rtcore_continuation_warp_boundary_state &state)
{
    return rtcore_continuation_request_has_warp_metadata(request) &&
           request.owner_hw_sid == state.owner_hw_sid &&
           request.warp_uid == state.warp_uid &&
           request.warp_id == state.warp_id &&
           request.active_mask == state.active_mask;
}

static void
rtcore_seed_continuation_boundary_state_from_completed_lanes(
    rtcore_continuation_warp_boundary_state *state)
{
    if (!state || !state->valid) {
        return;
    }

    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (!rtcore_continuation_request_matches_boundary_key(request,
                                                              *state)) {
            continue;
        }
        const unsigned lane_mask =
            rtcore_continuation_lane_mask(request.lane_id);
        if ((state->active_mask & lane_mask) == 0) {
            continue;
        }
        if (request.state == RTCORE_REPLAY_COMPLETED ||
            request.state == RTCORE_REPLAY_FINAL_WAIT_RETIRE) {
            state->boundary_reached_mask |= lane_mask;
            state->terminal_mask |= lane_mask;
            state->reason_final_mask |= lane_mask;
        }
    }
}

static void rtcore_note_terminal_continuation_boundary(
    const rtcore_replay_lane_request &request,
    unsigned long long service_cycle)
{
    if (!rtcore_continuation_model_enabled() ||
        !rtcore_continuation_request_has_warp_metadata(request)) {
        return;
    }

    const rtcore_replay_warp_completion_entry_key key =
        rtcore_make_continuation_warp_key(request);
    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_continuation_warp_boundary_state>::iterator it =
        g_rtcore_continuation_warp_boundary_states.find(key);
    if (it == g_rtcore_continuation_warp_boundary_states.end()) {
        return;
    }

    rtcore_continuation_warp_boundary_state &state = it->second;
    if (!state.valid || state.packet_published) {
        return;
    }
    const unsigned lane_mask =
        rtcore_continuation_lane_mask(request.lane_id);
    if ((state.active_mask & lane_mask) == 0) {
        return;
    }

    state.boundary_reached_mask |= lane_mask;
    state.terminal_mask |= lane_mask;
    state.reason_final_mask |= lane_mask;
    if ((state.boundary_reached_mask & state.active_mask) ==
        state.active_mask) {
        rtcore_publish_continuation_return_packet(&state, service_cycle);
    }
}

static void rtcore_mark_replay_request_completed(
    rtcore_replay_lane_request *request, unsigned long long service_cycle = 0)
{
    if (!request) {
        return;
    }

    const bool consumed_entry =
        rtcore_replay_lane_request_state_capacity_consumes_entry(*request);
    const bool hold_for_explicit_lifecycle_release =
        rtcore_continuation_model_enabled() &&
        rtcore_continuation_request_has_warp_metadata(*request);
    request->state = hold_for_explicit_lifecycle_release
                         ? RTCORE_REPLAY_FINAL_WAIT_RETIRE
                         : RTCORE_REPLAY_COMPLETED;
    rtcore_record_replay_request_state_write();
    rtcore_refresh_replay_lane_request_ready_bits(request);
    rtcore_note_terminal_continuation_boundary(*request, service_cycle);
    if (consumed_entry && !hold_for_explicit_lifecycle_release) {
        rtcore_record_replay_lane_request_state_capacity_release(*request,
                                                            service_cycle);
    }
}

static bool rtcore_replay_request_ready_for_state(
    const rtcore_replay_lane_request &request,
    rtcore_replay_lane_request_state unit_state);

static unsigned rtcore_count_replay_ready_unit_requests_for_owner(
    rtcore_replay_lane_request_state state, unsigned owner_hw_sid,
    unsigned excluded_thread_uid = 0, bool has_excluded_thread_uid = false)
{
    unsigned count = 0;
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (has_excluded_thread_uid &&
            request.thread_uid == excluded_thread_uid) {
            continue;
        }
        if (request.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        rtcore_record_replay_lane_request_state_identity_read();
        rtcore_record_replay_request_state_read();
        if (rtcore_replay_request_ready_for_state(request, state)) {
            count++;
        }
    }
    return count;
}

static bool rtcore_replay_request_state_has_memory_outstanding_work(
    const rtcore_replay_lane_request &request);

static unsigned rtcore_replay_request_state_bank_for_request(
    const rtcore_replay_lane_request &request)
{
    const unsigned bank_count =
        rtcore_replay_v03_hw_request_state_bank_count_config();
    return bank_count == 0 ? 0 : request.ready_order % bank_count;
}

static void rtcore_refresh_replay_lane_request_ready_bits(
    rtcore_replay_lane_request *request)
{
    if (!request) {
        return;
    }

    request->ready_node_bit = false;
    request->ready_primitive_bit = false;
    request->ready_stack_bit = false;
    request->ready_memory_bit = false;
    request->ready_result_bit = false;

    if (!request->valid || request->v04_private_frontier_init_pending ||
        request->continuation_boundary_pending ||
        request->unit_latency_gate_pending) {
        return;
    }
    if (request->memory_address_gen_latency_gate_pending) {
        request->ready_memory_bit =
            request->state == RTCORE_REPLAY_ISSUED_MEMORY;
        return;
    }

    switch (request->state) {
    case RTCORE_REPLAY_ISSUED_NODE:
        request->ready_node_bit = true;
        break;
    case RTCORE_REPLAY_ISSUED_PRIMITIVE:
        request->ready_primitive_bit = true;
        break;
    case RTCORE_REPLAY_ISSUED_STACK:
        request->ready_stack_bit = true;
        break;
    case RTCORE_REPLAY_ISSUED_MEMORY:
        request->ready_memory_bit =
            !rtcore_replay_request_state_has_memory_outstanding_work(*request);
        break;
    case RTCORE_REPLAY_COMPLETION_PENDING:
        request->ready_result_bit = true;
        break;
    default:
        break;
    }
}

static bool rtcore_replay_request_ready_for_state(
    const rtcore_replay_lane_request &request,
    rtcore_replay_lane_request_state unit_state)
{
    if (!request.valid || request.continuation_boundary_pending ||
        request.state != unit_state ||
        request.unit_latency_gate_pending) {
        return false;
    }
    if (request.memory_address_gen_latency_gate_pending &&
        unit_state != RTCORE_REPLAY_ISSUED_MEMORY) {
        return false;
    }

    switch (unit_state) {
    case RTCORE_REPLAY_ISSUED_NODE:
        return request.ready_node_bit;
    case RTCORE_REPLAY_ISSUED_PRIMITIVE:
        return request.ready_primitive_bit;
    case RTCORE_REPLAY_ISSUED_STACK:
        return request.ready_stack_bit;
    case RTCORE_REPLAY_ISSUED_MEMORY:
        return request.ready_memory_bit;
    case RTCORE_REPLAY_COMPLETION_PENDING:
        return request.ready_result_bit;
    default:
        return false;
    }
}

static unsigned long long rtcore_replay_v03_hw_ready_bank_cursor_key(
    unsigned owner_hw_sid, rtcore_replay_lane_request_state unit_state)
{
    return (static_cast<unsigned long long>(owner_hw_sid) << 32) |
           static_cast<unsigned long long>(unit_state);
}

static unsigned long long rtcore_replay_v03_hw_memory_wake_bank_cursor_key(
    unsigned owner_hw_sid)
{
    const unsigned memory_wake_cursor_id = 64;
    return (static_cast<unsigned long long>(owner_hw_sid) << 32) |
           static_cast<unsigned long long>(memory_wake_cursor_id);
}

struct rtcore_replay_v03_hw_banked_ready_candidate {
    bool valid;
    unsigned thread_uid;
    unsigned ready_order;

    rtcore_replay_v03_hw_banked_ready_candidate()
        : valid(false), thread_uid(0), ready_order(0)
    {
    }
};

static bool rtcore_select_banked_ready_request_for_owner(
    rtcore_replay_lane_request_state unit_state, unsigned owner_hw_sid,
    unsigned *thread_uid, const std::set<unsigned> *excluded_bank_ids,
    unsigned *selected_bank_id)
{
    const unsigned bank_count =
        rtcore_replay_v03_hw_request_state_bank_count_config();
    if (bank_count == 0) {
        return false;
    }

    std::vector<rtcore_replay_v03_hw_banked_ready_candidate> candidates(
        bank_count);
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (request.owner_hw_sid != owner_hw_sid ||
            !rtcore_replay_request_ready_for_state(request, unit_state)) {
            continue;
        }
        const unsigned bank_id = request.request_state_bank_id % bank_count;
        if (excluded_bank_ids &&
            excluded_bank_ids->find(bank_id) != excluded_bank_ids->end()) {
            continue;
        }
        rtcore_replay_v03_hw_banked_ready_candidate &candidate =
            candidates[bank_id];
        if (!candidate.valid || request.ready_order < candidate.ready_order) {
            candidate.valid = true;
            candidate.thread_uid = request.thread_uid;
            candidate.ready_order = request.ready_order;
        }
    }

    const unsigned long long cursor_key =
        rtcore_replay_v03_hw_ready_bank_cursor_key(owner_hw_sid, unit_state);
    unsigned cursor =
        g_rtcore_replay_v03_hw_ready_bank_rr_cursor_by_owner_unit[cursor_key] %
        bank_count;
    for (unsigned offset = 0; offset < bank_count; ++offset) {
        const unsigned bank_id = (cursor + offset) % bank_count;
        if (!candidates[bank_id].valid) {
            continue;
        }
        if (thread_uid) {
            *thread_uid = candidates[bank_id].thread_uid;
        }
        if (selected_bank_id) {
            *selected_bank_id = bank_id;
        }
        g_rtcore_replay_v03_hw_ready_bank_rr_cursor_by_owner_unit[cursor_key] =
            (bank_id + 1) % bank_count;
        return true;
    }
    return false;
}

static bool rtcore_replay_v03_hw_completion_entry_main_path_enabled()
{
    return true;
}

static bool rtcore_route_replay_completion_pending_to_warp_entry(
    rtcore_replay_lane_request *request)
{
    if (!request || !request->valid) {
        return false;
    }

    rtcore_refresh_replay_lane_request_ready_bits(request);
    request->ready_result_bit = true;
    return true;
}

static bool rtcore_enqueue_replay_request_by_state(
    rtcore_replay_lane_request *request_ptr, unsigned long long service_cycle = 0)
{
    (void)service_cycle;
    if (!request_ptr || !request_ptr->valid) {
        return false;
    }
    rtcore_replay_lane_request &request = *request_ptr;
    rtcore_refresh_replay_lane_request_ready_bits(&request);
    if (request.unit_latency_gate_pending) {
        return true;
    }

    switch (request.state) {
    case RTCORE_REPLAY_INVALID:
        return false;
    case RTCORE_REPLAY_ADMITTED:
        return true;
    case RTCORE_REPLAY_READY:
        return true;
    case RTCORE_REPLAY_ISSUED_NODE:
        return true;
    case RTCORE_REPLAY_ISSUED_PRIMITIVE:
        return true;
    case RTCORE_REPLAY_ISSUED_STACK:
        return true;
    case RTCORE_REPLAY_ISSUED_MEMORY:
        return true;
    case RTCORE_REPLAY_COMPLETION_PENDING:
        return rtcore_route_replay_completion_pending_to_warp_entry(
            &request);
    case RTCORE_REPLAY_WAITING_SHADER:
    case RTCORE_REPLAY_FINAL_WAIT_RETIRE:
    case RTCORE_REPLAY_COMPLETED:
        return true;
    }
    return false;
}

static bool rtcore_route_admitted_replay_request(
    unsigned thread_uid, unsigned long long service_cycle = 0)
{
    std::map<unsigned, rtcore_replay_lane_request>::iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    rtcore_record_replay_lane_request_state_identity_read();
    rtcore_record_replay_request_state_read();
    return rtcore_enqueue_replay_request_by_state(&it->second, service_cycle);
}

static bool rtcore_replay_request_owned_by_sm(unsigned thread_uid,
                                              unsigned owner_hw_sid)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    rtcore_record_replay_lane_request_state_identity_read();

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
    rtcore_record_replay_lane_request_state_identity_read();

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

static unsigned rtcore_count_replay_warp_completion_entry_lanes(unsigned mask)
{
    unsigned count = 0;
    while (mask != 0) {
        count += mask & 1u;
        mask >>= 1;
    }
    return count;
}

static bool rtcore_replay_request_has_warp_completion_entry_metadata(
    const rtcore_replay_lane_request &request)
{
    return request.valid && request.has_warp_metadata && request.lane_id < 32 &&
           request.active_mask != 0;
}

static rtcore_replay_warp_completion_entry_key
rtcore_make_replay_warp_completion_entry_key(
    const rtcore_replay_lane_request &request)
{
    rtcore_replay_warp_completion_entry_key key = {};
    key.owner_hw_sid = request.owner_hw_sid;
    key.warp_uid = request.warp_uid;
    key.warp_id = request.warp_id;
    key.active_mask = request.active_mask;
    return key;
}

static void rtcore_update_replay_warp_completion_entry_state(
    rtcore_replay_warp_completion_entry_state *state)
{
    if (!state || !state->valid) {
        return;
    }

    const unsigned active_completed_lanes =
        state->completed_lane_mask & state->key.active_mask;
    state->completed_lane_count =
        rtcore_count_replay_warp_completion_entry_lanes(
            active_completed_lanes);
    state->all_active_lanes_complete =
        state->key.active_mask != 0 &&
        active_completed_lanes == state->key.active_mask;
    state->scoreboard_handoff_ready =
        state->all_active_lanes_complete &&
        !state->scoreboard_handoff_delivered;
}

static unsigned rtcore_make_replay_result_data_slot(
    const rtcore_replay_lane_request &request)
{
    const unsigned reason =
        !request.valid
            ? RTCORE_REPLAY_CONTINUATION_PACKET_REASON_UNSUPPORTED
            : request.hit_geometry_summary_valid
                  ? RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY
                  : RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS;
    return 0x80000000u | reason;
}

static unsigned rtcore_make_replay_lane_status(
    const rtcore_replay_lane_request &request)
{
    (void)request;
    return static_cast<unsigned>(RTCORE_REPLAY_COMPLETED);
}

struct rtcore_replay_continuation_packet_lane_fact {
    unsigned reason;
    unsigned v_result;
    unsigned continuation_depth;
    bool context_profile_valid;
    unsigned context_layout_version;
    unsigned context_valid_flags;
    unsigned pipeline_profile_id;
    unsigned bvh_format_profile_id;
    unsigned handoff_words[32];
    bool terminal;
    bool continuation;
    bool unsupported;
    bool selector_valid;
    bool candidate_valid;
    bool software_return_valid;
};

static unsigned rtcore_replay_continuation_packet_reason_for_boundary(
    const char *boundary_reason)
{
    if (!boundary_reason) {
        return RTCORE_REPLAY_CONTINUATION_PACKET_REASON_UNSUPPORTED;
    }
    if (strcmp(boundary_reason, "oracle_anyhit") == 0) {
        return RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED;
    }
    if (strcmp(boundary_reason, "oracle_intersection") == 0) {
        return RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
    }
    return RTCORE_REPLAY_CONTINUATION_PACKET_REASON_UNSUPPORTED;
}

static bool rtcore_test_reported_attribute_packet_enabled()
{
    const char *value =
        getenv("VULKAN_SIM_RTCORE_TEST_REPORTED_ATTRIBUTE_PACKET");
    return value && value[0] && strcmp(value, "0") != 0;
}

static void rtcore_mark_continuation_reason_mask(
    rtcore_continuation_warp_boundary_state *state, unsigned lane_mask,
    const char *boundary_reason)
{
    if (!state) {
        return;
    }
    if (boundary_reason != NULL &&
        strcmp(boundary_reason, "synthetic_split") == 0) {
        state->reason_synthetic_split_mask |= lane_mask;
        return;
    }

    switch (rtcore_replay_continuation_packet_reason_for_boundary(
        boundary_reason)) {
    case RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED:
        state->reason_oracle_anyhit_mask |= lane_mask;
        break;
    case RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED:
        state->reason_oracle_intersection_mask |= lane_mask;
        break;
    default:
        state->reason_unsupported_mask |= lane_mask;
        break;
    }
}

static void rtcore_materialize_replay_continuation_packet_handoff_words(
    const rtcore_replay_lane_request &request,
    const rtcore_boundary_candidate_snapshot *boundary_candidate,
    rtcore_replay_continuation_packet_lane_fact *fact)
{
    if (fact == NULL) {
        return;
    }
    for (unsigned word = 0; word <= 12; ++word) {
        fact->handoff_words[word] = 0;
    }
    fact->handoff_words[0] = request.ray_flags;
    fact->handoff_words[1] =
        fact->candidate_valid ? request.sbt_record_offset
                              : request.miss_index;
    fact->handoff_words[2] =
        fact->candidate_valid ? request.sbt_record_stride : 0u;
    const bool boundary_candidate_valid =
        boundary_candidate != NULL && boundary_candidate->valid;
    fact->handoff_words[3] = fact->candidate_valid
                                 ? (boundary_candidate_valid
                                        ? boundary_candidate->hit_group_index
                                        : request.instance_sbt_contribution)
                                 : 0u;
    fact->handoff_words[4] =
        fact->candidate_valid
            ? (boundary_candidate_valid ? boundary_candidate->geometry_index
                                        : request.closest_hit_geometry_index)
            : 0u;
    if (fact->candidate_valid) {
        const bool procedural = boundary_candidate_valid
                                    ? boundary_candidate->geometry_type == 0x02u
                                    : request.closest_hit_geometry_type == 0x02u;
        const unsigned geometry_type = procedural ? 0x02u : 0x01u;
        const unsigned hit_kind = boundary_candidate_valid
                                      ? boundary_candidate->hit_kind
                                      : request.closest_hit_kind;
        const unsigned candidate_ref_kind = procedural ? 0x02u : 0x01u;
        const unsigned primitive_index =
            boundary_candidate_valid ? boundary_candidate->primitive_index
                                     : request.closest_hit_primitive_index;
        fact->handoff_words[6] = primitive_index;
        fact->handoff_words[7] = boundary_candidate_valid
                                     ? boundary_candidate->instance_index
                                     : request.closest_hit_instance_index;
        fact->handoff_words[9] = hit_kind | (geometry_type << 8);
        fact->handoff_words[11] =
            (primitive_index & 0x00ffffffu) |
            (candidate_ref_kind << 24) | (0x02u << 28);
    }
}

static rtcore_replay_continuation_packet_lane_fact
rtcore_make_replay_continuation_packet_lane_fact(
    const rtcore_replay_lane_request &request)
{
    rtcore_replay_continuation_packet_lane_fact fact = {};
    fact.reason = !request.valid
                      ? RTCORE_REPLAY_CONTINUATION_PACKET_REASON_UNSUPPORTED
                      : request.hit_geometry_summary_valid
                            ? RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY
                            : RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS;
    fact.v_result = 0x80000000u | fact.reason;
    fact.terminal =
        fact.reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS ||
        fact.reason ==
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY ||
        fact.reason ==
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_TRACE_DONE_NO_SHADER;
    fact.continuation = false;
    fact.unsupported =
        fact.reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_UNSUPPORTED;
    const bool selector_required =
        fact.reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS ||
        fact.reason ==
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY;
    const bool candidate_required =
        fact.reason ==
        RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY;
    const bool candidate_available =
        request.valid &&
        (request.hit_geometry_summary_valid ||
         request.oracle_anyhit_candidate_count != 0 ||
         request.oracle_requires_intersection_shader);
    fact.selector_valid =
        selector_required && request.valid && request.ray_sbt_inputs_valid;
    fact.candidate_valid = candidate_required && candidate_available;
    fact.software_return_valid = false;
    fact.continuation_depth = request.continuation_depth;
    fact.context_profile_valid = request.context_profile_valid;
    fact.context_layout_version = request.context_layout_version;
    fact.context_valid_flags = request.context_valid_flags;
    fact.pipeline_profile_id = request.pipeline_profile_id;
    fact.bvh_format_profile_id = request.bvh_format_profile_id;
    rtcore_materialize_replay_continuation_packet_handoff_words(
        request, NULL, &fact);
    return fact;
}

static bool rtcore_apply_v04_shadow_boundary_test_mutation(
    rtcore::abi_v04::shadow::boundary_values *values,
    bool *mutate_reserved_word)
{
    const char *mutation =
        getenv("VULKAN_SIM_RTCORE_TEST_V04_SHADOW_BOUNDARY_MUTATION");
    if (mutate_reserved_word != NULL) {
        *mutate_reserved_word = false;
    }
    if (mutation == NULL || mutation[0] == '\0' ||
        strcmp(mutation, "none") == 0) {
        return true;
    }
    if (values == NULL) {
        return false;
    }
    if (strcmp(mutation, "metadata_zero") == 0) {
        values->instance_metadata_reference = 0;
    } else if (strcmp(mutation, "metadata_misaligned") == 0) {
        values->instance_metadata_reference |= 0x08u;
    } else if (strcmp(mutation, "metadata_out_of_range") == 0) {
        values->instance_metadata_reference = UINT64_MAX & ~uint64_t{0x3f};
    } else if (strcmp(mutation, "sbt_overflow") == 0) {
        values->instance_sbt_contribution = 0x01000000u;
    } else if (strcmp(mutation, "hit_kind") == 0) {
        values->hit_kind = 0x80u;
    } else if (strcmp(mutation, "attribute_metadata") == 0) {
        values->input_attribute_word_count = 3;
    } else if (strcmp(mutation, "reserved") == 0) {
        if (mutate_reserved_word != NULL) {
            *mutate_reserved_word = true;
        }
    } else {
        return false;
    }
    return true;
}

static bool rtcore_apply_v04_tlas_binding_test_mutation(
    rtcore_tlas_binding_snapshot *snapshot)
{
    const char *mutation =
        getenv("VULKAN_SIM_RTCORE_TEST_V04_TLAS_BINDING_MUTATION");
    if (mutation == NULL || mutation[0] == '\0' ||
        strcmp(mutation, "none") == 0) {
        return true;
    }
    if (snapshot == NULL) {
        return false;
    }
    if (strcmp(mutation, "stale_generation") == 0) {
        snapshot->generation++;
        if (snapshot->generation == 0) snapshot->generation = 1;
    } else if (strcmp(mutation, "destroyed_binding") == 0) {
        snapshot->live = false;
    } else if (strcmp(mutation, "range_mismatch") == 0) {
        snapshot->size_bytes += 64;
        if (snapshot->size_bytes < 64) return false;
    } else {
        return false;
    }
    return true;
}

static unsigned rtcore_v04_live_publication_chunk_mask(uint32_t word_mask)
{
    unsigned chunk_mask = 0;
    for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
        if ((word_mask & (uint32_t{1} << word)) != 0) {
            chunk_mask |= 1u << (word / 8u);
        }
    }
    return chunk_mask;
}

static bool rtcore_v04_live_publication_matches_current_submit(
    const rtcore_replay_lane_request &request)
{
    return request.v04_live_publication_armed &&
           request.v04_live_publication_warp_uid == request.warp_uid;
}

static void rtcore_fail_v04_live_handoff_publication(
    const rtcore_replay_lane_request &request, const char *reason,
    uint32_t word_mask, unsigned chunk_mask)
{
    fprintf(stderr,
            "GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_FAULT "
            "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
            "publication_warp_uid=%u "
            "handoff_window_base=0x%llx word_mask=0x%08x "
            "chunk_mask=0x%x fault=%s\n",
            request.owner_hw_sid, request.thread_uid, request.lane_id,
            request.warp_uid, request.v04_live_publication_warp_uid,
            request.handoff_window_base, word_mask, chunk_mask,
            reason != NULL ? reason : "unknown");
    fflush(stderr);
    abort();
}

static bool rtcore_arm_v04_live_handoff_publication(
    rtcore_replay_lane_request *request, unsigned reason,
    const std::array<uint32_t, rtcore::abi_v04::kWordCount> &words,
    uint32_t word_mask, unsigned long long service_cycle)
{
    if (!rtcore_v04_live_handoff_publication_enabled()) {
        return true;
    }
    if (request == NULL || !request->valid ||
        !request->v04_shadow_boundary_enabled ||
        !rtcore_replay_memory_unit_path_active() ||
        request->v04_live_handoff_memory == NULL ||
        request->handoff_window_base == 0 || request->lane_id >= 32) {
        if (request != NULL) {
            rtcore_fail_v04_live_handoff_publication(
                *request, "arm_contract_invalid", word_mask, 0);
        }
        abort();
    }

    if (request->v04_live_publication_armed) {
        if (!rtcore_v04_live_publication_matches_current_submit(*request)) {
            rtcore_fail_v04_live_handoff_publication(
                *request, "stale_publication_submit_identity", word_mask,
                rtcore_v04_live_publication_chunk_mask(word_mask));
        }
        const bool same_publication =
            request->v04_live_publication_reason == reason &&
            request->v04_live_publication_word_mask == word_mask &&
            request->v04_live_publication_words == words;
        if (same_publication) {
            return request->v04_live_publication_committed;
        }
        rtcore_fail_v04_live_handoff_publication(
            *request, "overlapping_boundary_publication", word_mask,
            rtcore_v04_live_publication_chunk_mask(word_mask));
    }

    request->v04_live_publication_armed = false;
    request->v04_live_publication_committed = false;
    request->v04_live_publication_reason = reason;
    request->v04_live_publication_warp_uid = request->warp_uid;
    request->v04_live_publication_word_mask = word_mask;
    request->v04_live_publication_pending_chunk_mask =
        rtcore_v04_live_publication_chunk_mask(word_mask);
    request->v04_live_publication_acked_chunk_mask = 0;
    request->v04_live_publication_armed_cycle = service_cycle;
    request->v04_live_publication_words = words;

    const uint32_t shader_return_sentinel_word = uint32_t{1} << 24;
    const uint32_t allowed_rtcore_words =
        (uint32_t{0x3ff} << 14) | shader_return_sentinel_word |
        (uint32_t{0xf} << 28);
    if ((word_mask & ~allowed_rtcore_words) != 0) {
        rtcore_fail_v04_live_handoff_publication(
            *request, "word_ownership_violation", word_mask,
            request->v04_live_publication_pending_chunk_mask);
    }
    if ((word_mask & shader_return_sentinel_word) != 0 &&
        (!rtcore_v04_shader_return_consumer_enabled() ||
         !rtcore::abi_v04::shadow::
              shader_return_reason_requires_publication(reason) ||
         words[rtcore::abi_v04::kCommitRetainedCandidate.word] !=
             rtcore::abi_v04::shadow::shader_return_unpublished_sentinel())) {
        rtcore_fail_v04_live_handoff_publication(
            *request, "shader_return_sentinel_contract_invalid", word_mask,
            request->v04_live_publication_pending_chunk_mask);
    }

    const unsigned long long lane_offset =
        static_cast<unsigned long long>(request->lane_id) *
        rtcore::abi_v04::kLaneSlotBytes;
    const unsigned long long lane_address =
        request->handoff_window_base + lane_offset;
    if (lane_address < request->handoff_window_base) {
        rtcore_fail_v04_live_handoff_publication(
            *request, "lane_address_overflow", word_mask,
            request->v04_live_publication_pending_chunk_mask);
    }
    request->v04_live_handoff_memory->read_simulator_backing(
        lane_address, sizeof(request->v04_live_publication_preimage_words),
        request->v04_live_publication_preimage_words.data());

    uint32_t immutable_mismatch_mask = 0;
    const uint32_t immutable_mask =
        rtcore::abi_v04::shadow::trace_input_owned_word_mask();
    for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
        const uint32_t word_bit = uint32_t{1} << word;
        if ((immutable_mask & word_bit) != 0 &&
            request->v04_live_publication_preimage_words[word] !=
                words[word]) {
            immutable_mismatch_mask |= word_bit;
        }
    }
    if (immutable_mismatch_mask != 0) {
        rtcore_fail_v04_live_handoff_publication(
            *request, "compiler_owned_preimage_mismatch", word_mask,
            request->v04_live_publication_pending_chunk_mask);
    }

    request->v04_live_publication_armed = true;
    if (request->v04_live_publication_pending_chunk_mask == 0) {
        request->v04_live_publication_committed = true;
    } else {
        for (unsigned chunk = 0;
             chunk < RTCORE_V04_LIVE_PUBLICATION_CHUNK_COUNT; ++chunk) {
            if ((request->v04_live_publication_pending_chunk_mask &
                 (1u << chunk)) == 0) {
                continue;
            }
            rtcore_enqueue_memory_unit_handoff_window_request(
                request->owner_hw_sid, request->thread_uid,
                request->lane_id,
                RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE + chunk,
                RTCORE_V02_LSU_ACCESS_HANDOFF_PUBLICATION_STORE,
                lane_address +
                    chunk * RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES,
                1, true, service_cycle);
        }
    }

    printf("GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_ARMED "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "reason=%u lane_address=0x%llx word_mask=0x%08x "
           "chunk_mask=0x%x chunk_count=%u committed_without_write=%u "
           "rt_memory_unit=1 completion_release=after_matching_write_ack "
           "service_cycle=%llu\n",
           request->owner_hw_sid, request->thread_uid, request->lane_id,
           request->warp_uid, reason, lane_address, word_mask,
           request->v04_live_publication_pending_chunk_mask,
           rtcore_continuation_count_lanes(
               request->v04_live_publication_pending_chunk_mask),
           request->v04_live_publication_committed ? 1u : 0u,
           service_cycle);
    fflush(stdout);
    return request->v04_live_publication_committed;
}

static void rtcore_invalidate_v04_live_handoff_publication_for_resubmit(
    rtcore_replay_lane_request *request, unsigned next_warp_uid,
    unsigned long long service_cycle)
{
    if (!rtcore_v04_live_handoff_publication_enabled()) {
        return;
    }
    if (request == NULL || !request->valid ||
        !rtcore_v04_live_publication_matches_current_submit(*request) ||
        !request->v04_live_publication_committed) {
        if (request != NULL) {
            rtcore_fail_v04_live_handoff_publication(
                *request, "resubmit_before_current_publication_commit",
                request->v04_live_publication_word_mask,
                request->v04_live_publication_pending_chunk_mask);
        }
        abort();
    }

    printf("GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_INVALIDATE "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u "
           "publication_warp_uid=%u next_warp_uid=%u reason=%u "
           "service_cycle=%llu\n",
           request->owner_hw_sid, request->thread_uid, request->lane_id,
           request->v04_live_publication_warp_uid, next_warp_uid,
           request->v04_live_publication_reason, service_cycle);
    fflush(stdout);

    request->v04_live_publication_armed = false;
    request->v04_live_publication_committed = false;
    request->v04_live_publication_reason = 0;
    request->v04_live_publication_warp_uid = 0;
    request->v04_live_publication_word_mask = 0;
    request->v04_live_publication_pending_chunk_mask = 0;
    request->v04_live_publication_acked_chunk_mask = 0;
    request->v04_live_publication_armed_cycle = 0;
    request->v04_live_publication_words.fill(0);
    request->v04_live_publication_preimage_words.fill(0);
}

static bool rtcore_observe_v04_shadow_boundary_publication(
    const rtcore_replay_lane_request &request, unsigned reason,
    const rtcore_boundary_candidate_snapshot *boundary_candidate,
    std::array<uint32_t, rtcore::abi_v04::kWordCount> *published_words,
    uint32_t *published_word_mask)
{
    if (published_words != NULL) {
        published_words->fill(0);
    }
    if (published_word_mask != NULL) {
        *published_word_mask = 0;
    }
    if (!request.v04_shadow_boundary_enabled) {
        return true;
    }
    if (!request.v04_shadow_trace_input_valid) {
        return false;
    }

    const bool continuation_reason =
        reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED ||
        reason ==
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
    const bool terminal_hit_reason =
        reason ==
        RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY;
    const char *fact_source = continuation_reason
                                  ? "event_boundary_candidate"
                                  : terminal_hit_reason
                                        ? "terminal_boundary_fact"
                                        : "reason_has_no_candidate";
    rtcore::abi_v04::shadow::boundary_values values;
    if (continuation_reason && boundary_candidate != NULL) {
        values = boundary_candidate->v04_boundary_values;
        if (reason ==
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED) {
            rtcore::abi_v04::shadow::tighten_intersection_boundary(
                request.v04_replay_committed_boundary_values, &values);
        }
    } else if (terminal_hit_reason) {
        values = request.v04_replay_committed_boundary_values;
    }

    bool mutate_reserved_word = false;
    const bool mutation_valid =
        rtcore_apply_v04_shadow_boundary_test_mutation(
            &values, &mutate_reserved_word);
    rtcore_tlas_binding_snapshot tlas_binding =
        request.v04_tlas_binding;
    const bool tlas_mutation_valid =
        rtcore_apply_v04_tlas_binding_test_mutation(&tlas_binding);
    const char *tlas_binding_failure =
        request.v04_tlas_binding_enforcement_enabled ? "unvalidated"
                                                     : "disabled";
    bool tlas_binding_valid = true;
    if (request.v04_tlas_binding_enforcement_enabled) {
        const uint64_t instance_record_size =
            rtcore::abi_v04::shadow::boundary_reason_has_candidate(reason)
                ? 128u
                : 0u;
        tlas_binding_valid = tlas_mutation_valid &&
            VulkanRayTracing::validateTlasBinding(
                tlas_binding,
                instance_record_size != 0
                    ? values.instance_metadata_reference
                    : 0,
                instance_record_size, &tlas_binding_failure);
    }
    rtcore::abi_v04::shadow::boundary_publication publication =
        rtcore::abi_v04::shadow::build_boundary_publication(
            request.v04_shadow_trace_input_words, reason, values);
    if (mutate_reserved_word && publication.valid()) {
        publication.words[rtcore::abi_v04::kHitKind.word] |= 1u << 9;
    }
    const rtcore::abi_v04::shadow::boundary_validation validation =
        rtcore::abi_v04::shadow::validate_boundary_publication(
            publication.words, request.v04_shadow_trace_input_words,
            reason, values);
    const bool valid = mutation_valid && tlas_binding_valid &&
                       publication.valid() && validation.matches();
    if (valid && published_words != NULL) {
        *published_words = publication.words;
    }
    if (valid && published_word_mask != NULL) {
        *published_word_mask = publication.written_word_mask;
    }

    printf("GPGPU-Sim RTCORE_V04_SHADOW_BOUNDARY_PUBLICATION "
           "valid=%u reason=%u owner_hw_sid=%u thread_uid=%u lane_id=%u "
           "warp_uid=%u fact_source=%s candidate_event_seq=%u "
           "instance_metadata_ref=0x%llx "
           "instance_metadata_source=gen_rt_tlas_instance_leaf_device_address "
           "tlas_binding_enforcement=%u tlas_binding_valid=%u "
           "tlas_binding_failure=%s tlas_object_id=%llu "
           "tlas_generation=%u tlas_device_base=0x%llx "
           "tlas_size=%llu "
           "boundary_ray_tmax_fp32=0x%08x written_word_mask=0x%08x "
           "compared_word_mask=0x%08x mismatch_word_mask=0x%08x "
           "error=%s "
           "boundary_words={w14=0x%08x,w15=0x%08x,w16=0x%08x,"
           "w17=0x%08x,w18=0x%08x,w19=0x%08x,w20=0x%08x,"
           "w21=0x%08x,w22=0x%08x,w23=0x%08x} "
           "inline_attributes={w28=0x%08x,w29=0x%08x,w30=0x%08x,"
           "w31=0x%08x} event_local=1 live_handoff_write=0 "
           "v03_completion_authority=1 traversal_authority=0 "
           "shader_consumer_enabled=0\n",
           valid ? 1u : 0u, reason, request.owner_hw_sid,
           request.thread_uid, request.lane_id, request.warp_uid,
           fact_source,
           boundary_candidate != NULL ? boundary_candidate->event_seq : 0u,
           static_cast<unsigned long long>(
               values.instance_metadata_reference),
           request.v04_tlas_binding_enforcement_enabled ? 1u : 0u,
           tlas_binding_valid ? 1u : 0u, tlas_binding_failure,
           (unsigned long long)tlas_binding.object_id,
           tlas_binding.generation,
           (unsigned long long)tlas_binding.device_base_address,
           (unsigned long long)tlas_binding.size_bytes,
           values.boundary_ray_tmax_fp32, publication.written_word_mask,
           validation.compared_word_mask, validation.mismatch_word_mask,
           rtcore::abi_v04::shadow::boundary_error_name(
               publication.valid() ? validation.error : publication.error),
           publication.words[14], publication.words[15],
           publication.words[16], publication.words[17],
           publication.words[18], publication.words[19],
           publication.words[20], publication.words[21],
           publication.words[22], publication.words[23],
           publication.words[28], publication.words[29],
           publication.words[30], publication.words[31]);
    fflush(stdout);
    return valid;
}

static bool rtcore_prepare_v04_terminal_completion_publication(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!rtcore_v04_live_handoff_publication_enabled()) {
        return true;
    }
    if (request == NULL || !request->valid) {
        abort();
    }

    const rtcore_replay_continuation_packet_lane_fact packet_fact =
        rtcore_make_replay_continuation_packet_lane_fact(*request);
    if (!packet_fact.terminal) {
        rtcore_fail_v04_live_handoff_publication(
            *request, "completion_reason_not_terminal", 0, 0);
    }
    if (rtcore_v04_live_publication_matches_current_submit(*request) &&
        request->v04_live_publication_reason == packet_fact.reason) {
        return request->v04_live_publication_committed;
    }

    std::array<uint32_t, rtcore::abi_v04::kWordCount> publication_words = {};
    uint32_t publication_word_mask = 0;
    if (!rtcore_observe_v04_shadow_boundary_publication(
            *request, packet_fact.reason, NULL, &publication_words,
            &publication_word_mask)) {
        rtcore_fail_v04_live_handoff_publication(
            *request, "terminal_boundary_publication_invalid",
            publication_word_mask,
            rtcore_v04_live_publication_chunk_mask(publication_word_mask));
    }
    return rtcore_arm_v04_live_handoff_publication(
        request, packet_fact.reason, publication_words,
        publication_word_mask, service_cycle);
}

static void rtcore_populate_continuation_packet_handoff_summaries(
    rtcore_continuation_return_packet *packet,
    const rtcore_continuation_warp_boundary_state &state,
    unsigned long long service_cycle)
{
    if (!packet || !packet->valid) {
        return;
    }

    for (std::map<unsigned, rtcore_replay_lane_request>::iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        rtcore_replay_lane_request &request = it->second;
        if (!rtcore_continuation_request_matches_boundary_key(request,
                                                              state)) {
            continue;
        }
        const unsigned lane = request.lane_id;
        const unsigned lane_mask = rtcore_continuation_lane_mask(lane);
        if ((packet->boundary_reached_mask & lane_mask) == 0) {
            continue;
        }

        rtcore_replay_continuation_packet_lane_fact packet_fact =
            rtcore_make_replay_continuation_packet_lane_fact(request);
        if ((state.reason_oracle_anyhit_mask & lane_mask) != 0) {
            packet_fact.reason =
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED;
            packet_fact.terminal = false;
            packet_fact.continuation = true;
        } else if ((state.reason_oracle_intersection_mask & lane_mask) != 0) {
            packet_fact.reason =
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
            packet_fact.terminal = false;
            packet_fact.continuation = true;
            if (rtcore_test_reported_attribute_packet_enabled()) {
                packet_fact.software_return_valid = true;
                packet_fact.handoff_words[13] = 0x04u;
                packet_fact.handoff_words[14] = 0x3f000000u;
                packet_fact.handoff_words[15] =
                    (state.boundary_candidates[lane].hit_kind & 0xffu) |
                    (1u << 8) |
                    (16u << 16) | (2u << 24);
                packet_fact.handoff_words[16] = 0x3e800000u;
            }
        }
        const bool selector_required =
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS ||
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY ||
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED ||
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
        const bool candidate_required =
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY ||
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED ||
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
        const rtcore_boundary_candidate_snapshot &boundary_candidate =
            state.boundary_candidates[lane];
        const bool shader_continuation_reason =
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED ||
            packet_fact.reason ==
                RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
        const bool candidate_available =
            request.valid &&
            (shader_continuation_reason
                 ? boundary_candidate.valid != 0
                 : request.hit_geometry_summary_valid);
        packet_fact.selector_valid =
            selector_required && request.valid && request.ray_sbt_inputs_valid;
        packet_fact.candidate_valid = candidate_required && candidate_available;
        std::array<uint32_t, rtcore::abi_v04::kWordCount>
            v04_shadow_handoff_words = {};
        uint32_t v04_shadow_written_word_mask = 0;
        if (!rtcore_observe_v04_shadow_boundary_publication(
                request, packet_fact.reason,
                shader_continuation_reason ? &boundary_candidate : NULL,
                &v04_shadow_handoff_words,
                &v04_shadow_written_word_mask)) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_SHADOW_BOUNDARY_FAULT "
                    "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
                    "reason=%u fault=event_local_publication_invalid\n",
                    request.owner_hw_sid, request.thread_uid, request.lane_id,
                    request.warp_uid, packet_fact.reason);
            fflush(stderr);
            abort();
        }
        if (shader_continuation_reason &&
            rtcore_v04_shader_return_consumer_enabled()) {
            const uint32_t chunk_mask_before_sentinel =
                rtcore_v04_live_publication_chunk_mask(
                    v04_shadow_written_word_mask);
            v04_shadow_handoff_words[
                rtcore::abi_v04::kCommitRetainedCandidate.word] =
                rtcore::abi_v04::shadow::
                    shader_return_unpublished_sentinel();
            v04_shadow_written_word_mask |= uint32_t{1} <<
                rtcore::abi_v04::kCommitRetainedCandidate.word;
            const uint32_t chunk_mask_after_sentinel =
                rtcore_v04_live_publication_chunk_mask(
                    v04_shadow_written_word_mask);
            const bool added_publication_chunk =
                chunk_mask_after_sentinel != chunk_mask_before_sentinel;
            printf("GPGPU-Sim RTCORE_V04_LIVE_SHADER_RETURN_SENTINEL_ARMED "
                   "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
                   "reason=%u sentinel=0x%08x added_chunk=%u "
                   "live_memory_path=1\n",
                   request.owner_hw_sid, request.thread_uid, request.lane_id,
                   request.warp_uid, packet_fact.reason,
                   v04_shadow_handoff_words[
                       rtcore::abi_v04::kCommitRetainedCandidate.word],
                   added_publication_chunk ? 1u : 0u);
            fflush(stdout);
        }
        if (request.v04_shadow_boundary_enabled) {
            packet->v04_shadow_boundary_image_valid_mask |= lane_mask;
            for (unsigned word = 0; word < rtcore::abi_v04::kWordCount;
                 ++word) {
                packet->v04_shadow_handoff_words[lane][word] =
                    v04_shadow_handoff_words[word];
            }
            rtcore_arm_v04_live_handoff_publication(
                &request, packet_fact.reason, v04_shadow_handoff_words,
                v04_shadow_written_word_mask, service_cycle);
        }
        rtcore_materialize_replay_continuation_packet_handoff_words(
            request,
            shader_continuation_reason ? &boundary_candidate : NULL,
            &packet_fact);
        packet->boundary_candidates[lane] = boundary_candidate;
        packet_fact.v_result = 0x80000000u | packet_fact.reason;
        packet->v_result[lane] = packet_fact.v_result;
        packet->context_layout_version[lane] =
            packet_fact.context_layout_version;
        packet->context_valid_flags[lane] = packet_fact.context_valid_flags;
        packet->pipeline_profile_id[lane] = packet_fact.pipeline_profile_id;
        packet->bvh_format_profile_id[lane] =
            packet_fact.bvh_format_profile_id;
        if (packet_fact.context_profile_valid) {
            packet->context_profile_valid_mask |= lane_mask;
        }
        if (packet_fact.selector_valid) {
            packet->handoff_selector_valid_mask |= lane_mask;
        }
        if (packet_fact.candidate_valid) {
            packet->handoff_candidate_valid_mask |= lane_mask;
        }
        if (packet_fact.software_return_valid) {
            packet->handoff_software_return_valid_mask |= lane_mask;
        }
        for (unsigned word = 0; word < 32; ++word) {
            packet->handoff_words[lane][word] =
                packet_fact.handoff_words[word];
        }
    }
}

static unsigned rtcore_replay_warp_completion_reason_mask(
    const rtcore_replay_warp_completion_entry_state &state, unsigned reason)
{
    unsigned mask = 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((state.lane_completion_valid_mask & lane_mask) == 0) {
            continue;
        }
        if (state.lane_completion_reason[lane] == reason) {
            mask |= lane_mask;
        }
    }
    return mask;
}

static void rtcore_log_replay_continuation_packet_schema(
    const char *packet_source, unsigned owner_hw_sid, unsigned warp_uid,
    unsigned warp_id, unsigned active_mask, unsigned completion_valid_mask,
    unsigned terminal_mask, unsigned continuation_mask,
    unsigned selector_valid_mask, unsigned candidate_valid_mask,
    unsigned software_return_valid_mask, unsigned context_profile_valid_mask,
    unsigned reported_attribute_metadata_valid_mask,
    unsigned inline_payload_location_valid_mask, unsigned reason_miss_mask,
    unsigned reason_closest_hit_mask, unsigned reason_any_hit_mask,
    unsigned reason_intersection_mask, unsigned reason_trace_done_mask,
    unsigned reason_fault_mask, unsigned reason_unsupported_mask,
    unsigned continuation_depth, unsigned long long service_cycle)
{
    printf("GPGPU-Sim RTCORE_REPLAY_CONTINUATION_PACKET_SCHEMA "
           "packet_schema_version=%u packet_source=%s "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
           "completion_valid_mask=0x%08x terminal_mask=0x%08x "
           "continuation_mask=0x%08x selector_valid_mask=0x%08x "
           "candidate_valid_mask=0x%08x software_return_valid_mask=0x%08x "
           "context_profile_valid_mask=0x%08x "
           "reported_attribute_metadata_valid_mask=0x%08x "
           "inline_payload_location_valid_mask=0x%08x "
           "reason_miss_mask=0x%08x reason_closest_hit_mask=0x%08x "
           "reason_any_hit_mask=0x%08x reason_intersection_mask=0x%08x "
           "reason_trace_done_mask=0x%08x reason_fault_mask=0x%08x "
           "reason_unsupported_mask=0x%08x "
           "continuation_depth=%u service_cycle=%llu\n",
           RTCORE_REPLAY_CONTINUATION_PACKET_SCHEMA_VERSION,
           packet_source ? packet_source : "unknown", owner_hw_sid, warp_uid,
           warp_id, active_mask, completion_valid_mask, terminal_mask,
           continuation_mask, selector_valid_mask, candidate_valid_mask,
           software_return_valid_mask, context_profile_valid_mask,
           reported_attribute_metadata_valid_mask,
           inline_payload_location_valid_mask, reason_miss_mask,
           reason_closest_hit_mask, reason_any_hit_mask,
           reason_intersection_mask, reason_trace_done_mask, reason_fault_mask,
           reason_unsupported_mask, continuation_depth, service_cycle);
    fflush(stdout);
}

static unsigned rtcore_replay_lane_status_valid_mask(
    const rtcore_replay_warp_completion_entry_state &state)
{
    unsigned mask = 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
        if (state.lane_status[lane] != 0) {
            mask |= 1u << lane;
        }
    }
    return mask;
}

static unsigned rtcore_replay_first_active_lane(unsigned mask)
{
    for (unsigned lane = 0; lane < 32; ++lane) {
        if (mask & (1u << lane)) {
            return lane;
        }
    }
    return 0;
}

static void rtcore_log_replay_warp_completion_entry(
    const rtcore_replay_warp_completion_entry_state &state)
{
    printf("GPGPU-Sim RTCORE_REPLAY_WARP_COMPLETION_ENTRY "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
           "admitted_lane_mask=0x%08x completed_lane_mask=0x%08x "
           "result_valid_mask=0x%08x completed_lane_count=%u "
           "result_reg_base=%u lane_status_valid_mask=0x%08x "
           "packet_schema_version=%u packet_completion_valid_mask=0x%08x "
           "packet_terminal_lane_mask=0x%08x "
           "packet_continuation_lane_mask=0x%08x "
           "packet_handoff_selector_valid_mask=0x%08x "
           "packet_handoff_candidate_valid_mask=0x%08x "
           "packet_handoff_software_return_valid_mask=0x%08x "
           "all_active_lanes_complete=%u capacity=%u\n",
           state.key.owner_hw_sid, state.key.warp_uid, state.key.warp_id,
           state.key.active_mask, state.admitted_lane_mask,
           state.completed_lane_mask, state.result_valid_mask,
           state.completed_lane_count, state.result_reg_base,
           rtcore_replay_lane_status_valid_mask(state),
           state.packet_schema_version, state.lane_completion_valid_mask,
           state.terminal_lane_mask, state.continuation_lane_mask,
           state.handoff_selector_valid_mask,
           state.handoff_candidate_valid_mask,
           state.handoff_software_return_valid_mask,
           state.all_active_lanes_complete ? 1 : 0,
           rtcore_replay_warp_completion_entry_capacity_config());
    fflush(stdout);
}

static void rtcore_log_replay_scoreboard_result_packet(
    const rtcore_replay_warp_completion_entry_state &state,
    unsigned long long service_cycle)
{
    const unsigned first_lane =
        rtcore_replay_first_active_lane(state.result_valid_mask);
    rtcore_log_replay_continuation_packet_schema(
        "scoreboard_result_packet", state.key.owner_hw_sid, state.key.warp_uid,
        state.key.warp_id, state.key.active_mask,
        state.lane_completion_valid_mask, state.terminal_lane_mask,
        state.continuation_lane_mask, state.handoff_selector_valid_mask,
        state.handoff_candidate_valid_mask,
        state.handoff_software_return_valid_mask,
        state.context_profile_valid_mask,
        state.reported_attribute_metadata_valid_mask,
        state.inline_payload_location_valid_mask,
        rtcore_replay_warp_completion_reason_mask(
            state, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS),
        rtcore_replay_warp_completion_reason_mask(
            state,
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY),
        rtcore_replay_warp_completion_reason_mask(
            state, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED),
        rtcore_replay_warp_completion_reason_mask(
            state,
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED),
        rtcore_replay_warp_completion_reason_mask(
            state,
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_TRACE_DONE_NO_SHADER),
        rtcore_replay_warp_completion_reason_mask(
            state, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_FAULT),
        state.unsupported_reason_mask, state.lane_continuation_depth[first_lane],
        service_cycle);
    printf("GPGPU-Sim RTCORE_REPLAY_SCOREBOARD_RESULT_PACKET "
           "owner_hw_sid=%u service_cycle=%llu "
           "scoreboard_result_packet_valid=%u "
           "warp_uid=%u warp_id=%u active_mask=0x%08x "
           "packet_result_reg_base=%u packet_result_valid_mask=0x%08x "
           "packet_lane_status_valid_mask=0x%08x "
           "packet_schema_version=%u "
           "packet_completion_valid_mask=0x%08x "
           "packet_terminal_lane_mask=0x%08x "
           "packet_continuation_lane_mask=0x%08x "
           "packet_handoff_selector_valid_mask=0x%08x "
           "packet_handoff_candidate_valid_mask=0x%08x "
           "packet_handoff_software_return_valid_mask=0x%08x "
           "packet_first_result_data_slot=0x%08x "
           "packet_first_lane_status=0x%08x "
           "scoreboard_handoff_cycle=%llu\n",
           state.key.owner_hw_sid, service_cycle,
           state.scoreboard_handoff_delivered ? 1 : 0, state.key.warp_uid,
           state.key.warp_id, state.key.active_mask, state.result_reg_base,
           state.result_valid_mask, rtcore_replay_lane_status_valid_mask(state),
           state.packet_schema_version, state.lane_completion_valid_mask,
           state.terminal_lane_mask, state.continuation_lane_mask,
           state.handoff_selector_valid_mask,
           state.handoff_candidate_valid_mask,
           state.handoff_software_return_valid_mask,
           state.result_data_slot[first_lane], state.lane_status[first_lane],
           state.scoreboard_handoff_cycle);
    fflush(stdout);
}

static unsigned rtcore_continuation_packet_lane_reason(
    const rtcore_continuation_return_packet &packet, unsigned lane_mask)
{
    for (unsigned lane = 0; lane < 32; ++lane) {
        if ((lane_mask & (1u << lane)) != 0) {
            return packet.v_result[lane] & 0xffu;
        }
    }
    return RTCORE_REPLAY_CONTINUATION_PACKET_REASON_UNSUPPORTED;
}

static unsigned rtcore_continuation_packet_reason_mask(
    const rtcore_continuation_return_packet &packet, unsigned reason)
{
    unsigned mask = 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((packet.boundary_reached_mask & lane_mask) != 0 &&
            (packet.v_result[lane] & 0xffu) == reason) {
            mask |= lane_mask;
        }
    }
    return mask;
}

static unsigned rtcore_make_continuation_packet_result_data_slot(
    const rtcore_continuation_return_packet &packet, unsigned lane,
    unsigned reason)
{
    (void)reason;
    return packet.v_result[lane];
}

static bool rtcore_publish_scoreboard_visible_continuation_packet(
    const rtcore_continuation_return_packet &packet,
    unsigned long long service_cycle)
{
    if (!rtcore_replay_warp_completion_entry_enabled() || !packet.valid ||
        packet.active_mask == 0) {
        return false;
    }

    const unsigned completion_mask =
        packet.boundary_reached_mask & packet.active_mask;
    if (completion_mask == 0) {
        return false;
    }

    rtcore_replay_warp_completion_entry_key key = {};
    key.owner_hw_sid = packet.owner_hw_sid;
    key.warp_uid = packet.warp_uid;
    key.warp_id = packet.warp_id;
    key.active_mask = packet.active_mask;

    rtcore_replay_warp_completion_entry_state &state =
        g_rtcore_replay_warp_completion_entries[key];
    if (!state.valid) {
        state.valid = true;
        state.key = key;
    }

    const unsigned continuation_mask =
        packet.resume_required_mask & completion_mask;
    const unsigned terminal_mask = packet.terminal_mask & completion_mask;
    const unsigned unsupported_mask =
        packet.reason_unsupported_mask & completion_mask;

    state.packet_schema_version =
        RTCORE_REPLAY_CONTINUATION_PACKET_SCHEMA_VERSION;
    state.context_profile_valid_mask =
        (state.context_profile_valid_mask & ~completion_mask) |
        (packet.context_profile_valid_mask & completion_mask);
    state.reported_attribute_metadata_valid_mask =
        (state.reported_attribute_metadata_valid_mask & ~completion_mask) |
        (packet.reported_attribute_metadata_valid_mask & completion_mask);
    state.inline_payload_location_valid_mask =
        (state.inline_payload_location_valid_mask & ~completion_mask) |
        (packet.inline_payload_location_valid_mask & completion_mask);
    state.inline_payload_base_word = packet.inline_payload_base_word;
    state.max_inline_attribute_words = packet.max_inline_attribute_words;
    state.admitted_lane_mask |= completion_mask;
    state.completed_lane_mask =
        (state.completed_lane_mask & ~completion_mask) | completion_mask;
    state.result_valid_mask =
        (state.result_valid_mask & ~completion_mask) | completion_mask;
    state.lane_completion_valid_mask =
        (state.lane_completion_valid_mask & ~completion_mask) |
        completion_mask;
    state.terminal_lane_mask =
        (state.terminal_lane_mask & ~completion_mask) | terminal_mask;
    state.continuation_lane_mask =
        (state.continuation_lane_mask & ~completion_mask) |
        continuation_mask;
    state.unsupported_reason_mask =
        (state.unsupported_reason_mask & ~completion_mask) |
        unsupported_mask;
    state.handoff_selector_valid_mask =
        (state.handoff_selector_valid_mask & ~completion_mask) |
        (packet.handoff_selector_valid_mask & completion_mask);
    state.handoff_candidate_valid_mask =
        (state.handoff_candidate_valid_mask & ~completion_mask) |
        (packet.handoff_candidate_valid_mask & completion_mask);
    state.handoff_software_return_valid_mask =
        (state.handoff_software_return_valid_mask & ~completion_mask) |
        (packet.handoff_software_return_valid_mask & completion_mask);
    state.v04_shadow_boundary_image_valid_mask =
        (state.v04_shadow_boundary_image_valid_mask & ~completion_mask) |
        (packet.v04_shadow_boundary_image_valid_mask & completion_mask);
    state.scoreboard_handoff_delivered = false;
    state.scoreboard_handoff_cycle = 0;
    state.all_active_lanes_complete_logged = false;

    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((completion_mask & lane_mask) == 0) {
            continue;
        }
        const unsigned reason =
            rtcore_continuation_packet_lane_reason(packet, lane_mask);
        state.result_data_slot[lane] =
            rtcore_make_continuation_packet_result_data_slot(packet, lane,
                                                             reason);
        state.lane_status[lane] =
            static_cast<unsigned>(RTCORE_REPLAY_COMPLETED);
        state.lane_completion_reason[lane] = reason;
        state.lane_continuation_depth[lane] = packet.continuation_depth;
        state.context_layout_version[lane] =
            packet.context_layout_version[lane];
        state.context_valid_flags[lane] = packet.context_valid_flags[lane];
        state.pipeline_profile_id[lane] = packet.pipeline_profile_id[lane];
        state.bvh_format_profile_id[lane] =
            packet.bvh_format_profile_id[lane];
        state.boundary_candidates[lane] = packet.boundary_candidates[lane];
        for (unsigned word = 0; word < 32; ++word) {
            state.handoff_words[lane][word] =
                packet.handoff_words[lane][word];
            state.v04_shadow_handoff_words[lane][word] =
                packet.v04_shadow_handoff_words[lane][word];
        }
    }

    rtcore_update_replay_warp_completion_entry_state(&state);
    printf("GPGPU-Sim RTCORE_SCOREBOARD_VISIBLE_CONTINUATION_PACKET "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
           "completion_valid_mask=0x%08x terminal_mask=0x%08x "
           "continuation_mask=0x%08x software_return_valid_mask=0x%08x "
           "unsupported_reason_mask=0x%08x all_active_lanes_complete=%u "
           "scoreboard_handoff_ready=%u service_cycle=%llu\n",
           packet.owner_hw_sid, packet.warp_uid, packet.warp_id,
           packet.active_mask, state.lane_completion_valid_mask,
           state.terminal_lane_mask, state.continuation_lane_mask,
           state.handoff_software_return_valid_mask, state.unsupported_reason_mask,
           state.all_active_lanes_complete ? 1u : 0u,
           state.scoreboard_handoff_ready ? 1u : 0u, service_cycle);
    fflush(stdout);
    if (state.all_active_lanes_complete &&
        !state.all_active_lanes_complete_logged) {
        state.all_active_lanes_complete_logged = true;
        rtcore_log_replay_warp_completion_entry(state);
    }
    return state.all_active_lanes_complete;
}

static unsigned rtcore_count_scoreboard_handoff_ready_warp_entries(
    unsigned owner_hw_sid)
{
    unsigned count = 0;
    for (std::map<rtcore_replay_warp_completion_entry_key,
                  rtcore_replay_warp_completion_entry_state>::iterator it =
             g_rtcore_replay_warp_completion_entries.begin();
         it != g_rtcore_replay_warp_completion_entries.end(); ++it) {
        if (!it->second.valid || it->second.key.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        rtcore_update_replay_warp_completion_entry_state(&it->second);
        if (it->second.scoreboard_handoff_ready) {
            count++;
        }
    }
    return count;
}

static unsigned rtcore_service_completed_warp_entry_handoffs_for_owner(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    unsigned budget = rtcore_replay_scoreboard_result_handoff_budget_config();
    unsigned delivered = 0;
    const unsigned ready_before =
        rtcore_count_scoreboard_handoff_ready_warp_entries(owner_hw_sid);

    for (std::map<rtcore_replay_warp_completion_entry_key,
                  rtcore_replay_warp_completion_entry_state>::iterator it =
             g_rtcore_replay_warp_completion_entries.begin();
         it != g_rtcore_replay_warp_completion_entries.end(); ++it) {
        if (budget == 0) {
            break;
        }
        if (!it->second.valid || it->second.key.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        rtcore_update_replay_warp_completion_entry_state(&it->second);
        if (!it->second.scoreboard_handoff_ready) {
            continue;
        }
        it->second.scoreboard_handoff_delivered = true;
        it->second.scoreboard_handoff_ready = false;
        it->second.scoreboard_handoff_cycle = service_cycle;
        rtcore_log_replay_scoreboard_result_packet(it->second, service_cycle);
        budget--;
        delivered++;
    }
    const unsigned blocked = ready_before > delivered ? ready_before - delivered : 0;
    g_rtcore_replay_scoreboard_result_handoff_stats.ready_warp_count =
        ready_before;
    g_rtcore_replay_scoreboard_result_handoff_stats.delivered_count +=
        delivered;
    g_rtcore_replay_scoreboard_result_handoff_stats.blocked_count += blocked;
    if (ready_before >
        g_rtcore_replay_scoreboard_result_handoff_stats.max_ready_warp_count) {
        g_rtcore_replay_scoreboard_result_handoff_stats.max_ready_warp_count =
            ready_before;
    }
    if (blocked >
        g_rtcore_replay_scoreboard_result_handoff_stats.max_blocked_count) {
        g_rtcore_replay_scoreboard_result_handoff_stats.max_blocked_count =
            blocked;
    }

    if (ready_before > 0 || delivered > 0) {
        printf("GPGPU-Sim RTCORE_REPLAY_SCOREBOARD_RESULT_HANDOFF "
               "owner_hw_sid=%u service_cycle=%llu "
               "scoreboard_result_handoff_budget=%u "
               "scoreboard_handoff_ready_warp_count=%u "
               "scoreboard_handoff_delivered_count=%u "
               "scoreboard_handoff_blocked_count=%u\n",
               owner_hw_sid, service_cycle,
               rtcore_replay_scoreboard_result_handoff_budget_config(),
               ready_before, delivered, blocked);
        fflush(stdout);
    }
    return delivered;
}

static void rtcore_record_replay_lane_admission_entry(
    const rtcore_replay_lane_request &request)
{
    if (!rtcore_replay_warp_completion_entry_enabled() ||
        !rtcore_replay_request_has_warp_completion_entry_metadata(request)) {
        return;
    }

    const rtcore_replay_warp_completion_entry_key key =
        rtcore_make_replay_warp_completion_entry_key(request);
    rtcore_replay_warp_completion_entry_state &state =
        g_rtcore_replay_warp_completion_entries[key];
    if (!state.valid) {
        state.valid = true;
        state.key = key;
    }
    state.admitted_lane_mask |= 1u << request.lane_id;
    rtcore_update_replay_warp_completion_entry_state(&state);
}

static void rtcore_record_replay_lane_completion_entry(
    const rtcore_replay_lane_request &request)
{
    if (!rtcore_replay_warp_completion_entry_enabled() ||
        !rtcore_replay_request_has_warp_completion_entry_metadata(request)) {
        return;
    }

    const rtcore_replay_warp_completion_entry_key key =
        rtcore_make_replay_warp_completion_entry_key(request);
    rtcore_replay_warp_completion_entry_state &state =
        g_rtcore_replay_warp_completion_entries[key];
    if (!state.valid) {
        state.valid = true;
        state.key = key;
    }
    state.admitted_lane_mask |= 1u << request.lane_id;
    state.completed_lane_mask |= 1u << request.lane_id;
    state.result_valid_mask |= 1u << request.lane_id;
    state.result_data_slot[request.lane_id] =
        rtcore_make_replay_result_data_slot(request);
    state.lane_status[request.lane_id] =
        rtcore_make_replay_lane_status(request);
    const unsigned lane_mask = 1u << request.lane_id;
    const rtcore_replay_continuation_packet_lane_fact packet_fact =
        rtcore_make_replay_continuation_packet_lane_fact(request);
    state.packet_schema_version =
        RTCORE_REPLAY_CONTINUATION_PACKET_SCHEMA_VERSION;
    state.inline_payload_base_word = 16;
    state.max_inline_attribute_words = 4;
    state.lane_completion_valid_mask |= lane_mask;
    state.context_layout_version[request.lane_id] =
        packet_fact.context_layout_version;
    state.context_valid_flags[request.lane_id] =
        packet_fact.context_valid_flags;
    state.pipeline_profile_id[request.lane_id] =
        packet_fact.pipeline_profile_id;
    state.bvh_format_profile_id[request.lane_id] =
        packet_fact.bvh_format_profile_id;
    if (packet_fact.context_profile_valid) {
        state.context_profile_valid_mask |= lane_mask;
    }
    if (packet_fact.terminal) {
        state.terminal_lane_mask |= lane_mask;
    }
    if (packet_fact.continuation) {
        state.continuation_lane_mask |= lane_mask;
    }
    if (packet_fact.unsupported) {
        state.unsupported_reason_mask |= lane_mask;
    }
    if (packet_fact.selector_valid) {
        state.handoff_selector_valid_mask |= lane_mask;
    }
    if (packet_fact.candidate_valid) {
        state.handoff_candidate_valid_mask |= lane_mask;
    }
    if (packet_fact.software_return_valid) {
        state.handoff_software_return_valid_mask |= lane_mask;
    }
    if (rtcore_v04_live_handoff_publication_enabled()) {
        if (!rtcore_v04_live_publication_matches_current_submit(request) ||
            !request.v04_live_publication_committed ||
            request.v04_live_publication_reason != packet_fact.reason) {
            rtcore_fail_v04_live_handoff_publication(
                request, "completion_record_before_publication_commit",
                request.v04_live_publication_word_mask,
                request.v04_live_publication_pending_chunk_mask);
        }
        state.v04_shadow_boundary_image_valid_mask |= lane_mask;
    } else {
        state.v04_shadow_boundary_image_valid_mask &= ~lane_mask;
    }
    state.lane_completion_reason[request.lane_id] = packet_fact.reason;
    state.lane_continuation_depth[request.lane_id] =
        packet_fact.continuation_depth;
    for (unsigned word = 0; word < 32; ++word) {
        state.handoff_words[request.lane_id][word] =
            packet_fact.handoff_words[word];
        state.v04_shadow_handoff_words[request.lane_id][word] =
            rtcore_v04_live_handoff_publication_enabled()
                ? request.v04_live_publication_words[word]
                : 0;
    }
    rtcore_update_replay_warp_completion_entry_state(&state);
    if (state.all_active_lanes_complete &&
        !state.all_active_lanes_complete_logged) {
        state.all_active_lanes_complete_logged = true;
        rtcore_log_replay_warp_completion_entry(state);
    }
}

static bool rtcore_record_replay_lane_completion_entry(unsigned thread_uid)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }

    rtcore_record_replay_lane_completion_entry(it->second);
    return true;
}

static void rtcore_count_replay_model_lane_requests(
    unsigned owner_hw_sid, unsigned *admitted_lane_requests,
    unsigned *completed_lane_requests)
{
    unsigned admitted = 0;
    unsigned completed = 0;
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (!request.valid || request.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        admitted++;
        if (request.state == RTCORE_REPLAY_COMPLETED ||
            request.state == RTCORE_REPLAY_FINAL_WAIT_RETIRE) {
            completed++;
        }
    }
    if (admitted_lane_requests) {
        *admitted_lane_requests = admitted;
    }
    if (completed_lane_requests) {
        *completed_lane_requests = completed;
    }
}

static unsigned
rtcore_count_replay_model_completed_warp_aggregations(unsigned owner_hw_sid)
{
    unsigned completed_warps = 0;
    for (std::map<rtcore_replay_warp_completion_entry_key,
                  rtcore_replay_warp_completion_entry_state>::const_iterator
             it = g_rtcore_replay_warp_completion_entries.begin();
         it != g_rtcore_replay_warp_completion_entries.end(); ++it) {
        if (!it->second.valid || it->second.key.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        if (it->second.all_active_lanes_complete) {
            completed_warps++;
        }
    }
    return completed_warps;
}

static unsigned rtcore_replay_node_unit_busy_cycles()
{
    const unsigned node_latency = rtcore_replay_node_test_latency_config();
    const unsigned node_extra_cycles = node_latency > 0 ? node_latency - 1 : 0;
    return g_rtcore_replay_unit_arbitration_stats.node_unit_issued +
           g_rtcore_replay_unit_latency_gate_stats.node_gate_armed_count *
               node_extra_cycles;
}

static unsigned rtcore_replay_primitive_unit_busy_cycles()
{
    const unsigned primitive_latency =
        rtcore_replay_primitive_test_latency_config();
    const unsigned primitive_extra_cycles =
        primitive_latency > 0 ? primitive_latency - 1 : 0;
    return g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued +
           g_rtcore_replay_unit_latency_gate_stats.primitive_gate_armed_count *
               primitive_extra_cycles;
}

static unsigned rtcore_replay_stack_unit_busy_cycles()
{
    const unsigned stack_latency = rtcore_replay_stack_latency_cycles_config();
    const unsigned stack_extra_cycles =
        stack_latency > 0 ? stack_latency - 1 : 0;
    return g_rtcore_replay_unit_arbitration_stats.stack_unit_issued +
           g_rtcore_replay_unit_latency_gate_stats.stack_gate_armed_count *
               stack_extra_cycles;
}

static unsigned rtcore_replay_warp_completion_ingress_busy_cycles()
{
    return g_rtcore_replay_unit_arbitration_stats
        .warp_completion_ingress_issued;
}

static unsigned rtcore_replay_total_unit_busy_cycles()
{
    return rtcore_replay_node_unit_busy_cycles() +
           rtcore_replay_primitive_unit_busy_cycles() +
           rtcore_replay_stack_unit_busy_cycles() +
           rtcore_replay_warp_completion_ingress_busy_cycles();
}

static unsigned rtcore_replay_total_unit_issued()
{
    return g_rtcore_replay_unit_arbitration_stats.node_unit_issued +
           g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued +
           g_rtcore_replay_unit_arbitration_stats.stack_unit_issued +
           g_rtcore_replay_unit_arbitration_stats
               .warp_completion_ingress_issued;
}

static const char *rtcore_replay_dominant_issue_unit(unsigned *issue_count)
{
    const unsigned node_issued =
        g_rtcore_replay_unit_arbitration_stats.node_unit_issued;
    const unsigned primitive_issued =
        g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued;
    const unsigned stack_issued =
        g_rtcore_replay_unit_arbitration_stats.stack_unit_issued;
    const unsigned completion_issued =
        g_rtcore_replay_unit_arbitration_stats
            .warp_completion_ingress_issued;

    const char *unit = "none";
    unsigned count = 0;
    if (node_issued > count) {
        unit = "node";
        count = node_issued;
    }
    if (primitive_issued > count) {
        unit = "primitive";
        count = primitive_issued;
    }
    if (stack_issued > count) {
        unit = "stack";
        count = stack_issued;
    }
    if (completion_issued > count) {
        unit = "warp_completion_ingress";
        count = completion_issued;
    }
    if (issue_count) {
        *issue_count = count;
    }
    return unit;
}

static const char *rtcore_replay_dominant_busy_unit(unsigned *busy_cycles)
{
    const unsigned node_cycles = rtcore_replay_node_unit_busy_cycles();
    const unsigned primitive_cycles = rtcore_replay_primitive_unit_busy_cycles();
    const unsigned stack_cycles = rtcore_replay_stack_unit_busy_cycles();
    const unsigned completion_cycles =
        rtcore_replay_warp_completion_ingress_busy_cycles();

    const char *unit = "none";
    unsigned cycles = 0;
    if (node_cycles > cycles) {
        unit = "node";
        cycles = node_cycles;
    }
    if (primitive_cycles > cycles) {
        unit = "primitive";
        cycles = primitive_cycles;
    }
    if (stack_cycles > cycles) {
        unit = "stack";
        cycles = stack_cycles;
    }
    if (completion_cycles > cycles) {
        unit = "warp_completion_ingress";
        cycles = completion_cycles;
    }
    if (busy_cycles) {
        *busy_cycles = cycles;
    }
    return unit;
}

static unsigned rtcore_replay_memory_pressure_cycles()
{
    unsigned cycles = 0;
    if (g_rtcore_replay_memory_wake_latency_gate_stats.max_blocked_cycles >
        cycles) {
        cycles =
            g_rtcore_replay_memory_wake_latency_gate_stats.max_blocked_cycles;
    }
    if (g_rtcore_replay_memory_contention_gate_stats.max_contention_cycles >
        cycles) {
        cycles =
            g_rtcore_replay_memory_contention_gate_stats.max_contention_cycles;
    }
    if (g_rtcore_replay_memory_contention_gate_stats.max_queue_delay_cycles >
        cycles) {
        cycles =
            g_rtcore_replay_memory_contention_gate_stats.max_queue_delay_cycles;
    }
    return cycles;
}

static const char *rtcore_replay_dominant_pressure_source(
    const char *dominant_busy_unit, unsigned dominant_busy_cycles,
    unsigned memory_pressure_cycles, unsigned *pressure_cycles)
{
    const char *source = "none";
    unsigned cycles = 0;
    if (dominant_busy_cycles > cycles) {
        source = dominant_busy_unit;
        cycles = dominant_busy_cycles;
    }
    if (memory_pressure_cycles > cycles) {
        source = "memory";
        cycles = memory_pressure_cycles;
    }
    if (pressure_cycles) {
        *pressure_cycles = cycles;
    }
    return source;
}

static bool rtcore_should_log_replay_model_summary_stats(
    unsigned owner_hw_sid,
    const rtcore_replay_model_summary_progress_snapshot &snapshot)
{
    if (!snapshot.valid) {
        return false;
    }

    rtcore_replay_model_summary_progress_snapshot &last_snapshot =
        g_rtcore_replay_model_summary_progress_snapshots[owner_hw_sid];
    const bool data_path_pressure_changed =
        last_snapshot.data_path_lane_request_state_identity_accesses !=
            snapshot.data_path_lane_request_state_identity_accesses ||
        last_snapshot.data_path_request_state_accesses !=
            snapshot.data_path_request_state_accesses ||
        last_snapshot.data_path_max_lane_request_state_entries !=
            snapshot.data_path_max_lane_request_state_entries;
    const bool changed =
        !last_snapshot.valid ||
        last_snapshot.service_ticks_progressed !=
            snapshot.service_ticks_progressed ||
        last_snapshot.admitted_lane_requests !=
            snapshot.admitted_lane_requests ||
        last_snapshot.completed_lane_requests !=
            snapshot.completed_lane_requests ||
        last_snapshot.total_unit_issued != snapshot.total_unit_issued ||
        last_snapshot.total_unit_busy_cycles !=
            snapshot.total_unit_busy_cycles ||
        last_snapshot.memory_blocked_events !=
            snapshot.memory_blocked_events ||
        last_snapshot.memory_wake_blocked_count !=
            snapshot.memory_wake_blocked_count ||
        last_snapshot.memory_wake_max_blocked_cycles !=
            snapshot.memory_wake_max_blocked_cycles ||
        last_snapshot.memory_contention_gate_armed_count !=
            snapshot.memory_contention_gate_armed_count ||
        last_snapshot.memory_contention_gate_blocked_count !=
            snapshot.memory_contention_gate_blocked_count ||
        last_snapshot.memory_contention_gate_woken_count !=
            snapshot.memory_contention_gate_woken_count ||
        last_snapshot.memory_contention_max_contention_cycles !=
            snapshot.memory_contention_max_contention_cycles ||
        last_snapshot.memory_contention_max_queue_delay_cycles !=
            snapshot.memory_contention_max_queue_delay_cycles ||
        last_snapshot.memory_contention_capacity_blocked_count !=
            snapshot.memory_contention_capacity_blocked_count ||
        last_snapshot.lane_request_state_capacity_blocked_count !=
            snapshot.lane_request_state_capacity_blocked_count ||
        last_snapshot.lane_state_init_bandwidth_blocked_count !=
            snapshot.lane_state_init_bandwidth_blocked_count ||
        last_snapshot.lane_request_state_capacity_max_occupancy !=
            snapshot.lane_request_state_capacity_max_occupancy ||
        last_snapshot.lane_request_state_capacity_max_pending_admissions !=
            snapshot.lane_request_state_capacity_max_pending_admissions ||
        last_snapshot.warp_aggregated_completion_count !=
            snapshot.warp_aggregated_completion_count ||
        last_snapshot.scoreboard_handoff_ready_warp_count !=
            snapshot.scoreboard_handoff_ready_warp_count ||
        last_snapshot.scoreboard_handoff_delivered_count !=
            snapshot.scoreboard_handoff_delivered_count ||
        last_snapshot.scoreboard_handoff_blocked_count !=
            snapshot.scoreboard_handoff_blocked_count ||
        last_snapshot.scoreboard_handoff_max_blocked_count !=
            snapshot.scoreboard_handoff_max_blocked_count ||
        data_path_pressure_changed;
    if (!changed) {
        return false;
    }

    last_snapshot = snapshot;
    return true;
}

static unsigned long long rtcore_v02_lsu_align_32b(unsigned long long address)
{
    const unsigned long long mask =
        static_cast<unsigned long long>(RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES) -
        1ULL;
    return address & ~mask;
}

static rtcore_v02_lsu_memory_transaction_identity
rtcore_replay_make_memory_unit_request_descriptor_identity(
    const rtcore_replay_lane_request &request, unsigned memory_op_seq,
    unsigned chunk_id, unsigned chunk_count, unsigned access_kind,
    unsigned long long byte_address, unsigned long long issue_cycle)
{
    rtcore_v02_lsu_memory_transaction_identity identity;
    identity.response_target = RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE;
    identity.owner_hw_sid = request.owner_hw_sid;
    identity.rt_request_id = request.thread_uid;
    identity.lane_id = request.lane_id;
    identity.memory_op_seq = memory_op_seq;
    identity.chunk_id = chunk_id;
    identity.chunk_count = chunk_count;
    identity.access_kind = access_kind;
    identity.aligned_32b_addr = rtcore_v02_lsu_align_32b(byte_address);
    identity.is_write = false;
    identity.issue_cycle = issue_cycle;
    return identity;
}

static rtcore_v02_lsu_merge_key rtcore_v02_lsu_merge_key_for_identity(
    const rtcore_v02_lsu_memory_transaction_identity &identity)
{
    rtcore_v02_lsu_merge_key key;
    key.owner_hw_sid = identity.owner_hw_sid;
    key.issue_cycle = identity.issue_cycle;
    key.aligned_32b_addr = identity.aligned_32b_addr;
    key.is_write = identity.is_write;
    return key;
}

static void rtcore_maybe_log_replay_memory_unit_request_descriptor_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_memory_unit_request_descriptor_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_memory_unit_request_descriptor_stats_logs_emitted >= 256) {
        return;
    }
    g_rtcore_replay_memory_unit_request_descriptor_stats_logs_emitted++;

    const rtcore_replay_memory_unit_request_descriptor_stats &stats =
        g_rtcore_replay_memory_unit_request_descriptor_stats;
    printf("GPGPU-Sim RTCORE_REPLAY_MEMORY_UNIT_REQUEST_DESCRIPTOR_STATS "
           "owner_hw_sid=%u path_active=0 request_granule_bytes=%u "
           "descriptor_identity_fields=%u actual_fetch_event_count=%u "
           "node_fetch_event_count=%u primitive_fetch_event_count=%u "
           "descriptor_count=%u node_descriptor_count=%u "
           "primitive_descriptor_count=%u total_chunk_count=%u "
           "max_chunk_count=%u response_target_rtcore_count=%u "
           "synthetic_memory_latency_added=0 unique_transaction_count=%u "
           "same_cycle_merge_candidate_count=%u\n",
           owner_hw_sid, RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES,
           RTCORE_V02_LSU_TRANSACTION_IDENTITY_FIELD_COUNT,
           stats.actual_fetch_event_count, stats.node_fetch_event_count,
           stats.primitive_fetch_event_count, stats.descriptor_count,
           stats.node_descriptor_count, stats.primitive_descriptor_count,
           stats.total_chunk_count, stats.max_chunk_count,
           stats.response_target_rtcore_count, stats.unique_transaction_count,
           stats.same_cycle_merge_candidate_count);
    fflush(stdout);
}

static void rtcore_maybe_log_memory_unit_response_wait_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_memory_unit_response_wait_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_v02_lsu_response_wait_stats_logs_emitted >=
        rtcore_memory_unit_response_wait_stats_log_limit()) {
        return;
    }
    g_rtcore_v02_lsu_response_wait_stats_logs_emitted++;

    const rtcore_v02_lsu_response_wait_stats &stats =
        g_rtcore_v02_lsu_response_wait_stats;
    printf("GPGPU-Sim RTCORE_REPLAY_MEMORY_UNIT_RESPONSE_WAIT_STATS "
           "owner_hw_sid=%u path_active=1 gate_enabled=%u "
           "armed_count=%u blocked_count=%u woken_count=%u "
           "stack_load_armed_count=%u stack_store_armed_count=%u "
           "stack_load_woken_count=%u stack_store_woken_count=%u "
           "completed_event_count=%u response_chunk_count=%u "
           "duplicate_response_count=%u stale_response_count=%u "
           "pending_request_count=%u max_pending_request_count=%u "
           "max_chunk_count=%u synthetic_memory_latency_added=0 "
           "response_target_rtcore_count=%u\n",
           owner_hw_sid,
           rtcore_memory_unit_response_wait_enabled() ? 1u : 0u,
           stats.gate_armed_count, stats.gate_blocked_count,
           stats.gate_woken_count, stats.stack_load_armed_count,
           stats.stack_store_armed_count, stats.stack_load_woken_count,
           stats.stack_store_woken_count, stats.completed_event_count,
           stats.response_chunk_count, stats.duplicate_response_count,
           stats.stale_response_count, stats.pending_request_count,
           stats.max_pending_request_count, stats.max_chunk_count,
           stats.response_target_rtcore_count);
    fflush(stdout);
}

static void rtcore_record_replay_memory_unit_request_descriptor(
    const rtcore_replay_lane_request &request,
    const rtcore_compact_trace_event &event, unsigned long long service_cycle,
    unsigned chunk_count)
{
    const bool descriptor_log_enabled =
        rtcore_replay_memory_unit_request_descriptor_log_enabled();
    const bool sideband_offer_enabled =
        rtcore_replay_memory_unit_request_offer_enabled();
    if (!descriptor_log_enabled && !sideband_offer_enabled) {
        return;
    }

    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    unsigned access_kind = RTCORE_V02_LSU_ACCESS_KIND_COUNT;
    bool is_node_fetch = false;
    if (event_type == RTCORE_TRACE_NODE_FETCH) {
        access_kind = RTCORE_V02_LSU_ACCESS_NODE_FETCH;
        is_node_fetch = true;
    } else if (event_type == RTCORE_TRACE_PRIMITIVE_FETCH) {
        access_kind = RTCORE_V02_LSU_ACCESS_PRIMITIVE_FETCH;
    } else {
        return;
    }

    if (chunk_count == 0) {
        chunk_count = 1;
    }
    const unsigned long long base_address = event.address_or_ref;
    if (descriptor_log_enabled) {
        g_rtcore_replay_memory_unit_request_descriptor_stats
            .actual_fetch_event_count++;
        if (is_node_fetch) {
            g_rtcore_replay_memory_unit_request_descriptor_stats
                .node_fetch_event_count++;
            g_rtcore_replay_memory_unit_request_descriptor_stats
                .node_descriptor_count += chunk_count;
        } else {
            g_rtcore_replay_memory_unit_request_descriptor_stats
                .primitive_fetch_event_count++;
            g_rtcore_replay_memory_unit_request_descriptor_stats
                .primitive_descriptor_count += chunk_count;
        }
        g_rtcore_replay_memory_unit_request_descriptor_stats.descriptor_count +=
            chunk_count;
        g_rtcore_replay_memory_unit_request_descriptor_stats.total_chunk_count +=
            chunk_count;
        if (chunk_count >
            g_rtcore_replay_memory_unit_request_descriptor_stats.max_chunk_count) {
            g_rtcore_replay_memory_unit_request_descriptor_stats.max_chunk_count =
                chunk_count;
        }
    }

    for (unsigned chunk_id = 0; chunk_id < chunk_count; ++chunk_id) {
        const unsigned long long chunk_address =
            base_address +
            static_cast<unsigned long long>(
                chunk_id * RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES);
        const rtcore_v02_lsu_memory_transaction_identity identity =
            rtcore_replay_make_memory_unit_request_descriptor_identity(
                request, request.next_event_index, chunk_id, chunk_count,
                access_kind, chunk_address, service_cycle);
        if (sideband_offer_enabled) {
            rtcore_memory_unit_request_snapshot snapshot = {};
            snapshot.valid = true;
            snapshot.response_target = identity.response_target;
            snapshot.owner_hw_sid = identity.owner_hw_sid;
            snapshot.rt_request_id = identity.rt_request_id;
            snapshot.lane_id = identity.lane_id;
            snapshot.memory_op_seq = identity.memory_op_seq;
            snapshot.chunk_id = identity.chunk_id;
            snapshot.chunk_count = identity.chunk_count;
            snapshot.access_kind = identity.access_kind;
            snapshot.aligned_32b_addr = identity.aligned_32b_addr;
            snapshot.is_write = identity.is_write;
            snapshot.issue_cycle = identity.issue_cycle;
            g_rtcore_memory_unit_request_snapshots_by_owner
                [identity.owner_hw_sid]
                    .push_back(snapshot);
        }
        if (descriptor_log_enabled) {
            if (identity.response_target ==
                RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE) {
                g_rtcore_replay_memory_unit_request_descriptor_stats
                    .response_target_rtcore_count++;
            }
            const rtcore_v02_lsu_merge_key key =
                rtcore_v02_lsu_merge_key_for_identity(identity);
            unsigned &merge_count =
                g_rtcore_replay_memory_unit_request_descriptor_merge_counts[key];
            if (merge_count == 0) {
                g_rtcore_replay_memory_unit_request_descriptor_stats
                    .unique_transaction_count++;
            } else {
                g_rtcore_replay_memory_unit_request_descriptor_stats
                    .same_cycle_merge_candidate_count++;
            }
            merge_count++;
        }
    }

    if (descriptor_log_enabled) {
        rtcore_maybe_log_replay_memory_unit_request_descriptor_stats(
            request.owner_hw_sid);
    }
}

static unsigned rtcore_v02_lsu_stack_sideband_chunk_count(
    const rtcore_compact_trace_event &event)
{
    const unsigned count = rtcore_unpack_compact_trace_count(event);
    return count == 0 ? 1 : count;
}

static unsigned long long rtcore_v02_lsu_stack_sideband_base_address(
    const rtcore_replay_lane_request &request,
    const rtcore_compact_trace_event &event, unsigned event_index)
{
    if (event.address_or_ref != 0) {
        return event.address_or_ref;
    }
    return RTCORE_V02_LSU_SYNTHETIC_STACK_BASE +
           static_cast<unsigned long long>(request.owner_hw_sid) * 0x100000ull +
           static_cast<unsigned long long>(request.thread_uid) * 0x1000ull +
           static_cast<unsigned long long>(event_index) *
               RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES;
}

static bool rtcore_v02_lsu_stack_sideband_access_for_event(
    const rtcore_compact_trace_event &event, unsigned *access_kind,
    bool *is_write)
{
    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    if (event_type == RTCORE_TRACE_STACK_PUSH) {
        if (access_kind) {
            *access_kind = RTCORE_V02_LSU_ACCESS_STACK_STORE;
        }
        if (is_write) {
            *is_write = true;
        }
        return true;
    }
    if (event_type == RTCORE_TRACE_STACK_POP) {
        if (access_kind) {
            *access_kind = RTCORE_V02_LSU_ACCESS_STACK_LOAD;
        }
        if (is_write) {
            *is_write = false;
        }
        return true;
    }
    return false;
}

static bool rtcore_memory_unit_response_wait_manages_stack_load_event(
    const rtcore_compact_trace_event &event)
{
    unsigned access_kind = RTCORE_V02_LSU_ACCESS_KIND_COUNT;
    return rtcore_memory_unit_response_wait_enabled() &&
           rtcore_replay_memory_unit_request_offer_enabled() &&
           rtcore_replay_memory_unit_l1d_client_enabled() &&
           rtcore_v02_lsu_stack_sideband_enabled() &&
           rtcore_v02_lsu_stack_sideband_access_for_event(event, &access_kind,
                                                          NULL) &&
           access_kind == RTCORE_V02_LSU_ACCESS_STACK_LOAD;
}

static bool rtcore_maybe_enqueue_v02_lsu_stack_sideband(
    const rtcore_replay_lane_request &request, unsigned long long service_cycle)
{
    if (!request.valid || !rtcore_v02_lsu_stack_sideband_enabled() ||
        !rtcore_replay_memory_unit_request_offer_enabled()) {
        return false;
    }
    if (request.next_event_index >= request.events.size()) {
        return false;
    }

    const rtcore_compact_trace_event &event =
        request.events[request.next_event_index];
    unsigned access_kind = RTCORE_V02_LSU_ACCESS_KIND_COUNT;
    bool is_write = false;
    if (!rtcore_v02_lsu_stack_sideband_access_for_event(event, &access_kind,
                                                        &is_write)) {
        return false;
    }

    const unsigned chunk_count =
        rtcore_v02_lsu_stack_sideband_chunk_count(event);
    const unsigned long long base_address =
        rtcore_v02_lsu_stack_sideband_base_address(
            request, event, request.next_event_index);
    for (unsigned chunk_id = 0; chunk_id < chunk_count; ++chunk_id) {
        const unsigned long long chunk_address =
            base_address +
            static_cast<unsigned long long>(
                chunk_id * RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES);
        rtcore_memory_unit_request_snapshot snapshot = {};
        snapshot.valid = true;
        snapshot.response_target = RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE;
        snapshot.owner_hw_sid = request.owner_hw_sid;
        snapshot.rt_request_id = request.thread_uid;
        snapshot.lane_id = request.lane_id;
        snapshot.memory_op_seq = request.next_event_index;
        snapshot.chunk_id = chunk_id;
        snapshot.chunk_count = chunk_count;
        snapshot.access_kind = access_kind;
        snapshot.aligned_32b_addr = rtcore_v02_lsu_align_32b(chunk_address);
        snapshot.is_write = is_write;
        snapshot.issue_cycle = service_cycle;
        g_rtcore_memory_unit_request_snapshots_by_owner[request.owner_hw_sid]
            .push_back(snapshot);
    }
    return true;
}

static void rtcore_maybe_log_replay_model_summary_stats(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    if (!rtcore_replay_model_summary_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_model_summary_stats_logs_emitted >=
        rtcore_replay_model_summary_stats_log_limit()) {
        return;
    }
    if (g_rtcore_replay_service_tick_stats.ticks_progressed == 0) {
        return;
    }

    unsigned admitted_lane_requests = 0;
    unsigned completed_lane_requests = 0;
    rtcore_count_replay_model_lane_requests(
        owner_hw_sid, &admitted_lane_requests, &completed_lane_requests);
    if (admitted_lane_requests == 0) {
        return;
    }

    const rtcore_replay_issue_budget budget =
        rtcore_replay_issue_budget_config();
    const unsigned warp_aggregated_completion_count =
        rtcore_count_replay_model_completed_warp_aggregations(owner_hw_sid);
    const unsigned scoreboard_handoff_ready_warp_count =
        g_rtcore_replay_scoreboard_result_handoff_stats.ready_warp_count;
    const unsigned scoreboard_handoff_delivered_count =
        g_rtcore_replay_scoreboard_result_handoff_stats.delivered_count;
    const unsigned scoreboard_handoff_blocked_count =
        g_rtcore_replay_scoreboard_result_handoff_stats.blocked_count;
    const unsigned scoreboard_handoff_max_blocked_count =
        g_rtcore_replay_scoreboard_result_handoff_stats.max_blocked_count;
    const unsigned memory_blocked_events =
        g_rtcore_replay_memory_wake_latency_gate_stats.gate_blocked_count +
        g_rtcore_replay_memory_contention_gate_stats.gate_blocked_count;
    const unsigned total_unit_issued = rtcore_replay_total_unit_issued();
    const unsigned node_unit_busy_cycles =
        rtcore_replay_node_unit_busy_cycles();
    const unsigned primitive_unit_busy_cycles =
        rtcore_replay_primitive_unit_busy_cycles();
    const unsigned stack_unit_busy_cycles =
        rtcore_replay_stack_unit_busy_cycles();
    const unsigned warp_completion_ingress_busy_cycles =
        rtcore_replay_warp_completion_ingress_busy_cycles();
    const unsigned total_unit_busy_cycles =
        rtcore_replay_total_unit_busy_cycles();
    const unsigned memory_wake_blocked_count =
        g_rtcore_replay_memory_wake_latency_gate_stats.gate_blocked_count;
    const unsigned memory_wake_max_blocked_cycles =
        g_rtcore_replay_memory_wake_latency_gate_stats.max_blocked_cycles;
    const unsigned memory_contention_gate_armed_count =
        g_rtcore_replay_memory_contention_gate_stats.gate_armed_count;
    const unsigned memory_contention_gate_blocked_count =
        g_rtcore_replay_memory_contention_gate_stats.gate_blocked_count;
    const unsigned memory_contention_gate_woken_count =
        g_rtcore_replay_memory_contention_gate_stats.gate_woken_count;
    const unsigned memory_contention_max_contention_cycles =
        g_rtcore_replay_memory_contention_gate_stats.max_contention_cycles;
    const unsigned memory_contention_max_queue_delay_cycles =
        g_rtcore_replay_memory_contention_gate_stats.max_queue_delay_cycles;
    const unsigned memory_contention_capacity_blocked_count =
        g_rtcore_replay_memory_contention_gate_stats.capacity_blocked_count;
    const unsigned lane_request_state_capacity_blocked_count =
        g_rtcore_replay_lane_request_state_capacity_gate_stats.blocked_count;
    const unsigned lane_state_init_bandwidth_blocked_count =
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .lane_state_init_bandwidth_blocked_count;
    const unsigned lane_request_state_capacity_max_occupancy =
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_lane_request_state_occupancy;
    const unsigned lane_request_state_capacity_max_pending_admissions =
        g_rtcore_replay_lane_request_state_capacity_gate_stats
            .max_pending_admissions;
    rtcore_replay_data_path_access_snapshot data_path_access =
        rtcore_get_replay_data_path_access_snapshot();
    const unsigned data_path_max_lane_request_state_entries =
        g_rtcore_replay_data_path_access_stats.max_lane_request_state_entries;
    rtcore_replay_model_summary_progress_snapshot progress_snapshot = {};
    progress_snapshot.valid = true;
    progress_snapshot.service_ticks_progressed =
        g_rtcore_replay_service_tick_stats.ticks_progressed;
    progress_snapshot.admitted_lane_requests = admitted_lane_requests;
    progress_snapshot.completed_lane_requests = completed_lane_requests;
    progress_snapshot.total_unit_issued = total_unit_issued;
    progress_snapshot.total_unit_busy_cycles = total_unit_busy_cycles;
    progress_snapshot.memory_blocked_events = memory_blocked_events;
    progress_snapshot.memory_wake_blocked_count = memory_wake_blocked_count;
    progress_snapshot.memory_wake_max_blocked_cycles =
        memory_wake_max_blocked_cycles;
    progress_snapshot.memory_contention_gate_armed_count =
        memory_contention_gate_armed_count;
    progress_snapshot.memory_contention_gate_blocked_count =
        memory_contention_gate_blocked_count;
    progress_snapshot.memory_contention_gate_woken_count =
        memory_contention_gate_woken_count;
    progress_snapshot.memory_contention_max_contention_cycles =
        memory_contention_max_contention_cycles;
    progress_snapshot.memory_contention_max_queue_delay_cycles =
        memory_contention_max_queue_delay_cycles;
    progress_snapshot.memory_contention_capacity_blocked_count =
        memory_contention_capacity_blocked_count;
    progress_snapshot.lane_request_state_capacity_blocked_count =
        lane_request_state_capacity_blocked_count;
    progress_snapshot.lane_state_init_bandwidth_blocked_count =
        lane_state_init_bandwidth_blocked_count;
    progress_snapshot.lane_request_state_capacity_max_occupancy =
        lane_request_state_capacity_max_occupancy;
    progress_snapshot.lane_request_state_capacity_max_pending_admissions =
        lane_request_state_capacity_max_pending_admissions;
    progress_snapshot.warp_aggregated_completion_count =
        warp_aggregated_completion_count;
    progress_snapshot.scoreboard_handoff_ready_warp_count =
        scoreboard_handoff_ready_warp_count;
    progress_snapshot.scoreboard_handoff_delivered_count =
        scoreboard_handoff_delivered_count;
    progress_snapshot.scoreboard_handoff_blocked_count =
        scoreboard_handoff_blocked_count;
    progress_snapshot.scoreboard_handoff_max_blocked_count =
        scoreboard_handoff_max_blocked_count;
    progress_snapshot.data_path_lane_request_state_identity_accesses =
        data_path_access.lane_request_state_identity_accesses;
    progress_snapshot.data_path_request_state_accesses =
        data_path_access.request_state_accesses;
    progress_snapshot.data_path_max_lane_request_state_entries =
        data_path_max_lane_request_state_entries;
    progress_snapshot.max_observed_ready_cycle = service_cycle;
    if (!rtcore_should_log_replay_model_summary_stats(owner_hw_sid,
                                                      progress_snapshot)) {
        return;
    }

    g_rtcore_replay_model_summary_stats_logs_emitted++;
    unsigned dominant_busy_cycles = 0;
    const char *dominant_busy_unit =
        rtcore_replay_dominant_busy_unit(&dominant_busy_cycles);
    unsigned dominant_issue_count = 0;
    const char *dominant_issue_unit =
        rtcore_replay_dominant_issue_unit(&dominant_issue_count);
    const unsigned memory_pressure_cycles =
        rtcore_replay_memory_pressure_cycles();
    unsigned dominant_pressure_cycles = 0;
    const char *dominant_pressure_source =
        rtcore_replay_dominant_pressure_source(
            dominant_busy_unit, dominant_busy_cycles, memory_pressure_cycles,
            &dominant_pressure_cycles);

    printf("GPGPU-Sim RTCORE_REPLAY_MODEL_SUMMARY_STATS "
           "owner_hw_sid=%u model_name=%s model_version=0.1 "
           "rt_core_scope=per_sm_logical_attached_rt_core rt_cores_per_sm=1 "
           "internal_request_granularity=per_active_lane_rt_request "
           "external_wakeup_granularity=warp_aggregated_completion "
           "functional_oracle=VulkanRayTracing::traceRay "
           "claims_new_hardware_bvh_engine=0 "
           "max_trace_events_per_lane=%u "
           "node_issue_budget=%u primitive_issue_budget=%u "
           "stack_issue_budget=%u warp_completion_ingress_budget=%u "
           "node_event_base_cycles=%u "
           "primitive_event_base_cycles=%u "
           "memory_contention_cache_lines_per_cycle=%u "
           "memory_outstanding_capacity=%u "
           "memory_outstanding_alloc_budget=%u "
           "service_ticks_progressed=%u admitted_lane_requests=%u "
           "completed_lane_requests=%u node_unit_issued=%u "
           "primitive_unit_issued=%u stack_unit_issued=%u "
           "warp_completion_ingress_issued=%u total_unit_issued=%u "
           "dominant_issue_unit=%s dominant_issue_count=%u "
           "node_unit_busy_cycles=%u primitive_unit_busy_cycles=%u "
           "stack_unit_busy_cycles=%u "
           "warp_completion_ingress_busy_cycles=%u "
           "total_unit_busy_cycles=%u "
           "dominant_busy_unit=%s dominant_busy_cycles=%u "
           "memory_blocked_events=%u "
           "memory_wake_blocked_count=%u "
           "memory_wake_max_blocked_cycles=%u "
           "memory_contention_gate_armed_count=%u "
           "memory_contention_gate_blocked_count=%u "
           "memory_contention_gate_woken_count=%u "
           "memory_contention_max_contention_cycles=%u "
           "memory_contention_max_queue_delay_cycles=%u "
           "memory_contention_capacity_blocked_count=%u "
           "lane_request_state_capacity_blocked_count=%u "
           "lane_state_init_bandwidth_blocked_count=%u "
           "lane_state_init_bandwidth_max_used_per_cycle=%u "
           "lane_request_state_capacity_max_occupancy=%u "
           "lane_request_state_capacity_max_pending_admissions=%u "
           "memory_pressure_cycles=%u "
           "dominant_pressure_source=%s dominant_pressure_cycles=%u "
           "warp_aggregated_completion_count=%u "
           "scoreboard_result_handoff_budget=%u "
           "scoreboard_handoff_ready_warp_count=%u "
           "scoreboard_handoff_delivered_count=%u "
           "scoreboard_handoff_blocked_count=%u "
           "scoreboard_handoff_max_blocked_count=%u "
           "rtcore_stall_completion_backpressure_blocked_count=%u "
           "rtcore_continuation_packet_count=%llu "
           "rtcore_continuation_lane_count=%llu "
           "rtcore_continuation_warp_wakeup_count=%llu "
           "rtcore_continuation_wait_cycles=%llu "
           "rtcore_modeled_resubmit_count=%llu "
           "rtcore_modeled_resubmit_lane_count=%llu "
           "rtcore_continuation_synthetic_boundary_count=%llu "
           "rtcore_continuation_oracle_anyhit_boundary_count=%llu "
           "rtcore_continuation_oracle_intersection_boundary_count=%llu "
           "rtcore_continuation_max_depth=%u "
           "data_path_lane_request_state_identity_accesses=%u "
           "data_path_request_state_accesses=%u "
           "data_path_max_lane_request_state_entries=%u "
           "memory_unit_contract_enabled=1 "
           "memory_unit_path_active=%u "
           "memory_unit_request_granule_bytes=%u "
           "memory_unit_handoff_window_global_memory_contract=1 "
           "memory_unit_same_cycle_32b_merge_policy=1 "
           "memory_unit_cross_cycle_mshr_merge_required=0 "
           "memory_unit_response_fanout_contract=1 "
           "memory_unit_synthetic_memory_latency_disabled_when_active=1 "
           "memory_unit_response_target_rtcore=1 "
           "max_observed_ready_cycle=%llu\n",
           owner_hw_sid, RTCORE_TRACE_REPLAY_MODEL_NAME,
           rtcore_compact_trace_events_per_lane_config(),
           budget.node_issue_budget, budget.primitive_issue_budget,
           budget.stack_issue_budget, budget.warp_completion_ingress_budget,
           rtcore_replay_node_test_latency_config(),
           rtcore_replay_primitive_test_latency_config(),
           rtcore_replay_memory_contention_cache_lines_per_cycle_config(),
           rtcore_replay_memory_outstanding_capacity_config(),
           rtcore_replay_memory_outstanding_alloc_budget_config(),
           g_rtcore_replay_service_tick_stats.ticks_progressed,
           admitted_lane_requests, completed_lane_requests,
           g_rtcore_replay_unit_arbitration_stats.node_unit_issued,
           g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued,
           g_rtcore_replay_unit_arbitration_stats.stack_unit_issued,
           g_rtcore_replay_unit_arbitration_stats
               .warp_completion_ingress_issued,
           total_unit_issued, dominant_issue_unit, dominant_issue_count,
           node_unit_busy_cycles, primitive_unit_busy_cycles,
           stack_unit_busy_cycles, warp_completion_ingress_busy_cycles,
           total_unit_busy_cycles, dominant_busy_unit, dominant_busy_cycles,
           memory_blocked_events, memory_wake_blocked_count,
           memory_wake_max_blocked_cycles,
           memory_contention_gate_armed_count,
           memory_contention_gate_blocked_count,
           memory_contention_gate_woken_count,
           memory_contention_max_contention_cycles,
           memory_contention_max_queue_delay_cycles,
           memory_contention_capacity_blocked_count,
           lane_request_state_capacity_blocked_count,
           lane_state_init_bandwidth_blocked_count,
           g_rtcore_replay_lane_request_state_capacity_gate_stats
               .max_lane_state_init_bandwidth_used_per_cycle,
           lane_request_state_capacity_max_occupancy,
           lane_request_state_capacity_max_pending_admissions,
           memory_pressure_cycles, dominant_pressure_source,
           dominant_pressure_cycles,
           warp_aggregated_completion_count,
           rtcore_replay_scoreboard_result_handoff_budget_config(),
           scoreboard_handoff_ready_warp_count,
           scoreboard_handoff_delivered_count,
           scoreboard_handoff_blocked_count,
           scoreboard_handoff_max_blocked_count,
           scoreboard_handoff_blocked_count,
           g_rtcore_continuation_stats.rtcore_continuation_packet_count,
           g_rtcore_continuation_stats.rtcore_continuation_lane_count,
           g_rtcore_continuation_stats.rtcore_continuation_warp_wakeup_count,
           g_rtcore_continuation_stats.rtcore_continuation_wait_cycles,
           g_rtcore_continuation_stats.rtcore_modeled_resubmit_count,
           g_rtcore_continuation_stats.rtcore_modeled_resubmit_lane_count,
           g_rtcore_continuation_stats
               .rtcore_continuation_synthetic_boundary_count,
           g_rtcore_continuation_stats
               .rtcore_continuation_oracle_anyhit_boundary_count,
           g_rtcore_continuation_stats
               .rtcore_continuation_oracle_intersection_boundary_count,
           g_rtcore_continuation_stats.rtcore_continuation_max_depth,
           data_path_access.lane_request_state_identity_accesses,
           data_path_access.request_state_accesses,
           data_path_max_lane_request_state_entries,
           rtcore_replay_memory_unit_path_active() ? 1u : 0u,
           RTCORE_REPLAY_MEMORY_REQUEST_GRANULE_BYTES,
           service_cycle);
    fflush(stdout);
}

static void rtcore_maybe_log_replay_overflow_summary_estimate_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_overflow_summary_estimate_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_overflow_summary_estimate_stats_logs_emitted >=
        rtcore_replay_overflow_summary_estimate_log_limit()) {
        return;
    }
    g_rtcore_replay_overflow_summary_estimate_stats_logs_emitted++;

    printf("GPGPU-Sim RTCORE_REPLAY_OVERFLOW_SUMMARY_ESTIMATE_STATS "
           "owner_hw_sid=%u overflow_summary_requests_consumed=%u "
           "estimated_overflow_node_events=%u "
           "estimated_overflow_primitive_events=%u "
           "estimated_overflow_stack_events=%u "
           "estimated_overflow_memory_wait_events=%u "
           "estimated_overflow_memory_bytes=%u "
           "estimated_overflow_completion_events=%u\n",
           owner_hw_sid,
           g_rtcore_replay_overflow_summary_estimate_stats
               .overflow_summary_requests_consumed,
           g_rtcore_replay_overflow_summary_estimate_stats
               .estimated_overflow_node_events,
           g_rtcore_replay_overflow_summary_estimate_stats
               .estimated_overflow_primitive_events,
           g_rtcore_replay_overflow_summary_estimate_stats
               .estimated_overflow_stack_events,
           g_rtcore_replay_overflow_summary_estimate_stats
               .estimated_overflow_memory_wait_events,
           g_rtcore_replay_overflow_summary_estimate_stats
               .estimated_overflow_memory_bytes,
           g_rtcore_replay_overflow_summary_estimate_stats
               .estimated_overflow_completion_events);
    fflush(stdout);
}

static void rtcore_record_replay_overflow_summary_estimate(
    const rtcore_replay_lane_request &request)
{
    if (!request.timing_trace_overflowed ||
        request.overflow_summary_events == 0) {
        return;
    }

    const rtcore_compact_trace_overflow_summary &summary =
        request.overflow_summary;
    g_rtcore_replay_overflow_summary_estimate_stats
        .overflow_summary_requests_consumed++;
    g_rtcore_replay_overflow_summary_estimate_stats
        .estimated_overflow_node_events +=
        summary.overflow_node_fetch_count + summary.overflow_node_test_count;
    g_rtcore_replay_overflow_summary_estimate_stats
        .estimated_overflow_primitive_events +=
        summary.overflow_primitive_fetch_count +
        summary.overflow_primitive_test_count;
    g_rtcore_replay_overflow_summary_estimate_stats
        .estimated_overflow_stack_events +=
        summary.overflow_stack_push_count + summary.overflow_stack_pop_count;
    g_rtcore_replay_overflow_summary_estimate_stats
        .estimated_overflow_memory_wait_events +=
        summary.overflow_memory_wait_count;
    g_rtcore_replay_overflow_summary_estimate_stats
        .estimated_overflow_memory_bytes += summary.overflow_memory_bytes;
    g_rtcore_replay_overflow_summary_estimate_stats
        .estimated_overflow_completion_events +=
        summary.overflow_completion_count;

    rtcore_maybe_log_replay_overflow_summary_estimate_stats(
        request.owner_hw_sid);
}

static unsigned rtcore_replay_estimated_cache_lines_64b(unsigned bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (bytes + RTCORE_REPLAY_MEMORY_DEMAND_CACHE_LINE_BYTES - 1) /
           RTCORE_REPLAY_MEMORY_DEMAND_CACHE_LINE_BYTES;
}

static void rtcore_update_replay_memory_demand_totals()
{
    g_rtcore_replay_memory_demand_estimate_stats
        .estimated_memory_demand_bytes =
        g_rtcore_replay_memory_demand_estimate_stats
            .explicit_fetch_memory_bytes +
        g_rtcore_replay_memory_demand_estimate_stats.overflow_memory_bytes;
    g_rtcore_replay_memory_demand_estimate_stats.estimated_cache_lines_64b =
        rtcore_replay_estimated_cache_lines_64b(
            g_rtcore_replay_memory_demand_estimate_stats
                .estimated_memory_demand_bytes);
    g_rtcore_replay_memory_demand_estimate_stats
        .estimated_memory_demand_events =
        g_rtcore_replay_memory_demand_estimate_stats
            .explicit_memory_wait_events +
        g_rtcore_replay_memory_demand_estimate_stats
            .overflow_memory_wait_events +
        g_rtcore_replay_memory_demand_estimate_stats.estimated_cache_lines_64b;
}

static void rtcore_maybe_log_replay_memory_demand_estimate_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_memory_demand_estimate_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_memory_demand_estimate_stats_logs_emitted >=
        rtcore_replay_memory_demand_estimate_log_limit()) {
        return;
    }
    g_rtcore_replay_memory_demand_estimate_stats_logs_emitted++;

    printf("GPGPU-Sim RTCORE_REPLAY_MEMORY_DEMAND_ESTIMATE_STATS "
           "owner_hw_sid=%u explicit_memory_wait_events=%u "
           "explicit_fetch_memory_bytes=%u overflow_memory_wait_events=%u "
           "overflow_memory_bytes=%u estimated_memory_demand_events=%u "
           "estimated_memory_demand_bytes=%u estimated_cache_lines_64b=%u\n",
           owner_hw_sid,
           g_rtcore_replay_memory_demand_estimate_stats
               .explicit_memory_wait_events,
           g_rtcore_replay_memory_demand_estimate_stats
               .explicit_fetch_memory_bytes,
           g_rtcore_replay_memory_demand_estimate_stats
               .overflow_memory_wait_events,
           g_rtcore_replay_memory_demand_estimate_stats.overflow_memory_bytes,
           g_rtcore_replay_memory_demand_estimate_stats
               .estimated_memory_demand_events,
           g_rtcore_replay_memory_demand_estimate_stats
               .estimated_memory_demand_bytes,
           g_rtcore_replay_memory_demand_estimate_stats
               .estimated_cache_lines_64b);
    fflush(stdout);
}

static void rtcore_maybe_log_replay_memory_latency_policy_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_memory_latency_policy_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_memory_latency_policy_stats_logs_emitted >=
        rtcore_replay_memory_latency_policy_log_limit()) {
        return;
    }
    g_rtcore_replay_memory_latency_policy_stats_logs_emitted++;

    printf("GPGPU-Sim RTCORE_REPLAY_MEMORY_LATENCY_POLICY_STATS "
           "owner_hw_sid=%u policy_evaluations=%u "
           "cache_line_latency_cycles=%u memory_wait_latency_cycles=%u "
           "estimated_cache_latency_cycles=%u "
           "estimated_memory_wait_latency_cycles=%u "
           "estimated_total_memory_latency_cycles=%u\n",
           owner_hw_sid,
           g_rtcore_replay_memory_latency_policy_stats.policy_evaluations,
           g_rtcore_replay_memory_latency_policy_stats
               .cache_line_latency_cycles,
           g_rtcore_replay_memory_latency_policy_stats
               .memory_wait_latency_cycles,
           g_rtcore_replay_memory_latency_policy_stats
               .estimated_cache_latency_cycles,
           g_rtcore_replay_memory_latency_policy_stats
               .estimated_memory_wait_latency_cycles,
           g_rtcore_replay_memory_latency_policy_stats
               .estimated_total_memory_latency_cycles);
    fflush(stdout);
}

static void rtcore_maybe_log_replay_memory_latency_blocked_cycle_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_memory_latency_blocked_cycle_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_memory_latency_blocked_cycle_stats_logs_emitted >=
        rtcore_replay_memory_latency_blocked_cycle_stats_log_limit()) {
        return;
    }
    g_rtcore_replay_memory_latency_blocked_cycle_stats_logs_emitted++;

    printf("GPGPU-Sim RTCORE_REPLAY_MEMORY_LATENCY_BLOCKED_CYCLE_STATS "
           "owner_hw_sid=%u policy_evaluations=%u blocked_cycle_total=%u "
           "max_delta_blocked_cycles=%u last_delta_blocked_cycles=%u "
           "current_estimated_total_memory_latency_cycles=%u\n",
           owner_hw_sid,
           g_rtcore_replay_memory_latency_blocked_cycle_stats
               .policy_evaluations,
           g_rtcore_replay_memory_latency_blocked_cycle_stats
               .blocked_cycle_total,
           g_rtcore_replay_memory_latency_blocked_cycle_stats
               .max_delta_blocked_cycles,
           g_rtcore_replay_memory_latency_blocked_cycle_stats
               .last_delta_blocked_cycles,
           g_rtcore_replay_memory_latency_blocked_cycle_stats
               .current_estimated_total_memory_latency_cycles);
    fflush(stdout);
}

static void rtcore_record_replay_memory_latency_blocked_cycle_stats(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_memory_latency_blocked_cycle_stats_log_enabled()) {
        return;
    }

    const unsigned current_total =
        g_rtcore_replay_memory_latency_policy_stats
            .estimated_total_memory_latency_cycles;
    const unsigned previous_total =
        g_rtcore_replay_memory_latency_blocked_cycle_last_total_cycles;
    const unsigned delta =
        current_total > previous_total ? current_total - previous_total : 0;

    g_rtcore_replay_memory_latency_blocked_cycle_last_total_cycles =
        current_total;
    g_rtcore_replay_memory_latency_blocked_cycle_stats.policy_evaluations =
        g_rtcore_replay_memory_latency_policy_stats.policy_evaluations;
    g_rtcore_replay_memory_latency_blocked_cycle_stats
        .last_delta_blocked_cycles = delta;
    g_rtcore_replay_memory_latency_blocked_cycle_stats
        .current_estimated_total_memory_latency_cycles = current_total;
    g_rtcore_replay_memory_latency_blocked_cycle_stats.blocked_cycle_total +=
        delta;
    if (delta > g_rtcore_replay_memory_latency_blocked_cycle_stats
                    .max_delta_blocked_cycles) {
        g_rtcore_replay_memory_latency_blocked_cycle_stats
            .max_delta_blocked_cycles = delta;
    }

    rtcore_maybe_log_replay_memory_latency_blocked_cycle_stats(owner_hw_sid);
}

static unsigned rtcore_replay_memory_wake_latency_cycles_for_event(
    const rtcore_compact_trace_event &event)
{
    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    const rtcore_compact_trace_resource_class resource_class =
        rtcore_unpack_compact_trace_resource_class(event);
    const unsigned count = rtcore_unpack_compact_trace_count(event) == 0
                               ? 1
                               : rtcore_unpack_compact_trace_count(event);
    const unsigned bytes = rtcore_unpack_compact_trace_bytes(event);
    unsigned latency_cycles = 0;

    if (event_type == RTCORE_TRACE_MEMORY_WAIT) {
        latency_cycles += count * rtcore_replay_memory_wait_event_latency_config();
    }
    if (event_type == RTCORE_TRACE_NODE_FETCH ||
        event_type == RTCORE_TRACE_PRIMITIVE_FETCH ||
        resource_class == RTCORE_TRACE_RESOURCE_MEMORY) {
        latency_cycles +=
            rtcore_replay_estimated_cache_lines_64b(bytes * count) *
            rtcore_replay_memory_cache_line_latency_config();
    }
    return latency_cycles;
}

static unsigned rtcore_replay_memory_request_chunks_for_event(
    const rtcore_compact_trace_event &event);

static void rtcore_replay_memory_unit_refresh_transaction_accounting(
    unsigned long long service_cycle)
{
    if (!g_rtcore_replay_memory_unit_transaction_groups_cycle_valid ||
        g_rtcore_replay_memory_unit_transaction_groups_cycle != service_cycle) {
        g_rtcore_replay_memory_unit_transaction_groups_this_cycle.clear();
        g_rtcore_replay_memory_unit_transaction_groups_cycle = service_cycle;
        g_rtcore_replay_memory_unit_transaction_groups_cycle_valid = true;
    }
}

static bool rtcore_replay_memory_unit_transaction_keys_for_event(
    const rtcore_replay_lane_request &request,
    const rtcore_compact_trace_event &event, unsigned event_index,
    unsigned long long service_cycle,
    std::set<rtcore_replay_memory_unit_transaction_key> *keys)
{
    if (!keys) {
        return false;
    }

    unsigned chunk_count = rtcore_replay_memory_request_chunks_for_event(event);
    unsigned long long base_address = event.address_or_ref;
    bool is_write = false;
    if (chunk_count == 0 &&
        rtcore_memory_unit_response_wait_manages_stack_load_event(event)) {
        chunk_count = rtcore_v02_lsu_stack_sideband_chunk_count(event);
        base_address =
            rtcore_v02_lsu_stack_sideband_base_address(request, event,
                                                       event_index);
    }
    if (chunk_count == 0) {
        return false;
    }

    for (unsigned chunk_id = 0; chunk_id < chunk_count; ++chunk_id) {
        const unsigned long long chunk_address =
            base_address +
            static_cast<unsigned long long>(
                chunk_id * RTCORE_REPLAY_MEMORY_REQUEST_GRANULE_BYTES);
        rtcore_replay_memory_unit_transaction_key key = {};
        key.owner_hw_sid = request.owner_hw_sid;
        key.service_cycle = service_cycle;
        key.aligned_32b_addr = rtcore_v02_lsu_align_32b(chunk_address);
        key.is_write = is_write;
        keys->insert(key);
    }
    return !keys->empty();
}

static unsigned
rtcore_replay_memory_unit_new_transaction_count_for_current_event(
    const rtcore_replay_lane_request &request,
    unsigned long long service_cycle)
{
    if (!request.valid || request.next_event_index >= request.events.size()) {
        return 1;
    }
    rtcore_replay_memory_unit_refresh_transaction_accounting(service_cycle);
    std::set<rtcore_replay_memory_unit_transaction_key> keys;
    if (!rtcore_replay_memory_unit_transaction_keys_for_event(
            request, request.events[request.next_event_index],
            request.next_event_index, service_cycle, &keys)) {
        return 1;
    }
    for (std::set<rtcore_replay_memory_unit_transaction_key>::const_iterator
             key_it = keys.begin();
         key_it != keys.end(); ++key_it) {
        if (g_rtcore_replay_memory_unit_transaction_groups_this_cycle.find(
                *key_it) ==
            g_rtcore_replay_memory_unit_transaction_groups_this_cycle.end()) {
            return 1;
        }
    }
    return 0;
}

static void
rtcore_replay_memory_unit_note_issued_transactions_for_current_event(
    const rtcore_replay_lane_request &request,
    unsigned long long service_cycle)
{
    if (!request.valid || request.next_event_index >= request.events.size()) {
        return;
    }
    rtcore_replay_memory_unit_refresh_transaction_accounting(service_cycle);
    std::set<rtcore_replay_memory_unit_transaction_key> keys;
    if (!rtcore_replay_memory_unit_transaction_keys_for_event(
            request, request.events[request.next_event_index],
            request.next_event_index, service_cycle, &keys)) {
        return;
    }
    for (std::set<rtcore_replay_memory_unit_transaction_key>::const_iterator
             key_it = keys.begin();
         key_it != keys.end(); ++key_it) {
        g_rtcore_replay_memory_unit_transaction_groups_this_cycle[*key_it]++;
    }
}

static void rtcore_count_replay_memory_outstanding_for_owner(
    unsigned owner_hw_sid, unsigned *outstanding_entry_count,
    unsigned *response_fanout_waiter_count, unsigned *outstanding_chunk_count);

static rtcore_replay_memory_outstanding_key
rtcore_make_replay_memory_outstanding_key(
    const rtcore_replay_lane_request &request, unsigned kind,
    unsigned event_index)
{
    rtcore_replay_memory_outstanding_key key = {};
    key.owner_hw_sid = request.owner_hw_sid;
    key.thread_uid = request.thread_uid;
    key.event_index = event_index;
    key.kind = kind;
    return key;
}

static void rtcore_register_replay_memory_outstanding_entry(
    const rtcore_replay_lane_request &request, unsigned kind,
    unsigned event_index, unsigned chunk_count,
    unsigned long long service_cycle)
{
    if (!request.valid || chunk_count == 0) {
        return;
    }

    const rtcore_replay_memory_outstanding_key key =
        rtcore_make_replay_memory_outstanding_key(request, kind, event_index);
    rtcore_replay_memory_outstanding_entry &entry =
        g_rtcore_replay_memory_outstanding_table[key];
    if (!entry.active) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_outstanding_table_register_count++;
    }
    entry.active = true;
    entry.key = key;
    entry.chunk_count = chunk_count;
    entry.response_fanout_waiter_count = 1;
    entry.issue_cycle = service_cycle;
    entry.has_transaction_key = false;
    entry.transaction_keys.clear();
    if (event_index < request.events.size()) {
        std::set<rtcore_replay_memory_unit_transaction_key> transaction_keys;
        if (rtcore_replay_memory_unit_transaction_keys_for_event(
                request, request.events[event_index], event_index,
                service_cycle, &transaction_keys) &&
            !transaction_keys.empty()) {
            entry.has_transaction_key = true;
            entry.transaction_key = *transaction_keys.begin();
            entry.transaction_keys = transaction_keys;
        }
    }
}

static void rtcore_release_replay_memory_outstanding_entry(
    const rtcore_replay_lane_request &request, unsigned kind,
    unsigned event_index)
{
    if (!request.valid) {
        return;
    }

    const rtcore_replay_memory_outstanding_key key =
        rtcore_make_replay_memory_outstanding_key(request, kind, event_index);
    std::map<rtcore_replay_memory_outstanding_key,
             rtcore_replay_memory_outstanding_entry>::iterator it =
        g_rtcore_replay_memory_outstanding_table.find(key);
    if (it == g_rtcore_replay_memory_outstanding_table.end() ||
        !it->second.active) {
        return;
    }

    g_rtcore_replay_v03_hw_memory_outstanding_stats
        .memory_outstanding_table_release_count++;
    g_rtcore_replay_memory_outstanding_table.erase(it);
}

static void rtcore_maybe_log_replay_memory_wake_latency_gate_stats(
    const rtcore_replay_lane_request &request, bool armed, bool blocked,
    bool woken, unsigned long long service_cycle)
{
    if (!rtcore_replay_memory_wake_latency_gate_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_memory_wake_latency_gate_stats_logs_emitted >=
        rtcore_replay_memory_wake_latency_gate_stats_log_limit()) {
        return;
    }
    g_rtcore_replay_memory_wake_latency_gate_stats_logs_emitted++;

    const unsigned long long blocked_cycles =
        request.memory_wake_ready_cycle > service_cycle
            ? request.memory_wake_ready_cycle - service_cycle
            : 0;
    const unsigned event_type =
        request.memory_wake_event_index < request.events.size()
            ? static_cast<unsigned>(rtcore_unpack_compact_trace_event_type(
                  request.events[request.memory_wake_event_index]))
            : 0;

    printf("GPGPU-Sim RTCORE_REPLAY_MEMORY_WAKE_LATENCY_GATE_STATS "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u "
           "has_warp_metadata=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
           "static_inst_uid=%u event_index=%u event_type=%u gate_enabled=%u "
           "armed=%u blocked=%u woken=%u service_cycle=%llu "
           "ready_cycle=%llu latency_cycles=%u blocked_cycles=%llu "
           "gate_evaluations=%u gate_armed_count=%u gate_blocked_count=%u "
           "gate_woken_count=%u max_latency_cycles=%u max_blocked_cycles=%u\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.has_warp_metadata ? 1u : 0u, request.warp_uid,
           request.warp_id, request.active_mask, request.static_inst_uid,
           request.memory_wake_event_index, event_type,
           rtcore_replay_memory_wake_latency_gate_enabled() ? 1u : 0u,
           armed ? 1u : 0u, blocked ? 1u : 0u, woken ? 1u : 0u,
           service_cycle, request.memory_wake_ready_cycle,
           request.memory_wake_latency_cycles, blocked_cycles,
           g_rtcore_replay_memory_wake_latency_gate_stats.gate_evaluations,
           g_rtcore_replay_memory_wake_latency_gate_stats.gate_armed_count,
           g_rtcore_replay_memory_wake_latency_gate_stats.gate_blocked_count,
           g_rtcore_replay_memory_wake_latency_gate_stats.gate_woken_count,
           g_rtcore_replay_memory_wake_latency_gate_stats.max_latency_cycles,
           g_rtcore_replay_memory_wake_latency_gate_stats.max_blocked_cycles);
    fflush(stdout);
}

static bool rtcore_maybe_arm_replay_memory_wake_latency_gate(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->valid ||
        !rtcore_replay_memory_wake_latency_gate_enabled()) {
        return false;
    }
    if (request->next_event_index >= request->events.size()) {
        return false;
    }
    if (request->memory_wake_latency_gate_pending &&
        request->memory_wake_event_index == request->next_event_index) {
        return false;
    }

    const unsigned latency_cycles =
        rtcore_replay_memory_wake_latency_cycles_for_event(
            request->events[request->next_event_index]);
    if (latency_cycles == 0) {
        return false;
    }

    request->memory_wake_latency_gate_pending = true;
    request->memory_wake_event_index = request->next_event_index;
    request->memory_wake_latency_cycles = latency_cycles;
    request->memory_wake_armed_cycle = service_cycle;
    request->memory_wake_ready_cycle = service_cycle + latency_cycles;
    request->state = RTCORE_REPLAY_ISSUED_MEMORY;
    rtcore_record_replay_request_state_write();
    unsigned chunk_count = 1;
    if (request->memory_wake_event_index < request->events.size()) {
        const unsigned event_chunks =
            rtcore_replay_memory_request_chunks_for_event(
                request->events[request->memory_wake_event_index]);
        if (event_chunks > 0) {
            chunk_count = event_chunks;
        }
    }
    rtcore_register_replay_memory_outstanding_entry(
        *request, RTCORE_REPLAY_MEMORY_OUTSTANDING_WAKE_LATENCY,
        request->memory_wake_event_index, chunk_count, service_cycle);

    g_rtcore_replay_memory_wake_latency_gate_stats.gate_evaluations++;
    g_rtcore_replay_memory_wake_latency_gate_stats.gate_armed_count++;
    if (latency_cycles >
        g_rtcore_replay_memory_wake_latency_gate_stats.max_latency_cycles) {
        g_rtcore_replay_memory_wake_latency_gate_stats.max_latency_cycles =
            latency_cycles;
    }
    rtcore_maybe_log_replay_memory_wake_latency_gate_stats(
        *request, true, false, false, service_cycle);
    return true;
}

static bool rtcore_replay_memory_wake_latency_gate_ready(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->memory_wake_latency_gate_pending) {
        return true;
    }
    if (service_cycle < request->memory_wake_ready_cycle) {
        const unsigned blocked_cycles = static_cast<unsigned>(
            request->memory_wake_ready_cycle - service_cycle);
        g_rtcore_replay_memory_wake_latency_gate_stats.gate_evaluations++;
        g_rtcore_replay_memory_wake_latency_gate_stats.gate_blocked_count++;
        if (blocked_cycles >
            g_rtcore_replay_memory_wake_latency_gate_stats.max_blocked_cycles) {
            g_rtcore_replay_memory_wake_latency_gate_stats.max_blocked_cycles =
                blocked_cycles;
        }
        rtcore_maybe_log_replay_memory_wake_latency_gate_stats(
            *request, false, true, false, service_cycle);
        return false;
    }

    g_rtcore_replay_memory_wake_latency_gate_stats.gate_evaluations++;
    g_rtcore_replay_memory_wake_latency_gate_stats.gate_woken_count++;
    rtcore_maybe_log_replay_memory_wake_latency_gate_stats(
        *request, false, false, true, service_cycle);
    rtcore_release_replay_memory_outstanding_entry(
        *request, RTCORE_REPLAY_MEMORY_OUTSTANDING_WAKE_LATENCY,
        request->memory_wake_event_index);
    request->memory_wake_latency_gate_pending = false;
    return true;
}

static unsigned rtcore_replay_memory_request_chunks_for_event(
    const rtcore_compact_trace_event &event)
{
    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    if (event_type != RTCORE_TRACE_NODE_FETCH &&
        event_type != RTCORE_TRACE_PRIMITIVE_FETCH) {
        return 0;
    }

    const unsigned count = rtcore_unpack_compact_trace_count(event) == 0
                               ? 1
                               : rtcore_unpack_compact_trace_count(event);
    const unsigned bytes = rtcore_unpack_compact_trace_bytes(event);
    const unsigned total_bytes = bytes * count;
    if (total_bytes == 0) {
        return 1;
    }
    return (total_bytes + RTCORE_REPLAY_MEMORY_REQUEST_GRANULE_BYTES - 1) /
           RTCORE_REPLAY_MEMORY_REQUEST_GRANULE_BYTES;
}

static bool rtcore_replay_memory_address_gen_applies_to_event(
    const rtcore_compact_trace_event &event)
{
    if (rtcore_replay_memory_request_chunks_for_event(event) > 0) {
        return true;
    }
    return rtcore_memory_unit_response_wait_manages_stack_load_event(event);
}

static bool rtcore_replay_memory_address_gen_completed_for_current_event(
    const rtcore_replay_lane_request &request)
{
    return request.memory_address_gen_completed_valid &&
           request.memory_address_gen_event_index == request.next_event_index;
}

static bool rtcore_maybe_arm_memory_address_gen_latency(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->valid ||
        request->state != RTCORE_REPLAY_ISSUED_MEMORY) {
        return false;
    }
    if (request->next_event_index >= request->events.size()) {
        return false;
    }
    if (request->memory_address_gen_latency_gate_pending ||
        rtcore_replay_memory_address_gen_completed_for_current_event(*request)) {
        return false;
    }
    if (!rtcore_replay_memory_address_gen_applies_to_event(
            request->events[request->next_event_index])) {
        return false;
    }

    const unsigned latency_cycles =
        rtcore_replay_memory_address_gen_latency_config();
    request->memory_address_gen_latency_gate_pending = true;
    request->memory_address_gen_completed_valid = false;
    request->memory_address_gen_event_index = request->next_event_index;
    request->memory_address_gen_latency_cycles = latency_cycles;
    request->memory_address_gen_armed_cycle = service_cycle;
    request->memory_address_gen_ready_cycle =
        service_cycle + (latency_cycles > 0 ? latency_cycles - 1 : 0);
    rtcore_record_replay_request_state_write();
    return true;
}

static bool rtcore_replay_memory_address_gen_latency_ready(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->memory_address_gen_latency_gate_pending) {
        return true;
    }
    if (service_cycle < request->memory_address_gen_ready_cycle) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_address_gen_latency_blocked_count++;
        if (g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_address_gen_latency_blocked_count >
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .max_memory_address_gen_latency_blocked_count) {
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .max_memory_address_gen_latency_blocked_count =
                g_rtcore_replay_v03_hw_memory_outstanding_stats
                    .memory_address_gen_latency_blocked_count;
        }
        return false;
    }

    request->memory_address_gen_latency_gate_pending = false;
    request->memory_address_gen_completed_valid = true;
    rtcore_record_replay_request_state_write();
    return true;
}

static bool rtcore_maybe_arm_memory_unit_response_wait(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->valid ||
        !rtcore_memory_unit_response_wait_enabled() ||
        !rtcore_replay_memory_unit_request_offer_enabled() ||
        !rtcore_replay_memory_unit_l1d_client_enabled()) {
        return false;
    }
    if (request->next_event_index >= request->events.size()) {
        return false;
    }
    if (request->v02_lsu_response_wait_gate_pending) {
        return false;
    }

    const rtcore_compact_trace_event &event =
        request->events[request->next_event_index];
    unsigned access_kind = RTCORE_V02_LSU_ACCESS_KIND_COUNT;
    bool stack_response_wait = false;
    unsigned chunk_count = rtcore_replay_memory_request_chunks_for_event(event);
    if (chunk_count == 0 &&
        rtcore_memory_unit_response_wait_manages_stack_load_event(event)) {
        access_kind = RTCORE_V02_LSU_ACCESS_STACK_LOAD;
        chunk_count = rtcore_v02_lsu_stack_sideband_chunk_count(event);
        stack_response_wait = true;
    }
    if (chunk_count == 0) {
        return false;
    }

    if (stack_response_wait) {
        if (!rtcore_maybe_enqueue_v02_lsu_stack_sideband(*request,
                                                         service_cycle)) {
            return false;
        }
    } else {
        rtcore_record_replay_memory_unit_request_descriptor(*request, event,
                                                      service_cycle,
                                                      chunk_count);
    }

    request->v02_lsu_response_wait_gate_pending = true;
    request->v02_lsu_response_wait_event_index = request->next_event_index;
    request->v02_lsu_response_wait_chunk_count = chunk_count;
    request->v02_lsu_response_wait_completed_chunk_count = 0;
    request->v02_lsu_response_wait_access_kind = access_kind;
    request->v02_lsu_response_wait_armed_cycle = service_cycle;
    request->v02_lsu_response_wait_completed_chunks.clear();
    request->state = RTCORE_REPLAY_ISSUED_MEMORY;
    rtcore_record_replay_request_state_write();
    rtcore_register_replay_memory_outstanding_entry(
        *request, RTCORE_REPLAY_MEMORY_OUTSTANDING_V02_LSU_RESPONSE_WAIT,
        request->v02_lsu_response_wait_event_index, chunk_count,
        service_cycle);

    g_rtcore_v02_lsu_response_wait_stats.gate_evaluations++;
    g_rtcore_v02_lsu_response_wait_stats.gate_armed_count++;
    if (access_kind == RTCORE_V02_LSU_ACCESS_STACK_LOAD) {
        g_rtcore_v02_lsu_response_wait_stats.stack_load_armed_count++;
    } else if (access_kind == RTCORE_V02_LSU_ACCESS_STACK_STORE) {
        g_rtcore_v02_lsu_response_wait_stats.stack_store_armed_count++;
    }
    g_rtcore_v02_lsu_response_wait_stats.pending_request_count++;
    if (g_rtcore_v02_lsu_response_wait_stats.pending_request_count >
        g_rtcore_v02_lsu_response_wait_stats.max_pending_request_count) {
        g_rtcore_v02_lsu_response_wait_stats.max_pending_request_count =
            g_rtcore_v02_lsu_response_wait_stats.pending_request_count;
    }
    if (chunk_count >
        g_rtcore_v02_lsu_response_wait_stats.max_chunk_count) {
        g_rtcore_v02_lsu_response_wait_stats.max_chunk_count = chunk_count;
    }
    rtcore_maybe_log_memory_unit_response_wait_stats(request->owner_hw_sid);
    return true;
}

static bool rtcore_memory_unit_response_wait_ready(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    (void)service_cycle;
    if (!request || !request->v02_lsu_response_wait_gate_pending) {
        return true;
    }
    g_rtcore_v02_lsu_response_wait_stats.gate_evaluations++;
    if (request->v02_lsu_response_wait_completed_chunk_count <
        request->v02_lsu_response_wait_chunk_count) {
        g_rtcore_v02_lsu_response_wait_stats.gate_blocked_count++;
        rtcore_maybe_log_memory_unit_response_wait_stats(request->owner_hw_sid);
        return false;
    }

    g_rtcore_v02_lsu_response_wait_stats.gate_woken_count++;
    if (request->v02_lsu_response_wait_access_kind ==
        RTCORE_V02_LSU_ACCESS_STACK_LOAD) {
        g_rtcore_v02_lsu_response_wait_stats.stack_load_woken_count++;
    } else if (request->v02_lsu_response_wait_access_kind ==
               RTCORE_V02_LSU_ACCESS_STACK_STORE) {
        g_rtcore_v02_lsu_response_wait_stats.stack_store_woken_count++;
    }
    g_rtcore_v02_lsu_response_wait_stats.completed_event_count++;
    if (g_rtcore_v02_lsu_response_wait_stats.pending_request_count > 0) {
        g_rtcore_v02_lsu_response_wait_stats.pending_request_count--;
    }
    request->v02_lsu_response_wait_gate_pending = false;
    request->v02_lsu_response_wait_completed_chunks.clear();
    rtcore_release_replay_memory_outstanding_entry(
        *request, RTCORE_REPLAY_MEMORY_OUTSTANDING_V02_LSU_RESPONSE_WAIT,
        request->v02_lsu_response_wait_event_index);
    rtcore_maybe_log_memory_unit_response_wait_stats(request->owner_hw_sid);
    return true;
}

static bool rtcore_record_memory_unit_response_wait_chunk(
    unsigned owner_hw_sid, unsigned rt_request_id, unsigned memory_op_seq,
    unsigned chunk_id, unsigned chunk_count, unsigned response_target,
    unsigned long long response_cycle)
{
    (void)response_cycle;
    if (!rtcore_memory_unit_response_wait_enabled()) {
        return false;
    }

    std::map<unsigned, rtcore_replay_lane_request>::iterator it =
        g_rtcore_replay_lane_requests.find(rt_request_id);
    if (it == g_rtcore_replay_lane_requests.end() || !it->second.valid ||
        it->second.owner_hw_sid != owner_hw_sid ||
        !it->second.v02_lsu_response_wait_gate_pending ||
        it->second.v02_lsu_response_wait_event_index != memory_op_seq ||
        chunk_id >= it->second.v02_lsu_response_wait_chunk_count ||
        (chunk_count != 0 &&
         chunk_count != it->second.v02_lsu_response_wait_chunk_count)) {
        g_rtcore_v02_lsu_response_wait_stats.stale_response_count++;
        rtcore_maybe_log_memory_unit_response_wait_stats(owner_hw_sid);
        return false;
    }

    if (!it->second.v02_lsu_response_wait_completed_chunks.insert(chunk_id)
             .second) {
        g_rtcore_v02_lsu_response_wait_stats.duplicate_response_count++;
        rtcore_maybe_log_memory_unit_response_wait_stats(owner_hw_sid);
        return false;
    }

    it->second.v02_lsu_response_wait_completed_chunk_count++;
    g_rtcore_v02_lsu_response_wait_stats.response_chunk_count++;
    if (response_target == RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE) {
        g_rtcore_v02_lsu_response_wait_stats.response_target_rtcore_count++;
    }
    rtcore_maybe_log_memory_unit_response_wait_stats(owner_hw_sid);
    return true;
}

static bool rtcore_record_v04_live_handoff_publication_response(
    unsigned owner_hw_sid, unsigned rt_request_id, unsigned memory_op_seq,
    unsigned chunk_id, unsigned chunk_count, unsigned response_target,
    unsigned long long response_cycle)
{
    if (!rtcore_v04_live_handoff_publication_enabled() ||
        !rtcore_v04_live_publication_memory_op_seq(memory_op_seq)) {
        return false;
    }

    std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
        g_rtcore_replay_lane_requests.find(rt_request_id);
    if (request_it == g_rtcore_replay_lane_requests.end()) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_FAULT "
                "owner_hw_sid=%u rt_request_id=%u memory_op_seq=%u "
                "fault=ack_request_missing\n",
                owner_hw_sid, rt_request_id, memory_op_seq);
        fflush(stderr);
        abort();
    }
    rtcore_replay_lane_request &request = request_it->second;
    const unsigned publication_chunk =
        memory_op_seq - RTCORE_V04_LIVE_PUBLICATION_OP_SEQ_BASE;
    const unsigned publication_chunk_bit = 1u << publication_chunk;
    const bool response_valid =
        request.valid && request.owner_hw_sid == owner_hw_sid &&
        rtcore_v04_live_publication_matches_current_submit(request) &&
        !request.v04_live_publication_committed && chunk_id == 0 &&
        chunk_count == 1 &&
        response_target == RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE &&
        (request.v04_live_publication_pending_chunk_mask &
         publication_chunk_bit) != 0 &&
        (request.v04_live_publication_acked_chunk_mask &
         publication_chunk_bit) == 0;
    if (!response_valid) {
        rtcore_fail_v04_live_handoff_publication(
            request, "ack_identity_or_chunk_mismatch",
            request.v04_live_publication_word_mask,
            request.v04_live_publication_pending_chunk_mask);
    }

    request.v04_live_publication_acked_chunk_mask |= publication_chunk_bit;
    const bool all_chunks_acked =
        request.v04_live_publication_acked_chunk_mask ==
        request.v04_live_publication_pending_chunk_mask;
    printf("GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_ACK "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "memory_op_seq=%u publication_chunk=%u "
           "pending_chunk_mask=0x%x acked_chunk_mask=0x%x "
           "all_chunks_acked=%u response_cycle=%llu\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.warp_uid, memory_op_seq, publication_chunk,
           request.v04_live_publication_pending_chunk_mask,
           request.v04_live_publication_acked_chunk_mask,
           all_chunks_acked ? 1u : 0u, response_cycle);
    fflush(stdout);
    if (!all_chunks_acked) {
        return true;
    }

    const unsigned long long lane_address =
        request.handoff_window_base +
        static_cast<unsigned long long>(request.lane_id) *
            rtcore::abi_v04::kLaneSlotBytes;
    const char *mutation =
        getenv("VULKAN_SIM_RTCORE_TEST_V04_LIVE_HANDOFF_MUTATION");
    bool mutation_applied = false;
    for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
        const uint32_t word_bit = uint32_t{1} << word;
        if ((request.v04_live_publication_word_mask & word_bit) == 0) {
            continue;
        }
        uint32_t value = request.v04_live_publication_words[word];
        if (!mutation_applied && mutation != NULL &&
            strcmp(mutation, "owned_word") == 0) {
            value ^= 1u;
            mutation_applied = true;
        }
        request.v04_live_handoff_memory->write_simulator_backing(
            lane_address + word * sizeof(uint32_t), sizeof(value), &value);
    }

    std::array<uint32_t, rtcore::abi_v04::kWordCount> postimage = {};
    request.v04_live_handoff_memory->read_simulator_backing(
        lane_address, sizeof(postimage), postimage.data());
    uint32_t owned_mismatch_mask = 0;
    uint32_t preserved_mismatch_mask = 0;
    for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
        const uint32_t word_bit = uint32_t{1} << word;
        if ((request.v04_live_publication_word_mask & word_bit) != 0) {
            if (postimage[word] !=
                request.v04_live_publication_words[word]) {
                owned_mismatch_mask |= word_bit;
            }
        } else if (postimage[word] !=
                   request.v04_live_publication_preimage_words[word]) {
            preserved_mismatch_mask |= word_bit;
        }
    }
    if (owned_mismatch_mask != 0 || preserved_mismatch_mask != 0) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_FAULT "
                "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
                "word_mask=0x%08x owned_mismatch_mask=0x%08x "
                "preserved_mismatch_mask=0x%08x mutation=%s "
                "fault=post_ack_backing_validation\n",
                request.owner_hw_sid, request.thread_uid, request.lane_id,
                request.warp_uid, request.v04_live_publication_word_mask,
                owned_mismatch_mask, preserved_mismatch_mask,
                mutation != NULL && mutation[0] != '\0' ? mutation : "none");
        fflush(stderr);
        abort();
    }

    request.v04_live_publication_committed = true;
    printf("GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_COMMIT "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "reason=%u lane_address=0x%llx word_mask=0x%08x "
           "chunk_mask=0x%x acked_chunk_mask=0x%x "
           "owned_mismatch_mask=0x%08x preserved_mismatch_mask=0x%08x "
           "functional_backing_write=1 completion_release_ready=1 "
           "response_cycle=%llu\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.warp_uid, request.v04_live_publication_reason,
           lane_address, request.v04_live_publication_word_mask,
           request.v04_live_publication_pending_chunk_mask,
           request.v04_live_publication_acked_chunk_mask,
           owned_mismatch_mask, preserved_mismatch_mask, response_cycle);
    fflush(stdout);

    const rtcore_replay_warp_completion_entry_key boundary_key =
        rtcore_make_continuation_warp_key(request);
    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_continuation_warp_boundary_state>::iterator boundary_it =
        g_rtcore_continuation_warp_boundary_states.find(boundary_key);
    if (boundary_it != g_rtcore_continuation_warp_boundary_states.end() &&
        boundary_it->second.valid &&
        boundary_it->second.pending_packet_valid &&
        !boundary_it->second.packet_published) {
        (void)rtcore_publish_continuation_return_packet(
            &boundary_it->second, response_cycle);
        return true;
    }

    const rtcore_replay_warp_completion_entry_key completion_key =
        rtcore_make_replay_warp_completion_entry_key(request);
    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_replay_warp_completion_entry_state>::const_iterator
        completion_it =
            g_rtcore_replay_warp_completion_entries.find(completion_key);
    const unsigned lane_mask = 1u << request.lane_id;
    const bool terminal_completion_pending =
        request.state == RTCORE_REPLAY_COMPLETION_PENDING &&
        completion_it != g_rtcore_replay_warp_completion_entries.end() &&
        completion_it->second.valid &&
        (completion_it->second.completed_lane_mask & lane_mask) == 0;
    if (!terminal_completion_pending) {
        rtcore_fail_v04_live_handoff_publication(
            request, "ack_completion_owner_missing",
            request.v04_live_publication_word_mask,
            request.v04_live_publication_pending_chunk_mask);
    }

    printf("GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_TERMINAL_ACK_READY "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "reason=%u completion_path=generic_terminal "
           "response_cycle=%llu\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.warp_uid, request.v04_live_publication_reason,
           response_cycle);
    fflush(stdout);
    return true;
}

static unsigned rtcore_replay_memory_contention_cache_lines_for_event(
    const rtcore_compact_trace_event &event)
{
    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    const rtcore_compact_trace_resource_class resource_class =
        rtcore_unpack_compact_trace_resource_class(event);
    const unsigned count = rtcore_unpack_compact_trace_count(event) == 0
                               ? 1
                               : rtcore_unpack_compact_trace_count(event);
    const unsigned bytes = rtcore_unpack_compact_trace_bytes(event);

    if (event_type == RTCORE_TRACE_MEMORY_WAIT) {
        return count;
    }
    if (event_type == RTCORE_TRACE_NODE_FETCH ||
        event_type == RTCORE_TRACE_PRIMITIVE_FETCH ||
        resource_class == RTCORE_TRACE_RESOURCE_MEMORY) {
        const unsigned lines =
            rtcore_replay_estimated_cache_lines_64b(bytes * count);
        return lines == 0 ? count : lines;
    }
    return 0;
}

static void rtcore_maybe_log_replay_memory_contention_gate_stats(
    const rtcore_replay_lane_request &request, bool armed, bool blocked,
    bool woken, bool capacity_blocked, unsigned long long service_cycle)
{
    if (!rtcore_replay_memory_contention_gate_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_memory_contention_gate_stats_logs_emitted >=
        rtcore_replay_memory_contention_gate_stats_log_limit()) {
        return;
    }
    g_rtcore_replay_memory_contention_gate_stats_logs_emitted++;

    const unsigned long long blocked_cycles =
        request.memory_contention_ready_cycle > service_cycle
            ? request.memory_contention_ready_cycle - service_cycle
            : 0;
    const unsigned event_type =
        request.memory_contention_event_index < request.events.size()
            ? static_cast<unsigned>(rtcore_unpack_compact_trace_event_type(
                  request.events[request.memory_contention_event_index]))
            : 0;
    const unsigned queue_capacity = 0;
    unsigned inflight_reservations = 0;
    std::map<unsigned, rtcore_replay_memory_contention_owner_state>::
        const_iterator owner_it =
            g_rtcore_replay_memory_contention_owner_states.find(
                request.owner_hw_sid);
    if (owner_it != g_rtcore_replay_memory_contention_owner_states.end()) {
        inflight_reservations = owner_it->second.inflight_reservations;
    }

    printf("GPGPU-Sim RTCORE_REPLAY_MEMORY_CONTENTION_GATE_STATS "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u has_warp_metadata=%u "
           "warp_uid=%u warp_id=%u active_mask=0x%08x "
           "static_inst_uid=%u event_index=%u event_type=%u gate_enabled=%u "
           "armed=%u blocked=%u woken=%u service_cycle=%llu "
           "start_cycle=%llu ready_cycle=%llu cache_lines=%u "
           "contention_cycles=%u queue_delay_cycles=%u blocked_cycles=%llu "
           "gate_evaluations=%u gate_armed_count=%u gate_blocked_count=%u "
           "gate_woken_count=%u max_cache_lines=%u "
           "max_contention_cycles=%u max_queue_delay_cycles=%u "
           "queue_capacity=%u inflight_reservations=%u capacity_blocked=%u "
           "capacity_blocked_count=%u max_inflight_reservations=%u\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.has_warp_metadata ? 1u : 0u, request.warp_uid,
           request.warp_id, request.active_mask, request.static_inst_uid,
           request.memory_contention_event_index, event_type,
           rtcore_replay_memory_contention_gate_enabled() ? 1u : 0u,
           armed ? 1u : 0u, blocked ? 1u : 0u, woken ? 1u : 0u,
           service_cycle, request.memory_contention_start_cycle,
           request.memory_contention_ready_cycle,
           request.memory_contention_cache_lines,
           request.memory_contention_cycles,
           request.memory_contention_queue_delay_cycles, blocked_cycles,
           g_rtcore_replay_memory_contention_gate_stats.gate_evaluations,
           g_rtcore_replay_memory_contention_gate_stats.gate_armed_count,
           g_rtcore_replay_memory_contention_gate_stats.gate_blocked_count,
           g_rtcore_replay_memory_contention_gate_stats.gate_woken_count,
           g_rtcore_replay_memory_contention_gate_stats.max_cache_lines,
           g_rtcore_replay_memory_contention_gate_stats
               .max_contention_cycles,
           g_rtcore_replay_memory_contention_gate_stats
               .max_queue_delay_cycles,
           queue_capacity, inflight_reservations,
           capacity_blocked ? 1u : 0u,
           g_rtcore_replay_memory_contention_gate_stats
               .capacity_blocked_count,
           g_rtcore_replay_memory_contention_gate_stats
               .max_inflight_reservations);
    fflush(stdout);
}

static bool rtcore_maybe_arm_replay_memory_contention_gate(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->valid ||
        !rtcore_replay_memory_contention_gate_enabled()) {
        return false;
    }
    if (request->next_event_index >= request->events.size()) {
        return false;
    }
    if (request->memory_contention_gate_pending &&
        request->memory_contention_event_index == request->next_event_index) {
        return false;
    }

    const unsigned cache_lines =
        rtcore_replay_memory_contention_cache_lines_for_event(
            request->events[request->next_event_index]);
    if (cache_lines == 0) {
        return false;
    }

    const unsigned lines_per_cycle =
        rtcore_replay_memory_contention_cache_lines_per_cycle_config();
    const unsigned contention_cycles =
        (cache_lines + lines_per_cycle - 1) / lines_per_cycle;
    rtcore_replay_memory_contention_owner_state &owner_state =
        g_rtcore_replay_memory_contention_owner_states[request->owner_hw_sid];

    const unsigned long long start_cycle =
        owner_state.next_available_cycle > service_cycle
            ? owner_state.next_available_cycle
            : service_cycle;
    const unsigned long long ready_cycle =
        start_cycle + (contention_cycles == 0 ? 1 : contention_cycles);
    const unsigned long long queue_delay_cycles =
        start_cycle > service_cycle ? start_cycle - service_cycle : 0;

    owner_state.next_available_cycle = ready_cycle;
    owner_state.reservations++;
    owner_state.inflight_reservations++;

    request->memory_contention_gate_pending = true;
    request->memory_contention_event_index = request->next_event_index;
    request->memory_contention_cache_lines = cache_lines;
    request->memory_contention_cycles =
        contention_cycles == 0 ? 1 : contention_cycles;
    request->memory_contention_queue_delay_cycles =
        static_cast<unsigned>(queue_delay_cycles);
    request->memory_contention_armed_cycle = service_cycle;
    request->memory_contention_start_cycle = start_cycle;
    request->memory_contention_ready_cycle = ready_cycle;
    request->state = RTCORE_REPLAY_ISSUED_MEMORY;
    rtcore_record_replay_request_state_write();
    rtcore_register_replay_memory_outstanding_entry(
        *request, RTCORE_REPLAY_MEMORY_OUTSTANDING_CONTENTION,
        request->memory_contention_event_index, cache_lines, service_cycle);

    g_rtcore_replay_memory_contention_gate_stats.gate_evaluations++;
    g_rtcore_replay_memory_contention_gate_stats.gate_armed_count++;
    if (cache_lines >
        g_rtcore_replay_memory_contention_gate_stats.max_cache_lines) {
        g_rtcore_replay_memory_contention_gate_stats.max_cache_lines =
            cache_lines;
    }
    if (request->memory_contention_cycles >
        g_rtcore_replay_memory_contention_gate_stats.max_contention_cycles) {
        g_rtcore_replay_memory_contention_gate_stats.max_contention_cycles =
            request->memory_contention_cycles;
    }
    if (request->memory_contention_queue_delay_cycles >
        g_rtcore_replay_memory_contention_gate_stats.max_queue_delay_cycles) {
        g_rtcore_replay_memory_contention_gate_stats
            .max_queue_delay_cycles =
            request->memory_contention_queue_delay_cycles;
    }
    if (owner_state.inflight_reservations >
        g_rtcore_replay_memory_contention_gate_stats
            .max_inflight_reservations) {
        g_rtcore_replay_memory_contention_gate_stats
            .max_inflight_reservations =
            owner_state.inflight_reservations;
    }
    rtcore_maybe_log_replay_memory_contention_gate_stats(
        *request, true, false, false, false, service_cycle);
    return true;
}

static bool rtcore_replay_memory_contention_gate_ready(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->memory_contention_gate_pending) {
        return true;
    }
    if (service_cycle < request->memory_contention_ready_cycle) {
        g_rtcore_replay_memory_contention_gate_stats.gate_evaluations++;
        g_rtcore_replay_memory_contention_gate_stats.gate_blocked_count++;
        rtcore_maybe_log_replay_memory_contention_gate_stats(
            *request, false, true, false, false, service_cycle);
        return false;
    }

    g_rtcore_replay_memory_contention_gate_stats.gate_evaluations++;
    g_rtcore_replay_memory_contention_gate_stats.gate_woken_count++;
    std::map<unsigned, rtcore_replay_memory_contention_owner_state>::iterator
        owner_it =
            g_rtcore_replay_memory_contention_owner_states.find(
                request->owner_hw_sid);
    if (owner_it != g_rtcore_replay_memory_contention_owner_states.end() &&
        owner_it->second.inflight_reservations > 0) {
        owner_it->second.inflight_reservations--;
    }
    rtcore_maybe_log_replay_memory_contention_gate_stats(
        *request, false, false, true, false, service_cycle);
    rtcore_release_replay_memory_outstanding_entry(
        *request, RTCORE_REPLAY_MEMORY_OUTSTANDING_CONTENTION,
        request->memory_contention_event_index);
    request->memory_contention_gate_pending = false;
    return true;
}

static unsigned rtcore_replay_unit_latency_cycles_for_event(
    rtcore_compact_trace_event event, unsigned *unit)
{
    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    if (unit) {
        *unit = RTCORE_REPLAY_UNIT_LATENCY_NONE;
    }
    switch (event_type) {
    case RTCORE_TRACE_NODE_TEST:
        if (unit) {
            *unit = RTCORE_REPLAY_UNIT_LATENCY_NODE;
        }
        return rtcore_replay_node_test_latency_config();
    case RTCORE_TRACE_PRIMITIVE_TEST:
        if (unit) {
            *unit = RTCORE_REPLAY_UNIT_LATENCY_PRIMITIVE;
        }
        return rtcore_replay_primitive_test_latency_config();
    case RTCORE_TRACE_STACK_PUSH:
    case RTCORE_TRACE_STACK_POP:
        if (unit) {
            *unit = RTCORE_REPLAY_UNIT_LATENCY_STACK;
        }
        return rtcore_replay_stack_latency_cycles_config();
    default:
        return 0;
    }
}

static void rtcore_maybe_log_replay_unit_latency_gate_stats(
    const rtcore_replay_lane_request &request, bool armed, bool blocked,
    bool woken, unsigned long long service_cycle)
{
    if (!rtcore_replay_unit_latency_gate_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_unit_latency_gate_stats_logs_emitted >=
        rtcore_replay_unit_latency_gate_stats_log_limit()) {
        return;
    }
    g_rtcore_replay_unit_latency_gate_stats_logs_emitted++;

    const unsigned long long blocked_cycles =
        request.unit_latency_ready_cycle > service_cycle
            ? request.unit_latency_ready_cycle - service_cycle
            : 0;
    const unsigned event_type =
        request.unit_latency_event_index < request.events.size()
            ? static_cast<unsigned>(rtcore_unpack_compact_trace_event_type(
                  request.events[request.unit_latency_event_index]))
            : 0;

    printf("GPGPU-Sim RTCORE_REPLAY_UNIT_LATENCY_GATE_STATS "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u has_warp_metadata=%u "
           "warp_uid=%u warp_id=%u active_mask=0x%08x unit=%u "
           "event_index=%u event_type=%u gate_enabled=%u armed=%u "
           "blocked=%u woken=%u service_cycle=%llu ready_cycle=%llu "
           "latency_cycles=%u blocked_cycles=%llu gate_evaluations=%u "
           "node_gate_armed_count=%u primitive_gate_armed_count=%u "
           "stack_gate_armed_count=%u gate_blocked_count=%u "
           "gate_woken_count=%u max_latency_cycles=%u max_blocked_cycles=%u\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.has_warp_metadata ? 1u : 0u, request.warp_uid,
           request.warp_id, request.active_mask, request.unit_latency_unit,
           request.unit_latency_event_index, event_type,
           rtcore_replay_unit_latency_gate_enabled() ? 1u : 0u,
           armed ? 1u : 0u, blocked ? 1u : 0u, woken ? 1u : 0u,
           service_cycle, request.unit_latency_ready_cycle,
           request.unit_latency_cycles, blocked_cycles,
           g_rtcore_replay_unit_latency_gate_stats.gate_evaluations,
           g_rtcore_replay_unit_latency_gate_stats.node_gate_armed_count,
           g_rtcore_replay_unit_latency_gate_stats.primitive_gate_armed_count,
           g_rtcore_replay_unit_latency_gate_stats.stack_gate_armed_count,
           g_rtcore_replay_unit_latency_gate_stats.gate_blocked_count,
           g_rtcore_replay_unit_latency_gate_stats.gate_woken_count,
           g_rtcore_replay_unit_latency_gate_stats.max_latency_cycles,
           g_rtcore_replay_unit_latency_gate_stats.max_blocked_cycles);
    fflush(stdout);
}

static bool rtcore_maybe_arm_replay_unit_latency_gate(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->valid ||
        !rtcore_replay_unit_latency_gate_enabled()) {
        return false;
    }
    if (request->next_event_index >= request->events.size()) {
        return false;
    }
    if (request->unit_latency_gate_pending &&
        request->unit_latency_event_index == request->next_event_index) {
        return false;
    }

    unsigned unit = RTCORE_REPLAY_UNIT_LATENCY_NONE;
    const unsigned latency_cycles =
        rtcore_replay_unit_latency_cycles_for_event(
            request->events[request->next_event_index], &unit);
    if (latency_cycles == 0 || unit == RTCORE_REPLAY_UNIT_LATENCY_NONE) {
        return false;
    }

    request->unit_latency_gate_pending = true;
    request->unit_latency_unit = unit;
    request->unit_latency_event_index = request->next_event_index;
    request->unit_latency_cycles = latency_cycles;
    request->unit_latency_armed_cycle = service_cycle;
    request->unit_latency_ready_cycle = service_cycle + latency_cycles;
    if (unit == RTCORE_REPLAY_UNIT_LATENCY_NODE) {
        request->state = RTCORE_REPLAY_ISSUED_NODE;
        rtcore_record_replay_request_state_write();
    } else if (unit == RTCORE_REPLAY_UNIT_LATENCY_PRIMITIVE) {
        request->state = RTCORE_REPLAY_ISSUED_PRIMITIVE;
        rtcore_record_replay_request_state_write();
    } else if (unit == RTCORE_REPLAY_UNIT_LATENCY_STACK) {
        request->state = RTCORE_REPLAY_ISSUED_STACK;
        rtcore_record_replay_request_state_write();
    }

    g_rtcore_replay_unit_latency_gate_stats.gate_evaluations++;
    if (unit == RTCORE_REPLAY_UNIT_LATENCY_NODE) {
        g_rtcore_replay_unit_latency_gate_stats.node_gate_armed_count++;
    } else if (unit == RTCORE_REPLAY_UNIT_LATENCY_PRIMITIVE) {
        g_rtcore_replay_unit_latency_gate_stats.primitive_gate_armed_count++;
    } else if (unit == RTCORE_REPLAY_UNIT_LATENCY_STACK) {
        g_rtcore_replay_unit_latency_gate_stats.stack_gate_armed_count++;
    }
    if (latency_cycles >
        g_rtcore_replay_unit_latency_gate_stats.max_latency_cycles) {
        g_rtcore_replay_unit_latency_gate_stats.max_latency_cycles =
            latency_cycles;
    }
    rtcore_maybe_log_replay_unit_latency_gate_stats(
        *request, true, false, false, service_cycle);
    return true;
}

static bool rtcore_replay_unit_latency_gate_ready(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->unit_latency_gate_pending) {
        return true;
    }
    if (service_cycle < request->unit_latency_ready_cycle) {
        const unsigned blocked_cycles = static_cast<unsigned>(
            request->unit_latency_ready_cycle - service_cycle);
        g_rtcore_replay_unit_latency_gate_stats.gate_evaluations++;
        g_rtcore_replay_unit_latency_gate_stats.gate_blocked_count++;
        if (blocked_cycles >
            g_rtcore_replay_unit_latency_gate_stats.max_blocked_cycles) {
            g_rtcore_replay_unit_latency_gate_stats.max_blocked_cycles =
                blocked_cycles;
        }
        rtcore_maybe_log_replay_unit_latency_gate_stats(
            *request, false, true, false, service_cycle);
        return false;
    }

    g_rtcore_replay_unit_latency_gate_stats.gate_evaluations++;
    g_rtcore_replay_unit_latency_gate_stats.gate_woken_count++;
    rtcore_maybe_log_replay_unit_latency_gate_stats(
        *request, false, false, true, service_cycle);
    request->unit_latency_gate_pending = false;
    request->unit_latency_unit = RTCORE_REPLAY_UNIT_LATENCY_NONE;
    return true;
}

static void rtcore_record_replay_memory_latency_policy_estimate(
    unsigned owner_hw_sid)
{
    if (!rtcore_replay_memory_latency_policy_log_enabled()) {
        return;
    }

    const unsigned cache_line_latency_cycles =
        rtcore_replay_memory_cache_line_latency_config();
    const unsigned memory_wait_latency_cycles =
        rtcore_replay_memory_wait_event_latency_config();
    const unsigned memory_wait_events =
        g_rtcore_replay_memory_demand_estimate_stats
            .explicit_memory_wait_events +
        g_rtcore_replay_memory_demand_estimate_stats
            .overflow_memory_wait_events;

    g_rtcore_replay_memory_latency_policy_stats.policy_evaluations++;
    g_rtcore_replay_memory_latency_policy_stats.cache_line_latency_cycles =
        cache_line_latency_cycles;
    g_rtcore_replay_memory_latency_policy_stats.memory_wait_latency_cycles =
        memory_wait_latency_cycles;
    g_rtcore_replay_memory_latency_policy_stats
        .estimated_cache_latency_cycles =
        g_rtcore_replay_memory_demand_estimate_stats
            .estimated_cache_lines_64b *
        cache_line_latency_cycles;
    g_rtcore_replay_memory_latency_policy_stats
        .estimated_memory_wait_latency_cycles =
        memory_wait_events * memory_wait_latency_cycles;
    g_rtcore_replay_memory_latency_policy_stats
        .estimated_total_memory_latency_cycles =
        g_rtcore_replay_memory_latency_policy_stats
            .estimated_cache_latency_cycles +
        g_rtcore_replay_memory_latency_policy_stats
            .estimated_memory_wait_latency_cycles;

    rtcore_record_replay_memory_latency_blocked_cycle_stats(owner_hw_sid);
    rtcore_maybe_log_replay_memory_latency_policy_stats(owner_hw_sid);
}

static void rtcore_record_replay_memory_demand_estimate(
    const rtcore_replay_lane_request &request)
{
    if (request.next_event_index >= request.events.size()) {
        return;
    }

    const rtcore_compact_trace_event &event =
        request.events[request.next_event_index];
    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    const rtcore_compact_trace_resource_class resource_class =
        rtcore_unpack_compact_trace_resource_class(event);
    const unsigned count = rtcore_unpack_compact_trace_count(event) == 0
                               ? 1
                               : rtcore_unpack_compact_trace_count(event);
    const unsigned bytes = rtcore_unpack_compact_trace_bytes(event);
    bool changed = false;

    if (event_type == RTCORE_TRACE_MEMORY_WAIT) {
        g_rtcore_replay_memory_demand_estimate_stats
            .explicit_memory_wait_events += count;
        changed = true;
    }
    if (event_type == RTCORE_TRACE_NODE_FETCH ||
        event_type == RTCORE_TRACE_PRIMITIVE_FETCH ||
        resource_class == RTCORE_TRACE_RESOURCE_MEMORY) {
        g_rtcore_replay_memory_demand_estimate_stats
            .explicit_fetch_memory_bytes += bytes * count;
        changed = true;
    }
    if (event_type == RTCORE_TRACE_OVERFLOW_SUMMARY &&
        request.timing_trace_overflowed && request.overflow_summary_events) {
        g_rtcore_replay_memory_demand_estimate_stats
            .overflow_memory_wait_events +=
            request.overflow_summary.overflow_memory_wait_count;
        g_rtcore_replay_memory_demand_estimate_stats.overflow_memory_bytes +=
            request.overflow_summary.overflow_memory_bytes;
        changed = true;
    }

    if (!changed) {
        return;
    }
    rtcore_update_replay_memory_demand_totals();
    rtcore_record_replay_memory_latency_policy_estimate(request.owner_hw_sid);
    rtcore_maybe_log_replay_memory_demand_estimate_stats(request.owner_hw_sid);
}

static bool rtcore_step_admitted_replay_request(unsigned thread_uid,
                                                unsigned long long service_cycle = 0);

static bool rtcore_service_banked_ready_state_with_unit_budget_for_owner(
    rtcore_replay_lane_request_state unit_state, unsigned owner_hw_sid,
    unsigned *issue_budget, unsigned *issue_attempts, unsigned *issued,
    unsigned *budget_exhausted,
    rtcore_replay_service_cycle_identity_snapshot *last_identity,
    unsigned long long service_cycle)
{
    bool progressed = false;
    std::set<unsigned> selected_bank_ids;
    while (true) {
        unsigned thread_uid = 0;
        unsigned selected_bank_id = 0;
        if (!rtcore_select_banked_ready_request_for_owner(
                unit_state, owner_hw_sid, &thread_uid, &selected_bank_ids,
                &selected_bank_id)) {
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
        if (!rtcore_step_admitted_replay_request(thread_uid, service_cycle)) {
            rtcore_route_admitted_replay_request(thread_uid, service_cycle);
            break;
        }

        selected_bank_ids.insert(selected_bank_id);
        if (issued) {
            (*issued)++;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, false, true);
        }
        progressed = true;
        if (!rtcore_route_admitted_replay_request(thread_uid, service_cycle)) {
            break;
        }
    }
    return progressed;
}

static bool rtcore_service_replay_completion_ingress_requests_for_owner(
    unsigned owner_hw_sid, unsigned *ingress_budget, unsigned *ingress_attempts,
    unsigned *ingress_issued, unsigned *budget_exhausted,
    rtcore_replay_service_cycle_identity_snapshot *last_identity,
    unsigned long long service_cycle)
{
    bool progressed = false;
    std::set<unsigned> selected_bank_ids;
    while (true) {
        unsigned thread_uid = 0;
        unsigned selected_bank_id = 0;
        if (!rtcore_select_banked_ready_request_for_owner(
                RTCORE_REPLAY_COMPLETION_PENDING, owner_hw_sid,
                &thread_uid, &selected_bank_ids, &selected_bank_id)) {
            break;
        }
        if (ingress_attempts) {
            (*ingress_attempts)++;
        }
        if (!ingress_budget || *ingress_budget == 0) {
            if (budget_exhausted) {
                (*budget_exhausted)++;
            }
            break;
        }

        std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
            g_rtcore_replay_lane_requests.find(thread_uid);
        if (request_it == g_rtcore_replay_lane_requests.end()) {
            break;
        }
        rtcore_record_replay_lane_request_state_identity_read();
        rtcore_record_replay_request_state_read();
        rtcore_replay_lane_request &request = request_it->second;
        rtcore_refresh_replay_lane_request_ready_bits(&request);
        if (!rtcore_replay_request_ready_for_state(
                request, RTCORE_REPLAY_COMPLETION_PENDING)) {
            rtcore_route_admitted_replay_request(thread_uid, service_cycle);
            break;
        }

        const bool publication_was_armed =
            request.v04_live_publication_armed;
        if (!rtcore_prepare_v04_terminal_completion_publication(
                &request, service_cycle)) {
            selected_bank_ids.insert(selected_bank_id);
            if (!publication_was_armed &&
                request.v04_live_publication_armed) {
                progressed = true;
                if (last_identity) {
                    *last_identity =
                        rtcore_make_replay_service_progress_identity(
                            thread_uid, true, false);
                }
            }
            continue;
        }

        (*ingress_budget)--;
        rtcore_record_replay_lane_completion_entry(request);
        selected_bank_ids.insert(selected_bank_id);
        rtcore_mark_replay_request_completed(&request, service_cycle);
        if (ingress_issued) {
            (*ingress_issued)++;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, false, true);
        }
        progressed = true;
    }
    return progressed;
}

static void rtcore_try_service_replay_after_admission(unsigned owner_hw_sid);

static void rtcore_commit_replay_lane_request_state_admission(
    const rtcore_replay_lane_request &request)
{
    rtcore_replay_lane_request admitted_request = request;
    admitted_request.admitted_cycle = 0;
    admitted_request.request_state_bank_id =
        rtcore_replay_request_state_bank_for_request(admitted_request);
    rtcore_refresh_replay_lane_request_ready_bits(&admitted_request);
    g_rtcore_replay_lane_requests[admitted_request.thread_uid] =
        admitted_request;
    rtcore_record_replay_lane_request_state_identity_write();
    rtcore_record_replay_request_state_write();
    rtcore_record_replay_lane_admission_entry(admitted_request);
    if (admitted_request.state == RTCORE_REPLAY_COMPLETED) {
        rtcore_record_replay_lane_completion_entry(admitted_request);
    }
    rtcore_route_admitted_replay_request(request.thread_uid);
}

static bool rtcore_try_drain_replay_lane_request_state_capacity_pending_admissions(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    if (rtcore_retire_lifecycle_busy_for_owner(owner_hw_sid)) {
        return false;
    }

    const bool capacity_gate_enabled =
        rtcore_replay_lane_request_state_capacity_gate_enabled();
    const unsigned capacity = rtcore_replay_lane_request_state_capacity_config();

    bool admitted_any = false;
    for (std::deque<rtcore_replay_lane_request>::iterator it =
             g_rtcore_replay_lane_request_state_capacity_pending_admissions.begin();
         it !=
         g_rtcore_replay_lane_request_state_capacity_pending_admissions.end();) {
        if (it->owner_hw_sid != owner_hw_sid) {
            ++it;
            continue;
        }

        const bool consumes_entry =
            rtcore_replay_lane_request_state_capacity_consumes_entry(*it);
        const unsigned current_occupancy =
            rtcore_count_replay_lane_request_state_occupied_entries_for_owner(
                owner_hw_sid);
        if (capacity_gate_enabled && capacity != 0 && consumes_entry &&
            current_occupancy >= capacity) {
            break;
        }

        if (consumes_entry &&
            !rtcore_replay_lane_state_init_bandwidth_try_consume(
                owner_hw_sid, service_cycle)) {
            g_rtcore_replay_lane_request_state_capacity_gate_stats.evaluations++;
            g_rtcore_replay_lane_request_state_capacity_gate_stats.last_occupancy =
                current_occupancy;
            g_rtcore_replay_lane_request_state_capacity_gate_stats
                .last_capacity_blocked = 0u;
            g_rtcore_replay_lane_request_state_capacity_gate_stats
                .last_init_bandwidth_blocked = 1u;
            g_rtcore_replay_lane_request_state_capacity_gate_stats.last_admitted =
                0u;
            g_rtcore_replay_lane_request_state_capacity_gate_stats.last_released =
                0u;
            rtcore_update_replay_lane_request_state_capacity_max(
                current_occupancy);
            g_rtcore_replay_lane_request_state_capacity_gate_stats
                .lane_state_init_bandwidth_blocked_count++;
            rtcore_maybe_log_replay_lane_request_state_capacity_gate_stats(
                *it, current_occupancy, false, false, false);
            break;
        }

        rtcore_replay_lane_request request = *it;
        it = g_rtcore_replay_lane_request_state_capacity_pending_admissions.erase(
            it);
        const unsigned resulting_occupancy =
            current_occupancy + (consumes_entry ? 1u : 0u);
        if (capacity_gate_enabled && capacity != 0) {
            rtcore_record_replay_lane_request_state_capacity_admitted(
                request, resulting_occupancy);
        } else {
            g_rtcore_replay_lane_request_state_capacity_gate_stats
                .last_init_bandwidth_blocked = 0u;
        }
        rtcore_commit_replay_lane_request_state_admission(request);
        admitted_any = true;
    }
    return admitted_any;
}

static void rtcore_admit_compact_trace_for_replay(ptx_thread_info *thread)
{
    if (!rtcore_replay_admission_enabled() || !thread) {
        return;
    }
    if (rtcore_v04_request_owner_binding_enabled()) {
        // The complete active mask is admitted from the resident binding
        // commit point after every lane has published its compact trace.
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
    request.v04_live_handoff_memory = thread->get_global_memory();
    if (rtcore_v04_live_handoff_publication_enabled() &&
        (!request.v04_shadow_boundary_enabled ||
         !rtcore_replay_memory_unit_path_active() ||
         request.handoff_window_base == 0 ||
         request.v04_live_handoff_memory == NULL)) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_PUBLICATION_FAULT "
                "owner_hw_sid=%u thread_uid=%u lane_id=%u "
                "boundary_enabled=%u memory_path_active=%u "
                "handoff_window_base=0x%llx memory_backing_valid=%u "
                "fault=admission_contract_invalid\n",
                request.owner_hw_sid, request.thread_uid, request.lane_id,
                request.v04_shadow_boundary_enabled ? 1u : 0u,
                rtcore_replay_memory_unit_path_active() ? 1u : 0u,
                request.handoff_window_base,
                request.v04_live_handoff_memory != NULL ? 1u : 0u);
        fflush(stderr);
        abort();
    }
    rtcore_record_replay_resource_route_stats(request);
    request.ready_order = g_rtcore_next_replay_ready_order++;
    if (rtcore_maybe_block_replay_lane_request_state_capacity_admission(
            request, 0)) {
        return;
    }
    rtcore_commit_replay_lane_request_state_admission(request);
    rtcore_try_service_replay_after_admission(thread->get_hw_sid());
}

static bool rtcore_replay_request_done(
    const rtcore_replay_lane_request &request)
{
    rtcore_record_replay_request_state_read();
    return !request.valid || request.state == RTCORE_REPLAY_COMPLETED ||
           request.state == RTCORE_REPLAY_FINAL_WAIT_RETIRE;
}

static void rtcore_prepare_replay_request_completion_ingress(
    rtcore_replay_lane_request *request)
{
    if (!request || !request->valid ||
        request->state == RTCORE_REPLAY_COMPLETION_PENDING ||
        request->state == RTCORE_REPLAY_COMPLETED) {
        return;
    }

    request->state = RTCORE_REPLAY_COMPLETION_PENDING;
    rtcore_record_replay_request_state_write();
    rtcore_refresh_replay_lane_request_ready_bits(request);
}

static const char *rtcore_continuation_packet_kind_name(
    rtcore_continuation_packet_kind kind)
{
    switch (kind) {
    case RTCORE_CONTINUATION_PACKET_FINAL:
        return "FINAL";
    case RTCORE_CONTINUATION_PACKET_CONTINUATION:
        return "CONTINUATION";
    }
    return "UNKNOWN";
}

static bool rtcore_v04_live_handoff_packet_ready(
    const rtcore_continuation_warp_boundary_state &state,
    unsigned *armed_lane_mask, unsigned *committed_lane_mask)
{
    if (armed_lane_mask != NULL) {
        *armed_lane_mask = 0;
    }
    if (committed_lane_mask != NULL) {
        *committed_lane_mask = 0;
    }
    if (!rtcore_v04_live_handoff_publication_enabled()) {
        return true;
    }

    unsigned armed = 0;
    unsigned committed = 0;
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (!rtcore_continuation_request_matches_boundary_key(request,
                                                              state)) {
            continue;
        }
        const unsigned lane_mask = 1u << request.lane_id;
        if ((state.boundary_reached_mask & state.active_mask & lane_mask) ==
            0) {
            continue;
        }
        if (rtcore_v04_live_publication_matches_current_submit(request)) {
            armed |= lane_mask;
        }
        if (rtcore_v04_live_publication_matches_current_submit(request) &&
            request.v04_live_publication_committed) {
            committed |= lane_mask;
        }
    }
    if (armed_lane_mask != NULL) {
        *armed_lane_mask = armed;
    }
    if (committed_lane_mask != NULL) {
        *committed_lane_mask = committed;
    }
    const unsigned expected =
        state.boundary_reached_mask & state.active_mask;
    return armed == expected && committed == expected;
}

static bool rtcore_publish_continuation_return_packet(
    rtcore_continuation_warp_boundary_state *state,
    unsigned long long service_cycle)
{
    if (!state || !state->valid || state->packet_published) {
        return false;
    }

    rtcore_continuation_return_packet packet;
    packet.valid = true;
    packet.kind = state->resume_required_mask == 0
                      ? RTCORE_CONTINUATION_PACKET_FINAL
                      : RTCORE_CONTINUATION_PACKET_CONTINUATION;
    packet.owner_hw_sid = state->owner_hw_sid;
    packet.warp_uid = state->warp_uid;
    packet.warp_id = state->warp_id;
    packet.active_mask = state->active_mask;
    packet.boundary_reached_mask = state->boundary_reached_mask;
    packet.terminal_mask = state->terminal_mask;
    packet.resume_required_mask = state->resume_required_mask;
    packet.shader_required_mask = state->shader_required_mask;
    packet.continuation_depth = state->continuation_depth;
    packet.reason_oracle_anyhit_mask = state->reason_oracle_anyhit_mask;
    packet.reason_oracle_intersection_mask =
        state->reason_oracle_intersection_mask;
    packet.reason_synthetic_split_mask = state->reason_synthetic_split_mask;
    packet.reason_final_mask = state->reason_final_mask;
    packet.reason_unsupported_mask = state->reason_unsupported_mask;
    if (packet.reason_synthetic_split_mask != 0) {
        packet.kind = RTCORE_CONTINUATION_PACKET_CONTINUATION;
        state->packet_published = true;
        g_rtcore_continuation_stats.rtcore_continuation_packet_count++;
        g_rtcore_continuation_stats.rtcore_continuation_lane_count +=
            rtcore_continuation_count_lanes(packet.boundary_reached_mask &
                                            packet.active_mask);
        printf("GPGPU-Sim RTCORE_CONTINUATION_RETURN_PACKET "
               "packet_kind=CONTINUATION boundary_mode=synthetic_internal "
               "owner_hw_sid=%u warp_uid=%u warp_id=%u "
               "active_mask=0x%08x boundary_reached_mask=0x%08x "
               "terminal_mask=0x%08x resume_required_mask=0x%08x "
               "shader_required_mask=0x%08x continuation_depth=%u "
               "reason_synthetic_split_mask=0x%08x service_cycle=%llu\n",
               packet.owner_hw_sid, packet.warp_uid, packet.warp_id,
               packet.active_mask, packet.boundary_reached_mask,
               packet.terminal_mask, packet.resume_required_mask,
               packet.shader_required_mask, packet.continuation_depth,
               packet.reason_synthetic_split_mask, service_cycle);
        fflush(stdout);
        rtcore_mark_resident_warp_continuation_wakeup(packet, service_cycle);
        return true;
    }
    if (state->pending_packet_valid) {
        packet = state->pending_packet;
    } else {
        rtcore_populate_continuation_packet_handoff_summaries(
            &packet, *state, service_cycle);
        packet.terminal_mask = 0;
        packet.resume_required_mask = 0;
        packet.shader_required_mask = 0;
        for (unsigned lane = 0; lane < 32; ++lane) {
            const unsigned lane_mask = 1u << lane;
            if ((packet.boundary_reached_mask & lane_mask) == 0) {
                continue;
            }
            const unsigned reason = packet.v_result[lane] & 0xffu;
            if (reason ==
                    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED ||
                reason ==
                    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED) {
                packet.resume_required_mask |= lane_mask;
                packet.shader_required_mask |= lane_mask;
            } else if (reason ==
                           RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS ||
                       reason ==
                           RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY ||
                       reason ==
                           RTCORE_REPLAY_CONTINUATION_PACKET_REASON_TRACE_DONE_NO_SHADER) {
                packet.terminal_mask |= lane_mask;
            } else {
                packet.reason_unsupported_mask |= lane_mask;
            }
            if (reason ==
                    RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED &&
                (packet.handoff_software_return_valid_mask & lane_mask) != 0) {
                const unsigned metadata = packet.handoff_words[lane][15];
                const unsigned word_count = (metadata >> 8) & 0xffu;
                const unsigned base_word = (metadata >> 16) & 0xffu;
                const unsigned format = (metadata >> 24) & 0xffu;
                const bool metadata_valid =
                    (word_count == 0 && base_word == 0 && format == 0) ||
                    (word_count > 0 && word_count <= 4 && base_word == 16 &&
                     format == 2);
                if (metadata_valid && word_count > 0) {
                    packet.reported_attribute_metadata_valid_mask |=
                        lane_mask;
                    packet.inline_payload_location_valid_mask |= lane_mask;
                }
            }
        }
        packet.kind = packet.resume_required_mask == 0
                          ? RTCORE_CONTINUATION_PACKET_FINAL
                          : RTCORE_CONTINUATION_PACKET_CONTINUATION;
        state->pending_packet = packet;
        state->pending_packet_valid = true;
    }

    unsigned live_armed_lane_mask = 0;
    unsigned live_committed_lane_mask = 0;
    if (!rtcore_v04_live_handoff_packet_ready(
            *state, &live_armed_lane_mask, &live_committed_lane_mask)) {
        if (!state->live_publication_wait_logged) {
            printf("GPGPU-Sim RTCORE_V04_LIVE_HANDOFF_COMPLETION_WAIT "
                   "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                   "active_mask=0x%08x boundary_reached_mask=0x%08x "
                   "armed_lane_mask=0x%08x committed_lane_mask=0x%08x "
                   "completion_packet_published=0 wait_reason=write_ack "
                   "service_cycle=%llu\n",
                   state->owner_hw_sid, state->warp_uid, state->warp_id,
                   state->active_mask, state->boundary_reached_mask,
                   live_armed_lane_mask, live_committed_lane_mask,
                   service_cycle);
            fflush(stdout);
            state->live_publication_wait_logged = true;
        }
        return false;
    }
    state->packet_published = true;

    g_rtcore_continuation_stats.rtcore_continuation_packet_count++;
    g_rtcore_continuation_stats.rtcore_continuation_lane_count +=
        rtcore_continuation_count_lanes(state->boundary_reached_mask &
                                        state->active_mask);

    printf("GPGPU-Sim RTCORE_CONTINUATION_RETURN_PACKET "
           "packet_kind=%s owner_hw_sid=%u warp_uid=%u warp_id=%u "
           "active_mask=0x%08x boundary_reached_mask=0x%08x "
           "terminal_mask=0x%08x resume_required_mask=0x%08x "
           "shader_required_mask=0x%08x continuation_depth=%u "
           "reason_miss_mask=0x%08x reason_closest_hit_mask=0x%08x "
           "reason_any_hit_mask=0x%08x reason_intersection_mask=0x%08x "
           "reason_trace_done_mask=0x%08x reason_fault_mask=0x%08x "
           "reason_unsupported_mask=0x%08x "
           "context_profile_valid_mask=0x%08x "
           "reported_attribute_metadata_valid_mask=0x%08x "
           "inline_payload_location_valid_mask=0x%08x "
           "service_cycle=%llu\n",
           rtcore_continuation_packet_kind_name(packet.kind),
           packet.owner_hw_sid, packet.warp_uid, packet.warp_id,
           packet.active_mask, packet.boundary_reached_mask,
           packet.terminal_mask, packet.resume_required_mask,
           packet.shader_required_mask, packet.continuation_depth,
           rtcore_continuation_packet_reason_mask(
               packet, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS),
           rtcore_continuation_packet_reason_mask(
               packet,
               RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY),
           rtcore_continuation_packet_reason_mask(
               packet,
               RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED),
           rtcore_continuation_packet_reason_mask(
               packet,
               RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED),
           rtcore_continuation_packet_reason_mask(
               packet,
               RTCORE_REPLAY_CONTINUATION_PACKET_REASON_TRACE_DONE_NO_SHADER),
           rtcore_continuation_packet_reason_mask(
               packet, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_FAULT),
           packet.reason_unsupported_mask, packet.context_profile_valid_mask,
           packet.reported_attribute_metadata_valid_mask,
           packet.inline_payload_location_valid_mask, service_cycle);
    fflush(stdout);

    rtcore_log_replay_continuation_packet_schema(
        "continuation_return_packet", packet.owner_hw_sid, packet.warp_uid,
        packet.warp_id, packet.active_mask, packet.boundary_reached_mask,
        packet.terminal_mask, packet.resume_required_mask,
        packet.handoff_selector_valid_mask,
        packet.handoff_candidate_valid_mask,
        packet.handoff_software_return_valid_mask,
        packet.context_profile_valid_mask,
        packet.reported_attribute_metadata_valid_mask,
        packet.inline_payload_location_valid_mask,
        rtcore_continuation_packet_reason_mask(
            packet, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS),
        rtcore_continuation_packet_reason_mask(
            packet,
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY),
        rtcore_continuation_packet_reason_mask(
            packet, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED),
        rtcore_continuation_packet_reason_mask(
            packet,
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED),
        rtcore_continuation_packet_reason_mask(
            packet,
            RTCORE_REPLAY_CONTINUATION_PACKET_REASON_TRACE_DONE_NO_SHADER),
        rtcore_continuation_packet_reason_mask(
            packet, RTCORE_REPLAY_CONTINUATION_PACKET_REASON_FAULT),
        packet.reason_unsupported_mask, packet.continuation_depth,
        service_cycle);
    rtcore_publish_scoreboard_visible_continuation_packet(packet, service_cycle);

    if (packet.kind == RTCORE_CONTINUATION_PACKET_CONTINUATION) {
        rtcore_mark_resident_warp_continuation_wakeup(packet, service_cycle);
    }
    return true;
}

static void rtcore_initialize_continuation_boundary_state_from_request(
    rtcore_continuation_warp_boundary_state *state,
    const rtcore_replay_lane_request &request)
{
    if (!state) {
        return;
    }

    state->valid = true;
    state->owner_hw_sid = request.owner_hw_sid;
    state->warp_uid = request.warp_uid;
    state->warp_id = request.warp_id;
    state->active_mask = request.active_mask;
    state->boundary_reached_mask = 0;
    state->terminal_mask = 0;
    state->resume_required_mask = 0;
    state->shader_required_mask = 0;
    state->continuation_depth = request.continuation_depth + 1;
    state->packet_published = false;
    state->pending_packet_valid = false;
    state->live_publication_wait_logged = false;
    state->pending_packet = rtcore_continuation_return_packet();
    state->reason_oracle_anyhit_mask = 0;
    state->reason_oracle_intersection_mask = 0;
    state->reason_synthetic_split_mask = 0;
    state->reason_final_mask = 0;
    state->reason_unsupported_mask = 0;
    rtcore_seed_continuation_boundary_state_from_completed_lanes(state);
}

static void rtcore_record_continuation_boundary_reason(
    const char *boundary_reason)
{
    if (!boundary_reason) {
        return;
    }
    if (strcmp(boundary_reason, "synthetic_split") == 0) {
        g_rtcore_continuation_stats
            .rtcore_continuation_synthetic_boundary_count++;
    } else if (strcmp(boundary_reason, "oracle_anyhit") == 0) {
        g_rtcore_continuation_stats
            .rtcore_continuation_oracle_anyhit_boundary_count++;
    } else if (strcmp(boundary_reason, "oracle_intersection") == 0) {
        g_rtcore_continuation_stats
            .rtcore_continuation_oracle_intersection_boundary_count++;
    }
}

static bool rtcore_find_boundary_candidate_snapshot(
    const rtcore_replay_lane_request &request, unsigned event_index,
    rtcore_boundary_candidate_snapshot *snapshot)
{
    if (event_index >= request.events.size()) return false;
    const unsigned event_seq = request.events[event_index].event_seq;
    for (std::vector<rtcore_boundary_candidate_snapshot>::const_iterator it =
             request.boundary_candidates.begin();
         it != request.boundary_candidates.end(); ++it) {
        if (it->valid && it->event_seq == event_seq) {
            if (snapshot) *snapshot = *it;
            return true;
        }
    }
    return false;
}

static bool rtcore_apply_v04_shadow_consumed_traversal_event(
    rtcore_replay_lane_request *request, unsigned consumed_event_index)
{
    if (request == NULL || !request->valid) return false;
    if (!request->v04_shadow_boundary_enabled) return true;
    if (consumed_event_index >= request->events.size()) return false;

    const rtcore_compact_trace_event &event =
        request->events[consumed_event_index];
    if (rtcore_unpack_compact_trace_event_type(event) !=
            RTCORE_TRACE_PRIMITIVE_TEST ||
        (rtcore_unpack_compact_trace_flags(event) & 0x0fu) !=
            RTCORE_TRACE_PRIMITIVE_KIND_TRIANGLE_TEST) {
        return true;
    }

    rtcore_boundary_candidate_snapshot candidate;
    if (!rtcore_find_boundary_candidate_snapshot(
            *request, consumed_event_index, &candidate)) {
        // Triangle misses, farther hits, and any-hit candidates do not commit
        // traversal-owned state at the primitive-test event.
        return true;
    }
    if (!candidate.valid ||
        candidate.v04_boundary_values.geometry_type !=
            rtcore::abi_v04::shadow::kBoundaryGeometryTriangle) {
        return false;
    }

    rtcore::abi_v04::shadow::boundary_values next_terminal;
    if (!rtcore::abi_v04::shadow::
            commit_triangle_candidate_if_strictly_closer(
                request->v04_replay_committed_boundary_values,
                candidate.v04_boundary_values, &next_terminal)) {
        return false;
    }
    const bool selected =
        rtcore::abi_v04::shadow::triangle_candidate_is_strictly_closer(
            request->v04_replay_committed_boundary_values,
            candidate.v04_boundary_values);
    request->v04_replay_committed_boundary_values = next_terminal;
    const bool compatibility_summary_selected =
        selected && rtcore_v04_functional_shader_return_authority_enabled();
    if (compatibility_summary_selected) {
        const rtcore::abi_v04::shadow::boundary_values &selected_values =
            candidate.v04_boundary_values;
        request->hit_geometry_summary_valid = true;
        request->closest_hit_kind = selected_values.hit_kind;
        request->closest_hit_geometry_type = 0x01u;
        request->closest_hit_geometry_index = selected_values.geometry_index;
        request->closest_hit_primitive_index =
            selected_values.primitive_index;
        request->closest_hit_instance_index =
            selected_values.instance_custom_index;
        request->instance_sbt_contribution_valid = true;
        request->instance_sbt_contribution =
            selected_values.instance_sbt_contribution;
    }
    printf("GPGPU-Sim RTCORE_V04_SHADOW_OPAQUE_EVENT_COMMIT "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u event_seq=%u "
           "selected=%u candidate_t_fp32=0x%08x "
           "replay_order_only=1 oracle_final_preload=0 "
           "compatibility_summary_updated=%u "
           "compatibility_summary_source=%s\n",
           request->owner_hw_sid, request->thread_uid, request->lane_id,
           candidate.event_seq, selected ? 1u : 0u,
           candidate.v04_boundary_values.boundary_ray_tmax_fp32,
           compatibility_summary_selected ? 1u : 0u,
           compatibility_summary_selected ? "v04_event_candidate"
                                          : "unchanged_gate_off");
    fflush(stdout);
    return true;
}

static bool rtcore_mark_continuation_boundary(
    rtcore_replay_lane_request *request, unsigned long long service_cycle,
    const char *boundary_reason, unsigned boundary_event_index)
{
    if (!request || !request->valid || request->continuation_boundary_pending ||
        !rtcore_continuation_request_has_warp_metadata(*request)) {
        return false;
    }

    const unsigned lane_mask =
        rtcore_continuation_lane_mask(request->lane_id);
    if ((request->active_mask & lane_mask) == 0) {
        return false;
    }

    request->continuation_boundary_pending = true;
    request->state = RTCORE_REPLAY_WAITING_SHADER;
    rtcore_record_replay_request_state_write();
    rtcore_refresh_replay_lane_request_ready_bits(request);

    const rtcore_replay_warp_completion_entry_key key =
        rtcore_make_continuation_warp_key(*request);
    rtcore_continuation_warp_boundary_state &state =
        g_rtcore_continuation_warp_boundary_states[key];
    if (!state.valid) {
        rtcore_initialize_continuation_boundary_state_from_request(
            &state, *request);
    }

    state.boundary_reached_mask |= lane_mask;
    state.resume_required_mask |= lane_mask;
    state.shader_required_mask |= lane_mask;
    rtcore_mark_continuation_reason_mask(&state, lane_mask, boundary_reason);
    const bool shader_boundary =
        boundary_reason != NULL &&
        (strcmp(boundary_reason, "oracle_anyhit") == 0 ||
         strcmp(boundary_reason, "oracle_intersection") == 0);
    if (shader_boundary &&
        !rtcore_find_boundary_candidate_snapshot(
            *request, boundary_event_index, &request->boundary_candidate)) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_CONTINUATION_BOUNDARY_CANDIDATE_FAULT "
                "owner_hw_sid=%u thread_uid=%u lane_id=%u "
                "boundary_event_index=%u fault=event_local_candidate_missing\n",
                request->owner_hw_sid, request->thread_uid, request->lane_id,
                boundary_event_index);
        abort();
    }
    state.boundary_candidates[request->lane_id] = request->boundary_candidate;
    rtcore_record_continuation_boundary_reason(boundary_reason);
    const bool packet_ready =
        (state.boundary_reached_mask & state.active_mask) ==
        state.active_mask;

    printf("GPGPU-Sim RTCORE_CONTINUATION_LANE_BOUNDARY "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "warp_id=%u active_mask=0x%08x boundary_reached_mask=0x%08x "
           "terminal_mask=0x%08x resume_required_mask=0x%08x "
           "shader_required_mask=0x%08x continuation_depth=%u "
           "next_event_index=%u segment_event_count=%u packet_ready=%u "
           "boundary_reason=%s boundary_candidate_snapshot_valid=%u "
           "boundary_event_seq=%u boundary_shader_counter=%u "
           "boundary_hit_data_ref=0x%llx service_cycle=%llu\n",
           request->owner_hw_sid, request->thread_uid, request->lane_id,
           request->warp_uid, request->warp_id, state.active_mask,
           state.boundary_reached_mask, state.terminal_mask,
           state.resume_required_mask, state.shader_required_mask,
           state.continuation_depth, request->next_event_index,
           request->continuation_segment_event_count, packet_ready ? 1u : 0u,
           boundary_reason ? boundary_reason : "unknown",
           request->boundary_candidate.valid,
           request->boundary_candidate.event_seq,
           request->boundary_candidate.shader_counter,
           (unsigned long long)request->boundary_candidate.hit_data_ref,
           service_cycle);
    fflush(stdout);

    if (packet_ready) {
        rtcore_publish_continuation_return_packet(&state, service_cycle);
    }
    return true;
}

static bool rtcore_maybe_mark_synthetic_continuation_boundary(
    rtcore_replay_lane_request *request, unsigned long long service_cycle)
{
    if (!request || !request->valid || request->continuation_boundary_pending ||
        !rtcore_continuation_request_has_warp_metadata(*request)) {
        return false;
    }
    if (rtcore_continuation_model_config() !=
        RTCORE_CONTINUATION_MODEL_SYNTHETIC_SPLIT) {
        return false;
    }

    const unsigned segment_budget =
        rtcore_continuation_segment_event_budget_config();
    const unsigned max_resubmits =
        rtcore_continuation_max_resubmits_per_lane_config();
    if (segment_budget == 0 || max_resubmits == 0 ||
        request->continuation_depth >= max_resubmits ||
        request->continuation_segment_event_count < segment_budget) {
        return false;
    }

    return rtcore_mark_continuation_boundary(request, service_cycle,
                                             "synthetic_split", UINT_MAX);
}

static const char *rtcore_oracle_shader_boundary_reason_for_event(
    const rtcore_replay_lane_request &request, unsigned event_index)
{
    if (event_index >= request.events.size()) {
        return NULL;
    }

    const rtcore_compact_trace_event &event = request.events[event_index];
    const rtcore_compact_trace_event_type event_type =
        rtcore_unpack_compact_trace_event_type(event);
    const unsigned flags = rtcore_unpack_compact_trace_flags(event);

    const char *reason = NULL;
    if (event_type == RTCORE_TRACE_HIT_UPDATE &&
        (flags & 0xffu) == RTCORE_TRACE_HIT_UPDATE_KIND_ANY_HIT &&
        request.oracle_anyhit_candidate_count > 0) {
        reason = "oracle_anyhit";
    }

    if (event_type == RTCORE_TRACE_PRIMITIVE_TEST &&
        (flags & 0x0fu) == RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_DEFERRED &&
        (flags & 0x20u) != 0 &&
        request.oracle_requires_intersection_shader) {
        reason = "oracle_intersection";
    }
    const char *override_reason =
        getenv("VULKAN_SIM_RTCORE_TEST_BOUNDARY_REASON_OVERRIDE");
    if (reason && override_reason && strcmp(override_reason, "any_hit") == 0) {
        return "oracle_anyhit";
    }
    if (reason && override_reason &&
        strcmp(override_reason, "intersection") == 0) {
        return "oracle_intersection";
    }
    return reason;
}

static bool rtcore_maybe_mark_oracle_shader_continuation_boundary(
    rtcore_replay_lane_request *request, unsigned consumed_event_index,
    unsigned long long service_cycle)
{
    if (!request || !request->valid || request->continuation_boundary_pending ||
        !rtcore_continuation_request_has_warp_metadata(*request)) {
        return false;
    }
    if (rtcore_continuation_model_config() !=
        RTCORE_CONTINUATION_MODEL_ORACLE_SHADER_BOUNDARY) {
        return false;
    }

    const unsigned max_resubmits =
        rtcore_continuation_max_resubmits_per_lane_config();
    if (max_resubmits == 0 ||
        request->continuation_depth >= max_resubmits) {
        return false;
    }

    const char *reason = rtcore_oracle_shader_boundary_reason_for_event(
        *request, consumed_event_index);
    if (!reason) {
        return false;
    }

    return rtcore_mark_continuation_boundary(request, service_cycle, reason,
                                             consumed_event_index);
}

static bool rtcore_replay_advance_lane_request(
    rtcore_replay_lane_request *request, unsigned long long service_cycle = 0)
{
    if (!request || !request->valid) {
        return false;
    }
    if (request->continuation_boundary_pending) {
        return false;
    }
    if (request->events.empty()) {
        rtcore_prepare_replay_request_completion_ingress(request);
        return true;
    }
    if (request->next_event_index >= request->events.size()) {
        rtcore_prepare_replay_request_completion_ingress(request);
        return true;
    }

    const unsigned consumed_event_index = request->next_event_index;
    if (!rtcore_apply_v04_shadow_consumed_traversal_event(
            request, consumed_event_index)) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_SHADOW_TRAVERSAL_EVENT_FAULT "
                "owner_hw_sid=%u thread_uid=%u lane_id=%u event_index=%u\n",
                request->owner_hw_sid, request->thread_uid, request->lane_id,
                consumed_event_index);
        abort();
    }
    request->next_event_index++;
    request->continuation_segment_event_count++;
    if (rtcore_maybe_mark_oracle_shader_continuation_boundary(
            request, consumed_event_index, service_cycle)) {
        return true;
    }
    if (request->next_event_index >= request->events.size()) {
        rtcore_prepare_replay_request_completion_ingress(request);
        return true;
    }
    if (rtcore_maybe_mark_synthetic_continuation_boundary(request,
                                                          service_cycle)) {
        return true;
    }

    request->state = rtcore_classify_replay_state(
        rtcore_unpack_compact_trace_event_type(
            request->events[request->next_event_index]));
    rtcore_record_replay_request_state_write();
    rtcore_refresh_replay_lane_request_ready_bits(request);
    return true;
}

static bool rtcore_step_admitted_replay_request(unsigned thread_uid,
                                                unsigned long long service_cycle)
{
    std::map<unsigned, rtcore_replay_lane_request>::iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    rtcore_record_replay_lane_request_state_identity_read();
    if (rtcore_replay_request_done(it->second)) {
        return false;
    }
    if (it->second.memory_address_gen_latency_gate_pending) {
        if (!rtcore_replay_memory_address_gen_latency_ready(
                &it->second, service_cycle)) {
            return false;
        }
    }
    bool memory_contention_resolved_this_step = false;
    bool v02_lsu_response_wait_resolved_this_step = false;
    if (it->second.v02_lsu_response_wait_gate_pending) {
        if (!rtcore_memory_unit_response_wait_ready(&it->second,
                                                     service_cycle)) {
            return false;
        }
        v02_lsu_response_wait_resolved_this_step = true;
    }
    if (it->second.memory_contention_gate_pending) {
        if (!rtcore_replay_memory_contention_gate_ready(&it->second,
                                                        service_cycle)) {
            return false;
        }
        memory_contention_resolved_this_step = true;
    }
    if (it->second.memory_wake_latency_gate_pending) {
        if (!rtcore_replay_memory_wake_latency_gate_ready(&it->second,
                                                          service_cycle)) {
            return false;
        }
    } else if (it->second.unit_latency_gate_pending) {
        if (!rtcore_replay_unit_latency_gate_ready(&it->second,
                                                   service_cycle)) {
            return false;
        }
    } else {
        rtcore_record_replay_memory_demand_estimate(it->second);
        if (it->second.next_event_index < it->second.events.size() &&
            rtcore_unpack_compact_trace_event_type(
                it->second.events[it->second.next_event_index]) ==
                RTCORE_TRACE_OVERFLOW_SUMMARY) {
            rtcore_record_replay_overflow_summary_estimate(it->second);
        }
        if (!v02_lsu_response_wait_resolved_this_step &&
            rtcore_maybe_arm_memory_unit_response_wait(&it->second,
                                                        service_cycle)) {
            return true;
        }
        if (!v02_lsu_response_wait_resolved_this_step &&
            !memory_contention_resolved_this_step &&
            rtcore_maybe_arm_replay_memory_contention_gate(&it->second,
                                                           service_cycle)) {
            return true;
        }
        if (!v02_lsu_response_wait_resolved_this_step &&
            rtcore_maybe_arm_replay_memory_wake_latency_gate(&it->second,
                                                             service_cycle)) {
            return true;
        }
        const bool suppress_generic_stack_load =
            it->second.next_event_index < it->second.events.size() &&
            rtcore_memory_unit_response_wait_manages_stack_load_event(
                it->second.events[it->second.next_event_index]);
        if (!suppress_generic_stack_load) {
            rtcore_maybe_enqueue_v02_lsu_stack_sideband(it->second,
                                                        service_cycle);
        }
        if (rtcore_maybe_arm_replay_unit_latency_gate(&it->second,
                                                      service_cycle)) {
            return true;
        }
    }
    const bool advanced =
        rtcore_replay_advance_lane_request(&it->second, service_cycle);
    if (advanced && rtcore_replay_request_done(it->second)) {
        rtcore_record_replay_lane_completion_entry(it->second.thread_uid);
    }
    return advanced;
}

static void rtcore_collect_replay_request_owners(std::set<unsigned> *owners)
{
    if (!owners) {
        return;
    }
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (request.valid) {
            owners->insert(request.owner_hw_sid);
        }
    }
}

static bool rtcore_service_replay_non_completion_ready_requests_for_owner(
    unsigned owner_hw_sid, rtcore_replay_issue_budget budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL,
    unsigned long long service_cycle = 0)
{
    bool progressed = false;
    progressed |= rtcore_service_banked_ready_state_with_unit_budget_for_owner(
        RTCORE_REPLAY_ISSUED_NODE, owner_hw_sid,
        &budget.node_issue_budget,
        &g_rtcore_replay_unit_arbitration_stats.node_unit_issue_attempts,
        &g_rtcore_replay_unit_arbitration_stats.node_unit_issued,
        &g_rtcore_replay_unit_arbitration_stats.node_unit_budget_exhausted,
        last_identity, service_cycle);
    progressed |= rtcore_service_banked_ready_state_with_unit_budget_for_owner(
        RTCORE_REPLAY_ISSUED_PRIMITIVE, owner_hw_sid,
        &budget.primitive_issue_budget,
        &g_rtcore_replay_unit_arbitration_stats.primitive_unit_issue_attempts,
        &g_rtcore_replay_unit_arbitration_stats.primitive_unit_issued,
        &g_rtcore_replay_unit_arbitration_stats
             .primitive_unit_budget_exhausted,
        last_identity, service_cycle);
    progressed |= rtcore_service_banked_ready_state_with_unit_budget_for_owner(
        RTCORE_REPLAY_ISSUED_STACK, owner_hw_sid,
        &budget.stack_issue_budget,
        &g_rtcore_replay_unit_arbitration_stats.stack_unit_issue_attempts,
        &g_rtcore_replay_unit_arbitration_stats.stack_unit_issued,
        &g_rtcore_replay_unit_arbitration_stats.stack_unit_budget_exhausted,
        last_identity, service_cycle);
    return progressed;
}

static bool rtcore_service_replay_scoreboard_handoff_requests_for_owner(
    unsigned owner_hw_sid, rtcore_replay_issue_budget budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL,
    unsigned long long service_cycle = 0)
{
    const bool ingress_progressed =
        rtcore_service_replay_completion_ingress_requests_for_owner(
            owner_hw_sid, &budget.warp_completion_ingress_budget,
            &g_rtcore_replay_unit_arbitration_stats
                 .warp_completion_ingress_attempts,
            &g_rtcore_replay_unit_arbitration_stats.warp_completion_ingress_issued,
            &g_rtcore_replay_unit_arbitration_stats
                 .warp_completion_ingress_budget_exhausted,
            last_identity, service_cycle);
    const unsigned delivered =
        rtcore_service_completed_warp_entry_handoffs_for_owner(
            owner_hw_sid, service_cycle);
    return ingress_progressed || delivered > 0;
}

static bool rtcore_wake_request_state_unit_replay_request(
    unsigned thread_uid, unsigned long long service_cycle)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    if (!it->second.unit_latency_gate_pending) {
        return false;
    }
    if (!rtcore_step_admitted_replay_request(thread_uid, service_cycle)) {
        return false;
    }
    return rtcore_route_admitted_replay_request(thread_uid, service_cycle);
}

static bool rtcore_replay_request_state_has_unit_wake_service_work(
    const rtcore_replay_lane_request &request)
{
    return request.valid && request.unit_latency_gate_pending;
}

static bool rtcore_replay_request_state_unit_wake_service_ready(
    const rtcore_replay_lane_request &request,
    unsigned long long service_cycle)
{
    if (!rtcore_replay_request_state_has_unit_wake_service_work(request)) {
        return false;
    }
    if (request.unit_latency_gate_pending &&
        service_cycle >= request.unit_latency_ready_cycle) {
        return true;
    }
    return false;
}

static bool rtcore_select_banked_unit_wake_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid,
    unsigned long long service_cycle)
{
    const unsigned bank_count =
        rtcore_replay_v03_hw_request_state_bank_count_config();
    if (bank_count == 0) {
        return false;
    }

    std::vector<rtcore_replay_v03_hw_banked_ready_candidate> candidates(
        bank_count);
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (request.owner_hw_sid != owner_hw_sid ||
            !rtcore_replay_request_state_unit_wake_service_ready(
                request, service_cycle)) {
            continue;
        }
        const unsigned bank_id = request.request_state_bank_id % bank_count;
        rtcore_replay_v03_hw_banked_ready_candidate &candidate =
            candidates[bank_id];
        if (!candidate.valid || request.ready_order < candidate.ready_order) {
            candidate.valid = true;
            candidate.thread_uid = request.thread_uid;
            candidate.ready_order = request.ready_order;
        }
    }

    const unsigned long long cursor_key =
        rtcore_replay_v03_hw_ready_bank_cursor_key(owner_hw_sid,
                                                   RTCORE_REPLAY_READY);
    unsigned cursor =
        g_rtcore_replay_v03_hw_ready_bank_rr_cursor_by_owner_unit[cursor_key] %
        bank_count;
    for (unsigned offset = 0; offset < bank_count; ++offset) {
        const unsigned bank_id = (cursor + offset) % bank_count;
        if (!candidates[bank_id].valid) {
            continue;
        }
        if (thread_uid) {
            *thread_uid = candidates[bank_id].thread_uid;
        }
        g_rtcore_replay_v03_hw_ready_bank_rr_cursor_by_owner_unit[cursor_key] =
            (bank_id + 1) % bank_count;
        return true;
    }
    return false;
}

static bool rtcore_select_replay_request_state_unit_wake_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid,
    unsigned long long service_cycle)
{
    return rtcore_select_banked_unit_wake_request_for_owner(
        owner_hw_sid, thread_uid, service_cycle);
}

static bool rtcore_service_request_state_unit_wake_requests_for_owner(
    unsigned owner_hw_sid, unsigned wake_budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL,
    unsigned long long service_cycle = 0)
{
    bool progressed = false;
    while (wake_budget > 0) {
        unsigned thread_uid = 0;
        if (!rtcore_select_replay_request_state_unit_wake_request_for_owner(
                owner_hw_sid, &thread_uid, service_cycle)) {
            break;
        }
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .wake_attempt_count++;
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .request_state_unit_wake_attempt_count++;
        if (!rtcore_wake_request_state_unit_replay_request(thread_uid,
                                                           service_cycle)) {
            rtcore_route_admitted_replay_request(thread_uid, service_cycle);
            break;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, false, true);
        }
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .wake_progress_count++;
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .request_state_unit_wake_progress_count++;
        wake_budget--;
        progressed = true;
    }
    return progressed;
}

static bool rtcore_wake_memory_response_replay_request(
    unsigned thread_uid, unsigned long long service_cycle)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    rtcore_record_replay_lane_request_state_identity_read();
    rtcore_record_replay_request_state_read();
    if (it->second.state != RTCORE_REPLAY_ISSUED_MEMORY) {
        return false;
    }
    if (!rtcore_step_admitted_replay_request(thread_uid, service_cycle)) {
        return false;
    }
    return rtcore_route_admitted_replay_request(thread_uid, service_cycle);
}

static bool rtcore_issue_ready_memory_replay_request(
    unsigned thread_uid, unsigned long long service_cycle)
{
    std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
        g_rtcore_replay_lane_requests.find(thread_uid);
    if (it == g_rtcore_replay_lane_requests.end()) {
        return false;
    }
    rtcore_record_replay_lane_request_state_identity_read();
    rtcore_record_replay_request_state_read();
    if (it->second.state != RTCORE_REPLAY_ISSUED_MEMORY) {
        return false;
    }
    if (!rtcore_step_admitted_replay_request(thread_uid, service_cycle)) {
        return false;
    }
    return rtcore_route_admitted_replay_request(thread_uid, service_cycle);
}

static bool rtcore_replay_memory_outstanding_capacity_can_issue(
    unsigned owner_hw_sid, unsigned new_transaction_count)
{
    if (new_transaction_count == 0) {
        return true;
    }
    unsigned outstanding_entry_count = 0;
    rtcore_count_replay_memory_outstanding_for_owner(
        owner_hw_sid, &outstanding_entry_count, NULL, NULL);
    if (outstanding_entry_count + new_transaction_count <=
        rtcore_replay_memory_outstanding_capacity_config()) {
        return true;
    }

    g_rtcore_replay_v03_hw_memory_outstanding_stats
        .memory_outstanding_capacity_blocked_count++;
    if (g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_outstanding_capacity_blocked_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_capacity_blocked_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_capacity_blocked_count =
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_outstanding_capacity_blocked_count;
    }
    return false;
}

static bool rtcore_service_ready_memory_replay_requests_for_owner_with_budget(
    unsigned owner_hw_sid, unsigned *issue_budget, unsigned *address_gen_budget,
    unsigned *alloc_budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity,
    unsigned long long service_cycle)
{
    bool progressed = false;
    if (!issue_budget || !address_gen_budget || !alloc_budget) {
        return false;
    }
    std::set<unsigned> selected_bank_ids;
    while (true) {
        unsigned thread_uid = 0;
        unsigned selected_bank_id = 0;
        if (!rtcore_select_banked_ready_request_for_owner(
                RTCORE_REPLAY_ISSUED_MEMORY, owner_hw_sid, &thread_uid,
                &selected_bank_ids, &selected_bank_id)) {
            break;
        }
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_ready_issue_attempt_count++;

        std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
            g_rtcore_replay_lane_requests.find(thread_uid);
        if (request_it == g_rtcore_replay_lane_requests.end()) {
            break;
        }
        rtcore_replay_lane_request &request = request_it->second;
        bool address_gen_completed =
            rtcore_replay_memory_address_gen_completed_for_current_event(
                request);
        const bool address_gen_applicable =
            request.next_event_index < request.events.size() &&
            rtcore_replay_memory_address_gen_applies_to_event(
                request.events[request.next_event_index]);
        if (request.memory_address_gen_latency_gate_pending ||
            (!address_gen_completed && address_gen_applicable)) {
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_address_gen_attempt_count++;
        }
        if (request.memory_address_gen_latency_gate_pending) {
            if (!rtcore_replay_memory_address_gen_latency_ready(
                    &request, service_cycle)) {
                selected_bank_ids.insert(selected_bank_id);
                continue;
            }
            progressed = true;
            address_gen_completed =
                rtcore_replay_memory_address_gen_completed_for_current_event(
                    request);
        }
        if (!address_gen_completed && address_gen_applicable) {
            if (*address_gen_budget == 0) {
                g_rtcore_replay_v03_hw_memory_outstanding_stats
                    .memory_address_gen_blocked_count++;
                if (g_rtcore_replay_v03_hw_memory_outstanding_stats
                        .memory_address_gen_blocked_count >
                    g_rtcore_replay_v03_hw_memory_outstanding_stats
                        .max_memory_address_gen_blocked_count) {
                    g_rtcore_replay_v03_hw_memory_outstanding_stats
                        .max_memory_address_gen_blocked_count =
                        g_rtcore_replay_v03_hw_memory_outstanding_stats
                            .memory_address_gen_blocked_count;
                }
                break;
            }
            if (rtcore_maybe_arm_memory_address_gen_latency(&request,
                                                            service_cycle)) {
                (*address_gen_budget)--;
                g_rtcore_replay_v03_hw_memory_outstanding_stats
                    .memory_address_gen_issued_count++;
                if (!rtcore_replay_memory_address_gen_latency_ready(
                        &request, service_cycle)) {
                    selected_bank_ids.insert(selected_bank_id);
                    continue;
                }
                progressed = true;
            }
        }

        const unsigned new_transaction_count =
            rtcore_replay_memory_unit_new_transaction_count_for_current_event(
                request, service_cycle);
        if (*issue_budget == 0 || *alloc_budget < new_transaction_count) {
            break;
        }
        if (!rtcore_replay_memory_outstanding_capacity_can_issue(
                owner_hw_sid, new_transaction_count)) {
            break;
        }
        const rtcore_replay_lane_request issued_request_snapshot = request;
        if (!rtcore_issue_ready_memory_replay_request(thread_uid, service_cycle)) {
            rtcore_route_admitted_replay_request(thread_uid, service_cycle);
            break;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, true, false);
        }
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_ready_issue_count++;
        rtcore_replay_memory_unit_note_issued_transactions_for_current_event(
            issued_request_snapshot, service_cycle);
        selected_bank_ids.insert(selected_bank_id);
        (*issue_budget)--;
        (*alloc_budget) -= new_transaction_count;
        progressed = true;
    }
    return progressed;
}

static bool rtcore_service_ready_memory_replay_requests_for_owner(
    unsigned owner_hw_sid, unsigned issue_budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL,
    unsigned long long service_cycle = 0)
{
    unsigned alloc_budget = rtcore_replay_memory_outstanding_alloc_budget_config();
    unsigned address_gen_budget =
        rtcore_replay_memory_address_gen_budget_config();
    return rtcore_service_ready_memory_replay_requests_for_owner_with_budget(
        owner_hw_sid, &issue_budget, &address_gen_budget, &alloc_budget,
        last_identity,
        service_cycle);
}

static bool rtcore_replay_request_state_memory_wake_service_ready(
    const rtcore_replay_lane_request &request,
    unsigned long long service_cycle)
{
    if (!request.valid || request.state != RTCORE_REPLAY_ISSUED_MEMORY ||
        !rtcore_replay_request_state_has_memory_outstanding_work(request)) {
        return false;
    }
    if (request.v02_lsu_response_wait_gate_pending) {
        return request.v02_lsu_response_wait_completed_chunk_count >=
               request.v02_lsu_response_wait_chunk_count;
    }
    if (request.memory_contention_gate_pending) {
        return service_cycle >= request.memory_contention_ready_cycle;
    }
    if (request.memory_wake_latency_gate_pending) {
        return service_cycle >= request.memory_wake_ready_cycle;
    }
    return false;
}

static bool rtcore_select_banked_memory_wake_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid,
    unsigned long long service_cycle)
{
    const unsigned bank_count =
        rtcore_replay_v03_hw_request_state_bank_count_config();
    if (bank_count == 0) {
        return false;
    }

    std::vector<rtcore_replay_v03_hw_banked_ready_candidate> candidates(
        bank_count);
    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (request.owner_hw_sid != owner_hw_sid ||
            !rtcore_replay_request_state_memory_wake_service_ready(
                request, service_cycle)) {
            continue;
        }
        const unsigned bank_id = request.request_state_bank_id % bank_count;
        rtcore_replay_v03_hw_banked_ready_candidate &candidate =
            candidates[bank_id];
        if (!candidate.valid || request.ready_order < candidate.ready_order) {
            candidate.valid = true;
            candidate.thread_uid = request.thread_uid;
            candidate.ready_order = request.ready_order;
        }
    }

    const unsigned long long cursor_key =
        rtcore_replay_v03_hw_memory_wake_bank_cursor_key(owner_hw_sid);
    unsigned cursor =
        g_rtcore_replay_v03_hw_ready_bank_rr_cursor_by_owner_unit[cursor_key] %
        bank_count;
    for (unsigned offset = 0; offset < bank_count; ++offset) {
        const unsigned bank_id = (cursor + offset) % bank_count;
        if (!candidates[bank_id].valid) {
            continue;
        }
        if (thread_uid) {
            *thread_uid = candidates[bank_id].thread_uid;
        }
        g_rtcore_replay_v03_hw_ready_bank_rr_cursor_by_owner_unit[cursor_key] =
            (bank_id + 1) % bank_count;
        return true;
    }
    return false;
}

static bool rtcore_select_replay_request_state_memory_wake_request_for_owner(
    unsigned owner_hw_sid, unsigned *thread_uid,
    unsigned long long service_cycle)
{
    return rtcore_select_banked_memory_wake_request_for_owner(
        owner_hw_sid, thread_uid, service_cycle);
}

static bool rtcore_service_memory_response_wake_requests_for_owner(
    unsigned owner_hw_sid, unsigned wake_budget,
    rtcore_replay_service_cycle_identity_snapshot *last_identity = NULL,
    unsigned long long service_cycle = 0)
{
    bool progressed = false;
    while (wake_budget > 0) {
        unsigned thread_uid = 0;
        if (!rtcore_select_replay_request_state_memory_wake_request_for_owner(
                owner_hw_sid, &thread_uid, service_cycle)) {
            break;
        }
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_wake_attempt_count++;
        if (!rtcore_wake_memory_response_replay_request(thread_uid,
                                                        service_cycle)) {
            rtcore_route_admitted_replay_request(thread_uid, service_cycle);
            break;
        }
        if (last_identity) {
            *last_identity = rtcore_make_replay_service_progress_identity(
                thread_uid, true, false);
        }
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_wake_progress_count++;
        wake_budget--;
        progressed = true;
    }
    return progressed;
}

static bool rtcore_service_memory_response_wake_requests_for_owner(
    unsigned owner_hw_sid)
{
    return rtcore_service_memory_response_wake_requests_for_owner(
        owner_hw_sid, rtcore_replay_memory_wake_budget_config());
}

static bool rtcore_service_request_state_unit_wake_requests_for_owner(
    unsigned owner_hw_sid)
{
    return rtcore_service_request_state_unit_wake_requests_for_owner(
        owner_hw_sid, rtcore_replay_unit_wake_budget_config());
}

static unsigned rtcore_continuation_count_lanes(unsigned mask)
{
    unsigned count = 0;
    while (mask) {
        count += mask & 1u;
        mask >>= 1;
    }
    return count;
}

static rtcore_replay_warp_completion_entry_key
rtcore_make_continuation_warp_key(
    const rtcore_continuation_return_packet &packet)
{
    rtcore_replay_warp_completion_entry_key key = {};
    key.owner_hw_sid = packet.owner_hw_sid;
    key.warp_uid = packet.warp_uid;
    key.warp_id = packet.warp_id;
    key.active_mask = packet.active_mask;
    return key;
}

static rtcore_replay_warp_completion_entry_key
rtcore_make_continuation_warp_key(
    const rtcore_resident_warp_continuation_state &state)
{
    rtcore_replay_warp_completion_entry_key key = {};
    key.owner_hw_sid = state.owner_hw_sid;
    key.warp_uid = state.warp_uid;
    key.warp_id = state.warp_id;
    key.active_mask = state.active_mask;
    return key;
}

static bool rtcore_shader_continuation_bridge_scoreboard_packet_ready(
    const rtcore_continuation_return_packet &packet)
{
    const rtcore_replay_warp_completion_entry_key key =
        rtcore_make_continuation_warp_key(packet);
    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_replay_warp_completion_entry_state>::iterator it =
        g_rtcore_replay_warp_completion_entries.find(key);
    if (it == g_rtcore_replay_warp_completion_entries.end()) {
        return false;
    }

    rtcore_update_replay_warp_completion_entry_state(&it->second);
    const unsigned resume_mask =
        packet.resume_required_mask & packet.active_mask;
    return it->second.valid && it->second.all_active_lanes_complete &&
           (it->second.continuation_lane_mask & resume_mask) == resume_mask &&
           (it->second.handoff_software_return_valid_mask & resume_mask) ==
               resume_mask;
}

static bool rtcore_mark_resident_warp_continuation_wakeup(
    const rtcore_continuation_return_packet &packet,
    unsigned long long service_cycle)
{
    if (!rtcore_continuation_model_enabled() || !packet.valid ||
        packet.kind != RTCORE_CONTINUATION_PACKET_CONTINUATION) {
        return false;
    }
    const rtcore_resident_rt_warp_record_key resident_key = {
        packet.owner_hw_sid, packet.warp_id};
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::const_iterator resident_it =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    const bool explicit_rt_submit_authoritative =
        rtcore_continuation_model_config() ==
            RTCORE_CONTINUATION_MODEL_ORACLE_SHADER_BOUNDARY &&
        resident_it != g_rtcore_resident_rt_warp_records.end() &&
        resident_it->second.valid;
    if (rtcore_shader_continuation_resubmit_bridge_enabled() ||
        explicit_rt_submit_authoritative) {
        printf("GPGPU-Sim RTCORE_CONTINUATION_WARP_WAKEUP "
               "wakeup_result=deferred_until_shader_return_publication "
               "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
               "explicit_rt_submit_authoritative=%u "
               "scoreboard_packet_ready=%u "
               "service_cycle=%llu\n",
               packet.owner_hw_sid, packet.warp_uid, packet.warp_id,
               packet.active_mask,
               explicit_rt_submit_authoritative ? 1u : 0u,
               rtcore_shader_continuation_bridge_scoreboard_packet_ready(packet)
                   ? 1u
                   : 0u,
               service_cycle);
        fflush(stdout);
        return false;
    }

    const rtcore_replay_warp_completion_entry_key key =
        rtcore_make_continuation_warp_key(packet);
    if (g_rtcore_continuation_warp_boundary_states.find(key) ==
        g_rtcore_continuation_warp_boundary_states.end()) {
        printf("GPGPU-Sim RTCORE_CONTINUATION_WARP_WAKEUP "
               "wakeup_result=fail_closed owner_hw_sid=%u warp_uid=%u "
               "warp_id=%u active_mask=0x%08x service_cycle=%llu\n",
               packet.owner_hw_sid, packet.warp_uid, packet.warp_id,
               packet.active_mask, service_cycle);
        fflush(stdout);
        return false;
    }

    rtcore_resident_warp_continuation_state &state =
        g_rtcore_resident_warp_continuation_states[key];
    state.valid = true;
    state.owner_hw_sid = packet.owner_hw_sid;
    state.warp_uid = packet.warp_uid;
    state.warp_id = packet.warp_id;
    state.active_mask = packet.active_mask;
    state.resume_required_mask = packet.resume_required_mask;
    state.shader_required_mask = packet.shader_required_mask;
    state.ready_cycle =
        service_cycle + rtcore_continuation_shader_latency_cycles_config();
    state.continuation_depth = packet.continuation_depth;
    g_rtcore_continuation_stats.rtcore_continuation_warp_wakeup_count++;

    printf("GPGPU-Sim RTCORE_CONTINUATION_WARP_WAKEUP "
           "wakeup_result=matched owner_hw_sid=%u warp_uid=%u warp_id=%u "
           "active_mask=0x%08x resume_required_mask=0x%08x "
           "shader_required_mask=0x%08x continuation_depth=%u "
           "ready_cycle=%llu service_cycle=%llu\n",
           state.owner_hw_sid, state.warp_uid, state.warp_id,
           state.active_mask, state.resume_required_mask,
           state.shader_required_mask, state.continuation_depth,
           state.ready_cycle, service_cycle);
    fflush(stdout);
    return true;
}

extern "C" bool rtcore_record_shader_continuation_resubmit_decision(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned packet_schema_version,
    unsigned completion_visible_mask, unsigned terminal_lane_mask,
    unsigned continuation_lane_mask, unsigned unsupported_reason_mask,
    unsigned handoff_software_return_valid_mask, unsigned next_active_mask,
    unsigned final_like_mask, unsigned missing_resume_handoff_mask,
    unsigned handoff_result_consume_mask,
    unsigned resume_handoff_publish_mask,
    bool pre_submit_guard_passed,
    unsigned long long shader_side_decision_cycle)
{
    const bool bridge_enabled =
        rtcore_shader_continuation_resubmit_bridge_enabled();
    rtcore_replay_warp_completion_entry_key key = {};
    key.owner_hw_sid = owner_hw_sid;
    key.warp_uid = warp_uid;
    key.warp_id = warp_id;
    key.active_mask = active_mask;

    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_continuation_warp_boundary_state>::iterator boundary_it =
        g_rtcore_continuation_warp_boundary_states.find(key);
    const bool boundary_found =
        boundary_it != g_rtcore_continuation_warp_boundary_states.end();
    const rtcore_resident_rt_warp_record_key resident_key = {
        owner_hw_sid, warp_id};
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::const_iterator resident_it =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    const bool explicit_rt_submit_authoritative =
        resident_it != g_rtcore_resident_rt_warp_records.end() &&
        resident_it->second.valid;
    const bool common_enqueue_guard =
        rtcore_continuation_model_enabled() && bridge_enabled &&
        !explicit_rt_submit_authoritative &&
        pre_submit_guard_passed &&
        resume_handoff_publish_mask == next_active_mask &&
        missing_resume_handoff_mask == 0 && boundary_found;
    const bool can_enqueue_reactivation =
        common_enqueue_guard && next_active_mask != 0;
    const bool can_enqueue_release_only =
        common_enqueue_guard && next_active_mask == 0 &&
        continuation_lane_mask != 0;
    const bool can_enqueue =
        can_enqueue_reactivation || can_enqueue_release_only;
    const char *bridge_action = "disabled_observe_only";
    bool actual_resubmit_state_enqueued = false;

    if (!bridge_enabled) {
        bridge_action = "disabled_observe_only";
    } else if (!rtcore_continuation_model_enabled()) {
        bridge_action = "continuation_model_off";
    } else if (explicit_rt_submit_authoritative) {
        bridge_action = "explicit_rt_submit_authoritative_observe_only";
    } else if (!pre_submit_guard_passed ||
               missing_resume_handoff_mask != 0) {
        bridge_action = "pre_submit_guard_failed";
    } else if (resume_handoff_publish_mask != next_active_mask) {
        bridge_action = "resume_handoff_publish_incomplete";
    } else if (next_active_mask == 0) {
        bridge_action = "no_resubmit_lanes";
    } else if (!boundary_found) {
        bridge_action = "fail_closed_missing_boundary_state";
    }

    if (can_enqueue) {
        rtcore_resident_warp_continuation_state &state =
            g_rtcore_resident_warp_continuation_states[key];
        state.valid = true;
        state.owner_hw_sid = owner_hw_sid;
        state.warp_uid = warp_uid;
        state.warp_id = warp_id;
        state.active_mask = active_mask;
        state.resume_required_mask = next_active_mask;
        state.shader_required_mask = continuation_lane_mask;
        state.ready_cycle = shader_side_decision_cycle +
                            rtcore_continuation_shader_latency_cycles_config();
        state.continuation_depth = boundary_it->second.continuation_depth;
        g_rtcore_continuation_stats.rtcore_continuation_warp_wakeup_count++;
        actual_resubmit_state_enqueued = true;
        bridge_action = can_enqueue_reactivation
                            ? "enqueued_reactivation"
                            : "enqueued_release_only_reconciliation";
    }

    printf("GPGPU-Sim RTCORE_SHADER_CONTINUATION_RESUBMIT_BRIDGE "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
           "packet_schema_version=%u bridge_enabled=%u "
           "pre_submit_guard_passed=%u completion_visible_mask=0x%08x "
           "terminal_lane_mask=0x%08x continuation_lane_mask=0x%08x "
           "unsupported_reason_mask=0x%08x "
           "handoff_software_return_valid_mask=0x%08x "
           "next_active_mask=0x%08x final_like_mask=0x%08x "
           "missing_resume_handoff_mask=0x%08x boundary_state_found=%u "
           "explicit_rt_submit_authoritative=%u "
           "handoff_result_consume_mask=0x%08x "
           "resume_handoff_publish_mask=0x%08x "
           "bridge_action=%s actual_resubmit_state_enqueued=%u "
           "shader_side_decision_cycle=%llu\n",
           owner_hw_sid, warp_uid, warp_id, active_mask,
           packet_schema_version, bridge_enabled ? 1u : 0u,
           pre_submit_guard_passed ? 1u : 0u, completion_visible_mask,
           terminal_lane_mask, continuation_lane_mask, unsupported_reason_mask,
           handoff_software_return_valid_mask, next_active_mask, final_like_mask,
           missing_resume_handoff_mask, boundary_found ? 1u : 0u,
           explicit_rt_submit_authoritative ? 1u : 0u,
           handoff_result_consume_mask, resume_handoff_publish_mask,
           bridge_action, actual_resubmit_state_enqueued ? 1u : 0u,
           shader_side_decision_cycle);
    fflush(stdout);
    return actual_resubmit_state_enqueued;
}

static void rtcore_log_continuation_request_state_reconcile(
    const rtcore_replay_lane_request &request,
    const rtcore_resident_warp_continuation_state &state, const char *action,
    unsigned long long service_cycle)
{
    printf("GPGPU-Sim RTCORE_CONTINUATION_REQUEST_STATE_RECONCILE "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "warp_id=%u active_mask=0x%08x next_active_mask=0x%08x "
           "shader_required_mask=0x%08x continuation_depth=%u "
           "continuation_boundary_pending=%u request_state=%u action=%s "
           "service_cycle=%llu\n",
           request.owner_hw_sid, request.thread_uid, request.lane_id,
           request.warp_uid, request.warp_id, state.active_mask,
           state.resume_required_mask, state.shader_required_mask,
           state.continuation_depth,
           request.continuation_boundary_pending ? 1u : 0u,
           static_cast<unsigned>(request.state),
           action ? action : "unknown", service_cycle);
    fflush(stdout);
}

static unsigned rtcore_reconcile_resident_warp_continuation_lanes(
    const rtcore_resident_warp_continuation_state &state,
    unsigned long long service_cycle, unsigned *released_lane_count)
{
    unsigned reusable = 0;
    unsigned released = 0;
    if (!state.valid) {
        if (released_lane_count) {
            *released_lane_count = 0;
        }
        return reusable;
    }

    for (std::map<unsigned, rtcore_replay_lane_request>::iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        rtcore_replay_lane_request &request = it->second;
        if (!rtcore_continuation_request_has_warp_metadata(request) ||
            request.owner_hw_sid != state.owner_hw_sid ||
            request.warp_uid != state.warp_uid ||
            request.warp_id != state.warp_id ||
            request.active_mask != state.active_mask) {
            continue;
        }
        const bool waiting_shader =
            request.state == RTCORE_REPLAY_WAITING_SHADER &&
            request.continuation_boundary_pending;
        const bool final_wait_retire =
            request.state == RTCORE_REPLAY_FINAL_WAIT_RETIRE;
        if (!waiting_shader && !final_wait_retire) {
            continue;
        }

        const unsigned lane_mask =
            rtcore_continuation_lane_mask(request.lane_id);
        if ((state.resume_required_mask & lane_mask) != 0) {
            if (!waiting_shader) {
                rtcore_log_continuation_request_state_reconcile(
                    request, state, "reject_final_wait_retire", service_cycle);
                continue;
            }
            reusable++;
            rtcore_log_continuation_request_state_reconcile(
                request, state, "reuse_resubmit", service_cycle);
            continue;
        }

        const bool consumed_entry =
            rtcore_replay_lane_request_state_capacity_consumes_entry(request);
        request.continuation_boundary_pending = false;
        request.state = RTCORE_REPLAY_COMPLETED;
        rtcore_record_replay_request_state_write();
        rtcore_refresh_replay_lane_request_ready_bits(&request);
        if (consumed_entry) {
            rtcore_record_replay_lane_request_state_capacity_release(
                request, service_cycle);
        }
        released++;
        rtcore_log_continuation_request_state_reconcile(
            request, state, "release_mask_shrink", service_cycle);
    }

    if (released_lane_count) {
        *released_lane_count = released;
    }
    return reusable;
}

static unsigned rtcore_reactivate_resident_warp_continuation_lanes(
    const rtcore_resident_warp_continuation_state &state,
    unsigned long long service_cycle)
{
    unsigned reactivated = 0;
    if (!state.valid) {
        return reactivated;
    }

    unsigned released_lane_count = 0;
    const unsigned reusable_lane_count =
        rtcore_reconcile_resident_warp_continuation_lanes(
            state, service_cycle, &released_lane_count);

    for (std::map<unsigned, rtcore_replay_lane_request>::iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        rtcore_replay_lane_request &request = it->second;
        if (!rtcore_continuation_request_has_warp_metadata(request) ||
            request.owner_hw_sid != state.owner_hw_sid ||
            request.warp_uid != state.warp_uid ||
            request.warp_id != state.warp_id ||
            request.active_mask != state.active_mask) {
            continue;
        }
        const unsigned lane_mask =
            rtcore_continuation_lane_mask(request.lane_id);
        if ((state.resume_required_mask & lane_mask) == 0 ||
            request.state != RTCORE_REPLAY_WAITING_SHADER ||
            !request.continuation_boundary_pending) {
            continue;
        }

        request.continuation_boundary_pending = false;
        request.continuation_depth = state.continuation_depth;
        request.continuation_segment_event_count = 0;
        if (request.next_event_index >= request.events.size()) {
            rtcore_prepare_replay_request_completion_ingress(&request);
        } else {
            request.state = rtcore_classify_replay_state(
                rtcore_unpack_compact_trace_event_type(
                    request.events[request.next_event_index]));
            rtcore_record_replay_request_state_write();
            rtcore_refresh_replay_lane_request_ready_bits(&request);
        }
        rtcore_route_admitted_replay_request(request.thread_uid,
                                             service_cycle);
        reactivated++;
    }

    const rtcore_replay_warp_completion_entry_key key =
        rtcore_make_continuation_warp_key(state);
    printf("GPGPU-Sim RTCORE_CONTINUATION_REQUEST_STATE_RECONCILE_SUMMARY "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x "
           "next_active_mask=0x%08x reusable_lane_count=%u "
           "reactivated_lane_count=%u released_lane_count=%u "
           "continuation_depth=%u service_cycle=%llu\n",
           state.owner_hw_sid, state.warp_uid, state.warp_id,
           state.active_mask, state.resume_required_mask,
           reusable_lane_count, reactivated, released_lane_count,
           state.continuation_depth, service_cycle);
    fflush(stdout);
    g_rtcore_continuation_warp_boundary_states.erase(key);
    return reactivated;
}

static rtcore_resident_rt_warp_record_key
rtcore_make_resident_rt_warp_record_key(unsigned owner_hw_sid,
                                        unsigned warp_id)
{
    rtcore_resident_rt_warp_record_key key = {};
    key.owner_hw_sid = owner_hw_sid;
    key.warp_id = warp_id;
    return key;
}

static unsigned rtcore_resident_rt_warp_record_occupancy()
{
    unsigned occupancy = 0;
    for (std::map<rtcore_resident_rt_warp_record_key,
                  rtcore_resident_rt_warp_record>::const_iterator it =
             g_rtcore_resident_rt_warp_records.begin();
         it != g_rtcore_resident_rt_warp_records.end(); ++it) {
        if (it->second.valid) {
            occupancy++;
        }
    }
    return occupancy;
}

static unsigned rtcore_compute_resident_rt_warp_admitted_lane_mask(
    const rtcore_resident_rt_warp_record &record)
{
    if (!record.valid) {
        return 0;
    }

    unsigned admitted_lane_mask = 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        const rtcore_resident_rt_warp_lane_identity &identity =
            record.lane_identity[lane];
        if ((record.active_mask & lane_mask) == 0 || !identity.valid) {
            continue;
        }
        std::map<unsigned, rtcore_replay_lane_request>::const_iterator request =
            g_rtcore_replay_lane_requests.find(identity.thread_uid);
        if (request == g_rtcore_replay_lane_requests.end()) {
            continue;
        }
        const rtcore_replay_lane_request &lane_request = request->second;
        if (lane_request.valid &&
            !lane_request.v04_private_frontier_init_pending &&
            lane_request.owner_hw_sid == record.owner_hw_sid &&
            lane_request.warp_uid == record.current_warp_uid &&
            lane_request.warp_id == record.warp_id &&
            lane_request.active_mask == record.active_mask &&
            lane_request.lane_id == lane) {
            admitted_lane_mask |= lane_mask;
        }
    }
    return admitted_lane_mask;
}

static void rtcore_refresh_resident_rt_warp_admitted_lane_mask(
    rtcore_resident_rt_warp_record *record)
{
    if (!record) {
        return;
    }
    record->admitted_lane_mask =
        rtcore_compute_resident_rt_warp_admitted_lane_mask(*record);
}

static bool rtcore_commit_v04_whole_mask_request_owner_binding(
    rtcore_resident_rt_warp_record *record, const char **failure_reason,
    bool service_after_commit = true)
{
    namespace request_owner = rtcore::v04::request_owner;
    namespace private_shared = rtcore::v04::private_shared;
    const char *failure = "accepted";
    if (failure_reason != NULL) *failure_reason = failure;
    if (!rtcore_v04_request_owner_binding_enabled()) return true;
    if (record == NULL || !record->valid || record->active_mask == 0 ||
        (record->bound_lane_mask & record->active_mask) !=
            record->active_mask) {
        failure = "REQUEST_OWNER_RESIDENT_MASK_INCOMPLETE";
        if (failure_reason != NULL) *failure_reason = failure;
        return false;
    }
    if (record->v04_request_owner_binding_valid) {
        const unsigned expected_visible_mask =
            record->v04_private_frontier_live_init_valid &&
                    !record->v04_private_frontier_init_committed
                ? 0
                : record->active_mask;
        const bool matches =
            record->v04_request_owner_active_mask == record->active_mask &&
            record->admitted_lane_mask == expected_visible_mask;
        if (failure_reason != NULL) {
            *failure_reason = matches ? "already_committed"
                                      : "REQUEST_OWNER_DUPLICATE_MISMATCH";
        }
        return matches;
    }

    request_owner::warp_identity_v0 identity = {};
    identity.owner_hw_sid = record->owner_hw_sid;
    identity.warp_uid = record->current_warp_uid;
    identity.warp_id = record->warp_id;
    identity.active_mask = record->active_mask;

    request_owner::allocator_state_v0 &owner_allocator =
        rtcore_v04_request_owner_allocator_for(record->owner_hw_sid);
    request_owner::new_warp_plan_v0 owner_plan = {};
    request_owner::status_kind owner_status =
        request_owner::prepare_new_warp(owner_allocator, identity, &owner_plan);
    if (owner_status != request_owner::kStatusOk) {
        failure = request_owner::status_name(owner_status);
    }

    std::array<rtcore_replay_lane_request, 32> requests = {};
    rtcore::v04::private_frontier::owner_binding_v0 private_owners[32] = {};
    private_shared::new_warp_plan_v0 private_plan = {};
    private_shared::backing_state_v0 *private_backing = NULL;
    unsigned prepared_mask = 0;
    unsigned prepared_lane_count = 0;
    unsigned valid_packed_key_count = 0;
    std::set<unsigned> unique_request_slots;
    std::set<unsigned> unique_private_slots;
    const unsigned ready_order_base = g_rtcore_next_replay_ready_order;
    for (unsigned lane = 0;
         lane < 32 && strcmp(failure, "accepted") == 0; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((record->active_mask & lane_mask) == 0) continue;
        const rtcore_resident_rt_warp_lane_identity &lane_identity =
            record->lane_identity[lane];
        rtcore_compact_trace_export_record export_record = {};
        if (!lane_identity.valid ||
            !rtcore_get_compact_trace_export(lane_identity.thread_uid,
                                             &export_record) ||
            !export_record.valid || !export_record.has_warp_metadata ||
            export_record.thread_uid != lane_identity.thread_uid ||
            export_record.owner_hw_sid != record->owner_hw_sid ||
            export_record.warp_uid != record->current_warp_uid ||
            export_record.warp_id != record->warp_id ||
            export_record.active_mask != record->active_mask ||
            export_record.static_inst_uid !=
                record->current_static_inst_uid ||
            export_record.lane_id != lane ||
            export_record.v04_live_handoff_memory == NULL) {
            failure = "REQUEST_OWNER_COMPACT_TRACE_IDENTITY_MISMATCH";
            break;
        }
        std::map<unsigned, rtcore_replay_lane_request>::const_iterator existing =
            g_rtcore_replay_lane_requests.find(lane_identity.thread_uid);
        if (existing != g_rtcore_replay_lane_requests.end() &&
            rtcore_replay_lane_request_state_capacity_consumes_entry(
                existing->second)) {
            failure = "REQUEST_OWNER_LIVE_REPLAY_STATE_COLLISION";
            break;
        }

        requests[lane] = rtcore_build_replay_lane_request(export_record);
        if (ready_order_base >
            std::numeric_limits<unsigned>::max() - prepared_lane_count) {
            failure = "REQUEST_OWNER_READY_ORDER_OVERFLOW";
            break;
        }
        requests[lane].ready_order =
            ready_order_base + prepared_lane_count;
        const request_owner::lane_binding_v0 &binding =
            owner_plan.lane_bindings[lane];
        request_owner::internal_request_key_fields_v0 key_fields = {};
        if (request_owner::unpack_internal_request_key(
                binding.packed_request_key, &key_fields) !=
                request_owner::kStatusOk ||
            key_fields.resident_warp_slot != binding.resident_warp_slot ||
            key_fields.request_control_slot !=
                binding.request_control_slot ||
            key_fields.lane_id != lane ||
            key_fields.request_generation != binding.request_generation) {
            failure = "REQUEST_OWNER_PACKED_KEY_MISMATCH";
            break;
        }
        const rtcore::v04::private_frontier::owner_binding_v0 private_owner =
            request_owner::make_private_frontier_owner(binding);
        if (private_owner.owner_hw_sid != binding.owner_hw_sid ||
            private_owner.resident_warp_id != binding.resident_warp_slot ||
            private_owner.request_identity != binding.packed_request_key ||
            private_owner.generation != binding.request_generation ||
            private_owner.private_slot_id != binding.private_slot_id ||
            private_owner.lane_id != binding.lane_id) {
            failure = "REQUEST_OWNER_PRIVATE_TUPLE_MISMATCH";
            break;
        }
        requests[lane].v04_request_owner_binding_valid = true;
        requests[lane].v04_request_owner_binding = binding;
        requests[lane].v04_private_frontier_owner = private_owner;
        private_owners[lane] = private_owner;
        unique_request_slots.insert(binding.request_control_slot);
        unique_private_slots.insert(binding.private_slot_id);
        ++valid_packed_key_count;
        prepared_mask |= lane_mask;
        ++prepared_lane_count;
    }
    if (strcmp(failure, "accepted") == 0 && prepared_lane_count != 0 &&
        ready_order_base >
            std::numeric_limits<unsigned>::max() - prepared_lane_count) {
        failure = "REQUEST_OWNER_READY_ORDER_OVERFLOW";
    }
    if (strcmp(failure, "accepted") == 0 &&
        (unique_request_slots.size() != prepared_lane_count ||
         unique_private_slots.size() != prepared_lane_count ||
         valid_packed_key_count != prepared_lane_count)) {
        failure = "REQUEST_OWNER_LIVE_SLOT_ALIAS";
    }
    if (strcmp(failure, "accepted") == 0 &&
        rtcore_v04_private_frontier_live_init_enabled()) {
        private_backing =
            &rtcore_v04_private_shared_backing_for(record->owner_hw_sid);
        const private_shared::status_kind private_status =
            private_shared::prepare_new_warp(
                *private_backing, record->current_warp_uid,
                record->warp_id, record->active_mask, private_owners,
                &private_plan);
        if (private_status != private_shared::kStatusOk) {
            failure = private_shared::status_name(private_status);
        }
    }

    if (strcmp(failure, "accepted") != 0 ||
        prepared_mask != record->active_mask) {
        printf("GPGPU-Sim RTCORE_V04_REQUEST_OWNER_BINDING "
               "owner_hw_sid=%u warp_uid=%u warp_id=%u "
               "active_mask=0x%08x prepared_mask=0x%08x "
               "active_lanes=%u unique_request_slots=%zu "
               "unique_private_slots=%zu valid_packed_keys=%u "
               "whole_mask_committed=0 scheduler_visible_mask=0x00000000 "
               "allocator_scope=per_sm allocator_epoch=%llu result=%s\n",
               record->owner_hw_sid, record->current_warp_uid,
               record->warp_id, record->active_mask, prepared_mask,
               prepared_lane_count, unique_request_slots.size(),
               unique_private_slots.size(), valid_packed_key_count,
               static_cast<unsigned long long>(
                   owner_allocator.mutation_epoch),
               failure);
        fflush(stdout);
        if (failure_reason != NULL) *failure_reason = failure;
        return false;
    }

    owner_status = request_owner::commit_new_warp(
        &owner_allocator, owner_plan);
    if (owner_status != request_owner::kStatusOk) {
        failure = request_owner::status_name(owner_status);
        if (failure_reason != NULL) *failure_reason = failure;
        return false;
    }
    if (rtcore_v04_private_frontier_live_init_enabled()) {
        const private_shared::status_kind private_status =
            private_shared::commit_new_warp(private_backing, private_plan);
        if (private_status != private_shared::kStatusOk) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                    "owner_hw_sid=%u warp_uid=%u warp_id=%u fault=%s\n",
                    record->owner_hw_sid, record->current_warp_uid,
                    record->warp_id,
                    private_shared::status_name(private_status));
            fflush(stderr);
            abort();
        }
    }

    record->v04_request_owner_binding_valid = true;
    record->v04_resident_warp_slot = owner_plan.resident_warp_slot;
    record->v04_request_owner_active_mask = record->active_mask;
    record->v04_private_frontier_live_init_valid =
        rtcore_v04_private_frontier_live_init_enabled();
    record->v04_private_frontier_init_committed =
        !rtcore_v04_private_frontier_live_init_enabled();
    record->v04_private_frontier_init_active_mask =
        rtcore_v04_private_frontier_live_init_enabled()
            ? record->active_mask
            : 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((record->active_mask & lane_mask) == 0) continue;
        rtcore_resident_rt_warp_lane_identity &lane_identity =
            record->lane_identity[lane];
        lane_identity.v04_request_owner_binding_valid = true;
        lane_identity.v04_request_owner_binding =
            owner_plan.lane_bindings[lane];
        if (!request_owner::validate_live_binding(
                owner_allocator, owner_plan.lane_bindings[lane],
                record->current_warp_uid, record->active_mask)) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_REQUEST_OWNER_COMMIT_INVARIANT "
                    "owner_hw_sid=%u warp_uid=%u warp_id=%u lane_id=%u "
                    "fault=committed_binding_not_live\n",
                    record->owner_hw_sid, record->current_warp_uid,
                    record->warp_id, lane);
            fflush(stderr);
            abort();
        }
    }

    g_rtcore_next_replay_ready_order =
        ready_order_base + prepared_lane_count;

    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((record->active_mask & lane_mask) == 0) continue;
        requests[lane].v04_private_frontier_init_pending =
            rtcore_v04_private_frontier_live_init_enabled();
        rtcore_record_replay_resource_route_stats(requests[lane]);
        rtcore_commit_replay_lane_request_state_admission(requests[lane]);
    }
    rtcore_refresh_resident_rt_warp_admitted_lane_mask(record);
    const unsigned expected_scheduler_visible_mask =
        rtcore_v04_private_frontier_live_init_enabled()
            ? 0
            : record->active_mask;
    if (record->admitted_lane_mask != expected_scheduler_visible_mask) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_REQUEST_OWNER_COMMIT_INVARIANT "
                "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                "active_mask=0x%08x admitted_mask=0x%08x "
                "fault=partial_scheduler_visibility\n",
                record->owner_hw_sid, record->current_warp_uid,
                record->warp_id, record->active_mask,
                record->admitted_lane_mask);
        fflush(stderr);
        abort();
    }

    printf("GPGPU-Sim RTCORE_V04_REQUEST_OWNER_BINDING "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u "
           "active_mask=0x%08x prepared_mask=0x%08x "
           "active_lanes=%u unique_request_slots=%zu "
           "unique_private_slots=%zu valid_packed_keys=%u "
           "resident_warp_slot=%u request_slot_capacity=%u "
           "private_slot_one_to_one=1 whole_mask_committed=1 "
           "scheduler_visible_mask=0x%08x allocator_scope=per_sm "
           "private_live_init=%u resident_shared_charge_bytes=%u "
           "allocator_epoch=%llu "
           "functional_authority=0 timing_authority=0 result=accepted\n",
           record->owner_hw_sid, record->current_warp_uid, record->warp_id,
           record->active_mask, prepared_mask,
           prepared_lane_count, unique_request_slots.size(),
           unique_private_slots.size(), valid_packed_key_count,
           owner_plan.resident_warp_slot,
           request_owner::kRequestControlCapacity,
           record->admitted_lane_mask,
           rtcore_v04_private_frontier_live_init_enabled() ? 1u : 0u,
           rtcore_v04_private_frontier_live_init_enabled()
               ? private_plan.charged_bytes
               : 0u,
           static_cast<unsigned long long>(
               owner_allocator.mutation_epoch));
    fflush(stdout);
    if (service_after_commit) {
        rtcore_try_service_replay_after_admission(record->owner_hw_sid);
    }
    if (failure_reason != NULL) *failure_reason = "accepted";
    return true;
}

static bool rtcore_v04_request_owner_failure_is_resource_backpressure(
    const char *failure)
{
    if (failure == NULL) return false;
    return strcmp(
               failure,
               rtcore::v04::request_owner::status_name(
                   rtcore::v04::request_owner::
                       kStatusResidentCapacityExceeded)) == 0 ||
           strcmp(
               failure,
               rtcore::v04::request_owner::status_name(
                   rtcore::v04::request_owner::
                       kStatusRequestCapacityExceeded)) == 0 ||
           strcmp(
               failure,
               rtcore::v04::private_shared::status_name(
                   rtcore::v04::private_shared::
                       kStatusCapacityExceeded)) == 0;
}

static bool rtcore_retry_v04_whole_mask_request_owner_admission(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    if (!rtcore_v04_request_owner_binding_enabled()) return false;
    std::vector<rtcore_resident_rt_warp_record *> pending;
    for (std::map<rtcore_resident_rt_warp_record_key,
                  rtcore_resident_rt_warp_record>::iterator it =
             g_rtcore_resident_rt_warp_records.begin();
         it != g_rtcore_resident_rt_warp_records.end(); ++it) {
        rtcore_resident_rt_warp_record &record = it->second;
        if (!record.valid || record.owner_hw_sid != owner_hw_sid ||
            record.v04_request_owner_binding_valid ||
            (record.bound_lane_mask & record.active_mask) !=
                record.active_mask) {
            continue;
        }
        pending.push_back(&record);
    }
    std::sort(
        pending.begin(), pending.end(),
        [](const rtcore_resident_rt_warp_record *lhs,
           const rtcore_resident_rt_warp_record *rhs) {
            return lhs->resident_generation < rhs->resident_generation;
        });
    for (std::vector<rtcore_resident_rt_warp_record *>::iterator it =
             pending.begin();
         it != pending.end(); ++it) {
        rtcore_resident_rt_warp_record &record = **it;
        const char *failure = "accepted";
        if (rtcore_commit_v04_whole_mask_request_owner_binding(
                &record, &failure, false)) {
            printf("GPGPU-Sim RTCORE_V04_REQUEST_OWNER_ADMISSION_RETRY "
                   "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                   "active_mask=0x%08x service_cycle=%llu result=accepted\n",
                   record.owner_hw_sid, record.current_warp_uid,
                   record.warp_id, record.active_mask, service_cycle);
            fflush(stdout);
            return true;
        }
        if (rtcore_v04_request_owner_failure_is_resource_backpressure(
                failure)) {
            printf("GPGPU-Sim RTCORE_V04_REQUEST_OWNER_ADMISSION_RETRY "
                   "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                   "active_mask=0x%08x service_cycle=%llu "
                   "source_owner=sm_resident_warp "
                   "result=resource_backpressure reason=%s\n",
                   record.owner_hw_sid, record.current_warp_uid,
                   record.warp_id, record.active_mask, service_cycle,
                   failure);
            fflush(stdout);
            continue;
        }
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_REQUEST_OWNER_ADMISSION_INVARIANT "
                "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                "active_mask=0x%08x service_cycle=%llu fault=%s\n",
                record.owner_hw_sid, record.current_warp_uid,
                record.warp_id, record.active_mask, service_cycle,
                failure);
        fflush(stderr);
        abort();
    }
    return false;
}

extern "C" bool rtcore_query_resident_rt_warp_record(
    unsigned owner_hw_sid, unsigned warp_id, unsigned *current_warp_uid,
    unsigned *active_mask, unsigned *resident_generation,
    unsigned *resident_occupancy)
{
    if (resident_occupancy) {
        *resident_occupancy = rtcore_resident_rt_warp_record_occupancy();
    }
    if (!rtcore_continuation_model_enabled()) {
        return false;
    }

    const rtcore_resident_rt_warp_record_key key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::const_iterator it =
        g_rtcore_resident_rt_warp_records.find(key);
    if (it == g_rtcore_resident_rt_warp_records.end() || !it->second.valid) {
        return false;
    }
    if (current_warp_uid) {
        *current_warp_uid = it->second.current_warp_uid;
    }
    if (active_mask) {
        *active_mask = it->second.active_mask;
    }
    if (resident_generation) {
        *resident_generation = it->second.resident_generation;
    }
    return true;
}

extern "C" bool rtcore_bind_resident_rt_warp_lane_identity(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask, unsigned static_inst_uid, unsigned lane_id,
    unsigned thread_uid, unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned token_id,
    unsigned token_allocator_generation, unsigned window_generation)
{
    if (!rtcore_continuation_model_enabled()) {
        return true;
    }
    static const unsigned kSimulatorSmWarpSlotCapacity = 64;
    if (lane_id >= 32 || warp_id >= kSimulatorSmWarpSlotCapacity ||
        active_mask == 0 ||
        (active_mask & (1u << lane_id)) == 0 || context_ptr == 0 ||
        handoff_window_base == 0 || token_id == 0 ||
        token_allocator_generation == 0 || window_generation == 0) {
        return false;
    }

    const rtcore_resident_rt_warp_record_key key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    rtcore_resident_rt_warp_record &record =
        g_rtcore_resident_rt_warp_records[key];
    const bool allocated = !record.valid;
    if (allocated) {
        record.valid = true;
        record.owner_hw_sid = owner_hw_sid;
        record.warp_id = warp_id;
        record.current_warp_uid = warp_uid;
        record.current_static_inst_uid = static_inst_uid;
        record.active_mask = active_mask;
        record.resident_generation =
            g_rtcore_next_resident_rt_warp_generation++;
    }

    const bool metadata_matches =
        record.owner_hw_sid == owner_hw_sid && record.warp_id == warp_id &&
        record.current_warp_uid == warp_uid &&
        record.current_static_inst_uid == static_inst_uid &&
        record.active_mask == active_mask;
    rtcore_resident_rt_warp_lane_identity &identity =
        record.lane_identity[lane_id];
    const bool identity_matches =
        !identity.valid ||
        (identity.thread_uid == thread_uid && identity.context_ptr == context_ptr &&
         identity.handoff_window_base == handoff_window_base &&
         identity.token_id == token_id &&
         identity.token_allocator_generation == token_allocator_generation &&
         identity.window_generation == window_generation);
    if (!metadata_matches || !identity_matches) {
        printf("GPGPU-Sim RTCORE_RESIDENT_RT_WARP_BIND "
               "owner_hw_sid=%u warp_uid=%u warp_id=%u lane_id=%u "
               "active_mask=0x%08x metadata_match=%u identity_match=%u "
               "bind_result=fail_closed\n",
               owner_hw_sid, warp_uid, warp_id, lane_id, active_mask,
               metadata_matches ? 1u : 0u, identity_matches ? 1u : 0u);
        fflush(stdout);
        return false;
    }

    identity.valid = true;
    identity.thread_uid = thread_uid;
    identity.context_ptr = context_ptr;
    identity.handoff_window_base = handoff_window_base;
    identity.token_id = token_id;
    identity.token_allocator_generation = token_allocator_generation;
    identity.window_generation = window_generation;
    record.bound_lane_mask |= 1u << lane_id;
    const char *request_owner_failure = "not_required";
    if (rtcore_v04_request_owner_binding_enabled() &&
        (record.bound_lane_mask & record.active_mask) ==
            record.active_mask &&
        !rtcore_commit_v04_whole_mask_request_owner_binding(
            &record, &request_owner_failure)) {
        if (rtcore_v04_request_owner_failure_is_resource_backpressure(
                request_owner_failure)) {
            rtcore_refresh_resident_rt_warp_admitted_lane_mask(&record);
            printf("GPGPU-Sim RTCORE_RESIDENT_RT_WARP_BIND "
                   "owner_hw_sid=%u warp_uid=%u warp_id=%u lane_id=%u "
                   "active_mask=0x%08x bound_lane_mask=0x%08x "
                   "admitted_lane_mask=0x%08x "
                   "request_owner_binding=resource_backpressure "
                   "source_owner=sm_resident_warp "
                   "reason=%s bind_result=accepted_pending_retry\n",
                   owner_hw_sid, warp_uid, warp_id, lane_id, active_mask,
                   record.bound_lane_mask, record.admitted_lane_mask,
                   request_owner_failure);
            fflush(stdout);
            return true;
        }
        printf("GPGPU-Sim RTCORE_RESIDENT_RT_WARP_BIND "
               "owner_hw_sid=%u warp_uid=%u warp_id=%u lane_id=%u "
               "active_mask=0x%08x bound_lane_mask=0x%08x "
               "request_owner_binding=fail_closed reason=%s\n",
               owner_hw_sid, warp_uid, warp_id, lane_id, active_mask,
               record.bound_lane_mask, request_owner_failure);
        fflush(stdout);
        return false;
    }
    rtcore_refresh_resident_rt_warp_admitted_lane_mask(&record);

    printf("GPGPU-Sim RTCORE_RESIDENT_RT_WARP_BIND "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u lane_id=%u "
           "active_mask=0x%08x bound_lane_mask=0x%08x "
           "admitted_lane_mask=0x%08x resident_generation=%u "
           "resident_record_allocated=%u resident_occupancy=%u "
           "request_owner_binding=%u physical_resident_slot=%u "
           "bind_result=accepted\n",
           owner_hw_sid, warp_uid, warp_id, lane_id, active_mask,
           record.bound_lane_mask, record.admitted_lane_mask,
           record.resident_generation, allocated ? 1u : 0u,
           rtcore_resident_rt_warp_record_occupancy(),
           record.v04_request_owner_binding_valid ? 1u : 0u,
           record.v04_resident_warp_slot);
    fflush(stdout);
    return true;
}

static bool rtcore_resident_lane_identity_matches(
    const rtcore_resident_rt_warp_lane_identity &identity,
    unsigned thread_uid, unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned token_id,
    unsigned token_allocator_generation, unsigned window_generation)
{
    return identity.valid && identity.thread_uid == thread_uid &&
           identity.context_ptr == context_ptr &&
           identity.handoff_window_base == handoff_window_base &&
           identity.token_id == token_id &&
           identity.token_allocator_generation == token_allocator_generation &&
           identity.window_generation == window_generation;
}

static const char *rtcore_resubmit_admission_test_failpoint_mode()
{
    const char *value =
        getenv("VULKAN_SIM_RTCORE_TEST_RESUBMIT_ADMISSION_FAILPOINT");
    if (!value || !value[0] || strcmp(value, "0") == 0) {
        return NULL;
    }
    if (strcmp(value, "missing") == 0) {
        return "missing";
    }
    if (strcmp(value, "stale") == 0) {
        return "stale";
    }
    if (strcmp(value, "owner") == 0) {
        return "owner";
    }
    if (strcmp(value, "final_wait") == 0) {
        return "final_wait";
    }
    if (strcmp(value, "late") == 0) {
        return "late";
    }
    return NULL;
}

static bool rtcore_restore_v04_live_trace_input_after_legacy_shader_return(
    rtcore_replay_lane_request *request, unsigned long long refresh_cycle,
    const char **failure_reason)
{
    if (!rtcore_v04_live_handoff_publication_enabled()) {
        return true;
    }

    const char *failure = "accepted";
    if (request == NULL || !request->valid ||
        !request->v04_shadow_boundary_enabled ||
        !request->v04_shadow_trace_input_valid ||
        request->v04_live_handoff_memory == NULL ||
        request->handoff_window_base == 0 || request->lane_id >= 32) {
        failure = "V04_LIVE_TRACE_INPUT_RESTORE_CONTEXT_INVALID";
    }

    unsigned long long lane_address = 0;
    if (strcmp(failure, "accepted") == 0) {
        const unsigned long long lane_offset =
            static_cast<unsigned long long>(request->lane_id) *
            rtcore::abi_v04::kLaneSlotBytes;
        lane_address = request->handoff_window_base + lane_offset;
        if (lane_address < request->handoff_window_base) {
            failure = "V04_LIVE_TRACE_INPUT_RESTORE_ADDRESS_OVERFLOW";
        }
    }

    uint32_t changed_word_mask = 0;
    uint32_t restored_word_mask = 0;
    uint32_t post_restore_mismatch_mask = 0;
    const uint32_t immutable_mask =
        rtcore::abi_v04::shadow::trace_input_owned_word_mask();
    if (strcmp(failure, "accepted") == 0) {
        std::array<uint32_t, rtcore::abi_v04::kWordCount> preimage = {};
        request->v04_live_handoff_memory->read_simulator_backing(
            lane_address, sizeof(preimage), preimage.data());
        for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
            const uint32_t word_bit = uint32_t{1} << word;
            if ((immutable_mask & word_bit) == 0) {
                continue;
            }
            const uint32_t expected =
                request->v04_shadow_trace_input_words[word];
            if (preimage[word] != expected) {
                changed_word_mask |= word_bit;
            }
            request->v04_live_handoff_memory->write_simulator_backing(
                lane_address + word * sizeof(uint32_t), sizeof(expected),
                &expected);
            restored_word_mask |= word_bit;
        }

        std::array<uint32_t, rtcore::abi_v04::kWordCount> postimage = {};
        request->v04_live_handoff_memory->read_simulator_backing(
            lane_address, sizeof(postimage), postimage.data());
        for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
            const uint32_t word_bit = uint32_t{1} << word;
            if ((immutable_mask & word_bit) != 0 &&
                postimage[word] !=
                    request->v04_shadow_trace_input_words[word]) {
                post_restore_mismatch_mask |= word_bit;
            }
        }
        if (post_restore_mismatch_mask != 0) {
            failure = "V04_LIVE_TRACE_INPUT_RESTORE_VALIDATION_FAILED";
        }
    }

    printf("GPGPU-Sim RTCORE_V04_LIVE_TRACE_INPUT_COMPATIBILITY_RESTORE "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "lane_address=0x%llx immutable_word_mask=0x%08x "
           "changed_word_mask=0x%08x restored_word_mask=0x%08x "
           "post_restore_mismatch_mask=0x%08x "
           "compatibility_restore=1 architectural_write=0 "
           "rt_memory_unit_timing_charge=0 refresh_cycle=%llu result=%s\n",
           request != NULL ? request->owner_hw_sid : 0u,
           request != NULL ? request->thread_uid : 0u,
           request != NULL ? request->lane_id : 0u,
           request != NULL ? request->warp_uid : 0u, lane_address,
           immutable_mask, changed_word_mask, restored_word_mask,
           post_restore_mismatch_mask, refresh_cycle, failure);
    fflush(stdout);
    if (failure_reason != NULL && strcmp(failure, "accepted") != 0) {
        *failure_reason = failure;
    }
    return strcmp(failure, "accepted") == 0;
}

struct rtcore_v04_live_trace_input_restore_plan {
    rtcore_v04_live_trace_input_restore_plan()
        : valid(false), memory(NULL), owner_hw_sid(0), thread_uid(0),
          lane_id(0), warp_uid(0), handoff_window_base(0), lane_address(0),
          immutable_word_mask(0), changed_word_mask(0)
    {
        expected_words.fill(0);
    }

    bool valid;
    memory_space *memory;
    unsigned owner_hw_sid;
    unsigned thread_uid;
    unsigned lane_id;
    unsigned warp_uid;
    unsigned long long handoff_window_base;
    unsigned long long lane_address;
    uint32_t immutable_word_mask;
    uint32_t changed_word_mask;
    std::array<uint32_t, rtcore::abi_v04::kWordCount> expected_words;
};

static bool rtcore_prepare_v04_live_trace_input_compatibility_restore(
    const rtcore_replay_lane_request *request,
    rtcore_v04_live_trace_input_restore_plan *plan,
    const char **failure_reason)
{
    const char *failure = "accepted";
    if (plan == NULL) {
        failure = "V04_LIVE_TRACE_INPUT_RESTORE_PLAN_MISSING";
    } else {
        *plan = rtcore_v04_live_trace_input_restore_plan();
    }
    if (strcmp(failure, "accepted") == 0 &&
        (!rtcore_v04_live_handoff_publication_enabled() || request == NULL ||
         !request->valid || !request->v04_shadow_boundary_enabled ||
         !request->v04_shadow_trace_input_valid ||
         request->v04_live_handoff_memory == NULL ||
         request->handoff_window_base == 0 || request->lane_id >= 32)) {
        failure = "V04_LIVE_TRACE_INPUT_RESTORE_CONTEXT_INVALID";
    }

    unsigned long long lane_address = 0;
    if (strcmp(failure, "accepted") == 0) {
        lane_address = request->handoff_window_base +
            static_cast<unsigned long long>(request->lane_id) *
                rtcore::abi_v04::kLaneSlotBytes;
        if (lane_address < request->handoff_window_base) {
            failure = "V04_LIVE_TRACE_INPUT_RESTORE_ADDRESS_OVERFLOW";
        }
    }

    const char *fault_lane_value = getenv(
        "VULKAN_SIM_RTCORE_TEST_V04_RESTORE_PREFLIGHT_FAIL_LANE");
    if (strcmp(failure, "accepted") == 0 && fault_lane_value != NULL &&
        *fault_lane_value != '\0') {
        char *end = NULL;
        const unsigned long fault_lane =
            strtoul(fault_lane_value, &end, 0);
        if (end == fault_lane_value || *end != '\0' || fault_lane >= 32) {
            failure = "V04_LIVE_TRACE_INPUT_RESTORE_FAIL_LANE_INVALID";
        } else if (fault_lane == request->lane_id) {
            failure = "TEST_V04_RESTORE_PREFLIGHT_FAILURE";
        }
    }

    uint32_t changed_word_mask = 0;
    const uint32_t immutable_mask =
        rtcore::abi_v04::shadow::trace_input_owned_word_mask();
    if (strcmp(failure, "accepted") == 0) {
        std::array<uint32_t, rtcore::abi_v04::kWordCount> preimage = {};
        request->v04_live_handoff_memory->read_simulator_backing(
            lane_address, sizeof(preimage), preimage.data());
        for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
            const uint32_t word_bit = uint32_t{1} << word;
            if ((immutable_mask & word_bit) != 0 &&
                preimage[word] != request->v04_shadow_trace_input_words[word]) {
                changed_word_mask |= word_bit;
            }
        }
        plan->valid = true;
        plan->memory = request->v04_live_handoff_memory;
        plan->owner_hw_sid = request->owner_hw_sid;
        plan->thread_uid = request->thread_uid;
        plan->lane_id = request->lane_id;
        plan->warp_uid = request->warp_uid;
        plan->handoff_window_base = request->handoff_window_base;
        plan->lane_address = lane_address;
        plan->immutable_word_mask = immutable_mask;
        plan->changed_word_mask = changed_word_mask;
        plan->expected_words = request->v04_shadow_trace_input_words;
    }

    printf("GPGPU-Sim RTCORE_V04_LIVE_TRACE_INPUT_RESTORE_PREFLIGHT "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "lane_address=0x%llx immutable_word_mask=0x%08x "
           "changed_word_mask=0x%08x state_mutation=0 result=%s\n",
           request != NULL ? request->owner_hw_sid : 0u,
           request != NULL ? request->thread_uid : 0u,
           request != NULL ? request->lane_id : 0u,
           request != NULL ? request->warp_uid : 0u, lane_address,
           immutable_mask, changed_word_mask,
           strcmp(failure, "accepted") == 0 ? "prepared" : failure);
    fflush(stdout);
    if (failure_reason != NULL) *failure_reason = failure;
    return strcmp(failure, "accepted") == 0;
}

static void rtcore_commit_prepared_v04_live_trace_input_restore(
    const rtcore_v04_live_trace_input_restore_plan &plan,
    unsigned long long refresh_cycle)
{
    assert(plan.valid && plan.memory != NULL && plan.lane_address != 0);
    uint32_t restored_word_mask = 0;
    for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
        const uint32_t word_bit = uint32_t{1} << word;
        if ((plan.immutable_word_mask & word_bit) == 0) continue;
        const uint32_t expected = plan.expected_words[word];
        plan.memory->write_simulator_backing(
            plan.lane_address + word * sizeof(uint32_t), sizeof(expected),
            &expected);
        restored_word_mask |= word_bit;
    }

    std::array<uint32_t, rtcore::abi_v04::kWordCount> postimage = {};
    plan.memory->read_simulator_backing(
        plan.lane_address, sizeof(postimage), postimage.data());
    uint32_t mismatch_mask = 0;
    for (unsigned word = 0; word < rtcore::abi_v04::kWordCount; ++word) {
        const uint32_t word_bit = uint32_t{1} << word;
        if ((plan.immutable_word_mask & word_bit) != 0 &&
            postimage[word] != plan.expected_words[word]) {
            mismatch_mask |= word_bit;
        }
    }
    if (mismatch_mask != 0) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_LIVE_TRACE_INPUT_RESTORE_INVARIANT "
                "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
                "mismatch_mask=0x%08x\n",
                plan.owner_hw_sid, plan.thread_uid, plan.lane_id,
                plan.warp_uid, mismatch_mask);
        fflush(stderr);
        abort();
    }
    printf("GPGPU-Sim RTCORE_V04_LIVE_TRACE_INPUT_COMPATIBILITY_RESTORE "
           "owner_hw_sid=%u thread_uid=%u lane_id=%u warp_uid=%u "
           "lane_address=0x%llx immutable_word_mask=0x%08x "
           "changed_word_mask=0x%08x restored_word_mask=0x%08x "
           "post_restore_mismatch_mask=0x%08x "
           "compatibility_restore=1 architectural_write=0 "
           "rt_memory_unit_timing_charge=0 refresh_cycle=%llu "
           "preflight_complete=1 commit_cannot_fail=1 result=accepted\n",
           plan.owner_hw_sid, plan.thread_uid, plan.lane_id, plan.warp_uid,
           plan.lane_address, plan.immutable_word_mask,
           plan.changed_word_mask, restored_word_mask, mismatch_mask,
           refresh_cycle);
    fflush(stdout);
}

static bool rtcore_revalidate_prepared_v04_live_trace_input_restore(
    const rtcore_v04_live_trace_input_restore_plan &plan,
    const rtcore_replay_lane_request *request,
    const char **failure_reason)
{
    const bool valid =
        plan.valid && request != NULL && request->valid &&
        plan.memory == request->v04_live_handoff_memory &&
        plan.owner_hw_sid == request->owner_hw_sid &&
        plan.thread_uid == request->thread_uid &&
        plan.lane_id == request->lane_id &&
        plan.warp_uid == request->warp_uid &&
        plan.handoff_window_base == request->handoff_window_base &&
        plan.expected_words == request->v04_shadow_trace_input_words &&
        rtcore_v04_live_publication_matches_current_submit(*request) &&
        request->v04_live_publication_committed;
    if (failure_reason != NULL) {
        *failure_reason = valid
                              ? "accepted"
                              : "V04_LIVE_TRACE_INPUT_RESTORE_GENERATION_CHANGED";
    }
    return valid;
}

static bool rtcore_v04_retained_procedural_candidate_valid(
    const rtcore_boundary_candidate_snapshot &snapshot)
{
    const rtcore::abi_v04::shadow::boundary_values &values =
        snapshot.v04_boundary_values;
    return snapshot.valid != 0 && snapshot.geometry_type == 2u &&
           snapshot.shader_counter != UINT_MAX && values.candidate_valid &&
           values.geometry_type ==
               rtcore::abi_v04::shadow::kBoundaryGeometryProcedural &&
           values.instance_metadata_reference != 0 &&
           snapshot.hit_group_index == values.instance_sbt_contribution &&
           snapshot.geometry_index == values.geometry_index &&
           snapshot.primitive_index == values.primitive_index &&
           snapshot.instance_index == values.instance_custom_index &&
           values.input_attribute_word_count == 0u &&
           values.input_attribute_location == 0u &&
           values.input_attribute_format == 0u;
}

static bool rtcore_apply_v04_functional_return_test_mutation(
    std::array<uint32_t, rtcore::abi_v04::kWordCount> *words,
    const char **mutation_name)
{
    const char *mutation = getenv(
        "VULKAN_SIM_RTCORE_TEST_V04_FUNCTIONAL_SHADER_RETURN_MUTATION");
    if (mutation == NULL || mutation[0] == '\0' ||
        strcmp(mutation, "none") == 0) {
        if (mutation_name != NULL) *mutation_name = "none";
        return true;
    }
    if (mutation_name != NULL) *mutation_name = mutation;
    if (words == NULL) return false;
    if (strcmp(mutation, "effect") == 0) {
        (*words)[rtcore::abi_v04::kCommitRetainedCandidate.word] ^= 1u;
    } else if (strcmp(mutation, "reported_t") == 0) {
        (*words)[rtcore::abi_v04::kReportedTFp32.word] ^= 1u;
    } else if (strcmp(mutation, "reserved") == 0) {
        (*words)[rtcore::abi_v04::kCommitRetainedCandidate.word] |= 1u << 3;
    } else if (strcmp(mutation, "terminate") == 0) {
        (*words)[rtcore::abi_v04::kTerminateSearch.word] |=
            rtcore::abi_v04::kTerminateSearch.mask;
    } else if (strcmp(mutation, "attributes") == 0) {
        rtcore::abi_v04::insert_field(
            *words, rtcore::abi_v04::kReportedAttributeWordCount, 1u);
        rtcore::abi_v04::insert_field(
            *words, rtcore::abi_v04::kReportedAttributeFormat, 2u);
    } else if (strcmp(mutation, "unpublished") == 0) {
        (*words)[rtcore::abi_v04::kCommitRetainedCandidate.word] =
            rtcore::abi_v04::shadow::shader_return_unpublished_sentinel();
    } else {
        return false;
    }
    return true;
}

static float rtcore_v04_fp32_value(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool rtcore_v04_retained_triangle_candidate_valid(
    const rtcore_boundary_candidate_snapshot &snapshot)
{
    const rtcore::abi_v04::shadow::boundary_values &values =
        snapshot.v04_boundary_values;
    return snapshot.valid != 0 && snapshot.geometry_type == 1u &&
           snapshot.shader_counter != UINT_MAX && snapshot.hit_data_ref != 0 &&
           values.candidate_valid &&
           values.geometry_type ==
               rtcore::abi_v04::shadow::kBoundaryGeometryTriangle &&
           values.instance_metadata_reference != 0 &&
           snapshot.hit_group_index == values.instance_sbt_contribution &&
           snapshot.geometry_index == values.geometry_index &&
           snapshot.primitive_index == values.primitive_index &&
           snapshot.instance_index == values.instance_custom_index &&
           snapshot.hit_kind == values.hit_kind &&
           values.input_attribute_word_count == 2u &&
           values.input_attribute_location == 0x01u &&
           values.input_attribute_format == 0x01u;
}

static bool rtcore_apply_v04_functional_return_test_mutation_for_lane(
    unsigned lane_id,
    std::array<uint32_t, rtcore::abi_v04::kWordCount> *words,
    const char **mutation_name)
{
    const char *mutation = getenv(
        "VULKAN_SIM_RTCORE_TEST_V04_FUNCTIONAL_SHADER_RETURN_MUTATION");
    if (mutation == NULL || mutation[0] == '\0' ||
        strcmp(mutation, "none") == 0) {
        if (mutation_name != NULL) *mutation_name = "none";
        return true;
    }

    const char *lane_value = getenv(
        "VULKAN_SIM_RTCORE_TEST_V04_FUNCTIONAL_SHADER_RETURN_MUTATION_LANE");
    if (lane_value != NULL && lane_value[0] != '\0') {
        char *end = NULL;
        const unsigned long selected_lane = strtoul(lane_value, &end, 0);
        if (end == lane_value || *end != '\0' || selected_lane >= 32) {
            return false;
        }
        if (selected_lane != lane_id) {
            if (mutation_name != NULL) *mutation_name = "none_filtered";
            return true;
        }
    }

    return rtcore_apply_v04_functional_return_test_mutation(words,
                                                             mutation_name);
}

static bool rtcore_v04_project_terminal_to_compatibility_hit(
    const rtcore_replay_lane_request &request, memory_space *mem,
    const rtcore::abi_v04::shadow::boundary_values &terminal,
    bool *hit_geometry, Hit_data *closest_hit, const char **failure_reason)
{
    const char *failure = "accepted";
    if (hit_geometry == NULL || closest_hit == NULL || mem == NULL) {
        failure = "V04_FUNCTIONAL_RETURN_PROJECTION_CONTEXT_INVALID";
    }
    if (hit_geometry != NULL) *hit_geometry = false;
    if (closest_hit != NULL) *closest_hit = Hit_data();

    if (strcmp(failure, "accepted") == 0 && !terminal.candidate_valid) {
        if (failure_reason != NULL) *failure_reason = failure;
        return true;
    }
    if (strcmp(failure, "accepted") == 0 &&
        (terminal.instance_metadata_reference == 0 ||
         (terminal.instance_metadata_reference & 0x3fu) != 0 ||
         terminal.instance_sbt_contribution > 0x00ffffffu ||
         (terminal.geometry_type !=
              rtcore::abi_v04::shadow::kBoundaryGeometryTriangle &&
          terminal.geometry_type !=
              rtcore::abi_v04::shadow::kBoundaryGeometryProcedural))) {
        failure = "V04_FUNCTIONAL_RETURN_TERMINAL_IDENTITY_INVALID";
    }

    const bool triangle =
        terminal.geometry_type ==
        rtcore::abi_v04::shadow::kBoundaryGeometryTriangle;
    if (strcmp(failure, "accepted") == 0 &&
        ((triangle &&
          (terminal.input_attribute_word_count != 2u ||
           terminal.input_attribute_location != 0x01u ||
           terminal.input_attribute_format != 0x01u)) ||
         (!triangle &&
          (terminal.input_attribute_word_count != 0u ||
           terminal.input_attribute_location != 0u ||
           terminal.input_attribute_format != 0u)))) {
        failure = "V04_FUNCTIONAL_RETURN_TERMINAL_ATTRIBUTE_INVALID";
    }

    GEN_RT_BVH_INSTANCE_LEAF instance_leaf = {};
    if (strcmp(failure, "accepted") == 0) {
        alignas(8) std::array<uint8_t,
                             GEN_RT_BVH_INSTANCE_LEAF_length * 4>
            instance_bytes = {};
        mem->read(reinterpret_cast<void *>(static_cast<uintptr_t>(
                      terminal.instance_metadata_reference)),
                  instance_bytes.size(), instance_bytes.data());
        GEN_RT_BVH_INSTANCE_LEAF_unpack(&instance_leaf,
                                        instance_bytes.data());
        if (instance_leaf.InstanceContributionToHitGroupIndex !=
                terminal.instance_sbt_contribution ||
            instance_leaf.InstanceIndex != terminal.instance_index ||
            instance_leaf.InstanceID != terminal.instance_custom_index) {
            failure = "V04_FUNCTIONAL_RETURN_INSTANCE_METADATA_MISMATCH";
        }
    }

    float hit_t = 0.0f;
    if (strcmp(failure, "accepted") == 0) {
        hit_t = rtcore_v04_fp32_value(terminal.boundary_ray_tmax_fp32);
        if (!std::isfinite(hit_t)) {
            failure = "V04_FUNCTIONAL_RETURN_TERMINAL_T_INVALID";
        }
    }

    if (strcmp(failure, "accepted") == 0) {
        const std::array<uint32_t, rtcore::abi_v04::kWordCount> &trace_words =
            request.v04_shadow_trace_input_words;
        const float3 world_origin = make_float3(
            rtcore_v04_fp32_value(rtcore::abi_v04::extract_field(
                trace_words, rtcore::abi_v04::kWorldRayOriginXFp32)),
            rtcore_v04_fp32_value(rtcore::abi_v04::extract_field(
                trace_words, rtcore::abi_v04::kWorldRayOriginYFp32)),
            rtcore_v04_fp32_value(rtcore::abi_v04::extract_field(
                trace_words, rtcore::abi_v04::kWorldRayOriginZFp32)));
        const float3 world_direction = make_float3(
            rtcore_v04_fp32_value(rtcore::abi_v04::extract_field(
                trace_words, rtcore::abi_v04::kWorldRayDirectionXFp32)),
            rtcore_v04_fp32_value(rtcore::abi_v04::extract_field(
                trace_words, rtcore::abi_v04::kWorldRayDirectionYFp32)),
            rtcore_v04_fp32_value(rtcore::abi_v04::extract_field(
                trace_words, rtcore::abi_v04::kWorldRayDirectionZFp32)));

        closest_hit->geometryType =
            triangle ? VK_GEOMETRY_TYPE_TRIANGLES_KHR
                     : VK_GEOMETRY_TYPE_AABBS_KHR;
        closest_hit->hit_kind = terminal.hit_kind;
        closest_hit->world_min_thit = hit_t;
        closest_hit->geometry_index = terminal.geometry_index;
        closest_hit->primitive_index = terminal.primitive_index;
        closest_hit->intersection_point =
            world_origin + make_float3(world_direction.x * hit_t,
                                       world_direction.y * hit_t,
                                       world_direction.z * hit_t);
        if (triangle) {
            closest_hit->barycentric_coordinates.x = rtcore_v04_fp32_value(
                terminal.inline_attribute_words[0]);
            closest_hit->barycentric_coordinates.y = rtcore_v04_fp32_value(
                terminal.inline_attribute_words[1]);
            closest_hit->barycentric_coordinates.z =
                1.0f - closest_hit->barycentric_coordinates.x -
                closest_hit->barycentric_coordinates.y;
        }
        closest_hit->hitGroupIndex =
            static_cast<int32_t>(terminal.instance_sbt_contribution);
        closest_hit->instance_index = terminal.instance_custom_index;
        closest_hit->instance_id = terminal.instance_index;
        closest_hit->worldToObjectMatrix =
            instance_leaf_matrix_to_float4x4(&instance_leaf.WorldToObjectm00);
        closest_hit->objectToWorldMatrix =
            instance_leaf_matrix_to_float4x4(&instance_leaf.ObjectToWorldm00);
        *hit_geometry = true;
    }

    if (failure_reason != NULL) *failure_reason = failure;
    return strcmp(failure, "accepted") == 0;
}

static bool rtcore_v04_boundary_values_equal(
    const rtcore::abi_v04::shadow::boundary_values &lhs,
    const rtcore::abi_v04::shadow::boundary_values &rhs)
{
    return lhs.candidate_valid == rhs.candidate_valid &&
           lhs.instance_metadata_reference == rhs.instance_metadata_reference &&
           lhs.instance_sbt_contribution == rhs.instance_sbt_contribution &&
           lhs.geometry_index == rhs.geometry_index &&
           lhs.boundary_ray_tmax_fp32 == rhs.boundary_ray_tmax_fp32 &&
           lhs.primitive_index == rhs.primitive_index &&
           lhs.instance_index == rhs.instance_index &&
           lhs.instance_custom_index == rhs.instance_custom_index &&
           lhs.geometry_type == rhs.geometry_type &&
           lhs.hit_kind == rhs.hit_kind &&
           lhs.procedural_any_hit_eligible ==
               rhs.procedural_any_hit_eligible &&
           lhs.input_attribute_word_count == rhs.input_attribute_word_count &&
           lhs.input_attribute_location == rhs.input_attribute_location &&
           lhs.input_attribute_format == rhs.input_attribute_format &&
           lhs.inline_attribute_words == rhs.inline_attribute_words;
}

static bool rtcore_v04_boundary_candidate_snapshot_equal(
    const rtcore_boundary_candidate_snapshot &lhs,
    const rtcore_boundary_candidate_snapshot &rhs)
{
    return lhs.valid == rhs.valid && lhs.event_seq == rhs.event_seq &&
           lhs.shader_counter == rhs.shader_counter &&
           lhs.hit_data_ref == rhs.hit_data_ref &&
           lhs.hit_group_index == rhs.hit_group_index &&
           lhs.geometry_type == rhs.geometry_type &&
           lhs.geometry_index == rhs.geometry_index &&
           lhs.primitive_index == rhs.primitive_index &&
           lhs.instance_index == rhs.instance_index &&
           lhs.hit_kind == rhs.hit_kind &&
           rtcore_v04_boundary_values_equal(lhs.v04_boundary_values,
                                            rhs.v04_boundary_values);
}

struct rtcore_v04_functional_lane_return_plan {
    rtcore_v04_functional_lane_return_plan()
        : valid(false), thread(NULL), request(NULL), mem(NULL),
          traversal_data(NULL), compatibility_candidate_address(NULL),
          invalidate_compatibility_candidate(false),
          publish_triangle_attribute(false), projected_hit_geometry(false),
          reason(rtcore::abi_v04::kReasonNoneOrInvalid), traversal_effect(0),
          compatibility_hit_result(0), action("unsupported"),
          mutation("none"), lane_id(0), lane_address(0), apply_cycle(0)
    {
    }

    bool valid;
    ptx_thread_info *thread;
    rtcore_replay_lane_request *request;
    memory_space *mem;
    Traversal_data *traversal_data;
    Hit_data *compatibility_candidate_address;
    bool invalidate_compatibility_candidate;
    bool publish_triangle_attribute;
    bool projected_hit_geometry;
    Hit_data projected_closest_hit;
    rtcore_boundary_candidate_snapshot retained_candidate;
    rtcore::abi_v04::shadow::boundary_values previous_terminal;
    rtcore::abi_v04::shadow::boundary_values next_terminal;
    unsigned reason;
    unsigned traversal_effect;
    unsigned compatibility_hit_result;
    const char *action;
    const char *mutation;
    unsigned lane_id;
    unsigned long long lane_address;
    unsigned long long apply_cycle;
};

static bool rtcore_prepare_v04_functional_resubmit_lane_return(
    const ptx_instruction *pI, unsigned owner_hw_sid,
    unsigned previous_warp_uid, unsigned warp_id,
    unsigned previous_active_mask, unsigned lane_id,
    unsigned long long handoff_window_base, ptx_thread_info *thread,
    unsigned long long apply_cycle,
    rtcore_v04_functional_lane_return_plan *plan,
    const char **failure_reason)
{
    const char *failure = "accepted";
    if (plan == NULL) {
        failure = "V04_FUNCTIONAL_RETURN_PLAN_MISSING";
    } else {
        *plan = rtcore_v04_functional_lane_return_plan();
    }
    if (strcmp(failure, "accepted") == 0 &&
        (pI == NULL || thread == NULL || thread->RT_thread_data == NULL ||
         thread->RT_thread_data->traversal_data.empty() || lane_id >= 32 ||
         handoff_window_base == 0 ||
         (previous_active_mask & (1u << lane_id)) == 0)) {
        failure = "V04_FUNCTIONAL_RETURN_CONTEXT_MISSING";
    }

    std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
        g_rtcore_replay_lane_requests.end();
    if (strcmp(failure, "accepted") == 0) {
        request_it = g_rtcore_replay_lane_requests.find(thread->get_uid());
        if (request_it == g_rtcore_replay_lane_requests.end() ||
            !request_it->second.valid) {
            failure = "V04_FUNCTIONAL_RETURN_RETAINED_REQUEST_MISSING";
        }
    }
    rtcore_replay_lane_request *request =
        request_it != g_rtcore_replay_lane_requests.end()
            ? &request_it->second
            : NULL;
    if (request != NULL &&
        (request->owner_hw_sid != owner_hw_sid ||
         request->warp_uid != previous_warp_uid ||
         request->warp_id != warp_id ||
         request->active_mask != previous_active_mask ||
         request->lane_id != lane_id ||
         request->handoff_window_base != handoff_window_base ||
         request->state != RTCORE_REPLAY_WAITING_SHADER ||
         !request->continuation_boundary_pending)) {
        failure = "V04_FUNCTIONAL_RETURN_RETAINED_REQUEST_IDENTITY_MISMATCH";
    }
    if (strcmp(failure, "accepted") == 0 &&
        (!request->v04_shadow_boundary_enabled ||
         !request->v04_tlas_binding_enforcement_enabled ||
         !request->v04_shadow_trace_input_valid ||
         request->v04_live_handoff_memory == NULL ||
         request->v04_live_handoff_memory != thread->get_global_memory() ||
         !rtcore_v04_live_publication_matches_current_submit(*request) ||
         !request->v04_live_publication_committed)) {
        failure = "V04_FUNCTIONAL_RETURN_PUBLICATION_NOT_LIVE";
    }

    unsigned reason = rtcore::abi_v04::kReasonNoneOrInvalid;
    unsigned long long lane_address = 0;
    if (strcmp(failure, "accepted") == 0) {
        reason = request->v04_live_publication_reason;
        if (!rtcore::abi_v04::shadow::
                shader_return_reason_requires_publication(reason)) {
            failure = "V04_FUNCTIONAL_RETURN_BOUNDARY_REASON_INVALID";
        } else {
            lane_address = handoff_window_base +
                static_cast<unsigned long long>(lane_id) *
                    rtcore::abi_v04::kLaneSlotBytes;
            if (lane_address < handoff_window_base) {
                failure = "V04_FUNCTIONAL_RETURN_ADDRESS_OVERFLOW";
            }
        }
    }

    std::array<uint32_t, rtcore::abi_v04::kWordCount> words = {};
    const char *mutation = "none";
    rtcore::abi_v04::shadow::shader_return_observation observation;
    if (strcmp(failure, "accepted") == 0) {
        std::array<uint8_t, rtcore::abi_v04::kLaneSlotBytes> image = {};
        request->v04_live_handoff_memory->read_simulator_backing(
            lane_address, image.size(), image.data());
        words = rtcore::abi_v04::decode_words_le(image);
        if (!rtcore_apply_v04_functional_return_test_mutation_for_lane(
                lane_id, &words, &mutation)) {
            failure = "V04_FUNCTIONAL_RETURN_MUTATION_INVALID";
        }
    }
    if (strcmp(failure, "accepted") == 0) {
        observation = rtcore::abi_v04::shadow::decode_shader_return_words(
            words, reason);
        if (!observation.valid()) {
            failure = rtcore::abi_v04::shadow::shader_return_error_name(
                observation.error);
        }
    }

    const rtcore_boundary_candidate_snapshot *retained_candidate =
        request != NULL ? &request->boundary_candidate : NULL;
    const char *action = "unsupported";
    unsigned compatibility_hit_result = 0;
    bool invalidate_candidate = false;
    bool publish_triangle_attribute = false;
    if (strcmp(failure, "accepted") == 0 &&
        (retained_candidate == NULL || retained_candidate->valid == 0)) {
        failure = "V04_FUNCTIONAL_RETURN_RETAINED_CANDIDATE_INVALID";
    }
    if (strcmp(failure, "accepted") == 0 &&
        reason == rtcore::abi_v04::kReasonAnyHitRequired) {
        compatibility_hit_result =
            observation.update.action ==
                    rtcore::abi_v04::shadow::kBoundaryReturnCommitAnyHit
                ? 2u
                : 3u;
        if (!rtcore_v04_retained_triangle_candidate_valid(
                *retained_candidate)) {
            failure = "V04_FUNCTIONAL_RETURN_ANYHIT_BINDING_MISMATCH";
        } else if (observation.update.action ==
                   rtcore::abi_v04::shadow::kBoundaryReturnKeepExisting) {
            invalidate_candidate = true;
            action = "anyhit_ignore_invalidated";
        } else if (observation.update.action !=
                   rtcore::abi_v04::shadow::kBoundaryReturnCommitAnyHit) {
            failure = "V04_FUNCTIONAL_RETURN_ANYHIT_EFFECT_INVALID";
        } else {
            publish_triangle_attribute =
                rtcore::abi_v04::shadow::triangle_candidate_is_strictly_closer(
                    request->v04_replay_committed_boundary_values,
                    retained_candidate->v04_boundary_values);
            action = publish_triangle_attribute
                         ? "anyhit_accept_committed"
                         : "anyhit_accept_kept_closer";
        }
    } else if (strcmp(failure, "accepted") == 0 &&
               reason == rtcore::abi_v04::kReasonIntersectionRequired) {
        compatibility_hit_result =
            observation.update.action ==
                    rtcore::abi_v04::shadow::kBoundaryReturnCommitIntersection
                ? 4u
                : 1u;
        if (!rtcore_v04_retained_procedural_candidate_valid(
                *retained_candidate)) {
            failure = "V04_FUNCTIONAL_RETURN_INTERSECTION_BINDING_MISMATCH";
        } else if (observation.update.action ==
                   rtcore::abi_v04::shadow::kBoundaryReturnKeepExisting) {
            action = "intersection_no_hit";
        } else if (observation.update.action !=
                   rtcore::abi_v04::shadow::kBoundaryReturnCommitIntersection) {
            failure = "V04_FUNCTIONAL_RETURN_INTERSECTION_EFFECT_INVALID";
        } else {
            const float reported_t =
                rtcore_v04_fp32_value(observation.update.reported_t_fp32);
            const float tmin = rtcore_v04_fp32_value(
                rtcore::abi_v04::extract_field(
                    request->v04_shadow_trace_input_words,
                    rtcore::abi_v04::kRayTminFp32));
            const float boundary_tmax = rtcore_v04_fp32_value(
                retained_candidate->v04_boundary_values.
                    boundary_ray_tmax_fp32);
            const rtcore::abi_v04::shadow::boundary_values &current_terminal =
                request->v04_replay_committed_boundary_values;
            const float current_t =
                rtcore_v04_fp32_value(current_terminal.boundary_ray_tmax_fp32);
            const rtcore::procedural_report_ordering ordering =
                rtcore::classify_procedural_report(
                    reported_t, tmin, boundary_tmax,
                    current_terminal.candidate_valid, current_t);
            if (ordering == rtcore::RTCORE_PROCEDURAL_REPORT_INVALID) {
                failure = "V04_FUNCTIONAL_RETURN_INTERSECTION_REPORT_INVALID";
            } else {
                action = ordering == rtcore::RTCORE_PROCEDURAL_REPORT_COMMIT
                             ? "intersection_reported_committed"
                             : "intersection_reported_kept_closer";
            }
        }
    }

    rtcore::abi_v04::shadow::boundary_values next_terminal;
    if (strcmp(failure, "accepted") == 0 &&
        !rtcore::abi_v04::shadow::apply_boundary_return_update(
            request->v04_replay_committed_boundary_values,
            retained_candidate->v04_boundary_values, observation.update,
            &next_terminal)) {
        failure = "V04_FUNCTIONAL_RETURN_RETAINED_CANDIDATE_INVALID";
    }

    Traversal_data *traversal_data = NULL;
    memory_space *mem = NULL;
    bool projected_hit_geometry = false;
    Hit_data projected_closest_hit = {};
    if (strcmp(failure, "accepted") == 0) {
        traversal_data = thread->RT_thread_data->traversal_data.back();
        mem = thread->get_global_memory();
        if (traversal_data == NULL || mem == NULL) {
            failure = "V04_FUNCTIONAL_RETURN_TRAVERSAL_STORAGE_MISSING";
        } else if (!rtcore_v04_project_terminal_to_compatibility_hit(
                       *request, mem, next_terminal,
                       &projected_hit_geometry, &projected_closest_hit,
                       &failure)) {
            // Projection helper supplies the fail-closed reason.
        }
    }

    if (strcmp(failure, "accepted") == 0) {
        plan->valid = true;
        plan->thread = thread;
        plan->request = request;
        plan->mem = mem;
        plan->traversal_data = traversal_data;
        plan->compatibility_candidate_address =
            invalidate_candidate
                ? reinterpret_cast<Hit_data *>(static_cast<uintptr_t>(
                      retained_candidate->hit_data_ref))
                : NULL;
        plan->invalidate_compatibility_candidate = invalidate_candidate;
        plan->publish_triangle_attribute = publish_triangle_attribute;
        plan->projected_hit_geometry = projected_hit_geometry;
        plan->projected_closest_hit = projected_closest_hit;
        plan->retained_candidate = *retained_candidate;
        plan->previous_terminal =
            request->v04_replay_committed_boundary_values;
        plan->next_terminal = next_terminal;
        plan->reason = reason;
        plan->traversal_effect = observation.traversal_effect;
        plan->compatibility_hit_result = compatibility_hit_result;
        plan->action = action;
        plan->mutation = mutation;
        plan->lane_id = lane_id;
        plan->lane_address = lane_address;
        plan->apply_cycle = apply_cycle;
    }

    printf("GPGPU-Sim RTCORE_V04_FUNCTIONAL_SHADER_RETURN_PREPARE "
           "owner_hw_sid=%u previous_warp_uid=%u warp_id=%u lane_id=%u "
           "handoff_window_base=0x%llx reason=%u traversal_effect=%u "
           "mutation=%s decode_before_state_mutation=1 "
           "legacy_return_words_read=0 legacy_shader_type_read=0 "
           "legacy_traversal_state_read=0 result=%s\n",
           owner_hw_sid, previous_warp_uid, warp_id, lane_id,
           handoff_window_base, reason,
           observation.traversal_effect, mutation,
           strcmp(failure, "accepted") == 0 ? "prepared" : failure);
    fflush(stdout);
    if (failure_reason != NULL) *failure_reason = failure;
    return strcmp(failure, "accepted") == 0;
}

static bool rtcore_revalidate_v04_functional_resubmit_lane_plan(
    const rtcore_v04_functional_lane_return_plan &plan,
    const char **failure_reason)
{
    const char *failure = "accepted";
    if (!plan.valid || plan.thread == NULL || plan.request == NULL ||
        plan.mem == NULL || plan.traversal_data == NULL) {
        failure = "V04_FUNCTIONAL_RETURN_REVALIDATION_PLAN_INVALID";
    }

    if (strcmp(failure, "accepted") == 0) {
        std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
            g_rtcore_replay_lane_requests.find(plan.thread->get_uid());
        if (request_it == g_rtcore_replay_lane_requests.end() ||
            &request_it->second != plan.request || !plan.request->valid ||
            plan.request->thread_uid != plan.thread->get_uid() ||
            plan.request->lane_id != plan.lane_id ||
            plan.request->handoff_window_base == 0 ||
            plan.request->state != RTCORE_REPLAY_WAITING_SHADER ||
            !plan.request->continuation_boundary_pending ||
            !rtcore_v04_live_publication_matches_current_submit(
                *plan.request) ||
            !plan.request->v04_live_publication_committed) {
            failure = "V04_FUNCTIONAL_RETURN_REQUEST_GENERATION_CHANGED";
        }
    }
    if (strcmp(failure, "accepted") == 0 &&
        (!rtcore_v04_boundary_candidate_snapshot_equal(
             plan.request->boundary_candidate, plan.retained_candidate) ||
         !rtcore_v04_boundary_values_equal(
             plan.request->v04_replay_committed_boundary_values,
             plan.previous_terminal))) {
        failure = "V04_FUNCTIONAL_RETURN_RETAINED_STATE_CHANGED";
    }
    if (strcmp(failure, "accepted") == 0 &&
        (plan.thread->RT_thread_data == NULL ||
         plan.thread->RT_thread_data->traversal_data.empty() ||
         plan.thread->RT_thread_data->traversal_data.back() !=
             plan.traversal_data ||
         plan.thread->get_global_memory() != plan.mem)) {
        failure = "V04_FUNCTIONAL_RETURN_COMPATIBILITY_STORAGE_CHANGED";
    }
    if (strcmp(failure, "accepted") == 0 &&
        plan.invalidate_compatibility_candidate) {
        const unsigned shader_counter =
            plan.retained_candidate.shader_counter;
        if (shader_counter >=
                plan.thread->RT_thread_data->all_hit_data.size() ||
            plan.thread->RT_thread_data->all_hit_data[shader_counter] !=
                plan.compatibility_candidate_address ||
            reinterpret_cast<uint64_t>(plan.compatibility_candidate_address) !=
                plan.retained_candidate.hit_data_ref) {
            failure = "V04_FUNCTIONAL_RETURN_ANYHIT_CANDIDATE_CHANGED";
        }
    }
    if (failure_reason != NULL) *failure_reason = failure;
    return strcmp(failure, "accepted") == 0;
}

static void rtcore_commit_v04_functional_resubmit_lane_return(
    const ptx_instruction *pI, unsigned owner_hw_sid,
    unsigned previous_warp_uid, unsigned warp_uid, unsigned warp_id,
    unsigned long long handoff_window_base,
    const rtcore_v04_functional_lane_return_plan &plan)
{
    assert(plan.valid && plan.thread != NULL && plan.request != NULL &&
           plan.mem != NULL && plan.traversal_data != NULL);

    // The retained V0.4 terminal is authoritative. Compatibility storage is
    // projected only after the authoritative state has committed.
    plan.request->v04_replay_committed_boundary_values = plan.next_terminal;
    plan.request->hit_geometry_summary_valid = plan.projected_hit_geometry;
    plan.request->closest_hit_kind =
        plan.projected_hit_geometry ? plan.projected_closest_hit.hit_kind : 0u;
    plan.request->closest_hit_geometry_type =
        plan.projected_hit_geometry
            ? (plan.projected_closest_hit.geometryType ==
                       VK_GEOMETRY_TYPE_TRIANGLES_KHR
                   ? 0x01u
                   : 0x02u)
            : 0u;
    plan.request->closest_hit_geometry_index =
        plan.projected_hit_geometry
            ? plan.projected_closest_hit.geometry_index
            : 0u;
    plan.request->closest_hit_primitive_index =
        plan.projected_hit_geometry
            ? plan.projected_closest_hit.primitive_index
            : 0u;
    plan.request->closest_hit_instance_index =
        plan.projected_hit_geometry
            ? plan.projected_closest_hit.instance_index
            : 0u;
    plan.request->instance_sbt_contribution_valid =
        plan.projected_hit_geometry;
    plan.request->instance_sbt_contribution =
        plan.projected_hit_geometry
            ? static_cast<unsigned>(plan.projected_closest_hit.hitGroupIndex)
            : 0u;

    if (plan.invalidate_compatibility_candidate) {
        assert(plan.compatibility_candidate_address != NULL);
        const float invalid_hit = -1.0f;
        plan.mem->write(
            &(plan.compatibility_candidate_address->world_min_thit),
            sizeof(invalid_hit), &invalid_hit, plan.thread, pI);
    }

    plan.mem->write(&(plan.traversal_data->hit_geometry),
                    sizeof(plan.traversal_data->hit_geometry),
                    &plan.projected_hit_geometry, plan.thread, pI);
    if (plan.projected_hit_geometry) {
        plan.mem->write(&(plan.traversal_data->closest_hit),
                        sizeof(plan.projected_closest_hit),
                        &plan.projected_closest_hit, plan.thread, pI);
    }
    if (plan.publish_triangle_attribute && plan.projected_hit_geometry) {
        plan.thread->RT_thread_data->set_hitAttribute(
            plan.projected_closest_hit.barycentric_coordinates, pI,
            plan.thread);
    }

    printf("GPGPU-Sim RTCORE_SHADER_RETURN_DECISION_APPLY "
           "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u warp_id=%u "
           "lane_id=%u handoff_window_base=0x%llx reason=%u "
           "hit_result=%u traversal_effect=%u action=%s "
           "handoff_return_consumed=1 functional_state_applied=1 "
           "functional_oracle_replayed=0 functional_source=v04_handoff "
           "terminal_authority=v04_retained_boundary_values "
           "compatibility_traversal_projection=1 "
           "resubmit_membership_authority=active_mask "
           "apply_cycle=%llu apply_result=applied\n",
           owner_hw_sid, previous_warp_uid, warp_uid, warp_id, plan.lane_id,
           handoff_window_base, plan.reason, plan.compatibility_hit_result,
           plan.traversal_effect, plan.action, plan.apply_cycle);
    fflush(stdout);
}

extern "C" bool rtcore_refresh_shader_visible_resubmit_lane_terminal_facts(
    unsigned owner_hw_sid, unsigned previous_warp_uid, unsigned warp_id,
    unsigned previous_active_mask, unsigned lane_id, ptx_thread_info *thread,
    const rtcore::abi_v04::shadow::boundary_return_update *return_update,
    unsigned long long refresh_cycle, const char **failure_reason)
{
    const char *reason = "accepted";
    if (thread == NULL || thread->RT_thread_data == NULL ||
        thread->RT_thread_data->traversal_data.empty() || lane_id >= 32 ||
        (previous_active_mask & (1u << lane_id)) == 0) {
        reason = "REFRESH_THREAD_OR_TRAVERSAL_STATE_MISSING";
    }

    std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
        g_rtcore_replay_lane_requests.end();
    if (strcmp(reason, "accepted") == 0) {
        request_it = g_rtcore_replay_lane_requests.find(thread->get_uid());
        if (request_it == g_rtcore_replay_lane_requests.end() ||
            !request_it->second.valid) {
            reason = "REFRESH_RETAINED_REQUEST_MISSING";
        }
    }

    rtcore_replay_lane_request *request =
        request_it != g_rtcore_replay_lane_requests.end()
            ? &request_it->second
            : NULL;
    if (request != NULL &&
        (request->owner_hw_sid != owner_hw_sid ||
         request->warp_uid != previous_warp_uid ||
         request->warp_id != warp_id ||
         request->active_mask != previous_active_mask ||
         request->lane_id != lane_id ||
         request->state != RTCORE_REPLAY_WAITING_SHADER ||
         !request->continuation_boundary_pending)) {
        reason = "REFRESH_RETAINED_REQUEST_IDENTITY_MISMATCH";
    }
    if (strcmp(reason, "accepted") == 0 &&
        request->v04_shadow_boundary_enabled && return_update == NULL) {
        reason = "REFRESH_V04_RETURN_UPDATE_MISSING";
    }

    bool hit_geometry = false;
    Hit_data closest_hit = {};
    if (strcmp(reason, "accepted") == 0) {
        Traversal_data *traversal_data =
            thread->RT_thread_data->traversal_data.back();
        memory_space *mem = thread->get_global_memory();
        if (traversal_data == NULL || mem == NULL) {
            reason = "REFRESH_TRAVERSAL_MEMORY_MISSING";
        } else {
            mem->read(&(traversal_data->hit_geometry),
                      sizeof(traversal_data->hit_geometry), &hit_geometry);
            if (hit_geometry) {
                mem->read(&(traversal_data->closest_hit),
                          sizeof(traversal_data->closest_hit), &closest_hit);
                if (closest_hit.hitGroupIndex < 0 ||
                    (closest_hit.geometryType !=
                         VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
                     closest_hit.geometryType != VK_GEOMETRY_TYPE_AABBS_KHR)) {
                    reason = "REFRESH_CLOSEST_HIT_STATE_INVALID";
                }
            }
        }
    }

    if (strcmp(reason, "accepted") == 0) {
        if (request->v04_shadow_boundary_enabled) {
            rtcore::abi_v04::shadow::boundary_values next_terminal;
            if (!rtcore::abi_v04::shadow::apply_boundary_return_update(
                    request->v04_replay_committed_boundary_values,
                    request->boundary_candidate.v04_boundary_values,
                    *return_update, &next_terminal)) {
                reason = "REFRESH_V04_RETURN_UPDATE_INVALID";
            } else {
                request->v04_replay_committed_boundary_values = next_terminal;
            }
        }
    }

    if (strcmp(reason, "accepted") == 0) {
        request->hit_geometry_summary_valid = hit_geometry;
        request->closest_hit_kind =
            hit_geometry ? closest_hit.hit_kind : 0u;
        request->closest_hit_geometry_type =
            hit_geometry
                ? (closest_hit.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR
                       ? 0x01u
                       : 0x02u)
                : 0u;
        request->closest_hit_geometry_index =
            hit_geometry ? closest_hit.geometry_index : 0u;
        request->closest_hit_primitive_index =
            hit_geometry ? closest_hit.primitive_index : 0u;
        request->closest_hit_instance_index =
            hit_geometry ? closest_hit.instance_index : 0u;
        request->instance_sbt_contribution_valid = hit_geometry;
        request->instance_sbt_contribution =
            hit_geometry
                ? static_cast<unsigned>(closest_hit.hitGroupIndex)
                : 0u;
        printf("GPGPU-Sim RTCORE_SHADER_RETURN_TERMINAL_FACTS_REFRESH "
               "owner_hw_sid=%u previous_warp_uid=%u warp_id=%u "
               "lane_id=%u hit_geometry=%u closest_hit_kind=%u "
               "closest_hit_geometry_type=%u "
               "closest_hit_geometry_index=%u "
               "closest_hit_primitive_index=%u "
               "closest_hit_instance_index=%u "
               "instance_sbt_contribution=%u "
               "v04_return_action=%u v04_terminal_valid=%u "
               "v04_terminal_instance_metadata_ref=0x%llx "
               "v04_terminal_attribute_word_count=%u "
               "refresh_source=shader_return_terminal_facts_refresh "
               "refresh_cycle=%llu\n",
               owner_hw_sid, previous_warp_uid, warp_id, lane_id,
               hit_geometry ? 1u : 0u, request->closest_hit_kind,
               request->closest_hit_geometry_type,
               request->closest_hit_geometry_index,
               request->closest_hit_primitive_index,
               request->closest_hit_instance_index,
               request->instance_sbt_contribution,
               return_update != NULL
                   ? static_cast<unsigned>(return_update->action)
                   : 0u,
               request->v04_replay_committed_boundary_values.candidate_valid
                   ? 1u
                                                                       : 0u,
               static_cast<unsigned long long>(
                   request->v04_replay_committed_boundary_values
                       .instance_metadata_reference),
               request->v04_replay_committed_boundary_values
                   .input_attribute_word_count,
               refresh_cycle);
        fflush(stdout);
    }

    if (strcmp(reason, "accepted") == 0 &&
        !rtcore_restore_v04_live_trace_input_after_legacy_shader_return(
            request, refresh_cycle, &reason)) {
        // The caller converts a failed refresh into the existing fail-closed
        // resubmit path.
    }

    if (failure_reason != NULL) {
        *failure_reason = reason;
    }
    return strcmp(reason, "accepted") == 0;
}

extern "C" bool rtcore_validate_shader_visible_resubmit_lane(
    unsigned owner_hw_sid, unsigned new_warp_uid, unsigned warp_id,
    unsigned next_active_mask, unsigned lane_id, unsigned thread_uid,
    unsigned long long context_ptr, unsigned long long handoff_window_base,
    unsigned token_id, unsigned token_allocator_generation,
    unsigned window_generation, const char **failure_reason)
{
    const char *reason = "accepted";
    const char *test_failpoint_mode =
        rtcore_resubmit_admission_test_failpoint_mode();
    const bool inject_missing =
        test_failpoint_mode && strcmp(test_failpoint_mode, "missing") == 0;
    const bool inject_stale =
        test_failpoint_mode && strcmp(test_failpoint_mode, "stale") == 0;
    const bool inject_owner =
        test_failpoint_mode && strcmp(test_failpoint_mode, "owner") == 0;
    const bool inject_final_wait =
        test_failpoint_mode && strcmp(test_failpoint_mode, "final_wait") == 0;
    const rtcore_resident_rt_warp_record_key key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::iterator resident =
        g_rtcore_resident_rt_warp_records.find(key);
    if (inject_missing ||
        resident == g_rtcore_resident_rt_warp_records.end() ||
        !resident->second.valid) {
        reason = "MISSING_RESIDENT_RECORD";
    } else if (new_warp_uid == resident->second.current_warp_uid) {
        reason = "DUPLICATE_CURRENT_SUBMIT";
    } else if (next_active_mask == 0 ||
               (next_active_mask & ~resident->second.active_mask) != 0) {
        reason = "NEXT_MASK_NOT_NONZERO_SUBSET";
    } else if (lane_id >= 32 ||
               (next_active_mask & (1u << lane_id)) == 0) {
        reason = "LANE_NOT_IN_NEXT_MASK";
    } else if (inject_stale ||
               !rtcore_resident_lane_identity_matches(
                   resident->second.lane_identity[lane_id], thread_uid,
                   context_ptr, handoff_window_base, token_id,
                   token_allocator_generation, window_generation)) {
        reason = "STALE_OR_OWNER_IDENTITY_MISMATCH";
    } else {
        const rtcore_resident_rt_warp_lane_identity &identity =
            resident->second.lane_identity[lane_id];
        std::map<unsigned, rtcore_replay_lane_request>::const_iterator request =
            g_rtcore_replay_lane_requests.find(identity.thread_uid);
        if (request == g_rtcore_replay_lane_requests.end() ||
            !request->second.valid) {
            reason = "MISSING_PINNED_REQUEST_STATE";
        } else if (inject_owner ||
                   request->second.owner_hw_sid != owner_hw_sid ||
                   request->second.warp_uid !=
                       resident->second.current_warp_uid ||
                   request->second.warp_id != warp_id ||
                   request->second.active_mask != resident->second.active_mask ||
                   request->second.lane_id != lane_id) {
            reason = "REQUEST_STATE_OWNER_MISMATCH";
        } else if (request->second.v04_tlas_binding_enforcement_enabled) {
            const char *binding_failure = "unvalidated";
            const bool binding_valid =
                VulkanRayTracing::validateTlasBinding(
                    request->second.v04_tlas_binding, 0, 0,
                    &binding_failure);
            printf("GPGPU-Sim RTCORE_V04_TLAS_RESUBMIT_REVALIDATION "
                   "owner_hw_sid=%u previous_warp_uid=%u warp_id=%u "
                   "lane_id=%u tlas_object_id=%llu tlas_generation=%u "
                   "result=%s failure=%s\n",
                   owner_hw_sid, resident->second.current_warp_uid, warp_id,
                   lane_id,
                   (unsigned long long)
                       request->second.v04_tlas_binding.object_id,
                   request->second.v04_tlas_binding.generation,
                   binding_valid ? "accepted" : "rejected",
                   binding_failure);
            fflush(stdout);
            if (!binding_valid) {
                reason = "TLAS_BINDING_STALE_OR_RELEASED";
            } else if (inject_final_wait ||
                       request->second.state ==
                           RTCORE_REPLAY_FINAL_WAIT_RETIRE) {
                reason = "FINAL_WAIT_RETIRE_RESUBMIT";
            } else if (request->second.state != RTCORE_REPLAY_WAITING_SHADER ||
                       !request->second.continuation_boundary_pending) {
                reason = "REQUEST_STATE_NOT_WAITING_SHADER";
            }
        } else if (inject_final_wait ||
                   request->second.state == RTCORE_REPLAY_FINAL_WAIT_RETIRE) {
            reason = "FINAL_WAIT_RETIRE_RESUBMIT";
        } else if (request->second.state != RTCORE_REPLAY_WAITING_SHADER ||
                   !request->second.continuation_boundary_pending) {
            reason = "REQUEST_STATE_NOT_WAITING_SHADER";
        }
    }

    if (failure_reason) {
        *failure_reason = reason;
    }
    return strcmp(reason, "accepted") == 0;
}

struct rtcore_shader_visible_resubmit_admission_plan {
    rtcore_shader_visible_resubmit_admission_plan()
        : valid(false), record(NULL), old_active_mask(0),
          continuation_depth(0), occupancy_before(0),
          v04_request_owner_plan(), v04_private_shared_plan()
    {
        old_key = rtcore_replay_warp_completion_entry_key();
    }

    bool valid;
    rtcore_resident_rt_warp_record *record;
    rtcore_replay_warp_completion_entry_key old_key;
    unsigned old_active_mask;
    unsigned continuation_depth;
    unsigned occupancy_before;
    rtcore::v04::request_owner::mask_shrink_plan_v0
        v04_request_owner_plan;
    rtcore::v04::private_shared::mask_shrink_plan_v0
        v04_private_shared_plan;
};

static bool rtcore_prepare_shader_visible_resubmit_admission(
    unsigned owner_hw_sid, unsigned new_warp_uid, unsigned warp_id,
    unsigned next_active_mask, unsigned expected_previous_warp_uid,
    unsigned expected_resident_generation,
    rtcore_shader_visible_resubmit_admission_plan *plan,
    const char **failure_reason)
{
    const char *reason = "accepted";
    if (plan == NULL) {
        reason = "RESUBMIT_ADMISSION_PLAN_MISSING";
    } else {
        *plan = rtcore_shader_visible_resubmit_admission_plan();
        plan->occupancy_before = rtcore_resident_rt_warp_record_occupancy();
    }

    const rtcore_resident_rt_warp_record_key resident_key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::iterator resident =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    if (strcmp(reason, "accepted") == 0 &&
        (resident == g_rtcore_resident_rt_warp_records.end() ||
         !resident->second.valid)) {
        reason = "MISSING_RESIDENT_RECORD";
    }

    rtcore_resident_rt_warp_record *record =
        strcmp(reason, "accepted") == 0 ? &resident->second : NULL;
    const unsigned old_active_mask = record ? record->active_mask : 0;
    if (record &&
        (record->current_warp_uid != expected_previous_warp_uid ||
         record->resident_generation != expected_resident_generation ||
         new_warp_uid == record->current_warp_uid || next_active_mask == 0 ||
         (next_active_mask & ~old_active_mask) != 0)) {
        reason = "RESIDENT_GENERATION_OR_SUBMIT_ID_INVALID";
    }

    rtcore_replay_warp_completion_entry_key old_key = {};
    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_continuation_warp_boundary_state>::const_iterator boundary =
        g_rtcore_continuation_warp_boundary_states.end();
    if (record && strcmp(reason, "accepted") == 0) {
        old_key.owner_hw_sid = record->owner_hw_sid;
        old_key.warp_uid = record->current_warp_uid;
        old_key.warp_id = record->warp_id;
        old_key.active_mask = record->active_mask;
        boundary = g_rtcore_continuation_warp_boundary_states.find(old_key);
        if (boundary == g_rtcore_continuation_warp_boundary_states.end() ||
            !boundary->second.valid || !boundary->second.packet_published) {
            reason = "MISSING_OR_UNPUBLISHED_BOUNDARY_STATE";
        }
    }
    if (strcmp(reason, "accepted") == 0 &&
        g_rtcore_replay_warp_completion_entries.find(old_key) !=
            g_rtcore_replay_warp_completion_entries.end()) {
        reason = "OLD_COMPLETION_PACKET_STILL_LIVE";
    }

    if (record && strcmp(reason, "accepted") == 0) {
        const unsigned admitted_lane_mask =
            rtcore_compute_resident_rt_warp_admitted_lane_mask(*record);
        if ((record->bound_lane_mask & old_active_mask) != old_active_mask ||
            (admitted_lane_mask & old_active_mask) != old_active_mask) {
            reason = "RESIDENT_LANE_BINDING_INCOMPLETE";
        }
    }

    if (record && strcmp(reason, "accepted") == 0) {
        for (unsigned lane = 0; lane < 32; ++lane) {
            const unsigned lane_mask = 1u << lane;
            if ((old_active_mask & lane_mask) == 0) {
                continue;
            }
            const rtcore_resident_rt_warp_lane_identity &identity =
                record->lane_identity[lane];
            std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
                g_rtcore_replay_lane_requests.find(identity.thread_uid);
            if (!identity.valid ||
                it == g_rtcore_replay_lane_requests.end() ||
                !it->second.valid) {
                reason = "MISSING_PINNED_REQUEST_STATE";
                break;
            }
            const rtcore_replay_lane_request &request = it->second;
            if (request.owner_hw_sid != record->owner_hw_sid ||
                request.warp_uid != record->current_warp_uid ||
                request.warp_id != record->warp_id ||
                request.active_mask != old_active_mask ||
                request.lane_id != lane) {
                reason = "REQUEST_STATE_OWNER_MISMATCH";
                break;
            }
            if (request.memory_address_gen_latency_gate_pending ||
                request.memory_wake_latency_gate_pending ||
                request.memory_contention_gate_pending ||
                request.v02_lsu_response_wait_gate_pending ||
                request.unit_latency_gate_pending ||
                rtcore_replay_request_state_has_memory_outstanding_work(
                    request)) {
                reason = "REQUEST_STATE_IN_FLIGHT_CONFLICT";
                break;
            }
            if ((next_active_mask & lane_mask) != 0) {
                if (request.state == RTCORE_REPLAY_FINAL_WAIT_RETIRE) {
                    reason = "FINAL_WAIT_RETIRE_RESUBMIT";
                    break;
                }
                if (request.state != RTCORE_REPLAY_WAITING_SHADER ||
                    !request.continuation_boundary_pending) {
                    reason = "REQUEST_STATE_NOT_WAITING_SHADER";
                    break;
                }
            } else if (!((request.state == RTCORE_REPLAY_WAITING_SHADER &&
                          request.continuation_boundary_pending) ||
                         request.state == RTCORE_REPLAY_FINAL_WAIT_RETIRE)) {
                reason = "MASK_SHRINK_REQUEST_NOT_RELEASABLE";
                break;
            }
        }
    }

    if (record && strcmp(reason, "accepted") == 0 &&
        rtcore_v04_request_owner_binding_enabled()) {
        if (!record->v04_request_owner_binding_valid ||
            record->v04_resident_warp_slot >=
                rtcore::v04::request_owner::kResidentWarpCapacity) {
            reason = "REQUEST_OWNER_RESUBMIT_BINDING_MISSING";
        } else {
            const rtcore::v04::request_owner::status_kind owner_status =
                rtcore::v04::request_owner::prepare_mask_shrink(
                    rtcore_v04_request_owner_allocator_for(owner_hw_sid),
                    static_cast<uint8_t>(record->v04_resident_warp_slot),
                    owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
                    warp_id, next_active_mask,
                    &plan->v04_request_owner_plan);
            if (owner_status !=
                rtcore::v04::request_owner::kStatusOk) {
                reason = rtcore::v04::request_owner::status_name(
                    owner_status);
            }
        }
    }
    if (record && strcmp(reason, "accepted") == 0 &&
        rtcore_v04_private_frontier_live_init_enabled()) {
        if (!record->v04_private_frontier_live_init_valid ||
            !record->v04_private_frontier_init_committed) {
            reason = "PRIVATE_FRONTIER_RESUBMIT_INIT_INCOMPLETE";
        } else {
            const rtcore::v04::private_shared::status_kind private_status =
                rtcore::v04::private_shared::prepare_mask_shrink(
                    rtcore_v04_private_shared_backing_for(owner_hw_sid),
                    static_cast<uint8_t>(record->v04_resident_warp_slot),
                    expected_previous_warp_uid, new_warp_uid, warp_id,
                    next_active_mask, &plan->v04_private_shared_plan);
            if (private_status !=
                rtcore::v04::private_shared::kStatusOk) {
                reason =
                    rtcore::v04::private_shared::status_name(private_status);
            }
        }
    }

    const char *test_failpoint_mode =
        rtcore_resubmit_admission_test_failpoint_mode();
    if (strcmp(reason, "accepted") == 0 && test_failpoint_mode != NULL &&
        strcmp(test_failpoint_mode, "late") == 0) {
        reason = "TEST_LATE_RESUBMIT_ADMISSION_FAILURE";
    }

    if (strcmp(reason, "accepted") == 0) {
        plan->valid = true;
        plan->record = record;
        plan->old_key = old_key;
        plan->old_active_mask = old_active_mask;
        plan->continuation_depth = boundary->second.continuation_depth;
    }
    if (failure_reason != NULL) *failure_reason = reason;
    return strcmp(reason, "accepted") == 0;
}

static bool rtcore_commit_prepared_shader_visible_resubmit_admission(
    const rtcore_shader_visible_resubmit_admission_plan &plan,
    unsigned owner_hw_sid, unsigned new_warp_uid, unsigned warp_id,
    unsigned new_static_inst_uid, unsigned next_active_mask,
    unsigned expected_previous_warp_uid,
    unsigned expected_resident_generation,
    unsigned long long service_cycle, unsigned *previous_active_mask,
    unsigned *released_lane_mask, unsigned *reactivated_lane_mask,
    unsigned *resident_occupancy_before, unsigned *resident_occupancy_after)
{
    assert(plan.valid && plan.record != NULL);
    rtcore_resident_rt_warp_record *record = plan.record;
    const unsigned old_active_mask = plan.old_active_mask;
    unsigned released_mask = 0;
    unsigned reactivated_mask = 0;

    if (rtcore_v04_request_owner_binding_enabled()) {
        const rtcore::v04::request_owner::status_kind owner_status =
            rtcore::v04::request_owner::commit_mask_shrink(
                &rtcore_v04_request_owner_allocator_for(owner_hw_sid),
                plan.v04_request_owner_plan);
        if (owner_status != rtcore::v04::request_owner::kStatusOk) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_REQUEST_OWNER_COMMIT_INVARIANT "
                    "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u "
                    "warp_id=%u fault=%s\n",
                    owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
                    warp_id,
                    rtcore::v04::request_owner::status_name(owner_status));
            fflush(stderr);
            abort();
        }
        printf("GPGPU-Sim RTCORE_V04_REQUEST_OWNER_MASK_SHRINK "
               "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u "
               "warp_id=%u resident_warp_slot=%u "
               "previous_active_mask=0x%08x next_active_mask=0x%08x "
               "released_lane_mask=0x%08x retained_lane_mask=0x%08x "
               "allocator_scope=per_sm result=accepted\n",
               owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
               warp_id, plan.v04_request_owner_plan.resident_warp_slot,
               plan.v04_request_owner_plan.previous_active_mask,
               plan.v04_request_owner_plan.next_active_mask,
               plan.v04_request_owner_plan.release_mask,
               plan.v04_request_owner_plan.next_active_mask);
        fflush(stdout);
    }
    if (rtcore_v04_private_frontier_live_init_enabled()) {
        const rtcore::v04::private_shared::status_kind private_status =
            rtcore::v04::private_shared::commit_mask_shrink(
                &rtcore_v04_private_shared_backing_for(owner_hw_sid),
                plan.v04_private_shared_plan);
        if (private_status !=
            rtcore::v04::private_shared::kStatusOk) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                    "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u "
                    "warp_id=%u fault=%s\n",
                    owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
                    warp_id,
                    rtcore::v04::private_shared::status_name(
                        private_status));
            fflush(stderr);
            abort();
        }
    }

    rtcore_resident_warp_continuation_state log_state;
    log_state.valid = true;
    log_state.owner_hw_sid = record->owner_hw_sid;
    log_state.warp_uid = record->current_warp_uid;
    log_state.warp_id = record->warp_id;
    log_state.active_mask = old_active_mask;
    log_state.resume_required_mask = next_active_mask;
    log_state.shader_required_mask = next_active_mask;
    log_state.continuation_depth = plan.continuation_depth;

    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((old_active_mask & lane_mask) == 0) {
            continue;
        }
        const unsigned thread_uid = record->lane_identity[lane].thread_uid;
        std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
            g_rtcore_replay_lane_requests.find(thread_uid);
        assert(request_it != g_rtcore_replay_lane_requests.end());
        rtcore_replay_lane_request &request = request_it->second;
        if ((next_active_mask & lane_mask) == 0) {
            const bool consumed_entry =
                rtcore_replay_lane_request_state_capacity_consumes_entry(
                    request);
            request.continuation_boundary_pending = false;
            request.state = RTCORE_REPLAY_COMPLETED;
            rtcore_record_replay_request_state_write();
            rtcore_refresh_replay_lane_request_ready_bits(&request);
            if (consumed_entry) {
                rtcore_record_replay_lane_request_state_capacity_release(
                    request, service_cycle);
            }
            if (rtcore_v04_request_owner_binding_enabled()) {
                request.v04_request_owner_binding_valid = false;
                record->lane_identity[lane]
                    .v04_request_owner_binding_valid = false;
            }
            released_mask |= lane_mask;
            rtcore_log_continuation_request_state_reconcile(
                request, log_state, "release_mask_shrink", service_cycle);
            continue;
        }

        rtcore_invalidate_v04_live_handoff_publication_for_resubmit(
            &request, new_warp_uid, service_cycle);
        request.warp_uid = new_warp_uid;
        request.static_inst_uid = new_static_inst_uid;
        request.active_mask = next_active_mask;
        request.continuation_boundary_pending = false;
        request.continuation_depth = plan.continuation_depth;
        request.continuation_segment_event_count = 0;
        request.ready_order = g_rtcore_next_replay_ready_order++;
        request.request_state_bank_id =
            rtcore_replay_request_state_bank_for_request(request);
        if (rtcore_v04_request_owner_binding_enabled() &&
            (!request.v04_request_owner_binding_valid ||
             !record->lane_identity[lane]
                  .v04_request_owner_binding_valid ||
             !rtcore::v04::request_owner::validate_live_binding(
                 rtcore_v04_request_owner_allocator_for(owner_hw_sid),
                 request.v04_request_owner_binding, new_warp_uid,
                 next_active_mask))) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_REQUEST_OWNER_COMMIT_INVARIANT "
                    "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u "
                    "warp_id=%u lane_id=%u "
                    "fault=retained_binding_not_live\n",
                    owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
                    warp_id, lane);
            fflush(stderr);
            abort();
        }
        if (request.next_event_index >= request.events.size()) {
            rtcore_prepare_replay_request_completion_ingress(&request);
        } else {
            request.state = rtcore_classify_replay_state(
                rtcore_unpack_compact_trace_event_type(
                    request.events[request.next_event_index]));
            rtcore_record_replay_request_state_write();
            rtcore_refresh_replay_lane_request_ready_bits(&request);
        }
        rtcore_record_replay_lane_admission_entry(request);
        const bool routed =
            rtcore_route_admitted_replay_request(thread_uid, service_cycle);
        if (!routed) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_RESUBMIT_COMMIT_INVARIANT "
                    "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u "
                    "warp_id=%u lane_id=%u fault=prepared_route_failed\n",
                    owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
                    warp_id, lane);
            fflush(stderr);
            abort();
        }
        reactivated_mask |= lane_mask;
        rtcore_log_continuation_request_state_reconcile(
            request, log_state, "reactivate_resubmit", service_cycle);
    }

    record->current_warp_uid = new_warp_uid;
    record->current_static_inst_uid = new_static_inst_uid;
    record->active_mask = next_active_mask;
    record->admitted_lane_mask = next_active_mask;
    record->resubmit_count++;
    if (rtcore_v04_request_owner_binding_enabled()) {
        record->v04_request_owner_active_mask = next_active_mask;
    }
    if (rtcore_v04_private_frontier_live_init_enabled()) {
        record->v04_private_frontier_init_active_mask = next_active_mask;
    }
    g_rtcore_continuation_warp_boundary_states.erase(plan.old_key);
    g_rtcore_resident_warp_continuation_states.erase(plan.old_key);
    g_rtcore_continuation_stats.rtcore_modeled_resubmit_count++;
    g_rtcore_continuation_stats.rtcore_modeled_resubmit_lane_count +=
        rtcore_count_replay_warp_completion_entry_lanes(reactivated_mask);

    printf("GPGPU-Sim RTCORE_CONTINUATION_MODELED_RESUBMIT "
           "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u warp_id=%u "
           "resident_generation=%u "
           "active_mask=0x%08x resume_required_mask=0x%08x "
           "released_lane_mask=0x%08x reactivated_lane_mask=0x%08x "
           "continuation_depth=%u resubmit_source=shader_visible_rt_submit "
           "service_cycle=%llu\n",
           owner_hw_sid, expected_previous_warp_uid, new_warp_uid, warp_id,
           expected_resident_generation, old_active_mask, next_active_mask,
           released_mask, reactivated_mask, log_state.continuation_depth,
           service_cycle);
    fflush(stdout);

    const unsigned occupancy_after =
        rtcore_resident_rt_warp_record_occupancy();
    if (previous_active_mask) *previous_active_mask = old_active_mask;
    if (released_lane_mask) *released_lane_mask = released_mask;
    if (reactivated_lane_mask) *reactivated_lane_mask = reactivated_mask;
    if (resident_occupancy_before) {
        *resident_occupancy_before = plan.occupancy_before;
    }
    if (resident_occupancy_after) *resident_occupancy_after = occupancy_after;
    return reactivated_mask == next_active_mask &&
           plan.occupancy_before == occupancy_after;
}

extern "C" bool rtcore_commit_shader_visible_resubmit_admission(
    unsigned owner_hw_sid, unsigned new_warp_uid, unsigned warp_id,
    unsigned new_static_inst_uid, unsigned next_active_mask,
    unsigned expected_previous_warp_uid,
    unsigned expected_resident_generation,
    unsigned long long service_cycle, unsigned *previous_active_mask,
    unsigned *released_lane_mask, unsigned *reactivated_lane_mask,
    unsigned *resident_occupancy_before, unsigned *resident_occupancy_after,
    const char **failure_reason)
{
    const char *reason = "accepted";
    rtcore_shader_visible_resubmit_admission_plan prepared_plan;
    rtcore_prepare_shader_visible_resubmit_admission(
        owner_hw_sid, new_warp_uid, warp_id, next_active_mask,
        expected_previous_warp_uid, expected_resident_generation,
        &prepared_plan, &reason);
    const unsigned old_active_mask = prepared_plan.old_active_mask;
    const unsigned occupancy_before = prepared_plan.occupancy_before;
    if (strcmp(reason, "accepted") != 0) {
        if (previous_active_mask) *previous_active_mask = old_active_mask;
        if (released_lane_mask) *released_lane_mask = 0;
        if (reactivated_lane_mask) *reactivated_lane_mask = 0;
        if (resident_occupancy_before) {
            *resident_occupancy_before = occupancy_before;
        }
        if (resident_occupancy_after) {
            *resident_occupancy_after = occupancy_before;
        }
        if (failure_reason) *failure_reason = reason;
        return false;
    }

    const bool committed =
        rtcore_commit_prepared_shader_visible_resubmit_admission(
            prepared_plan, owner_hw_sid, new_warp_uid, warp_id,
            new_static_inst_uid, next_active_mask,
            expected_previous_warp_uid, expected_resident_generation,
            service_cycle, previous_active_mask, released_lane_mask,
            reactivated_lane_mask, resident_occupancy_before,
            resident_occupancy_after);
    if (failure_reason) {
        *failure_reason = committed ? "accepted"
                                    : "RESUBMIT_COMMIT_INVARIANT_FAILED";
    }
    return committed;
}

struct rtcore_v04_atomic_state_snapshot {
    rtcore_v04_atomic_state_snapshot()
        : valid(false), lane_mask(0), live_handoff_digest(0),
          retained_state_digest(0), compatibility_state_digest(0),
          request_state_digest(0), resident_state_digest(0)
    {
    }

    bool valid;
    unsigned lane_mask;
    uint64_t live_handoff_digest;
    uint64_t retained_state_digest;
    uint64_t compatibility_state_digest;
    uint64_t request_state_digest;
    uint64_t resident_state_digest;
};

static bool rtcore_v04_atomic_state_snapshot_enabled()
{
    return rtcore_candidate_gate_state_for(
               "VULKAN_SIM_RTCORE_TEST_V04_ATOMIC_STATE_SNAPSHOT") ==
           RTCORE_CANDIDATE_GATE_ENABLED;
}

static void rtcore_v04_atomic_digest_bytes(
    uint64_t *digest, const void *data, size_t size)
{
    assert(digest != NULL && (data != NULL || size == 0));
    const unsigned char *bytes =
        static_cast<const unsigned char *>(data);
    for (size_t index = 0; index < size; ++index) {
        *digest ^= bytes[index];
        *digest *= UINT64_C(1099511628211);
    }
}

static bool rtcore_capture_v04_atomic_state_snapshot(
    unsigned owner_hw_sid, unsigned previous_warp_uid, unsigned warp_id,
    unsigned previous_active_mask, unsigned next_active_mask,
    unsigned long long handoff_window_base,
    ptx_thread_info *const *lane_threads,
    rtcore_v04_atomic_state_snapshot *snapshot)
{
    if (snapshot == NULL) return false;
    *snapshot = rtcore_v04_atomic_state_snapshot();
    if (lane_threads == NULL || previous_active_mask == 0 ||
        next_active_mask == 0 ||
        (next_active_mask & ~previous_active_mask) != 0 ||
        handoff_window_base == 0) {
        return false;
    }

    const uint64_t digest_seed = UINT64_C(1469598103934665603);
    snapshot->lane_mask = next_active_mask;
    snapshot->live_handoff_digest = digest_seed;
    snapshot->retained_state_digest = digest_seed;
    snapshot->compatibility_state_digest = digest_seed;
    snapshot->request_state_digest = digest_seed;
    snapshot->resident_state_digest = digest_seed;

    const rtcore_resident_rt_warp_record_key resident_key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::const_iterator resident =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    if (resident == g_rtcore_resident_rt_warp_records.end() ||
        !resident->second.valid ||
        resident->second.current_warp_uid != previous_warp_uid ||
        resident->second.active_mask != previous_active_mask) {
        return false;
    }
    rtcore_v04_atomic_digest_bytes(
        &snapshot->resident_state_digest, &resident->second,
        sizeof(resident->second));
    const size_t resident_record_count =
        g_rtcore_resident_rt_warp_records.size();
    const size_t boundary_state_count =
        g_rtcore_continuation_warp_boundary_states.size();
    const size_t continuation_state_count =
        g_rtcore_resident_warp_continuation_states.size();
    const size_t completion_entry_count =
        g_rtcore_replay_warp_completion_entries.size();
    rtcore_v04_atomic_digest_bytes(
        &snapshot->resident_state_digest, &resident_record_count,
        sizeof(resident_record_count));
    rtcore_v04_atomic_digest_bytes(
        &snapshot->resident_state_digest, &boundary_state_count,
        sizeof(boundary_state_count));
    rtcore_v04_atomic_digest_bytes(
        &snapshot->resident_state_digest, &continuation_state_count,
        sizeof(continuation_state_count));
    rtcore_v04_atomic_digest_bytes(
        &snapshot->resident_state_digest, &completion_entry_count,
        sizeof(completion_entry_count));

    rtcore_replay_warp_completion_entry_key boundary_key = {};
    boundary_key.owner_hw_sid = owner_hw_sid;
    boundary_key.warp_uid = previous_warp_uid;
    boundary_key.warp_id = warp_id;
    boundary_key.active_mask = previous_active_mask;
    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_continuation_warp_boundary_state>::const_iterator boundary =
        g_rtcore_continuation_warp_boundary_states.find(boundary_key);
    if (boundary == g_rtcore_continuation_warp_boundary_states.end() ||
        !boundary->second.valid) {
        return false;
    }
    rtcore_v04_atomic_digest_bytes(
        &snapshot->resident_state_digest, &boundary->second,
        sizeof(boundary->second));

    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((next_active_mask & lane_mask) == 0) continue;
        ptx_thread_info *thread = lane_threads[lane];
        if (thread == NULL || thread->RT_thread_data == NULL ||
            thread->RT_thread_data->traversal_data.empty()) {
            return false;
        }
        std::map<unsigned, rtcore_replay_lane_request>::const_iterator request =
            g_rtcore_replay_lane_requests.find(thread->get_uid());
        if (request == g_rtcore_replay_lane_requests.end() ||
            !request->second.valid ||
            request->second.owner_hw_sid != owner_hw_sid ||
            request->second.warp_uid != previous_warp_uid ||
            request->second.warp_id != warp_id ||
            request->second.active_mask != previous_active_mask ||
            request->second.lane_id != lane ||
            request->second.handoff_window_base != handoff_window_base ||
            request->second.v04_live_handoff_memory == NULL) {
            return false;
        }
        const rtcore_replay_lane_request &lane_request = request->second;
        rtcore_v04_atomic_digest_bytes(
            &snapshot->request_state_digest, &lane_request,
            sizeof(lane_request));
        rtcore_v04_atomic_digest_bytes(
            &snapshot->retained_state_digest,
            &lane_request.v04_replay_committed_boundary_values,
            sizeof(lane_request.v04_replay_committed_boundary_values));
        rtcore_v04_atomic_digest_bytes(
            &snapshot->retained_state_digest,
            &lane_request.boundary_candidate,
            sizeof(lane_request.boundary_candidate));

        const unsigned long long lane_address =
            handoff_window_base +
            static_cast<unsigned long long>(lane) *
                rtcore::abi_v04::kLaneSlotBytes;
        if (lane_address < handoff_window_base) return false;
        std::array<uint8_t, rtcore::abi_v04::kLaneSlotBytes> lane_image = {};
        lane_request.v04_live_handoff_memory->read_simulator_backing(
            lane_address, lane_image.size(), lane_image.data());
        rtcore_v04_atomic_digest_bytes(
            &snapshot->live_handoff_digest, lane_image.data(),
            lane_image.size());

        Traversal_data *traversal_data =
            thread->RT_thread_data->traversal_data.back();
        if (traversal_data == NULL ||
            thread->get_global_memory() !=
                lane_request.v04_live_handoff_memory) {
            return false;
        }
        Traversal_data traversal_image = {};
        lane_request.v04_live_handoff_memory->read(
            traversal_data, sizeof(traversal_image), &traversal_image);
        rtcore_v04_atomic_digest_bytes(
            &snapshot->compatibility_state_digest, &traversal_image,
            sizeof(traversal_image));
        const size_t hit_data_count =
            thread->RT_thread_data->all_hit_data.size();
        rtcore_v04_atomic_digest_bytes(
            &snapshot->compatibility_state_digest, &hit_data_count,
            sizeof(hit_data_count));
        for (size_t index = 0; index < hit_data_count; ++index) {
            Hit_data *hit_data = thread->RT_thread_data->all_hit_data[index];
            rtcore_v04_atomic_digest_bytes(
                &snapshot->compatibility_state_digest, &hit_data,
                sizeof(hit_data));
        }
        if (lane_request.boundary_candidate.hit_data_ref != 0) {
            Hit_data candidate_image = {};
            Hit_data *candidate = reinterpret_cast<Hit_data *>(
                lane_request.boundary_candidate.hit_data_ref);
            lane_request.v04_live_handoff_memory->read(
                candidate, sizeof(candidate_image), &candidate_image);
            rtcore_v04_atomic_digest_bytes(
                &snapshot->compatibility_state_digest, &candidate_image,
                sizeof(candidate_image));
        }
    }
    snapshot->valid = true;
    return true;
}

static bool rtcore_v04_atomic_state_snapshots_equal(
    const rtcore_v04_atomic_state_snapshot &before,
    const rtcore_v04_atomic_state_snapshot &after)
{
    return before.valid && after.valid &&
           before.lane_mask == after.lane_mask &&
           before.live_handoff_digest == after.live_handoff_digest &&
           before.retained_state_digest == after.retained_state_digest &&
           before.compatibility_state_digest ==
               after.compatibility_state_digest &&
           before.request_state_digest == after.request_state_digest &&
           before.resident_state_digest == after.resident_state_digest;
}

extern "C" bool
rtcore_commit_v04_functional_shader_visible_resubmit_admission(
    const ptx_instruction *pI, unsigned owner_hw_sid,
    unsigned new_warp_uid, unsigned warp_id,
    unsigned new_static_inst_uid, unsigned next_active_mask,
    unsigned expected_previous_warp_uid,
    unsigned expected_resident_generation,
    unsigned expected_previous_active_mask,
    unsigned long long handoff_window_base,
    ptx_thread_info *const *lane_threads,
    unsigned long long service_cycle, unsigned *previous_active_mask,
    unsigned *released_lane_mask, unsigned *reactivated_lane_mask,
    unsigned *resident_occupancy_before, unsigned *resident_occupancy_after,
    const char **failure_reason)
{
    const char *failure = "accepted";
    unsigned prepared_lane_mask = 0;
    unsigned restore_prepared_lane_mask = 0;
    unsigned committed_lane_mask = 0;
    std::array<rtcore_v04_functional_lane_return_plan, 32> lane_plans = {};
    std::array<rtcore_v04_live_trace_input_restore_plan, 32>
        restore_plans = {};
    rtcore_shader_visible_resubmit_admission_plan admission_plan;
    const bool atomic_snapshot_enabled =
        rtcore_v04_atomic_state_snapshot_enabled();
    rtcore_v04_atomic_state_snapshot atomic_state_before;
    rtcore_v04_atomic_state_snapshot atomic_state_after;
    bool atomic_snapshot_captured = false;

    if (pI == NULL || lane_threads == NULL || next_active_mask == 0) {
        failure = "V04_FUNCTIONAL_RESUBMIT_TRANSACTION_CONTEXT_INVALID";
    }
    if (strcmp(failure, "accepted") == 0 && atomic_snapshot_enabled) {
        atomic_snapshot_captured = rtcore_capture_v04_atomic_state_snapshot(
            owner_hw_sid, expected_previous_warp_uid, warp_id,
            expected_previous_active_mask, next_active_mask,
            handoff_window_base, lane_threads, &atomic_state_before);
        if (!atomic_snapshot_captured) {
            failure = "V04_FUNCTIONAL_ATOMIC_PRE_SNAPSHOT_FAILED";
        }
    }
    for (unsigned lane = 0;
         lane < 32 && strcmp(failure, "accepted") == 0; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((next_active_mask & lane_mask) == 0) {
            continue;
        }
        if (lane_threads[lane] == NULL) {
            failure = "V04_FUNCTIONAL_RESUBMIT_LANE_THREAD_MISSING";
            break;
        }
        if (!rtcore_prepare_v04_functional_resubmit_lane_return(
                pI, owner_hw_sid, expected_previous_warp_uid, warp_id,
                expected_previous_active_mask, lane, handoff_window_base,
                lane_threads[lane], service_cycle, &lane_plans[lane],
                &failure)) {
            break;
        }
        prepared_lane_mask |= lane_mask;
    }

    if (strcmp(failure, "accepted") == 0 &&
        !rtcore_prepare_shader_visible_resubmit_admission(
            owner_hw_sid, new_warp_uid, warp_id, next_active_mask,
            expected_previous_warp_uid, expected_resident_generation,
            &admission_plan, &failure)) {
        // Admission failure remains side-effect free after all lane prepares.
    }
    if (strcmp(failure, "accepted") == 0 &&
        admission_plan.old_active_mask != expected_previous_active_mask) {
        failure = "V04_FUNCTIONAL_RESUBMIT_PREVIOUS_MASK_CHANGED";
    }

    for (unsigned lane = 0;
         lane < 32 && strcmp(failure, "accepted") == 0; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((next_active_mask & lane_mask) == 0) continue;
        if (!rtcore_prepare_v04_live_trace_input_compatibility_restore(
                lane_plans[lane].request, &restore_plans[lane], &failure)) {
            break;
        }
        restore_prepared_lane_mask |= lane_mask;
    }

    for (unsigned lane = 0;
         lane < 32 && strcmp(failure, "accepted") == 0; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((next_active_mask & lane_mask) == 0) continue;
        if (!rtcore_revalidate_prepared_v04_live_trace_input_restore(
                restore_plans[lane], lane_plans[lane].request, &failure) ||
            !rtcore_revalidate_v04_functional_resubmit_lane_plan(
                lane_plans[lane], &failure)) {
            break;
        }
    }

    if (strcmp(failure, "accepted") != 0) {
        const bool atomic_after_captured =
            atomic_snapshot_enabled && atomic_snapshot_captured &&
            rtcore_capture_v04_atomic_state_snapshot(
                owner_hw_sid, expected_previous_warp_uid, warp_id,
                expected_previous_active_mask, next_active_mask,
                handoff_window_base, lane_threads, &atomic_state_after);
        const bool atomic_state_unchanged =
            atomic_after_captured && rtcore_v04_atomic_state_snapshots_equal(
                                         atomic_state_before,
                                         atomic_state_after);
        const unsigned precommit_state_mutation =
            atomic_snapshot_enabled && atomic_snapshot_captured &&
                    !atomic_state_unchanged
                ? 1u
                : 0u;
        const unsigned occupancy =
            rtcore_resident_rt_warp_record_occupancy();
        if (previous_active_mask) {
            *previous_active_mask = admission_plan.valid
                                        ? admission_plan.old_active_mask
                                        : expected_previous_active_mask;
        }
        if (released_lane_mask) *released_lane_mask = 0;
        if (reactivated_lane_mask) *reactivated_lane_mask = 0;
        if (resident_occupancy_before) *resident_occupancy_before = occupancy;
        if (resident_occupancy_after) *resident_occupancy_after = occupancy;
        printf("GPGPU-Sim RTCORE_V04_FUNCTIONAL_RESUBMIT_TRANSACTION "
               "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u "
               "warp_id=%u previous_active_mask=0x%08x "
               "next_active_mask=0x%08x prepared_lane_mask=0x%08x "
               "restore_prepared_lane_mask=0x%08x "
               "committed_lane_mask=0x%08x whole_mask_committed=0 "
               "atomic_state_snapshot=%u atomic_state_unchanged=%u "
               "precommit_state_mutation=%u "
               "before_live_handoff_digest=0x%016llx "
               "after_live_handoff_digest=0x%016llx "
               "before_retained_state_digest=0x%016llx "
               "after_retained_state_digest=0x%016llx "
               "before_compatibility_state_digest=0x%016llx "
               "after_compatibility_state_digest=0x%016llx "
               "before_request_state_digest=0x%016llx "
               "after_request_state_digest=0x%016llx "
               "before_resident_state_digest=0x%016llx "
               "after_resident_state_digest=0x%016llx "
               "resubmit_membership_authority=active_mask result=%s\n",
               owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
               warp_id, expected_previous_active_mask, next_active_mask,
               prepared_lane_mask, restore_prepared_lane_mask,
               committed_lane_mask, atomic_snapshot_captured ? 1u : 0u,
               atomic_state_unchanged ? 1u : 0u,
               precommit_state_mutation,
               static_cast<unsigned long long>(
                   atomic_state_before.live_handoff_digest),
               static_cast<unsigned long long>(
                   atomic_state_after.live_handoff_digest),
               static_cast<unsigned long long>(
                   atomic_state_before.retained_state_digest),
               static_cast<unsigned long long>(
                   atomic_state_after.retained_state_digest),
               static_cast<unsigned long long>(
                   atomic_state_before.compatibility_state_digest),
               static_cast<unsigned long long>(
                   atomic_state_after.compatibility_state_digest),
               static_cast<unsigned long long>(
                   atomic_state_before.request_state_digest),
               static_cast<unsigned long long>(
                   atomic_state_after.request_state_digest),
               static_cast<unsigned long long>(
                   atomic_state_before.resident_state_digest),
               static_cast<unsigned long long>(
                   atomic_state_after.resident_state_digest),
               failure);
        fflush(stdout);
        if (failure_reason != NULL) *failure_reason = failure;
        return false;
    }

    // Every operation below is either infallible or a fail-stop invariant.
    // No transaction path can return false after the first write.
    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((next_active_mask & lane_mask) == 0) continue;
        rtcore_commit_prepared_v04_live_trace_input_restore(
            restore_plans[lane], service_cycle);
    }

    for (unsigned lane = 0; lane < 32; ++lane) {
        const unsigned lane_mask = 1u << lane;
        if ((next_active_mask & lane_mask) == 0) {
            continue;
        }
        rtcore_commit_v04_functional_resubmit_lane_return(
            pI, owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
            warp_id, handoff_window_base, lane_plans[lane]);
        committed_lane_mask |= lane_mask;
    }

    const bool admission_committed =
        rtcore_commit_prepared_shader_visible_resubmit_admission(
            admission_plan, owner_hw_sid, new_warp_uid, warp_id,
            new_static_inst_uid, next_active_mask,
            expected_previous_warp_uid, expected_resident_generation,
            service_cycle, previous_active_mask, released_lane_mask,
            reactivated_lane_mask, resident_occupancy_before,
            resident_occupancy_after);
    if (!admission_committed) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_FUNCTIONAL_RESUBMIT_COMMIT_INVARIANT "
                "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u "
                "warp_id=%u prepared_lane_mask=0x%08x "
                "committed_lane_mask=0x%08x fault=admission_commit_failed\n",
                owner_hw_sid, expected_previous_warp_uid, new_warp_uid,
                warp_id, prepared_lane_mask, committed_lane_mask);
        fflush(stderr);
        abort();
    }

    printf("GPGPU-Sim RTCORE_V04_FUNCTIONAL_RESUBMIT_TRANSACTION "
           "owner_hw_sid=%u previous_warp_uid=%u warp_uid=%u warp_id=%u "
           "previous_active_mask=0x%08x next_active_mask=0x%08x "
           "prepared_lane_mask=0x%08x "
           "restore_prepared_lane_mask=0x%08x "
           "committed_lane_mask=0x%08x "
           "whole_mask_committed=1 "
           "resubmit_membership_authority=active_mask result=accepted\n",
           owner_hw_sid, expected_previous_warp_uid, new_warp_uid, warp_id,
           expected_previous_active_mask, next_active_mask,
           prepared_lane_mask, restore_prepared_lane_mask,
           committed_lane_mask);
    fflush(stdout);
    if (failure_reason != NULL) *failure_reason = "accepted";
    return true;
}

static const char *rtcore_validate_resident_rt_warp_lane_retire(
    unsigned owner_hw_sid, unsigned warp_id, unsigned lane_id,
    unsigned thread_uid, unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned token_id,
    unsigned token_allocator_generation, unsigned window_generation,
    rtcore_resident_rt_warp_record **validated_record,
    rtcore_replay_lane_request **validated_request)
{
    if (validated_record) {
        *validated_record = NULL;
    }
    if (validated_request) {
        *validated_request = NULL;
    }

    const char *reason = "accepted";
    const rtcore_resident_rt_warp_record_key resident_key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::iterator resident =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    if (resident == g_rtcore_resident_rt_warp_records.end() ||
        !resident->second.valid) {
        reason = "MISSING_RESIDENT_RECORD";
    }

    rtcore_resident_rt_warp_record *record =
        strcmp(reason, "accepted") == 0 ? &resident->second : NULL;
    const unsigned lane_mask = lane_id < 32 ? 1u << lane_id : 0;
    if (record &&
        (lane_mask == 0 || (record->bound_lane_mask & lane_mask) == 0)) {
        reason = "RETIRE_LANE_NOT_BOUND";
    }
    if (record && strcmp(reason, "accepted") == 0 &&
        (record->retired_lane_mask & lane_mask) != 0) {
        reason = "DUPLICATE_RESIDENT_LANE_RETIRE";
    }
    if (record && strcmp(reason, "accepted") == 0 &&
        !rtcore_resident_lane_identity_matches(
            record->lane_identity[lane_id], thread_uid, context_ptr,
            handoff_window_base, token_id, token_allocator_generation,
            window_generation)) {
        reason = "RETIRE_STALE_OR_OWNER_IDENTITY_MISMATCH";
    }

    rtcore_replay_lane_request *request = NULL;
    if (record && strcmp(reason, "accepted") == 0) {
        std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
            g_rtcore_replay_lane_requests.find(thread_uid);
        if (request_it == g_rtcore_replay_lane_requests.end() ||
            !request_it->second.valid) {
            reason = "RETIRE_MISSING_PINNED_REQUEST_STATE";
        } else {
            request = &request_it->second;
            const bool waiting_shader =
                request->state == RTCORE_REPLAY_WAITING_SHADER &&
                request->continuation_boundary_pending;
            const bool final_wait =
                request->state == RTCORE_REPLAY_FINAL_WAIT_RETIRE;
            const bool already_released =
                request->state == RTCORE_REPLAY_COMPLETED;
            if (!waiting_shader && !final_wait && !already_released) {
                reason = "RETIRE_REQUEST_STATE_NOT_QUIESCENT";
            }
        }
    }
    if (strcmp(reason, "accepted") == 0) {
        if (validated_record) {
            *validated_record = record;
        }
        if (validated_request) {
            *validated_request = request;
        }
    }
    return reason;
}

extern "C" bool rtcore_preflight_retire_resident_rt_warp_lane(
    unsigned owner_hw_sid, unsigned warp_id, unsigned lane_id,
    unsigned thread_uid, unsigned long long context_ptr,
    unsigned long long handoff_window_base, unsigned token_id,
    unsigned token_allocator_generation, unsigned window_generation,
    const char **failure_reason)
{
    if (!rtcore_continuation_model_enabled()) {
        if (failure_reason) {
            *failure_reason = "continuation_model_disabled";
        }
        return true;
    }
    const char *reason = rtcore_validate_resident_rt_warp_lane_retire(
        owner_hw_sid, warp_id, lane_id, thread_uid, context_ptr,
        handoff_window_base, token_id, token_allocator_generation,
        window_generation, NULL, NULL);
    if (failure_reason) {
        *failure_reason = reason;
    }
    return strcmp(reason, "accepted") == 0;
}

extern "C" bool rtcore_begin_retire_resident_rt_warp_transaction(
    unsigned owner_hw_sid, unsigned retire_warp_uid, unsigned warp_id,
    unsigned retire_active_mask, unsigned *resident_generation,
    unsigned *bound_lane_mask, unsigned *pending_release_mask,
    unsigned *already_released_lane_mask, unsigned *resident_occupancy,
    const char **failure_reason)
{
    if (!rtcore_retire_lifecycle_busy_for_owner(owner_hw_sid)) {
        if (failure_reason) {
            *failure_reason = "RETIRE_LIFECYCLE_FRONTEND_NOT_RESERVED";
        }
        return false;
    }

    if (!rtcore_continuation_model_enabled()) {
        if (resident_generation) {
            *resident_generation = 0;
        }
        if (bound_lane_mask) {
            *bound_lane_mask = retire_active_mask;
        }
        if (pending_release_mask) {
            *pending_release_mask = retire_active_mask;
        }
        if (already_released_lane_mask) {
            *already_released_lane_mask = 0;
        }
        if (resident_occupancy) {
            *resident_occupancy = 0;
        }
        if (failure_reason) {
            *failure_reason = "continuation_model_disabled";
        }
        return true;
    }

    const unsigned occupancy =
        rtcore_resident_rt_warp_record_occupancy();
    const rtcore_resident_rt_warp_record_key resident_key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::iterator resident =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    const char *reason = "accepted";
    if (resident == g_rtcore_resident_rt_warp_records.end() ||
        !resident->second.valid) {
        reason = "MISSING_RESIDENT_RECORD";
    }

    rtcore_resident_rt_warp_record *record =
        strcmp(reason, "accepted") == 0 ? &resident->second : NULL;
    if (record && (retire_active_mask == 0 ||
                   retire_active_mask != record->bound_lane_mask ||
                   record->retired_lane_mask != 0)) {
        reason = "RETIRE_WHOLE_BOUND_MASK_REQUIRED";
    }
    if (record && strcmp(reason, "accepted") == 0) {
        for (unsigned lane = 0; lane < 32; ++lane) {
            const unsigned lane_mask = 1u << lane;
            if ((retire_active_mask & lane_mask) == 0) {
                continue;
            }
            const rtcore_resident_rt_warp_lane_identity &identity =
                record->lane_identity[lane];
            const char *lane_reason =
                rtcore_validate_resident_rt_warp_lane_retire(
                    owner_hw_sid, warp_id, lane, identity.thread_uid,
                    identity.context_ptr, identity.handoff_window_base,
                    identity.token_id, identity.token_allocator_generation,
                    identity.window_generation, NULL, NULL);
            if (strcmp(lane_reason, "accepted") != 0) {
                reason = lane_reason;
                break;
            }
        }
    }

    unsigned already_released_mask = 0;
    if (record && strcmp(reason, "accepted") == 0) {
        for (unsigned lane = 0; lane < 32; ++lane) {
            const unsigned lane_mask = 1u << lane;
            if ((retire_active_mask & lane_mask) == 0) {
                continue;
            }
            const unsigned thread_uid = record->lane_identity[lane].thread_uid;
            std::map<unsigned, rtcore_replay_lane_request>::const_iterator request =
                g_rtcore_replay_lane_requests.find(thread_uid);
            if (request != g_rtcore_replay_lane_requests.end() &&
                request->second.valid &&
                request->second.state == RTCORE_REPLAY_COMPLETED) {
                already_released_mask |= lane_mask;
            }
        }
        record->retired_lane_mask = already_released_mask;
    }
    const unsigned pending_mask =
        record ? record->bound_lane_mask & ~already_released_mask : 0;
    if (strcmp(reason, "accepted") == 0) {
        printf("GPGPU-Sim RTCORE_RETIRE_TRANSACTION_BEGIN "
               "owner_hw_sid=%u retire_warp_uid=%u warp_id=%u "
               "retire_active_mask=0x%08x resident_generation=%u "
               "bound_lane_mask=0x%08x pending_release_mask=0x%08x "
               "already_released_lane_mask=0x%08x resident_occupancy=%u\n",
               owner_hw_sid, retire_warp_uid, warp_id, retire_active_mask,
               record->resident_generation, record->bound_lane_mask,
               pending_mask, already_released_mask, occupancy);
        fflush(stdout);
    }
    if (resident_generation) {
        *resident_generation = record ? record->resident_generation : 0;
    }
    if (bound_lane_mask) {
        *bound_lane_mask = record ? record->bound_lane_mask : 0;
    }
    if (pending_release_mask) {
        *pending_release_mask = pending_mask;
    }
    if (already_released_lane_mask) {
        *already_released_lane_mask = already_released_mask;
    }
    if (resident_occupancy) {
        *resident_occupancy = occupancy;
    }
    if (failure_reason) {
        *failure_reason = reason;
    }
    return strcmp(reason, "accepted") == 0;
}

extern "C" bool rtcore_drain_retire_resident_rt_warp_lane(
    unsigned owner_hw_sid, unsigned retire_warp_uid, unsigned warp_id,
    unsigned expected_resident_generation, unsigned lane_id,
    unsigned long long service_cycle, unsigned *retired_lane_mask,
    unsigned *remaining_lane_mask, const char **failure_reason)
{
    if (!rtcore_continuation_model_enabled()) {
        if (retired_lane_mask) {
            *retired_lane_mask = lane_id < 32 ? 1u << lane_id : 0;
        }
        if (remaining_lane_mask) {
            *remaining_lane_mask = 0;
        }
        if (failure_reason) {
            *failure_reason = "continuation_model_disabled";
        }
        return true;
    }

    const rtcore_resident_rt_warp_record_key resident_key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::iterator resident =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    const char *reason = "accepted";
    if (resident == g_rtcore_resident_rt_warp_records.end() ||
        !resident->second.valid) {
        reason = "MISSING_RESIDENT_RECORD";
    }

    rtcore_resident_rt_warp_record *record =
        strcmp(reason, "accepted") == 0 ? &resident->second : NULL;
    const unsigned lane_mask = lane_id < 32 ? 1u << lane_id : 0;
    if (record && (record->resident_generation !=
                       expected_resident_generation ||
                   lane_mask == 0 ||
                   (record->bound_lane_mask & lane_mask) == 0)) {
        reason = "RETIRE_DRAIN_RESIDENT_IDENTITY_MISMATCH";
    }
    if (record && strcmp(reason, "accepted") == 0 &&
        (record->retired_lane_mask & lane_mask) != 0) {
        reason = "DUPLICATE_RESIDENT_LANE_RETIRE";
    }

    rtcore_replay_lane_request *request = NULL;
    if (record && strcmp(reason, "accepted") == 0) {
        const rtcore_resident_rt_warp_lane_identity &identity =
            record->lane_identity[lane_id];
        std::map<unsigned, rtcore_replay_lane_request>::iterator request_it =
            g_rtcore_replay_lane_requests.find(identity.thread_uid);
        if (request_it == g_rtcore_replay_lane_requests.end() ||
            !request_it->second.valid) {
            reason = "RETIRE_MISSING_PINNED_REQUEST_STATE";
        } else {
            request = &request_it->second;
            const bool waiting_shader =
                request->state == RTCORE_REPLAY_WAITING_SHADER &&
                request->continuation_boundary_pending;
            const bool final_wait =
                request->state == RTCORE_REPLAY_FINAL_WAIT_RETIRE;
            const bool already_released =
                request->state == RTCORE_REPLAY_COMPLETED;
            if (!waiting_shader && !final_wait && !already_released) {
                reason = "RETIRE_REQUEST_STATE_NOT_QUIESCENT";
            }
        }
    }

    const char *request_state_action = "none";
    if (record && request && strcmp(reason, "accepted") == 0) {
        const bool consumed_entry =
            rtcore_replay_lane_request_state_capacity_consumes_entry(*request);
        if (request->state != RTCORE_REPLAY_COMPLETED) {
            request->continuation_boundary_pending = false;
            request->state = RTCORE_REPLAY_COMPLETED;
            rtcore_record_replay_request_state_write();
            rtcore_refresh_replay_lane_request_ready_bits(request);
            if (consumed_entry) {
                rtcore_record_replay_lane_request_state_capacity_release(
                    *request, service_cycle);
            }
            request_state_action = "release_on_staged_retire";
        } else {
            request_state_action = "already_released_by_mask_shrink";
        }

        record->retired_lane_mask |= lane_mask;
    }

    const unsigned retired_mask = record ? record->retired_lane_mask : 0;
    const unsigned remaining_mask =
        record ? record->bound_lane_mask & ~record->retired_lane_mask : 0;
    if (strcmp(reason, "accepted") == 0) {
        printf("GPGPU-Sim RTCORE_RESIDENT_RT_WARP_RETIRE_DRAIN "
               "owner_hw_sid=%u retire_warp_uid=%u warp_id=%u lane_id=%u "
               "resident_generation=%u "
               "retired_lane_mask=0x%08x request_state_action=%s "
               "remaining_lane_mask=0x%08x resident_record_live=1 "
               "service_cycle=%llu\n",
               owner_hw_sid, retire_warp_uid, warp_id, lane_id,
               expected_resident_generation, retired_mask,
               request_state_action, remaining_mask, service_cycle);
        fflush(stdout);
    }
    if (retired_lane_mask) {
        *retired_lane_mask = retired_mask;
    }
    if (remaining_lane_mask) {
        *remaining_lane_mask = remaining_mask;
    }
    if (failure_reason) {
        *failure_reason = reason;
    }
    return strcmp(reason, "accepted") == 0;
}

extern "C" bool rtcore_commit_retire_resident_rt_warp_lifecycle(
    unsigned owner_hw_sid, unsigned retire_warp_uid, unsigned warp_id,
    unsigned expected_resident_generation, unsigned retire_active_mask,
    bool external_resources_released, unsigned long long service_cycle,
    unsigned *resident_occupancy_before,
    unsigned *resident_occupancy_after, const char **failure_reason)
{
    if (!rtcore_continuation_model_enabled()) {
        const bool busy_released =
            g_rtcore_retire_lifecycle_busy_owners.erase(owner_hw_sid) == 1;
        if (resident_occupancy_before) {
            *resident_occupancy_before = 0;
        }
        if (resident_occupancy_after) {
            *resident_occupancy_after = 0;
        }
        if (failure_reason) {
            *failure_reason = busy_released
                                  ? "continuation_model_disabled"
                                  : "RETIRE_LIFECYCLE_BUSY_MISSING";
        }
        return busy_released;
    }

    const unsigned occupancy_before =
        rtcore_resident_rt_warp_record_occupancy();
    const rtcore_resident_rt_warp_record_key resident_key =
        rtcore_make_resident_rt_warp_record_key(owner_hw_sid, warp_id);
    std::map<rtcore_resident_rt_warp_record_key,
             rtcore_resident_rt_warp_record>::iterator resident =
        g_rtcore_resident_rt_warp_records.find(resident_key);
    const char *reason = "accepted";
    if (resident == g_rtcore_resident_rt_warp_records.end() ||
        !resident->second.valid) {
        reason = "MISSING_RESIDENT_RECORD";
    }

    rtcore_resident_rt_warp_record *record =
        strcmp(reason, "accepted") == 0 ? &resident->second : NULL;
    if (record &&
        (record->resident_generation != expected_resident_generation ||
         retire_active_mask != record->bound_lane_mask ||
         (record->retired_lane_mask & record->bound_lane_mask) !=
             record->bound_lane_mask)) {
        reason = "RETIRE_LIFECYCLE_DRAIN_INCOMPLETE";
    }
    if (record && strcmp(reason, "accepted") == 0 &&
        !external_resources_released) {
        reason = "RETIRE_EXTERNAL_RESOURCES_NOT_RELEASED";
    }
    if (record && strcmp(reason, "accepted") == 0 &&
        !rtcore_retire_lifecycle_busy_for_owner(owner_hw_sid)) {
        reason = "RETIRE_LIFECYCLE_BUSY_MISSING";
    }

    rtcore::v04::request_owner::release_warp_plan_v0
        v04_request_owner_release_plan = {};
    rtcore::v04::private_shared::release_warp_plan_v0
        v04_private_shared_release_plan = {};
    if (record && strcmp(reason, "accepted") == 0 &&
        rtcore_v04_request_owner_binding_enabled()) {
        if (!record->v04_request_owner_binding_valid ||
            record->v04_resident_warp_slot >=
                rtcore::v04::request_owner::kResidentWarpCapacity) {
            reason = "REQUEST_OWNER_RETIRE_BINDING_MISSING";
        } else {
            const rtcore::v04::request_owner::status_kind owner_status =
                rtcore::v04::request_owner::prepare_release_warp(
                    rtcore_v04_request_owner_allocator_for(owner_hw_sid),
                    static_cast<uint8_t>(record->v04_resident_warp_slot),
                    owner_hw_sid, record->current_warp_uid, warp_id,
                    &v04_request_owner_release_plan);
            if (owner_status !=
                rtcore::v04::request_owner::kStatusOk) {
                reason = rtcore::v04::request_owner::status_name(
                    owner_status);
            }
        }
    }
    if (record && strcmp(reason, "accepted") == 0 &&
        rtcore_v04_private_frontier_live_init_enabled()) {
        if (!record->v04_private_frontier_live_init_valid ||
            !record->v04_private_frontier_init_committed) {
            reason = "PRIVATE_FRONTIER_RETIRE_INIT_INCOMPLETE";
        } else {
            const rtcore::v04::private_shared::status_kind private_status =
                rtcore::v04::private_shared::prepare_release_warp(
                    rtcore_v04_private_shared_backing_for(owner_hw_sid),
                    static_cast<uint8_t>(record->v04_resident_warp_slot),
                    record->current_warp_uid, warp_id,
                    &v04_private_shared_release_plan);
            if (private_status !=
                rtcore::v04::private_shared::kStatusOk) {
                reason =
                    rtcore::v04::private_shared::status_name(private_status);
            }
        }
    }

    unsigned resident_generation =
        record ? record->resident_generation : 0;
    unsigned bound_mask = record ? record->bound_lane_mask : 0;
    unsigned submit_warp_uid = record ? record->current_warp_uid : 0;
    if (record && strcmp(reason, "accepted") == 0) {
        if (rtcore_v04_request_owner_binding_enabled()) {
            const rtcore::v04::request_owner::status_kind owner_status =
                rtcore::v04::request_owner::commit_release_warp(
                    &rtcore_v04_request_owner_allocator_for(owner_hw_sid),
                    v04_request_owner_release_plan);
            if (owner_status !=
                rtcore::v04::request_owner::kStatusOk) {
                fprintf(stderr,
                        "GPGPU-Sim "
                        "RTCORE_V04_REQUEST_OWNER_COMMIT_INVARIANT "
                        "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                        "fault=%s\n",
                        owner_hw_sid, record->current_warp_uid, warp_id,
                        rtcore::v04::request_owner::status_name(
                            owner_status));
                fflush(stderr);
                abort();
            }
            printf("GPGPU-Sim RTCORE_V04_REQUEST_OWNER_RELEASE "
                   "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                   "resident_warp_slot=%u released_lane_mask=0x%08x "
                   "allocator_scope=per_sm result=accepted\n",
                   owner_hw_sid, record->current_warp_uid, warp_id,
                   v04_request_owner_release_plan.resident_warp_slot,
                   v04_request_owner_release_plan.identity.active_mask);
            fflush(stdout);
            for (unsigned lane = 0; lane < 32; ++lane) {
                if ((record->v04_request_owner_active_mask &
                     (1u << lane)) == 0) {
                    continue;
                }
                const unsigned thread_uid =
                    record->lane_identity[lane].thread_uid;
                std::map<unsigned, rtcore_replay_lane_request>::iterator it =
                    g_rtcore_replay_lane_requests.find(thread_uid);
                if (it != g_rtcore_replay_lane_requests.end()) {
                    it->second.v04_request_owner_binding_valid = false;
                }
                record->lane_identity[lane]
                    .v04_request_owner_binding_valid = false;
            }
        }
        if (rtcore_v04_private_frontier_live_init_enabled()) {
            const rtcore::v04::private_shared::status_kind private_status =
                rtcore::v04::private_shared::commit_release_warp(
                    &rtcore_v04_private_shared_backing_for(owner_hw_sid),
                    v04_private_shared_release_plan);
            if (private_status !=
                rtcore::v04::private_shared::kStatusOk) {
                fprintf(stderr,
                        "GPGPU-Sim "
                        "RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                        "owner_hw_sid=%u warp_uid=%u warp_id=%u fault=%s\n",
                        owner_hw_sid, record->current_warp_uid, warp_id,
                        rtcore::v04::private_shared::status_name(
                            private_status));
                fflush(stderr);
                abort();
            }
        }
        rtcore_replay_warp_completion_entry_key current_key = {};
        current_key.owner_hw_sid = record->owner_hw_sid;
        current_key.warp_uid = record->current_warp_uid;
        current_key.warp_id = record->warp_id;
        current_key.active_mask = record->active_mask;
        g_rtcore_continuation_warp_boundary_states.erase(current_key);
        g_rtcore_resident_warp_continuation_states.erase(current_key);
        g_rtcore_resident_rt_warp_records.erase(resident_key);
        const size_t busy_released =
            g_rtcore_retire_lifecycle_busy_owners.erase(owner_hw_sid);
        if (busy_released != 1) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_RETIRE_RESIDENT_COMMIT_INVARIANT "
                    "reason=RETIRE_LIFECYCLE_BUSY_MISSING owner_hw_sid=%u "
                    "retire_warp_uid=%u warp_id=%u\n",
                    owner_hw_sid, retire_warp_uid, warp_id);
            fflush(stderr);
            abort();
        }
    }

    const unsigned occupancy_after =
        rtcore_resident_rt_warp_record_occupancy();
    if (strcmp(reason, "accepted") == 0) {
        printf("GPGPU-Sim RTCORE_RESIDENT_RT_WARP_RETIRE_COMMIT "
               "owner_hw_sid=%u retire_warp_uid=%u submit_warp_uid=%u "
               "warp_id=%u resident_generation=%u "
               "bound_lane_mask=0x%08x external_resources_released=1 "
               "resident_record_released=1 resident_occupancy_before=%u "
               "resident_occupancy_after=%u service_cycle=%llu\n",
               owner_hw_sid, retire_warp_uid, submit_warp_uid, warp_id,
               resident_generation, bound_mask, occupancy_before,
               occupancy_after, service_cycle);
        fflush(stdout);
    }
    if (resident_occupancy_before) {
        *resident_occupancy_before = occupancy_before;
    }
    if (resident_occupancy_after) {
        *resident_occupancy_after = occupancy_after;
    }
    if (failure_reason) {
        *failure_reason = reason;
    }
    return strcmp(reason, "accepted") == 0;
}

static bool rtcore_service_resident_warp_continuation_for_owner(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    if (!rtcore_continuation_model_enabled()) {
        return false;
    }

    for (std::map<rtcore_replay_warp_completion_entry_key,
                  rtcore_resident_warp_continuation_state>::iterator it =
             g_rtcore_resident_warp_continuation_states.begin();
         it != g_rtcore_resident_warp_continuation_states.end(); ++it) {
        rtcore_resident_warp_continuation_state &state = it->second;
        if (!state.valid || state.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        if (service_cycle < state.ready_cycle) {
            g_rtcore_continuation_stats.rtcore_continuation_wait_cycles++;
            continue;
        }

        const unsigned reactivated_lane_count =
            rtcore_reactivate_resident_warp_continuation_lanes(
                state, service_cycle);
        if (state.continuation_depth >
            g_rtcore_continuation_stats.rtcore_continuation_max_depth) {
            g_rtcore_continuation_stats.rtcore_continuation_max_depth =
                state.continuation_depth;
        }

        if (reactivated_lane_count != 0) {
            g_rtcore_continuation_stats.rtcore_modeled_resubmit_count++;
            g_rtcore_continuation_stats.rtcore_modeled_resubmit_lane_count +=
                reactivated_lane_count;
            printf("GPGPU-Sim RTCORE_CONTINUATION_MODELED_RESUBMIT "
                   "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                   "active_mask=0x%08x resume_required_mask=0x%08x "
                   "shader_required_mask=0x%08x continuation_depth=%u "
                   "reactivated_lane_count=%u service_cycle=%llu\n",
                   state.owner_hw_sid, state.warp_uid, state.warp_id,
                   state.active_mask, state.resume_required_mask,
                   state.shader_required_mask, state.continuation_depth,
                   reactivated_lane_count, service_cycle);
        } else {
            printf("GPGPU-Sim RTCORE_CONTINUATION_RECONCILE_ONLY "
                   "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                   "active_mask=0x%08x next_active_mask=0x%08x "
                   "shader_required_mask=0x%08x continuation_depth=%u "
                   "service_cycle=%llu\n",
                   state.owner_hw_sid, state.warp_uid, state.warp_id,
                   state.active_mask, state.resume_required_mask,
                   state.shader_required_mask, state.continuation_depth,
                   service_cycle);
        }
        fflush(stdout);
        g_rtcore_resident_warp_continuation_states.erase(it);
        return true;
    }
    return false;
}

static bool rtcore_service_v04_private_frontier_live_init(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *scheduler_ready_progressed)
{
    namespace private_shared = rtcore::v04::private_shared;
    if (scheduler_ready_progressed != NULL) {
        *scheduler_ready_progressed = false;
    }
    if (!rtcore_v04_private_frontier_live_init_enabled()) {
        return false;
    }

    private_shared::backing_state_v0 &backing =
        rtcore_v04_private_shared_backing_for(owner_hw_sid);
    const unsigned acknowledged =
        private_shared::service_write_acks(
            &backing, service_cycle, private_shared::kResponseFillWidth);
    if (backing.fault_status != private_shared::kStatusOk) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                "owner_hw_sid=%u service_cycle=%llu fault=%s\n",
                owner_hw_sid, service_cycle,
                private_shared::status_name(
                    static_cast<private_shared::status_kind>(
                        backing.fault_status)));
        fflush(stderr);
        abort();
    }
    bool ready_progressed = false;
    private_shared::ready_commit_v0 ready_commit = {};
    while (private_shared::pop_ready_commit(&backing, &ready_commit)) {
        const rtcore_resident_rt_warp_record_key record_key =
            rtcore_make_resident_rt_warp_record_key(
                ready_commit.owner_hw_sid, ready_commit.warp_id);
        std::map<rtcore_resident_rt_warp_record_key,
                 rtcore_resident_rt_warp_record>::iterator record_it =
            g_rtcore_resident_rt_warp_records.find(record_key);
        if (record_it == g_rtcore_resident_rt_warp_records.end() ||
            !record_it->second.valid ||
            !record_it->second.v04_private_frontier_live_init_valid ||
            record_it->second.v04_private_frontier_init_committed ||
            record_it->second.current_warp_uid != ready_commit.warp_uid ||
            record_it->second.active_mask != ready_commit.active_mask ||
            record_it->second.v04_resident_warp_slot !=
                ready_commit.resident_warp_slot) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                    "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                    "active_mask=0x%08x fault=ready_commit_owner_mismatch\n",
                    ready_commit.owner_hw_sid, ready_commit.warp_uid,
                    ready_commit.warp_id, ready_commit.active_mask);
            fflush(stderr);
            abort();
        }

        rtcore_resident_rt_warp_record &record = record_it->second;
        for (unsigned lane = 0; lane < 32; ++lane) {
            if ((ready_commit.active_mask & (1u << lane)) == 0) continue;
            const unsigned thread_uid =
                record.lane_identity[lane].thread_uid;
            std::map<unsigned, rtcore_replay_lane_request>::iterator request =
                g_rtcore_replay_lane_requests.find(thread_uid);
            if (request == g_rtcore_replay_lane_requests.end() ||
                !request->second.valid ||
                !request->second.v04_private_frontier_init_pending ||
                !rtcore::v04::private_frontier::owners_equal(
                    request->second.v04_private_frontier_owner,
                    backing
                        .resident_warps[ready_commit.resident_warp_slot]
                        .lanes[lane]
                        .owner)) {
                fprintf(stderr,
                        "GPGPU-Sim "
                        "RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                        "owner_hw_sid=%u warp_uid=%u warp_id=%u lane_id=%u "
                        "fault=lane_ready_commit_mismatch\n",
                        ready_commit.owner_hw_sid, ready_commit.warp_uid,
                        ready_commit.warp_id, lane);
                fflush(stderr);
                abort();
            }
        }

        for (unsigned lane = 0; lane < 32; ++lane) {
            if ((ready_commit.active_mask & (1u << lane)) == 0) continue;
            const unsigned thread_uid =
                record.lane_identity[lane].thread_uid;
            g_rtcore_replay_lane_requests[thread_uid]
                .v04_private_frontier_init_pending = false;
        }
        record.v04_private_frontier_init_committed = true;
        for (unsigned lane = 0; lane < 32; ++lane) {
            if ((ready_commit.active_mask & (1u << lane)) == 0) continue;
            const unsigned thread_uid =
                record.lane_identity[lane].thread_uid;
            rtcore_replay_lane_request &request =
                g_rtcore_replay_lane_requests[thread_uid];
            rtcore_refresh_replay_lane_request_ready_bits(&request);
            if (!rtcore_route_admitted_replay_request(thread_uid,
                                                       service_cycle)) {
                fprintf(stderr,
                        "GPGPU-Sim "
                        "RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                        "owner_hw_sid=%u warp_uid=%u warp_id=%u lane_id=%u "
                        "fault=whole_mask_route_failed\n",
                        ready_commit.owner_hw_sid, ready_commit.warp_uid,
                        ready_commit.warp_id, lane);
                fflush(stderr);
                abort();
            }
        }
        rtcore_refresh_resident_rt_warp_admitted_lane_mask(&record);
        if (record.admitted_lane_mask != ready_commit.active_mask) {
            fprintf(stderr,
                    "GPGPU-Sim RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                    "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                    "active_mask=0x%08x scheduler_visible_mask=0x%08x "
                    "fault=partial_ready_commit\n",
                    ready_commit.owner_hw_sid, ready_commit.warp_uid,
                    ready_commit.warp_id, ready_commit.active_mask,
                    record.admitted_lane_mask);
            fflush(stderr);
            abort();
        }
        const private_shared::resident_warp_state_v0 &private_warp =
            backing.resident_warps[ready_commit.resident_warp_slot];
        const unsigned planned_chunks =
            rtcore_continuation_count_lanes(ready_commit.active_mask) * 2u;
        if (private_warp.enqueued_chunk_count != planned_chunks ||
            private_warp.accepted_chunk_count != planned_chunks ||
            private_warp.acknowledged_chunk_count != planned_chunks) {
            fprintf(stderr,
                    "GPGPU-Sim "
                    "RTCORE_V04_PRIVATE_FRONTIER_INIT_INVARIANT "
                    "owner_hw_sid=%u warp_uid=%u warp_id=%u "
                    "planned_chunks=%u enqueued_chunks=%u "
                    "accepted_chunks=%u acknowledged_chunks=%u "
                    "fault=transport_accounting_mismatch\n",
                    ready_commit.owner_hw_sid, ready_commit.warp_uid,
                    ready_commit.warp_id, planned_chunks,
                    private_warp.enqueued_chunk_count,
                    private_warp.accepted_chunk_count,
                    private_warp.acknowledged_chunk_count);
            fflush(stderr);
            abort();
        }
        printf("GPGPU-Sim RTCORE_V04_PRIVATE_FRONTIER_INIT_COMMIT "
               "owner_hw_sid=%u warp_uid=%u warp_id=%u "
               "resident_warp_slot=%u active_mask=0x%08x "
               "scheduler_visible_mask=0x%08x whole_mask_committed=1 "
               "planned_chunks=%u enqueued_chunks=%u accepted_chunks=%u "
               "acknowledged_chunks=%u shared_queue_capacity=%u "
               "shared_queue_max_depth=%u outstanding_capacity=%u "
               "outstanding_max_depth=%u enqueue_width=%u ack_width=%u "
               "shared_queue_capacity_blocked=%llu "
               "outstanding_capacity_blocked=%llu service_cycle=%llu\n",
               ready_commit.owner_hw_sid, ready_commit.warp_uid,
               ready_commit.warp_id, ready_commit.resident_warp_slot,
               ready_commit.active_mask, record.admitted_lane_mask,
               planned_chunks, private_warp.enqueued_chunk_count,
               private_warp.accepted_chunk_count,
               private_warp.acknowledged_chunk_count,
               private_shared::kSharedQueueCapacity,
               backing.max_shared_queue_depth,
               private_shared::kOutstandingCapacity,
               backing.max_outstanding_depth,
               private_shared::kPrivateWriteEnqueueWidth,
               private_shared::kResponseFillWidth,
               static_cast<unsigned long long>(
                   backing.shared_queue_capacity_blocked_count),
               static_cast<unsigned long long>(
                   backing.outstanding_capacity_blocked_count),
               service_cycle);
        fflush(stdout);
        ready_progressed = true;
    }

    const unsigned enqueued =
        private_shared::service_init_enqueue(
            &backing, service_cycle,
            private_shared::kPrivateWriteEnqueueWidth);
    if (scheduler_ready_progressed != NULL) {
        *scheduler_ready_progressed = ready_progressed;
    }
    return acknowledged != 0 || enqueued != 0 || ready_progressed;
}

static rtcore_replay_service_tick_result
rtcore_service_replay_tick_for_owner(unsigned owner_hw_sid,
                                     unsigned long long service_cycle = 0)
{
    rtcore_replay_service_tick_result result = {};
    rtcore_replay_service_cycle_identity_snapshot memory_identity = {};
    rtcore_replay_service_cycle_identity_snapshot unit_identity = {};
    rtcore_replay_service_cycle_identity_snapshot ready_identity = {};
    rtcore_replay_service_cycle_identity_snapshot scoreboard_handoff_identity = {};
    rtcore_replay_service_cycle_identity_snapshot continuation_identity = {};
    const bool owner_admission_progressed =
        rtcore_retry_v04_whole_mask_request_owner_admission(
            owner_hw_sid, service_cycle);
    bool private_ready_progressed = false;
    const bool private_shared_progressed =
        rtcore_service_v04_private_frontier_live_init(
            owner_hw_sid, service_cycle, &private_ready_progressed);
    const bool lane_state_admission_progressed =
        rtcore_try_drain_replay_lane_request_state_capacity_pending_admissions(
            owner_hw_sid, service_cycle);
    const bool memory_issue_progressed =
        rtcore_service_ready_memory_replay_requests_for_owner(
            owner_hw_sid, rtcore_replay_memory_issue_budget_config(),
            &memory_identity, service_cycle);
    const bool memory_wake_progressed =
        rtcore_service_memory_response_wake_requests_for_owner(
            owner_hw_sid, rtcore_replay_memory_wake_budget_config(),
            &memory_identity, service_cycle);
    result.memory_progressed =
        private_shared_progressed || memory_issue_progressed ||
        memory_wake_progressed;
    const bool unit_progressed =
        rtcore_service_request_state_unit_wake_requests_for_owner(
            owner_hw_sid, rtcore_replay_unit_wake_budget_config(),
            &unit_identity, service_cycle);
    const bool ready_issue_progressed =
        rtcore_service_replay_non_completion_ready_requests_for_owner(
            owner_hw_sid, rtcore_replay_issue_budget_config(), &ready_identity,
            service_cycle);
    const bool scoreboard_handoff_progressed =
        rtcore_service_replay_scoreboard_handoff_requests_for_owner(
            owner_hw_sid, rtcore_replay_issue_budget_config(),
            &scoreboard_handoff_identity, service_cycle);
    const bool continuation_progressed =
        rtcore_service_resident_warp_continuation_for_owner(owner_hw_sid,
                                                            service_cycle);
    if (continuation_progressed) {
        continuation_identity.valid = true;
        continuation_identity.ready_progressed = true;
        continuation_identity.owner_hw_sid = owner_hw_sid;
    }
    result.unit_wake_progressed = unit_progressed;
    result.ready_issue_progressed = ready_issue_progressed;
    result.scoreboard_handoff_progressed = scoreboard_handoff_progressed;
    result.ready_progressed =
        owner_admission_progressed || private_ready_progressed ||
        lane_state_admission_progressed ||
        unit_progressed ||
        ready_issue_progressed || scoreboard_handoff_progressed ||
        continuation_progressed;
    result.progressed = result.memory_progressed || result.ready_progressed;
    if (scoreboard_handoff_progressed) {
        result.last_progress_identity = scoreboard_handoff_identity;
    } else if (continuation_progressed) {
        result.last_progress_identity = continuation_identity;
    } else if (ready_issue_progressed) {
        result.last_progress_identity = ready_identity;
    } else if (unit_progressed) {
        result.last_progress_identity = unit_identity;
    } else if (result.memory_progressed) {
        result.last_progress_identity = memory_identity;
    }
    return result;
}

static rtcore_replay_service_tick_result
rtcore_maybe_service_replay_tick(unsigned owner_hw_sid,
                                 unsigned long long service_cycle = 0)
{
    rtcore_replay_service_tick_result result = {};
    if (!rtcore_replay_service_tick_enabled()) {
        return result;
    }
    return rtcore_service_replay_tick_for_owner(owner_hw_sid, service_cycle);
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
        g_rtcore_replay_service_tick_stats
            .service_stage_memory_wake_progressed_count++;
    }
    if (result.ready_progressed) {
        g_rtcore_replay_service_tick_stats.ready_ticks_progressed++;
    }
    if (result.unit_wake_progressed) {
        g_rtcore_replay_service_tick_stats
            .service_stage_unit_wake_progressed_count++;
    }
    if (result.ready_issue_progressed) {
        g_rtcore_replay_service_tick_stats
            .service_stage_ready_issue_progressed_count++;
    }
    if (result.scoreboard_handoff_progressed) {
        g_rtcore_replay_service_tick_stats.scoreboard_handoff_progressed_count++;
    }
    if (result.memory_progressed && result.unit_wake_progressed) {
        g_rtcore_replay_service_tick_stats
            .service_stage_memory_and_unit_progressed_count++;
    }
    if (result.memory_progressed && result.ready_issue_progressed) {
        g_rtcore_replay_service_tick_stats
            .service_stage_memory_and_ready_issue_progressed_count++;
    }
    if (result.unit_wake_progressed && result.ready_issue_progressed) {
        g_rtcore_replay_service_tick_stats
            .service_stage_unit_and_ready_issue_progressed_count++;
    }
    if (result.memory_progressed && result.unit_wake_progressed &&
        result.ready_issue_progressed) {
        g_rtcore_replay_service_tick_stats
            .service_stage_all_progressed_count++;
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
           g_rtcore_replay_unit_arbitration_stats
               .warp_completion_ingress_issued;
}

static unsigned rtcore_replay_unit_arbitration_total_budget_exhausted()
{
    return g_rtcore_replay_unit_arbitration_stats.node_unit_budget_exhausted +
           g_rtcore_replay_unit_arbitration_stats
               .primitive_unit_budget_exhausted +
           g_rtcore_replay_unit_arbitration_stats.stack_unit_budget_exhausted +
           g_rtcore_replay_unit_arbitration_stats
               .warp_completion_ingress_budget_exhausted;
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
           "warp_completion_ingress_attempts=%u "
           "warp_completion_ingress_issued=%u "
           "warp_completion_ingress_budget_exhausted=%u\n",
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
               .warp_completion_ingress_attempts,
           g_rtcore_replay_unit_arbitration_stats
               .warp_completion_ingress_issued,
           g_rtcore_replay_unit_arbitration_stats
               .warp_completion_ingress_budget_exhausted);
    fflush(stdout);
}

static unsigned rtcore_replay_data_path_total_accesses()
{
    return g_rtcore_replay_data_path_access_stats.lane_request_state_identity_reads +
           g_rtcore_replay_data_path_access_stats.lane_request_state_identity_writes +
           g_rtcore_replay_data_path_access_stats.request_state_reads +
           g_rtcore_replay_data_path_access_stats.request_state_writes;
}

static bool rtcore_replay_data_path_contract_observed()
{
    return g_rtcore_replay_data_path_access_stats.lane_request_state_identity_reads > 0 &&
           g_rtcore_replay_data_path_access_stats.lane_request_state_identity_writes > 0 &&
           g_rtcore_replay_data_path_access_stats.request_state_reads > 0 &&
           g_rtcore_replay_data_path_access_stats.request_state_writes > 0 &&
           g_rtcore_replay_data_path_access_stats.max_lane_request_state_entries > 0;
}

static void rtcore_maybe_log_replay_data_path_access_stats(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    if (!rtcore_replay_data_path_access_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_data_path_access_stats_logs_emitted >=
        rtcore_replay_data_path_access_stats_log_limit()) {
        return;
    }
    if (rtcore_replay_data_path_total_accesses() == 0) {
        return;
    }
    if (!rtcore_replay_data_path_contract_observed()) {
        return;
    }

    g_rtcore_replay_data_path_access_stats_logs_emitted++;
    printf("GPGPU-Sim RTCORE_REPLAY_DATA_PATH_ACCESS_STATS "
           "owner_hw_sid=%u service_cycle=%llu "
           "lane_request_state_identity_reads=%u "
           "lane_request_state_identity_writes=%u request_state_reads=%u "
           "request_state_writes=%u "
           "max_lane_request_state_entries=%u\n",
           owner_hw_sid, service_cycle,
           g_rtcore_replay_data_path_access_stats.lane_request_state_identity_reads,
           g_rtcore_replay_data_path_access_stats.lane_request_state_identity_writes,
           g_rtcore_replay_data_path_access_stats.request_state_reads,
           g_rtcore_replay_data_path_access_stats.request_state_writes,
           g_rtcore_replay_data_path_access_stats.max_lane_request_state_entries);
    fflush(stdout);
}

static void rtcore_maybe_log_replay_v03_hw_request_state_scoreboard_stats(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    const rtcore_replay_data_path_access_snapshot &before,
    const rtcore_replay_data_path_access_snapshot &after)
{
    if (!rtcore_replay_v03_hw_request_state_scoreboard_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_v03_hw_request_state_scoreboard_stats_logs_emitted >=
        rtcore_replay_v03_hw_request_state_scoreboard_stats_log_limit()) {
        return;
    }

    const unsigned request_state_hot_delta =
        rtcore_replay_data_path_access_delta(after.request_state_reads,
                                             before.request_state_reads) +
        rtcore_replay_data_path_access_delta(after.request_state_writes,
                                             before.request_state_writes);
    const unsigned lane_request_state_identity_delta =
        rtcore_replay_data_path_access_delta(after.lane_request_state_identity_reads,
                                             before.lane_request_state_identity_reads) +
        rtcore_replay_data_path_access_delta(after.lane_request_state_identity_writes,
                                             before.lane_request_state_identity_writes);
    if (request_state_hot_delta == 0) {
        return;
    }

    const unsigned scoreboard_update_count =
        g_rtcore_replay_data_path_access_stats.request_state_writes;
    const unsigned payload_cache_update_count = scoreboard_update_count;
    if (request_state_hot_delta >
        g_rtcore_replay_v03_hw_request_state_scoreboard_stats
            .max_request_state_hot_delta) {
        g_rtcore_replay_v03_hw_request_state_scoreboard_stats
            .max_request_state_hot_delta = request_state_hot_delta;
    }
    if (scoreboard_update_count >
        g_rtcore_replay_v03_hw_request_state_scoreboard_stats
            .max_scoreboard_update_count) {
        g_rtcore_replay_v03_hw_request_state_scoreboard_stats
            .max_scoreboard_update_count = scoreboard_update_count;
    }

    g_rtcore_replay_v03_hw_request_state_scoreboard_stats.evaluations++;
    g_rtcore_replay_v03_hw_request_state_scoreboard_stats_logs_emitted++;
    printf("GPGPU-Sim RTCORE_REPLAY_V03_HW_REQUEST_STATE_SCOREBOARD_STATS "
           "owner_hw_sid=%u service_cycle=%llu stats_enabled=1 "
           "per_entry_update_model=1 "
           "banked_ready_selection_model=%u request_state_bank_count=%u "
           "per_bank_select_width=1 ready_bank_rr_cursor_model=1 "
           "request_state_read_count=%u request_state_write_count=%u "
           "lane_request_state_identity_read_count=%u lane_request_state_identity_write_count=%u "
           "request_state_hot_delta=%u lane_request_state_identity_delta=%u "
           "scoreboard_update_count=%u payload_cache_update_count=%u "
           "evaluations=%u max_request_state_hot_delta=%u "
           "max_scoreboard_update_count=%u\n",
           owner_hw_sid, service_cycle,
           rtcore_replay_v03_hw_banked_ready_selection_enabled() ? 1u : 0u,
           rtcore_replay_v03_hw_request_state_bank_count_config(),
           g_rtcore_replay_data_path_access_stats.request_state_reads,
           g_rtcore_replay_data_path_access_stats.request_state_writes,
           g_rtcore_replay_data_path_access_stats.lane_request_state_identity_reads,
           g_rtcore_replay_data_path_access_stats.lane_request_state_identity_writes,
           request_state_hot_delta, lane_request_state_identity_delta,
           scoreboard_update_count, payload_cache_update_count,
           g_rtcore_replay_v03_hw_request_state_scoreboard_stats.evaluations,
           g_rtcore_replay_v03_hw_request_state_scoreboard_stats
               .max_request_state_hot_delta,
           g_rtcore_replay_v03_hw_request_state_scoreboard_stats
               .max_scoreboard_update_count);
    fflush(stdout);
}

static bool rtcore_replay_request_state_has_unit_executing_work(
    const rtcore_replay_lane_request &request)
{
    return request.unit_latency_gate_pending;
}

static bool rtcore_replay_request_state_has_unit_ready_pending_work(
    const rtcore_replay_lane_request &request)
{
    (void)request;
    return false;
}

static void rtcore_count_replay_unit_request_state_for_owner(
    unsigned owner_hw_sid, unsigned *executing_count,
    unsigned *ready_pending_count)
{
    if (executing_count) {
        *executing_count = 0;
    }
    if (ready_pending_count) {
        *ready_pending_count = 0;
    }

    for (std::map<unsigned, rtcore_replay_lane_request>::const_iterator it =
             g_rtcore_replay_lane_requests.begin();
         it != g_rtcore_replay_lane_requests.end(); ++it) {
        const rtcore_replay_lane_request &request = it->second;
        if (!request.valid || request.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        if (executing_count &&
            rtcore_replay_request_state_has_unit_executing_work(request)) {
            (*executing_count)++;
        }
        if (ready_pending_count &&
            rtcore_replay_request_state_has_unit_ready_pending_work(request)) {
            (*ready_pending_count)++;
        }
    }
}

static void rtcore_maybe_log_replay_v03_hw_unit_state_wake_service_stats(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    if (!rtcore_replay_v03_hw_unit_state_wake_service_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_v03_hw_unit_state_wake_service_stats_logs_emitted >=
        rtcore_replay_v03_hw_unit_state_wake_service_stats_log_limit()) {
        return;
    }

    unsigned executing_count = 0;
    unsigned ready_pending_count = 0;
    rtcore_count_replay_unit_request_state_for_owner(
        owner_hw_sid, &executing_count, &ready_pending_count);

    if (executing_count == 0 && ready_pending_count == 0 &&
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
                .wake_attempt_count == 0) {
        return;
    }

    if (executing_count >
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_request_state_executing_count) {
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_request_state_executing_count = executing_count;
    }
    if (ready_pending_count >
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_request_state_ready_pending_count) {
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_request_state_ready_pending_count = ready_pending_count;
    }
    if (g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .wake_progress_count >
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_wake_progress_count) {
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_wake_progress_count =
            g_rtcore_replay_v03_hw_unit_state_wake_service_stats
                .wake_progress_count;
    }
    if (g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .request_state_unit_wake_progress_count >
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_request_state_unit_wake_progress_count) {
        g_rtcore_replay_v03_hw_unit_state_wake_service_stats
            .max_request_state_unit_wake_progress_count =
            g_rtcore_replay_v03_hw_unit_state_wake_service_stats
                .request_state_unit_wake_progress_count;
    }

    g_rtcore_replay_v03_hw_unit_state_wake_service_stats.evaluations++;
    g_rtcore_replay_v03_hw_unit_state_wake_service_stats_logs_emitted++;
    printf("GPGPU-Sim RTCORE_REPLAY_V03_HW_UNIT_STATE_WAKE_SERVICE_STATS "
           "owner_hw_sid=%u service_cycle=%llu stats_enabled=1 "
           "request_state_executing_model=1 "
           "request_state_ready_pending_model=1 "
           "request_state_unit_wake_service_path=1 "
           "request_state_unit_wake_attempt_count=%u "
           "request_state_unit_wake_progress_count=%u "
           "request_state_executing_count=%u "
           "request_state_ready_pending_count=%u "
           "wake_attempt_count=%u wake_progress_count=%u evaluations=%u "
           "max_request_state_executing_count=%u "
           "max_request_state_ready_pending_count=%u "
           "max_wake_progress_count=%u "
           "max_request_state_unit_wake_progress_count=%u\n",
           owner_hw_sid, service_cycle,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .request_state_unit_wake_attempt_count,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .request_state_unit_wake_progress_count,
           executing_count, ready_pending_count,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .wake_attempt_count,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .wake_progress_count,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats.evaluations,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .max_request_state_executing_count,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .max_request_state_ready_pending_count,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .max_wake_progress_count,
           g_rtcore_replay_v03_hw_unit_state_wake_service_stats
               .max_request_state_unit_wake_progress_count);
    fflush(stdout);
}

static unsigned rtcore_replay_memory_outstanding_chunk_count(
    const rtcore_replay_lane_request &request)
{
    unsigned chunk_count = 0;
    if (request.v02_lsu_response_wait_gate_pending) {
        const unsigned pending_chunks =
            request.v02_lsu_response_wait_chunk_count >
                    request.v02_lsu_response_wait_completed_chunk_count
                ? request.v02_lsu_response_wait_chunk_count -
                      request.v02_lsu_response_wait_completed_chunk_count
                : 0;
        chunk_count += pending_chunks;
    }
    if (request.memory_wake_latency_gate_pending &&
        request.memory_wake_event_index < request.events.size()) {
        chunk_count += rtcore_replay_memory_request_chunks_for_event(
            request.events[request.memory_wake_event_index]);
    }
    return chunk_count;
}

static bool rtcore_replay_request_state_has_memory_outstanding_work(
    const rtcore_replay_lane_request &request)
{
    return request.v02_lsu_response_wait_gate_pending ||
           request.memory_contention_gate_pending ||
           request.memory_wake_latency_gate_pending;
}

static void rtcore_count_replay_memory_outstanding_for_owner(
    unsigned owner_hw_sid, unsigned *outstanding_entry_count,
    unsigned *response_fanout_waiter_count, unsigned *outstanding_chunk_count)
{
    if (outstanding_entry_count) {
        *outstanding_entry_count = 0;
    }
    if (response_fanout_waiter_count) {
        *response_fanout_waiter_count = 0;
    }
    if (outstanding_chunk_count) {
        *outstanding_chunk_count = 0;
    }

    std::set<rtcore_replay_memory_unit_transaction_key> unique_transactions;
    unsigned transactionless_entry_count = 0;
    for (std::map<rtcore_replay_memory_outstanding_key,
                  rtcore_replay_memory_outstanding_entry>::
             const_iterator it =
                 g_rtcore_replay_memory_outstanding_table.begin();
         it != g_rtcore_replay_memory_outstanding_table.end(); ++it) {
        const rtcore_replay_memory_outstanding_entry &entry =
            it->second;
        if (!entry.active || entry.key.owner_hw_sid != owner_hw_sid) {
            continue;
        }
        if (entry.has_transaction_key && !entry.transaction_keys.empty()) {
            unique_transactions.insert(entry.transaction_keys.begin(),
                                       entry.transaction_keys.end());
        } else if (entry.has_transaction_key) {
            unique_transactions.insert(entry.transaction_key);
        } else {
            transactionless_entry_count++;
        }
        if (response_fanout_waiter_count) {
            *response_fanout_waiter_count +=
                entry.response_fanout_waiter_count;
        }
        if (outstanding_chunk_count) {
            *outstanding_chunk_count += entry.chunk_count;
        }
    }
    if (outstanding_entry_count) {
        *outstanding_entry_count =
            static_cast<unsigned>(unique_transactions.size()) +
            transactionless_entry_count;
    }
}

static void rtcore_maybe_log_replay_v03_hw_memory_outstanding_stats(
    unsigned owner_hw_sid, unsigned long long service_cycle)
{
    if (!rtcore_replay_v03_hw_memory_outstanding_stats_log_enabled()) {
        return;
    }
    if (g_rtcore_replay_v03_hw_memory_outstanding_stats_logs_emitted >=
        rtcore_replay_v03_hw_memory_outstanding_stats_log_limit()) {
        return;
    }

    unsigned outstanding_entry_count = 0;
    unsigned response_fanout_waiter_count = 0;
    unsigned outstanding_chunk_count = 0;
    rtcore_count_replay_memory_outstanding_for_owner(
        owner_hw_sid, &outstanding_entry_count, &response_fanout_waiter_count,
        &outstanding_chunk_count);

    if (outstanding_entry_count == 0 && response_fanout_waiter_count == 0 &&
        outstanding_chunk_count == 0 &&
        g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_ready_issue_attempt_count == 0 &&
        g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_address_gen_attempt_count == 0 &&
        g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_wake_attempt_count == 0) {
        return;
    }

    if (outstanding_entry_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_outstanding_entry_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_outstanding_entry_count = outstanding_entry_count;
    }
    if (response_fanout_waiter_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_response_fanout_waiter_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_response_fanout_waiter_count =
            response_fanout_waiter_count;
    }
    if (outstanding_chunk_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_outstanding_chunk_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_outstanding_chunk_count = outstanding_chunk_count;
    }
    if (outstanding_entry_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_active_entry_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_active_entry_count = outstanding_entry_count;
    }
    if (response_fanout_waiter_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_response_fanout_waiter_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_response_fanout_waiter_count =
            response_fanout_waiter_count;
    }
    if (g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_outstanding_table_register_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_register_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_register_count =
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_outstanding_table_register_count;
    }
    if (g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_outstanding_table_release_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_release_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_outstanding_table_release_count =
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_outstanding_table_release_count;
    }
    if (g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_wake_progress_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_wake_progress_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_wake_progress_count =
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_wake_progress_count;
    }
    if (g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_ready_issue_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_ready_issue_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_ready_issue_count =
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_ready_issue_count;
    }
    if (g_rtcore_replay_v03_hw_memory_outstanding_stats
            .memory_address_gen_issued_count >
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_address_gen_issued_count) {
        g_rtcore_replay_v03_hw_memory_outstanding_stats
            .max_memory_address_gen_issued_count =
            g_rtcore_replay_v03_hw_memory_outstanding_stats
                .memory_address_gen_issued_count;
    }

    g_rtcore_replay_v03_hw_memory_outstanding_stats.evaluations++;
    g_rtcore_replay_v03_hw_memory_outstanding_stats_logs_emitted++;
    printf("GPGPU-Sim RTCORE_REPLAY_V03_HW_MEMORY_OUTSTANDING_STATS "
           "owner_hw_sid=%u service_cycle=%llu stats_enabled=1 "
           "memory_unit_internal=1 outstanding_table_model=1 "
           "response_fanout_model=1 "
           "request_state_memory_pending_model=1 "
           "memory_ready_bit_issue_model=1 "
           "same_cycle_transaction_alloc_model=1 "
           "outstanding_capacity_transaction_model=1 "
           "memory_address_gen_budget=%u "
           "memory_address_gen_latency=%u "
           "memory_outstanding_capacity=%u "
           "memory_outstanding_alloc_budget=%u "
           "memory_response_wake_budget=%u "
           "memory_address_gen_attempt_count=%u "
           "memory_address_gen_issued_count=%u "
           "memory_address_gen_blocked_count=%u "
           "memory_address_gen_latency_blocked_count=%u "
           "memory_outstanding_capacity_blocked_count=%u "
           "memory_ready_issue_attempt_count=%u "
           "memory_ready_issue_count=%u "
           "outstanding_entry_count=%u response_fanout_waiter_count=%u "
           "outstanding_chunk_count=%u "
           "memory_outstanding_table_active_entry_count=%u "
           "memory_outstanding_table_response_fanout_waiter_count=%u "
           "memory_outstanding_table_register_count=%u "
           "memory_outstanding_table_release_count=%u "
           "memory_wake_attempt_count=%u memory_wake_progress_count=%u "
           "evaluations=%u "
           "max_memory_address_gen_issued_count=%u "
           "max_memory_address_gen_blocked_count=%u "
           "max_memory_address_gen_latency_blocked_count=%u "
           "max_memory_outstanding_capacity_blocked_count=%u "
           "max_memory_ready_issue_count=%u max_outstanding_entry_count=%u "
           "max_response_fanout_waiter_count=%u "
           "max_outstanding_chunk_count=%u "
           "max_memory_outstanding_table_active_entry_count=%u "
           "max_memory_outstanding_table_response_fanout_waiter_count=%u "
           "max_memory_outstanding_table_register_count=%u "
           "max_memory_outstanding_table_release_count=%u "
           "max_memory_wake_progress_count=%u\n",
           owner_hw_sid, service_cycle,
           rtcore_replay_memory_address_gen_budget_config(),
           rtcore_replay_memory_address_gen_latency_config(),
           rtcore_replay_memory_outstanding_capacity_config(),
           rtcore_replay_memory_outstanding_alloc_budget_config(),
           rtcore_replay_memory_wake_budget_config(),
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_address_gen_attempt_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_address_gen_issued_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_address_gen_blocked_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_address_gen_latency_blocked_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_outstanding_capacity_blocked_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_ready_issue_attempt_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_ready_issue_count,
           outstanding_entry_count,
           response_fanout_waiter_count, outstanding_chunk_count,
           outstanding_entry_count, response_fanout_waiter_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_outstanding_table_register_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_outstanding_table_release_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_wake_attempt_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .memory_wake_progress_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats.evaluations,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_address_gen_issued_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_address_gen_blocked_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_address_gen_latency_blocked_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_outstanding_capacity_blocked_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_ready_issue_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_outstanding_entry_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_response_fanout_waiter_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_outstanding_chunk_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_outstanding_table_active_entry_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_outstanding_table_response_fanout_waiter_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_outstanding_table_register_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_outstanding_table_release_count,
           g_rtcore_replay_v03_hw_memory_outstanding_stats
               .max_memory_wake_progress_count);
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
    if (rtcore_continuation_model_enabled()) {
        rtcore_register_continuation_final_summary();
    }
    if (!result.service_enabled) {
        return result;
    }
    const rtcore_replay_data_path_access_snapshot data_path_before =
        rtcore_get_replay_data_path_access_snapshot();
    result.tick_result =
        rtcore_maybe_service_replay_tick(owner_hw_sid, service_cycle);
    const rtcore_replay_data_path_access_snapshot data_path_after =
        rtcore_get_replay_data_path_access_snapshot();
    rtcore_record_replay_service_tick_result(result.tick_result);
    rtcore_maybe_log_replay_v03_hw_request_state_scoreboard_stats(
        owner_hw_sid, service_cycle, data_path_before, data_path_after);
    rtcore_maybe_log_replay_v03_hw_unit_state_wake_service_stats(
        owner_hw_sid, service_cycle);
    rtcore_maybe_log_replay_v03_hw_memory_outstanding_stats(owner_hw_sid,
                                                            service_cycle);
    rtcore_publish_replay_service_tick_stats_snapshot();
    rtcore_maybe_log_replay_data_path_access_stats(owner_hw_sid, service_cycle);
    rtcore_maybe_log_replay_unit_arbitration_stats(owner_hw_sid);
    rtcore_maybe_log_replay_model_summary_stats(owner_hw_sid, service_cycle);
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

static rtcore_memory_unit_request_snapshot
rtcore_v04_private_shared_snapshot(
    const rtcore::v04::private_shared::shared_write_v0 &operation)
{
    rtcore_memory_unit_request_snapshot snapshot = {};
    snapshot.valid = operation.valid;
    snapshot.address_space = RTCORE_MEMORY_ADDRESS_SPACE_SHARED;
    snapshot.operation = RTCORE_MEMORY_OPERATION_WRITE;
    snapshot.destination =
        RTCORE_MEMORY_DESTINATION_PRIVATE_COMMIT_ACK;
    snapshot.response_target = RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE;
    snapshot.owner_hw_sid = operation.owner.owner_hw_sid;
    snapshot.rt_request_id = operation.owner.request_identity;
    snapshot.lane_id = operation.owner.lane_id;
    snapshot.resident_warp_id = operation.owner.resident_warp_id;
    snapshot.request_generation = operation.owner.generation;
    snapshot.private_slot_id = operation.owner.private_slot_id;
    snapshot.memory_op_seq = operation.memory_op_seq;
    snapshot.chunk_id = operation.chunk_id;
    snapshot.chunk_count = operation.chunk_count;
    snapshot.access_kind =
        RTCORE_MEMORY_ACCESS_PRIVATE_FRONTIER_INIT;
    snapshot.aligned_32b_addr = operation.aligned_32b_address;
    snapshot.byte_mask = operation.byte_mask;
    std::memcpy(snapshot.payload, operation.payload,
                sizeof(snapshot.payload));
    snapshot.is_write = true;
    snapshot.issue_cycle = operation.enqueue_cycle;
    return snapshot;
}

static bool rtcore_v04_private_shared_operation_from_snapshot(
    const rtcore_memory_unit_request_snapshot &snapshot,
    rtcore::v04::private_shared::shared_write_v0 *operation)
{
    if (operation == NULL || !snapshot.valid ||
        snapshot.address_space != RTCORE_MEMORY_ADDRESS_SPACE_SHARED ||
        snapshot.operation != RTCORE_MEMORY_OPERATION_WRITE ||
        snapshot.destination !=
            RTCORE_MEMORY_DESTINATION_PRIVATE_COMMIT_ACK ||
        snapshot.response_target !=
            RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE ||
        snapshot.access_kind !=
            RTCORE_MEMORY_ACCESS_PRIVATE_FRONTIER_INIT ||
        snapshot.resident_warp_id >=
            rtcore::v04::private_shared::kResidentWarpCapacity ||
        snapshot.lane_id >=
            rtcore::v04::private_shared::kLaneCapacity ||
        snapshot.private_slot_id >= 256 ||
        snapshot.chunk_count != 2 ||
        snapshot.chunk_id >= snapshot.chunk_count ||
        !snapshot.is_write) {
        return false;
    }
    *operation = rtcore::v04::private_shared::shared_write_v0();
    operation->valid = true;
    operation->owner.owner_hw_sid = snapshot.owner_hw_sid;
    operation->owner.resident_warp_id = snapshot.resident_warp_id;
    operation->owner.request_identity = snapshot.rt_request_id;
    operation->owner.generation = snapshot.request_generation;
    operation->owner.private_slot_id = snapshot.private_slot_id;
    operation->owner.lane_id =
        static_cast<uint8_t>(snapshot.lane_id);
    operation->memory_op_seq = snapshot.memory_op_seq;
    operation->chunk_id = static_cast<uint8_t>(snapshot.chunk_id);
    operation->chunk_count =
        static_cast<uint8_t>(snapshot.chunk_count);
    operation->aligned_32b_address = snapshot.aligned_32b_addr;
    operation->byte_mask = snapshot.byte_mask;
    std::memcpy(operation->payload, snapshot.payload,
                sizeof(operation->payload));
    operation->enqueue_cycle = snapshot.issue_cycle;
    return true;
}

extern "C" bool rtcore_accept_v04_private_shared_request(
    const rtcore_memory_unit_request_snapshot *snapshot,
    unsigned long long accepted_cycle)
{
    namespace private_shared = rtcore::v04::private_shared;
    if (snapshot == NULL ||
        !rtcore_v04_private_frontier_live_init_enabled()) {
        return false;
    }
    private_shared::shared_write_v0 operation = {};
    if (!rtcore_v04_private_shared_operation_from_snapshot(
            *snapshot, &operation)) {
        return false;
    }
    private_shared::backing_state_v0 &backing =
        rtcore_v04_private_shared_backing_for(
            snapshot->owner_hw_sid);
    return private_shared::accept_shared_offer(
               &backing, operation, accepted_cycle) ==
           private_shared::kStatusOk;
}

extern "C" bool
rtcore_service_replay_cycle_for_sm_with_identity_and_memory_unit(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *service_enabled, bool *memory_progressed, bool *ready_progressed,
    rtcore_replay_service_cycle_identity_snapshot *identity_snapshot,
    rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    rtcore_replay_service_cycle_result result =
        rtcore_service_replay_cycle(owner_hw_sid, service_cycle);
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
    if (sideband_snapshot) {
        *sideband_snapshot = rtcore_memory_unit_request_snapshot();
        bool shared_offered = false;
        if (rtcore_v04_private_frontier_live_init_enabled()) {
            rtcore::v04::private_shared::shared_write_v0 operation = {};
            shared_offered =
                rtcore::v04::private_shared::pop_shared_offer(
                    &rtcore_v04_private_shared_backing_for(owner_hw_sid),
                    service_cycle, &operation);
            if (shared_offered) {
                *sideband_snapshot =
                    rtcore_v04_private_shared_snapshot(operation);
            }
        }
        if (!shared_offered) {
            std::map<unsigned,
                     std::deque<rtcore_memory_unit_request_snapshot> >::
                iterator queue_it =
                    g_rtcore_memory_unit_request_snapshots_by_owner.find(
                        owner_hw_sid);
            if (queue_it !=
                    g_rtcore_memory_unit_request_snapshots_by_owner.end() &&
                !queue_it->second.empty()) {
                *sideband_snapshot = queue_it->second.front();
                queue_it->second.pop_front();
            }
        }
    }
    return result.tick_result.progressed;
}

extern "C" bool
rtcore_service_replay_cycle_for_sm_with_identity_and_lsu_sideband(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *service_enabled, bool *memory_progressed, bool *ready_progressed,
    rtcore_replay_service_cycle_identity_snapshot *identity_snapshot,
    rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    return rtcore_service_replay_cycle_for_sm_with_identity_and_memory_unit(
        owner_hw_sid, service_cycle, service_enabled, memory_progressed,
        ready_progressed, identity_snapshot, sideband_snapshot);
}

extern "C" bool rtcore_pop_memory_unit_request_for_sm(
    unsigned owner_hw_sid,
    rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    if (sideband_snapshot) {
        *sideband_snapshot = rtcore_memory_unit_request_snapshot();
    }
    std::map<unsigned,
             std::deque<rtcore_memory_unit_request_snapshot> >::
        iterator queue_it =
            g_rtcore_memory_unit_request_snapshots_by_owner.find(
                owner_hw_sid);
    if (queue_it ==
            g_rtcore_memory_unit_request_snapshots_by_owner.end() ||
        queue_it->second.empty()) {
        return false;
    }
    if (sideband_snapshot) {
        *sideband_snapshot = queue_it->second.front();
    }
    queue_it->second.pop_front();
    return true;
}

extern "C" bool rtcore_pop_v02_lsu_sideband_request_for_sm(
    unsigned owner_hw_sid,
    rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    return rtcore_pop_memory_unit_request_for_sm(owner_hw_sid,
                                                 sideband_snapshot);
}

extern "C" bool rtcore_push_front_memory_unit_request_for_sm(
    unsigned owner_hw_sid,
    const rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    if (!sideband_snapshot || !sideband_snapshot->valid) {
        return false;
    }
    g_rtcore_memory_unit_request_snapshots_by_owner[owner_hw_sid]
        .push_front(*sideband_snapshot);
    return true;
}

extern "C" bool rtcore_push_front_v02_lsu_sideband_request_for_sm(
    unsigned owner_hw_sid,
    const rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    return rtcore_push_front_memory_unit_request_for_sm(owner_hw_sid,
                                                        sideband_snapshot);
}

extern "C" bool rtcore_push_back_memory_unit_request_for_sm(
    unsigned owner_hw_sid,
    const rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    if (!sideband_snapshot || !sideband_snapshot->valid) {
        return false;
    }
    g_rtcore_memory_unit_request_snapshots_by_owner[owner_hw_sid]
        .push_back(*sideband_snapshot);
    return true;
}

extern "C" bool rtcore_push_back_v02_lsu_sideband_request_for_sm(
    unsigned owner_hw_sid,
    const rtcore_memory_unit_request_snapshot *sideband_snapshot)
{
    return rtcore_push_back_memory_unit_request_for_sm(owner_hw_sid,
                                                       sideband_snapshot);
}

extern "C" unsigned rtcore_count_memory_unit_requests_for_sm(
    unsigned owner_hw_sid)
{
    std::map<unsigned,
             std::deque<rtcore_memory_unit_request_snapshot> >::
        const_iterator queue_it =
            g_rtcore_memory_unit_request_snapshots_by_owner.find(
                owner_hw_sid);
    if (queue_it ==
        g_rtcore_memory_unit_request_snapshots_by_owner.end()) {
        return 0;
    }
    return static_cast<unsigned>(queue_it->second.size());
}

extern "C" unsigned rtcore_count_v02_lsu_sideband_requests_for_sm(
    unsigned owner_hw_sid)
{
    return rtcore_count_memory_unit_requests_for_sm(owner_hw_sid);
}

extern "C" void rtcore_enqueue_memory_unit_handoff_window_request(
    unsigned owner_hw_sid, unsigned rt_request_id, unsigned lane_id,
    unsigned memory_op_seq, unsigned access_kind,
    unsigned long long byte_address, unsigned chunk_count, bool is_write,
    unsigned long long issue_cycle)
{
    if (chunk_count == 0) {
        chunk_count = 1;
    }
    for (unsigned chunk_id = 0; chunk_id < chunk_count; ++chunk_id) {
        const unsigned long long chunk_address =
            byte_address +
            static_cast<unsigned long long>(
                chunk_id * RTCORE_V02_LSU_MEMORY_REQUEST_GRANULE_BYTES);
        rtcore_memory_unit_request_snapshot snapshot = {};
        snapshot.valid = true;
        snapshot.response_target = RTCORE_V02_LSU_RESPONSE_TARGET_RTCORE;
        snapshot.owner_hw_sid = owner_hw_sid;
        snapshot.rt_request_id = rt_request_id;
        snapshot.lane_id = lane_id;
        snapshot.memory_op_seq = memory_op_seq;
        snapshot.chunk_id = chunk_id;
        snapshot.chunk_count = chunk_count;
        snapshot.access_kind = access_kind;
        snapshot.aligned_32b_addr = rtcore_v02_lsu_align_32b(chunk_address);
        snapshot.is_write = is_write;
        snapshot.issue_cycle = issue_cycle;
        g_rtcore_memory_unit_request_snapshots_by_owner[owner_hw_sid]
            .push_back(snapshot);
    }
}

extern "C" void rtcore_enqueue_v02_lsu_handoff_window_sideband_request(
    unsigned owner_hw_sid, unsigned rt_request_id, unsigned lane_id,
    unsigned memory_op_seq, unsigned access_kind,
    unsigned long long byte_address, unsigned chunk_count, bool is_write,
    unsigned long long issue_cycle)
{
    rtcore_enqueue_memory_unit_handoff_window_request(
        owner_hw_sid, rt_request_id, lane_id, memory_op_seq, access_kind,
        byte_address, chunk_count, is_write, issue_cycle);
}

extern "C" bool rtcore_record_memory_unit_response(
    unsigned owner_hw_sid, unsigned rt_request_id, unsigned memory_op_seq,
    unsigned chunk_id, unsigned chunk_count, unsigned response_target,
    unsigned long long response_cycle)
{
    if (rtcore_record_v04_live_handoff_publication_response(
            owner_hw_sid, rt_request_id, memory_op_seq, chunk_id,
            chunk_count, response_target, response_cycle)) {
        return true;
    }
    return rtcore_record_memory_unit_response_wait_chunk(
        owner_hw_sid, rt_request_id, memory_op_seq, chunk_id, chunk_count,
        response_target, response_cycle);
}

extern "C" bool rtcore_record_v02_lsu_sideband_memory_response(
    unsigned owner_hw_sid, unsigned rt_request_id, unsigned memory_op_seq,
    unsigned chunk_id, unsigned chunk_count, unsigned response_target,
    unsigned long long response_cycle)
{
    return rtcore_record_memory_unit_response(
        owner_hw_sid, rt_request_id, memory_op_seq, chunk_id, chunk_count,
        response_target, response_cycle);
}

extern "C" bool rtcore_service_replay_cycle_for_sm(
    unsigned owner_hw_sid, unsigned long long service_cycle,
    bool *service_enabled, bool *memory_progressed, bool *ready_progressed)
{
    return rtcore_service_replay_cycle_for_sm_with_identity(
        owner_hw_sid, service_cycle, service_enabled, memory_progressed,
        ready_progressed, NULL);
}

extern "C" bool rtcore_query_replay_warp_completion_entry(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask,
    rtcore_replay_warp_completion_entry_snapshot *snapshot)
{
    rtcore_replay_warp_completion_entry_snapshot local_snapshot = {};
    local_snapshot.enabled = rtcore_replay_warp_completion_entry_enabled();
    local_snapshot.owner_hw_sid = owner_hw_sid;
    local_snapshot.warp_uid = warp_uid;
    local_snapshot.warp_id = warp_id;
    local_snapshot.active_mask = active_mask;

    if (local_snapshot.enabled) {
        rtcore_replay_warp_completion_entry_key key = {};
        key.owner_hw_sid = owner_hw_sid;
        key.warp_uid = warp_uid;
        key.warp_id = warp_id;
        key.active_mask = active_mask;
        std::map<rtcore_replay_warp_completion_entry_key,
                 rtcore_replay_warp_completion_entry_state>::const_iterator it =
            g_rtcore_replay_warp_completion_entries.find(key);
        if (it != g_rtcore_replay_warp_completion_entries.end()) {
            local_snapshot.found = true;
            local_snapshot.all_active_lanes_complete =
                it->second.all_active_lanes_complete;
            local_snapshot.admitted_lane_mask = it->second.admitted_lane_mask;
            local_snapshot.completed_lane_mask = it->second.completed_lane_mask;
            local_snapshot.result_valid_mask = it->second.result_valid_mask;
            local_snapshot.completed_lane_count = it->second.completed_lane_count;
            local_snapshot.result_reg_base = it->second.result_reg_base;
            local_snapshot.packet_schema_version =
                it->second.packet_schema_version;
            local_snapshot.context_profile_valid_mask =
                it->second.context_profile_valid_mask;
            local_snapshot.reported_attribute_metadata_valid_mask =
                it->second.reported_attribute_metadata_valid_mask;
            local_snapshot.inline_payload_location_valid_mask =
                it->second.inline_payload_location_valid_mask;
            local_snapshot.inline_payload_base_word =
                it->second.inline_payload_base_word;
            local_snapshot.max_inline_attribute_words =
                it->second.max_inline_attribute_words;
            local_snapshot.lane_completion_valid_mask =
                it->second.lane_completion_valid_mask;
            local_snapshot.terminal_lane_mask = it->second.terminal_lane_mask;
            local_snapshot.continuation_lane_mask =
                it->second.continuation_lane_mask;
            local_snapshot.unsupported_reason_mask =
                it->second.unsupported_reason_mask;
            local_snapshot.handoff_selector_valid_mask =
                it->second.handoff_selector_valid_mask;
            local_snapshot.handoff_candidate_valid_mask =
                it->second.handoff_candidate_valid_mask;
            local_snapshot.handoff_software_return_valid_mask =
                it->second.handoff_software_return_valid_mask;
            local_snapshot.v04_shadow_boundary_image_valid_mask =
                it->second.v04_shadow_boundary_image_valid_mask;
            for (unsigned lane = 0; lane < 32; ++lane) {
                local_snapshot.result_data_slot[lane] =
                    it->second.result_data_slot[lane];
                local_snapshot.lane_status[lane] = it->second.lane_status[lane];
                local_snapshot.lane_completion_reason[lane] =
                    it->second.lane_completion_reason[lane];
                local_snapshot.lane_continuation_depth[lane] =
                    it->second.lane_continuation_depth[lane];
                local_snapshot.context_layout_version[lane] =
                    it->second.context_layout_version[lane];
                local_snapshot.context_valid_flags[lane] =
                    it->second.context_valid_flags[lane];
                local_snapshot.pipeline_profile_id[lane] =
                    it->second.pipeline_profile_id[lane];
                local_snapshot.bvh_format_profile_id[lane] =
                    it->second.bvh_format_profile_id[lane];
                local_snapshot.boundary_candidates[lane] =
                    it->second.boundary_candidates[lane];
                for (unsigned word = 0; word < 32; ++word) {
                    local_snapshot.handoff_words[lane][word] =
                        it->second.handoff_words[lane][word];
                    local_snapshot.v04_shadow_handoff_words[lane][word] =
                        it->second.v04_shadow_handoff_words[lane][word];
                }
            }
            local_snapshot.scoreboard_handoff_ready =
                it->second.scoreboard_handoff_ready;
            local_snapshot.scoreboard_handoff_delivered =
                it->second.scoreboard_handoff_delivered;
            local_snapshot.scoreboard_handoff_cycle =
                it->second.scoreboard_handoff_cycle;
        }
    }

    if (snapshot) {
        *snapshot = local_snapshot;
    }
    return local_snapshot.enabled && local_snapshot.found &&
           local_snapshot.all_active_lanes_complete;
}

extern "C" bool rtcore_release_replay_warp_completion_entry(
    unsigned owner_hw_sid, unsigned warp_uid, unsigned warp_id,
    unsigned active_mask)
{
    rtcore_replay_warp_completion_entry_key key = {};
    key.owner_hw_sid = owner_hw_sid;
    key.warp_uid = warp_uid;
    key.warp_id = warp_id;
    key.active_mask = active_mask;
    std::map<rtcore_replay_warp_completion_entry_key,
             rtcore_replay_warp_completion_entry_state>::iterator it =
        g_rtcore_replay_warp_completion_entries.find(key);
    if (it == g_rtcore_replay_warp_completion_entries.end() ||
        !it->second.scoreboard_handoff_delivered) {
        return false;
    }
    g_rtcore_replay_warp_completion_entries.erase(it);
    printf("GPGPU-Sim RTCORE_REPLAY_COMPLETION_PACKET_RELEASE "
           "owner_hw_sid=%u warp_uid=%u warp_id=%u active_mask=0x%08x\n",
           owner_hw_sid, warp_uid, warp_id, active_mask);
    fflush(stdout);
    return true;
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

static bool rtcore_v04_make_typed_node_candidate_input(
    const uint8_t *raw_node, const float3 &origin, const float3 &direction,
    float t_min, float t_max, float committed_t, uint32_t ray_flags,
    uint32_t cull_mask, bool top_level,
    rtcore::v04::typed_node::candidate_input_v0 *input)
{
    namespace typed_node = rtcore::v04::typed_node;
    if (raw_node == NULL || input == NULL || (cull_mask & ~0xffu) != 0) {
        return false;
    }
    *input = typed_node::candidate_input_v0();
    input->profile_id = typed_node::kGenRtDerivedProfileId;
    input->level = top_level ? typed_node::kLevelTlas
                             : typed_node::kLevelBlas;
    input->ray.origin[0] = origin.x;
    input->ray.origin[1] = origin.y;
    input->ray.origin[2] = origin.z;
    input->ray.direction[0] = direction.x;
    input->ray.direction[1] = direction.y;
    input->ray.direction[2] = direction.z;
    input->ray.t_min = t_min;
    input->ray.t_max = t_max;
    input->policy.ray_flags = ray_flags;
    input->policy.cull_mask = static_cast<uint8_t>(cull_mask);
    input->committed_t = committed_t;
    return typed_node::make_raw_node_payload(raw_node, &input->raw_node);
}

struct rtcore_v04_typed_node_candidate_stats {
    unsigned tlas_nodes;
    unsigned blas_nodes;
    unsigned evaluated_children;
    unsigned hit_candidates;
    unsigned mismatches;

    rtcore_v04_typed_node_candidate_stats()
        : tlas_nodes(0), blas_nodes(0), evaluated_children(0),
          hit_candidates(0), mismatches(0) {}
};

static uint8_t rtcore_v04_popcount6(uint8_t value)
{
    unsigned count = 0;
    for (unsigned bit = 0; bit < 6; ++bit) {
        count += (value >> bit) & 1u;
    }
    return static_cast<uint8_t>(count);
}

static void rtcore_v04_observe_typed_node_candidates(
    const uint8_t *raw_node, const GEN_RT_BVH_INTERNAL_NODE &legacy_node,
    const float3 &origin, const float3 &direction, float t_min, float t_max,
    float committed_t, uint32_t ray_flags, uint32_t cull_mask, bool top_level,
    const bool legacy_child_hit[6], const float legacy_near_t[6],
    rtcore_v04_typed_node_candidate_stats *stats)
{
    namespace typed_node = rtcore::v04::typed_node;
    assert(raw_node != NULL);
    assert(stats != NULL);

    typed_node::candidate_input_v0 input = {};
    if (!rtcore_v04_make_typed_node_candidate_input(
            raw_node, origin, direction, t_min, t_max, committed_t,
            ray_flags, cull_mask, top_level, &input)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_KERNEL "
               "adapter_failure=1 level=%s\n",
               top_level ? "tlas" : "blas");
        fflush(stdout);
        abort();
    }

    const typed_node::candidate_result_v0 result = typed_node::execute(input);
    if (result.status != typed_node::kStatusOk) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_KERNEL "
               "kernel_failure=1 level=%s status=%s raw_node=%p\n",
               top_level ? "tlas" : "blas",
               typed_node::status_name(
                   static_cast<typed_node::status_kind>(result.status)),
               static_cast<const void *>(raw_node));
        fflush(stdout);
        abort();
    }

    uint8_t expected_evaluated_mask = 0;
    uint8_t expected_hit_mask = 0;
    uint8_t expected_order[6] = {};
    uint32_t legacy_near_t_bits[6] = {};
    uint8_t child_descriptor_mismatch_mask = 0;
    uint8_t near_t_mismatch_mask = 0;
    uint8_t order_mismatch_mask = 0;
    unsigned expected_count = 0;
    const bool legacy_node_visible =
        (legacy_node.NodeRayMask & (cull_mask & 0xffu)) != 0;
    const bool node_descriptor_mismatch =
        result.child_offset_blocks != legacy_node.ChildOffset ||
        result.node_ray_mask != legacy_node.NodeRayMask;
    bool mismatch = node_descriptor_mismatch;
    for (unsigned child = 0; child < 6; ++child) {
        const uint8_t legacy_size =
            static_cast<uint8_t>(legacy_node.ChildSize[child]);
        const uint8_t legacy_kind =
            static_cast<uint8_t>(legacy_node.ChildType[child]);
        if (result.child_size[child] != legacy_size ||
            result.child_kind[child] != legacy_kind) {
            mismatch = true;
            child_descriptor_mismatch_mask |=
                static_cast<uint8_t>(1u << child);
        }
        if (legacy_size == 0) continue;
        expected_evaluated_mask |= static_cast<uint8_t>(1u << child);
        legacy_near_t_bits[child] =
            rtcore_v04_fp32_bits(legacy_near_t[child]);
        const bool legacy_candidate_hit =
            legacy_child_hit[child] && legacy_node_visible;
        if (legacy_candidate_hit &&
            result.near_t_bits[child] != legacy_near_t_bits[child]) {
            mismatch = true;
            near_t_mismatch_mask |= static_cast<uint8_t>(1u << child);
        }
        if (legacy_candidate_hit) {
            expected_hit_mask |= static_cast<uint8_t>(1u << child);
            expected_order[expected_count++] = static_cast<uint8_t>(child);
        }
    }

    for (unsigned lhs = 1; lhs < expected_count; ++lhs) {
        const uint8_t slot = expected_order[lhs];
        unsigned rhs = lhs;
        while (rhs > 0) {
            const uint8_t previous_slot = expected_order[rhs - 1];
            if (legacy_near_t[previous_slot] < legacy_near_t[slot] ||
                (legacy_near_t[previous_slot] == legacy_near_t[slot] &&
                 previous_slot < slot)) {
                break;
            }
            expected_order[rhs] = previous_slot;
            --rhs;
        }
        expected_order[rhs] = slot;
    }

    if (result.evaluated_child_mask != expected_evaluated_mask ||
        result.hit_child_mask != expected_hit_mask ||
        result.candidate_count != expected_count) {
        mismatch = true;
    }
    for (unsigned index = 0; index < expected_count; ++index) {
        if (result.ordered_child_slots[index] != expected_order[index]) {
            mismatch = true;
            order_mismatch_mask |= static_cast<uint8_t>(1u << index);
        }
    }

    if (top_level) {
        ++stats->tlas_nodes;
    } else {
        ++stats->blas_nodes;
    }
    stats->evaluated_children += rtcore_v04_popcount6(expected_evaluated_mask);
    stats->hit_candidates += expected_count;
    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_KERNEL "
               "mismatch=1 level=%s raw_node=%p "
               "typed_evaluated=0x%02x legacy_evaluated=0x%02x "
               "typed_hits=0x%02x legacy_hits=0x%02x "
               "typed_count=%u legacy_count=%u "
               "node_descriptor_mismatch=%u "
               "child_descriptor_mismatch_mask=0x%02x "
               "near_t_mismatch_mask=0x%02x "
               "order_mismatch_mask=0x%02x "
               "legacy_node_visible=%u cull_mask=0x%02x "
               "typed_offset=%d legacy_offset=%d "
               "typed_node_mask=0x%02x legacy_node_mask=0x%02x "
               "typed_near=%08x,%08x,%08x,%08x,%08x,%08x "
               "legacy_near=%08x,%08x,%08x,%08x,%08x,%08x "
               "typed_order=%u,%u,%u,%u,%u,%u "
               "legacy_order=%u,%u,%u,%u,%u,%u\n",
               top_level ? "tlas" : "blas",
               static_cast<const void *>(raw_node),
               result.evaluated_child_mask, expected_evaluated_mask,
               result.hit_child_mask, expected_hit_mask,
               result.candidate_count, expected_count,
               node_descriptor_mismatch ? 1u : 0u,
               child_descriptor_mismatch_mask, near_t_mismatch_mask,
               order_mismatch_mask, legacy_node_visible ? 1u : 0u,
               static_cast<unsigned>(cull_mask & 0xffu),
               result.child_offset_blocks,
               legacy_node.ChildOffset, result.node_ray_mask,
               static_cast<unsigned>(legacy_node.NodeRayMask),
               result.near_t_bits[0], result.near_t_bits[1],
               result.near_t_bits[2], result.near_t_bits[3],
               result.near_t_bits[4], result.near_t_bits[5],
               legacy_near_t_bits[0], legacy_near_t_bits[1],
               legacy_near_t_bits[2], legacy_near_t_bits[3],
               legacy_near_t_bits[4], legacy_near_t_bits[5],
               result.ordered_child_slots[0],
               result.ordered_child_slots[1],
               result.ordered_child_slots[2],
               result.ordered_child_slots[3],
               result.ordered_child_slots[4],
               result.ordered_child_slots[5], expected_order[0],
               expected_order[1], expected_order[2], expected_order[3],
               expected_order[4], expected_order[5]);
        fflush(stdout);
        abort();
    }
}

struct rtcore_v04_typed_node_child_route_stats {
    unsigned seeded_roots;
    unsigned tlas_nodes;
    unsigned blas_nodes;
    unsigned evaluated_references;
    unsigned selected_routes;
    unsigned miss_routes;
    unsigned frontier_items;
    unsigned mismatches;

    rtcore_v04_typed_node_child_route_stats()
        : seeded_roots(0), tlas_nodes(0), blas_nodes(0),
          evaluated_references(0),
          selected_routes(0), miss_routes(0), frontier_items(0),
          mismatches(0) {}
};

struct rtcore_v04_typed_node_current_reference {
    rtcore::v04::typed_blas::as_decode_context_v0 decode_context;
    uint64_t payload_offset;
};

typedef std::map<uint64_t, rtcore_v04_typed_node_current_reference>
    rtcore_v04_typed_node_reference_tracker;

static rtcore::v04::typed_blas::as_decode_context_v0
rtcore_v04_typed_node_tlas_context(
    const rtcore_tlas_binding_snapshot &binding)
{
    namespace typed_blas = rtcore::v04::typed_blas;
    typed_blas::as_decode_context_v0 context = {};
    context.bvh_format_profile_id = typed_blas::kGenRtDerivedProfileId;
    context.as_object.object_id = binding.object_id;
    context.as_object.generation = binding.generation;
    context.as_object.as_type = 1;
    context.device_base = binding.device_base_address;
    context.device_range_bytes = binding.size_bytes;
    return context;
}

static bool rtcore_v04_typed_node_child_item_matches(
    const rtcore::v04::typed_node::compact_child_work_item_v0 &item,
    uint64_t payload_offset, uint16_t payload_bytes, uint8_t payload_kind,
    uint8_t child_slot, uint32_t near_t_bits)
{
    return item.payload_offset == payload_offset &&
           item.payload_byte_count == payload_bytes &&
           item.payload_kind == payload_kind &&
           item.child_slot == child_slot && item.near_t_bits == near_t_bits;
}

static bool rtcore_v04_typed_node_reference_matches(
    const rtcore_v04_typed_node_current_reference &reference,
    const rtcore::v04::typed_blas::as_decode_context_v0 &decode_context,
    uint64_t payload_offset)
{
    return reference.payload_offset == payload_offset &&
           memcmp(&reference.decode_context, &decode_context,
                  sizeof(decode_context)) == 0;
}

static bool rtcore_v04_publish_typed_node_reference(
    rtcore_v04_typed_node_reference_tracker *tracker,
    uint64_t comparison_host_address,
    const rtcore::v04::typed_blas::as_decode_context_v0 &decode_context,
    uint64_t payload_offset)
{
    if (tracker == NULL || comparison_host_address == 0) return false;
    rtcore_v04_typed_node_reference_tracker::const_iterator existing =
        tracker->find(comparison_host_address);
    if (existing != tracker->end()) {
        return rtcore_v04_typed_node_reference_matches(
            existing->second, decode_context, payload_offset);
    }
    rtcore_v04_typed_node_current_reference reference = {};
    reference.decode_context = decode_context;
    reference.payload_offset = payload_offset;
    (*tracker)[comparison_host_address] = reference;
    return true;
}

static bool rtcore_v04_checked_add_u64(uint64_t lhs, uint64_t rhs,
                                       uint64_t *result)
{
    if (result == NULL || lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

static void rtcore_v04_seed_typed_tlas_root_reference(
    const uint8_t *raw_tlas_header,
    const rtcore_tlas_binding_snapshot &binding,
    uint64_t comparison_root_offset,
    rtcore_v04_typed_node_reference_tracker *reference_tracker,
    rtcore_v04_typed_node_child_route_stats *stats)
{
    namespace typed_blas = rtcore::v04::typed_blas;
    namespace typed_node = rtcore::v04::typed_node;
    assert(raw_tlas_header != NULL);
    assert(reference_tracker != NULL);
    assert(stats != NULL);

    typed_node::root_reference_seed_input_v0 input = {};
    input.decode_context = rtcore_v04_typed_node_tlas_context(binding);
    if (!typed_blas::make_raw_bvh_header(
            raw_tlas_header, binding.size_bytes, &input.raw_header)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "root_seed_failure=1 reason=raw_header_capture\n");
        fflush(stdout);
        abort();
    }
    const typed_node::root_reference_seed_result_v0 result =
        typed_node::execute_root_reference_seed(input);
    uint64_t comparison_root_host = 0;
    if (result.status != typed_node::kStatusOk ||
        result.root_payload_offset != comparison_root_offset ||
        !rtcore_v04_checked_add_u64(
            reinterpret_cast<uint64_t>(raw_tlas_header),
            result.root_payload_offset, &comparison_root_host) ||
        !rtcore_v04_publish_typed_node_reference(
            reference_tracker, comparison_root_host,
            input.decode_context, result.root_payload_offset)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "root_seed_failure=1 status=%s typed_offset=0x%llx "
               "comparison_offset=0x%llx\n",
               typed_node::status_name(
                   static_cast<typed_node::status_kind>(result.status)),
               static_cast<unsigned long long>(result.root_payload_offset),
               static_cast<unsigned long long>(comparison_root_offset));
        fflush(stdout);
        abort();
    }
    ++stats->seeded_roots;
}

static bool rtcore_v04_apply_signed_byte_offset(uint64_t base, int64_t delta,
                                                 uint64_t *result)
{
    if (result == NULL) return false;
    if (delta >= 0) {
        const uint64_t magnitude = static_cast<uint64_t>(delta);
        if (base > std::numeric_limits<uint64_t>::max() - magnitude) {
            return false;
        }
        *result = base + magnitude;
        return true;
    }

    const uint64_t magnitude =
        static_cast<uint64_t>(-(delta + 1)) + uint64_t{1};
    if (base < magnitude) return false;
    *result = base - magnitude;
    return true;
}

static bool rtcore_v04_apply_signed_block_offset(uint64_t base,
                                                  int32_t blocks,
                                                  uint64_t *result)
{
    return rtcore_v04_apply_signed_byte_offset(
        base, static_cast<int64_t>(blocks) * int64_t{64}, result);
}

struct rtcore_v04_typed_stack_push_remainder_stats {
    unsigned operations;
    unsigned input_frontier_items;
    unsigned written_items;
    unsigned pruned_items;
    unsigned mismatches;

    rtcore_v04_typed_stack_push_remainder_stats()
        : operations(0), input_frontier_items(0), written_items(0),
          pruned_items(0), mismatches(0) {}
};

struct rtcore_v04_typed_stack_pop_next_stats {
    unsigned operations;
    unsigned selected_items;
    unsigned pruned_items;
    unsigned popped_items;
    unsigned mismatches;

    rtcore_v04_typed_stack_pop_next_stats()
        : operations(0), selected_items(0), pruned_items(0),
          popped_items(0), mismatches(0) {}
};

struct rtcore_v04_private_frontier_owner_layout_stats {
    unsigned initialized_slots;
    unsigned append_operations;
    unsigned pop_operations;
    unsigned read_plans;
    unsigned write_plans;
    unsigned planned_chunks;
    unsigned metadata_chunks;
    unsigned entry_chunks;
    unsigned multi_chunk_entry_plans;
    unsigned mismatches;

    rtcore_v04_private_frontier_owner_layout_stats()
        : initialized_slots(0), append_operations(0), pop_operations(0),
          read_plans(0), write_plans(0), planned_chunks(0),
          metadata_chunks(0), entry_chunks(0),
          multi_chunk_entry_plans(0), mismatches(0) {}
};

static bool rtcore_v04_record_private_frontier_plan(
    const rtcore::v04::private_frontier::access_plan_v0 &plan,
    const rtcore::v04::private_frontier::owner_binding_v0 &owner,
    bool is_read,
    rtcore_v04_private_frontier_owner_layout_stats *stats)
{
    namespace private_frontier = rtcore::v04::private_frontier;
    assert(stats != NULL);
    if (!private_frontier::owners_equal(plan.owner, owner) ||
        plan.access_count == 0 ||
        plan.access_count > private_frontier::kMaxAccessChunks) {
        return false;
    }

    unsigned entry_chunks = 0;
    for (unsigned index = 0; index < plan.access_count; ++index) {
        const private_frontier::shared_chunk_access_v0 &chunk =
            plan.accesses[index];
        if ((chunk.aligned_32b_address %
             private_frontier::kSharedAccessChunkBytes) != 0 ||
            chunk.byte_mask == 0 || chunk.byte_count == 0 ||
            chunk.byte_count > private_frontier::kSharedAccessChunkBytes ||
            chunk.access_kind !=
                (is_read ? private_frontier::kAccessRead
                         : private_frontier::kAccessWrite) ||
            (chunk.field_kind != private_frontier::kFieldFrontierMetadata &&
             chunk.field_kind != private_frontier::kFieldFrontierEntry)) {
            return false;
        }
        ++stats->planned_chunks;
        if (chunk.field_kind == private_frontier::kFieldFrontierMetadata) {
            ++stats->metadata_chunks;
        } else {
            ++stats->entry_chunks;
            ++entry_chunks;
        }
    }
    if (entry_chunks > 1) ++stats->multi_chunk_entry_plans;
    if (is_read) {
        ++stats->read_plans;
    } else {
        ++stats->write_plans;
    }
    return true;
}

static void rtcore_v04_observe_typed_stack_pop_next(
    const rtcore::v04::typed_stack::push_result_v0 &push_result,
    float current_traversal_bound,
    rtcore_v04_typed_stack_pop_next_stats *stats)
{
    namespace typed_stack = rtcore::v04::typed_stack;
    assert(stats != NULL);
    assert(push_result.status == typed_stack::kStatusOk);
    assert(push_result.result_kind == typed_stack::kStackPushedAndSelected);
    assert(push_result.frontier_delta.append_base_index == 0);
    assert(push_result.frontier_delta.new_frontier_top ==
           push_result.frontier_delta.write_count);
    assert(push_result.frontier_delta.new_frontier_count ==
           push_result.frontier_delta.write_count);

    typed_stack::frontier_metadata_v0 frontier = {};
    frontier.frontier_top = push_result.frontier_delta.new_frontier_top;
    frontier.frontier_count = push_result.frontier_delta.new_frontier_count;
    frontier.frontier_capacity = typed_stack::kMaxRemainderChildren;
    while (frontier.frontier_top != 0) {
        const uint32_t expected_index = frontier.frontier_top - 1;
        const rtcore::v04::typed_node::compact_child_work_item_v0 &expected =
            push_result.frontier_delta.written_items[expected_index];

        typed_stack::pop_input_v0 input = {};
        input.profile_id = typed_stack::kGenRtDerivedProfileId;
        input.operation_kind = typed_stack::kPopNext;
        input.has_top_entry = 1;
        input.frontier = frontier;
        input.current_traversal_bound_bits =
            rtcore_v04_fp32_bits(current_traversal_bound);
        input.top_entry = expected;
        input.current_decode_context =
            push_result.selected_fetch.decode_context;

        const typed_stack::pop_result_v0 result =
            typed_stack::execute_pop(input);
        const uint8_t expected_valid =
            static_cast<uint8_t>(typed_stack::kFrontierDeltaValid |
                                 typed_stack::kSelectedFetchValid);
        const bool mismatch =
            result.status != typed_stack::kStatusOk ||
            result.result_kind != typed_stack::kStackSelectedNext ||
            result.output_valid_mask != expected_valid ||
            result.frontier_delta.action !=
                typed_stack::kFrontierActionPopChild ||
            result.frontier_delta.pop_count != 1 ||
            result.frontier_delta.popped_index != expected_index ||
            result.frontier_delta.new_frontier_top != expected_index ||
            result.frontier_delta.new_frontier_count != expected_index ||
            memcmp(&result.selected_fetch.child, &expected,
                   sizeof(expected)) != 0 ||
            memcmp(&result.selected_fetch.decode_context,
                   &input.current_decode_context,
                   sizeof(input.current_decode_context)) != 0;

        ++stats->operations;
        ++stats->popped_items;
        if (result.result_kind == typed_stack::kStackSelectedNext) {
            ++stats->selected_items;
        } else if (result.result_kind ==
                   typed_stack::kStackPrunedRetryPop) {
            ++stats->pruned_items;
        }
        if (mismatch) {
            ++stats->mismatches;
            printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_STACK_POP_NEXT "
                   "mismatch=1 status=%s expected_index=%u new_top=%u "
                   "new_count=%u result_kind=%u\n",
                   typed_stack::status_name(
                       static_cast<typed_stack::status_kind>(result.status)),
                   expected_index, result.frontier_delta.new_frontier_top,
                   result.frontier_delta.new_frontier_count,
                   result.result_kind);
            fflush(stdout);
            abort();
        }
        frontier.frontier_top = result.frontier_delta.new_frontier_top;
        frontier.frontier_count = result.frontier_delta.new_frontier_count;
    }
}

static void rtcore_v04_observe_private_frontier_owner_layout(
    const rtcore::v04::typed_stack::push_result_v0 &push_result,
    float current_traversal_bound,
    rtcore_v04_private_frontier_owner_layout_stats *stats)
{
    namespace private_frontier = rtcore::v04::private_frontier;
    namespace typed_stack = rtcore::v04::typed_stack;
    assert(stats != NULL);
    assert(push_result.status == typed_stack::kStatusOk);
    assert(push_result.result_kind == typed_stack::kStackPushedAndSelected);

    private_frontier::owner_binding_v0 owner = {};
    owner.request_identity = stats->initialized_slots + 1;
    owner.generation = 1;
    owner.resident_warp_id =
        (owner.request_identity - 1) / 32;
    owner.lane_id = static_cast<uint8_t>(
        (owner.request_identity - 1) % 32);
    owner.private_slot_id = owner.lane_id;

    private_frontier::region_binding_v0 region = {};
    region.profile_id = private_frontier::kLayoutProfileId;
    region.slot_count = 32;
    region.private_region_base = 0x100000;

    private_frontier::frontier_metadata_image_v0 metadata = {};
    metadata.frontier_capacity =
        private_frontier::kFrontierEntryCapacity;
    metadata.current_level =
        push_result.selected_fetch.decode_context.as_object.as_type == 1
            ? 0u
            : 1u;
    metadata.level_frame_depth = metadata.current_level;
    metadata.max_level_depth = 1;

    private_frontier::shadow_slot_v0 slot = {};
    private_frontier::access_plan_v0 plan = {};
    private_frontier::status_kind status =
        private_frontier::initialize_shadow_slot(
            &slot, owner, region, metadata, &plan);
    bool mismatch = status != private_frontier::kStatusOk ||
        !rtcore_v04_record_private_frontier_plan(
            plan, owner, false, stats);
    if (!mismatch) {
        ++stats->initialized_slots;
        status = private_frontier::apply_append_delta(
            &slot, owner, region, push_result.frontier_delta, &plan);
        mismatch = status != private_frontier::kStatusOk ||
            !rtcore_v04_record_private_frontier_plan(
                plan, owner, false, stats);
    }
    if (!mismatch) ++stats->append_operations;

    unsigned observed_pops = 0;
    while (!mismatch) {
        status = private_frontier::decode_metadata(slot, owner, &metadata);
        if (status != private_frontier::kStatusOk) {
            mismatch = true;
            break;
        }
        if (metadata.frontier_top == 0) break;

        rtcore::v04::typed_node::compact_child_work_item_v0 top_entry = {};
        uint32_t top_index = 0;
        status = private_frontier::read_top_entry(
            slot, owner, region, &top_entry, &top_index, &plan);
        if (status != private_frontier::kStatusOk ||
            !rtcore_v04_record_private_frontier_plan(
                plan, owner, true, stats) ||
            top_index >= push_result.frontier_delta.write_count ||
            memcmp(&top_entry,
                   &push_result.frontier_delta.written_items[top_index],
                   sizeof(top_entry)) != 0) {
            mismatch = true;
            break;
        }

        typed_stack::pop_input_v0 pop_input = {};
        pop_input.profile_id = typed_stack::kGenRtDerivedProfileId;
        pop_input.operation_kind = typed_stack::kPopNext;
        pop_input.has_top_entry = 1;
        pop_input.frontier.frontier_top = metadata.frontier_top;
        pop_input.frontier.frontier_count = metadata.frontier_count;
        pop_input.frontier.frontier_capacity = metadata.frontier_capacity;
        pop_input.current_traversal_bound_bits =
            rtcore_v04_fp32_bits(current_traversal_bound);
        pop_input.top_entry = top_entry;
        pop_input.current_decode_context =
            push_result.selected_fetch.decode_context;
        const typed_stack::pop_result_v0 pop_result =
            typed_stack::execute_pop(pop_input);
        if (pop_result.status != typed_stack::kStatusOk ||
            pop_result.result_kind != typed_stack::kStackSelectedNext) {
            mismatch = true;
            break;
        }

        status = private_frontier::apply_pop_delta(
            &slot, owner, region, pop_result.frontier_delta, &plan);
        if (status != private_frontier::kStatusOk ||
            !rtcore_v04_record_private_frontier_plan(
                plan, owner, false, stats)) {
            mismatch = true;
            break;
        }

        rtcore::v04::typed_node::compact_child_work_item_v0 stale_entry = {};
        if (private_frontier::decode_entry(
                slot, owner, top_index, &stale_entry) !=
                private_frontier::kStatusOk ||
            memcmp(&stale_entry, &top_entry, sizeof(top_entry)) != 0) {
            mismatch = true;
            break;
        }
        ++stats->pop_operations;
        ++observed_pops;
    }

    if (!mismatch) {
        status = private_frontier::decode_metadata(slot, owner, &metadata);
        mismatch = status != private_frontier::kStatusOk ||
            metadata.frontier_top != 0 ||
            metadata.frontier_count != 0 ||
            observed_pops != push_result.frontier_delta.write_count;
    }
    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_PRIVATE_FRONTIER_OWNER_LAYOUT "
               "mismatch=1 status=%s request_identity=%u "
               "written_items=%u observed_pops=%u\n",
               private_frontier::status_name(status),
               owner.request_identity,
               push_result.frontier_delta.write_count, observed_pops);
        fflush(stdout);
        abort();
    }
}

static void rtcore_v04_observe_typed_stack_push_remainder(
    const rtcore::v04::typed_node::route_result_v0 &node_route,
    float current_traversal_bound,
    rtcore_v04_typed_stack_push_remainder_stats *stats,
    rtcore_v04_typed_stack_pop_next_stats *pop_stats,
    rtcore_v04_private_frontier_owner_layout_stats *layout_stats)
{
    namespace typed_node = rtcore::v04::typed_node;
    namespace typed_stack = rtcore::v04::typed_stack;
    assert(stats != NULL);
    assert(node_route.result_kind == typed_node::kRouteResultSelected);
    assert(node_route.frontier_count != 0);

    typed_stack::push_input_v0 input = {};
    input.profile_id = typed_stack::kGenRtDerivedProfileId;
    input.operation_kind =
        typed_stack::kPushRemainderAndForwardSelected;
    input.frontier.frontier_capacity =
        typed_stack::kMaxRemainderChildren;
    input.current_traversal_bound_bits =
        rtcore_v04_fp32_bits(current_traversal_bound);
    input.node_route = node_route;

    const typed_stack::push_result_v0 result =
        typed_stack::execute_push(input);
    const uint8_t expected_valid =
        static_cast<uint8_t>(typed_stack::kFrontierDeltaValid |
                             typed_stack::kSelectedFetchValid);
    bool mismatch =
        result.status != typed_stack::kStatusOk ||
        result.result_kind != typed_stack::kStackPushedAndSelected ||
        result.output_valid_mask != expected_valid ||
        result.pruned_count != 0 ||
        result.frontier_delta.action !=
            typed_stack::kFrontierActionAppendChildren ||
        result.frontier_delta.write_count != node_route.frontier_count ||
        result.frontier_delta.append_base_index != 0 ||
        result.frontier_delta.new_frontier_top !=
            node_route.frontier_count ||
        result.frontier_delta.new_frontier_count !=
            node_route.frontier_count ||
        memcmp(&result.selected_fetch, &node_route.selected_fetch,
               sizeof(result.selected_fetch)) != 0;
    for (unsigned index = 0; index < node_route.frontier_count; ++index) {
        const unsigned source_index = node_route.frontier_count - index - 1;
        mismatch = mismatch ||
            memcmp(&result.frontier_delta.written_items[index],
                   &node_route.frontier[source_index],
                   sizeof(node_route.frontier[source_index])) != 0;
    }

    ++stats->operations;
    stats->input_frontier_items += node_route.frontier_count;
    stats->written_items += result.frontier_delta.write_count;
    stats->pruned_items += result.pruned_count;
    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_STACK_PUSH_REMAINDER "
               "mismatch=1 status=%s input_frontier=%u written=%u "
               "pruned=%u new_top=%u new_count=%u\n",
               typed_stack::status_name(
                   static_cast<typed_stack::status_kind>(result.status)),
               node_route.frontier_count,
               result.frontier_delta.write_count, result.pruned_count,
               result.frontier_delta.new_frontier_top,
               result.frontier_delta.new_frontier_count);
        fflush(stdout);
        abort();
    }
    if (pop_stats != NULL) {
        rtcore_v04_observe_typed_stack_pop_next(
            result, current_traversal_bound, pop_stats);
    }
    if (layout_stats != NULL) {
        rtcore_v04_observe_private_frontier_owner_layout(
            result, current_traversal_bound, layout_stats);
    }
}

static void rtcore_v04_observe_typed_node_child_route(
    const uint8_t *raw_node, const GEN_RT_BVH_INTERNAL_NODE &legacy_node,
    const float3 &origin, const float3 &direction, float t_min, float t_max,
    float committed_t, uint32_t ray_flags, uint32_t cull_mask, bool top_level,
    const bool legacy_child_hit[6], const float legacy_near_t[6],
    uint64_t comparison_host_as_base,
    rtcore_v04_typed_node_reference_tracker *reference_tracker,
    rtcore_v04_typed_node_child_route_stats *stats,
    rtcore_v04_typed_stack_push_remainder_stats *stack_push_stats,
    rtcore_v04_typed_stack_pop_next_stats *stack_pop_stats,
    rtcore_v04_private_frontier_owner_layout_stats *frontier_layout_stats)
{
    namespace typed_node = rtcore::v04::typed_node;
    assert(raw_node != NULL);
    assert(reference_tracker != NULL);
    assert(stats != NULL);

    const uint64_t raw_node_address = reinterpret_cast<uint64_t>(raw_node);
    rtcore_v04_typed_node_reference_tracker::const_iterator current =
        reference_tracker->find(raw_node_address);
    if (current == reference_tracker->end()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "adapter_failure=1 reason=typed_current_reference_missing "
               "level=%s\n",
               top_level ? "tlas" : "blas");
        fflush(stdout);
        abort();
    }

    typed_node::route_input_v0 input = {};
    if (!rtcore_v04_make_typed_node_candidate_input(
            raw_node, origin, direction, t_min, t_max, committed_t,
            ray_flags, cull_mask, top_level, &input.candidate)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "adapter_failure=1 reason=candidate_input level=%s\n",
               top_level ? "tlas" : "blas");
        fflush(stdout);
        abort();
    }
    input.decode_context = current->second.decode_context;
    input.current_payload_offset = current->second.payload_offset;

    const typed_node::route_result_v0 result =
        typed_node::execute_route(input);
    if (result.status != typed_node::kStatusOk) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "kernel_failure=1 level=%s status=%s current_offset=0x%llx\n",
               top_level ? "tlas" : "blas",
               typed_node::status_name(
                   static_cast<typed_node::status_kind>(result.status)),
               static_cast<unsigned long long>(input.current_payload_offset));
        fflush(stdout);
        abort();
    }
    if (stack_push_stats != NULL && result.frontier_count != 0) {
        rtcore_v04_observe_typed_stack_push_remainder(
            result, committed_t, stack_push_stats, stack_pop_stats,
            frontier_layout_stats);
    }

    typed_node::compact_child_work_item_v0 expected_items[6] = {};
    uint64_t comparison_child_host[6] = {};
    uint8_t expected_order[6] = {};
    unsigned expected_count = 0;
    const bool node_visible =
        (legacy_node.NodeRayMask & (cull_mask & 0xffu)) != 0;
    uint64_t child_host = 0;
    bool expected_address_invalid = !rtcore_v04_apply_signed_block_offset(
        raw_node_address, legacy_node.ChildOffset,
        &child_host);
    if (raw_node_address < comparison_host_as_base ||
        input.current_payload_offset !=
            raw_node_address - comparison_host_as_base) {
        expected_address_invalid = true;
    }
    for (unsigned child = 0; child < 6; ++child) {
        const uint64_t payload_bytes =
            static_cast<uint64_t>(legacy_node.ChildSize[child]) * uint64_t{64};
        if (payload_bytes == 0) continue;
        ++stats->evaluated_references;
        comparison_child_host[child] = child_host;
        if (expected_address_invalid || child_host < comparison_host_as_base) {
            expected_address_invalid = true;
        } else {
            expected_items[child].payload_offset =
                child_host - comparison_host_as_base;
        }
        expected_items[child].near_t_bits =
            rtcore_v04_fp32_bits(legacy_near_t[child]);
        expected_items[child].payload_byte_count =
            static_cast<uint16_t>(payload_bytes);
        expected_items[child].payload_kind =
            static_cast<uint8_t>(legacy_node.ChildType[child]);
        expected_items[child].child_slot = static_cast<uint8_t>(child);
        if (legacy_child_hit[child] && node_visible) {
            expected_order[expected_count++] = static_cast<uint8_t>(child);
        }
        if (child_host >
            std::numeric_limits<uint64_t>::max() - payload_bytes) {
            expected_address_invalid = true;
        } else {
            child_host += payload_bytes;
        }
    }
    for (unsigned lhs = 1; lhs < expected_count; ++lhs) {
        const uint8_t slot = expected_order[lhs];
        unsigned rhs = lhs;
        while (rhs > 0) {
            const uint8_t previous = expected_order[rhs - 1];
            if (legacy_near_t[previous] < legacy_near_t[slot] ||
                (legacy_near_t[previous] == legacy_near_t[slot] &&
                 previous < slot)) {
                break;
            }
            expected_order[rhs] = previous;
            --rhs;
        }
        expected_order[rhs] = slot;
    }

    bool mismatch = expected_address_invalid;
    if (expected_count == 0) {
        ++stats->miss_routes;
        mismatch = mismatch || result.result_kind != typed_node::kRouteResultMiss ||
                   result.output_valid_mask != 0 || result.frontier_count != 0;
    } else {
        ++stats->selected_routes;
        stats->frontier_items += expected_count - 1;
        const uint8_t selected = expected_order[0];
        const uint8_t expected_valid =
            expected_count == 1
                ? typed_node::kSelectedFetchValid
                : static_cast<uint8_t>(typed_node::kSelectedFetchValid |
                                       typed_node::kFrontierItemsValid);
        mismatch = mismatch ||
            result.result_kind != typed_node::kRouteResultSelected ||
            result.output_valid_mask != expected_valid ||
            result.frontier_count != expected_count - 1 ||
            !rtcore_v04_typed_node_child_item_matches(
                result.selected_fetch.child,
                expected_items[selected].payload_offset,
                expected_items[selected].payload_byte_count,
                expected_items[selected].payload_kind,
                expected_items[selected].child_slot,
                expected_items[selected].near_t_bits) ||
            memcmp(&result.selected_fetch.decode_context,
                   &input.decode_context,
                   sizeof(input.decode_context)) != 0;
        for (unsigned index = 0; index + 1 < expected_count; ++index) {
            const uint8_t slot = expected_order[index + 1];
            mismatch = mismatch || !rtcore_v04_typed_node_child_item_matches(
                result.frontier[index], expected_items[slot].payload_offset,
                expected_items[slot].payload_byte_count,
                expected_items[slot].payload_kind,
                expected_items[slot].child_slot,
                expected_items[slot].near_t_bits);
        }
    }

    if (top_level) {
        ++stats->tlas_nodes;
    } else {
        ++stats->blas_nodes;
    }
    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "mismatch=1 level=%s current_offset=0x%llx typed_kind=%u "
               "typed_valid=0x%x typed_frontier=%u expected_count=%u\n",
               top_level ? "tlas" : "blas",
               static_cast<unsigned long long>(input.current_payload_offset),
               result.result_kind, result.output_valid_mask,
               result.frontier_count, expected_count);
        fflush(stdout);
        abort();
    }

    if (result.result_kind == typed_node::kRouteResultSelected) {
        if (!rtcore_v04_publish_typed_node_reference(
                reference_tracker,
                comparison_child_host[result.selected_fetch.child.child_slot],
                result.selected_fetch.decode_context,
                result.selected_fetch.child.payload_offset)) {
            printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
                   "adapter_failure=1 reason=selected_reference_conflict "
                   "level=%s\n",
                   top_level ? "tlas" : "blas");
            fflush(stdout);
            abort();
        }
        for (unsigned index = 0; index < result.frontier_count; ++index) {
            const typed_node::compact_child_work_item_v0 &item =
                result.frontier[index];
            if (!rtcore_v04_publish_typed_node_reference(
                    reference_tracker,
                    comparison_child_host[item.child_slot],
                    input.decode_context, item.payload_offset)) {
                printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
                       "adapter_failure=1 reason=frontier_reference_conflict "
                       "level=%s slot=%u\n",
                       top_level ? "tlas" : "blas", item.child_slot);
                fflush(stdout);
                abort();
            }
        }
    }
}

struct rtcore_v04_typed_primitive_candidate_stats {
    unsigned leaves;
    unsigned geometric_hits;
    unsigned candidate_hits;
    unsigned mismatches;

    rtcore_v04_typed_primitive_candidate_stats()
        : leaves(0), geometric_hits(0), candidate_hits(0), mismatches(0) {}
};

static float rtcore_v04_typed_primitive_fp32_value(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool rtcore_v04_typed_primitive_bary_matches(float typed,
                                                     float legacy)
{
    static const float kLegacyBarycentricAbsoluteTolerance = 2.0e-5f;
    return std::isfinite(typed) && std::isfinite(legacy) &&
           fabsf(typed - legacy) <= kLegacyBarycentricAbsoluteTolerance;
}

static void rtcore_v04_observe_typed_primitive_candidate(
    const uint8_t *raw_leaf, const GEN_RT_BVH_QUAD_LEAF &legacy_leaf,
    const Ray &object_ray, float world_to_object_t_multiplier,
    float world_t_min, float world_t_max, float committed_world_t,
    uint32_t instance_flags, bool legacy_hit, float legacy_object_t,
    bool legacy_counter_clockwise,
    bool legacy_front_facing, uint32_t legacy_hit_kind,
    const float3 &legacy_barycentric,
    rtcore_v04_typed_primitive_candidate_stats *stats)
{
    namespace typed_primitive = rtcore::v04::typed_primitive;
    assert(raw_leaf != NULL);
    assert(stats != NULL);

    typed_primitive::candidate_input_v0 input = {};
    input.profile_id = typed_primitive::kGenRtDerivedProfileId;
    input.instance_flags = instance_flags;
    input.object_ray.origin[0] = object_ray.get_origin().x;
    input.object_ray.origin[1] = object_ray.get_origin().y;
    input.object_ray.origin[2] = object_ray.get_origin().z;
    input.object_ray.direction[0] = object_ray.get_direction().x;
    input.object_ray.direction[1] = object_ray.get_direction().y;
    input.object_ray.direction[2] = object_ray.get_direction().z;
    input.object_ray.t_min = object_ray.get_tmin();
    input.object_ray.t_max = object_ray.get_tmax();
    input.world_to_object_t_multiplier = world_to_object_t_multiplier;
    input.committed_world_t = committed_world_t;
    input.world_t_min = world_t_min;
    input.world_t_max = world_t_max;
    if (!typed_primitive::make_raw_primitive_payload(
            raw_leaf, &input.raw_primitive)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PRIMITIVE_KERNEL "
               "adapter_failure=1 raw_leaf=%p\n",
               static_cast<const void *>(raw_leaf));
        fflush(stdout);
        abort();
    }

    const typed_primitive::candidate_result_v0 result =
        typed_primitive::execute(input);
    if (result.status != typed_primitive::kStatusOk) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PRIMITIVE_KERNEL "
               "kernel_failure=1 status=%s raw_leaf=%p raw_control=0x%08x\n",
               typed_primitive::status_name(
                   static_cast<typed_primitive::status_kind>(result.status)),
               static_cast<const void *>(raw_leaf), result.raw_quad_control);
        fflush(stdout);
        abort();
    }

    const float legacy_world_t =
        legacy_hit ? legacy_object_t / world_to_object_t_multiplier : 0.0f;
    const bool legacy_candidate =
        legacy_hit && world_t_min <= legacy_world_t &&
        legacy_world_t <= world_t_max &&
        legacy_world_t < committed_world_t;
    const bool descriptor_mismatch =
        result.shader_index != legacy_leaf.LeafDescriptor.ShaderIndex ||
        result.geometry_ray_mask !=
            legacy_leaf.LeafDescriptor.GeometryRayMask ||
        result.geometry_index != legacy_leaf.LeafDescriptor.GeometryIndex ||
        result.geometry_flags != legacy_leaf.LeafDescriptor.GeometryFlags ||
        result.primitive_index != legacy_leaf.PrimitiveIndex0;
    bool t_mismatch = false;
    bool facing_mismatch = false;
    bool barycentric_mismatch = false;
    if (legacy_hit) {
        t_mismatch =
            result.object_t_bits != rtcore_v04_fp32_bits(legacy_object_t) ||
            result.world_t_bits != rtcore_v04_fp32_bits(legacy_world_t);
        facing_mismatch =
            result.counter_clockwise_facing !=
                (legacy_counter_clockwise ? 1u : 0u) ||
            result.front_facing != (legacy_front_facing ? 1u : 0u) ||
            result.hit_kind != legacy_hit_kind;
        const float typed_bary1 = rtcore_v04_typed_primitive_fp32_value(
            result.bary_vertex1_bits);
        const float typed_bary2 = rtcore_v04_typed_primitive_fp32_value(
            result.bary_vertex2_bits);
        barycentric_mismatch =
            !rtcore_v04_typed_primitive_bary_matches(
                typed_bary1, legacy_barycentric.x) ||
            !rtcore_v04_typed_primitive_bary_matches(
                typed_bary2, legacy_barycentric.y);
    }
    const bool mismatch =
        descriptor_mismatch ||
        result.geometric_hit != (legacy_hit ? 1u : 0u) ||
        result.candidate_hit != (legacy_candidate ? 1u : 0u) || t_mismatch ||
        facing_mismatch || barycentric_mismatch;

    ++stats->leaves;
    stats->geometric_hits += legacy_hit ? 1u : 0u;
    stats->candidate_hits += legacy_candidate ? 1u : 0u;
    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PRIMITIVE_KERNEL "
               "mismatch=1 raw_leaf=%p descriptor_mismatch=%u "
               "typed_geometric=%u legacy_geometric=%u "
               "typed_candidate=%u legacy_candidate=%u "
               "t_mismatch=%u facing_mismatch=%u bary_mismatch=%u "
               "typed_object_t=0x%08x legacy_object_t=0x%08x "
               "typed_world_t=0x%08x legacy_world_t=0x%08x "
               "typed_bary=0x%08x,0x%08x legacy_bary=0x%08x,0x%08x "
               "typed_hit_kind=0x%02x legacy_hit_kind=0x%02x\n",
               static_cast<const void *>(raw_leaf),
               descriptor_mismatch ? 1u : 0u, result.geometric_hit,
               legacy_hit ? 1u : 0u, result.candidate_hit,
               legacy_candidate ? 1u : 0u, t_mismatch ? 1u : 0u,
               facing_mismatch ? 1u : 0u,
               barycentric_mismatch ? 1u : 0u, result.object_t_bits,
               rtcore_v04_fp32_bits(legacy_object_t), result.world_t_bits,
               rtcore_v04_fp32_bits(legacy_world_t),
               result.bary_vertex1_bits, result.bary_vertex2_bits,
               rtcore_v04_fp32_bits(legacy_barycentric.x),
               rtcore_v04_fp32_bits(legacy_barycentric.y), result.hit_kind,
               legacy_hit_kind);
        fflush(stdout);
        abort();
    }
}

struct rtcore_v04_typed_procedural_boundary_stats {
    unsigned leaves;
    unsigned mask_visible;
    unsigned mismatches;

    rtcore_v04_typed_procedural_boundary_stats()
        : leaves(0), mask_visible(0), mismatches(0) {}
};

static void rtcore_v04_observe_typed_procedural_boundary_seed(
    const uint8_t *raw_leaf,
    const GEN_RT_BVH_PROCEDURAL_LEAF &legacy_leaf, uint32_t cull_mask,
    rtcore_v04_typed_procedural_boundary_stats *stats)
{
    namespace typed_primitive = rtcore::v04::typed_primitive;
    assert(raw_leaf != NULL);
    assert(stats != NULL);

    typed_primitive::procedural_input_v0 input = {};
    input.profile_id = typed_primitive::kGenRtDerivedProfileId;
    input.cull_mask = cull_mask;
    if (!typed_primitive::make_raw_procedural_payload(
            raw_leaf, &input.raw_primitive)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PROCEDURAL_BOUNDARY "
               "adapter_failure=1 raw_leaf=%p\n",
               static_cast<const void *>(raw_leaf));
        fflush(stdout);
        abort();
    }

    const typed_primitive::procedural_result_v0 result =
        typed_primitive::execute_procedural(input);
    if (result.status != typed_primitive::kStatusOk) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PROCEDURAL_BOUNDARY "
               "kernel_failure=1 status=%s raw_leaf=%p raw_control=0x%08x\n",
               typed_primitive::status_name(
                   static_cast<typed_primitive::status_kind>(result.status)),
               static_cast<const void *>(raw_leaf), result.raw_control);
        fflush(stdout);
        abort();
    }

    const bool legacy_mask_visible =
        (legacy_leaf.LeafDescriptor.GeometryRayMask & cull_mask) != 0;
    const bool mismatch =
        result.shader_index != legacy_leaf.LeafDescriptor.ShaderIndex ||
        result.geometry_ray_mask !=
            legacy_leaf.LeafDescriptor.GeometryRayMask ||
        result.geometry_index != legacy_leaf.LeafDescriptor.GeometryIndex ||
        result.leaf_type != legacy_leaf.LeafDescriptor.LeafType ||
        result.geometry_flags != legacy_leaf.LeafDescriptor.GeometryFlags ||
        result.primitive_count != legacy_leaf.NumPrimitives ||
        result.last_primitive != legacy_leaf.LastPrimitive ||
        result.primitive_index != legacy_leaf.PrimitiveIndex[0] ||
        result.mask_visible != (legacy_mask_visible ? 1u : 0u);

    ++stats->leaves;
    stats->mask_visible += legacy_mask_visible ? 1u : 0u;
    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PROCEDURAL_BOUNDARY "
               "mismatch=1 raw_leaf=%p typed_shader=%u legacy_shader=%u "
               "typed_mask=0x%02x legacy_mask=0x%02x "
               "typed_geometry=%u legacy_geometry=%u "
               "typed_flags=%u legacy_flags=%u "
               "typed_count=%u legacy_count=%u "
               "typed_last=%u legacy_last=%u "
               "typed_primitive=%u legacy_primitive=%u "
               "typed_visible=%u legacy_visible=%u\n",
               static_cast<const void *>(raw_leaf), result.shader_index,
               legacy_leaf.LeafDescriptor.ShaderIndex,
               result.geometry_ray_mask,
               legacy_leaf.LeafDescriptor.GeometryRayMask,
               result.geometry_index,
               legacy_leaf.LeafDescriptor.GeometryIndex,
               result.geometry_flags,
               legacy_leaf.LeafDescriptor.GeometryFlags,
               result.primitive_count, legacy_leaf.NumPrimitives,
               result.last_primitive, legacy_leaf.LastPrimitive,
               result.primitive_index, legacy_leaf.PrimitiveIndex[0],
               result.mask_visible, legacy_mask_visible ? 1u : 0u);
        fflush(stdout);
        abort();
    }
}

struct rtcore_v04_typed_instance_boundary_stats {
    unsigned instances;
    unsigned mismatches;

    rtcore_v04_typed_instance_boundary_stats()
        : instances(0), mismatches(0) {}
};

static void rtcore_v04_observe_typed_instance_boundary_seed(
    const uint8_t *raw_leaf, const GEN_RT_BVH_INSTANCE_LEAF &legacy_leaf,
    rtcore_v04_typed_instance_boundary_stats *stats)
{
    namespace typed_instance = rtcore::v04::typed_instance;
    assert(raw_leaf != NULL);
    assert(stats != NULL);

    typed_instance::boundary_input_v0 input = {};
    input.profile_id = typed_instance::kGenRtDerivedProfileId;
    if (!typed_instance::make_raw_instance_payload(
            raw_leaf, &input.raw_instance)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_BOUNDARY "
               "adapter_failure=1 raw_leaf=%p\n",
               static_cast<const void *>(raw_leaf));
        fflush(stdout);
        abort();
    }

    const typed_instance::boundary_result_v0 result =
        typed_instance::execute(input);
    if (result.status != typed_instance::kStatusOk) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_BOUNDARY "
               "kernel_failure=1 status=%s raw_leaf=%p\n",
               typed_instance::status_name(
                   static_cast<typed_instance::status_kind>(result.status)),
               static_cast<const void *>(raw_leaf));
        fflush(stdout);
        abort();
    }

    const uint32_t legacy_world_to_object_bits[] = {
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm00),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm01),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm02),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm10),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm11),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm12),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm20),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm21),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm22),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm30),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm31),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm32),
    };
    const uint32_t legacy_object_to_world_bits[] = {
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm00),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm01),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm02),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm10),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm11),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm12),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm20),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm21),
        rtcore_v04_fp32_bits(legacy_leaf.ObjectToWorldm22),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm30),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm31),
        rtcore_v04_fp32_bits(legacy_leaf.WorldToObjectm32),
    };
    bool matrix_mismatch = false;
    for (unsigned element = 0;
         element < typed_instance::kMatrixElementCount; ++element) {
        matrix_mismatch =
            matrix_mismatch ||
            result.world_to_object_bits[element] !=
                legacy_world_to_object_bits[element] ||
            result.object_to_world_bits[element] !=
                legacy_object_to_world_bits[element];
    }

    const bool mismatch =
        result.shader_index != legacy_leaf.ShaderIndex ||
        result.geometry_ray_mask != legacy_leaf.GeometryRayMask ||
        result.instance_sbt_contribution !=
            legacy_leaf.InstanceContributionToHitGroupIndex ||
        result.leaf_type != legacy_leaf.LeafType ||
        result.geometry_flags != legacy_leaf.GeometryFlags ||
        result.start_node_address != legacy_leaf.StartNodeAddress ||
        result.instance_flags != legacy_leaf.InstanceFlags ||
        result.bvh_address != legacy_leaf.BVHAddress ||
        result.instance_custom_index != legacy_leaf.InstanceID ||
        result.instance_index != legacy_leaf.InstanceIndex || matrix_mismatch;

    ++stats->instances;
    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_BOUNDARY "
               "mismatch=1 raw_leaf=%p typed_shader=%u legacy_shader=%u "
               "typed_sbt=%u legacy_sbt=%u typed_start=0x%llx "
               "legacy_start=0x%llx typed_bvh=0x%llx legacy_bvh=0x%llx "
               "typed_custom=%u legacy_custom=%u typed_index=%u "
               "legacy_index=%u matrix_mismatch=%u\n",
               static_cast<const void *>(raw_leaf), result.shader_index,
               legacy_leaf.ShaderIndex, result.instance_sbt_contribution,
               legacy_leaf.InstanceContributionToHitGroupIndex,
               static_cast<unsigned long long>(result.start_node_address),
               static_cast<unsigned long long>(legacy_leaf.StartNodeAddress),
               static_cast<unsigned long long>(result.bvh_address),
               static_cast<unsigned long long>(legacy_leaf.BVHAddress),
               result.instance_custom_index, legacy_leaf.InstanceID,
               result.instance_index, legacy_leaf.InstanceIndex,
               matrix_mismatch ? 1u : 0u);
        fflush(stdout);
        abort();
    }
}

struct rtcore_v04_typed_blas_decode_context_stats {
    unsigned contexts;
    unsigned mismatches;
    unsigned root_descriptors;
    unsigned root_descriptor_mismatches;
    unsigned instance_references;
    unsigned instance_reference_mismatches;

    rtcore_v04_typed_blas_decode_context_stats()
        : contexts(0), mismatches(0), root_descriptors(0),
          root_descriptor_mismatches(0), instance_references(0),
          instance_reference_mismatches(0) {}
};

static bool rtcore_v04_resolve_instance_blas_reference(
    const rtcore_tlas_binding_snapshot &tlas,
    uint64_t instance_metadata_reference,
    rtcore_blas_binding_snapshot *blas,
    rtcore_v04_instance_blas_reference_snapshot *reference,
    const char **failure_reason)
{
    if (blas == NULL || reference == NULL) {
        if (failure_reason != NULL) *failure_reason = "null_output";
        return false;
    }
    *blas = rtcore_blas_binding_snapshot();
    *reference = rtcore_v04_instance_blas_reference_snapshot();
    const char *reason = "unvalidated";
    if (!g_rtcore_tlas_binding_registry.validate(
            tlas, instance_metadata_reference, 128, &reason) ||
        !g_rtcore_instance_blas_reference_registry.resolve(
            tlas.object_id, tlas.generation,
            instance_metadata_reference, reference, &reason) ||
        !g_rtcore_blas_binding_registry.capture_by_object_id(
            reference->blas_object_id, blas, &reason) ||
        !g_rtcore_blas_binding_registry.validate(
            *blas, 0, 0, &reason)) {
        if (failure_reason != NULL) *failure_reason = reason;
        return false;
    }
    if (blas->generation != reference->blas_generation ||
        blas->host_root_address != reference->blas_host_root_address ||
        !blas->root_descriptor_valid ||
        blas->root_build_generation == 0) {
        if (failure_reason != NULL) {
            *failure_reason = "referenced_blas_generation_mismatch";
        }
        return false;
    }
    if (failure_reason != NULL) *failure_reason = "none";
    return true;
}

struct rtcore_v04_typed_instance_enter_stats {
    unsigned observations;
    unsigned culled;
    unsigned blas_roots;
    unsigned mismatches;

    rtcore_v04_typed_instance_enter_stats()
        : observations(0), culled(0), blas_roots(0), mismatches(0) {}
};

static bool rtcore_v04_typed_instance_fp32_matches(float typed,
                                                    float legacy)
{
    static const float kAbsoluteTolerance = 2.0e-5f;
    const float scale = fmaxf(1.0f, fmaxf(fabsf(typed), fabsf(legacy)));
    return std::isfinite(typed) && std::isfinite(legacy) &&
           fabsf(typed - legacy) <= kAbsoluteTolerance * scale;
}

static void rtcore_v04_observe_typed_instance_enter_transition(
    const uint8_t *raw_instance,
    const GEN_RT_BVH_INSTANCE_LEAF &legacy_instance,
    const rtcore_tlas_binding_snapshot &tlas_binding,
    uint64_t instance_metadata_reference, const Ray &world_ray,
    uint32_t ray_flags, uint32_t cull_mask,
    rtcore_v04_typed_instance_enter_stats *stats,
    rtcore::v04::typed_instance::enter_result_v0 *observed_result)
{
    namespace typed_blas = rtcore::v04::typed_blas;
    namespace typed_instance = rtcore::v04::typed_instance;
    assert(raw_instance != NULL);
    assert(stats != NULL);

    rtcore_blas_binding_snapshot blas_binding;
    rtcore_v04_instance_blas_reference_snapshot relation;
    const char *capture_failure = "unvalidated";
    if (!rtcore_v04_resolve_instance_blas_reference(
            tlas_binding, instance_metadata_reference, &blas_binding,
            &relation, &capture_failure)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_ENTER "
               "capture_failure=1 reason=%s metadata_ref=0x%llx\n",
               capture_failure,
               static_cast<unsigned long long>(
                   instance_metadata_reference));
        fflush(stdout);
        abort();
    }
    if ((cull_mask & ~0xffu) != 0) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_ENTER "
               "adapter_failure=1 reason=cull_mask_width cull_mask=0x%x\n",
               cull_mask);
        fflush(stdout);
        abort();
    }

    typed_instance::enter_input_v0 input = {};
    input.profile_id = typed_instance::kGenRtDerivedProfileId;
    input.current_level = typed_instance::kLevelTlas;
    const float world_origin[] = {world_ray.get_origin().x,
                                  world_ray.get_origin().y,
                                  world_ray.get_origin().z};
    const float world_direction[] = {world_ray.get_direction().x,
                                     world_ray.get_direction().y,
                                     world_ray.get_direction().z};
    if (!typed_instance::make_mutable_ray_state(
            world_origin, world_direction, world_ray.get_tmin(),
            world_ray.get_tmax(), &input.world_ray) ||
        !typed_instance::make_raw_instance_payload(
            raw_instance, &input.raw_instance)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_ENTER "
               "adapter_failure=1 reason=ray_or_raw_payload\n");
        fflush(stdout);
        abort();
    }
    input.policy.ray_flags = ray_flags;
    input.policy.cull_mask = static_cast<uint8_t>(cull_mask);

    input.instance_blas_reference.tlas_object_id =
        relation.tlas_object_id;
    input.instance_blas_reference.instance_metadata_reference =
        relation.instance_metadata_reference;
    input.instance_blas_reference.blas_object_id = relation.blas_object_id;
    input.instance_blas_reference.tlas_generation =
        relation.tlas_generation;
    input.instance_blas_reference.tlas_build_generation =
        relation.tlas_build_generation;
    input.instance_blas_reference.blas_generation =
        relation.blas_generation;
    input.instance_blas_reference.valid = relation.valid ? 1 : 0;

    input.tlas_decode_context.bvh_format_profile_id =
        typed_instance::kGenRtDerivedProfileId;
    input.tlas_decode_context.as_object.object_id = tlas_binding.object_id;
    input.tlas_decode_context.as_object.generation = tlas_binding.generation;
    input.tlas_decode_context.as_object.as_type =
        typed_instance::kAsTypeTlas;
    input.tlas_decode_context.device_base =
        tlas_binding.device_base_address;
    input.tlas_decode_context.device_range_bytes = tlas_binding.size_bytes;

    input.blas_decode_context.bvh_format_profile_id =
        typed_instance::kGenRtDerivedProfileId;
    input.blas_decode_context.as_object.object_id = blas_binding.object_id;
    input.blas_decode_context.as_object.generation =
        blas_binding.generation;
    input.blas_decode_context.as_object.as_type =
        typed_instance::kAsTypeBlas;
    input.blas_decode_context.device_base =
        blas_binding.device_base_address;
    input.blas_decode_context.device_range_bytes = blas_binding.size_bytes;

    input.blas_root_descriptor.object_id = blas_binding.object_id;
    input.blas_root_descriptor.root_payload_offset =
        blas_binding.root_payload_offset;
    input.blas_root_descriptor.object_generation =
        blas_binding.generation;
    input.blas_root_descriptor.build_generation =
        blas_binding.root_build_generation;
    input.blas_root_descriptor.bvh_format_profile_id =
        blas_binding.root_bvh_profile_id;
    input.blas_root_descriptor.payload_format_id =
        blas_binding.root_payload_format_id;
    input.blas_root_descriptor.as_type = typed_blas::kAsTypeBlas;
    input.blas_root_descriptor.root_payload_kind =
        blas_binding.root_payload_kind;
    input.blas_root_descriptor.valid =
        blas_binding.root_descriptor_valid ? 1 : 0;

    const typed_instance::enter_result_v0 result =
        typed_instance::execute_enter(input);
    if (result.status != typed_instance::kStatusOk) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_ENTER "
               "kernel_failure=1 status=%s metadata_ref=0x%llx\n",
               typed_instance::status_name(
                   static_cast<typed_instance::status_kind>(result.status)),
               static_cast<unsigned long long>(
                   instance_metadata_reference));
        fflush(stdout);
        abort();
    }

    ++stats->observations;
    const bool expected_visible =
        (legacy_instance.GeometryRayMask & cull_mask) != 0;
    bool mismatch = result.mask_visible != (expected_visible ? 1u : 0u);
    if (!expected_visible) {
        ++stats->culled;
        mismatch = mismatch ||
                   result.result_kind !=
                       typed_instance::kEnterResultCulled ||
                   result.output_valid_mask != 0;
    } else {
        ++stats->blas_roots;
        Ray mutable_world_ray = world_ray;
        GEN_RT_BVH_INSTANCE_LEAF legacy_instance_copy = legacy_instance;
        const float4x4 legacy_matrix = instance_leaf_matrix_to_float4x4(
            &legacy_instance_copy.WorldToObjectm00);
        float legacy_multiplier = 0.0f;
        const Ray legacy_object_ray = make_transformed_ray(
            mutable_world_ray, legacy_matrix, &legacy_multiplier);
        const float legacy_raw_direction[] = {
            legacy_object_ray.get_direction().x * legacy_multiplier,
            legacy_object_ray.get_direction().y * legacy_multiplier,
            legacy_object_ray.get_direction().z * legacy_multiplier,
        };
        const uint64_t legacy_host_blas =
            reinterpret_cast<uint64_t>(raw_instance) +
            legacy_instance.BVHAddress;
        GEN_RT_BVH legacy_header = {};
        GEN_RT_BVH_unpack(
            &legacy_header,
            reinterpret_cast<uint8_t *>(legacy_host_blas));
        const uint64_t typed_root_device =
            result.root_fetch.decode_context.device_base +
            result.root_fetch.encoded_reference;
        const uint64_t legacy_root_device =
            blas_binding.device_base_address + legacy_header.RootNodeOffset;

        mismatch = mismatch ||
            result.result_kind != typed_instance::kEnterResultBlasRoot ||
            result.output_valid_mask !=
                (typed_instance::kObjectRayValid |
                 typed_instance::kInstanceProjectionValid |
                 typed_instance::kRootFetchValid) ||
            result.instance_projection.instance_metadata_reference !=
                instance_metadata_reference ||
            result.instance_projection.shader_index !=
                legacy_instance.ShaderIndex ||
            result.instance_projection.instance_sbt_contribution !=
                legacy_instance.InstanceContributionToHitGroupIndex ||
            result.instance_projection.instance_custom_index !=
                legacy_instance.InstanceID ||
            result.instance_projection.instance_index !=
                legacy_instance.InstanceIndex ||
            result.instance_projection.instance_flags !=
                legacy_instance.InstanceFlags ||
            result.instance_projection.geometry_flags !=
                legacy_instance.GeometryFlags ||
            result.root_fetch.decode_context.as_object.object_id !=
                relation.blas_object_id ||
            result.root_fetch.decode_context.as_object.generation !=
                relation.blas_generation ||
            result.root_fetch.build_generation !=
                blas_binding.root_build_generation ||
            result.root_fetch.expected_payload_kind !=
                blas_binding.root_payload_kind ||
            typed_root_device != legacy_root_device;
        for (unsigned component = 0; component < 3; ++component) {
            const float legacy_origin = component == 0
                ? legacy_object_ray.get_origin().x
                : (component == 1 ? legacy_object_ray.get_origin().y
                                  : legacy_object_ray.get_origin().z);
            mismatch = mismatch ||
                !rtcore_v04_typed_instance_fp32_matches(
                    result.object_ray.origin[component], legacy_origin) ||
                !rtcore_v04_typed_instance_fp32_matches(
                    result.object_ray.direction[component],
                    legacy_raw_direction[component]);
        }
        mismatch = mismatch ||
            !rtcore_v04_typed_instance_fp32_matches(
                result.object_ray.t_min * legacy_multiplier,
                legacy_object_ray.get_tmin()) ||
            !rtcore_v04_typed_instance_fp32_matches(
                result.object_ray.t_max * legacy_multiplier,
                legacy_object_ray.get_tmax());
    }

    if (mismatch) {
        ++stats->mismatches;
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_ENTER "
               "mismatch=1 metadata_ref=0x%llx visible=%u kind=%u "
               "valid_mask=0x%x typed_root=0x%llx\n",
               static_cast<unsigned long long>(
                   instance_metadata_reference),
               result.mask_visible, result.result_kind,
               result.output_valid_mask,
               static_cast<unsigned long long>(
                   result.root_fetch.decode_context.device_base +
                   result.root_fetch.encoded_reference));
        fflush(stdout);
        abort();
    }
    if (observed_result != NULL) {
        *observed_result = result;
    }
}

static void rtcore_v04_observe_typed_blas_decode_context(
    const uint8_t *raw_instance,
    const GEN_RT_BVH_INSTANCE_LEAF &legacy_instance,
    const rtcore_tlas_binding_snapshot &tlas_binding,
    uint64_t instance_metadata_reference,
    rtcore_v04_typed_blas_decode_context_stats *stats)
{
    namespace typed_blas = rtcore::v04::typed_blas;
    assert(raw_instance != NULL);
    assert(stats != NULL);

    const uint64_t legacy_host_blas_header =
        reinterpret_cast<uint64_t>(raw_instance) + legacy_instance.BVHAddress;
    rtcore_blas_binding_snapshot binding;
    rtcore_v04_instance_blas_reference_snapshot instance_reference;
    const char *capture_failure = "unvalidated";
    const bool use_instance_reference =
        rtcore_v04_producer_backed_instance_blas_reference_enabled();
    const bool binding_captured = use_instance_reference
        ? rtcore_v04_resolve_instance_blas_reference(
              tlas_binding, instance_metadata_reference, &binding,
              &instance_reference, &capture_failure)
        : VulkanRayTracing::captureBlasBinding(
              legacy_host_blas_header, &binding, &capture_failure);
    if (!binding_captured) {
        if (use_instance_reference) {
            ++stats->instance_reference_mismatches;
        }
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT "
               "capture_failure=1 reason=%s host_blas=0x%llx "
               "instance_metadata_ref=0x%llx relation_authority=%u\n",
               capture_failure,
               static_cast<unsigned long long>(legacy_host_blas_header),
               static_cast<unsigned long long>(
                   instance_metadata_reference),
               use_instance_reference ? 1u : 0u);
        fflush(stdout);
        abort();
    }
    if (use_instance_reference) {
        ++stats->instance_references;
    }
    const uint64_t producer_host_blas_header = binding.host_root_address;

    typed_blas::boundary_input_v0 input = {};
    input.profile_id = typed_blas::kGenRtDerivedProfileId;
    input.binding.object_id = binding.object_id;
    input.binding.generation = binding.generation;
    input.binding.as_type = typed_blas::kAsTypeBlas;
    input.binding.device_base = binding.device_base_address;
    input.binding.device_range_bytes = binding.size_bytes;
    if (!typed_blas::make_raw_bvh_header(
            reinterpret_cast<const void *>(producer_host_blas_header),
            binding.size_bytes,
            &input.raw_header)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT "
               "adapter_failure=1 producer_host_blas=0x%llx\n",
               static_cast<unsigned long long>(
                   producer_host_blas_header));
        fflush(stdout);
        abort();
    }

    const typed_blas::boundary_result_v0 result =
        typed_blas::execute(input);
    if (result.status != typed_blas::kStatusOk) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT "
               "kernel_failure=1 status=%s producer_host_blas=0x%llx\n",
               typed_blas::status_name(
                   static_cast<typed_blas::status_kind>(result.status)),
               static_cast<unsigned long long>(
                   producer_host_blas_header));
        fflush(stdout);
        abort();
    }

    const uint64_t root_device_address =
        binding.device_base_address + result.root_payload_offset;
    const char *range_failure = "unvalidated";
    if (!VulkanRayTracing::validateBlasBinding(
            binding, root_device_address, 64, &range_failure)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT "
               "range_failure=1 reason=%s root_device=0x%llx\n",
               range_failure,
               static_cast<unsigned long long>(root_device_address));
        fflush(stdout);
        abort();
    }

    GEN_RT_BVH legacy_header;
    GEN_RT_BVH_unpack(
        &legacy_header,
        reinterpret_cast<uint8_t *>(legacy_host_blas_header));
    const uint32_t legacy_bounds_min[] = {
        rtcore_v04_fp32_bits(legacy_header.BoundsMin.X),
        rtcore_v04_fp32_bits(legacy_header.BoundsMin.Y),
        rtcore_v04_fp32_bits(legacy_header.BoundsMin.Z),
    };
    const uint32_t legacy_bounds_max[] = {
        rtcore_v04_fp32_bits(legacy_header.BoundsMax.X),
        rtcore_v04_fp32_bits(legacy_header.BoundsMax.Y),
        rtcore_v04_fp32_bits(legacy_header.BoundsMax.Z),
    };
    bool bounds_mismatch = false;
    for (unsigned component = 0; component < 3; ++component) {
        bounds_mismatch = bounds_mismatch ||
                          result.bounds_min_bits[component] !=
                              legacy_bounds_min[component] ||
                          result.bounds_max_bits[component] !=
                              legacy_bounds_max[component];
    }

    const bool legacy_map_valid =
        VulkanRayTracing::validateBlasLegacyAlias(
            legacy_host_blas_header, binding.device_base_address);
    const typed_blas::as_decode_context_v0 &context = result.decode_context;
    const bool instance_reference_mismatch =
        use_instance_reference &&
        (instance_reference.instance_host_address !=
             reinterpret_cast<uint64_t>(raw_instance) ||
         instance_reference.instance_metadata_reference !=
             instance_metadata_reference ||
         instance_reference.blas_host_root_address !=
             producer_host_blas_header ||
         producer_host_blas_header != legacy_host_blas_header);
    const bool mismatch =
        !binding.valid || !binding.live ||
        instance_reference_mismatch ||
        context.bvh_format_profile_id !=
            typed_blas::kGenRtDerivedProfileId ||
        context.reserved_zero != 0 ||
        context.as_object.object_id != binding.object_id ||
        context.as_object.generation != binding.generation ||
        context.as_object.as_type != typed_blas::kAsTypeBlas ||
        context.device_base != binding.device_base_address ||
        context.device_range_bytes != binding.size_bytes ||
        result.root_payload_offset != legacy_header.RootNodeOffset ||
        result.root_payload_kind_valid != 0 || !legacy_map_valid ||
        bounds_mismatch;

    ++stats->contexts;
    if (mismatch) {
        ++stats->mismatches;
        if (instance_reference_mismatch) {
            ++stats->instance_reference_mismatches;
        }
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT "
               "mismatch=1 producer_host_blas=0x%llx "
               "legacy_host_blas=0x%llx object_id=%llu generation=%u "
               "typed_base=0x%llx binding_base=0x%llx "
               "typed_range=%llu binding_range=%llu "
               "typed_root=0x%llx legacy_root=0x%llx "
               "legacy_map_valid=%u bounds_mismatch=%u\n",
               static_cast<unsigned long long>(producer_host_blas_header),
               static_cast<unsigned long long>(legacy_host_blas_header),
               static_cast<unsigned long long>(binding.object_id),
               binding.generation,
               static_cast<unsigned long long>(context.device_base),
               static_cast<unsigned long long>(binding.device_base_address),
               static_cast<unsigned long long>(context.device_range_bytes),
               static_cast<unsigned long long>(binding.size_bytes),
               static_cast<unsigned long long>(result.root_payload_offset),
               static_cast<unsigned long long>(legacy_header.RootNodeOffset),
               legacy_map_valid ? 1u : 0u,
               bounds_mismatch ? 1u : 0u);
        fflush(stdout);
        abort();
    }

    if (rtcore_v04_producer_backed_blas_root_descriptor_enabled()) {
        typed_blas::root_binding_input_v0 root_input = {};
        root_input.decode_context = context;
        root_input.root_descriptor.object_id = binding.object_id;
        root_input.root_descriptor.root_payload_offset =
            binding.root_payload_offset;
        root_input.root_descriptor.object_generation = binding.generation;
        root_input.root_descriptor.build_generation =
            binding.root_build_generation;
        root_input.root_descriptor.bvh_format_profile_id =
            binding.root_bvh_profile_id;
        root_input.root_descriptor.payload_format_id =
            binding.root_payload_format_id;
        root_input.root_descriptor.as_type = typed_blas::kAsTypeBlas;
        root_input.root_descriptor.root_payload_kind =
            binding.root_payload_kind;
        root_input.root_descriptor.valid =
            binding.root_descriptor_valid ? 1 : 0;

        const typed_blas::root_binding_result_v0 root_result =
            typed_blas::execute_root_binding(root_input);
        const char *root_range_failure = "unvalidated";
        const bool root_range_valid =
            root_result.status == typed_blas::kStatusOk &&
            VulkanRayTracing::validateBlasBinding(
                binding, root_result.root_device_address, 64,
                &root_range_failure);
        const bool root_mismatch =
            root_result.status != typed_blas::kStatusOk ||
            !binding.root_descriptor_valid ||
            binding.root_reserved_zero != 0 ||
            root_result.root_payload_kind_valid != 1 ||
            root_result.root_payload_kind != binding.root_payload_kind ||
            root_result.root_payload_offset !=
                legacy_header.RootNodeOffset ||
            root_result.root_payload_offset != binding.root_payload_offset ||
            root_result.root_device_address !=
                binding.device_base_address + legacy_header.RootNodeOffset ||
            root_result.build_generation !=
                binding.root_build_generation ||
            !root_range_valid;

        ++stats->root_descriptors;
        if (root_mismatch) {
            ++stats->root_descriptor_mismatches;
            printf("GPGPU-Sim PTX: "
                   "RTCORE_V04_PRODUCER_BACKED_BLAS_ROOT_DESCRIPTOR "
                   "mismatch=1 producer_host_blas=0x%llx object_id=%llu "
                   "generation=%u build_generation=%u status=%s "
                   "typed_root=0x%llx producer_root=0x%llx "
                   "legacy_root=0x%llx typed_kind=%u producer_kind=%u "
                   "range_valid=%u range_reason=%s\n",
                   static_cast<unsigned long long>(
                       producer_host_blas_header),
                   static_cast<unsigned long long>(binding.object_id),
                   binding.generation, binding.root_build_generation,
                   typed_blas::status_name(
                       static_cast<typed_blas::status_kind>(
                           root_result.status)),
                   static_cast<unsigned long long>(
                       root_result.root_payload_offset),
                   static_cast<unsigned long long>(
                       binding.root_payload_offset),
                   static_cast<unsigned long long>(
                       legacy_header.RootNodeOffset),
                   static_cast<unsigned>(root_result.root_payload_kind),
                   static_cast<unsigned>(binding.root_payload_kind),
                   root_range_valid ? 1u : 0u, root_range_failure);
            fflush(stdout);
            abort();
        }
    }
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

bool VulkanRayTracing::traceRayFromRtcoreAbi(
    const rtcore_trace_ray_abi_entry& entry,
    const ptx_instruction *pI,
    ptx_thread_info *thread)
{
    const bool source_valid =
        entry.source != nullptr &&
        strcmp(entry.source, "decoded_context_window_abi_value_record") == 0;
    const bool root_valid =
        entry.top_level_as != 0 &&
        entry.top_level_as == entry.root_metadata_handle &&
        entry.root_address_space != nullptr &&
        strcmp(entry.root_address_space, "actual_as_bvh_memory") == 0 &&
        entry.root_node_reference != 0 &&
        entry.layout_profile_reference != nullptr &&
        strcmp(entry.layout_profile_reference,
               "RTCORE_BVH_FORMAT_VULKAN_SIM_GEN_RT") == 0 &&
        entry.bvh_memory_binding;
    const bool context_valid =
        entry.context_layout_version == 1 &&
        (entry.context_valid_flags & 0x01u) != 0 &&
        (entry.context_valid_flags & ~0x07u) == 0 &&
        entry.pipeline_profile_id == 1 &&
        entry.bvh_format_profile_id == 1;
    const bool authority_valid =
        entry.all_field_bundles_present &&
        entry.decoded_source_authority_valid &&
        entry.root_descriptor_authority_valid &&
        entry.runtime_lifetime_valid;
    const bool v04_shadow_input_valid =
        !entry.v04_shadow_boundary_enabled ||
        entry.v04_shadow_trace_input_valid;
    const bool v04_tlas_root_matches =
        !entry.v04_tlas_binding_enforcement_enabled ||
        (entry.v04_tlas_binding.valid && entry.v04_tlas_binding.live &&
         entry.v04_tlas_binding.host_root_address == entry.top_level_as);
    const char *v04_tlas_binding_failure =
        entry.v04_tlas_binding_enforcement_enabled ? "unvalidated"
                                                   : "disabled";
    const bool v04_tlas_binding_valid =
        !entry.v04_tlas_binding_enforcement_enabled ||
        (v04_tlas_root_matches &&
         validateTlasBinding(entry.v04_tlas_binding, 0, 0,
                             &v04_tlas_binding_failure));
    if (!entry.valid || !source_valid || !root_valid || !context_valid ||
        !authority_valid || !v04_shadow_input_valid ||
        !v04_tlas_binding_valid || pI == nullptr || thread == nullptr) {
        printf("GPGPU-Sim PTX: RT_SUBMIT "
               "abi-native-trace-ray-entry-rejected=1, "
               "entry_valid=%u, source_valid=%u, root_valid=%u, "
               "context_valid=%u, authority_valid=%u, "
               "v04_shadow_input_valid=%u, "
               "v04_tlas_binding_enforcement=%u, "
               "v04_tlas_root_matches=%u, v04_tlas_binding_valid=%u, "
               "v04_tlas_binding_failure=%s\n",
               entry.valid ? 1 : 0, source_valid ? 1 : 0,
               root_valid ? 1 : 0, context_valid ? 1 : 0,
               authority_valid ? 1 : 0,
               v04_shadow_input_valid ? 1 : 0,
               entry.v04_tlas_binding_enforcement_enabled ? 1u : 0u,
               v04_tlas_root_matches ? 1u : 0u,
               v04_tlas_binding_valid ? 1u : 0u,
               v04_tlas_binding_failure);
        fflush(stdout);
        return false;
    }

    printf("GPGPU-Sim PTX: RT_SUBMIT "
           "abi-native-trace-ray-entry=1, source=%s, "
           "top_level_as=0x%llx, root_node_reference=0x%llx, "
           "context_layout_version=%u, pipeline_profile_id=%u, "
           "bvh_format_profile_id=%u, v04_tlas_binding_enforcement=%u, "
           "tlas_object_id=%llu, tlas_generation=%u, "
           "tlas_device_base=0x%llx, tlas_size=%llu\n",
           entry.source,
           (unsigned long long)entry.top_level_as,
           (unsigned long long)entry.root_node_reference,
           entry.context_layout_version, entry.pipeline_profile_id,
           entry.bvh_format_profile_id,
           entry.v04_tlas_binding_enforcement_enabled ? 1u : 0u,
           (unsigned long long)entry.v04_tlas_binding.object_id,
           entry.v04_tlas_binding.generation,
           (unsigned long long)entry.v04_tlas_binding.device_base_address,
           (unsigned long long)entry.v04_tlas_binding.size_bytes);
    fflush(stdout);

    traceRay((VkAccelerationStructureKHR)entry.top_level_as,
             entry.ray_flags, entry.cull_mask, entry.sbt_record_offset,
             entry.sbt_record_stride, entry.miss_index, entry.ray_origin,
             entry.ray_tmin, entry.ray_direction, entry.ray_tmax,
             entry.context_layout_version, entry.context_valid_flags,
             entry.pipeline_profile_id, entry.bvh_format_profile_id,
             NULL, &entry, pI, thread);
    return true;
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
                   uint32_t context_layout_version,
                   uint32_t context_valid_flags,
                   uint32_t pipeline_profile_id,
                   uint32_t bvh_format_profile_id,
                   int payload,
                   const rtcore_trace_ray_abi_entry *rtcore_abi_entry,
                   const ptx_instruction *pI,
                   ptx_thread_info *thread)
{
    const bool v04_shadow_boundary_enabled =
        rtcore_abi_entry != NULL &&
        rtcore_abi_entry->v04_shadow_boundary_enabled;
    const bool v04_tlas_binding_enforcement_enabled =
        rtcore_abi_entry != NULL &&
        rtcore_abi_entry->v04_tlas_binding_enforcement_enabled;
    const uint64_t selected_tlas_device_base =
        v04_tlas_binding_enforcement_enabled
            ? rtcore_abi_entry->v04_tlas_binding.device_base_address
            : (uint64_t)tlas_addr;
    const bool v04_typed_node_candidate_enabled =
        rtcore_v04_typed_node_candidate_kernel_enabled();
    const bool v04_typed_node_child_route_enabled =
        rtcore_v04_typed_node_child_route_kernel_enabled();
    const bool v04_typed_stack_push_remainder_enabled =
        rtcore_v04_typed_stack_push_remainder_kernel_enabled();
    const bool v04_typed_stack_pop_next_enabled =
        rtcore_v04_typed_stack_pop_next_kernel_enabled();
    const bool v04_private_frontier_owner_layout_enabled =
        rtcore_v04_private_frontier_owner_layout_enabled();
    const bool v04_request_owner_binding_enabled =
        rtcore_v04_request_owner_binding_enabled();
    const bool v04_private_frontier_live_init_enabled =
        rtcore_v04_private_frontier_live_init_enabled();
    if (v04_typed_node_candidate_enabled &&
        (rtcore_abi_entry == NULL || !v04_shadow_boundary_enabled ||
         !rtcore_abi_entry->v04_shadow_trace_input_valid ||
         !v04_tlas_binding_enforcement_enabled ||
         !rtcore_abi_entry->v04_tlas_binding.valid ||
         !rtcore_abi_entry->v04_tlas_binding.live ||
         bvh_format_profile_id != 1)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_KERNEL "
               "configuration_invalid=1 abi_entry=%u boundary=%u "
               "trace_input=%u tlas_enforcement=%u tlas_valid=%u "
               "tlas_live=%u bvh_format_profile=%u\n",
               rtcore_abi_entry != NULL ? 1u : 0u,
               v04_shadow_boundary_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_shadow_trace_input_valid
                   ? 1u
                   : 0u,
               v04_tlas_binding_enforcement_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.valid
                   ? 1u
                   : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.live
                   ? 1u
                   : 0u,
               bvh_format_profile_id);
        fflush(stdout);
        abort();
    }
    const bool v04_typed_primitive_candidate_enabled =
        rtcore_v04_typed_primitive_candidate_kernel_enabled();
    if (v04_typed_primitive_candidate_enabled &&
        (rtcore_abi_entry == NULL || !v04_shadow_boundary_enabled ||
         !rtcore_abi_entry->v04_shadow_trace_input_valid ||
         !v04_tlas_binding_enforcement_enabled ||
         !rtcore_abi_entry->v04_tlas_binding.valid ||
         !rtcore_abi_entry->v04_tlas_binding.live ||
         bvh_format_profile_id != 1)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PRIMITIVE_KERNEL "
               "configuration_invalid=1 abi_entry=%u boundary=%u "
               "trace_input=%u tlas_enforcement=%u tlas_valid=%u "
               "tlas_live=%u bvh_format_profile=%u\n",
               rtcore_abi_entry != NULL ? 1u : 0u,
               v04_shadow_boundary_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_shadow_trace_input_valid
                   ? 1u
                   : 0u,
               v04_tlas_binding_enforcement_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.valid
                   ? 1u
                   : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.live
                   ? 1u
                   : 0u,
               bvh_format_profile_id);
        fflush(stdout);
        abort();
    }
    const bool v04_typed_procedural_boundary_enabled =
        rtcore_v04_typed_procedural_boundary_seed_enabled();
    if (v04_typed_procedural_boundary_enabled &&
        (rtcore_abi_entry == NULL || !v04_shadow_boundary_enabled ||
         !rtcore_abi_entry->v04_shadow_trace_input_valid ||
         !v04_tlas_binding_enforcement_enabled ||
         !rtcore_abi_entry->v04_tlas_binding.valid ||
         !rtcore_abi_entry->v04_tlas_binding.live ||
         bvh_format_profile_id != 1)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PROCEDURAL_BOUNDARY "
               "configuration_invalid=1 abi_entry=%u boundary=%u "
               "trace_input=%u tlas_enforcement=%u tlas_valid=%u "
               "tlas_live=%u bvh_format_profile=%u\n",
               rtcore_abi_entry != NULL ? 1u : 0u,
               v04_shadow_boundary_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_shadow_trace_input_valid
                   ? 1u
                   : 0u,
               v04_tlas_binding_enforcement_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.valid
                   ? 1u
                   : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.live
                   ? 1u
                   : 0u,
               bvh_format_profile_id);
        fflush(stdout);
        abort();
    }
    const bool v04_typed_instance_boundary_enabled =
        rtcore_v04_typed_instance_boundary_seed_enabled();
    if (v04_typed_instance_boundary_enabled &&
        (rtcore_abi_entry == NULL || !v04_shadow_boundary_enabled ||
         !rtcore_abi_entry->v04_shadow_trace_input_valid ||
         !v04_tlas_binding_enforcement_enabled ||
         !rtcore_abi_entry->v04_tlas_binding.valid ||
         !rtcore_abi_entry->v04_tlas_binding.live ||
         bvh_format_profile_id != 1)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_BOUNDARY "
               "configuration_invalid=1 abi_entry=%u boundary=%u "
               "trace_input=%u tlas_enforcement=%u tlas_valid=%u "
               "tlas_live=%u bvh_format_profile=%u\n",
               rtcore_abi_entry != NULL ? 1u : 0u,
               v04_shadow_boundary_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_shadow_trace_input_valid
                   ? 1u
                   : 0u,
               v04_tlas_binding_enforcement_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.valid
                   ? 1u
                   : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.live
                   ? 1u
                   : 0u,
               bvh_format_profile_id);
        fflush(stdout);
        abort();
    }
    const bool v04_typed_blas_decode_context_enabled =
        rtcore_v04_typed_blas_decode_context_bridge_enabled();
    const bool v04_producer_backed_blas_root_descriptor_enabled =
        rtcore_v04_producer_backed_blas_root_descriptor_enabled();
    const bool v04_producer_backed_instance_blas_reference_enabled =
        rtcore_v04_producer_backed_instance_blas_reference_enabled();
    const bool v04_typed_instance_enter_enabled =
        rtcore_v04_typed_instance_enter_transition_enabled();
    if (v04_typed_node_child_route_enabled &&
        !rtcore_v04_typed_node_child_route_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "configuration_invalid=1 node_candidate=%u "
               "instance_enter=%u producer_chain=%u\n",
               v04_typed_node_candidate_enabled ? 1u : 0u,
               v04_typed_instance_enter_enabled ? 1u : 0u,
               rtcore_v04_typed_instance_enter_prerequisites_enabled()
                   ? 1u
                   : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_typed_stack_push_remainder_enabled &&
        !rtcore_v04_typed_stack_push_remainder_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_STACK_PUSH_REMAINDER "
               "configuration_invalid=1 node_child_route=%u "
               "producer_chain=%u\n",
               v04_typed_node_child_route_enabled ? 1u : 0u,
               rtcore_v04_typed_node_child_route_prerequisites_enabled()
                   ? 1u
                   : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_typed_stack_pop_next_enabled &&
        !rtcore_v04_typed_stack_pop_next_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_STACK_POP_NEXT "
               "configuration_invalid=1 stack_push=%u producer_chain=%u\n",
               v04_typed_stack_push_remainder_enabled ? 1u : 0u,
               rtcore_v04_typed_stack_push_remainder_prerequisites_enabled()
                   ? 1u
                   : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_private_frontier_owner_layout_enabled &&
        !rtcore_v04_private_frontier_owner_layout_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_PRIVATE_FRONTIER_OWNER_LAYOUT "
               "configuration_invalid=1 stack_pop=%u producer_chain=%u\n",
               v04_typed_stack_pop_next_enabled ? 1u : 0u,
               rtcore_v04_typed_stack_pop_next_prerequisites_enabled()
                   ? 1u
                   : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_request_owner_binding_enabled &&
        !rtcore_v04_request_owner_binding_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_REQUEST_OWNER_BINDING "
               "configuration_invalid=1 private_frontier_layout=%u "
               "producer_chain=%u replay_admission=%u "
               "continuation_model=%u\n",
               v04_private_frontier_owner_layout_enabled ? 1u : 0u,
               rtcore_v04_private_frontier_owner_layout_prerequisites_enabled()
                   ? 1u
                   : 0u,
               rtcore_replay_admission_enabled() ? 1u : 0u,
               rtcore_continuation_model_enabled() ? 1u : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_private_frontier_live_init_enabled &&
        !rtcore_v04_private_frontier_live_init_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_PRIVATE_FRONTIER_LIVE_INIT "
               "configuration_invalid=1 request_owner_binding=%u "
               "request_owner_prerequisites=%u memory_unit_offer=%u\n",
               v04_request_owner_binding_enabled ? 1u : 0u,
               rtcore_v04_request_owner_binding_prerequisites_enabled()
                   ? 1u
                   : 0u,
               rtcore_replay_memory_unit_request_offer_enabled() ? 1u : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_typed_instance_enter_enabled &&
        !rtcore_v04_typed_instance_enter_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_ENTER "
               "configuration_invalid=1 relation=%u instance_boundary=%u "
               "context_bridge=%u root_descriptor=%u "
               "tlas_binding_enforcement=%u\n",
               v04_producer_backed_instance_blas_reference_enabled ? 1u : 0u,
               v04_typed_instance_boundary_enabled ? 1u : 0u,
               v04_typed_blas_decode_context_enabled ? 1u : 0u,
               v04_producer_backed_blas_root_descriptor_enabled ? 1u : 0u,
               rtcore_v04_tlas_binding_enforcement_gate_enabled() ? 1u : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_producer_backed_instance_blas_reference_enabled &&
        !rtcore_v04_instance_blas_reference_prerequisites_enabled()) {
        printf("GPGPU-Sim PTX: "
               "RTCORE_V04_PRODUCER_BACKED_INSTANCE_BLAS_REFERENCE "
               "configuration_invalid=1 instance_boundary=%u "
               "context_bridge=%u root_descriptor=%u "
               "tlas_binding_enforcement=%u\n",
               v04_typed_instance_boundary_enabled ? 1u : 0u,
               v04_typed_blas_decode_context_enabled ? 1u : 0u,
               v04_producer_backed_blas_root_descriptor_enabled ? 1u : 0u,
               rtcore_v04_tlas_binding_enforcement_gate_enabled() ? 1u : 0u);
        fflush(stdout);
        abort();
    }
    if (v04_producer_backed_blas_root_descriptor_enabled &&
        !v04_typed_blas_decode_context_enabled) {
        printf("GPGPU-Sim PTX: "
               "RTCORE_V04_PRODUCER_BACKED_BLAS_ROOT_DESCRIPTOR "
               "configuration_invalid=1 context_bridge=0\n");
        fflush(stdout);
        abort();
    }
    if (v04_typed_blas_decode_context_enabled &&
        (!v04_typed_instance_boundary_enabled || rtcore_abi_entry == NULL ||
         !v04_shadow_boundary_enabled ||
         !rtcore_abi_entry->v04_shadow_trace_input_valid ||
         !v04_tlas_binding_enforcement_enabled ||
         !rtcore_abi_entry->v04_tlas_binding.valid ||
         !rtcore_abi_entry->v04_tlas_binding.live ||
         bvh_format_profile_id != 1)) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT "
               "configuration_invalid=1 instance_boundary=%u abi_entry=%u "
               "boundary=%u trace_input=%u tlas_enforcement=%u "
               "tlas_valid=%u tlas_live=%u bvh_format_profile=%u\n",
               v04_typed_instance_boundary_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL ? 1u : 0u,
               v04_shadow_boundary_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_shadow_trace_input_valid
                   ? 1u
                   : 0u,
               v04_tlas_binding_enforcement_enabled ? 1u : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.valid
                   ? 1u
                   : 0u,
               rtcore_abi_entry != NULL &&
                       rtcore_abi_entry->v04_tlas_binding.live
                   ? 1u
                   : 0u,
               bvh_format_profile_id);
        fflush(stdout);
        abort();
    }
    rtcore_v04_typed_node_candidate_stats v04_typed_node_candidate_stats;
    rtcore_v04_typed_node_child_route_stats
        v04_typed_node_child_route_stats;
    rtcore_v04_typed_stack_push_remainder_stats
        v04_typed_stack_push_remainder_stats;
    rtcore_v04_typed_stack_pop_next_stats
        v04_typed_stack_pop_next_stats;
    rtcore_v04_private_frontier_owner_layout_stats
        v04_private_frontier_owner_layout_stats;
    rtcore_v04_typed_node_reference_tracker
        v04_typed_node_reference_tracker;
    rtcore_v04_typed_primitive_candidate_stats
        v04_typed_primitive_candidate_stats;
    rtcore_v04_typed_procedural_boundary_stats
        v04_typed_procedural_boundary_stats;
    rtcore_v04_typed_instance_boundary_stats
        v04_typed_instance_boundary_stats;
    rtcore_v04_typed_blas_decode_context_stats
        v04_typed_blas_decode_context_stats;
    rtcore_v04_typed_instance_enter_stats v04_typed_instance_enter_stats;
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
    int64_t device_offset =
        selected_tlas_device_base - (uint64_t)_topLevelAS;
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

    Traversal_data traversal_data = {};

    traversal_data.n_all_hits = 0;
    traversal_data.hit_geometry = false;
    traversal_data.closest_hit.hitGroupIndex = -1;
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
    rtcore_maybe_run_compact_trace_boundary_overflow_self_test();

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
    uint64_t closest_instance_metadata_ref = 0;
    float4x4 closest_worldToObject, closest_objectToWorld;
    Ray closest_objectRay;
    float min_thit_object;
    uint32_t closest_hit_kind = 0;

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
    if (v04_typed_node_child_route_enabled) {
        rtcore_v04_seed_typed_tlas_root_reference(
            reinterpret_cast<const uint8_t *>(_topLevelAS),
            rtcore_abi_entry->v04_tlas_binding,
            topBVH.RootNodeOffset, &v04_typed_node_reference_tracker,
            &v04_typed_node_child_route_stats);
    }
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
                selected_tlas_device_base - (uint64_t)_topLevelAS;
            rtcore_compact_trace.append_stack_pop(
                (uint64_t)stack.back().addr + tlas_stack_device_offset,
                stack.back().topLevel, stack.back().leaf);
            stack.pop_back();
        }

        while (next_node_addr != NULL)
        {
            // TLAS offset
            device_offset =
                selected_tlas_device_base - (uint64_t)_topLevelAS;

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

            if (v04_typed_node_candidate_enabled) {
                rtcore_v04_observe_typed_node_candidates(
                    node_addr, node, ray.get_origin(), ray.get_direction(),
                    ray.get_tmin(), ray.get_tmax(), min_thit, rayFlags,
                    cullMask, true, child_hit, thit,
                    &v04_typed_node_candidate_stats);
            }
            if (v04_typed_node_child_route_enabled) {
                rtcore_v04_observe_typed_node_child_route(
                    node_addr, node, ray.get_origin(), ray.get_direction(),
                    ray.get_tmin(), ray.get_tmax(), min_thit, rayFlags,
                    cullMask, true, child_hit, thit,
                    reinterpret_cast<uint64_t>(_topLevelAS),
                    &v04_typed_node_reference_tracker,
                    &v04_typed_node_child_route_stats,
                    v04_typed_stack_push_remainder_enabled
                        ? &v04_typed_stack_push_remainder_stats
                        : NULL,
                    v04_typed_stack_pop_next_enabled
                        ? &v04_typed_stack_pop_next_stats
                        : NULL,
                    v04_private_frontier_owner_layout_enabled
                        ? &v04_private_frontier_owner_layout_stats
                        : NULL);
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
            device_offset =
                selected_tlas_device_base - (uint64_t)_topLevelAS;

            assert(stack.back().topLevel);

            uint8_t* leaf_addr = stack.back().addr;
            rtcore_compact_trace.append_stack_pop(
                (uint64_t)leaf_addr + device_offset, stack.back().topLevel,
                stack.back().leaf);
            stack.pop_back();

            GEN_RT_BVH_INSTANCE_LEAF instanceLeaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&instanceLeaf, leaf_addr);
            if (v04_typed_instance_boundary_enabled) {
                rtcore_v04_observe_typed_instance_boundary_seed(
                    leaf_addr, instanceLeaf,
                    &v04_typed_instance_boundary_stats);
            }
            const uint64_t instance_metadata_ref =
                (uint64_t)leaf_addr + device_offset;
            if (v04_typed_blas_decode_context_enabled) {
                rtcore_v04_observe_typed_blas_decode_context(
                    leaf_addr, instanceLeaf,
                    rtcore_abi_entry->v04_tlas_binding,
                    instance_metadata_ref,
                    &v04_typed_blas_decode_context_stats);
            }
            rtcore::v04::typed_instance::enter_result_v0
                v04_route_instance_enter_result = {};
            bool v04_route_instance_enter_result_valid = false;
            if (v04_typed_instance_enter_enabled) {
                rtcore_v04_observe_typed_instance_enter_transition(
                    leaf_addr, instanceLeaf,
                    rtcore_abi_entry->v04_tlas_binding,
                    instance_metadata_ref, ray, rayFlags, cullMask,
                    &v04_typed_instance_enter_stats,
                    v04_typed_node_child_route_enabled
                        ? &v04_route_instance_enter_result
                        : NULL);
                v04_route_instance_enter_result_valid =
                    v04_typed_node_child_route_enabled;
            }
            rtcore::v04::typed_blas::as_decode_context_v0
                v04_route_blas_context = {};
            uint64_t v04_route_blas_root_offset = 0;
            if (v04_typed_node_child_route_enabled) {
                namespace typed_instance = rtcore::v04::typed_instance;
                if (!v04_route_instance_enter_result_valid ||
                    v04_route_instance_enter_result.status !=
                        typed_instance::kStatusOk ||
                    v04_route_instance_enter_result.result_kind !=
                        typed_instance::kEnterResultBlasRoot ||
                    (v04_route_instance_enter_result.output_valid_mask &
                     typed_instance::kRootFetchValid) == 0) {
                    printf("GPGPU-Sim PTX: "
                           "RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
                           "capture_failure=1 "
                           "reason=typed_instance_root_unavailable "
                           "metadata_ref=0x%llx\n",
                           static_cast<unsigned long long>(
                               instance_metadata_ref));
                    fflush(stdout);
                    abort();
                }
                v04_route_blas_context =
                    v04_route_instance_enter_result.root_fetch.decode_context;
                v04_route_blas_root_offset =
                    v04_route_instance_enter_result.root_fetch
                        .encoded_reference;
            }
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
            const uint64_t v04_route_blas_host_base =
                reinterpret_cast<uint64_t>(botLevelRootAddr);
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
            if (v04_typed_node_child_route_enabled) {
                if (v04_route_blas_root_offset !=
                        botLevelASAddr.RootNodeOffset ||
                    !rtcore_v04_publish_typed_node_reference(
                        &v04_typed_node_reference_tracker,
                        reinterpret_cast<uint64_t>(botLevelRootAddr),
                        v04_route_blas_context,
                        v04_route_blas_root_offset)) {
                    printf("GPGPU-Sim PTX: "
                           "RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
                           "root_seed_failure=1 reason=typed_instance_root "
                           "typed_offset=0x%llx comparison_offset=0x%llx\n",
                           static_cast<unsigned long long>(
                               v04_route_blas_root_offset),
                           static_cast<unsigned long long>(
                               botLevelASAddr.RootNodeOffset));
                    fflush(stdout);
                    abort();
                }
                ++v04_typed_node_child_route_stats.seeded_roots;
            }
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

                    if (v04_typed_node_candidate_enabled) {
                        rtcore_v04_observe_typed_node_candidates(
                            node_addr, node, objectRay.get_origin(),
                            objectRay.get_direction(), objectRay.get_tmin(),
                            objectRay.get_tmax(),
                            min_thit * worldToObject_tMultiplier, rayFlags,
                            cullMask, false, child_hit, thit,
                            &v04_typed_node_candidate_stats);
                    }
                    if (v04_typed_node_child_route_enabled) {
                        rtcore_v04_observe_typed_node_child_route(
                            node_addr, node, objectRay.get_origin(),
                            objectRay.get_direction(), objectRay.get_tmin(),
                            objectRay.get_tmax(),
                            min_thit * worldToObject_tMultiplier, rayFlags,
                            cullMask, false, child_hit, thit,
                            v04_route_blas_host_base,
                            &v04_typed_node_reference_tracker,
                            &v04_typed_node_child_route_stats,
                            v04_typed_stack_push_remainder_enabled
                                ? &v04_typed_stack_push_remainder_stats
                                : NULL,
                            v04_typed_stack_pop_next_enabled
                                ? &v04_typed_stack_pop_next_stats
                                : NULL,
                            v04_private_frontier_owner_layout_enabled
                                ? &v04_private_frontier_owner_layout_stats
                                : NULL);
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
                        bool counter_clockwise_facing = false;
                        total_primitive_tests++;
                        bool hit = VulkanRayTracing::mt_ray_triangle_test(
                            p[0], p[1], p[2], objectRay, &thit,
                            &counter_clockwise_facing);
                        const float world_thit =
                            hit ? thit / worldToObject_tMultiplier : 0.0f;
                        const bool opaque_hit_selected =
                            hit && Tmin <= world_thit && world_thit <= Tmax &&
                            skipAnyHitShader && world_thit < min_thit;
                        const bool opaque_commit =
                            v04_shadow_boundary_enabled && opaque_hit_selected;
                        const bool front_facing =
                            (instanceLeaf.InstanceFlags &
                             TRIANGLE_FRONT_COUNTERCLOCKWISE)
                                ? counter_clockwise_facing
                                : !counter_clockwise_facing;
                        const uint32_t triangle_hit_kind =
                            front_facing ? 0xfeu : 0xffu;
                        if (v04_typed_primitive_candidate_enabled) {
                            float3 legacy_barycentric = {};
                            if (hit) {
                                const float3 object_intersection_point =
                                    objectRay.get_origin() +
                                    make_float3(
                                        objectRay.get_direction().x * thit,
                                        objectRay.get_direction().y * thit,
                                        objectRay.get_direction().z * thit);
                                legacy_barycentric = Barycentric(
                                    object_intersection_point, p[0], p[1],
                                    p[2]);
                            }
                            rtcore_v04_observe_typed_primitive_candidate(
                                leaf_addr, leaf, objectRay,
                                worldToObject_tMultiplier, Tmin, Tmax,
                                min_thit,
                                instanceLeaf.InstanceFlags, hit, thit,
                                counter_clockwise_facing, front_facing,
                                triangle_hit_kind, legacy_barycentric,
                                &v04_typed_primitive_candidate_stats);
                        }
                        const unsigned primitive_test_event_seq =
                            rtcore_compact_trace.append_primitive_test(
                            (uint64_t)leaf_addr + device_offset, hit,
                            rtcore_trace_primitive_flags(
                                RTCORE_TRACE_PRIMITIVE_KIND_TRIANGLE_TEST,
                                hit, false, opaque_commit));

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

                        //TODO: why the Tmin Tmax consition wasn't handled in the object coordinates?
                        if(hit && Tmin <= world_thit && world_thit <= Tmax)
                        {
                            if (debugTraversal)
                            {
                                traversalFile << "quad node " << (void *)leaf_addr << ", primitiveID " << leaf.PrimitiveIndex0 << " is the closest hit. world_thit " << thit / worldToObject_tMultiplier;
                            }

                            if (opaque_hit_selected) {
                                min_thit = world_thit;
                                if (v04_shadow_boundary_enabled) {
                                    Hit_data opaque_candidate = {};
                                    opaque_candidate.geometryType =
                                        VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                                    opaque_candidate.hit_kind = triangle_hit_kind;
                                    opaque_candidate.geometry_index =
                                        leaf.LeafDescriptor.GeometryIndex;
                                    opaque_candidate.primitive_index =
                                        leaf.PrimitiveIndex0;
                                    opaque_candidate.instance_index =
                                        instanceLeaf.InstanceID;
                                    opaque_candidate.instance_id =
                                        instanceLeaf.InstanceIndex;
                                    opaque_candidate.hitGroupIndex =
                                        instanceLeaf
                                            .InstanceContributionToHitGroupIndex;
                                    opaque_candidate.world_min_thit = world_thit;
                                    const float3 object_intersection_point =
                                        objectRay.get_origin() +
                                        make_float3(
                                            objectRay.get_direction().x * thit,
                                            objectRay.get_direction().y * thit,
                                            objectRay.get_direction().z * thit);
                                    opaque_candidate.barycentric_coordinates =
                                        Barycentric(object_intersection_point,
                                                    p[0], p[1], p[2]);
                                    rtcore_compact_trace
                                        .record_boundary_candidate(
                                            primitive_test_event_seq, UINT_MAX,
                                            0,
                                            instanceLeaf
                                                .InstanceContributionToHitGroupIndex,
                                            1,
                                            leaf.LeafDescriptor.GeometryIndex,
                                            leaf.PrimitiveIndex0,
                                            instanceLeaf.InstanceID,
                                            triangle_hit_kind,
                                            rtcore_make_v04_boundary_values_from_hit(
                                                opaque_candidate,
                                                instance_metadata_ref,
                                                instanceLeaf
                                                    .InstanceContributionToHitGroupIndex,
                                                instanceLeaf.InstanceIndex,
                                                instanceLeaf.InstanceID,
                                                rtcore_v04_fp32_bits(world_thit),
                                                true));
                                }
                            }
                            // The legacy functional path overwrites these
                            // fields for every valid hit, even when min_thit
                            // did not change. Keep that behavior for V0.3,
                            // but make the default-off V0.4 candidate pair
                            // distance and identity from the same commit.
                            if (!v04_shadow_boundary_enabled ||
                                !skipAnyHitShader || opaque_hit_selected) {
                                min_thit_object = thit;
                                closest_leaf = leaf;
                                closest_instanceLeaf = instanceLeaf;
                                closest_instance_metadata_ref =
                                    instance_metadata_ref;
                                closest_worldToObject = worldToObjectMatrix;
                                closest_objectToWorld = objectToWorldMatrix;
                                closest_objectRay = objectRay;
                                closest_hit_kind = triangle_hit_kind;
                            }
                            thread->add_ray_intersect();
                            transactions.push_back(MemoryTransactionRecord((uint8_t*)((uint64_t)leaf_addr + device_offset), GEN_RT_BVH_QUAD_LEAF_length * 4, TransactionType::BVH_QUAD_LEAF_HIT));
                            ctx->func_sim->g_rt_mem_access_type[static_cast<int>(TransactionType::BVH_QUAD_LEAF_HIT)]++;
                            total_nodes_accessed++;

                            if (!skipAnyHitShader) {
                                VSIM_DPRINTF("gpgpusim: Adding triangle intersection to anyhit shader table\n");
                                warp_intersection_table* table = anyhit_table[thread->get_ctaid().x][thread->get_ctaid().y];
                                
                                uint32_t hit_group_index = instanceLeaf.InstanceContributionToHitGroupIndex;
                                uint32_t boundary_shader_counter = UINT_MAX;
                                auto intersectionTransactions = table->add_intersection(hit_group_index, thread->get_tid().x, leaf.PrimitiveIndex0, instanceLeaf.InstanceID, instanceLeaf.InstanceIndex, leaf.LeafDescriptor.GeometryIndex, pI, thread, &boundary_shader_counter); // TODO: switch these to device addresses

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

                                Hit_data anyhit_hit_attributes = {};
                                anyhit_hit_attributes.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                                anyhit_hit_attributes.hit_kind = triangle_hit_kind;
                                anyhit_hit_attributes.geometry_index = leaf.LeafDescriptor.GeometryIndex;
                                anyhit_hit_attributes.primitive_index = leaf.PrimitiveIndex0;
                                anyhit_hit_attributes.instance_index = instanceLeaf.InstanceID;
                                anyhit_hit_attributes.instance_id = instanceLeaf.InstanceIndex;
                                anyhit_hit_attributes.hitGroupIndex =
                                    hit_group_index;

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

                                if (traversal_data.n_all_hits == 0) {
                                    traversal_data.closest_hit =
                                        anyhit_hit_attributes;
                                }

                                VSIM_DPRINTF("gpgpusim: Ray hit geomID %d primID %d at (%5.3f, %5.3f, %5.3f) with t = %5.3f\n", anyhit_hit_attributes.geometry_index, anyhit_hit_attributes.primitive_index, barycentric.x, barycentric.y, barycentric.z, thit);

                                // Allocate memory to store hit attributes
                                memory_space *mem = thread->get_global_memory();
                                Hit_data* device_hit_attributes = (Hit_data*) VulkanRayTracing::gpgpusim_alloc(sizeof(Hit_data));
                                mem->write(device_hit_attributes, sizeof(Hit_data), &anyhit_hit_attributes, thread, pI);
                                thread->RT_thread_data->all_hit_data.push_back(device_hit_attributes);

                                traversal_data.n_all_hits++;
                                const unsigned boundary_event_seq =
                                    rtcore_compact_trace.append_hit_update(
                                    (uint64_t)device_hit_attributes,
                                    traversal_data.n_all_hits,
                                    rtcore_trace_hit_update_flags(
                                        RTCORE_TRACE_HIT_UPDATE_KIND_ANY_HIT));
                                rtcore_compact_trace.record_boundary_candidate(
                                    boundary_event_seq,
                                    boundary_shader_counter,
                                    (uint64_t)device_hit_attributes,
                                    hit_group_index, 1,
                                    leaf.LeafDescriptor.GeometryIndex,
                                    leaf.PrimitiveIndex0, instanceLeaf.InstanceID,
                                    triangle_hit_kind,
                                    v04_shadow_boundary_enabled
                                        ? rtcore_make_v04_boundary_values_from_hit(
                                              anyhit_hit_attributes,
                                              instance_metadata_ref,
                                              instanceLeaf.InstanceContributionToHitGroupIndex,
                                              instanceLeaf.InstanceIndex,
                                              instanceLeaf.InstanceID,
                                              rtcore_v04_fp32_bits(anyhit_thit),
                                              true)
                                        : rtcore::abi_v04::shadow::boundary_values());
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
                        if (v04_typed_procedural_boundary_enabled) {
                            rtcore_v04_observe_typed_procedural_boundary_seed(
                                leaf_addr, leaf, cullMask,
                                &v04_typed_procedural_boundary_stats);
                        }
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

                        traversal_data.closest_hit.geometryType =
                            VK_GEOMETRY_TYPE_AABBS_KHR;
                        traversal_data.closest_hit.hit_kind = 0;
                        traversal_data.closest_hit.geometry_index =
                            leaf.LeafDescriptor.GeometryIndex;
                        traversal_data.closest_hit.primitive_index =
                            leaf.PrimitiveIndex[0];
                        traversal_data.closest_hit.instance_index =
                            instanceLeaf.InstanceID;
                        traversal_data.closest_hit.instance_id =
                            instanceLeaf.InstanceIndex;
                        traversal_data.closest_hit.hitGroupIndex =
                            hit_group_index;

                        warp_intersection_table* table = intersection_table[thread->get_ctaid().x][thread->get_ctaid().y];
                        uint32_t boundary_shader_counter = UINT_MAX;
                        auto intersectionTransactions = table->add_intersection(hit_group_index, thread->get_tid().x, leaf.PrimitiveIndex[0], instanceLeaf.InstanceID, instanceLeaf.InstanceIndex, leaf.LeafDescriptor.GeometryIndex, pI, thread, &boundary_shader_counter); // TODO: switch these to device addresses
                        const unsigned boundary_event_seq =
                            rtcore_compact_trace.append_primitive_test(
                            (uint64_t)leaf_addr + device_offset, true,
                            rtcore_trace_primitive_flags(
                                RTCORE_TRACE_PRIMITIVE_KIND_PROCEDURAL_DEFERRED,
                                true, true));
                        rtcore_compact_trace.record_boundary_candidate(
                            boundary_event_seq, boundary_shader_counter, 0,
                            hit_group_index, 2,
                            leaf.LeafDescriptor.GeometryIndex,
                            leaf.PrimitiveIndex[0], instanceLeaf.InstanceID, 0,
                            v04_shadow_boundary_enabled
                                ? rtcore_make_v04_boundary_values_from_hit(
                                      traversal_data.closest_hit,
                                      instance_metadata_ref,
                                      instanceLeaf.InstanceContributionToHitGroupIndex,
                                      instanceLeaf.InstanceIndex,
                                      instanceLeaf.InstanceID,
                                      rtcore_v04_fp32_bits(min_thit), false)
                                : rtcore::abi_v04::shadow::boundary_values());
                        
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
        traversal_data.closest_hit.hit_kind = closest_hit_kind;
        traversal_data.closest_hit.geometry_index = closest_leaf.LeafDescriptor.GeometryIndex;
        traversal_data.closest_hit.primitive_index = closest_leaf.PrimitiveIndex0;
        traversal_data.closest_hit.instance_index = closest_instanceLeaf.InstanceID;
        traversal_data.closest_hit.instance_id = closest_instanceLeaf.InstanceIndex;
        traversal_data.closest_hit.hitGroupIndex =
            closest_instanceLeaf.InstanceContributionToHitGroupIndex;
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

    if (v04_typed_node_candidate_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_KERNEL summary=1 "
               "thread_uid=%u tlas_nodes=%u blas_nodes=%u "
               "evaluated_children=%u hit_candidates=%u mismatches=%u "
               "functional_authority=0 timing_authority=0\n",
               thread->get_uid(), v04_typed_node_candidate_stats.tlas_nodes,
               v04_typed_node_candidate_stats.blas_nodes,
               v04_typed_node_candidate_stats.evaluated_children,
               v04_typed_node_candidate_stats.hit_candidates,
               v04_typed_node_candidate_stats.mismatches);
        fflush(stdout);
    }
    if (v04_typed_node_child_route_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_NODE_CHILD_ROUTE "
               "summary=1 thread_uid=%u seeded_roots=%u "
               "tlas_nodes=%u blas_nodes=%u "
               "evaluated_references=%u selected_routes=%u miss_routes=%u "
               "frontier_items=%u mismatches=%u memory_issue_authority=0 "
               "frontier_commit_authority=0 private_state_authority=0 "
               "functional_authority=0 timing_authority=0\n",
               thread->get_uid(),
               v04_typed_node_child_route_stats.seeded_roots,
               v04_typed_node_child_route_stats.tlas_nodes,
               v04_typed_node_child_route_stats.blas_nodes,
               v04_typed_node_child_route_stats.evaluated_references,
               v04_typed_node_child_route_stats.selected_routes,
               v04_typed_node_child_route_stats.miss_routes,
               v04_typed_node_child_route_stats.frontier_items,
               v04_typed_node_child_route_stats.mismatches);
        fflush(stdout);
    }
    if (v04_typed_stack_push_remainder_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_STACK_PUSH_REMAINDER "
               "summary=1 thread_uid=%u operations=%u "
               "input_frontier_items=%u written_items=%u pruned_items=%u "
               "mismatches=%u persistent_frontier_authority=0 "
               "memory_issue_authority=0 result_commit_authority=0 "
               "functional_authority=0 timing_authority=0\n",
               thread->get_uid(),
               v04_typed_stack_push_remainder_stats.operations,
               v04_typed_stack_push_remainder_stats.input_frontier_items,
               v04_typed_stack_push_remainder_stats.written_items,
               v04_typed_stack_push_remainder_stats.pruned_items,
               v04_typed_stack_push_remainder_stats.mismatches);
        fflush(stdout);
    }
    if (v04_typed_stack_pop_next_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_STACK_POP_NEXT "
               "summary=1 thread_uid=%u operations=%u selected_items=%u "
               "pruned_items=%u popped_items=%u mismatches=%u "
               "persistent_frontier_authority=0 memory_issue_authority=0 "
               "result_commit_authority=0 functional_authority=0 "
               "timing_authority=0\n",
               thread->get_uid(), v04_typed_stack_pop_next_stats.operations,
               v04_typed_stack_pop_next_stats.selected_items,
               v04_typed_stack_pop_next_stats.pruned_items,
               v04_typed_stack_pop_next_stats.popped_items,
               v04_typed_stack_pop_next_stats.mismatches);
        fflush(stdout);
    }
    if (v04_private_frontier_owner_layout_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_PRIVATE_FRONTIER_OWNER_LAYOUT "
               "summary=1 thread_uid=%u initialized_slots=%u "
               "append_operations=%u pop_operations=%u read_plans=%u "
               "write_plans=%u planned_chunks=%u metadata_chunks=%u "
               "entry_chunks=%u multi_chunk_entry_plans=%u "
               "mismatches=%u producer_owner_authority=0 "
               "live_private_slot_authority=0 shared_memory_mutation=0 "
               "memory_issue_authority=0 result_commit_authority=0 "
               "functional_authority=0 timing_authority=0\n",
               thread->get_uid(),
               v04_private_frontier_owner_layout_stats.initialized_slots,
               v04_private_frontier_owner_layout_stats.append_operations,
               v04_private_frontier_owner_layout_stats.pop_operations,
               v04_private_frontier_owner_layout_stats.read_plans,
               v04_private_frontier_owner_layout_stats.write_plans,
               v04_private_frontier_owner_layout_stats.planned_chunks,
               v04_private_frontier_owner_layout_stats.metadata_chunks,
               v04_private_frontier_owner_layout_stats.entry_chunks,
               v04_private_frontier_owner_layout_stats
                   .multi_chunk_entry_plans,
               v04_private_frontier_owner_layout_stats.mismatches);
        fflush(stdout);
    }
    if (v04_typed_primitive_candidate_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PRIMITIVE_KERNEL summary=1 "
               "thread_uid=%u leaves=%u geometric_hits=%u "
               "candidate_hits=%u mismatches=%u "
               "functional_authority=0 timing_authority=0\n",
               thread->get_uid(),
               v04_typed_primitive_candidate_stats.leaves,
               v04_typed_primitive_candidate_stats.geometric_hits,
               v04_typed_primitive_candidate_stats.candidate_hits,
               v04_typed_primitive_candidate_stats.mismatches);
        fflush(stdout);
    }
    if (v04_typed_procedural_boundary_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_PROCEDURAL_BOUNDARY summary=1 "
               "thread_uid=%u leaves=%u mask_visible=%u mismatches=%u "
               "functional_authority=0 timing_authority=0\n",
               thread->get_uid(), v04_typed_procedural_boundary_stats.leaves,
               v04_typed_procedural_boundary_stats.mask_visible,
               v04_typed_procedural_boundary_stats.mismatches);
        fflush(stdout);
    }
    if (v04_typed_instance_boundary_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_BOUNDARY summary=1 "
               "thread_uid=%u instances=%u mismatches=%u "
               "transform_authority=0 functional_authority=0 "
               "timing_authority=0\n",
               thread->get_uid(), v04_typed_instance_boundary_stats.instances,
               v04_typed_instance_boundary_stats.mismatches);
        fflush(stdout);
    }
    if (v04_typed_blas_decode_context_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_BLAS_DECODE_CONTEXT "
               "summary=1 thread_uid=%u contexts=%u mismatches=%u "
               "root_kind_authority=0 transform_authority=0 "
               "blas_transition_authority=0 functional_authority=0 "
               "timing_authority=0\n",
               thread->get_uid(),
               v04_typed_blas_decode_context_stats.contexts,
               v04_typed_blas_decode_context_stats.mismatches);
        fflush(stdout);
    }
    if (v04_producer_backed_blas_root_descriptor_enabled) {
        printf("GPGPU-Sim PTX: "
               "RTCORE_V04_PRODUCER_BACKED_BLAS_ROOT_DESCRIPTOR "
               "summary=1 thread_uid=%u descriptors=%u mismatches=%u "
               "transform_authority=0 blas_transition_authority=0 "
               "memory_issue_authority=0 private_state_authority=0 "
               "functional_authority=0 timing_authority=0\n",
               thread->get_uid(),
               v04_typed_blas_decode_context_stats.root_descriptors,
               v04_typed_blas_decode_context_stats
                   .root_descriptor_mismatches);
        fflush(stdout);
    }
    if (v04_producer_backed_instance_blas_reference_enabled) {
        printf("GPGPU-Sim PTX: "
               "RTCORE_V04_PRODUCER_BACKED_INSTANCE_BLAS_REFERENCE "
               "summary=1 thread_uid=%u resolutions=%u mismatches=%u "
               "legacy_pointer_authority=0 transform_authority=0 "
               "blas_transition_authority=0 memory_issue_authority=0 "
               "private_state_authority=0 functional_authority=0 "
               "timing_authority=0\n",
               thread->get_uid(),
               v04_typed_blas_decode_context_stats.instance_references,
               v04_typed_blas_decode_context_stats
                   .instance_reference_mismatches);
        fflush(stdout);
    }
    if (v04_typed_instance_enter_enabled) {
        printf("GPGPU-Sim PTX: RTCORE_V04_TYPED_INSTANCE_ENTER "
               "summary=1 thread_uid=%u observations=%u culled=%u "
               "blas_roots=%u mismatches=%u transform_authority=0 "
               "blas_transition_authority=0 memory_issue_authority=0 "
               "private_state_authority=0 functional_authority=0 "
               "timing_authority=0\n",
               thread->get_uid(), v04_typed_instance_enter_stats.observations,
               v04_typed_instance_enter_stats.culled,
               v04_typed_instance_enter_stats.blas_roots,
               v04_typed_instance_enter_stats.mismatches);
        fflush(stdout);
    }

    memory_space *mem = thread->get_global_memory();
    Traversal_data* device_traversal_data = (Traversal_data*) VulkanRayTracing::gpgpusim_alloc(sizeof(Traversal_data));
    traversal_data.rtcore_node_visits = total_nodes_accessed;
    traversal_data.rtcore_primitive_tests = total_primitive_tests;
    rtcore_compact_trace.append_completion_summary(total_nodes_accessed,
                                                   total_primitive_tests);
    rtcore_compact_trace.set_oracle_shader_boundary_reason(
        traversal_data.n_all_hits, hit_procedural);
    rtcore_compact_trace_export_record rtcore_trace_export = rtcore_compact_trace.export_record();
    rtcore_trace_export.context_profile_valid =
        context_layout_version != 0 && context_valid_flags != 0 &&
        pipeline_profile_id != 0 && bvh_format_profile_id != 0;
    rtcore_trace_export.context_layout_version = context_layout_version;
    rtcore_trace_export.context_valid_flags = context_valid_flags;
    rtcore_trace_export.pipeline_profile_id = pipeline_profile_id;
    rtcore_trace_export.bvh_format_profile_id = bvh_format_profile_id;
    rtcore_trace_export.ray_sbt_inputs_valid = true;
    rtcore_trace_export.sbt_record_offset = traversal_data.sbtRecordOffset;
    rtcore_trace_export.sbt_record_stride = traversal_data.sbtRecordStride;
    rtcore_trace_export.miss_index = traversal_data.missIndex;
    rtcore_trace_export.ray_flags = traversal_data.rayFlags;
    rtcore_trace_export.cull_mask = traversal_data.cullMask;
    rtcore_trace_export.hit_geometry_summary_valid =
        traversal_data.hit_geometry;
    const bool candidate_summary_valid =
        traversal_data.hit_geometry || traversal_data.n_all_hits != 0 ||
        hit_procedural;
    rtcore_trace_export.closest_hit_kind =
        candidate_summary_valid
            ? traversal_data.closest_hit.hit_kind
            : 0u;
    rtcore_trace_export.closest_hit_geometry_type =
        candidate_summary_valid
            ? (traversal_data.closest_hit.geometryType ==
                       VK_GEOMETRY_TYPE_TRIANGLES_KHR
                   ? 0x01u
                   : 0x02u)
            : 0u;
    rtcore_trace_export.closest_hit_geometry_index =
        candidate_summary_valid
            ? traversal_data.closest_hit.geometry_index
            : 0u;
    rtcore_trace_export.closest_hit_primitive_index =
        candidate_summary_valid
            ? traversal_data.closest_hit.primitive_index
            : 0u;
    rtcore_trace_export.closest_hit_instance_index =
        candidate_summary_valid
            ? traversal_data.closest_hit.instance_index
            : 0u;
    rtcore_trace_export.instance_sbt_contribution_valid =
        candidate_summary_valid;
    rtcore_trace_export.instance_sbt_contribution =
        candidate_summary_valid
            ? (unsigned)traversal_data.closest_hit.hitGroupIndex
            : 0u;
    rtcore_trace_export.v04_shadow_boundary_enabled =
        v04_shadow_boundary_enabled;
    rtcore_trace_export.v04_tlas_binding_enforcement_enabled =
        v04_tlas_binding_enforcement_enabled;
    if (v04_tlas_binding_enforcement_enabled) {
        rtcore_trace_export.v04_tlas_binding =
            rtcore_abi_entry->v04_tlas_binding;
    }
    rtcore_trace_export.handoff_window_base =
        rtcore_abi_entry != NULL ? rtcore_abi_entry->handoff_window_base : 0;
    rtcore_trace_export.v04_shadow_trace_input_valid =
        rtcore_trace_export.v04_shadow_boundary_enabled &&
        rtcore_abi_entry->v04_shadow_trace_input_valid;
    if (rtcore_trace_export.v04_shadow_trace_input_valid) {
        rtcore_trace_export.v04_shadow_trace_input_words =
            rtcore_abi_entry->v04_shadow_trace_input_words;
    }
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

bool VulkanRayTracing::mt_ray_triangle_test(
    float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit,
    bool* counter_clockwise_facing)
{
    // Moller Trumbore algorithm (from scratchapixel.com)
    float3 v0v1 = p1 - p0;
    float3 v0v2 = p2 - p0;
    float3 pvec = cross(ray_properties.get_direction(), v0v2);
    float det = dot(v0v1, pvec);

    if (counter_clockwise_facing) {
        *counter_clockwise_facing = det > 0.0f;
    }

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
                      void *raygen_sbt_device_addr,
                      void *miss_sbt_device_addr,
                      void *hit_sbt_device_addr,
                      void *callable_sbt_device_addr,
                      uint64_t raygen_sbt_stride,
                      uint64_t raygen_sbt_size,
                      uint64_t miss_sbt_stride,
                      uint64_t miss_sbt_size,
                      uint64_t hit_sbt_stride,
                      uint64_t hit_sbt_size,
                      uint64_t callable_sbt_stride,
                      uint64_t callable_sbt_size,
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

    uint32_t shaderId = 0;
    if (!rtcoreLoadCompatibilitySbtShaderId(
            raygen_sbt, raygen_sbt_stride, raygen_sbt_size, 0, 0,
            &shaderId)) {
        printf("GPGPU-Sim PTX: compatibility SBT fail-closed, "
               "region=raygen, base=%p, stride=%llu, size=%llu\n",
               raygen_sbt, (unsigned long long)raygen_sbt_stride,
               (unsigned long long)raygen_sbt_size);
        fflush(stdout);
        abort();
    }
    ctx->func_sim->g_total_shaders = shaders.size();

    const shader_stage_info *raygen_shader = NULL;
    for (size_t index = 0; index < shaders.size(); ++index) {
        if (shaders[index].ID == shaderId &&
            shaders[index].type == MESA_SHADER_RAYGEN) {
            raygen_shader = &shaders[index];
            break;
        }
    }
    if (raygen_shader == NULL) {
        printf("GPGPU-Sim PTX: compatibility SBT fail-closed, "
               "region=raygen, shader_id=%u\n", shaderId);
        fflush(stdout);
        abort();
    }
    function_info *entry = context->get_kernel(raygen_shader->function_name);
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
      raygen_shader->function_name, args, gridDim, blockDim, context);
    grid->vulkan_metadata.raygen_sbt = raygen_sbt;
    grid->vulkan_metadata.miss_sbt = miss_sbt;
    grid->vulkan_metadata.hit_sbt = hit_sbt;
    grid->vulkan_metadata.callable_sbt = callable_sbt;
    grid->vulkan_metadata.raygen_sbt_device_addr = raygen_sbt_device_addr;
    grid->vulkan_metadata.miss_sbt_device_addr = miss_sbt_device_addr;
    grid->vulkan_metadata.hit_sbt_device_addr = hit_sbt_device_addr;
    grid->vulkan_metadata.callable_sbt_device_addr = callable_sbt_device_addr;
    grid->vulkan_metadata.raygen_sbt_stride = raygen_sbt_stride;
    grid->vulkan_metadata.raygen_sbt_size = raygen_sbt_size;
    grid->vulkan_metadata.miss_sbt_stride = miss_sbt_stride;
    grid->vulkan_metadata.miss_sbt_size = miss_sbt_size;
    grid->vulkan_metadata.hit_sbt_stride = hit_sbt_stride;
    grid->vulkan_metadata.hit_sbt_size = hit_sbt_size;
    grid->vulkan_metadata.callable_sbt_stride = callable_sbt_stride;
    grid->vulkan_metadata.callable_sbt_size = callable_sbt_size;
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

bool VulkanRayTracing::rtcoreLoadCompatibilitySbtShaderId(
    const void *base, uint64_t stride, uint64_t size, uint32_t record_index,
    uint32_t component_index, uint32_t *shader_id) {
    if (getenv("VULKAN_SIM_RTCORE_TEST_COMPAT_SBT_BOUNDS") != NULL) {
        size = 1;
    }
    if (base == NULL || shader_id == NULL || stride == 0 || size == 0) {
        return false;
    }
    const uint64_t component_offset =
        static_cast<uint64_t>(component_index) * sizeof(uint32_t);
    if (component_offset + sizeof(uint32_t) > stride ||
        record_index > UINT64_MAX / stride) {
        return false;
    }
    const uint64_t record_offset = static_cast<uint64_t>(record_index) * stride;
    if (record_offset > size || component_offset > size - record_offset ||
        sizeof(uint32_t) > size - record_offset - component_offset) {
        return false;
    }
    memcpy(shader_id,
           static_cast<const uint8_t *>(base) + record_offset +
               component_offset,
           sizeof(*shader_id));
    if (getenv("VULKAN_SIM_RTCORE_TEST_COMPAT_SBT_SHADER_ID_OOB") != NULL) {
        *shader_id = shaders.size();
    }
    return *shader_id < shaders.size();
}

static bool rtcore_require_compat_sbt_shader_id(
    const ptx_instruction *pI, const char *region_name, const void *base,
    uint64_t stride, uint64_t size, uint32_t record_index,
    uint32_t component_index, uint32_t *shader_id) {
    if (VulkanRayTracing::rtcoreLoadCompatibilitySbtShaderId(
            base, stride, size, record_index, component_index, shader_id)) {
        return true;
    }
    printf("GPGPU-Sim PTX: compatibility SBT fail-closed (%s:%u), "
           "region=%s, base=%p, stride=%llu, size=%llu, "
           "record_index=%u, component_index=%u\n",
           pI->source_file(), pI->source_line(), region_name, base,
           (unsigned long long)stride, (unsigned long long)size,
           record_index, component_index);
    fflush(stdout);
    abort();
    return false;
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

    const vulkan_kernel_metadata &metadata = thread->get_kernel().vulkan_metadata;
    uint32_t shaderID = 0;
    if (!rtcore_require_compat_sbt_shader_id(
            pI, "miss", metadata.miss_sbt, metadata.miss_sbt_stride,
            metadata.miss_sbt_size, missIndex, 0, &shaderID)) {
        return;
    }
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
        const vulkan_kernel_metadata &metadata = thread->get_kernel().vulkan_metadata;
        uint32_t shaderID = 0;
        if (!rtcore_require_compat_sbt_shader_id(
                pI, "hit", metadata.hit_sbt, metadata.hit_sbt_stride,
                metadata.hit_sbt_size, 0, 0, &shaderID)) {
            return;
        }
        closesthit_shader = shaders[shaderID];
        VSIM_DPRINTF("gpgpusim: Calling Closest Hit Shader at ID %d\n", shaderID);

    }
    else {
        int32_t hitGroupIndex;
        mem->read(&(traversal_data->closest_hit.hitGroupIndex), sizeof(traversal_data->closest_hit.hitGroupIndex), &hitGroupIndex);
        const vulkan_kernel_metadata &metadata = thread->get_kernel().vulkan_metadata;
        uint32_t shaderID = 0;
        if (!rtcore_require_compat_sbt_shader_id(
                pI, "hit", metadata.hit_sbt, metadata.hit_sbt_stride,
                metadata.hit_sbt_size, hitGroupIndex, 0, &shaderID)) {
            return;
        }
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

    const vulkan_kernel_metadata &metadata = thread->get_kernel().vulkan_metadata;
    uint32_t shaderID = 0;
    if (!rtcore_require_compat_sbt_shader_id(
            pI, "hit", metadata.hit_sbt, metadata.hit_sbt_stride,
            metadata.hit_sbt_size, hitGroupIndex, 1, &shaderID)) {
        return;
    }
    shader_stage_info intersection_shader = shaders[shaderID];
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

    const vulkan_kernel_metadata &metadata = thread->get_kernel().vulkan_metadata;
    uint32_t shaderID = 0;
    if (!rtcore_require_compat_sbt_shader_id(
            pI, "hit", metadata.hit_sbt, metadata.hit_sbt_stride,
            metadata.hit_sbt_size, hitGroupIndex, 1, &shaderID)) {
        return;
    }
    shader_stage_info anyhit_shader = shaders[shaderID];
    function_info *entry = context->get_kernel(anyhit_shader.function_name);
    callShader(pI, thread, entry);
}

function_info *VulkanRayTracing::rtcoreResolveCompatibilityShaderFunction(
    uint32_t shaderID) {
    if (shaderID >= shaders.size()) {
        return NULL;
    }
    gpgpu_context *ctx = GPGPU_Context();
    if (ctx == NULL) {
        return NULL;
    }
    CUctx_st *context = GPGPUSim_Context(ctx);
    if (context == NULL) {
        return NULL;
    }
    return context->get_kernel(shaders[shaderID].function_name);
}

extern "C" function_info *rtcore_resolve_compatibility_shader_function(
    unsigned shaderID) {
    return VulkanRayTracing::rtcoreResolveCompatibilityShaderFunction(shaderID);
}

int VulkanRayTracing::rtcoreCompatibilityShaderTargetKind(unsigned shader_id,
                                                           unsigned reason) {
    if (shader_id == VK_SHADER_UNUSED_KHR) {
        return 0;
    }
    if (shader_id >= VulkanRayTracing::shaders.size()) {
        return -1;
    }
    gl_shader_stage expected_stage = MESA_SHADER_NONE;
    switch (reason) {
    case RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS:
        expected_stage = MESA_SHADER_MISS;
        break;
    case RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY:
        expected_stage = MESA_SHADER_CLOSEST_HIT;
        break;
    case RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED:
        expected_stage = MESA_SHADER_ANY_HIT;
        break;
    case RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED:
        expected_stage = MESA_SHADER_INTERSECTION;
        break;
    default:
        break;
    }
    if (expected_stage == MESA_SHADER_NONE ||
        VulkanRayTracing::shaders[shader_id].type != expected_stage) {
        return -1;
    }
    return 1;
}

extern "C" int rtcore_compatibility_shader_target_kind(unsigned shader_id,
                                                         unsigned reason) {
    return VulkanRayTracing::rtcoreCompatibilityShaderTargetKind(shader_id,
                                                                  reason);
}

static bool rtcore_apply_v04_shader_builtin_compatibility_test_mutation(
    std::array<uint32_t, rtcore::abi_v04::kWordCount> *actual) {
    const char *mutation = getenv(
        "VULKAN_SIM_RTCORE_TEST_V04_SHADER_BUILTIN_COMPATIBILITY_MUTATION");
    if (mutation == NULL || mutation[0] == '\0' ||
        strcmp(mutation, "none") == 0) {
        return true;
    }
    if (actual == NULL) {
        return false;
    }

    unsigned word = rtcore::abi_v04::kWordCount;
    if (strcmp(mutation, "launch_tmax") == 0) {
        word = rtcore::abi_v04::kLaunchRayTmaxFp32.word;
    } else if (strcmp(mutation, "geometry_index") == 0) {
        word = rtcore::abi_v04::kGeometryIndex.word;
    } else if (strcmp(mutation, "primitive_index") == 0) {
        word = rtcore::abi_v04::kPrimitiveIndex.word;
    } else if (strcmp(mutation, "instance_index") == 0) {
        word = rtcore::abi_v04::kInstanceIndex.word;
    } else if (strcmp(mutation, "instance_custom_index") == 0) {
        word = rtcore::abi_v04::kInstanceCustomIndex.word;
    } else if (strcmp(mutation, "hit_kind") == 0) {
        word = rtcore::abi_v04::kHitKind.word;
    } else {
        return false;
    }
    (*actual)[word] ^= 1u;
    return true;
}

extern "C" int rtcore_validate_v04_shader_builtin_compatibility_context(
    const ptx_instruction *source_inst, ptx_thread_info *thread,
    unsigned reason, unsigned lane_slot_index,
    unsigned long long handoff_window_base) {
    if (source_inst == NULL || thread == NULL ||
        thread->RT_thread_data == NULL ||
        thread->RT_thread_data->traversal_data.empty() ||
        lane_slot_index >= 32 || handoff_window_base == 0) {
        return 0;
    }

    const unsigned long long lane_address =
        handoff_window_base +
        static_cast<unsigned long long>(lane_slot_index) *
            rtcore::abi_v04::kLaneSlotBytes;
    if (lane_address < handoff_window_base) {
        return 0;
    }

    std::array<uint32_t, rtcore::abi_v04::kWordCount> actual = {};
    memory_space *mem = thread->get_global_memory();
    mem->read_simulator_backing(lane_address, sizeof(actual), actual.data());
    if (!rtcore_apply_v04_shader_builtin_compatibility_test_mutation(
            &actual)) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_SHADER_BUILTIN_COMPATIBILITY_FAULT "
                "thread_uid=%u lane_id=%u reason=%u "
                "fault=unknown_test_mutation\n",
                thread->get_uid(), lane_slot_index, reason);
        fflush(stderr);
        return 0;
    }

    Traversal_data *traversal_data =
        thread->RT_thread_data->traversal_data.back();
    float3 world_origin = {};
    float3 world_direction = {};
    float ray_tmin = 0.0f;
    float launch_ray_tmax = 0.0f;
    uint32_t ray_flags = 0;
    uint32_t cull_mask = 0;
    bool hit_geometry = false;
    Hit_data closest_hit = {};
    mem->read(&(traversal_data->ray_world_origin), sizeof(world_origin),
              &world_origin);
    mem->read(&(traversal_data->ray_world_direction), sizeof(world_direction),
              &world_direction);
    mem->read(&(traversal_data->Tmin), sizeof(ray_tmin), &ray_tmin);
    mem->read(&(traversal_data->Tmax), sizeof(launch_ray_tmax),
              &launch_ray_tmax);
    mem->read(&(traversal_data->rayFlags), sizeof(ray_flags), &ray_flags);
    mem->read(&(traversal_data->cullMask), sizeof(cull_mask), &cull_mask);
    mem->read(&(traversal_data->hit_geometry), sizeof(hit_geometry),
              &hit_geometry);
    if (hit_geometry) {
        mem->read(&(traversal_data->closest_hit), sizeof(closest_hit),
                  &closest_hit);
    }

    const uint32_t expected_words[] = {
        rtcore_v04_fp32_bits(world_origin.x),
        rtcore_v04_fp32_bits(world_origin.y),
        rtcore_v04_fp32_bits(world_origin.z),
        rtcore_v04_fp32_bits(ray_tmin),
        rtcore_v04_fp32_bits(world_direction.x),
        rtcore_v04_fp32_bits(world_direction.y),
        rtcore_v04_fp32_bits(world_direction.z),
    };
    const unsigned actual_word_indices[] = {4, 5, 6, 7, 8, 9, 10};
    uint32_t mismatch_mask = 0;
    for (unsigned index = 0;
         index < sizeof(expected_words) / sizeof(expected_words[0]); ++index) {
        if (actual[actual_word_indices[index]] != expected_words[index]) {
            mismatch_mask |= uint32_t{1} << index;
        }
    }
    if (actual[rtcore::abi_v04::kRayFlags.word] != ray_flags) {
        mismatch_mask |= uint32_t{1} << 8;
    }
    if (rtcore::abi_v04::extract_field(actual,
                                       rtcore::abi_v04::kCullMask) !=
        (cull_mask & 0xffu)) {
        mismatch_mask |= uint32_t{1} << 9;
    }
    if (actual[rtcore::abi_v04::kLaunchRayTmaxFp32.word] !=
        rtcore_v04_fp32_bits(launch_ray_tmax)) {
        mismatch_mask |= uint32_t{1} << 12;
    }

    const bool boundary_tmax_is_consumed =
        reason != RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS;
    const float legacy_ray_tmax =
        [&]() {
            uint32_t override_valid = 0;
            uint32_t override_bits = 0;
            mem->read(&(traversal_data->current_shader_ray_tmax_valid),
                      sizeof(override_valid), &override_valid);
            if (override_valid != 0) {
                mem->read(&(traversal_data->current_shader_ray_tmax_fp32),
                          sizeof(override_bits), &override_bits);
                float override_value = 0.0f;
                memcpy(&override_value, &override_bits,
                       sizeof(override_value));
                return override_value;
            }
            return hit_geometry ? closest_hit.world_min_thit
                                : launch_ray_tmax;
        }();
    if (boundary_tmax_is_consumed &&
        actual[rtcore::abi_v04::kBoundaryRayTmaxFp32.word] !=
            rtcore_v04_fp32_bits(legacy_ray_tmax)) {
        mismatch_mask |= uint32_t{1} << 7;
    }

    const bool terminal_closest_hit =
        reason ==
        RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY;
    const bool anyhit =
        reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED;
    const bool intersection =
        reason ==
        RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
    const bool identity_fields_consumed =
        terminal_closest_hit || anyhit || intersection;
    const bool hit_kind_consumed = terminal_closest_hit || anyhit;
    uint32_t expected_geometry_index = 0;
    uint32_t expected_primitive_index = 0;
    uint32_t expected_instance_index = 0;
    uint32_t expected_instance_custom_index = 0;
    uint32_t expected_hit_kind = 0;
    bool identity_source_valid = !identity_fields_consumed;
    if (terminal_closest_hit) {
        identity_source_valid = hit_geometry;
        if (identity_source_valid) {
            expected_geometry_index = closest_hit.geometry_index;
            expected_primitive_index = closest_hit.primitive_index;
            expected_instance_index = closest_hit.instance_id;
            expected_instance_custom_index = closest_hit.instance_index;
            expected_hit_kind = closest_hit.hit_kind & 0xffu;
        }
    } else if (anyhit || intersection) {
        int32_t shader_counter = -1;
        int32_t shader_type = -1;
        mem->read(&(traversal_data->current_shader_counter),
                  sizeof(shader_counter), &shader_counter);
        mem->read(&(traversal_data->current_shader_type),
                  sizeof(shader_type), &shader_type);
        const int32_t expected_shader_type = intersection ? 1 : 2;
        const uint32_t cta_x = thread->get_ctaid().x;
        const uint32_t cta_y = thread->get_ctaid().y;
        warp_intersection_table *table =
            intersection ? VulkanRayTracing::intersection_table[cta_x][cta_y]
                         : VulkanRayTracing::anyhit_table[cta_x][cta_y];
        const uint32_t tid = thread->get_tid().x;
        identity_source_valid =
            shader_counter >= 0 && shader_type == expected_shader_type &&
            table != NULL &&
            static_cast<unsigned>(shader_counter) <
                INTERSECTION_TABLE_MAX_LENGTH &&
            table->shader_exists(tid, static_cast<unsigned>(shader_counter),
                                 source_inst, thread);
        if (identity_source_valid) {
            const unsigned counter = static_cast<unsigned>(shader_counter);
            expected_geometry_index = table->get_geometryID(
                counter, tid, source_inst, thread);
            expected_primitive_index = table->get_primitiveID(
                counter, tid, source_inst, thread);
            expected_instance_index = table->get_instanceIndex(
                counter, tid, source_inst, thread);
            expected_instance_custom_index = table->get_instanceID(
                counter, tid, source_inst, thread);
            if (anyhit) {
                identity_source_valid =
                    counter < thread->RT_thread_data->all_hit_data.size() &&
                    thread->RT_thread_data->all_hit_data[counter] != NULL;
                if (identity_source_valid) {
                    Hit_data candidate = {};
                    mem->read(thread->RT_thread_data->all_hit_data[counter],
                              sizeof(candidate), &candidate);
                    expected_hit_kind = candidate.hit_kind & 0xffu;
                }
            }
        }
    }
    if (identity_fields_consumed) {
        if (!identity_source_valid ||
            actual[rtcore::abi_v04::kPrimitiveIndex.word] !=
                expected_primitive_index) {
            mismatch_mask |= uint32_t{1} << 10;
        }
        if (!identity_source_valid ||
            actual[rtcore::abi_v04::kInstanceCustomIndex.word] !=
                expected_instance_custom_index) {
            mismatch_mask |= uint32_t{1} << 11;
        }
        if (!identity_source_valid ||
            actual[rtcore::abi_v04::kGeometryIndex.word] !=
                expected_geometry_index) {
            mismatch_mask |= uint32_t{1} << 13;
        }
        if (!identity_source_valid ||
            actual[rtcore::abi_v04::kInstanceIndex.word] !=
                expected_instance_index) {
            mismatch_mask |= uint32_t{1} << 14;
        }
    }
    if (hit_kind_consumed &&
        (!identity_source_valid ||
         rtcore::abi_v04::extract_field(actual,
                                        rtcore::abi_v04::kHitKind) !=
             expected_hit_kind)) {
        mismatch_mask |= uint32_t{1} << 15;
    }

    if (mismatch_mask != 0) {
        fprintf(stderr,
                "GPGPU-Sim RTCORE_V04_SHADER_BUILTIN_COMPATIBILITY_FAULT "
                "thread_uid=%u lane_id=%u reason=%u lane_address=0x%llx "
                "mismatch_mask=0x%08x "
                "handoff_origin={0x%08x,0x%08x,0x%08x} "
                "legacy_origin={0x%08x,0x%08x,0x%08x} "
                "handoff_direction={0x%08x,0x%08x,0x%08x} "
                "legacy_direction={0x%08x,0x%08x,0x%08x} "
                "handoff_tmin=0x%08x legacy_tmin=0x%08x "
                "handoff_launch_tmax=0x%08x legacy_launch_tmax=0x%08x "
                "handoff_boundary_tmax=0x%08x legacy_tmax=0x%08x "
                "handoff_flags=0x%08x legacy_flags=0x%08x "
                "handoff_cull=0x%02x legacy_cull=0x%02x "
                "handoff_geometry=%u legacy_geometry=%u "
                "handoff_primitive=%u legacy_primitive=%u "
                "handoff_instance=%u legacy_instance=%u "
                "handoff_instance_custom=%u legacy_instance_custom=%u "
                "handoff_hit_kind=%u legacy_hit_kind=%u\n",
                thread->get_uid(), lane_slot_index, reason, lane_address,
                mismatch_mask, actual[4], actual[5], actual[6],
                expected_words[0], expected_words[1], expected_words[2],
                actual[8], actual[9], actual[10], expected_words[4],
                expected_words[5], expected_words[6], actual[7],
                expected_words[3],
                actual[rtcore::abi_v04::kLaunchRayTmaxFp32.word],
                rtcore_v04_fp32_bits(launch_ray_tmax),
                actual[rtcore::abi_v04::kBoundaryRayTmaxFp32.word],
                rtcore_v04_fp32_bits(legacy_ray_tmax),
                actual[rtcore::abi_v04::kRayFlags.word], ray_flags,
                rtcore::abi_v04::extract_field(actual,
                                               rtcore::abi_v04::kCullMask),
                cull_mask & 0xffu,
                actual[rtcore::abi_v04::kGeometryIndex.word],
                expected_geometry_index,
                actual[rtcore::abi_v04::kPrimitiveIndex.word],
                expected_primitive_index,
                actual[rtcore::abi_v04::kInstanceIndex.word],
                expected_instance_index,
                actual[rtcore::abi_v04::kInstanceCustomIndex.word],
                expected_instance_custom_index,
                rtcore::abi_v04::extract_field(actual,
                                               rtcore::abi_v04::kHitKind),
                expected_hit_kind);
        fflush(stderr);
        return 0;
    }
    return 1;
}

extern "C" int rtcore_prepare_compatibility_shader_continuation_context(
    const ptx_instruction *pI, ptx_thread_info *thread, unsigned reason,
    unsigned hit_record_selector, unsigned boundary_event_seq,
    unsigned boundary_shader_counter,
    unsigned long long boundary_hit_data_ref,
    unsigned boundary_hit_group_index, unsigned boundary_geometry_type,
    unsigned boundary_geometry_index, unsigned primitive_index,
    unsigned instance_index, unsigned hit_kind,
    unsigned boundary_ray_tmax_fp32, bool boundary_ray_tmax_valid) {
    if (pI == NULL || thread == NULL || thread->RT_thread_data == NULL ||
        thread->RT_thread_data->traversal_data.empty()) {
        return 0;
    }

    const bool intersection =
        reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_INTERSECTION_REQUIRED;
    const bool anyhit =
        reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_ANY_HIT_REQUIRED;
    const bool terminal_miss =
        reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_MISS;
    const bool terminal_closest_hit =
        reason == RTCORE_REPLAY_CONTINUATION_PACKET_REASON_CLOSEST_HIT_READY;
    if (!intersection && !anyhit && !terminal_miss &&
        !terminal_closest_hit) {
        return 0;
    }

    const uint32_t tid = thread->get_tid().x;
    Traversal_data *traversal_data =
        thread->RT_thread_data->traversal_data.back();
    memory_space *mem = thread->get_global_memory();

    if (terminal_miss || terminal_closest_hit) {
        bool hit_geometry = false;
        uint32_t miss_index = 0;
        Hit_data closest_hit = {};
        const bool v04_functional_authority =
            rtcore_v04_functional_shader_return_authority_enabled();
        const char *terminal_context_source =
            v04_functional_authority ? "v04_terminal_projection"
                                     : "terminal_traversal_state";

        if (v04_functional_authority) {
            std::map<unsigned, rtcore_replay_lane_request>::const_iterator
                request = g_rtcore_replay_lane_requests.find(
                    thread->get_uid());
            const char *projection_failure = "accepted";
            if (request == g_rtcore_replay_lane_requests.end() ||
                !request->second.valid ||
                request->second.thread_uid != thread->get_uid() ||
                !request->second.v04_shadow_boundary_enabled ||
                !request->second.v04_shadow_trace_input_valid ||
                !rtcore_v04_project_terminal_to_compatibility_hit(
                    request->second, mem,
                    request->second.v04_replay_committed_boundary_values,
                    &hit_geometry, &closest_hit, &projection_failure)) {
                fprintf(stderr,
                        "GPGPU-Sim "
                        "RTCORE_SHADER_CONTINUATION_COMPAT_CONTEXT_FAULT "
                        "lane_id=%u reason=%u "
                        "fault=v04_terminal_projection_invalid "
                        "projection_failure=%s\n",
                        tid, reason, projection_failure);
                fflush(stderr);
                return 0;
            }
            const std::array<uint32_t, rtcore::abi_v04::kWordCount> &
                trace_words = request->second.v04_shadow_trace_input_words;
            miss_index = rtcore::abi_v04::extract_field(
                trace_words, rtcore::abi_v04::kMissIndex);
        } else {
            mem->read(&(traversal_data->hit_geometry),
                      sizeof(traversal_data->hit_geometry), &hit_geometry);
            mem->read(&(traversal_data->missIndex),
                      sizeof(traversal_data->missIndex), &miss_index);
            if (terminal_closest_hit) {
                mem->read(&(traversal_data->closest_hit),
                          sizeof(traversal_data->closest_hit), &closest_hit);
            }
        }

        bool terminal_state_valid = false;
        uint64_t expected_selector = miss_index;
        unsigned expected_geometry_type = 0;
        if (terminal_miss) {
            terminal_state_valid =
                !hit_geometry && hit_record_selector == miss_index;
        } else {
            uint32_t sbt_record_offset = 0;
            uint32_t sbt_record_stride = 0;
            if (v04_functional_authority) {
                const rtcore_replay_lane_request &request =
                    g_rtcore_replay_lane_requests.find(thread->get_uid())
                        ->second;
                sbt_record_offset = rtcore::abi_v04::extract_field(
                    request.v04_shadow_trace_input_words,
                    rtcore::abi_v04::kSbtRecordOffset);
                sbt_record_stride = rtcore::abi_v04::extract_field(
                    request.v04_shadow_trace_input_words,
                    rtcore::abi_v04::kSbtRecordStride);
            } else {
                mem->read(&(traversal_data->sbtRecordOffset),
                          sizeof(traversal_data->sbtRecordOffset),
                          &sbt_record_offset);
                mem->read(&(traversal_data->sbtRecordStride),
                          sizeof(traversal_data->sbtRecordStride),
                          &sbt_record_stride);
            }
            expected_selector =
                closest_hit.hitGroupIndex < 0
                    ? UINT64_MAX
                    : static_cast<uint64_t>(sbt_record_offset) +
                          static_cast<uint64_t>(closest_hit.geometry_index) *
                              sbt_record_stride +
                          static_cast<uint32_t>(closest_hit.hitGroupIndex);
            expected_geometry_type =
                closest_hit.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR
                    ? 1u
                    : 2u;
            terminal_state_valid =
                hit_geometry && expected_selector <= UINT_MAX &&
                hit_record_selector == expected_selector &&
                boundary_hit_group_index ==
                    static_cast<unsigned>(closest_hit.hitGroupIndex) &&
                boundary_geometry_type == expected_geometry_type &&
                boundary_geometry_index == closest_hit.geometry_index &&
                primitive_index == closest_hit.primitive_index &&
                instance_index == closest_hit.instance_index &&
                hit_kind == closest_hit.hit_kind;
        }
        if (!terminal_state_valid) {
            fprintf(stderr,
                    "GPGPU-Sim "
                    "RTCORE_SHADER_CONTINUATION_COMPAT_CONTEXT_FAULT "
                    "lane_id=%u reason=%u hit_record_selector=%u "
                    "expected_selector=%llu miss_index=%u hit_geometry=%u "
                    "packet_hit_group_index=%u traversal_hit_group_index=%d "
                    "packet_geometry_type=%u traversal_geometry_type=%u "
                    "packet_geometry_index=%u traversal_geometry_index=%u "
                    "packet_primitive_index=%u traversal_primitive_index=%u "
                    "packet_instance_index=%u traversal_instance_index=%u "
                    "packet_hit_kind=%u traversal_hit_kind=%u "
                    "terminal_context_source=%s "
                    "fault=terminal_traversal_state_mismatch\n",
                    tid, reason, hit_record_selector,
                    static_cast<unsigned long long>(expected_selector),
                    miss_index, hit_geometry ? 1u : 0u,
                    boundary_hit_group_index,
                    terminal_closest_hit ? closest_hit.hitGroupIndex : -1,
                    boundary_geometry_type,
                    terminal_closest_hit ? expected_geometry_type : 0u,
                    boundary_geometry_index,
                    terminal_closest_hit ? closest_hit.geometry_index : 0u,
                    primitive_index,
                    terminal_closest_hit ? closest_hit.primitive_index : 0u,
                    instance_index,
                    terminal_closest_hit ? closest_hit.instance_index : 0u,
                    hit_kind,
                    terminal_closest_hit ? closest_hit.hit_kind : 0u,
                    terminal_context_source);
            fflush(stderr);
            return 0;
        }

        if (v04_functional_authority) {
            mem->write(&(traversal_data->hit_geometry),
                       sizeof(traversal_data->hit_geometry), &hit_geometry,
                       thread, pI);
            if (hit_geometry) {
                mem->write(&(traversal_data->closest_hit),
                           sizeof(traversal_data->closest_hit), &closest_hit,
                           thread, pI);
                if (closest_hit.geometryType ==
                    VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                    thread->RT_thread_data->set_hitAttribute(
                        closest_hit.barycentric_coordinates, pI, thread);
                }
            }
            printf("GPGPU-Sim "
                   "RTCORE_V04_FUNCTIONAL_TERMINAL_CONTEXT_PROJECTION "
                   "lane_id=%u reason=%u hit_geometry=%u "
                   "hit_record_selector=%u compatibility_storage_write=1 "
                   "terminal_authority=v04_retained_boundary_values\n",
                   tid, reason, hit_geometry ? 1u : 0u,
                   hit_record_selector);
            fflush(stdout);
        }

        const int32_t current_shader_counter = -1;
        const int32_t current_shader_type = -1;
        mem->write(&(traversal_data->current_shader_counter),
                   sizeof(traversal_data->current_shader_counter),
                   &current_shader_counter, thread, pI);
        mem->write(&(traversal_data->current_shader_type),
                   sizeof(traversal_data->current_shader_type),
                   &current_shader_type, thread, pI);
        const uint32_t current_shader_ray_tmax_valid = 0;
        mem->write(&(traversal_data->current_shader_ray_tmax_valid),
                   sizeof(traversal_data->current_shader_ray_tmax_valid),
                   &current_shader_ray_tmax_valid, thread, pI);
        printf("GPGPU-Sim RTCORE_SHADER_CONTINUATION_COMPAT_CONTEXT_PREPARE "
               "lane_id=%u reason=%u hit_record_selector=%u "
               "terminal_kind=%s shader_counter=-1 shader_type=-1 "
               "context_source=%s\n",
               tid, reason, hit_record_selector,
               terminal_miss ? "terminal_miss" : "terminal_closest_hit",
               terminal_context_source);
        fflush(stdout);
        return 1;
    }

    const uint32_t cta_x = thread->get_ctaid().x;
    const uint32_t cta_y = thread->get_ctaid().y;
    warp_intersection_table *table =
        intersection ? VulkanRayTracing::intersection_table[cta_x][cta_y]
                     : VulkanRayTracing::anyhit_table[cta_x][cta_y];
    const uint32_t shader_counter = boundary_shader_counter;
    const bool table_identity_valid =
        table != NULL && shader_counter < INTERSECTION_TABLE_MAX_LENGTH &&
        table->shader_exists(tid, shader_counter, pI, thread) &&
        table->get_hitGroupIndex(shader_counter, tid, pI, thread) ==
            boundary_hit_group_index &&
        table->get_primitiveID(shader_counter, tid, pI, thread) ==
            primitive_index &&
        table->get_instanceID(shader_counter, tid, pI, thread) ==
            instance_index;
    if (!table_identity_valid) {
        fprintf(stderr,
                "GPGPU-Sim "
                "RTCORE_SHADER_CONTINUATION_COMPAT_CONTEXT_FAULT "
                "lane_id=%u reason=%u hit_record_selector=%u "
                "boundary_event_seq=%u boundary_shader_counter=%u "
                "boundary_hit_group_index=%u primitive_index=%u "
                "instance_index=%u hit_kind=%u "
                "fault=event_local_table_identity_mismatch\n",
                tid, reason, hit_record_selector, boundary_event_seq,
                shader_counter, boundary_hit_group_index, primitive_index,
                instance_index, hit_kind);
        fflush(stderr);
        return 0;
    }

    const int32_t current_shader_counter =
        static_cast<int32_t>(shader_counter);
    const int32_t current_shader_type = intersection ? 1 : 2;
    mem->write(&(traversal_data->current_shader_counter),
               sizeof(traversal_data->current_shader_counter),
               &current_shader_counter, thread, pI);
    mem->write(&(traversal_data->current_shader_type),
               sizeof(traversal_data->current_shader_type),
               &current_shader_type, thread, pI);
    const uint32_t current_shader_ray_tmax_valid =
        boundary_ray_tmax_valid ? 1u : 0u;
    mem->write(&(traversal_data->current_shader_ray_tmax_fp32),
               sizeof(traversal_data->current_shader_ray_tmax_fp32),
               &boundary_ray_tmax_fp32, thread, pI);
    mem->write(&(traversal_data->current_shader_ray_tmax_valid),
               sizeof(traversal_data->current_shader_ray_tmax_valid),
               &current_shader_ray_tmax_valid, thread, pI);

    if (anyhit) {
        if (shader_counter >= thread->RT_thread_data->all_hit_data.size() ||
            boundary_hit_data_ref == 0 ||
            reinterpret_cast<uint64_t>(
                thread->RT_thread_data->all_hit_data[shader_counter]) !=
                boundary_hit_data_ref) {
            return 0;
        }
        Hit_data hit_attributes;
        mem->read(reinterpret_cast<void *>(boundary_hit_data_ref),
                  sizeof(hit_attributes), &hit_attributes);
        if (boundary_geometry_type != 1 ||
            hit_attributes.geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR ||
            hit_attributes.geometry_index != boundary_geometry_index ||
            hit_attributes.primitive_index != primitive_index ||
            hit_attributes.instance_index != instance_index ||
            hit_attributes.hitGroupIndex != boundary_hit_group_index ||
            hit_attributes.hit_kind != hit_kind) {
            return 0;
        }
        thread->RT_thread_data->set_hitAttribute(
            hit_attributes.barycentric_coordinates, pI, thread);
    } else if (boundary_geometry_type != 2 || boundary_hit_data_ref != 0) {
        return 0;
    }

    printf("GPGPU-Sim RTCORE_SHADER_CONTINUATION_COMPAT_CONTEXT_PREPARE "
           "lane_id=%u reason=%u shader_counter=%u shader_type=%d "
           "hit_record_selector=%u boundary_event_seq=%u "
           "boundary_shader_counter=%u boundary_hit_data_ref=0x%llx "
           "boundary_hit_group_index=%u boundary_geometry_type=%u "
           "boundary_geometry_index=%u primitive_index=%u instance_index=%u "
           "hit_kind=%u boundary_ray_tmax_fp32=0x%08x "
           "boundary_ray_tmax_valid=%u "
           "context_source=boundary_event_local_candidate\n",
           tid, reason, shader_counter, current_shader_type,
           hit_record_selector, boundary_event_seq, shader_counter,
           boundary_hit_data_ref, boundary_hit_group_index,
           boundary_geometry_type, boundary_geometry_index, primitive_index,
           instance_index, hit_kind, boundary_ray_tmax_fp32,
           boundary_ray_tmax_valid ? 1u : 0u);
    fflush(stdout);
    return 1;
}

extern "C" int rtcore_call_compatibility_shader_function(
    const ptx_instruction *pI, ptx_thread_info *thread,
    function_info *target_func) {
    if (pI == NULL || thread == NULL || target_func == NULL) {
        return 0;
    }
    VulkanRayTracing::callShader(pI, thread, target_func);
    return 1;
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

void VulkanRayTracing::allocBLAS(void* objectKey, void* rootAddr,
                                 uint64_t bufferSize, void* gpgpusimAddr) {
    printf("gpgpusim: set BLAS address for 0x%lx at %p to %p\n", bufferSize, rootAddr, gpgpusimAddr);
    blas_addr_map[rootAddr] = gpgpusimAddr;
    if (rtcore_v04_producer_backed_blas_root_descriptor_enabled() &&
        !rtcore_v04_typed_blas_decode_context_bridge_enabled()) {
        rtcore_fail_blas_binding("root_descriptor_requires_context_bridge",
                                (uint64_t)rootAddr,
                                (uint64_t)gpgpusimAddr, bufferSize,
                                (uint64_t)objectKey);
    }
    if (!rtcore_v04_typed_blas_decode_context_bridge_enabled()) {
        return;
    }

    const uint64_t driver_object_key = (uint64_t)objectKey;
    const uint64_t host_root_address = (uint64_t)rootAddr;
    const uint64_t device_base_address = (uint64_t)gpgpusimAddr;
    rtcore_blas_binding_snapshot snapshot;
    const char *failure_reason = "unvalidated";
    if (!g_rtcore_blas_binding_registry.register_binding(
            driver_object_key, host_root_address, device_base_address,
            bufferSize, &snapshot, &failure_reason)) {
        rtcore_fail_blas_binding(failure_reason, host_root_address,
                                device_base_address, bufferSize,
                                driver_object_key);
    }
    printf("GPGPU-Sim RTCORE_BLAS_BINDING_REGISTERED "
           "object_id=%llu generation=%u driver_object_key=0x%llx "
           "host_root=0x%llx device_base=0x%llx size=%llu live=1\n",
           (unsigned long long)snapshot.object_id, snapshot.generation,
           (unsigned long long)driver_object_key,
           (unsigned long long)snapshot.host_root_address,
           (unsigned long long)snapshot.device_base_address,
           (unsigned long long)snapshot.size_bytes);
    fflush(stdout);
}

void VulkanRayTracing::publishBLASRootDescriptor(
    void* objectKey, uint64_t rootPayloadOffset, uint32_t rootPayloadKind) {
    namespace typed_blas = rtcore::v04::typed_blas;
    if (!rtcore_v04_producer_backed_blas_root_descriptor_enabled()) {
        return;
    }
    if (!rtcore_v04_typed_blas_decode_context_bridge_enabled()) {
        rtcore_fail_blas_binding("root_descriptor_requires_context_bridge",
                                0, 0, 0, (uint64_t)objectKey);
    }
    if (rootPayloadKind > UINT8_MAX ||
        (rootPayloadKind != typed_blas::kInternalPayloadKind &&
         rootPayloadKind != typed_blas::kProceduralPayloadKind &&
         rootPayloadKind != typed_blas::kQuadPayloadKind)) {
        rtcore_fail_blas_binding("invalid_root_payload_kind", 0, 0,
                                rootPayloadOffset, (uint64_t)objectKey);
    }

    rtcore_blas_binding_snapshot snapshot;
    const char *failure_reason = "unvalidated";
    if (!g_rtcore_blas_binding_registry.publish_root_descriptor(
            (uint64_t)objectKey, typed_blas::kGenRtDerivedProfileId,
            typed_blas::kGenRtPayloadFormatId, rootPayloadOffset,
            static_cast<uint8_t>(rootPayloadKind), &snapshot,
            &failure_reason)) {
        rtcore_fail_blas_binding(failure_reason, snapshot.host_root_address,
                                snapshot.device_base_address,
                                snapshot.size_bytes, (uint64_t)objectKey);
    }
    printf("GPGPU-Sim RTCORE_BLAS_ROOT_DESCRIPTOR_PUBLISHED "
           "object_id=%llu generation=%u build_generation=%u "
           "driver_object_key=0x%llx profile=%u payload_format=%u "
           "root_offset=%llu root_kind=%u valid=1\n",
           (unsigned long long)snapshot.object_id, snapshot.generation,
           snapshot.root_build_generation,
           (unsigned long long)(uint64_t)objectKey,
           snapshot.root_bvh_profile_id,
           snapshot.root_payload_format_id,
           (unsigned long long)snapshot.root_payload_offset,
           (unsigned)snapshot.root_payload_kind);
    fflush(stdout);
}

void VulkanRayTracing::beginTLASInstanceReferences(void* objectKey) {
    if (!rtcore_v04_producer_backed_instance_blas_reference_enabled()) {
        return;
    }
    if (!rtcore_v04_instance_blas_reference_prerequisites_enabled()) {
        rtcore_fail_instance_blas_reference(
            "instance_reference_prerequisite_missing", 0, 0, 0);
    }

    rtcore_tlas_binding_snapshot tlas;
    const char *capture_failure = "unvalidated";
    if (!g_rtcore_tlas_binding_registry.capture_by_driver_object(
            (uint64_t)objectKey, &tlas, &capture_failure)) {
        rtcore_fail_instance_blas_reference(
            capture_failure, 0, 0, 0);
    }
    const char *binding_failure = "unvalidated";
    if (!g_rtcore_tlas_binding_registry.validate(
            tlas, 0, 0, &binding_failure)) {
        rtcore_fail_instance_blas_reference(
            binding_failure, tlas.object_id, tlas.generation, 0);
    }

    uint32_t build_generation = 0;
    const char *begin_failure = "unvalidated";
    if (!g_rtcore_instance_blas_reference_registry.begin_build(
            tlas.object_id, tlas.generation, tlas.host_root_address,
            tlas.device_base_address, tlas.size_bytes,
            &build_generation, &begin_failure)) {
        rtcore_fail_instance_blas_reference(
            begin_failure, tlas.object_id, tlas.generation, 0);
    }
    printf("GPGPU-Sim RTCORE_INSTANCE_BLAS_REFERENCE_BUILD_BEGIN "
           "tlas_object_id=%llu tlas_generation=%u "
           "build_generation=%u\n",
           (unsigned long long)tlas.object_id, tlas.generation,
           build_generation);
    fflush(stdout);
}

void VulkanRayTracing::publishTLASInstanceReference(
    void* objectKey, void* instanceLeafAddress,
    const void* blasRootAddress) {
    if (!rtcore_v04_producer_backed_instance_blas_reference_enabled()) {
        return;
    }
    if (!rtcore_v04_instance_blas_reference_prerequisites_enabled()) {
        rtcore_fail_instance_blas_reference(
            "instance_reference_prerequisite_missing", 0, 0, 0);
    }

    rtcore_tlas_binding_snapshot tlas;
    rtcore_blas_binding_snapshot blas;
    const char *capture_failure = "unvalidated";
    if (!g_rtcore_tlas_binding_registry.capture_by_driver_object(
            (uint64_t)objectKey, &tlas, &capture_failure)) {
        rtcore_fail_instance_blas_reference(
            capture_failure, 0, 0, 0);
    }
    if (!g_rtcore_blas_binding_registry.capture(
            (uint64_t)blasRootAddress, &blas, &capture_failure)) {
        rtcore_fail_instance_blas_reference(
            capture_failure, tlas.object_id, tlas.generation, 0);
    }
    if (!blas.root_descriptor_valid ||
        blas.root_build_generation == 0) {
        rtcore_fail_instance_blas_reference(
            "referenced_blas_root_not_published", tlas.object_id,
            tlas.generation, 0, blas.object_id, blas.generation);
    }

    const uint64_t instance_host_address =
        (uint64_t)instanceLeafAddress;
    if (instance_host_address < tlas.host_root_address) {
        rtcore_fail_instance_blas_reference(
            "instance_host_before_tlas", tlas.object_id,
            tlas.generation, 0, blas.object_id, blas.generation);
    }
    const uint64_t instance_offset =
        instance_host_address - tlas.host_root_address;
    if (instance_offset > tlas.size_bytes ||
        uint64_t{128} > tlas.size_bytes - instance_offset) {
        rtcore_fail_instance_blas_reference(
            "instance_host_out_of_range", tlas.object_id,
            tlas.generation, 0, blas.object_id, blas.generation);
    }
    const uint64_t instance_metadata_reference =
        tlas.device_base_address + instance_offset;
    rtcore_v04_instance_blas_reference_snapshot published;
    const char *publish_failure = "unvalidated";
    if (!g_rtcore_instance_blas_reference_registry.publish(
            tlas.object_id, tlas.generation, instance_host_address,
            instance_metadata_reference, blas.object_id,
            blas.generation, blas.host_root_address, &published,
            &publish_failure)) {
        rtcore_fail_instance_blas_reference(
            publish_failure, tlas.object_id, tlas.generation,
            instance_metadata_reference, blas.object_id,
            blas.generation);
    }

    printf("GPGPU-Sim RTCORE_INSTANCE_BLAS_REFERENCE_PUBLISHED "
           "tlas_object_id=%llu tlas_generation=%u "
           "build_generation=%u instance_metadata_ref=0x%llx "
           "blas_object_id=%llu blas_generation=%u\n",
           (unsigned long long)published.tlas_object_id,
           published.tlas_generation,
           published.tlas_build_generation,
           (unsigned long long)published.instance_metadata_reference,
           (unsigned long long)published.blas_object_id,
           published.blas_generation);
    fflush(stdout);
}

void VulkanRayTracing::endTLASInstanceReferences(void* objectKey) {
    if (!rtcore_v04_producer_backed_instance_blas_reference_enabled()) {
        return;
    }
    rtcore_tlas_binding_snapshot tlas;
    const char *capture_failure = "unvalidated";
    if (!g_rtcore_tlas_binding_registry.capture_by_driver_object(
            (uint64_t)objectKey, &tlas, &capture_failure)) {
        rtcore_fail_instance_blas_reference(
            capture_failure, 0, 0, 0);
    }
    uint32_t build_generation = 0;
    uint64_t reference_count = 0;
    const char *end_failure = "unvalidated";
    if (!g_rtcore_instance_blas_reference_registry.end_build(
            tlas.object_id, tlas.generation, &build_generation,
            &reference_count, &end_failure)) {
        rtcore_fail_instance_blas_reference(
            end_failure, tlas.object_id, tlas.generation, 0);
    }
    printf("GPGPU-Sim RTCORE_INSTANCE_BLAS_REFERENCE_BUILD_END "
           "tlas_object_id=%llu tlas_generation=%u "
           "build_generation=%u references=%llu\n",
           (unsigned long long)tlas.object_id, tlas.generation,
           build_generation, (unsigned long long)reference_count);
    fflush(stdout);
}

void VulkanRayTracing::releaseBLAS(void* objectKey, void* rootAddr,
                                   void* gpgpusimAddr) {
    if (!rtcore_v04_typed_blas_decode_context_bridge_enabled()) {
        return;
    }

    std::map<void *, void *>::iterator legacy = blas_addr_map.find(rootAddr);
    if (legacy == blas_addr_map.end() || legacy->second != gpgpusimAddr) {
        rtcore_fail_blas_binding("legacy_map_release_mismatch",
                                (uint64_t)rootAddr, (uint64_t)gpgpusimAddr,
                                0, (uint64_t)objectKey);
    }

    const uint64_t driver_object_key = (uint64_t)objectKey;
    const uint64_t host_root_address = (uint64_t)rootAddr;
    const uint64_t device_base_address = (uint64_t)gpgpusimAddr;
    rtcore_blas_binding_snapshot released;
    const char *failure_reason = "unvalidated";
    if (!g_rtcore_blas_binding_registry.release_binding(
            driver_object_key, host_root_address, device_base_address,
            &released, &failure_reason)) {
        rtcore_fail_blas_binding(failure_reason, host_root_address,
                                device_base_address, 0,
                                driver_object_key);
    }
    printf("GPGPU-Sim RTCORE_BLAS_BINDING_RELEASED "
           "object_id=%llu generation=%u driver_object_key=0x%llx "
           "host_root=0x%llx device_base=0x%llx size=%llu live=0\n",
           (unsigned long long)released.object_id, released.generation,
           (unsigned long long)driver_object_key,
           (unsigned long long)released.host_root_address,
           (unsigned long long)released.device_base_address,
           (unsigned long long)released.size_bytes);
    fflush(stdout);
    blas_addr_map.erase(legacy);
}

void VulkanRayTracing::allocTLAS(void* objectKey, void* rootAddr,
                                 uint64_t bufferSize, void* gpgpusimAddr) {
    const uint64_t driver_object_key = (uint64_t)objectKey;
    const uint64_t host_root_address = (uint64_t)rootAddr;
    const uint64_t device_base_address = (uint64_t)gpgpusimAddr;
    rtcore_tlas_binding_snapshot snapshot;
    const char *failure_reason = "unvalidated";
    if (!g_rtcore_tlas_binding_registry.register_binding(
            driver_object_key, host_root_address, device_base_address,
            bufferSize, &snapshot, &failure_reason)) {
        rtcore_fail_tlas_binding(failure_reason, host_root_address,
                                device_base_address, bufferSize,
                                driver_object_key);
    }

    printf("GPGPU-Sim RTCORE_TLAS_BINDING_REGISTERED "
           "object_id=%llu generation=%u driver_object_key=0x%llx "
           "host_root=0x%llx "
           "device_base=0x%llx size=%llu live=1\n",
           (unsigned long long)snapshot.object_id, snapshot.generation,
           (unsigned long long)driver_object_key,
           (unsigned long long)snapshot.host_root_address,
           (unsigned long long)snapshot.device_base_address,
           (unsigned long long)snapshot.size_bytes);
    fflush(stdout);
    tlas_addr = gpgpusimAddr;
}

void VulkanRayTracing::releaseTLAS(void* objectKey, void* rootAddr,
                                   void* gpgpusimAddr) {
    const uint64_t driver_object_key = (uint64_t)objectKey;
    const uint64_t host_root_address = (uint64_t)rootAddr;
    const uint64_t device_base_address = (uint64_t)gpgpusimAddr;
    rtcore_tlas_binding_snapshot released;
    const char *failure_reason = "unvalidated";
    if (!g_rtcore_tlas_binding_registry.release_binding(
            driver_object_key, host_root_address, device_base_address,
            &released, &failure_reason)) {
        rtcore_fail_tlas_binding(failure_reason, host_root_address,
                                device_base_address, 0, driver_object_key);
    }
    if (rtcore_v04_producer_backed_instance_blas_reference_enabled()) {
        const char *reference_failure = "unvalidated";
        if (!g_rtcore_instance_blas_reference_registry.release(
                released.object_id, released.generation,
                &reference_failure)) {
            rtcore_fail_instance_blas_reference(
                reference_failure, released.object_id,
                released.generation, 0);
        }
    }
    printf("GPGPU-Sim RTCORE_TLAS_BINDING_RELEASED "
           "object_id=%llu generation=%u driver_object_key=0x%llx "
           "host_root=0x%llx "
           "device_base=0x%llx size=%llu live=0\n",
           (unsigned long long)released.object_id, released.generation,
           (unsigned long long)driver_object_key,
           (unsigned long long)released.host_root_address,
           (unsigned long long)released.device_base_address,
           (unsigned long long)released.size_bytes);
    fflush(stdout);
}

bool VulkanRayTracing::captureTlasBinding(
    uint64_t hostRootAddress, rtcore_tlas_binding_snapshot *snapshot,
    const char **failureReason) {
    return g_rtcore_tlas_binding_registry.capture(
        hostRootAddress, snapshot, failureReason);
}

bool VulkanRayTracing::validateTlasBinding(
    const rtcore_tlas_binding_snapshot &snapshot,
    uint64_t instanceMetadataReference, uint64_t recordSize,
    const char **failureReason) {
    return g_rtcore_tlas_binding_registry.validate(
        snapshot, instanceMetadataReference, recordSize, failureReason);
}

bool VulkanRayTracing::captureBlasBinding(
    uint64_t hostRootAddress, rtcore_blas_binding_snapshot *snapshot,
    const char **failureReason) {
    return g_rtcore_blas_binding_registry.capture(
        hostRootAddress, snapshot, failureReason);
}

bool VulkanRayTracing::validateBlasBinding(
    const rtcore_blas_binding_snapshot &snapshot,
    uint64_t payloadReference, uint64_t recordSize,
    const char **failureReason) {
    return g_rtcore_blas_binding_registry.validate(
        snapshot, payloadReference, recordSize, failureReason);
}

bool VulkanRayTracing::validateBlasLegacyAlias(
    uint64_t hostRootAddress, uint64_t deviceBaseAddress) {
    std::map<void *, void *>::const_iterator legacy = blas_addr_map.find(
        reinterpret_cast<void *>(hostRootAddress));
    return legacy != blas_addr_map.end() &&
           reinterpret_cast<uint64_t>(legacy->second) == deviceBaseAddress;
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
