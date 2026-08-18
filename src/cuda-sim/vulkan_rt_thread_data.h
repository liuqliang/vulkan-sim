#ifndef VULKAN_RT_THREAD_DATA_H
#define VULKAN_RT_THREAD_DATA_H

#include "vulkan/vulkan.h"

#if defined(MESA_USE_INTEL_DRIVER)
#include "vulkan/vulkan_intel.h"
#endif

#include "vulkan_ray_tracing.h"

// #include "ptx_ir.h"
#include "ptx_ir.h"
#include "../../libcuda/gpgpu_context.h"
#include "compiler/shader_enums.h"
#include "rtcore_v04_shader_input.h"
#include <fstream>
#include <cmath>
#include <stack>
#include <array>

#include "compiler/nir/nir.h"

static const size_t RTCORE_KHR_CCS_LOGICAL_DEPTH = 8;
static const size_t RTCORE_KHR_CCS_ATTRIBUTE_BYTES = 64;

// Fixed storage used to model one lane row of the scheduler Continuation
// Control Store (CCS).  Unlike std::vector, capacity is part of the modeled
// resource and no allocation can occur after construction.
template <typename T, size_t Capacity>
class rtcore_khr_ccs_fixed_stack {
 public:
  typedef T *iterator;
  typedef const T *const_iterator;

  rtcore_khr_ccs_fixed_stack() : depth_(0) {}

  size_t size() const { return depth_; }
  size_t capacity() const { return Capacity; }
  bool empty() const { return depth_ == 0; }
  T &back() { assert(depth_ > 0); return entries_[depth_ - 1]; }
  const T &back() const { assert(depth_ > 0); return entries_[depth_ - 1]; }
  T &operator[](size_t index) { assert(index < depth_); return entries_[index]; }
  const T &operator[](size_t index) const {
    assert(index < depth_); return entries_[index];
  }
  iterator begin() { return entries_.data(); }
  iterator end() { return entries_.data() + depth_; }
  const_iterator begin() const { return entries_.data(); }
  const_iterator end() const { return entries_.data() + depth_; }

  void push_back(const T &value) {
    assert(depth_ < Capacity);
    entries_[depth_++] = value;
  }
  void pop_back() {
    assert(depth_ > 0);
    entries_[--depth_] = T();
  }
  iterator erase(iterator position) {
    assert(position >= begin() && position < end());
    const size_t index = static_cast<size_t>(position - begin());
    for (size_t current = index + 1; current < depth_; ++current)
      entries_[current - 1] = entries_[current];
    entries_[--depth_] = T();
    return entries_.data() + index;
  }
  void clear() {
    while (depth_ > 0) entries_[--depth_] = T();
  }

 private:
  std::array<T, Capacity> entries_;
  size_t depth_;
};

class rtcore_khr_ccs_attribute_image {
 public:
  rtcore_khr_ccs_attribute_image() : size_(0) { bytes_.fill(0); }
  size_t size() const { return size_; }
  void resize(size_t size) {
    assert(size <= bytes_.size());
    size_ = size;
  }
  unsigned char *data() { return bytes_.data(); }
  const unsigned char *data() const { return bytes_.data(); }
  unsigned char operator[](size_t index) const {
    assert(index < size_);
    return bytes_[index];
  }

 private:
  std::array<unsigned char, RTCORE_KHR_CCS_ATTRIBUTE_BYTES> bytes_;
  size_t size_;
};

typedef struct variable_decleration_entry{
  nir_variable_mode type;
  std::string name;
  uint64_t address;
  uint32_t size;
} variable_decleration_entry;

typedef struct callable_data_binding_entry {
    uint64_t address;
    uint32_t size;
    function_info *callee;
    uint32_t sbt_index;
    uint32_t shader_id;
} callable_data_binding_entry;

enum report_intersection_decision {
    REPORT_INTERSECTION_ACCEPT = 0,
    REPORT_INTERSECTION_IGNORE = 1,
    REPORT_INTERSECTION_TERMINATE = 2,
};

typedef struct Hit_data{
    VkGeometryTypeKHR geometryType;
    uint32_t hit_kind;
    float world_min_thit;
    uint32_t geometry_index;
    uint32_t primitive_index;
    float3 intersection_point;
    float3 barycentric_coordinates;
    int32_t hitGroupIndex; // Shader ID of the closest hit for procedural geometries

    uint32_t instance_index; // Legacy storage for gl_InstanceCustomIndexEXT.
    uint32_t instance_id; // TLAS instance ordinal exposed by gl_InstanceID.
    float4x4 worldToObjectMatrix;
    float4x4 objectToWorldMatrix;
} Hit_data;

typedef struct Traversal_data {
    bool hit_geometry;
    Hit_data closest_hit;
    float3 ray_world_direction;
    float3 ray_world_origin;
    float Tmin;
    float Tmax;
    int32_t current_shader_counter; // set to shader_counter in call_intersection and -1 in call_miss and call_closest_hit
    int32_t current_shader_type;
    uint32_t current_shader_ray_tmax_fp32;
    uint32_t current_shader_ray_tmax_valid;
    uint32_t n_all_hits;
    uint32_t rayFlags;
    uint32_t cullMask;
    uint32_t sbtRecordOffset;
    uint32_t sbtRecordStride;
    uint32_t missIndex;
    uint64_t rtcore_trace_input_top_level_as;
    uint32_t rtcore_trace_input_has_top_level_as;
    uint64_t rtcore_traversable_proxy_id;
    uint64_t rtcore_root_proxy_id;
    uint32_t rtcore_node_visits;
    uint32_t rtcore_primitive_tests;
} Traversal_data;

typedef struct report_intersection_frame_entry {
    // Numeric identities are the CCS authority.  Pointers below are decoded
    // simulator caches and are checked against these stable values.
    uint64_t traversal_address;
    uint32_t instruction_pc;
    uint32_t caller_entry_pc;
    const ptx_instruction *instruction;
    function_info *caller;
    function_info *anyhit;
    Traversal_data *traversal;
    Hit_data candidate;
    uint32_t shader_counter;
    uint32_t shader_id;
    uint64_t attribute_address;
    uint32_t attribute_size;
    rtcore_khr_ccs_attribute_image attribute_image;
    size_t trace_depth;
    size_t callable_depth;
    report_intersection_decision decision;
} report_intersection_frame_entry;

typedef struct committed_procedural_attribute_entry {
    uint64_t traversal_address;
    Traversal_data *traversal;
    uint64_t address;
    uint32_t size;
    rtcore_khr_ccs_attribute_image image;
} committed_procedural_attribute_entry;

typedef struct rtcore_khr_ccs_lane_control {
    uint8_t trace_depth = 0;
    uint8_t callable_depth = 0;
    uint8_t report_depth = 0;
    uint8_t generation = 1;
    bool report_terminated = false;
} rtcore_khr_ccs_lane_control;

typedef struct Vulkan_RT_thread_data {
    std::vector<variable_decleration_entry> variable_decleration_table;
    rtcore_khr_ccs_fixed_stack<callable_data_binding_entry,
        RTCORE_KHR_CCS_LOGICAL_DEPTH> callable_data_bindings;
    rtcore_khr_ccs_fixed_stack<report_intersection_frame_entry,
        RTCORE_KHR_CCS_LOGICAL_DEPTH> report_intersection_frames;
    rtcore_khr_ccs_fixed_stack<committed_procedural_attribute_entry,
        RTCORE_KHR_CCS_LOGICAL_DEPTH>
        committed_procedural_attributes;
    // Modeled scheduler CCS lane-current-control row.
    rtcore_khr_ccs_lane_control ccs_lane_control;

    // Compatibility address cache indexed by the physical CCS trace depth.
    // Traversal bytes remain authoritative in Global384/RequestControlEntry.
    rtcore_khr_ccs_fixed_stack<Traversal_data*,
        RTCORE_KHR_CCS_LOGICAL_DEPTH> traversal_data;
    std::vector<Hit_data*> all_hit_data;

    bool push_traversal_binding(Traversal_data *traversal) {
        if (traversal == NULL ||
            ccs_lane_control.trace_depth >= RTCORE_KHR_CCS_LOGICAL_DEPTH ||
            traversal_data.size() != ccs_lane_control.trace_depth) {
            return false;
        }
        traversal_data.push_back(traversal);
        ccs_lane_control.trace_depth++;
        return true;
    }

    bool pop_traversal_binding(Traversal_data **traversal) {
        if (ccs_lane_control.trace_depth == 0 || traversal_data.empty() ||
            traversal_data.size() != ccs_lane_control.trace_depth) {
            return false;
        }
        if (traversal != NULL) *traversal = traversal_data.back();
        traversal_data.pop_back();
        ccs_lane_control.trace_depth--;
        return true;
    }

    bool push_callable_data_binding(uint64_t address, uint32_t size,
                                    function_info *callee,
                                    uint32_t sbt_index,
                                    uint32_t shader_id) {
        if (address == 0 || size == 0 || callee == NULL ||
            callable_data_bindings.size() !=
                ccs_lane_control.callable_depth ||
            callable_data_bindings.size() >=
                RTCORE_KHR_CCS_LOGICAL_DEPTH) {
            return false;
        }
        callable_data_binding_entry entry = {};
        entry.address = address;
        entry.size = size;
        entry.callee = callee;
        entry.sbt_index = sbt_index;
        entry.shader_id = shader_id;
        callable_data_bindings.push_back(entry);
        ccs_lane_control.callable_depth++;
        return true;
    }

    bool current_callable_data_binding(
            function_info *callee,
            callable_data_binding_entry *entry) const {
        if (callee == NULL || entry == NULL ||
            callable_data_bindings.empty() ||
            callable_data_bindings.size() !=
                ccs_lane_control.callable_depth ||
            callable_data_bindings.back().callee != callee) {
            return false;
        }
        *entry = callable_data_bindings.back();
        return entry->address != 0 && entry->size != 0;
    }

    bool pop_callable_data_binding(function_info *callee,
                                   callable_data_binding_entry *entry) {
        if (callee == NULL || callable_data_bindings.empty() ||
            callable_data_bindings.size() !=
                ccs_lane_control.callable_depth ||
            callable_data_bindings.back().callee != callee) {
            return false;
        }
        if (entry != NULL) {
            *entry = callable_data_bindings.back();
        }
        callable_data_bindings.pop_back();
        assert(ccs_lane_control.callable_depth > 0);
        ccs_lane_control.callable_depth--;
        return true;
    }

    bool mark_report_intersection_decision(
            function_info *callee, report_intersection_decision decision) {
        if (callee == NULL || report_intersection_frames.empty() ||
            report_intersection_frames.size() !=
                ccs_lane_control.report_depth ||
            report_intersection_frames.back().anyhit != callee) {
            return false;
        }
        report_intersection_frames.back().decision = decision;
        return true;
    }

    committed_procedural_attribute_entry *committed_procedural_attribute(
            Traversal_data *traversal) {
        const uint64_t traversal_address =
            reinterpret_cast<uint64_t>(traversal);
        for (size_t index = 0;
             index < committed_procedural_attributes.size(); ++index) {
            if (committed_procedural_attributes[index].traversal_address ==
                    traversal_address &&
                committed_procedural_attributes[index].traversal == traversal) {
                return &committed_procedural_attributes[index];
            }
        }
        return NULL;
    }

    void clear_committed_procedural_attribute(Traversal_data *traversal) {
        const uint64_t traversal_address =
            reinterpret_cast<uint64_t>(traversal);
        for (rtcore_khr_ccs_fixed_stack<
                 committed_procedural_attribute_entry,
                 RTCORE_KHR_CCS_LOGICAL_DEPTH>::iterator it =
                 committed_procedural_attributes.begin();
             it != committed_procedural_attributes.end(); ++it) {
            if (it->traversal_address == traversal_address &&
                it->traversal == traversal) {
                committed_procedural_attributes.erase(it);
                return;
            }
        }
    }


    variable_decleration_entry* get_variable_decleration_entry(nir_variable_mode type, std::string name, uint32_t size) {
        if(type == nir_var_ray_hit_attrib)
            return get_hitAttribute();
        
        for (int i = 0; i < variable_decleration_table.size(); i++) {
            if (variable_decleration_table[i].name == name) {
                assert (variable_decleration_table[i].address != NULL);
                return &(variable_decleration_table[i]);
            }
        }
        return NULL;
    }

    uint64_t add_variable_decleration_entry(nir_variable_mode type, std::string name, uint32_t size) {
        variable_decleration_entry entry;
        entry.type = type;
        entry.name = name;
        // entry.address = (uint64_t) malloc(size);
        entry.address = (uint64_t) VulkanRayTracing::gpgpusim_alloc(size);
        entry.size = size;
        variable_decleration_table.push_back(entry);

        return entry.address;
    }

    variable_decleration_entry* get_hitAttribute() {
        variable_decleration_entry* hitAttribute = NULL;
        for (int i = 0; i < variable_decleration_table.size(); i++) {
            if (variable_decleration_table[i].type == nir_var_ray_hit_attrib) {
                assert (variable_decleration_table[i].address != NULL);
                assert (hitAttribute == NULL); // There should be only 1 hitAttribute
                hitAttribute = &(variable_decleration_table[i]);
            }
        }
        return hitAttribute;
    }

    void set_hitAttribute(float3 barycentric, const ptx_instruction *pI, ptx_thread_info *thread) {
        variable_decleration_entry* hitAttribute = get_hitAttribute();
        float* address;
        if(hitAttribute == NULL) {
            address = (float*)add_variable_decleration_entry(nir_var_ray_hit_attrib, "attribs", 12);
        }
        else {
            assert (hitAttribute->type == nir_var_ray_hit_attrib);
            assert (hitAttribute->address != NULL);
            // hitAttribute->name = name;
            address = (float*)(hitAttribute->address);
        }
        // address[0] = barycentric.x;
        // address[1] = barycentric.y;
        // address[2] = barycentric.z;

        memory_space *mem = thread->get_global_memory();
        mem->write(address, sizeof(float3), &barycentric, thread, pI);
    }

    bool set_hitAttributeWords(const uint32_t *words, uint32_t word_count,
                               const ptx_instruction *pI,
                               ptx_thread_info *thread) {
        if (words == NULL || word_count == 0 || word_count > 4 ||
            pI == NULL || thread == NULL) {
            return false;
        }

        const uint32_t required_size = word_count * sizeof(uint32_t);
        variable_decleration_entry *hit_attribute = get_hitAttribute();
        if (hit_attribute == NULL) {
            add_variable_decleration_entry(
                nir_var_ray_hit_attrib, "attribs", required_size);
            hit_attribute = get_hitAttribute();
        } else if (hit_attribute->size < required_size) {
            hit_attribute->address =
                (uint64_t)VulkanRayTracing::gpgpusim_alloc(required_size);
            hit_attribute->size = required_size;
        }
        if (hit_attribute == NULL || hit_attribute->address == 0 ||
            hit_attribute->size < required_size) {
            return false;
        }

        std::array<uint32_t, 4> attribute_words = {};
        for (uint32_t word = 0; word < word_count; ++word) {
            attribute_words[word] = words[word];
        }
        std::vector<unsigned char> image(hit_attribute->size, 0);
        if (!rtcore::abi_v04::shader_input::encode_words_little_endian(
                attribute_words, word_count, image.data(), image.size())) {
            return false;
        }
        thread->get_global_memory()->write(
            hit_attribute->address, image.size(), image.data(), thread, pI);
        return true;
    }
} Vulkan_RT_thread_data;

#endif /* VULKAN_RT_THREAD_DATA_H */
