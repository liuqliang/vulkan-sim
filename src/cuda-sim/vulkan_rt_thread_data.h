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

#include "compiler/nir/nir.h"

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
    const ptx_instruction *instruction;
    function_info *caller;
    function_info *anyhit;
    Traversal_data *traversal;
    Hit_data candidate;
    uint32_t shader_counter;
    uint32_t shader_id;
    uint64_t attribute_address;
    uint32_t attribute_size;
    std::vector<unsigned char> attribute_image;
    size_t trace_depth;
    size_t callable_depth;
    report_intersection_decision decision;
} report_intersection_frame_entry;

typedef struct committed_procedural_attribute_entry {
    Traversal_data *traversal;
    uint64_t address;
    uint32_t size;
    std::vector<unsigned char> image;
} committed_procedural_attribute_entry;


typedef struct Vulkan_RT_thread_data {
    std::vector<variable_decleration_entry> variable_decleration_table;
    std::vector<callable_data_binding_entry> callable_data_bindings;
    std::vector<report_intersection_frame_entry> report_intersection_frames;
    std::vector<committed_procedural_attribute_entry>
        committed_procedural_attributes;
    bool last_report_intersection_terminated = false;

    std::vector<Traversal_data*> traversal_data;
    std::vector<Hit_data*> all_hit_data;

    bool push_callable_data_binding(uint64_t address, uint32_t size,
                                    function_info *callee,
                                    uint32_t sbt_index,
                                    uint32_t shader_id) {
        if (address == 0 || size == 0 || callee == NULL ||
            callable_data_bindings.size() >= 32) {
            return false;
        }
        callable_data_binding_entry entry = {};
        entry.address = address;
        entry.size = size;
        entry.callee = callee;
        entry.sbt_index = sbt_index;
        entry.shader_id = shader_id;
        callable_data_bindings.push_back(entry);
        return true;
    }

    bool current_callable_data_binding(
            function_info *callee,
            callable_data_binding_entry *entry) const {
        if (callee == NULL || entry == NULL ||
            callable_data_bindings.empty() ||
            callable_data_bindings.back().callee != callee) {
            return false;
        }
        *entry = callable_data_bindings.back();
        return entry->address != 0 && entry->size != 0;
    }

    bool pop_callable_data_binding(function_info *callee,
                                   callable_data_binding_entry *entry) {
        if (callee == NULL || callable_data_bindings.empty() ||
            callable_data_bindings.back().callee != callee) {
            return false;
        }
        if (entry != NULL) {
            *entry = callable_data_bindings.back();
        }
        callable_data_bindings.pop_back();
        return true;
    }

    bool mark_report_intersection_decision(
            function_info *callee, report_intersection_decision decision) {
        if (callee == NULL || report_intersection_frames.empty() ||
            report_intersection_frames.back().anyhit != callee) {
            return false;
        }
        report_intersection_frames.back().decision = decision;
        return true;
    }

    committed_procedural_attribute_entry *committed_procedural_attribute(
            Traversal_data *traversal) {
        for (size_t index = 0;
             index < committed_procedural_attributes.size(); ++index) {
            if (committed_procedural_attributes[index].traversal ==
                traversal) {
                return &committed_procedural_attributes[index];
            }
        }
        return NULL;
    }

    void clear_committed_procedural_attribute(Traversal_data *traversal) {
        for (std::vector<committed_procedural_attribute_entry>::iterator it =
                 committed_procedural_attributes.begin();
             it != committed_procedural_attributes.end(); ++it) {
            if (it->traversal == traversal) {
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
