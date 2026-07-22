#ifndef RTCORE_V04_INSTANCE_BLAS_REFERENCE_REGISTRY_H
#define RTCORE_V04_INSTANCE_BLAS_REFERENCE_REGISTRY_H

#include <stdint.h>

#include <map>

struct rtcore_v04_instance_blas_reference_snapshot {
  bool valid;
  uint8_t reserved_zero[3];
  uint32_t tlas_generation;
  uint32_t tlas_build_generation;
  uint32_t blas_generation;
  uint64_t tlas_object_id;
  uint64_t instance_host_address;
  uint64_t instance_metadata_reference;
  uint64_t blas_object_id;
  uint64_t blas_host_root_address;

  rtcore_v04_instance_blas_reference_snapshot()
      : valid(false), reserved_zero(), tlas_generation(0),
        tlas_build_generation(0), blas_generation(0), tlas_object_id(0),
        instance_host_address(0), instance_metadata_reference(0),
        blas_object_id(0), blas_host_root_address(0) {}
};

class rtcore_v04_instance_blas_reference_registry {
 public:
  bool begin_build(uint64_t tlas_object_id, uint32_t tlas_generation,
                   uint64_t tlas_host_root, uint64_t tlas_device_base,
                   uint64_t tlas_size_bytes, uint32_t *build_generation,
                   const char **failure_reason) {
    if (build_generation != nullptr) *build_generation = 0;
    if (tlas_object_id == 0 || tlas_generation == 0 ||
        tlas_host_root == 0 || tlas_device_base == 0 ||
        tlas_size_bytes < 128 ||
        tlas_host_root > UINT64_MAX - tlas_size_bytes ||
        tlas_device_base > UINT64_MAX - tlas_size_bytes) {
      return fail("invalid_tlas_build_binding", failure_reason);
    }

    build_record &build = builds_[tlas_object_id];
    if (build.active) {
      return fail("tlas_build_already_active", failure_reason);
    }
    if (build.generation != 0) {
      if (build.generation != tlas_generation) {
        return fail("tlas_generation_mismatch", failure_reason);
      }
      if (build.host_root != tlas_host_root ||
          build.device_base != tlas_device_base ||
          build.size_bytes != tlas_size_bytes) {
        return fail("tlas_build_binding_mismatch", failure_reason);
      }
    }
    if (build.build_generation == UINT32_MAX) {
      return fail("tlas_build_generation_exhausted", failure_reason);
    }

    build.generation = tlas_generation;
    build.build_generation =
        build.build_generation == 0 ? 1 : build.build_generation + 1;
    build.host_root = tlas_host_root;
    build.device_base = tlas_device_base;
    build.size_bytes = tlas_size_bytes;
    build.active = true;
    build.sealed = false;
    build.references.clear();
    if (build_generation != nullptr) {
      *build_generation = build.build_generation;
    }
    return succeed(failure_reason);
  }

  bool publish(uint64_t tlas_object_id, uint32_t tlas_generation,
               uint64_t instance_host_address,
               uint64_t instance_metadata_reference,
               uint64_t blas_object_id, uint32_t blas_generation,
               uint64_t blas_host_root_address,
               rtcore_v04_instance_blas_reference_snapshot *published,
               const char **failure_reason) {
    reset_snapshot(published);
    std::map<uint64_t, build_record>::iterator build =
        builds_.find(tlas_object_id);
    if (published == nullptr || build == builds_.end() ||
        !build->second.active || build->second.sealed ||
        build->second.generation != tlas_generation) {
      return fail("tlas_build_not_active", failure_reason);
    }
    if (instance_host_address < build->second.host_root ||
        instance_metadata_reference < build->second.device_base ||
        blas_object_id == 0 || blas_generation == 0 ||
        blas_host_root_address == 0) {
      return fail("invalid_instance_blas_reference", failure_reason);
    }
    const uint64_t host_offset =
        instance_host_address - build->second.host_root;
    const uint64_t device_offset =
        instance_metadata_reference - build->second.device_base;
    if (host_offset != device_offset ||
        (instance_metadata_reference & uint64_t{0x3f}) != 0 ||
        host_offset > build->second.size_bytes ||
        uint64_t{128} > build->second.size_bytes - host_offset) {
      return fail("instance_metadata_out_of_range", failure_reason);
    }
    if (build->second.references.find(instance_metadata_reference) !=
        build->second.references.end()) {
      return fail("duplicate_instance_metadata_reference", failure_reason);
    }

    rtcore_v04_instance_blas_reference_snapshot reference;
    reference.valid = true;
    reference.tlas_object_id = tlas_object_id;
    reference.tlas_generation = tlas_generation;
    reference.tlas_build_generation = build->second.build_generation;
    reference.instance_host_address = instance_host_address;
    reference.instance_metadata_reference = instance_metadata_reference;
    reference.blas_object_id = blas_object_id;
    reference.blas_generation = blas_generation;
    reference.blas_host_root_address = blas_host_root_address;
    build->second.references[instance_metadata_reference] = reference;
    *published = reference;
    return succeed(failure_reason);
  }

  bool end_build(uint64_t tlas_object_id, uint32_t tlas_generation,
                 uint32_t *build_generation, uint64_t *reference_count,
                 const char **failure_reason) {
    if (build_generation != nullptr) *build_generation = 0;
    if (reference_count != nullptr) *reference_count = 0;
    std::map<uint64_t, build_record>::iterator build =
        builds_.find(tlas_object_id);
    if (build == builds_.end() || !build->second.active ||
        build->second.sealed ||
        build->second.generation != tlas_generation) {
      return fail("tlas_build_not_active", failure_reason);
    }
    build->second.active = false;
    build->second.sealed = true;
    if (build_generation != nullptr) {
      *build_generation = build->second.build_generation;
    }
    if (reference_count != nullptr) {
      *reference_count = build->second.references.size();
    }
    return succeed(failure_reason);
  }

  bool resolve(uint64_t tlas_object_id, uint32_t tlas_generation,
               uint64_t instance_metadata_reference,
               rtcore_v04_instance_blas_reference_snapshot *resolved,
               const char **failure_reason) const {
    reset_snapshot(resolved);
    std::map<uint64_t, build_record>::const_iterator build =
        builds_.find(tlas_object_id);
    if (resolved == nullptr || build == builds_.end() ||
        build->second.active || !build->second.sealed ||
        build->second.generation != tlas_generation) {
      return fail("tlas_reference_table_not_ready", failure_reason);
    }
    std::map<uint64_t,
             rtcore_v04_instance_blas_reference_snapshot>::const_iterator
        reference =
            build->second.references.find(instance_metadata_reference);
    if (reference == build->second.references.end()) {
      return fail("instance_reference_not_found", failure_reason);
    }
    *resolved = reference->second;
    return validate(*resolved, failure_reason);
  }

  bool validate(
      const rtcore_v04_instance_blas_reference_snapshot &reference,
      const char **failure_reason) const {
    if (!reference.valid || !reserved_fields_zero(reference) ||
        reference.tlas_object_id == 0 ||
        reference.tlas_generation == 0 ||
        reference.tlas_build_generation == 0 ||
        reference.instance_host_address == 0 ||
        reference.instance_metadata_reference == 0 ||
        reference.blas_object_id == 0 || reference.blas_generation == 0 ||
        reference.blas_host_root_address == 0) {
      return fail("invalid_reference_snapshot", failure_reason);
    }
    std::map<uint64_t, build_record>::const_iterator build =
        builds_.find(reference.tlas_object_id);
    if (build == builds_.end() || build->second.active ||
        !build->second.sealed) {
      return fail("tlas_reference_table_not_ready", failure_reason);
    }
    if (build->second.generation != reference.tlas_generation ||
        build->second.build_generation !=
            reference.tlas_build_generation) {
      return fail("stale_tlas_reference_build", failure_reason);
    }
    std::map<uint64_t,
             rtcore_v04_instance_blas_reference_snapshot>::const_iterator
        current = build->second.references.find(
            reference.instance_metadata_reference);
    if (current == build->second.references.end() ||
        !same_reference(current->second, reference)) {
      return fail("instance_reference_mismatch", failure_reason);
    }
    return succeed(failure_reason);
  }

  bool release(uint64_t tlas_object_id, uint32_t tlas_generation,
               const char **failure_reason) {
    std::map<uint64_t, build_record>::iterator build =
        builds_.find(tlas_object_id);
    if (build == builds_.end()) {
      return succeed(failure_reason);
    }
    if (build->second.generation != tlas_generation) {
      return fail("tlas_reference_release_mismatch", failure_reason);
    }
    if (build->second.active) {
      return fail("tlas_reference_release_during_build", failure_reason);
    }
    builds_.erase(build);
    return succeed(failure_reason);
  }

 private:
  struct build_record {
    build_record()
        : generation(0), build_generation(0), host_root(0), device_base(0),
          size_bytes(0), active(false), sealed(false), references() {}

    uint32_t generation;
    uint32_t build_generation;
    uint64_t host_root;
    uint64_t device_base;
    uint64_t size_bytes;
    bool active;
    bool sealed;
    std::map<uint64_t, rtcore_v04_instance_blas_reference_snapshot>
        references;
  };

  static bool same_reference(
      const rtcore_v04_instance_blas_reference_snapshot &lhs,
      const rtcore_v04_instance_blas_reference_snapshot &rhs) {
    return lhs.valid == rhs.valid &&
           lhs.reserved_zero[0] == rhs.reserved_zero[0] &&
           lhs.reserved_zero[1] == rhs.reserved_zero[1] &&
           lhs.reserved_zero[2] == rhs.reserved_zero[2] &&
           lhs.tlas_object_id == rhs.tlas_object_id &&
           lhs.tlas_generation == rhs.tlas_generation &&
           lhs.tlas_build_generation == rhs.tlas_build_generation &&
           lhs.instance_host_address == rhs.instance_host_address &&
           lhs.instance_metadata_reference ==
               rhs.instance_metadata_reference &&
           lhs.blas_object_id == rhs.blas_object_id &&
           lhs.blas_generation == rhs.blas_generation &&
           lhs.blas_host_root_address == rhs.blas_host_root_address;
  }

  static bool reserved_fields_zero(
      const rtcore_v04_instance_blas_reference_snapshot &reference) {
    return reference.reserved_zero[0] == 0 &&
           reference.reserved_zero[1] == 0 &&
           reference.reserved_zero[2] == 0;
  }

  static void reset_snapshot(
      rtcore_v04_instance_blas_reference_snapshot *snapshot) {
    if (snapshot != nullptr) {
      *snapshot = rtcore_v04_instance_blas_reference_snapshot();
    }
  }

  static bool fail(const char *reason, const char **failure_reason) {
    if (failure_reason != nullptr) *failure_reason = reason;
    return false;
  }

  static bool succeed(const char **failure_reason) {
    if (failure_reason != nullptr) *failure_reason = "none";
    return true;
  }

  std::map<uint64_t, build_record> builds_;
};

#endif
