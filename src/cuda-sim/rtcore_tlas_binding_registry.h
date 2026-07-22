#ifndef RTCORE_TLAS_BINDING_REGISTRY_H
#define RTCORE_TLAS_BINDING_REGISTRY_H

#include <stdint.h>

#include <map>

template <typename Snapshot>
class rtcore_tlas_binding_registry {
 public:
  rtcore_tlas_binding_registry()
      : next_object_id_(1), next_generation_(1) {}

  bool register_binding(uint64_t driver_object_key,
                        uint64_t host_root_address,
                        uint64_t device_base_address, uint64_t size_bytes,
                        Snapshot *published, const char **failure_reason) {
    reset_snapshot(published);
    if (published == nullptr || driver_object_key == 0 ||
        host_root_address == 0 || device_base_address == 0 ||
        size_bytes == 0 || host_root_address > UINT64_MAX - size_bytes ||
        device_base_address > UINT64_MAX - size_bytes) {
      return fail("invalid_registration", failure_reason);
    }
    if (object_id_by_driver_object_.find(driver_object_key) !=
        object_id_by_driver_object_.end()) {
      return fail("duplicate_driver_object_registration", failure_reason);
    }
    if (next_object_id_ == 0 || next_generation_ == 0 ||
        next_generation_ > UINT32_MAX) {
      return fail("identity_exhausted", failure_reason);
    }

    record new_record;
    new_record.driver_object_key = driver_object_key;
    new_record.snapshot.valid = true;
    new_record.snapshot.live = true;
    new_record.snapshot.object_id = next_object_id_++;
    new_record.snapshot.generation =
        static_cast<uint32_t>(next_generation_++);
    new_record.snapshot.host_root_address = host_root_address;
    new_record.snapshot.device_base_address = device_base_address;
    new_record.snapshot.size_bytes = size_bytes;
    bindings_by_object_id_[new_record.snapshot.object_id] = new_record;
    object_id_by_driver_object_[driver_object_key] =
        new_record.snapshot.object_id;
    *published = new_record.snapshot;
    return succeed(failure_reason);
  }

  bool publish_root_descriptor(uint64_t driver_object_key,
                               uint32_t bvh_profile_id,
                               uint32_t payload_format_id,
                               uint64_t root_payload_offset,
                               uint8_t root_payload_kind,
                               Snapshot *published,
                               const char **failure_reason) {
    reset_snapshot(published);
    typename std::map<uint64_t, uint64_t>::iterator object =
        object_id_by_driver_object_.find(driver_object_key);
    typename std::map<uint64_t, record>::iterator binding =
        object != object_id_by_driver_object_.end()
            ? bindings_by_object_id_.find(object->second)
            : bindings_by_object_id_.end();
    if (published == nullptr || object == object_id_by_driver_object_.end() ||
        binding == bindings_by_object_id_.end() ||
        !binding->second.snapshot.valid || !binding->second.snapshot.live ||
        binding->second.driver_object_key != driver_object_key) {
      return fail("root_publish_binding_mismatch", failure_reason);
    }
    if (bvh_profile_id == 0 || payload_format_id == 0 ||
        root_payload_offset < 64 ||
        (root_payload_offset & uint64_t{0x3f}) != 0 ||
        root_payload_offset > binding->second.snapshot.size_bytes ||
        uint64_t{64} >
            binding->second.snapshot.size_bytes - root_payload_offset) {
      return fail("root_descriptor_out_of_range", failure_reason);
    }
    if (binding->second.snapshot.root_descriptor_valid &&
        binding->second.snapshot.root_build_generation == UINT32_MAX) {
      return fail("root_build_generation_exhausted", failure_reason);
    }

    Snapshot &snapshot = binding->second.snapshot;
    snapshot.root_descriptor_valid = true;
    snapshot.root_payload_kind = root_payload_kind;
    snapshot.root_reserved_zero = 0;
    snapshot.root_build_generation =
        snapshot.root_build_generation == 0
            ? 1
            : snapshot.root_build_generation + 1;
    snapshot.root_bvh_profile_id = bvh_profile_id;
    snapshot.root_payload_format_id = payload_format_id;
    snapshot.root_payload_offset = root_payload_offset;
    *published = snapshot;
    return succeed(failure_reason);
  }

  bool release_binding(uint64_t driver_object_key,
                       uint64_t host_root_address,
                       uint64_t device_base_address, Snapshot *released,
                       const char **failure_reason) {
    reset_snapshot(released);
    typename std::map<uint64_t, uint64_t>::iterator object =
        object_id_by_driver_object_.find(driver_object_key);
    typename std::map<uint64_t, record>::iterator binding =
        object != object_id_by_driver_object_.end()
            ? bindings_by_object_id_.find(object->second)
            : bindings_by_object_id_.end();
    if (released == nullptr || object == object_id_by_driver_object_.end() ||
        binding == bindings_by_object_id_.end() ||
        !binding->second.snapshot.valid || !binding->second.snapshot.live ||
        binding->second.driver_object_key != driver_object_key ||
        binding->second.snapshot.host_root_address != host_root_address ||
        binding->second.snapshot.device_base_address != device_base_address) {
      return fail("release_mismatch", failure_reason);
    }

    binding->second.snapshot.live = false;
    clear_root_descriptor(&binding->second.snapshot);
    *released = binding->second.snapshot;
    object_id_by_driver_object_.erase(object);
    return succeed(failure_reason);
  }

  bool capture(uint64_t host_root_address, Snapshot *snapshot,
               const char **failure_reason) const {
    reset_snapshot(snapshot);
    if (snapshot == nullptr) {
      return fail("snapshot_null", failure_reason);
    }

    const record *selected = nullptr;
    for (typename std::map<uint64_t, record>::const_iterator binding =
             bindings_by_object_id_.begin();
         binding != bindings_by_object_id_.end(); ++binding) {
      if (!binding->second.snapshot.valid ||
          !binding->second.snapshot.live ||
          binding->second.snapshot.host_root_address != host_root_address) {
        continue;
      }
      if (selected != nullptr) {
        return fail("ambiguous_live_alias", failure_reason);
      }
      selected = &binding->second;
    }
    if (selected == nullptr) {
      return fail("binding_not_found", failure_reason);
    }

    *snapshot = selected->snapshot;
    return succeed(failure_reason);
  }

  bool capture_by_driver_object(uint64_t driver_object_key,
                                Snapshot *snapshot,
                                const char **failure_reason) const {
    reset_snapshot(snapshot);
    typename std::map<uint64_t, uint64_t>::const_iterator object =
        object_id_by_driver_object_.find(driver_object_key);
    typename std::map<uint64_t, record>::const_iterator binding =
        object != object_id_by_driver_object_.end()
            ? bindings_by_object_id_.find(object->second)
            : bindings_by_object_id_.end();
    if (snapshot == nullptr || object == object_id_by_driver_object_.end() ||
        binding == bindings_by_object_id_.end() ||
        !binding->second.snapshot.valid || !binding->second.snapshot.live ||
        binding->second.driver_object_key != driver_object_key) {
      return fail("driver_object_binding_not_found", failure_reason);
    }
    *snapshot = binding->second.snapshot;
    return succeed(failure_reason);
  }

  bool capture_by_object_id(uint64_t object_id, Snapshot *snapshot,
                            const char **failure_reason) const {
    reset_snapshot(snapshot);
    typename std::map<uint64_t, record>::const_iterator binding =
        bindings_by_object_id_.find(object_id);
    if (snapshot == nullptr || object_id == 0 ||
        binding == bindings_by_object_id_.end() ||
        !binding->second.snapshot.valid || !binding->second.snapshot.live) {
      return fail("object_binding_not_found", failure_reason);
    }
    *snapshot = binding->second.snapshot;
    return succeed(failure_reason);
  }

  bool validate(const Snapshot &snapshot, uint64_t instance_metadata_reference,
                uint64_t record_size, const char **failure_reason) const {
    if (!snapshot.valid || !snapshot.live || snapshot.object_id == 0 ||
        snapshot.generation == 0 || snapshot.host_root_address == 0 ||
        snapshot.device_base_address == 0 || snapshot.size_bytes == 0) {
      return fail("captured_binding_invalid", failure_reason);
    }

    typename std::map<uint64_t, record>::const_iterator current =
        bindings_by_object_id_.find(snapshot.object_id);
    if (current == bindings_by_object_id_.end()) {
      return fail("binding_not_found", failure_reason);
    }
    if (!current->second.snapshot.valid || !current->second.snapshot.live) {
      return fail("binding_not_live", failure_reason);
    }
    if (current->second.snapshot.object_id != snapshot.object_id ||
        current->second.snapshot.generation != snapshot.generation) {
      return fail("binding_generation_mismatch", failure_reason);
    }
    if (current->second.snapshot.host_root_address !=
            snapshot.host_root_address ||
        current->second.snapshot.device_base_address !=
            snapshot.device_base_address ||
        current->second.snapshot.size_bytes != snapshot.size_bytes) {
      return fail("binding_range_mismatch", failure_reason);
    }
    if (current->second.snapshot.root_descriptor_valid !=
            snapshot.root_descriptor_valid ||
        current->second.snapshot.root_payload_kind !=
            snapshot.root_payload_kind ||
        current->second.snapshot.root_reserved_zero !=
            snapshot.root_reserved_zero ||
        current->second.snapshot.root_build_generation !=
            snapshot.root_build_generation ||
        current->second.snapshot.root_bvh_profile_id !=
            snapshot.root_bvh_profile_id ||
        current->second.snapshot.root_payload_format_id !=
            snapshot.root_payload_format_id ||
        current->second.snapshot.root_payload_offset !=
            snapshot.root_payload_offset) {
      return fail("root_descriptor_mismatch", failure_reason);
    }

    if (record_size != 0) {
      if (instance_metadata_reference == 0 ||
          (instance_metadata_reference & uint64_t{0x3f}) != 0) {
        return fail("instance_record_alignment", failure_reason);
      }
      if (instance_metadata_reference < snapshot.device_base_address) {
        return fail("instance_record_before_range", failure_reason);
      }
      const uint64_t offset =
          instance_metadata_reference - snapshot.device_base_address;
      if (offset > snapshot.size_bytes ||
          record_size > snapshot.size_bytes - offset) {
        return fail("instance_record_out_of_range", failure_reason);
      }
    }
    return succeed(failure_reason);
  }

 private:
  struct record {
    record() : driver_object_key(0), snapshot() {}

    uint64_t driver_object_key;
    Snapshot snapshot;
  };

  static void reset_snapshot(Snapshot *snapshot) {
    if (snapshot != nullptr) {
      *snapshot = Snapshot();
    }
  }

  static void clear_root_descriptor(Snapshot *snapshot) {
    snapshot->root_descriptor_valid = false;
    snapshot->root_payload_kind = 0;
    snapshot->root_reserved_zero = 0;
    snapshot->root_build_generation = 0;
    snapshot->root_bvh_profile_id = 0;
    snapshot->root_payload_format_id = 0;
    snapshot->root_payload_offset = 0;
  }

  static bool fail(const char *reason, const char **failure_reason) {
    if (failure_reason != nullptr) {
      *failure_reason = reason;
    }
    return false;
  }

  static bool succeed(const char **failure_reason) {
    if (failure_reason != nullptr) {
      *failure_reason = "none";
    }
    return true;
  }

  std::map<uint64_t, record> bindings_by_object_id_;
  std::map<uint64_t, uint64_t> object_id_by_driver_object_;
  uint64_t next_object_id_;
  uint64_t next_generation_;
};

#endif
