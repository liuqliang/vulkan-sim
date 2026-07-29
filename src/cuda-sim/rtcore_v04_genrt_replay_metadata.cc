#include "rtcore_v04_genrt_replay_metadata.h"

#include <cstring>
#include <limits>
#include <set>
#include <vector>

namespace rtcore {
namespace v04 {
namespace genrt_replay {
namespace {

static const uint8_t kAsTypeTlas = 1;

static bool bytes_are_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (bytes[index] != 0) return false;
  }
  return true;
}

static int32_t read_le_i32(const uint8_t *bytes) {
  const uint32_t value =
      static_cast<uint32_t>(bytes[0]) |
      (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) |
      (static_cast<uint32_t>(bytes[3]) << 24);
  int32_t signed_value = 0;
  std::memcpy(&signed_value, &value, sizeof(signed_value));
  return signed_value;
}

static bool valid_identity(const object_identity_v0 &identity) {
  return identity.object_id != 0 && identity.generation != 0 &&
         identity.build_generation != 0 &&
         (identity.as_type == kAsTypeTlas ||
          identity.as_type == typed_blas::kAsTypeBlas) &&
         bytes_are_zero(identity.reserved_zero,
                        sizeof(identity.reserved_zero));
}

static bool payload_allowed(uint8_t as_type, uint8_t kind) {
  if (as_type == kAsTypeTlas) {
    return kind == typed_node::kInternalPayloadKind ||
           kind == typed_node::kInstancePayloadKind;
  }
  return kind == typed_node::kInternalPayloadKind ||
         kind == typed_node::kProceduralPayloadKind ||
         kind == typed_node::kQuadPayloadKind;
}

static bool payload_layout(uint8_t kind, uint8_t size_blocks,
                           uint64_t *bytes) {
  if (bytes == NULL) return false;
  switch (kind) {
    case typed_node::kInternalPayloadKind:
    case typed_node::kProceduralPayloadKind:
    case typed_node::kQuadPayloadKind:
      *bytes = 64;
      return size_blocks == 1;
    case typed_node::kInstancePayloadKind:
      *bytes = 128;
      return size_blocks == 2;
  }
  return false;
}

static bool range_contains(uint64_t image_bytes, uint64_t offset,
                           uint64_t bytes) {
  return offset <= image_bytes && bytes <= image_bytes - offset;
}

static bool add_signed_blocks(uint64_t base, int32_t blocks,
                              uint64_t *result) {
  if (result == NULL) return false;
  const int64_t byte_delta =
      static_cast<int64_t>(blocks) * int64_t{64};
  if (byte_delta >= 0) {
    const uint64_t magnitude = static_cast<uint64_t>(byte_delta);
    if (base > std::numeric_limits<uint64_t>::max() - magnitude) {
      return false;
    }
    *result = base + magnitude;
    return true;
  }
  const uint64_t magnitude =
      static_cast<uint64_t>(-(byte_delta + 1)) + uint64_t{1};
  if (base < magnitude) return false;
  *result = base - magnitude;
  return true;
}

struct build_state {
  const image_view_v0 *image;
  object_metadata_v0 *metadata;
  std::set<uint64_t> visiting;
  std::set<uint64_t> complete;
};

static status_kind insert_record(
    build_state *state, uint64_t payload_offset, uint8_t payload_kind,
    bool root, uint64_t parent_offset, uint8_t parent_child_slot) {
  if (state == NULL || state->metadata == NULL) {
    return kStatusInvalidArgument;
  }
  std::map<uint64_t, parent_record_v0>::const_iterator existing =
      state->metadata->records.find(payload_offset);
  if (existing != state->metadata->records.end()) {
    return kStatusDuplicateParent;
  }
  parent_record_v0 record = {};
  record.payload_offset = payload_offset;
  record.parent_payload_offset = root ? 0 : parent_offset;
  record.parent_child_slot = root ? 0 : parent_child_slot;
  record.payload_kind = payload_kind;
  record.root = root ? 1 : 0;
  state->metadata->records[payload_offset] = record;
  return kStatusOk;
}

static status_kind visit_internal(build_state *state,
                                  uint64_t node_offset) {
  if (state == NULL || state->image == NULL ||
      state->metadata == NULL) {
    return kStatusInvalidArgument;
  }
  if (state->complete.find(node_offset) != state->complete.end()) {
    return kStatusOk;
  }
  if (!state->visiting.insert(node_offset).second) {
    return kStatusCycle;
  }
  if (!range_contains(state->image->byte_count, node_offset, 64)) {
    return kStatusChildOutOfRange;
  }

  const uint8_t *raw = state->image->bytes + node_offset;
  if (raw[16] != typed_node::kInternalPayloadKind || raw[17] != 0) {
    return kStatusMalformedNode;
  }
  uint64_t first_child = 0;
  if (!add_signed_blocks(node_offset, read_le_i32(raw + 12),
                         &first_child)) {
    return kStatusChildOutOfRange;
  }

  bool prefix_ended = false;
  uint64_t child_offset = first_child;
  for (uint8_t child = 0; child < typed_node::kMaxChildren; ++child) {
    const uint8_t info = raw[22 + child];
    if ((info & 0xc0u) != 0) return kStatusMalformedNode;
    const uint8_t size_blocks = info & 0x03u;
    const uint8_t kind = static_cast<uint8_t>((info >> 2) & 0x0fu);
    if (size_blocks == 0) {
      prefix_ended = true;
      continue;
    }
    if (prefix_ended ||
        !payload_allowed(state->image->identity.as_type, kind)) {
      return kStatusMalformedNode;
    }
    uint64_t child_bytes = 0;
    if (!payload_layout(kind, size_blocks, &child_bytes) ||
        !range_contains(state->image->byte_count, child_offset,
                        child_bytes)) {
      return kStatusChildOutOfRange;
    }
    if (kind == typed_node::kInternalPayloadKind &&
        state->visiting.find(child_offset) !=
            state->visiting.end()) {
      return kStatusCycle;
    }
    const status_kind inserted =
        insert_record(state, child_offset, kind, false, node_offset,
                      child);
    if (inserted != kStatusOk) return inserted;
    if (kind == typed_node::kInternalPayloadKind) {
      const status_kind child_status =
          visit_internal(state, child_offset);
      if (child_status != kStatusOk) return child_status;
    }
    if (child_offset >
        std::numeric_limits<uint64_t>::max() - child_bytes) {
      return kStatusChildOutOfRange;
    }
    child_offset += child_bytes;
  }

  state->visiting.erase(node_offset);
  state->complete.insert(node_offset);
  return kStatusOk;
}

static status_kind build_object(const image_view_v0 &image,
                                object_metadata_v0 *metadata) {
  if (metadata == NULL || !valid_identity(image.identity) ||
      image.bytes == NULL || image.byte_count < 64 ||
      image.root_payload_offset < 64 ||
      (image.root_payload_offset & uint64_t{0x3f}) != 0 ||
      !range_contains(image.byte_count, image.root_payload_offset, 64)) {
    return kStatusInvalidArgument;
  }
  *metadata = object_metadata_v0();
  metadata->identity = image.identity;
  metadata->root_payload_offset = image.root_payload_offset;
  const uint8_t root_kind = image.bytes[image.root_payload_offset + 16];
  if (root_kind != typed_node::kInternalPayloadKind) {
    return kStatusMalformedNode;
  }
  build_state state = {};
  state.image = &image;
  state.metadata = metadata;
  status_kind status =
      insert_record(&state, image.root_payload_offset, root_kind,
                    true, 0, 0);
  if (status != kStatusOk) return status;
  status = visit_internal(&state, image.root_payload_offset);
  if (status != kStatusOk) {
    *metadata = object_metadata_v0();
  }
  return status;
}

}  // namespace

static_assert(sizeof(object_identity_v0) == 24,
              "GEN_RT replay identity must remain 24 bytes");
static_assert(sizeof(parent_record_v0) == 24,
              "GEN_RT replay parent record must remain 24 bytes");

status_kind registry_v0::publish(const image_view_v0 &image) {
  if (!valid_identity(image.identity)) return kStatusInvalidArgument;
  std::map<uint64_t, object_metadata_v0>::iterator existing =
      objects_.find(image.identity.object_id);
  if (existing != objects_.end()) {
    if (existing->second.identity.generation !=
        image.identity.generation) {
      return kStatusGenerationMismatch;
    }
    if (existing->second.identity.build_generation ==
        image.identity.build_generation) {
      return kStatusObjectAlreadyPublished;
    }
    if (image.identity.build_generation <
        existing->second.identity.build_generation) {
      return kStatusBuildGenerationMismatch;
    }
  }
  object_metadata_v0 metadata;
  const status_kind status = build_object(image, &metadata);
  if (status != kStatusOk) return status;
  objects_[image.identity.object_id] = metadata;
  return kStatusOk;
}

status_kind registry_v0::resolve(
    const object_identity_v0 &identity, uint64_t payload_offset,
    parent_record_v0 *record) const {
  if (record == NULL || !valid_identity(identity)) {
    return kStatusInvalidArgument;
  }
  *record = parent_record_v0();
  std::map<uint64_t, object_metadata_v0>::const_iterator object =
      objects_.find(identity.object_id);
  if (object == objects_.end()) return kStatusObjectNotFound;
  if (object->second.identity.generation != identity.generation) {
    return kStatusGenerationMismatch;
  }
  if (object->second.identity.build_generation !=
      identity.build_generation) {
    return kStatusBuildGenerationMismatch;
  }
  if (object->second.identity.as_type != identity.as_type) {
    return kStatusUnsupportedAsType;
  }
  std::map<uint64_t, parent_record_v0>::const_iterator found =
      object->second.records.find(payload_offset);
  if (found == object->second.records.end()) {
    return kStatusRecordNotFound;
  }
  *record = found->second;
  return kStatusOk;
}

status_kind registry_v0::release(uint64_t object_id,
                                 uint32_t generation) {
  std::map<uint64_t, object_metadata_v0>::iterator object =
      objects_.find(object_id);
  if (object == objects_.end()) return kStatusObjectNotFound;
  if (generation == 0 ||
      object->second.identity.generation != generation) {
    return kStatusGenerationMismatch;
  }
  objects_.erase(object);
  return kStatusOk;
}

size_t registry_v0::object_count() const {
  return objects_.size();
}

size_t registry_v0::record_count(uint64_t object_id) const {
  std::map<uint64_t, object_metadata_v0>::const_iterator object =
      objects_.find(object_id);
  return object == objects_.end() ? 0 : object->second.records.size();
}

bool to_parent_edge(const parent_record_v0 &record,
                    short_stack::parent_edge_v0 *edge) {
  if (edge == NULL || record.root > 1 ||
      !bytes_are_zero(record.reserved_zero,
                      sizeof(record.reserved_zero))) {
    return false;
  }
  *edge = short_stack::parent_edge_v0();
  edge->root = record.root;
  if (record.root != 0) return true;
  if (record.parent_payload_offset == 0 ||
      (record.parent_payload_offset & uint64_t{0x3f}) != 0 ||
      record.parent_child_slot >= typed_node::kMaxChildren) {
    return false;
  }
  edge->parent_payload_offset = record.parent_payload_offset;
  edge->parent_child_slot = record.parent_child_slot;
  return true;
}

const char *status_name(status_kind status) {
  switch (status) {
    case kStatusOk:
      return "ok";
    case kStatusInvalidArgument:
      return "invalid_argument";
    case kStatusUnsupportedAsType:
      return "unsupported_as_type";
    case kStatusRootOutOfRange:
      return "root_out_of_range";
    case kStatusMalformedNode:
      return "malformed_node";
    case kStatusChildOutOfRange:
      return "child_out_of_range";
    case kStatusDuplicateParent:
      return "duplicate_parent";
    case kStatusCycle:
      return "cycle";
    case kStatusObjectAlreadyPublished:
      return "object_already_published";
    case kStatusObjectNotFound:
      return "object_not_found";
    case kStatusGenerationMismatch:
      return "generation_mismatch";
    case kStatusBuildGenerationMismatch:
      return "build_generation_mismatch";
    case kStatusRecordNotFound:
      return "record_not_found";
  }
  return "unknown";
}

}  // namespace genrt_replay
}  // namespace v04
}  // namespace rtcore
