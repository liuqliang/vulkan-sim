#ifndef RTCORE_V04_GENRT_REPLAY_METADATA_H
#define RTCORE_V04_GENRT_REPLAY_METADATA_H

#include <cstdint>
#include <map>

#include "rtcore_v04_short_stack_replay.h"

namespace rtcore {
namespace v04 {
namespace genrt_replay {

enum status_kind : uint8_t {
  kStatusOk = 0,
  kStatusInvalidArgument,
  kStatusUnsupportedAsType,
  kStatusRootOutOfRange,
  kStatusMalformedNode,
  kStatusChildOutOfRange,
  kStatusDuplicateParent,
  kStatusCycle,
  kStatusObjectAlreadyPublished,
  kStatusObjectNotFound,
  kStatusGenerationMismatch,
  kStatusBuildGenerationMismatch,
  kStatusRecordNotFound,
};

struct object_identity_v0 {
  uint64_t object_id;
  uint32_t generation;
  uint32_t build_generation;
  uint8_t as_type;
  uint8_t reserved_zero[7];
};

struct parent_record_v0 {
  uint64_t payload_offset;
  uint64_t parent_payload_offset;
  uint8_t parent_child_slot;
  uint8_t payload_kind;
  uint8_t root;
  uint8_t reserved_zero[5];
};

struct image_view_v0 {
  object_identity_v0 identity;
  const uint8_t *bytes;
  uint64_t byte_count;
  uint64_t root_payload_offset;
};

struct object_metadata_v0 {
  object_identity_v0 identity;
  uint64_t root_payload_offset;
  std::map<uint64_t, parent_record_v0> records;
};

class registry_v0 {
 public:
  status_kind publish(const image_view_v0 &image);
  status_kind resolve(const object_identity_v0 &identity,
                      uint64_t payload_offset,
                      parent_record_v0 *record) const;
  status_kind release(uint64_t object_id, uint32_t generation);
  size_t object_count() const;
  size_t record_count(uint64_t object_id) const;

 private:
  std::map<uint64_t, object_metadata_v0> objects_;
};

bool to_parent_edge(const parent_record_v0 &record,
                    short_stack::parent_edge_v0 *edge);

const char *status_name(status_kind status);

}  // namespace genrt_replay
}  // namespace v04
}  // namespace rtcore

#endif
