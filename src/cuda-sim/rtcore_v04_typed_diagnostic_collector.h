#ifndef RTCORE_V04_TYPED_DIAGNOSTIC_COLLECTOR_H
#define RTCORE_V04_TYPED_DIAGNOSTIC_COLLECTOR_H

#include <cstddef>
#include <cstdint>

#include "rtcore_v04_private_frontier_layout.h"

namespace rtcore {
namespace v04 {
namespace typed_diagnostic {

enum driver_kind : uint8_t {
  kDriverInvalid = 0,
  kDriverFunctionalOnly = 1,
  kDriverTiming = 2,
};

enum unit_kind : uint8_t {
  kUnitInvalid = 0,
  kUnitNode = 1,
  kUnitStack = 2,
  kUnitInstance = 3,
  kUnitPrimitive = 4,
};

struct record_v0 {
  private_frontier::owner_binding_v0 owner;
  uint32_t operation_seq;
  uint8_t driver;
  uint8_t unit;
  uint8_t operation_kind;
  uint8_t semantic_plan_kind;
  uint8_t route_kind;
  uint8_t boundary_kind;
  uint8_t reserved_zero[2];
  const void *typed_input;
  size_t typed_input_bytes;
  const void *typed_result;
  size_t typed_result_bytes;
  const void *semantic_plan;
  size_t semantic_plan_bytes;
};

bool enabled();
bool recording_required();

bool emit_record(const record_v0 &record);

const char *driver_name(driver_kind driver);
const char *unit_name(unit_kind unit);

}  // namespace typed_diagnostic
}  // namespace v04
}  // namespace rtcore

#endif
