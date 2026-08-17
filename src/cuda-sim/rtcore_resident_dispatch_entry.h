#ifndef RTCORE_RESIDENT_DISPATCH_ENTRY_H
#define RTCORE_RESIDENT_DISPATCH_ENTRY_H

#include <stdint.h>

class function_info;

enum rtcore_resident_dispatch_stage_v0 {
  RTCORE_RESIDENT_DISPATCH_STAGE_INVALID = 0,
  RTCORE_RESIDENT_DISPATCH_STAGE_RAYGEN = 1,
  RTCORE_RESIDENT_DISPATCH_STAGE_MISS = 2,
  RTCORE_RESIDENT_DISPATCH_STAGE_CLOSEST_HIT = 3,
  RTCORE_RESIDENT_DISPATCH_STAGE_ANY_HIT = 4,
  RTCORE_RESIDENT_DISPATCH_STAGE_INTERSECTION = 5,
  RTCORE_RESIDENT_DISPATCH_STAGE_CALLABLE = 6,
};

struct rtcore_resident_dispatch_entry_v0 {
  uint32_t valid;
  uint32_t entry_id;
  uint32_t stage;
  uint32_t registration_index;
  function_info *function;
};

extern "C" int rtcore_resolve_resident_dispatch_entry_v0(
    unsigned entry_id, unsigned expected_stage,
    rtcore_resident_dispatch_entry_v0 *entry_out);

extern "C" unsigned rtcore_resident_dispatch_stage_for_reason_v0(
    unsigned reason);

const char *rtcore_resident_dispatch_stage_name_v0(unsigned stage);

#endif
