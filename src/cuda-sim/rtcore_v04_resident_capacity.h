#ifndef RTCORE_V04_RESIDENT_CAPACITY_H
#define RTCORE_V04_RESIDENT_CAPACITY_H

#include <cstdint>

namespace rtcore {
namespace v04 {
namespace resident_capacity {

template <typename Iterator>
uint32_t occupancy_for_owner(Iterator begin, Iterator end,
                             uint32_t owner_hw_sid) {
  uint32_t occupancy = 0;
  for (Iterator it = begin; it != end; ++it) {
    if (it->second.valid && it->second.owner_hw_sid == owner_hw_sid) {
      ++occupancy;
    }
  }
  return occupancy;
}

template <typename Iterator>
bool available_for_owner(Iterator begin, Iterator end,
                         uint32_t owner_hw_sid, uint32_t capacity) {
  return occupancy_for_owner(begin, end, owner_hw_sid) < capacity;
}

}  // namespace resident_capacity
}  // namespace v04
}  // namespace rtcore

#endif
