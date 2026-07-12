#ifndef RTCORE_PROCEDURAL_HIT_ORDERING_H
#define RTCORE_PROCEDURAL_HIT_ORDERING_H

#include <cmath>

namespace rtcore {

enum procedural_report_ordering {
  RTCORE_PROCEDURAL_REPORT_INVALID = 0,
  RTCORE_PROCEDURAL_REPORT_KEEP_EXISTING,
  RTCORE_PROCEDURAL_REPORT_COMMIT,
};

inline procedural_report_ordering classify_procedural_report(
    float reported_t, float tmin, float tmax, bool has_existing_hit,
    float existing_closest_t) {
  if (!std::isfinite(reported_t) || reported_t < tmin || reported_t > tmax) {
    return RTCORE_PROCEDURAL_REPORT_INVALID;
  }
  if (has_existing_hit && !(reported_t < existing_closest_t)) {
    return RTCORE_PROCEDURAL_REPORT_KEEP_EXISTING;
  }
  return RTCORE_PROCEDURAL_REPORT_COMMIT;
}

}  // namespace rtcore

#endif  // RTCORE_PROCEDURAL_HIT_ORDERING_H
