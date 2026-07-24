#ifndef RTCORE_V04_CANONICAL_RAY_H
#define RTCORE_V04_CANONICAL_RAY_H

#include <cmath>

namespace rtcore {
namespace v04 {
namespace canonical_ray {

inline float inverse_direction(float direction) {
  const float minimum_magnitude = std::ldexp(1.0f, -80);
  const float denominator =
      std::fabs(direction) > minimum_magnitude
          ? direction
          : std::copysign(minimum_magnitude, direction);
  return 1.0f / denominator;
}

inline bool inverse_direction_matches(float direction, float inverse) {
  return std::isfinite(direction) && std::isfinite(inverse) &&
         inverse == inverse_direction(direction);
}

}  // namespace canonical_ray
}  // namespace v04
}  // namespace rtcore

#endif
