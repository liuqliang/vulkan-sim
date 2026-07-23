#ifndef RTCORE_V04_PRIVATE_SHARED_BACKING_INTERNAL_H
#define RTCORE_V04_PRIVATE_SHARED_BACKING_INTERNAL_H

#include "rtcore_v04_private_shared_backing.h"

namespace rtcore {
namespace v04 {
namespace private_shared {
namespace detail {

status_kind peek_runtime_write_ack(const backing_state_v0 &state,
                                   uint64_t service_cycle,
                                   runtime_write_ack_v0 *ack);

status_kind commit_runtime_write_ack(backing_state_v0 *state,
                                     uint64_t service_cycle,
                                     const runtime_write_ack_v0 &ack);

}  // namespace detail
}  // namespace private_shared
}  // namespace v04
}  // namespace rtcore

#endif
