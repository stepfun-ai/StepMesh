/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */
#ifndef PS_TRACE_H_
#define  PS_TRACE_H_

namespace ps {

#ifdef STEPAF_ENABLE_TRACE
struct Trace {
  // on start of a reqiest/response
  uint64_t start = 0;
  // on rdma post send
  uint64_t postsend = 0;
  // on receive an reqiest/response
  uint64_t postrecv = 0;
  // on processed
  uint64_t process = 0;
};
#endif

}  // namespace ps

#endif  // PS_TRACE_H_
