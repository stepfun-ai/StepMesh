/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */
#ifndef PS_INTERNAL_TRACE_H_
#define  PS_INTERNAL_TRACE_H_

namespace ps {

struct Trace {
  uint64_t pre_start = 0;
  // on start of a request/response
  uint64_t start = 0;
  // on rdma post send
  uint64_t postsend = 0;
  // on receive an reqiest/response
  uint64_t postrecv = 0;
  // on processed
  uint64_t process = 0;
};

}  // namespace ps

#endif  // PS_INTERNAL_TRACE_H_
