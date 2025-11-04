/**
 *  Copyright (c) 2015 by Contributors
 *  Modifications Copyright (C) by StepAI Contributors. 2025.
 */

#include "ps/internal/customer.h"

#include <emmintrin.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <limits>
#include <list>
#include <utility>

#include "ps/internal/postoffice.h"
#include "ps/internal/threadsafe_queue.h"

namespace ps {

const int Node::kEmpty = std::numeric_limits<int16_t>::max();
const int Meta::kEmpty = std::numeric_limits<int16_t>::max();
const int kMaxSpinCount = 10000;

Customer::Customer(int app_id, int customer_id,
                   const Customer::RecvHandle& recv_handle,
                   Postoffice* postoffice)
    : app_id_(app_id),
      customer_id_(customer_id),
      recv_handle_(recv_handle),
      postoffice_(postoffice) {
  postoffice_->AddCustomer(this);
}

Customer::~Customer() { postoffice_->RemoveCustomer(this); }

int Customer::NewRequest(int recver) {
  PS_CHECK(recver == kServerGroup) << recver;
  std::lock_guard<std::mutex> lk(tracker_mu_);
  // for push/pull requests, the worker only communication with one instance
  // from each server instance group
  int num = postoffice_->GetNodeIDs(recver).size() / postoffice_->group_size();
  auto* t = new CustomerTracker();
  t->count = num;
  t->response_count.store(0);
  t->response_count_cache = 0;
  t->start_time = GetNanosecond(false);
  tracker_.push_back(t);
  return tracker_.size() - 1;
}

void Customer::WaitRequest(int timestamp, uint64_t timeout_ms) {
  uint64_t timeout_ns = timeout_ms * 1000000;
  auto* req = tracker_[timestamp];
  int spin_count = 0;
  while (req->count != req->response_count.load(std::memory_order_acquire)) {
    if (spin_count < kMaxSpinCount) {
      spin_count++;
    } else {
      _mm_pause();
      uint64_t now = GetNanosecond(false);
      // 1s for timeout
      if (now - req->start_time > timeout_ns) {
        PS_LOG(FATAL) << "request timeout " << timeout_ms << "ms, handler "
                      << timestamp << " " << (now - req->start_time) / 1000
                      << "us"
                      << " "
                      << req->response_count.load(std::memory_order_acquire)
                      << " " << req->count;
      }
    }
  }
}

int Customer::NumResponse(int timestamp) {
  // std::unique_lock<std::mutex> lk(tracker_mu_);
  return tracker_[timestamp]->count;
}

void Customer::AddResponse(int timestamp, int num) {
  // std::unique_lock<std::mutex> lk(tracker_mu_);
  tracker_[timestamp]->response_count.fetch_add(num, std::memory_order_release);
}

void Customer::Receiving() {
  while (true) {
    Message recv;
    recv_queue_.WaitAndPop(&recv);
    if (!recv.meta.control.empty() &&
        recv.meta.control.cmd == Control::TERMINATE) {
      break;
    }
    recv_handle_(recv);
    if (!recv.meta.request) {
      auto t = tracker_[recv.meta.timestamp];
      PS_CHECK_NE(t, nullptr) << "could not find tracker";
#ifdef STEPMESH_ENABLE_TRACE
      t->request = recv.meta.request_trace;
      t->response = recv.meta.response_trace;
#endif  // STEPMESH_ENABLE_TRACE
      t->response_count.fetch_add(1, std::memory_order_release);
    }
  }
}

void Customer::DirectProcess(Message& recv) {
  if (!recv.meta.control.empty() &&
      recv.meta.control.cmd == Control::TERMINATE) {
    return;
  }
  recv_handle_(recv);

  if (!recv.meta.request) {
    auto t = tracker_[recv.meta.timestamp];
    PS_CHECK_NE(t, nullptr) << "could not find tracker";
#ifdef STEPMESH_ENABLE_TRACE
    t->request = recv.meta.request_trace;
    t->response = recv.meta.response_trace;
#endif  // STEPMESH_ENABLE_TRACE
    PS_VLOG(4) << "recv response " << recv.meta.timestamp << " "
               << recv.meta.DebugString();
    t->response_count.fetch_add(1, std::memory_order_release);
    t->response_count_cache += 1;
  }
}

std::pair<struct Trace, struct Trace> Customer::FetchTrace(int timestamp) {
#ifdef STEPMESH_ENABLE_TRACE
  std::unique_lock<std::mutex> lk(tracker_mu_);
  auto p = tracker_[timestamp];
  return std::make_pair(p->request, p->response);
#endif  // STEPMESH_ENABLE_TRACE
  return std::make_pair(Trace(), Trace());
}

}  // namespace ps
