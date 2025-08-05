/**
 *  Copyright (c) 2015 by Contributors
 *  Modifications Copyright (C) by StepAI Contributors. 2025.
 */

#include "ps/internal/customer.h"

#include <emmintrin.h>

#include <atomic>
#include <fstream>
#include <limits>
#include <list>
#include <utility>

#include "ps/internal/postoffice.h"
#include "ps/internal/threadsafe_queue.h"

namespace ps {

const int Node::kEmpty = std::numeric_limits<int16_t>::max();
const int Meta::kEmpty = std::numeric_limits<int16_t>::max();
const int kMaxSpinCount = 1000;

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
  t->count.store(num);
  t->response_count.store(0);
  tracker_.push_back(t);
  return tracker_.size() - 1;
}

void Customer::WaitRequest(int timestamp) {
  auto* req = tracker_[timestamp];
  int spin_count = 0;
  while (req->count.load(std::memory_order_acquire) !=
         req->response_count.load(std::memory_order_acquire)) {
    if (spin_count < kMaxSpinCount) {
      spin_count++;
    } else {
      // _mm_pause();
    }
  }
#ifdef STEPMESH_ENABLE_TRACE
  req->response.process = GetNanosecond();
#endif  // STEPMESH_ENABLE_TRACE
}

int Customer::NumResponse(int timestamp) {
  // std::unique_lock<std::mutex> lk(tracker_mu_);
  return tracker_[timestamp]->count.load(std::memory_order_acquire);
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
    t->response_count.fetch_add(1, std::memory_order_release);
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
