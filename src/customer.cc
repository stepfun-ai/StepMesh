/**
 *  Copyright (c) 2015 by Contributors
 *  Modifications Copyright (C) by StepAI Contributors. 2025.
 */
#include "ps/internal/customer.h"

#include <atomic>
#include <fstream>
#include <list>

#include "ps/internal/postoffice.h"
#include "ps/internal/threadsafe_queue.h"
#include "emmintrin.h"

namespace ps {
const int Node::kEmpty = std::numeric_limits<short>::max();
const int Meta::kEmpty = std::numeric_limits<short>::max();

Customer::Customer(int app_id, int customer_id,
                   const Customer::RecvHandle& recv_handle,
                   Postoffice* postoffice)
    : app_id_(app_id),
      customer_id_(customer_id),
      recv_handle_(recv_handle),
      postoffice_(postoffice) {
  postoffice_->AddCustomer(this);
  recv_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&Customer::Receiving, this));
}

Customer::~Customer() {
  postoffice_->RemoveCustomer(this);
  Message msg;
  msg.meta.control.cmd = Control::TERMINATE;
  recv_queue_.Push(msg);
  recv_thread_->join();
}

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
  // std::unique_lock<std::mutex> lk(tracker_mu_);
  int spin_count = 0;
  const int kMaxSpinCount = 1000;
  // tracker_mu_.lock();
  auto* req = tracker_[timestamp];
  // tracker_mu_.unlock();
  while (req->count.load(std::memory_order_acquire)
         != req->response_count.load(std::memory_order_acquire)) {
    if (spin_count < kMaxSpinCount) {
     spin_count++;
    } else {
      _mm_pause();
    }
  }
#ifdef STEPAF_ENABLE_TRACE
  req->response.process = GetNanosecond();
#endif
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
      // std::lock_guard<std::mutex> lk(tracker_mu_);
      tracker_[recv.meta.timestamp]->response_count.fetch_add(
          1, std::memory_order_release);
      // tracker_cond_.notify_all();
    }
  }
}

void Customer::DirectProcess(Message& recv) {
  if (!recv.meta.control.empty() &&
      recv.meta.control.cmd == Control::TERMINATE) {
    return;
  }
  recv_handle_(recv);
  PS_LOG(TRACE) << "Processing Ts:" <<recv.meta.timestamp;

  if (!recv.meta.request) {
    auto t = tracker_[recv.meta.timestamp];
    PS_CHECK_NE(t, nullptr) << "could not find tracker";
#ifdef STEPAF_ENABLE_TRACE
    t->request = recv.meta.request_trace;
    t->response = recv.meta.response_trace;
#endif  // STEPAF_ENABLE_TRACE
    t->response_count.fetch_add(1, std::memory_order_release);
  }
}

#ifdef STEPAF_ENABLE_TRACE
std::pair<struct Trace, struct Trace> Customer::FetchTrace(int timestamp) {
  std::unique_lock<std::mutex> lk(tracker_mu_);
  auto p = tracker_[timestamp];
  return std::make_pair(p->request, p->response);
}
#endif  // STEPAF_ENABLE_TRACE

}  // namespace ps
