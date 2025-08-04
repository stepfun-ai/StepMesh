/**
 *  Copyright (c) 2015 by Contributors
 *  Modifications Copyright (C) by StepAI Contributors. 2025.
 */
#ifndef PS_INTERNAL_THREADSAFE_QUEUE_H_
#define PS_INTERNAL_THREADSAFE_QUEUE_H_
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include "ps/base.h"
#include "ps/internal/env.h"
#include "ps/internal/spsc_queue.h"

namespace ps {

/**
 * \brief thread-safe queue allowing push and waited pop
 */
template <typename T>
class ThreadsafeQueue {
 public:
  ThreadsafeQueue() : lockless_queue_(32768) {
    Environment::Get()->find("STEPAF_BUSY_POLL_ENABLE", &lockless_, 1);
    head_.store(0);
    tail_.store(0);
  }

  ~ThreadsafeQueue() {}

  /**
   * \brief push an value into the end. threadsafe.
   * \param new_value the value
   */
  inline void Push(T new_value) {
    if (lockless_) {
      // PushLockless(std::move(new_value));
      PushAtomic(std::move(new_value));
      return;
    }
    {
      std::unique_lock<std::mutex> lk(mu_);
      queue_.push(std::move(new_value));
    }
    cond_.notify_all();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  inline void WaitAndPop(T* value) {
    if (lockless_) {
      // WaitAndPopLockless(value);
      WaitAndPopAtomic(value);
      return;
    }
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this] { return !queue_.empty(); });
    *value = std::move(queue_.front());
    queue_.pop();
  }

  /**
   * \brief peek queue size
   */
  int Size() {
    if (lockless_) {
      return SizeAtomic();
    }
    std::unique_lock<std::mutex> lk(mu_);
    return queue_.size();
  }

 private:
  // lockless impl
  void PushLockless(T new_value) {
    write_mu_.lock();
    lockless_queue_.push(std::move(new_value));
    write_mu_.unlock();
  }

  void WaitAndPopLockless(T* value) {
    for (;;) {
      read_mu_.lock();
      if (lockless_queue_.front()) {
        *value = *(lockless_queue_.front());
        lockless_queue_.pop();
        read_mu_.unlock();
        break;
      }
      read_mu_.unlock();
      std::this_thread::yield();
    }
  }

  void PushAtomic(T new_value) {
    const size_t current_tail = tail_.load(std::memory_order_relaxed);
    const size_t next_tail = (current_tail + 1) % capacity_;
    while (next_tail == head_.load(std::memory_order_acquire)) {
      // Queue is full, spin and yield
      // std::this_thread::yield()
      continue;
    }

    buffer_[current_tail] = new_value;
    // release: ensures writes preceding this store in this thread are visible
    // to other threads that perform an acquire load on 'tail_'.
    tail_.store(next_tail, std::memory_order_release);
    return;
  }

  void WaitAndPopAtomic(T* value) {
    const size_t current_head = head_.load(std::memory_order_relaxed);

    // Check if the queue is empty
    // acquire: ensures writes preceding this load in other threads are
    // visible. Specifically, ensures the producer's writes to 'tail_' are
    // visible.
    int max_count = 1000;
    int count = 0;
    while (current_head == tail_.load(std::memory_order_acquire)) {
      // Queue is empty, spin and yield
      count++;
      if (count > max_count) {
        count = 0;
        // _mm_pause();
      }
    }

    *value = std::move(buffer_[current_head]);
    // release: ensures writes preceding this store in this thread are visible
    // to other threads that perform an acquire load on 'head_'.
    head_.store((current_head + 1) % capacity_, std::memory_order_release);
  }

  int SizeAtomic() {
    // Use acquire-release for consistent view, though still approximate
    auto current_tail = tail_.load(std::memory_order_acquire);
    auto current_head = head_.load(std::memory_order_acquire);
    if (current_tail >= current_head) {
      return static_cast<int>(current_tail - current_head);
    } else {
      return static_cast<int>(capacity_ - (current_head - current_tail));
    }
  }

  int SizeLockless() {
    std::unique_lock<std::mutex> lk_read(read_mu_);
    std::unique_lock<std::mutex> lk_write(write_mu_);
    return lockless_queue_.size();
  }

  int lockless_;

  // cv implementation
  mutable std::mutex mu_;
  std::queue<T> queue_;
  std::condition_variable cond_;

  // lockless implementation
  mutable std::mutex read_mu_;
  mutable std::mutex write_mu_;
  rigtorp::SPSCQueue<T> lockless_queue_;
  T buffer_[32768];

  std::atomic<size_t> head_;  // Consumer's pointer
  std::atomic<size_t> tail_;  // Producer's pointer
  const size_t capacity_ = 32768;
};

}  // namespace ps

#endif  // PS_INTERNAL_THREADSAFE_QUEUE_H_
