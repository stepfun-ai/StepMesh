/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */
#ifndef PS_INTERNAL_BACKEND_H_
#define  PS_INTERNAL_BACKEND_H_

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>

#include "ps/internal/env.h"
#include "dmlc/logging.h"

namespace ps {

enum {
  BACKEND_OK = 0,
  BACKEND_FAILED = -1
};

/**
 * \brief Abstract Backend Class
 */
class Backend {
 public:
  /**
   * \brief Set device index for current thread
   * @param dev device index
   * @return BACKEND_OK if success to set device, otherwise BACKEND_FAILED
   */
  virtual int SetDevice(int dev) = 0;

  /**
   * \brief Get device index for current thread
   * @return device index
   */
  virtual int GetDeviceId() = 0;

  /**
   * \brief Get the torch device of current device
   * @return torch device
   */
  virtual at::Device GetDevice() = 0;

  /**
   * \brief Alloc memory over the device
   * @param size size to alloc
   * @return nullptr if failed to alloc, otherwise the memory pointer
   */
  virtual void* Alloc(uint64_t size) = 0;

  /**
   * \brief Free memory allocated over device
   * @param m
   */
  virtual void Free(void* m) = 0;

  /**
   * \brief Create stream event
   * @return nullptr or the event pointer
   */
  virtual void* CreateEvent() = 0;

  /**
   * \brief Free the event
   * @return BACKEND_OK if succeed to free the event, otherwise BACKEND_FAILED
   */
  virtual int FreeEvent(void* event) = 0;

  /**
   *\brief Record the event over the stream
   * @param event the created event
   * @param stream user designated stream, can be nullptr (using default stream)
   * @return BACKEND_OK if succeed to record the event, otherwise BACKEND_FAILED
   */
  virtual int RecordEvent(void* event, void* stream) = 0;

  /**
   *\brief Sync and wait the event
   * @param event the created event
   * @return BACKEND_OK if succeed to synchronize the event,
   *    otherwise BACKEND_FAILED
   */
  virtual int SyncEvent(void* event) = 0;

  /**
   * \brief Get the backend implementation
   * @return the backend implementation
   */
  static inline Backend* Get() {
    return GetImpl();
  }

  static void Register(const std::string& name,
                       Backend* backend) {
    RegisterImpl(name, backend);
  }

 protected:
  Backend() = default;

 private:
  static std::mutex backends_mutex_;
  static std::unordered_map<std::string, Backend*> backends_;

  static Backend* GetImpl() {
    static Backend* backend_impl = nullptr;
    if (backend_impl == nullptr) {
      std::unique_lock<std::mutex> lock(backends_mutex_);
      std::string backend_type = "GPU";
      backend_type = Environment::Get()->find("STEPMESH_BAKCEND", backend_type);
      PS_CHECK_NE(backends_.find(backend_type), backends_.end())
          << "failed to get backend impl: " <<  backend_type;
      backend_impl = backends_[backend_type];
    }
    return backend_impl;
  }

  static void RegisterImpl(const std::string& name,
                           Backend* backend) {
    std::unique_lock<std::mutex> lock(backends_mutex_);
    backends_[name] = backend;
  }
};


}  // namespace ps

#endif  // PS_INTERNAL_BACKEND_H_
