/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#ifndef PS_INTERNAL_GPU_BACKEND_H_
#define  PS_INTERNAL_GPU_BACKEND_H_

#include <memory>

#include "ps/internal/backend.h"

namespace ps {

/**
 * \brief Nvidia GPU Backend Class
 *  the device for GPU Backend stands for gpu index
 */
class GpuBackend : public Backend {
 public:
  GpuBackend();
  int SetDevice(int dev) override;
  int GetDeviceId() override;
  at::Device GetDevice() override;
  void* Alloc(uint64_t size) override;
  void Free(void* m) override;
  void* CreateEvent() override;
  int FreeEvent(void* event) override;
  int RecordEvent(void* event, void* stream) override;
  int SyncEvent(void* event) override;

 private:
  void* CreateCudaEvent();
  int FreeCudaEvent(void* event);
  int RecordCudaEvent(void* event, void* stream);
  int SyncCudaEvent(void* event);

  void* CreateMemEvent();
  int FreeMemEvent(void* event);
  int RecordMemEvent(void* event, void* stream);
  int SyncMemEvent(void* event);

 private:
  inline void DoInitGpu() {
    static thread_local int gpu_idx = -1;
    if (gpu_idx == -1) {
      PS_CHECK_GE(gpu_idx_, 0)
          << "cannot set device " << gpu_idx_ << " for gpu backend";
      SetDevice(gpu_idx_);
      gpu_idx = gpu_idx_;
    }
  }

  /** \brief for cpu backend, the device stands for numa id */
  int gpu_idx_ = -1;
  int mem_sync_ = 1;
};

}  // namespace ps

#endif  // PS_INTERNAL_GPU_BACKEND_H_
