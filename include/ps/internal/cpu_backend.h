/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#ifndef PS_INTERNAL_CPU_BACKEND_H_
#define PS_INTERNAL_CPU_BACKEND_H_

#include <memory>

#include "ps/internal/backend.h"

namespace ps {

class CpuBackend : public Backend {
 public:
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
  /** \brief for cpu backend, the device stands for numa id */
  int numa_id_ = -1;
};

}  // namespace ps

#endif  // PS_INTERNAL_CPU_BACKEND_H_
