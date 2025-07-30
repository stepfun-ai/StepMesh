/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#include "ps/internal/cpu_backend.h"

#include "ps/internal/backend.h"

namespace ps {

int CpuBackend::SetDevice(int dev) {
  PS_CHECK_GE(dev, 0) << "cannot set dev=" << dev << " for cpu backend";
  numa_id_ = dev;
  return BACKEND_OK;
}

int CpuBackend::GetDeviceId() { return numa_id_; }

at::Device CpuBackend::GetDevice() { return at::Device(at::kCPU); }

void* CpuBackend::Alloc(uint64_t size) { return malloc(size); }

void CpuBackend::Free(void* m) {
  PS_CHECK_NE(m, nullptr) << "cpu backend cannot free null memory";
  free(m);
}

void* CpuBackend::CreateEvent() { return nullptr; }

int CpuBackend::FreeEvent(void* event) { return BACKEND_OK; }

int CpuBackend::RecordEvent(void* event, void* stream) { return BACKEND_OK; }

int CpuBackend::SyncEvent(void* event) { return BACKEND_OK; }

}  // namespace ps
