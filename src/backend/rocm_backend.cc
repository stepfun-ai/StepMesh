/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#include "ps/internal/gpu_backend.h"

#ifdef DMLC_USE_ROCM
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPEvent.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "ps/internal/backend.h"

namespace ps {

GpuBackend::GpuBackend() {
  Environment::Get()->find("STEPMESH_MEM_SYNC", &mem_sync_, mem_sync_);
}

int GpuBackend::SetDevice(int dev) {
  PS_CHECK_GE(dev, 0) << "cannot set dev=" << dev << " for gpu backend";
  PS_CHECK_LE(dev, 7) << "cannot set dev=" << dev << " for gpu backend";
  static thread_local int gpu_idx = -1;

  gpu_idx_ = dev;
  if (gpu_idx == -1 || gpu_idx != gpu_idx_) {
    gpu_idx = gpu_idx_;
    auto result = hipSetDevice(gpu_idx_);
    if (result != hipSuccess) {
      PS_LOG(WARNING) << "failed to set device to " << gpu_idx
                      << " cuda result=" << result;
    }
  }

  return BACKEND_OK;
}

int GpuBackend::GetDeviceId() {
  static thread_local int gpu_idx = -1;
  if (gpu_idx != -1) {
    auto result = hipGetDevice(&gpu_idx);
    PS_CHECK_EQ(result, hipSuccess)
        << "failed to get device cuda result=" << result;
  }
  return gpu_idx;
}

at::Device GpuBackend::GetDevice() {
  PS_CHECK_GE(gpu_idx_, 0) << "device index is not initialized for gpu backend";
  return {at::kCUDA, static_cast<char>(gpu_idx_)};
}

void* GpuBackend::Alloc(uint64_t size) {
  DoInitGpu();
  void* ptr = nullptr;
  auto cuda_err = hipMalloc(&ptr, size);
  PS_CHECK_EQ(cuda_err, hipSuccess)
      << "hipMalloc failed for gpu " << gpu_idx_ << " with size " << size
      << " (" << hipGetErrorString(cuda_err) << ")";
  return ptr;
}

void GpuBackend::Free(void* m) {
  PS_CHECK_NE(m, nullptr) << "backend cannot free null memory";
  hipError_t err = hipFree(m);
  PS_CHECK_EQ(err, hipSuccess)
      << "hipFree failed for ptr " << reinterpret_cast<void*>(m) << " ("
      << hipGetErrorString(err) << ")";
}

void* GpuBackend::CreateEvent() {
  DoInitGpu();
  if (!mem_sync_) {
    return CreateCudaEvent();
  } else {
    return CreateMemEvent();
  }
}

int GpuBackend::FreeEvent(void* event) {
  DoInitGpu();
  PS_CHECK_NE(event, nullptr) << "backend cannot free null event";
  if (!mem_sync_) {
    return FreeCudaEvent(event);
  } else {
    return FreeMemEvent(event);
  }
}

int GpuBackend::RecordEvent(void* event, void* stream) {
  DoInitGpu();
  PS_CHECK_NE(event, nullptr) << "backend cannot record null event";
  if (!mem_sync_) {
    return RecordCudaEvent(event, stream);
  } else {
    return RecordMemEvent(event, stream);
  }
}

int GpuBackend::SyncEvent(void* event) {
  DoInitGpu();
  PS_CHECK_NE(event, nullptr) << "backend cannot sync null event";
  if (!mem_sync_) {
    return SyncCudaEvent(event);
  } else {
    return SyncMemEvent(event);
  }
}

void* GpuBackend::CreateCudaEvent() {
  hipEvent_t* ev = nullptr;
  hipHostMalloc(&ev, sizeof(hipEvent_t));
  auto status = hipEventCreateWithFlags(ev, hipEventDisableTiming);
  PS_CHECK_EQ(status, hipSuccess)
      << "hipEventCreateWithFlags failed for gpu " << gpu_idx_;
  return reinterpret_cast<void*>(ev);
}

int GpuBackend::FreeCudaEvent(void* event) {
  auto ev = reinterpret_cast<hipEvent_t*>(event);
  hipError_t err = hipEventDestroy(*ev);
  PS_CHECK_EQ(err, hipSuccess)
      << "hipEventDestroy failed for event " << reinterpret_cast<void*>(event)
      << " (" << hipGetErrorString(err) << ")";
  hipHostFree(ev);
  return BACKEND_OK;
}

int GpuBackend::RecordCudaEvent(void* event, void* stream) {
  hipStream_t hip_stream;
  if (stream == nullptr) {
    hip_stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  } else {
    hip_stream = reinterpret_cast<hipStream_t>(stream);
  }

  auto ev = reinterpret_cast<hipEvent_t*>(event);
  auto status = hipEventRecord(*ev, hip_stream);
  if (status == hipSuccess) {
    return BACKEND_OK;
  } else {
    PS_LOG(WARNING) << "failed to record cuda event: "
                    << " (" << hipGetErrorString(status) << ")";
    return BACKEND_FAILED;
  }
}

int GpuBackend::SyncCudaEvent(void* event) {
  auto ev = reinterpret_cast<hipEvent_t*>(event);
  hipError_t status;
  while (true) {
    status = hipEventQuery(*ev);
    if (status == hipErrorNotReady) {
      sched_yield();
      continue;
    }
    break;
  }
  if (status != hipSuccess) {
    PS_LOG(WARNING) << "failed to sync cuda event: "
                    << " (" << hipGetErrorString(status) << ")";
    return BACKEND_FAILED;
  }

  return BACKEND_OK;
}

struct GpuBackendMemEvent {
  int* gpu_flag = nullptr;
  int* cpu_flag = nullptr;
};

void* GpuBackend::CreateMemEvent() {
  struct GpuBackendMemEvent* ev = nullptr;
  AT_CUDA_CHECK(hipHostMalloc(&ev, sizeof(GpuBackendMemEvent)));
  AT_CUDA_CHECK(hipMalloc(&(ev->gpu_flag), sizeof(int)));
  AT_CUDA_CHECK(hipMemset(ev->gpu_flag, 0, sizeof(int)));
  AT_CUDA_CHECK(
      hipHostMalloc(reinterpret_cast<void**>(&(ev->cpu_flag)), sizeof(int)));
  *ev->cpu_flag = 0;
  return reinterpret_cast<void*>(ev);
}

int GpuBackend::FreeMemEvent(void* event) {
  auto ev = reinterpret_cast<GpuBackendMemEvent*>(event);
  AT_CUDA_CHECK(hipFree(ev->gpu_flag));
  AT_CUDA_CHECK(hipHostFree(reinterpret_cast<void*>(ev->cpu_flag)));
  AT_CUDA_CHECK(hipHostFree(ev));
  return BACKEND_OK;
}

int GpuBackend::RecordMemEvent(void* event, void* stream) {
  auto ev = reinterpret_cast<GpuBackendMemEvent*>(event);
  *(ev->cpu_flag) = 1;
  hipStream_t hip_stream;
  if (stream == nullptr) {
    hip_stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  } else {
    hip_stream = reinterpret_cast<hipStream_t>(stream);
  }

  AT_CUDA_CHECK(hipMemcpyAsync(reinterpret_cast<void*>(ev->cpu_flag),
                                ev->gpu_flag, sizeof(int),
                                hipMemcpyDeviceToHost, hip_stream));
  return BACKEND_OK;
}

int GpuBackend::SyncMemEvent(void* event) {
  auto ev = reinterpret_cast<GpuBackendMemEvent*>(event);
  while (*(ev->cpu_flag) == 1) {
    _mm_pause();
  }
  return BACKEND_OK;
}

}  // namespace ps
#endif  // DMLC_USE_CUDA
