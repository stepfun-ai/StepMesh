/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#include "ps/internal/gpu_backend.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

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
    auto result = cudaSetDevice(gpu_idx_);
    if (result != cudaSuccess) {
      PS_LOG(WARNING) << "failed to set device to " << gpu_idx
                      << " cuda result=" << result;
    }
  }

  return BACKEND_OK;
}

int GpuBackend::GetDeviceId() {
  static thread_local int gpu_idx = -1;
  if (gpu_idx != -1) {
    auto result = cudaGetDevice(&gpu_idx);
    PS_CHECK_EQ(result, cudaSuccess)
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
  auto cuda_err = cudaMalloc(&ptr, size);
  PS_CHECK_EQ(cuda_err, cudaSuccess)
      << "cudaMalloc failed for gpu " << gpu_idx_ << " with size " << size
      << " (" << cudaGetErrorString(cuda_err) << ")";
  return ptr;
}

void GpuBackend::Free(void* m) {
  PS_CHECK_NE(m, nullptr) << "backend cannot free null memory";
  cudaError_t err = cudaFree(m);
  PS_CHECK_EQ(err, cudaSuccess)
      << "cudaFree failed for ptr " << reinterpret_cast<void*>(m) << " ("
      << cudaGetErrorString(err) << ")";
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
  cudaEvent_t* ev = nullptr;
  cudaMallocHost(&ev, sizeof(cudaEvent_t));
  auto status = cudaEventCreateWithFlags(ev, cudaEventDisableTiming);
  PS_CHECK_EQ(status, cudaSuccess)
      << "cudaEventCreateWithFlags failed for gpu " << gpu_idx_;
  return reinterpret_cast<void*>(ev);
}

int GpuBackend::FreeCudaEvent(void* event) {
  auto ev = reinterpret_cast<cudaEvent_t*>(event);
  cudaError_t err = cudaEventDestroy(*ev);
  PS_CHECK_EQ(err, cudaSuccess)
      << "cudaEventDestroy failed for event " << reinterpret_cast<void*>(event)
      << " (" << cudaGetErrorString(err) << ")";
  cudaFreeHost(ev);
  return BACKEND_OK;
}

int GpuBackend::RecordCudaEvent(void* event, void* stream) {
  cudaStream_t cuda_stream;
  if (stream == nullptr) {
    cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  } else {
    cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  }

  auto ev = reinterpret_cast<cudaEvent_t*>(event);
  auto status = cudaEventRecord(*ev, cuda_stream);
  if (status == cudaSuccess) {
    return BACKEND_OK;
  } else {
    PS_LOG(WARNING) << "failed to record cuda event: "
                    << " (" << cudaGetErrorString(status) << ")";
    return BACKEND_FAILED;
  }
}

int GpuBackend::SyncCudaEvent(void* event) {
  auto ev = reinterpret_cast<cudaEvent_t*>(event);
  cudaError_t status;
  while (true) {
    status = cudaEventQuery(*ev);
    if (status == cudaErrorNotReady) {
      sched_yield();
      continue;
    }
    break;
  }
  if (status != cudaSuccess) {
    PS_LOG(WARNING) << "failed to sync cuda event: "
                    << " (" << cudaGetErrorString(status) << ")";
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
  AT_CUDA_CHECK(cudaMallocHost(&ev, sizeof(GpuBackendMemEvent)));
  AT_CUDA_CHECK(cudaMalloc(&(ev->gpu_flag), sizeof(int)));
  AT_CUDA_CHECK(cudaMemset(ev->gpu_flag, 0, sizeof(int)));
  AT_CUDA_CHECK(
      cudaMallocHost(reinterpret_cast<void**>(&(ev->cpu_flag)), sizeof(int)));
  *ev->cpu_flag = 0;
  return reinterpret_cast<void*>(ev);
}

int GpuBackend::FreeMemEvent(void* event) {
  auto ev = reinterpret_cast<GpuBackendMemEvent*>(event);
  AT_CUDA_CHECK(cudaFree(ev->gpu_flag));
  AT_CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(ev->cpu_flag)));
  AT_CUDA_CHECK(cudaFreeHost(ev));
  return BACKEND_OK;
}

int GpuBackend::RecordMemEvent(void* event, void* stream) {
  auto ev = reinterpret_cast<GpuBackendMemEvent*>(event);
  *(ev->cpu_flag) = 1;
  cudaStream_t cuda_stream;
  if (stream == nullptr) {
    cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  } else {
    cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  }

  AT_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(ev->cpu_flag),
                                ev->gpu_flag, sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream));
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
