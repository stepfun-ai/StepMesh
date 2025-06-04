/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include "ps/internal/backend.h"
#include "ps/internal/gpu_backend.h"

namespace ps {

int GpuBackend::SetDevice(int dev) {
  PS_CHECK_GE(dev, 0) << "cannot set dev=" << dev << " for gpu backend";
  PS_CHECK_LE(dev, 7) << "cannot set dev=" << dev << " for gpu backend";
  static thread_local int gpu_idx = -1;

  gpu_idx_ = dev;
  if (gpu_idx == -1 || gpu_idx != gpu_idx_) {
    gpu_idx = gpu_idx_;
    auto result = cudaSetDevice(gpu_idx_);
    if (result != cudaSuccess) {
      PS_LOG(WARNING) << "failed to set device to "
                      << gpu_idx << " cuda result=" << result;
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
  return at::Device(at::kCUDA, gpu_idx_);
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
  PS_CHECK_EQ(err, cudaSuccess) << "cudaFree failed for ptr " << (void*)(m)
                                << " (" << cudaGetErrorString(err) << ")";
}

void* GpuBackend::CreateEvent() {
  DoInitGpu();
  cudaEvent_t ev;
  auto status = cudaEventCreateWithFlags(
      &ev, cudaEventDisableTiming | cudaEventBlockingSync);
  PS_CHECK_EQ(status, cudaSuccess) << "cudaEventCreateWithFlags failed for gpu "
                                   << gpu_idx_;
  return (void*)(ev);
}

int GpuBackend::FreeEvent(void* event) {
  DoInitGpu();

  PS_CHECK_NE(event, nullptr) << "backend cannot free null event";
  cudaError_t err = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
  PS_CHECK_EQ(err, cudaSuccess) << "cudaEventDestroy failed for event "
                                << (void*)(event)
                                << " (" << cudaGetErrorString(err) << ")";
  return BACKEND_OK;
}

int GpuBackend::RecordEvent(void* event, void* stream) {
  DoInitGpu();
  cudaStream_t cuda_stream;
  PS_CHECK_NE(event, nullptr) << "backend cannot record null event";
  if (stream == nullptr) {
    cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  } else {
    cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  }
  auto status = cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                cuda_stream);
  if (status == cudaSuccess) {
    return BACKEND_OK;
  } else {
    PS_LOG(WARNING) << "failed to record cuda event: "
                    << " (" << cudaGetErrorString(status) << ")";
    return BACKEND_FAILED;
  }
}

int GpuBackend::SyncEvent(void* event) {
  DoInitGpu();

  PS_CHECK_NE(event, nullptr) << "backend cannot sync null event";
  auto status = cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event));
  if (status != cudaSuccess) {
    PS_LOG(WARNING) << "failed to sync cuda event: "
                    << " (" << cudaGetErrorString(status) << ")";
    return BACKEND_FAILED;
  }
  return BACKEND_OK;
}

}  // namespace ps