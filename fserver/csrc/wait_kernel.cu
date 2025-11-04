#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void write_flag_kernel(int64_t* flag, int64_t seq) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    flag[0] = seq;
  }
}

__global__ void wait_flag_kernel(int64_t* flag, int64_t seq) {
  if (threadIdx.x == 0) {
    // Mark pointer volatile so we reload host-written values each iteration.
    volatile int64_t* flag_ptr = flag;
    int64_t value = flag_ptr[0];
    while (value < seq) {
      __nanosleep(128);
      value = flag_ptr[0];
    }
  }
}

static void check_cuda(cudaError_t err, const char* msg) {
  TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor map_pinned_tensor(torch::Tensor tensor, int64_t device_index) {
  TORCH_CHECK(tensor.is_pinned(), "tensor must be pinned");
  void* host_ptr = tensor.data_ptr();
  void* device_ptr = nullptr;
  check_cuda(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0),
             "cudaHostGetDevicePointer failed");
  auto options = tensor.options().device(torch::kCUDA, device_index);
  auto sizes = tensor.sizes();
  auto strides = tensor.strides();
  return torch::from_blob(device_ptr, sizes, strides, [](void*){}, options);
}

void write_flag(torch::Tensor flag, int64_t seq) {
  TORCH_CHECK(flag.is_cuda(), "flag must be a CUDA tensor");
  auto stream = at::cuda::getCurrentCUDAStream(flag.device().index());
  write_flag_kernel<<<1, 1, 0, stream>>>(flag.data_ptr<int64_t>(), seq);
  check_cuda(cudaGetLastError(), "write_flag_kernel launch failed");
}

void wait_flag(torch::Tensor flag, int64_t seq) {
  TORCH_CHECK(flag.is_cuda(), "flag must be a CUDA tensor");
  auto stream = at::cuda::getCurrentCUDAStream(flag.device().index());
  wait_flag_kernel<<<1, 1, 0, stream>>>(flag.data_ptr<int64_t>(), seq);
  check_cuda(cudaGetLastError(), "wait_flag_kernel launch failed");
}
