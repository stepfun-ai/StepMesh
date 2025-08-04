/*************************************************************************
* Copyright (c) 2024, STEP AI. All rights reserved.
************************************************************************/
#ifndef PS_TEST_COMMON_H_
#define PS_TEST_COMMON_H_

#include <unistd.h>

#include <algorithm>
#include <vector>
#include <string>
#include <climits>

#include <ATen/ATen.h>
#include <torch/torch.h>

#ifdef DMLC_USE_CUDA
  #include <ATen/cuda/CUDAContext.h>
  #include <ATen/cuda/CUDAEvent.h>
#endif

#include "ps/ps.h"

using namespace ps;

static struct {
  int gpu = 0;
} g_conf;

static struct {
  int batch_max = 128;
  int tensor_num = 1;
} g_worker_conf;

#define  CUDA_CALL(func)                                      \
{                                                          \
  cudaError_t e = (func);                                  \
  PS_CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
      << "CUDA: " << cudaGetErrorString(e);                \
}

static inline void InitUtestConfig() {
  Environment::Get()->find("UTEST_WORKER_BATCH_MAX",
                           &g_worker_conf.batch_max,
                           g_worker_conf.batch_max);
  Environment::Get()->find("UTEST_WORKER_TENSOR_NUM",
                           &g_worker_conf.tensor_num,
                           g_worker_conf.tensor_num);

  Environment::Get()->find("STEPMESH_GPU", &g_conf.gpu, g_conf.gpu);
}

static inline at::Tensor CreateTensor(
                               std::vector<int64_t> shape,
                               at::ScalarType dtype,
                               int gpu,
                               bool random = false) {
  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .memory_format(at::MemoryFormat::Contiguous)
                     .device(at::Device(at::kCUDA, gpu));
  if (random) {
    return torch::rand(shape, options);
  } else {
    return torch::zeros(shape, options);
  }
}

#endif  // PS_TEST_COMMON_H_
