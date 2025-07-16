/* Copyright (c) 2025, StepFun Authors. All rights reserved. */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>   
#include <iostream>
#include <stdlib.h>
#include <mutex>
#include <pybind11/stl.h>

#include "ps/ps.h"

#if DMLC_USE_CUDA
#include <cuda_runtime.h>
#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }
#endif

#define DIVUP(x, y) (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) (DIVUP((x), (y))*(y))

uint64_t DecodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

inline int GetKeyIndex(int global_rank, int microbacth, bool a2f) {
  int key = 0;
  if (!a2f) {
    key = 1 << 16;
  }
  key |= global_rank << 8;
  key |= microbacth;
  return key;
}

void aligned_memory_alloc(void** ptr, size_t size, int device_idx, ps::DeviceType device) {
  if (device == ps::CPU) {
    // CPU Alloc
    size_t page_size = sysconf(_SC_PAGESIZE);
    void* p;
    int size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    PS_CHECK_EQ(ret, 0) << "posix_memalign error: " << strerror(ret);
    PS_CHECK(p);
    memset(p, 1, size);
    *ptr = p;
  } else {
    CHECK(device == ps::GPU);
#if DMLC_USE_CUDA
    // GPU Alloc, malloc should automatically gives page aligned.
    CUDA_CALL(cudaSetDevice(device_idx));
    CUDA_CALL(cudaMalloc(ptr, size));
#else
    CHECK(false) << "Please build with USE_CUDA=1";
#endif
  }
}

// signal(SIGSEGV, printStackTrace);
void printStackTrace(int sig) {
    void *array[10];
    size_t size;
    char **strings;
    size_t i;

    // 获取当前堆栈信息
    size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);

    printf("Obtained %zd stack frames.\n", size);

    // 打印堆栈信息
    for (i = 0; i < size; i++) {
        printf("%d: %s\n", getpid(), strings[i]);
    }

    free(strings);
    signal(sig, SIG_DFL); // 恢复默认的信号处理函数
    kill(getpid(), sig); // 重新发送信号，以便默认处理函数可以工作
}
