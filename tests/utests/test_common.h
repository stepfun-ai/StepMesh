/**
 *  Copyright (C) 2025 by StepAI Contributors.
 */
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

#include "ps/af_tensor_app.h"

#include "ps/ps.h"

using namespace ps;

static struct {
  int warmup_iter = 10;
  int iter = INT_MAX;
  int size = 7168;
  int debug = false;
  int gpu_num = 1;
  int gpu = 0;
  int mb_num = 3;
  int group_size;
  int node_rank ;
  std::string role_str;
  ps::Node::Role role;
} g_conf;

#define DIVUP(x, y) (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) (DIVUP((x), (y))*(y))

void InitConfig() {
  Environment::Get()->find("BENCHMARK_WARMUP",
                           &g_conf.warmup_iter, g_conf.warmup_iter);
  Environment::Get()->find("BENCHMARK_ITER", &g_conf.iter, g_conf.iter);
  Environment::Get()->find("BENCHMARK_SIZE", &g_conf.size, g_conf.size);
  Environment::Get()->find("STEPMESH_GPU", &g_conf.gpu, g_conf.gpu);

  g_conf.gpu = GetGpuId();
  g_conf.group_size = GetGroupSize();
  g_conf.node_rank = GetNodeRank();
  const char* val = PS_CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
  g_conf.role_str = std::string(val);
  g_conf.role = GetRole(g_conf.role_str);
  g_conf.debug = Environment::Get()->find("DEBUG_MODE") != nullptr;
}

static inline void AlignedMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  PS_CHECK_EQ(ret, 0) << "posix_memalign error: " << strerror(ret);
  PS_CHECK(p);
  memset(p, 1, size);
  *ptr = p;
}

static inline std::vector<int64_t> GetPercentile(
    std::vector<int64_t>& vec, std::vector<int> percentiles) {
  std::vector<int64_t> result;
  result.reserve(percentiles.size());
  std::sort(vec.begin(), vec.end());
  for (auto percentile : percentiles) {
    PS_CHECK(percentile >= 0 && percentile < 100);
    auto percentile_idx = int(vec.size() * percentile / 100) - 1;
    result.emplace_back(vec[percentile_idx]);
  }
  return result;
}

static inline int64_t GetMean(std::vector<int64_t>& vec) {
  PS_CHECK(!vec.empty());
  int64_t result = 0;
  for (auto data : vec) {
    result += data;
  }
  return result / vec.size();
}

static inline void DumpLatency(const std::string& head, std::vector<int64_t>& vec) {
  auto pull_mean = GetMean(vec);
  auto pull_percentile = GetPercentile(vec, {50, 90, 99});
  LL << head << ": mean=" << pull_mean  / 1000.0
     << "us, min=" << vec[0] / 1000.0
     << "us, p50=" << pull_percentile[0] / 1000.0
     << "us, p90=" << pull_percentile[1] / 1000.0
     << "us, p99=" << pull_percentile[2] / 1000.0
     << "us, max=" << vec[vec.size() - 1]  / 1000.0 << "us";
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

static inline void StartServer(
    std::function<void(const AFTensorMeta& req_meta, AFTensorServer* server)>
        func) {
  AFTensorServer* server = new AFTensorServer(g_conf.gpu);
  server->SetRequestHandle(func);
  RegisterExitCallback([server]() { delete server; });
}

static inline void InitWorker(AFTensorWorker* kv) {
  Postoffice::GetWorker(g_conf.gpu)->Barrier(0, ps::kWorkerGroup);
  PS_LOG(INFO) << "finish worker init.";
}

static inline void StartScheduler() {
  PS_LOG(INFO) << "Scheduler starts";
  StartPS(0, Node::SCHEDULER, -1, true);
  Finalize(0, Node::SCHEDULER, true);
  PS_LOG(INFO) << "Scheduler ends";
}

#endif  // PS_TEST_COMMON_H_
