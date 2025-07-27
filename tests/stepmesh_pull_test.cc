/**
 *  Copyright (C) 2025 by StepAI Contributors.
 */
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdlib>

#include <unistd.h>

#include "test_common.h"

std::unordered_map<uint64_t, at::Tensor> g_mem;

void PullHandler(const AFTensorMeta& req_meta, AFTensorServer* server) {
  auto key = req_meta.pull_tensors[0].key;
  KeyTensor key_tensor;
  key_tensor.key = key;
  auto iter = g_mem.find(key);
  if (iter != g_mem.end()) {
    key_tensor.val = iter->second;
  } else {
    key_tensor.val = CreateTensor({g_conf.size},
                 at::kBFloat16, g_conf.gpu, true);
    g_mem[key] = key_tensor.val;
  }

  server->Response(req_meta, { key_tensor });
}

void RunWorker(AFTensorWorker* kv) {
  auto pull_tensor = CreateTensor({g_conf.size},
                                  at::kBFloat16, g_conf.gpu, false);
  auto PushPull = [kv, pull_tensor] () {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> timestamps;
    auto push_batch = KeyTensorBatch();
    auto pull_batch = KeyTensorBatch();
    pull_batch.push_back(KeyTensor{
        1, pull_tensor,
    });

    kv->Wait(kv->ZBatchPushPull(push_batch, pull_batch));
    auto end = std::chrono::high_resolution_clock::now();
    return (end - start).count();
  };

  PS_LOG(INFO) << "warmup starts";
  std::vector<int64_t> timestamps;

  for (int iter = 0; iter < g_conf.iter; iter++) {
    auto pushpull_ts = PushPull();
    timestamps.emplace_back(pushpull_ts);
    if ((iter % 1000 == 999)) {
      DumpLatency("pull batch latency: ", timestamps);
      timestamps.clear();
    }
  }
}

void StartFFNServer() {
  PS_LOG(INFO) << "Run server: gpu=" << g_conf.gpu
               << ", node rank=" << g_conf.node_rank
               << ", group size=" << g_conf.group_size;
  StartPS(0, Node::SERVER,
          g_conf.node_rank * g_conf.group_size + g_conf.gpu, true);
  Backend::Get()->SetDevice(g_conf.gpu);
  StartServer(PullHandler);
  Finalize(0, Node::SERVER, true);
  PS_LOG(INFO) << "Server ends";
}

void StartWorkers() {
  PS_LOG(INFO) << "run worker: gpu=" << g_conf.gpu
               << ", node rank=" << g_conf.node_rank
               << ", group size=" << g_conf.group_size;
  StartPS(0, Node::WORKER,
          g_conf.node_rank * g_conf.group_size + g_conf.gpu, true);
  Backend::Get()->SetDevice(g_conf.gpu);
  AFTensorWorker af_worker(g_conf.gpu);
  InitWorker(&af_worker);
  RunWorker(&af_worker);
  Finalize(0, Node::WORKER, true);
  PS_LOG(INFO) << "Simulated worker is DONE";
}

int main(int argc, char *argv[]) {
  InitConfig();
  PS_LOG(INFO) << "StepMesh Echo Tests: gpu_num="
            << g_conf.gpu_num << ", role=" << g_conf.role_str;
  if (g_conf.role == Node::SCHEDULER) {
    StartScheduler();
  } else if (g_conf.role == Node::SERVER) {
    StartFFNServer();
  } else if (g_conf.role == Node::WORKER) {
    StartWorkers();
  }
  return 0;
}
