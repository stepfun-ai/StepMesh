/**
 *  Copyright (C) 2025 by StepAI Contributors.
 */
#include <chrono>
#include <thread>
#include <unistd.h>

#include "test_common.h"

std::unordered_map<uint64_t, at::Tensor> g_mem;

std::vector<at::Tensor> g_af_push_tensors;
std::vector<Key> g_af_push_keys;

std::vector<at::Tensor> g_af_pull_tensors;
std::vector<Key>  g_af_pull_keys;

at::Tensor g_recv_tensor;

void SimulatedHandler(const AFTensorMeta& req_meta, AFTensorServer* server) {
  auto key = req_meta.pull_tensors[0].key;
  KeyTensor key_tensor;
  key_tensor.key = key;
  auto iter = g_mem.find(key);
  if (iter != g_mem.end()) {
    key_tensor.val = iter->second;
  } else {
    key_tensor.val = CreateTensor({g_conf.size},
                                  at::kByte, g_conf.gpu, false);
    g_mem[key] = key_tensor.val;
  }
  server->Response(req_meta, { key_tensor });
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

void StartServer() {
  AFTensorServer* server = new AFTensorServer(g_conf.gpu);
  g_recv_tensor = torch::zeros({g_conf.mb_num * g_conf.size},
                              torch::TensorOptions().dtype(at::kByte)
                                  .memory_format(at::MemoryFormat::Contiguous)
                                  .device(at::Device(at::kCUDA, g_conf.gpu)));
  std::vector<int> ranks = {0, 0, 0};
  std::vector<Key> keys = {
      static_cast<unsigned long long>(GetKeyIndex(0, 0, true)),
      static_cast<unsigned long long>(GetKeyIndex(0, 1, true)),
      static_cast<unsigned long long>(GetKeyIndex(0, 2, true))
  };

  server->RegisterRecvTensor(g_recv_tensor, ranks, keys);

  server->SetRequestHandle(SimulatedHandler);
  RegisterExitCallback([server]() { delete server; });

  Postoffice::GetServer(g_conf.gpu)->Barrier(
      0, ps::kWorkerGroup + ps::kServerGroup);
}

void InitWorker() {
  Postoffice::GetWorker(g_conf.gpu)->Barrier(
      0, ps::kWorkerGroup+ ps::kServerGroup);
  for (int mb = 0; mb < g_conf.mb_num; mb++) {
    g_af_push_keys.push_back(GetKeyIndex(0, mb, true));
    g_af_push_tensors.push_back(
        CreateTensor({g_conf.size}, at::kByte, g_conf.gpu));
    g_af_pull_keys.push_back(GetKeyIndex(0, mb, false));
    g_af_pull_tensors.push_back(
        CreateTensor({g_conf.size}, at::kByte, g_conf.gpu));
  }
  PS_LOG(INFO) << "finish worker init.";
}

void RunWorker(AFTensorWorker* kv) {
  auto PushPull = [kv] (int mb) {
    auto start = std::chrono::high_resolution_clock::now();
    auto push_batch = KeyTensorBatch();
    push_batch.push_back(
        KeyTensor{
            g_af_push_keys[mb],
            g_af_push_tensors[mb],
        });
    auto pull_batch = KeyTensorBatch();
    pull_batch.push_back(KeyTensor{
        g_af_pull_keys[mb],
        g_af_pull_tensors[mb],
    });

    kv->Wait(kv->ZBatchPushPull(push_batch, pull_batch));
    auto end = std::chrono::high_resolution_clock::now();
    return (end - start).count();
  };

  std::vector<int64_t> timestamps;
  for (int iter = 0; iter < g_conf.iter; iter++) {
    for (int mb = 0; mb < g_conf.mb_num; mb++) {
      auto ts = PushPull(mb);
      timestamps.emplace_back(ts);
    }

    if ((iter % 10 == 9)) {
      DumpLatency("pushpull batch latency: ", timestamps);
      timestamps.clear();
    }
  }
}

void StartRegisterServer() {
  PS_LOG(INFO) << "Register server Starts";
  StartPS(0, Node::SERVER,
          g_conf.node_rank * g_conf.group_size + g_conf.gpu, true);
  Backend::Get()->SetDevice(g_conf.gpu);
  StartServer();
  Finalize(0, Node::SERVER, true);
  PS_LOG(INFO) << "Register server ends";
}

void StartWorkers() {
#ifdef DMLC_USE_CUDA
  PS_LOG(INFO) << "run worker: gpu=" << g_conf.gpu
               << ", node rank=" << g_conf.node_rank
               << ", group size=" << g_conf.group_size;
#endif
  StartPS(0, Node::WORKER,
          g_conf.node_rank * g_conf.group_size + g_conf.gpu, true);
  AFTensorWorker af_worker(g_conf.gpu);
  InitWorker();
  RunWorker(&af_worker);

  Finalize(0, Node::WORKER, true);
  PS_LOG(INFO) << "Simulated attention worker is DONE";
}

int main(int argc, char *argv[]) {
  InitConfig();
  PS_LOG(INFO) << "StepMesh benchmark: gpu_num="
            << g_conf.gpu_num << ", role=" << g_conf.role_str;
  if (g_conf.role == Node::SCHEDULER) {
    StartScheduler();
  } else if (g_conf.role == Node::SERVER) {
    StartRegisterServer();
  } else if (g_conf.role == Node::WORKER) {
    StartWorkers();
  }
  return 0;
}
