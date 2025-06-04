/**
*  Copyright (c) 2015 by Step AI
*/
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdlib>

#include <unistd.h>

#include "test_common.h"

#ifdef STEPAF_USE_TORCH
std::unordered_map<uint64_t, at::Tensor> g_mem;

std::vector<at::Tensor> g_af_push_tensors;
std::vector<SArray<Key>> g_af_push_keys;

std::vector<at::Tensor> g_af_pull_tensors;
std::vector<SArray<Key>>  g_af_pull_keys;

void SimulatedHandler(const AFTensorMeta& req_meta, AFTensorServer* server) {
  if (req_meta.single) {
    if (req_meta.push_tensors.size() == 1) {
      auto& req_data = req_meta.push_tensors[0];
      auto key = req_data.key;

      if (g_mem.find(key) == g_mem.end()) {
        g_mem[key] = req_data.val;
      }

      server->Response(req_meta);
    } else {
      auto key = req_meta.pull_tensors[0].key;
      auto iter = g_mem.find(key);
      KeyTensor key_tensor;
      key_tensor.key = key;
      key_tensor.val =  iter->second;
      server->Response(req_meta, { key_tensor });
    }
  } else {
    auto key = req_meta.pull_tensors[0].key;
    auto iter = g_mem.find(key);
    KeyTensor key_tensor;
    key_tensor.key = key;
    key_tensor.val =  iter->second;
    server->Response(req_meta, { key_tensor });
  }
}

void StartServer() {
  AFTensorServer* server = new AFTensorServer(0, g_conf.gpu, false, g_conf.gpu);
  server->set_request_handle(SimulatedHandler);
  RegisterExitCallback([server]() { delete server; });
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

inline void InitOneKeyThenPush(ps::Key ps_key,
                               std::vector<SArray<Key>> &server_keys,
                               at::Tensor &vals,
                               AFTensorWorker* kv) {
  void* ptr_key;
  AlignedMalloc(&ptr_key, sizeof(Key));
  SArray<Key> keys;
  keys.reset(reinterpret_cast<Key*>(ptr_key), 1, [](void *){});
  memcpy(ptr_key, &ps_key, sizeof(Key));
  server_keys.push_back(keys);

  kv->Wait(kv->ZPush(keys, vals));
}

void InitWorker(AFTensorWorker* kv) {
  auto af_push_len = g_conf.batch_start * g_conf.base_size;
  auto af_pull_len = g_conf.batch_start * g_conf.base_size * 2;
  PS_LOG(INFO) << "init work push buffer: microbatch="
            << g_conf.mb_num << ", val_size=" << af_push_len;
  InitTensors(g_af_push_tensors, g_conf.mb_num,
           {g_conf.batch_start, g_conf.base_size},
           at::kByte, g_conf.gpu);

  PS_LOG(INFO) << "init work pull buffer: microbatch="
            << g_conf.mb_num << ", val_size=" << af_pull_len;
  InitTensors(g_af_pull_tensors, g_conf.mb_num,
              {g_conf.batch_start, g_conf.base_size},
              at::kBFloat16, g_conf.gpu);

  auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
  for (int mb = 0; mb < g_conf.mb_num; mb++) {
    ps::Key ps_key = GetKeyIndex(g_conf.worker_rank, mb, true);
    InitOneKeyThenPush(ps_key, g_af_push_keys, g_af_push_tensors[mb], kv);

    ps_key = GetKeyIndex(g_conf.worker_rank, mb, false);
    InitOneKeyThenPush(ps_key, g_af_pull_keys, g_af_pull_tensors[mb], kv);
  }
  Postoffice::GetWorker(g_conf.gpu)->Barrier(0, ps::kWorkerGroup);
  PS_LOG(INFO) << "finish worker init.";
}

void RunWorker(AFTensorWorker* kv, int tid) {
  std::vector<int64_t> batch_push_netcosts;
  std::vector<int64_t> batch_pull_netcosts;

  auto af_push = [kv, &batch_push_netcosts] (int mb) {
    auto start = std::chrono::high_resolution_clock::now();
    auto keys = g_af_push_keys[mb];
    auto vals = g_af_push_tensors[mb];
    auto ts = kv->ZPush(keys, vals);
    kv->Wait(ts);
    auto end = std::chrono::high_resolution_clock::now();

#ifdef STEPAF_ENABLE_TRACE
    auto p = kv->FetchTrace(ts);
    int64_t net_cost = (p.second.postrecv - p.first.postsend -
                        (p.second.postsend -  p.first.postrecv));
    batch_push_netcosts.push_back(net_cost);
#endif

    return (end - start).count();
  };

  auto af_pull = [kv, &batch_pull_netcosts] (int mb) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> timestamps;

    auto keys = g_af_pull_keys[mb];
    auto vals = g_af_pull_tensors[mb];
    auto ts = kv->ZPull(keys, vals);
    kv->Wait(ts);
    auto end = std::chrono::high_resolution_clock::now();

#ifdef STEPAF_ENABLE_TRACE
    auto p = kv->FetchTrace(ts);
    batch_pull_netcosts.push_back(
        p.second.postrecv - p.first.postsend -
        (p.second.postsend -  p.first.postrecv));
#endif

    return (end - start).count();
  };

  auto af_pushpull = [kv] (int mb) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> timestamps;
    auto push_batch = KeyTensorBatch();
    push_batch.push_back(
        KeyTensor{
            g_af_push_keys[mb][0],
            g_af_push_tensors[mb],
        });
    auto pull_batch = KeyTensorBatch();
    pull_batch.push_back(KeyTensor{
        g_af_pull_keys[mb][0],
        g_af_pull_tensors[mb],
    });

    kv->Wait(kv->ZBatchPushPull(push_batch, pull_batch));
    auto end = std::chrono::high_resolution_clock::now();
    return (end - start).count();
  };

  PS_LOG(INFO) << "warmup starts";
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0 ; iter < g_conf.warmup_iter; iter++) {
    for (int mb = 0; mb < g_conf.mb_num; mb++) {
      af_push(mb);
      af_pull(mb);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  PS_LOG(INFO) << "warmup finishes: count=" << g_conf.warmup_iter
            << ", cost=" << (end - start).count() / 1000 << "us";

  std::vector<int64_t> overall_push_timestamps;
  overall_push_timestamps.reserve(g_conf.iter);
  std::vector<int64_t> overall_pull_timestamps;
  overall_pull_timestamps.reserve(g_conf.iter);
  std::vector<int64_t> overall_pushpull_timestamps;
  overall_pull_timestamps.reserve(g_conf.iter);

  std::vector<int64_t> batch_push_timestamps;
  batch_push_timestamps.reserve(100);
  std::vector<int64_t> batch_pull_timestamps;
  batch_pull_timestamps.reserve(100);
  std::vector<int64_t> batch_pushpull_timestamps;
  batch_pull_timestamps.reserve(100);

  for (int iter = 0; iter < g_conf.iter; iter++) {
    for (int mb = 0; mb < g_conf.mb_num; mb++) {
      auto push_ts = af_push(mb);
      auto pull_ts = af_pull(mb);
      auto pushpull_ts = af_pushpull(mb);

      overall_push_timestamps.emplace_back(push_ts);
      batch_push_timestamps.emplace_back(push_ts);

      overall_pull_timestamps.emplace_back(pull_ts);
      batch_pull_timestamps.emplace_back(pull_ts);

      overall_pushpull_timestamps.emplace_back(pushpull_ts);
      batch_pushpull_timestamps.emplace_back(pushpull_ts);
    }
    
    if ((iter % 1000 == 999)) {
#ifdef STEPAF_ENABLE_TRACE
      DumpLatency("push net cost: ", batch_push_netcosts);
      DumpLatency("pull net cost: ", batch_pull_netcosts);
#endif
      DumpLatency("push batch latency: ", batch_push_timestamps);
      DumpLatency("pull batch latency: ", batch_pull_timestamps);
      DumpLatency("pushpull batch latency: ", batch_pushpull_timestamps);
      batch_pull_timestamps.clear();
      batch_push_timestamps.clear();
      batch_pushpull_timestamps.clear();
#ifdef STEPAF_ENABLE_TRACE
      batch_push_netcosts.clear();
      batch_pull_netcosts.clear();
#endif
    }
  }

  DumpLatency("push overall latency: ", overall_push_timestamps);
  DumpLatency("pull overall latency: ", overall_pull_timestamps);
  DumpLatency("pushpull overall latency: ", overall_pushpull_timestamps);
}

void StartFFNServer() {
#ifdef DMLC_USE_CUDA
  CUDA_CALL(cudaSetDevice(g_conf.gpu));
  PS_LOG(INFO) << "run server: gpu=" << g_conf.gpu
               << ", node rank=" << g_conf.node_rank
               << ", group size=" << g_conf.group_size;
#endif
  PS_LOG(INFO) << "FFN server Starts";
  StartPS(0, Node::SERVER,
          g_conf.node_rank * g_conf.group_size + g_conf.gpu, true);
  StartServer();
  Finalize(0, Node::SERVER, true);
  PS_LOG(INFO) << "FFN server ends";
}

void StartWorkers() {
#ifdef DMLC_USE_CUDA
  // GPU Alloc, malloc should automatically gives page aligned.
  CUDA_CALL(cudaSetDevice(g_conf.gpu));
  PS_LOG(INFO) << "run worker: gpu=" << g_conf.gpu
               << ", node rank=" << g_conf.node_rank
               << ", group size=" << g_conf.group_size;
#endif
  StartPS(0, Node::WORKER,
          g_conf.node_rank * g_conf.group_size + g_conf.gpu, true);
  AFTensorWorker af_worker(0, 0, g_conf.gpu);
  CHECK(g_conf.worker_rank >= 0) << "please set attention worker rank";
  InitWorker(&af_worker);

  RunWorker(&af_worker, 0);

  Finalize(0, Node::WORKER, true);
  PS_LOG(INFO) << "Simulated attention worker is DONE";
}

int main(int argc, char *argv[]) {
  InitConfig();
  PS_LOG(INFO) << "AF-Communication benchmark: worker_num="
            << g_conf.worker_num << ", gpu_num="
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
#else

int main(int argc, char *argv[]) {
  PS_CHECK(0) << argv[0] << " is compiled without STEPAF_USE_TORCH";
  return 0;
}

#endif  // STEPAF_USE_TORCH