
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdlib>

#include <unistd.h>

#include "test_common.h"

std::unordered_map<uint64_t, KVPairs<char> > g_mem;

std::vector<SArray<char>> g_af_push_vals;
std::vector<SArray<Key>>  g_af_push_keys;
std::vector<SArray<int>>  g_af_push_lens;

std::vector<SArray<char>> g_af_pull_vals;
std::vector<SArray<Key>>  g_af_pull_keys;
std::vector<SArray<int>>  g_af_pull_lens;

template <typename Val>
void SimulatedHandler(const AFMeta<Val>& req_meta, AFServer<Val>* server) {
  if (req_meta.single) {
    if (req_meta.push_data_batch.size() == 1) {
      auto& req_data = req_meta.push_data_batch[0];
      auto key = req_data.keys[0];

      PS_CHECK(req_data.lens.size());
      PS_CHECK_EQ(req_data.vals.size(), static_cast<size_t>(req_data.lens[0]))
          << "key=" << key << ", " << req_data.vals.size() << ", "
          << req_data.lens[0];
      if (g_mem.find(key) == g_mem.end()) {
        size_t len = req_data.vals.size();

        void* ptr_val;
        AlignedMalloc(&ptr_val, len);
        g_mem[key].vals.reset(reinterpret_cast<char*>(ptr_val), len,
                              [](void*) {});

        void* ptr_key;
        AlignedMalloc(&ptr_key, sizeof(Key));
        g_mem[key].keys.reset(reinterpret_cast<Key*>(ptr_key), 1, [](void*) {});
        memcpy(ptr_key, &key, sizeof(Key));

        void* ptr_len;
        AlignedMalloc(&ptr_len, sizeof(int));
        g_mem[key].lens.reset(reinterpret_cast<int*>(ptr_len), 1, [](void*) {});
        memcpy(ptr_len, &len, sizeof(int));
      }

      server->Response(req_meta, std::vector<KVPairs<Val>>());
    } else {
      auto key = req_meta.pull_data_batch[0].keys[0];
      auto& meta = req_meta.pull_meta_batch[0];
      auto iter = g_mem.find(key);
      PS_CHECK_NE(meta.val_len, 0);
      server->Response(req_meta, {iter->second});
    }
  } else {
    auto key = req_meta.pull_data_batch[0].keys[0];
    auto& meta = req_meta.pull_meta_batch[0];
    auto iter = g_mem.find(key);
    PS_CHECK_NE(meta.val_len, 0);
    server->Response(req_meta, {iter->second});
  }
}

void StartServer() {
  AFServer<char>* server = new AFServer<char>(0);
  server->set_request_handle(SimulatedHandler<char>);
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
                               std::vector<SArray<int>> &server_lens,
                               SArray<char> &vals,
                               int len,
                               AFWorker<char>* kv) {
  void* ptr_key;
  AlignedMalloc(&ptr_key, sizeof(Key));
  SArray<Key> keys;
  keys.reset(reinterpret_cast<Key*>(ptr_key), 1, [](void *){});
  memcpy(ptr_key, &ps_key, sizeof(Key));
  server_keys.push_back(keys);

  void* ptr_len;
  AlignedMalloc(&ptr_len, sizeof(int));
  SArray<int> lens;
  lens.reset(reinterpret_cast<int*>(ptr_len), 1, [](void *){});
  memcpy(ptr_len, &len, sizeof(len));
  server_lens.push_back(lens);

  kv->Wait(kv->ZPush(keys, vals, lens));
}

void InitWorker(AFWorker<char>* kv) {
  auto af_push_len = g_conf.batch_start * g_conf.base_size;
  auto af_pull_len = g_conf.batch_start * g_conf.base_size * 2;
  PS_LOG(INFO) << "init work push buffer: microbatch="
            << g_conf.mb_num << ", val_size=" << af_push_len;
  InitVals(g_af_push_vals, g_conf.mb_num, af_push_len);

  PS_LOG(INFO) << "init work pull buffer: microbatch="
            << g_conf.mb_num << ", val_size=" << af_pull_len;
  InitVals(g_af_pull_vals, g_conf.mb_num, af_pull_len);

  auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
  for (int mb = 0; mb < g_conf.mb_num; mb++) {
    for (int server = 0; server < krs.size(); server++) {
      ps::Key ps_key = GetKeyIndex(g_conf.worker_rank, mb, true);
      InitOneKeyThenPush(ps_key, g_af_push_keys, g_af_push_lens, g_af_push_vals[mb], af_push_len, kv);

      ps_key = GetKeyIndex(g_conf.worker_rank, mb, false);
      InitOneKeyThenPush(ps_key, g_af_pull_keys, g_af_pull_lens, g_af_pull_vals[mb], af_pull_len, kv);
    }
  }
  Postoffice::GetWorker()->Barrier(0, ps::kWorkerGroup);
  PS_LOG(INFO) << "finish worker init.";
}

void RunWorker(AFWorker<char>* kv, int tid) {
  auto krs = ps::Postoffice::Get()->GetServerKeyRanges();

  const int num_servers = krs.size();
  PS_CHECK_GT(num_servers, 0);

  std::vector<int64_t> batch_push_netcosts;
  std::vector<int64_t> batch_pull_netcosts;

  auto af_push = [kv, &batch_push_netcosts] (int mb) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> timestamps;
    auto lens = g_af_push_lens[mb];
    auto keys = g_af_push_keys[mb];
    auto vals = g_af_push_vals[mb];
    auto ts = kv->ZPush(keys, vals, lens);
    kv->Wait(ts);
    auto end = std::chrono::high_resolution_clock::now();

#ifdef STEPAF_ENABLE_TRACE
    auto p = kv->FetchTrace(ts);
    int64_t net_cost = (p.second.postrecv - p.first.postsend - (p.second.postsend -  p.first.postrecv));
    batch_push_netcosts.push_back(net_cost);
    /*
auto end = GetNanosecond();
PS_LOG(INFO) << (recv.meta.push ? "Push" : "Pull") << " Request Tracing: "
      << ", req_before_send: " << recv.meta.request_trace.postsend - recv.meta.request_trace.start
      << ", req_send_recv: " << recv.meta.request_trace.postrecv - recv.meta.request_trace.postsend
      << ", req_after_recv: " << recv.meta.request_trace.process - recv.meta.request_trace.postrecv
      << ", req_process: " << recv.meta.response_trace.start - recv.meta.request_trace.process
      << ", rsp_before_send: " << recv.meta.response_trace.postsend - recv.meta.response_trace.start
      << ", rsp_send_recv: " << recv.meta.response_trace.postrecv - recv.meta.response_trace.postsend
      << ", rsp_after_recv: " << recv.meta.response_trace.process - recv.meta.response_trace.postrecv
      << ", rsp_process: " << end - recv.meta.response_trace.process
      << ", server_recv_to_send " <<  recv.meta.response_trace.postsend -  recv.meta.request_trace.postrecv
      << ", worker_send_to_recv " <<  recv.meta.response_trace.postrecv -  recv.meta.request_trace.postsend
      << ", e2e_net " <<  recv.meta.response_trace.postrecv -  recv.meta.request_trace.postsend - (recv.meta.response_trace.postsend -  recv.meta.request_trace.postrecv)
      << ", e2e: " << end - recv.meta.request_trace.start;
*/
#endif

    return (end - start).count();
  };

  auto af_pull = [kv, &batch_pull_netcosts] (int mb) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> timestamps;

    auto lens = g_af_pull_lens[mb];
    auto keys = g_af_pull_keys[mb];
    auto vals = g_af_pull_vals[mb];
    auto ts = kv->ZPull(keys, &vals, &lens);
    kv->Wait(ts);
    auto end = std::chrono::high_resolution_clock::now();

#ifdef STEPAF_ENABLE_TRACE
    auto p = kv->FetchTrace(ts);
    batch_pull_netcosts.push_back(p.second.postrecv - p.first.postsend - (p.second.postsend -  p.first.postrecv));
#endif

    return (end - start).count();
  };

  auto af_pushpull = [num_servers, kv] (int mb) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> timestamps;
    auto push_batch = AFWorker<char>::KVPairBatch();
    push_batch.push_back(
        AFWorker<char>::KVPair{
            g_af_push_keys[mb],
            &g_af_push_vals[mb],
            &g_af_push_lens[mb]
        });
    auto pull_batch = AFWorker<char>::KVPairBatch();
    pull_batch.push_back(AFWorker<char>::KVPair{
        g_af_pull_keys[mb],
        &g_af_pull_vals[mb],
        &g_af_pull_lens[mb]
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
  // GPU Alloc, malloc should automatically gives page aligned.
  CUDA_CALL(cudaSetDevice(g_conf.gpu));
  PS_LOG(INFO) << "run server over gpu" << g_conf.gpu;
#endif
  PS_LOG(INFO) << "FFN server Starts";
  StartPS(0, Node::SERVER, -1, true);
  StartServer();
  Finalize(0, Node::SERVER, true);
  PS_LOG(INFO) << "FFN server ends";
}

void StartWorkers() {
#ifdef DMLC_USE_CUDA
  // GPU Alloc, malloc should automatically gives page aligned.
  CUDA_CALL(cudaSetDevice(g_conf.gpu));
  PS_LOG(INFO) << "run worker over gpu" << g_conf.gpu;
#endif
  StartPS(0, Node::WORKER, -1, true);
  AFWorker<char> af_worker(0, 0);

  auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
  const int num_servers = krs.size();
  PS_LOG(INFO) << krs.size() << " ffn servers";
  PS_CHECK_GT(num_servers, 0);
  PS_CHECK(g_conf.worker_rank >= 0) << "please set attention worker rank";
  InitWorker(&af_worker);

  std::vector<std::thread> threads;
  for (int i = 0; i < g_conf.gpu_num; ++i) {
    threads.emplace_back(RunWorker, &af_worker, threads.size());
  }

  // wait for workers
  for (auto& t : threads) {
    t.join();
    PS_LOG(INFO) << "worker thread " << t.get_id() << " ends";
  }

  // stop system
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
