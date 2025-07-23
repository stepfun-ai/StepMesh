/**
*  Copyright (c) 2015 by Step AI
*/
#include <chrono>
#include <thread>

#include <unistd.h>

#include "ut_common.h"

int ServerCount = ps::GetEnv("DMLC_NUM_SERVER", 1);

void StartWorkers() {
  PS_LOG(INFO) << "run worker over gpu" << g_conf.gpu;
  StartPS(0, Node::WORKER, g_conf.gpu, true);
  Backend::Get()->SetDevice(g_conf.gpu);
  AFTensorWorker af_worker = AFTensorWorker(g_conf.gpu);
  ps::Postoffice::GetWorker(g_conf.gpu)->Barrier(
      0, ps::kServerGroup + ps::kWorkerGroup);

  std::vector<at::Tensor> tensors;
  for (int b = 1; b < g_worker_conf.batch_max; b++) {
    // g_worker_conf.batch_max; b++) {
    auto start = std::chrono::high_resolution_clock::now();
    auto push_batch = KeyTensorBatch();
    int failed_count = 0;
    for (int t = 0; t < g_worker_conf.tensor_num; t++) {
      auto push = CreateTensor({b, 7168}, at::kBFloat16, g_conf.gpu, true);
      push_batch.push_back(KeyTensor{
          uint64_t((b << 16) + t),
          push,
      });
      tensors.push_back(push);
    }
    auto pull_batch = KeyTensorBatch();
    for (int i = 0 ;i < ServerCount ; i ++) {
      pull_batch.push_back(KeyTensor{
          uint64_t((b << 16) + g_worker_conf.tensor_num),
          CreateTensor({b, 7168}, at::kBFloat16, g_conf.gpu),
      });
    }
    for (int i = 0; i < 32; i++) {
      for (int j = 0; j < ServerCount; j++) {
        pull_batch[j].val.zero_();
      }

      af_worker.Wait(af_worker.ZBatchPushPull(push_batch, pull_batch));
      auto sum = CreateTensor({b, 7168}, at::kBFloat16, g_conf.gpu);
      for (auto t : push_batch) {
        sum += t.val;
      }


      bool fail_flag = false;
      for (int j = 0; j < ServerCount; j++) {
        if (!sum.allclose(pull_batch[j].val)) {
          LOG(WARNING) << "check failed: batch=" << b << " , iter=" << i;
          fail_flag = true;
          break;
        }
      }
      if (fail_flag) failed_count += 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    if (failed_count == 0) {
      std::cout << "GPU " << g_conf.gpu << " Batch " << b << ": ALL PASS"
                << " duration=" << (end - start).count() << "ns" << std::endl;
    } else {
      std::cout << "GPU " << g_conf.gpu << " Batch " << b << " FAILED "
                << failed_count << "/" << 32
                << " duration=" << (end - start).count() << "ns" << std::endl;
    }
  }

  // stop worker
  Finalize(0, Node::WORKER, true);
  PS_LOG(INFO) << "Simulated attention worker is DONE";
}

int main(int argc, char *argv[]) {
  InitUtestConfig();
  StartWorkers();
  return 0;
}
