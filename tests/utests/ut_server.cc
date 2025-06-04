/**
*  Copyright (c) 2015 by Step AI
*/
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdlib>

#include <unistd.h>

#include "ut_common.h"

std::vector<at::Tensor> g_tensors;

void SumHandler(const AFTensorMeta& req_meta, AFTensorServer* server) {
  auto sum = at::zeros_like(req_meta.push_tensors[0].val, req_meta.push_tensors[0].val.scalar_type());
  for (auto t : req_meta.push_tensors) {
    sum += t.val;
  }
  g_tensors.push_back(sum);
  server->Response(req_meta, {{ req_meta.pull_tensors[0].key, sum }});
}

void StartFFNServer() {
  // GPU Alloc, malloc should automatically gives page aligned.
  PS_LOG(INFO) << "run server over gpu" << g_conf.gpu;
  StartPS(0, Node::SERVER, -1, true);
  AFTensorServer* server = new AFTensorServer(g_conf.gpu);
  server->SetRequestHandle(SumHandler);
  ps::Postoffice::GetServer()->Barrier(0, ps::kServerGroup + ps::kWorkerGroup);
  RegisterExitCallback([server]() { delete server; });
  Finalize(0, Node::SERVER, true);
  PS_LOG(INFO) << "FFN server ends";
}

int main(int argc, char *argv[]) {
  InitUtestConfig();
  PS_LOG(INFO) << "AF-Communication utest server: gpu="
            << g_conf.gpu;
  StartFFNServer();
  return 0;
}
