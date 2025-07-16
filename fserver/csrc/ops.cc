/* Copyright (c) 2025, StepFun Authors. All rights reserved. */
#include <atomic>

#include <execinfo.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "util.h"
using namespace ps;

ps::AFTensorServer* fserver_;
ps::AFTensorWorker* fworker_;
Node::Role role_;
int gpu_ = 0;
int node_rank_ = 0;
int group_size_ = 1;
int instance_id_ = 0;

typedef std::tuple<uint64_t, std::vector<torch::Tensor>, std::vector<uint64_t>> ServerDataBatch;

std::mutex mu_;
uint64_t handler_counter_ = 0;
std::unordered_map<uint64_t, AFTensorMeta> meta_map_;
std::vector<ServerDataBatch> q_;
std::atomic<bool> q_signal_;

void SumHandler(const AFTensorMeta& req_meta, AFTensorServer* server) {
  std::vector<torch::Tensor> tensors;
  std::vector<uint64_t> keys;
  for (auto& t : req_meta.push_tensors) {
    tensors.emplace_back(std::move(t.val));
    keys.emplace_back(t.key);
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    meta_map_[handler_counter_] = req_meta;
    q_.emplace_back(handler_counter_, std::move(tensors), keys);
    q_signal_.store(true);
  }
  ++handler_counter_;
}

std::vector<ServerDataBatch> get_batch() {
  while (!q_signal_.load()) {
    sched_yield();
  }
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<ServerDataBatch> res = std::move(q_);
  q_.clear();
  q_signal_.store(false);
  return res;
}

void respond(std::vector<torch::Tensor>& tensors, uint64_t handler) {
  AFTensorMeta reqmeta;
  {
    std::lock_guard<std::mutex> lock(mu_);
    PS_CHECK_NE(meta_map_.find(handler), meta_map_.end());
    reqmeta = meta_map_[handler];
  }
  PS_CHECK_EQ(tensors.size(), reqmeta.pull_tensors.size());
  std::vector<KeyTensor> result;
  for (size_t i = 0; i < tensors.size(); ++i) {
    result.push_back({reqmeta.pull_tensors[i].key, std::move(tensors[i])});
  }
  fserver_->Response(reqmeta, result);
}

int push_pull(std::vector<torch::Tensor>& push_tensors,
              std::vector<uint64_t>& push_keys,
              std::vector<torch::Tensor>& pull_tensors,
              std::vector<uint64_t>& pull_keys) {
  auto push_batch = KeyTensorBatch(push_tensors.size());
  auto pull_batch = KeyTensorBatch(pull_tensors.size());
  for (size_t i = 0; i < push_tensors.size(); i++) {
    push_batch[i] = KeyTensor{uint64_t(push_keys[i]), std::move(push_tensors[i])};
  }
  for (size_t i = 0; i < pull_tensors.size(); i++) {
    pull_batch[i] = KeyTensor{uint64_t(pull_keys[i]), std::move(pull_tensors[i])};
  }
  return fworker_->ZBatchPushPull(push_batch, pull_batch);
}

void wait(int handler) { 
  fworker_->Wait(handler);
}

void barrier(bool include_server, bool include_worker) {
  int node_group = 0;
  if (include_server) {
    node_group += ps::kServerGroup;
  }
  if (include_worker) {
    node_group += ps::kWorkerGroup;
  }

  if (role_ == Node::WORKER && include_worker) {
    ps::Postoffice::GetWorker(instance_id_)->Barrier(0, node_group);
  } else if (role_ == Node::SERVER && include_server) {
    ps::Postoffice::GetServer(instance_id_)->Barrier(0, node_group);
  } else {
    ps::Postoffice::Get()->Barrier(0, node_group);
  }
}

void init() {
  q_signal_.store(false);
  std::string role_str = GetEnv("DMLC_ROLE", "server");
  role_ = GetRole(role_str);

  ps::Environment::Get()->find("STEPPS_GPU", &gpu_, gpu_);
  ps::Environment::Get()->find("DMLC_GROUP_SIZE", &group_size_, group_size_);
  ps::Environment::Get()->find("DMLC_NODE_RANK", &node_rank_, node_rank_);
  // 默认的Instance ID与GPU相同
  ps::Environment::Get()->find("DMLC_INSTANCE_ID", &instance_id_, gpu_);

  CUDA_CALL(cudaSetDevice(gpu_));
  std::cout << "Rank ID " << group_size_ * node_rank_ + gpu_ 
    << " Instance_ID " << instance_id_ <<  std::endl;
  ps::StartPS(0, role_,  group_size_ * node_rank_ + gpu_, true);
  if (role_ == Node::WORKER) {
    fworker_ = new AFTensorWorker(instance_id_);
    barrier(true,true);
  } else if (role_ == Node::SERVER) {
    fserver_ = new AFTensorServer(instance_id_);
    // 下面这个能跑通
    // fserver_ = new AFTensorServer(0, -1, false, instance_id_);
    fserver_->SetRequestHandle(SumHandler);
    RegisterExitCallback([fserver_]() { delete fserver_; });
    barrier(true,true);
  }
}

void register_recv_buffer(torch::Tensor& tensor, std::vector<int> worker_ranks, std::vector<uint64_t> push_keys) {
  fserver_->RegisterRecvTensor(tensor, worker_ranks, push_keys);
}

void stop() {
  // LOG(INFO) << (role_ == Node::WORKER ? "worker" : (role_ == Node::SCHEDULER ? "scheduler" : "server"))
            // << " stopping...";
  if (role_ == Node::WORKER) {
    ps::Postoffice::GetWorker(gpu_)->Barrier(0,
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  } else if (role_ == Node::SERVER) {
    ps::Postoffice::GetServer(gpu_)->Barrier(0,
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  } else {
    ps::Postoffice::Get()->Barrier(0,
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }

  ps::Finalize(0, role_, true);
}

std::vector<int> get_all_handlers(int handler) {
    return fworker_->GetAllHandlers(handler);
}

std::vector<uint64_t> fetch_trace(int handler) {
  auto p = fworker_->FetchTrace(handler);
  std::vector<uint64_t> vec;
  vec.push_back(p.first.start);
  vec.push_back(p.first.postsend);
  vec.push_back(p.first.postrecv);
  vec.push_back(p.first.process);
  vec.push_back(p.second.start);
  vec.push_back(p.second.postsend);
  vec.push_back(p.second.postrecv);
  vec.push_back(p.second.process);
  return vec;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, py::call_guard<py::gil_scoped_release>());

  m.def("register_recv_buffer", &register_recv_buffer, py::call_guard<py::gil_scoped_release>());
  m.def("stop", &stop, py::call_guard<py::gil_scoped_release>());

  m.def("wait", &wait, py::call_guard<py::none>());
  m.def("push_pull", &push_pull, py::call_guard<py::none>());
  m.def("respond", &respond,
        py::arg("tensors"),
        py::arg("handler"),
        py::call_guard<py::none>());

  // get batch和fetch trace需要加gil scoped release
  m.def("get_batch", &get_batch, py::call_guard<py::gil_scoped_release>());
  m.def("fetch_trace", &fetch_trace, py::call_guard<py::gil_scoped_release>());
  // functions for communication performance tracing
  
  m.def("get_all_handlers", &get_all_handlers, py::call_guard<py::none>());
  m.def("barrier", &barrier, py::call_guard<py::none>());
}
