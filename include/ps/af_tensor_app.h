/**
 *  Copyright (C) 2025 by StepAI Contributors.
 */
#ifndef PS_AF_TENSOR_APP_H_
#define PS_AF_TENSOR_APP_H_

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ps/base.h"
#include "ps/internal/backend.h"
#include "ps/internal/utils.h"
#include "ps/kv_app.h"

namespace ps {

constexpr int kDefaultEventBufferSize = 16;

enum {
  AF_FLAG_BATCH_START = 1,
  AF_FLAG_BATCH_MIDDLE = 2,
  AF_FLAG_BATCH_END = 3,
};

struct KeyTensor {
  Key key = 0;
  at::Tensor val;
};

typedef std::vector<KeyTensor> KeyTensorBatch;

/** \brief AF request */
struct AFTensorRequest {
  KeyTensorBatch push;
  KeyTensorBatch pull;
  std::vector<int> push_timestamps;
  std::vector<int> pull_timestamps;
  TensorEvent* event = nullptr;
};

/**
 * \brief Attention-FFN Disaggregation Worker
 */
class AFTensorWorker {
 public:
  /**
   * \brief constructor for AF worker specific to tensor-based communication
   *
   * \param instance_idx the instance id within a group
   */
  explicit AFTensorWorker(int instance_idx = 0)
      : kv_(0, 0, instance_idx),
        instance_id_(instance_idx),
        pushpull_stop_(false) {
    gpu_ = -1;
    Environment::Get()->find("STEPMESH_GPU", &gpu_, gpu_);
    PS_CHECK_GE(gpu_, 0) << "STEPMESH_GPU is not set";
    Backend::Get()->SetDevice(gpu_);
    for (int i = 0; i < kDefaultEventBufferSize; i++) {
      events_.push_back(new TensorEvent());
    }

    pushpull_thread_ = std::thread([this] { this->PushPullWorker(); });
  }

  virtual ~AFTensorWorker() {
    pushpull_stop_.store(true);
    pushpull_queue_.Push(AFTensorRequest());
    if (pushpull_thread_.joinable()) {
      pushpull_thread_.join();
    }
  }

  /**
   * \brief Performs a batch operation of
   *    pushing and pulling tensors to/from FFN.
   *
   * @param push_tensors A reference to the KeyTensorBatch object
   *    containing the tensors to be pushed and their associated keys.
   * @param pull_tensors A reference to the KeyTensorBatch object
   *    where the pulled tensors and their associated keys will be stored.
   * @return An integer indicating the result of the operation.
   */
  int ZBatchPushPull(KeyTensorBatch& push_tensors,
                     KeyTensorBatch& pull_tensors) {
    Backend::Get()->SetDevice(gpu_);
    auto server_ranges =
        Postoffice::GetWorker(instance_id_)->GetServerKeyRanges();
    int server_count = server_ranges.size();
    PS_CHECK_GT(server_count, 0) << "zero servers and cannot pushpull";

    std::unique_lock<std::mutex> lock(mu_);
    auto req = AFTensorRequest();
    std::vector<int> timestamps;
    bool first = true;
    int start_ts = 0;
    for (size_t i = 0; i < push_tensors.size(); i++) {
      int ts = kv_.AllocTimestamp();
      if (i == 0) {
        first = false;
        start_ts = ts;
      } else {
        timestamps.push_back(ts);
      }
      req.push_timestamps.push_back(ts);
    }

    int pull_batch_size = static_cast<int>(pull_tensors.size() / server_count);
    for (int i = 0; i < pull_batch_size; i++) {
      int ts = kv_.AllocTimestamp();
      if (first && i == 0) {
        start_ts = ts;
      } else {
        timestamps.push_back(ts);
      }
      req.pull_timestamps.push_back(ts);
    }

    req.push = push_tensors;
    req.pull = pull_tensors;
    req.event = GetEvent();
    req.event->Record();

    pushpull_queue_.Push(std::move(req));

    std::unique_lock<std::mutex> timestamp_lock(timestamp_mu_);
    batch_timestamps_[start_ts] = std::move(timestamps);
    return start_ts;
  }

  /**
   * \brief Wait for the operation to complete
   * @param timestamp return by push, pull or push-pull operations
   */
  void Wait(int timestamp) {
    kv_.Wait(timestamp);
    std::unique_lock<std::mutex> lock(timestamp_mu_);
    auto itr = batch_timestamps_.find(timestamp);
    if (itr != batch_timestamps_.end()) {
      for (auto ts : itr->second) {
        kv_.Wait(ts);
      }
      batch_timestamps_.erase(itr);
    }
  }

  /**
   * Get all handlers for batch push-pull operations
   * @param timestamp return by push, pull or pushpull operations
   */
  std::vector<int> GetAllHandlers(int timestamp) {
    std::vector<int> handlers;
    handlers.push_back(timestamp);
    std::unique_lock<std::mutex> lock(timestamp_mu_);
    auto itr = batch_timestamps_.find(timestamp);
    if (itr != batch_timestamps_.end()) {
      for (auto ts : itr->second) {
        handlers.push_back(ts);
      }
    }
    return handlers;
  }

  /**
   * \brief Get performance trace for an operation
   * @param timestamp return by push, pull or pushpull operations
   */
  std::pair<struct Trace, struct Trace> FetchTrace(int timestamp) {
#ifdef STEPMESH_ENABLE_TRACE
    return kv_.FetchTrace(timestamp);
#endif  // STEPMESH_ENABLE_TRACE
    return std::make_pair(Trace(), Trace());
  }

 private:
  TensorEvent* GetEvent() {
    for (auto ev : events_) {
      if (ev->Occupy()) {
        return ev;
      }
    }

    auto* ev = new TensorEvent();
    ev->Occupy();
    events_.push_back(ev);
    return ev;
  }

  void PushPullWorker() {
    BindCpuCore(4, 1);
    Backend::Get()->SetDevice(gpu_);
    while (!pushpull_stop_.load()) {
      AFTensorRequest req;
      pushpull_queue_.WaitAndPop(&req);

      if (pushpull_stop_.load()) {
        break;
      }

      if (req.event != nullptr) {
        req.event->Sync();
        req.event->Release();
        req.event = nullptr;
      }
      ZBatchPushPull_(req.push, req.push_timestamps, req.pull,
                      req.pull_timestamps);
    }
    PS_LOG(INFO) << "Stop PushPullWorker" << gpu_;
  }

  void ZPush_(int ts, const SArray<Key>& keys, const at::Tensor& tensor,
              int cmd = 0) {
    SArray<char> val;
    val.reset(reinterpret_cast<char*>(tensor.data_ptr()),
              tensor.numel() * tensor.itemsize(), [tensor](void*) {});

    Message msg;
    msg.meta.request = true;
    msg.meta.head = cmd;
    msg.meta.push = true;
    msg.meta.timestamp = ts;
    msg.meta.addr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    msg.meta.val_len = tensor.numel() * tensor.itemsize();
    msg.meta.key = keys[0];
    msg.meta.is_tensor = 1;
    msg.meta.dtype = static_cast<int>(tensor.scalar_type());
    msg.meta.shape.clear();
    for (int64_t i = 0; i < tensor.dim(); i++) {
      msg.meta.shape.push_back(tensor.size(i));
    }
    msg.data.clear();
    msg.AddData(keys);
    msg.AddData(val);
    msg.meta.tensor_ev = nullptr;
    auto server_ranges =
        Postoffice::GetWorker(instance_id_)->GetServerKeyRanges();
    int server_count = server_ranges.size();
    // broadcast
    for (int i = 0; i < server_count; i++) {
      kv_.SendMsg(msg, i);
    }
  }

  void ZPull_(int ts, const SArray<Key>& keys, KeyTensorBatch& pull_tensors,
              int index, int cmd = 0) {
    auto server_ranges =
        Postoffice::GetWorker(instance_id_)->GetServerKeyRanges();
    int server_count = server_ranges.size();
    int pull_batch_size = static_cast<int>(pull_tensors.size() / server_count);
    for (int i = 0; i < server_count; i++) {
      Message msg;
      msg.meta.timestamp = ts;
      SArray<char> val;
      SArray<Key> key(1);

      auto tensor = pull_tensors[i * pull_batch_size + index].val;

      *key.data() = pull_tensors[i * pull_batch_size + index].key;

      val.reset(reinterpret_cast<char*>(tensor.data_ptr()),
                tensor.numel() * tensor.itemsize(), [tensor](void*) {});

      msg.meta.request = true;
      msg.meta.head = cmd;
      msg.meta.push = false;
      msg.meta.addr = reinterpret_cast<uint64_t>(tensor.data_ptr());
      msg.meta.val_len = tensor.numel() * tensor.itemsize();
      msg.meta.key = key[0];
      msg.meta.is_tensor = 1;
      msg.meta.dtype = static_cast<int>(tensor.scalar_type());
      msg.meta.shape.clear();
      for (int64_t s = 0; s < tensor.dim(); s++) {
        msg.meta.shape.push_back(tensor.size(i));
      }
      msg.data.clear();
      msg.AddData(key);
      msg.AddData(val);

      kv_.SendMsg(msg, i);

      kv_.AddCallback(msg.meta.timestamp,
                      [this, val, ts{msg.meta.timestamp}]() mutable {
                        this->kv_.EraseRecvKvs(ts);
                      });
    }
  }

  void ZBatchPushPull_(KeyTensorBatch& push_tensors,
                       std::vector<int>& push_timestamps,
                       KeyTensorBatch& pull_tensors,
                       std::vector<int>& pull_timestamps) {
    PS_CHECK_GE(push_tensors.size() + pull_tensors.size(), 1);
    Backend::Get()->SetDevice(gpu_);
    auto server_ranges =
        Postoffice::GetWorker(instance_id_)->GetServerKeyRanges();
    int server_count = server_ranges.size();
    PS_CHECK_GT(server_count, 0) << "zero servers and cannot pushpull";

    if (push_tensors.size() + pull_tensors.size() == 1) {
      SArray<Key> key(1);
      if (push_tensors.size() == 1) {
        *key.data() = push_tensors[0].key;
        ZPush_(push_timestamps[0], key, push_tensors[0].val);
      } else {
        ZPull_(pull_timestamps[0], key, pull_tensors, 0);
      }
      return;
    }

    bool first = true;
    for (size_t i = 0; i < push_tensors.size(); i++) {
      SArray<Key> key(1);
      *key.data() = push_tensors[i].key;

      if (i == 0) {
        ZPush_(push_timestamps[i], key, push_tensors[0].val,
               AF_FLAG_BATCH_START);
        first = false;
      } else if (pull_tensors.empty() && i == push_tensors.size() - 1) {
        ZPush_(push_timestamps[i], key, push_tensors[i].val, AF_FLAG_BATCH_END);
      } else {
        ZPush_(push_timestamps[i], key, push_tensors[i].val,
               AF_FLAG_BATCH_MIDDLE);
      }
    }

    int pull_batch_size = static_cast<int>(pull_tensors.size() / server_count);
    for (int i = 0; i < pull_batch_size; i++) {
      SArray<Key> key(1);
      if (first && i == 0) {
        ZPull_(pull_timestamps[i], key, pull_tensors, i, AF_FLAG_BATCH_START);
      } else if (i == pull_batch_size - 1) {
        ZPull_(pull_timestamps[i], key, pull_tensors, i, AF_FLAG_BATCH_END);
      } else {
        ZPull_(pull_timestamps[i], key, pull_tensors, i, AF_FLAG_BATCH_MIDDLE);
      }
    }
  }

  /** \brief key-value works */
  KVWorker<char> kv_;
  /** \brief API mutex */
  mutable std::mutex mu_;
  /** \brief record timestamps for each batch */
  std::unordered_map<int, std::vector<int>> batch_timestamps_;
  /** \brief mutex for record timestamps */
  std::mutex timestamp_mu_;
  /** \brief tensor events */
  std::vector<TensorEvent*> events_;
  /** \brief gpu id */
  int gpu_;
  /** \brief instance id in one group */
  int instance_id_;
  /** \brief queue for transmitting data from user thread to response thread */
  ThreadsafeQueue<AFTensorRequest> pushpull_queue_;
  /** \brief response thread */
  std::thread pushpull_thread_;
  /** \brief response stop signal */
  std::atomic_bool pushpull_stop_;
};

/** \brief meta information about a kv request */
struct AFTensorMeta {
  /** \brief sender's node id */
  int sender = 0;
  /** \brief sender's node rank */
  int sender_rank = 0;
  /** \brief whether is a single request */
  bool single = false;
  int last_timestamp = 0;
  /** \brief meta information for push operations */
  std::vector<KVMeta> push_metas;
  /** \brief tensors for push operations */
  std::vector<KeyTensor> push_tensors;
  /** \brief meta information for pull operations */
  std::vector<KVMeta> pull_metas;
  /** \brief tensors for pull operations */
  std::vector<KeyTensor> pull_tensors;

  /**
   * \brief Append kv metadata and tensor into af metadata
   * @param kv_meta kv meta received from kv server
   * @param tensor tensor received from server
   */
  void Add(const KVMeta& kv_meta, const KeyTensor& tensor) {
    last_timestamp = kv_meta.timestamp;
    if (kv_meta.push) {
      push_tensors.emplace_back(tensor);
      push_metas.emplace_back(kv_meta);
    } else {
      pull_tensors.emplace_back(tensor);
      pull_metas.emplace_back(kv_meta);
    }
  }
};

/** \brief AF response Buffer */
struct AFTensorResponse {
  /** \brief kv metadata for response */
  KVMeta kv_meta = {};
  /** \brief kv pairs for response */
  KVPairs<char> kv_pair = {};
  /** \brief event to synchronize */
  TensorEvent* event = nullptr;
  uint64_t rsp_start;
};

/**
 * \brief Attention-FFN Disggregation Server
 */
class AFTensorServer {
 public:
  /**
   * \brief Constructor for AF server specific to tensor-based communication
   *
   * @param gpu the local gpu rank
   */
  explicit AFTensorServer(int gpu)
      : kv_(0, false, gpu), gpu_(gpu), response_stop_(false) {
    PS_LOG(INFO) << "AFTensorServer runs on gpu " << gpu;
    Backend::Get()->SetDevice(gpu_);
    for (int i = 0; i < 64; i++) {
      events_.push_back(new TensorEvent());
    }
    kv_.set_request_handle([this](const KVMeta& req_meta,
                                  const KVPairs<char>& req_data,
                                  KVServer<char>* server) {
      this->KVHandler(req_meta, req_data);
    });
    response_thread_ = std::thread([this] { this->ResponseWorker(); });
  }

  virtual ~AFTensorServer() {
    response_stop_.store(true);
    response_queue_.Push(AFTensorResponse());
    if (response_thread_.joinable()) {
      response_thread_.join();
    }
  }

  /**
   * \brief Response to a pushpull operation
   *
   * @param meta handler metatda
   * @param tensors the pull tensors to respond
   * @param stream the gpu stream used for event synchronize
   */
  void Response(const AFTensorMeta& meta, KeyTensorBatch tensors = {},
                bool need_event = true) {
    Backend::Get()->SetDevice(gpu_);
    if (meta.single) {
      if (meta.pull_tensors.size() == 1) {
        KVPairs<char> res;
        SArray<Key> key(1);
        *key.data() = tensors[0].key;
        res.keys = key;

        SArray<char> tensor_val;
        tensor_val.reset(reinterpret_cast<char*>(tensors[0].val.data_ptr()),
                         tensors[0].val.numel() * tensors[0].val.itemsize(),
                         [](void*) {});
        res.vals = tensor_val;

        kv_.Response(meta.pull_metas[0], res, GetEvent());
      } else if (meta.push_metas.size() == 1) {
        kv_.Response(meta.push_metas[0]);
      }
    } else {
      unsigned response_count = 0;
      for (uint32_t i = 0; i < meta.pull_tensors.size(); i++) {
        auto& kv_meta = meta.pull_metas[i];
        bool found = false;
        for (auto& res_kv : tensors) {
          if (res_kv.key == kv_meta.key) {
            response_count++;
            AFTensorResponse rsp = {};
            SArray<Key> key(1);
            *key.data() = res_kv.key;
            rsp.kv_pair.keys = key;

            rsp.kv_pair.vals.reset(
                reinterpret_cast<char*>(res_kv.val.data_ptr()),
                res_kv.val.numel() * res_kv.val.itemsize(), [](void*) {});

            rsp.kv_meta = kv_meta;
            if (need_event) {
              rsp.event = GetEvent();
              rsp.event->Record();
            } else {
              rsp.event = nullptr;
            }
            rsp.rsp_start = GetNanosecond();
            response_queue_.Push(std::move(rsp));
            found = true;
            break;
          }
        }
        if (!found) {
          PS_LOG(ERROR) << "failed to found key " << kv_meta.key;
        }
      }

      if (response_count < tensors.size()) {
        PS_LOG(ERROR) << "too many response keys";
      }
    }
  }

  using AFServerRequestHandle =
      std::function<void(const AFTensorMeta& req_meta, AFTensorServer* server)>;
  /**
   * \brief Set the handle to process AF request
   * @param request_handle user-defined handle
   */
  void SetRequestHandle(const AFServerRequestHandle& request_handle) {
    PS_CHECK(request_handle) << "invalid request handle for AF server";
    request_handle_ = request_handle;
  }

  /**
   * \brief Register a tensor with local rdma devices
   *
   * @param tensor the tensor to register
   * @param worker_ranks the worker ranks to register,
   *    and the tensor will be sliced to register for different ranks
   * @param keys the keys to register
   */
  void RegisterRecvTensor(const at::Tensor& tensor,
                          std::vector<int>& worker_ranks,
                          std::vector<Key>& keys) {
    PS_CHECK_GT(worker_ranks.size(), 0) << "ranks or keys should not be empty";
    PS_CHECK_EQ(worker_ranks.size(), keys.size())
        << "rank list and key list have unequal size";
    char* buffer_ptr = reinterpret_cast<char*>(tensor.data_ptr());
    uint64_t data_size = tensor.numel() * tensor.element_size();
    int chunk_size = data_size / worker_ranks.size();
    PS_CHECK_EQ(data_size % worker_ranks.size(), 0)
        << "tensor data size cannot be evenly chunked to different ranks";
    for (uint32_t i = 0; i < worker_ranks.size(); i++) {
      RegisterRecvBuffer_(worker_ranks[i], keys[i], buffer_ptr + chunk_size * i,
                          chunk_size);
    }
  }

 private:
  TensorEvent* GetEvent() {
    std::unique_lock<std::mutex> lock(events_mu_);
    for (auto ev : events_) {
      if (ev->Occupy()) {
        return ev;
      }
    }

    auto* ev = new TensorEvent();
    ev->Occupy();
    events_.push_back(ev);
    return ev;
  }

  KeyTensor FromBlob(const KVMeta& req_meta, const KVPairs<char>& req_data) {
    KeyTensor key_tensor;
    if (req_meta.push) {
      auto options = torch::TensorOptions()
                         .dtype(at::ScalarType(req_meta.dtype))
                         .memory_format(at::MemoryFormat::Contiguous)
                         .device(Backend::Get()->GetDevice());
      key_tensor.val =
          at::from_blob(req_data.vals.data(), req_meta.shape, options);
    }
    key_tensor.key = req_data.keys[0];
    return key_tensor;
  }

  void ResponseWorker() {
    BindCpuCore(3, 1);
    Backend::Get()->SetDevice(gpu_);
    PS_LOG(INFO) << "Start ResponseWorker " << gpu_;
    while (!response_stop_.load()) {
      AFTensorResponse rsp;
      rsp.event = nullptr;
      response_queue_.WaitAndPop(&rsp);

      if (response_stop_.load()) {
        break;
      }
      if (rsp.event != nullptr) {
        rsp.event->Sync();
        rsp.event->Release();
        rsp.event = nullptr;
      }
      kv_.Response(rsp.kv_meta, rsp.kv_pair, nullptr);
    }
    PS_LOG(INFO) << "Stop ResponseWorker";
  }

  void KVHandler(const KVMeta& req_meta, const KVPairs<char>& req_data) {
    Backend::Get()->SetDevice(gpu_);
    if (req_meta.cmd == 0) {
      struct AFTensorMeta af_meta;
      af_meta.sender = req_meta.sender;
      af_meta.Add(req_meta, FromBlob(req_meta, req_data));
      af_meta.single = true;
      af_meta.sender_rank =
          Postoffice::GetServer(gpu_)->IDtoRank(af_meta.sender);
      request_handle_(af_meta, this);
    } else {
      AFTensorMeta* af_meta = nullptr;
      bool is_reorder = false;
      if (req_meta.push) {
        kv_.Response(req_meta);
      }

      if (req_meta.cmd == AF_FLAG_BATCH_START) {
        af_meta = new AFTensorMeta;
        af_meta->sender = req_meta.sender;
        af_meta->sender_rank =
            Postoffice::GetServer(gpu_)->IDtoRank(af_meta->sender);
        af_meta->single = false;
        if (batch_data_.find(req_meta.sender) != batch_data_.end() &&
            batch_data_[req_meta.sender] != nullptr) {
          reorder_buffer_.push_back(batch_data_[req_meta.sender]);
        }

        batch_data_[req_meta.sender] = af_meta;
      } else {
        for (size_t i = 0; i < reorder_buffer_.size(); i++) {
          if (reorder_buffer_[i]->sender == req_meta.sender) {
            if (req_meta.timestamp == reorder_buffer_[i]->last_timestamp + 1) {
              af_meta = reorder_buffer_[i];
              is_reorder = true;
              break;
            }
          }
        }

        if (af_meta == nullptr) {
          af_meta = batch_data_[req_meta.sender];
        }
      }

      af_meta->Add(req_meta, FromBlob(req_meta, req_data));

      if (req_meta.cmd == AF_FLAG_BATCH_END) {
        request_handle_(*af_meta, this);
        if (!is_reorder) {
          batch_data_[req_meta.sender] = nullptr;
        } else {
          for (size_t i = 0; i < reorder_buffer_.size(); i++) {
            if (reorder_buffer_[i] == af_meta) {
              reorder_buffer_.erase(reorder_buffer_.begin() + i);
              break;
            }
          }
        }
        delete af_meta;
      }
    }
  }

  void RegisterRecvBuffer_(int worker_rank, Key k, char* data, int data_len) {
    SArray<Key> key(1);
    *key.data() = k;

    SArray<char> tensor_val;
    tensor_val.reset(reinterpret_cast<char*>(data), data_len, [](void*) {});

    SArray<int> len(1);
    *len.data() = data_len;
    kv_.RegisterRecvBufferWithRank(worker_rank, key, tensor_val, len);
  }

  /** \brief kv server used for process data */
  KVServer<char> kv_;
  /** \brief batch data */
  std::unordered_map<int, AFTensorMeta*> batch_data_;
  /** \brief data handle for af server */
  AFServerRequestHandle request_handle_;
  /** \brief gpu device index */
  int gpu_;
  /** \brief tensor event mutex */
  std::mutex events_mu_;
  /** \brief tensor event vector */
  std::vector<TensorEvent*> events_;
  /** \brief queue for transmitting data from user thread to response thread */
  ThreadsafeQueue<AFTensorResponse> response_queue_;
  std::vector<AFTensorMeta*> reorder_buffer_;
  /** \brief response thread */
  std::thread response_thread_;
  /** \brief response stop signal */
  std::atomic_bool response_stop_;
};

}  // namespace ps

#endif  // PS_AF_TENSOR_APP_H_
