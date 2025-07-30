// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
// Modifications Copyright (C) by StepAI Contributors. 2025.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef RDMA_VAN_H_
#define RDMA_VAN_H_

#ifdef DMLC_USE_RDMA

#include <cstdio>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./rdma_transport.h"
#include "./rdma_utils.h"
#include "dmlc/logging.h"

namespace ps {

class RDMAVan : public Van {
 public:
  explicit RDMAVan(Postoffice *postoffice)
      : Van(postoffice), postoffice_(postoffice) {
    PS_CHECK_EQ(ibv_fork_init(), 0) << strerror(errno);
  }
  ~RDMAVan() = default;

  virtual std::string GetType() const { return std::string("rdma"); }

  Postoffice *postoffice_;

 protected:
  void Start(int customer_id, bool standalone) override {
    start_mu_.lock();
    should_stop_ = false;

    auto val = Environment::Get()->find("BYTEPS_ENABLE_IPC");
    disable_ipc_ = val ? !atoi(val) : true;
    if (disable_ipc_) {
      LOG(INFO) << "Shared memory IPC has been disabled";
    } else {
      std::string role = Environment::Get()->find("DMLC_ROLE");
      if (role == "joint") {
        LOG(INFO) << "You are using IPC in joint mode, make sure no P2P "
                     "operations are involved";
      }
    }
    if (event_channel_ == nullptr) {
      event_channel_ = rdma_create_event_channel();
      PS_CHECK(event_channel_) << "Create RDMA event channel failed";

      cm_event_polling_thread_.reset(
          new std::thread(&RDMAVan::PollEvents, this));
    }

    // enable logging
    val = Environment::Get()->find("BYTEPS_PRINT_RDMA_LOG");
    enable_log_ = val ? atoi(val) : false;
    if (enable_log_) LOG(INFO) << "Enable RDMA logging.";

    val = Environment::Get()->find("BYTEPS_RDMA_MAX_CONCURR_WR");
    if (val) {
      // should make sure: kMaxConcurrentWorkRequest >= kStartDepth +
      // kReplyDepth + kRxDepth
      kMaxConcurrentWorkRequest = atoi(val);

      auto start_depth_env =
          Environment::Get()->find("BYTEPS_RDMA_START_DEPTH");
      auto rx_depth_env = Environment::Get()->find("BYTEPS_RDMA_RX_DEPTH");

      auto start_depth = start_depth_env ? atoi(start_depth_env) : 128;
      auto rx_depth = rx_depth_env ? atoi(rx_depth_env) : 2048;
      auto reply_depth = rx_depth;

      PS_CHECK_GE(kMaxConcurrentWorkRequest,
                  start_depth + reply_depth + rx_depth)
          << "Should make sure: kMaxConcurrentWorkRequest >= kStartDepth + "
             "kReplyDepth + kRxDepth";
    }

    start_mu_.unlock();
    if (!standalone) Van::Start(customer_id, false);
  }

  void Stop() override {
    PS_VLOG(1) << my_node_.ShortDebugString() << " is stopping";
    Van::Stop();

    should_stop_ = true;
    PS_CHECK(should_stop_);

    PS_VLOG(1) << "Stopping cq_polling_thread_.";
    cq_polling_thread_->join();
    cq_polling_thread_.reset();

    PS_VLOG(1) << "Stopping cm_event_polling_thread_.";
    cm_event_polling_thread_->join();
    cm_event_polling_thread_.reset();

    PS_VLOG(1) << "Clearing memory allocator.";
    mem_allocator_.reset();

    PS_VLOG(1) << "Clearing endpoints.";
    incoming_.clear();
    {
      std::lock_guard<std::mutex> lk(endpoints_mu_);
      endpoints_.clear();
    }

    PS_VLOG(1) << "Destroying cq and pd.";

    PS_CHECK(!ibv_destroy_cq(cq_)) << "Failed to destroy CQ";

    for (auto &it : mem_mr_) ibv_dereg_mr(it.second);

    // TODO(non): ibv_dealloc_pd sometimes complains resource busy, need to fix
    // PS_CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD: " <<
    // strerror(errno);

    PS_VLOG(1) << "Destroying listener.";
    rdma_destroy_id(listener_);
    rdma_destroy_event_channel(event_channel_);
  }

  int Bind(Node &node, int max_retry) override {
    PS_CHECK_EQ(my_node_.num_ports, 1)
        << "RDMA van does not support multiple ports";
    PS_CHECK(rdma_create_id(event_channel_, &listener_, nullptr, RDMA_PS_TCP) ==
             0)
        << "Create RDMA connection identifier failed";

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));

    auto val = Environment::Get()->find("DMLC_NODE_HOST");
    if (val) {
      PS_VLOG(1) << "bind to DMLC_NODE_HOST: " << std::string(val);
      addr.sin_addr.s_addr = inet_addr(val);
    }

    addr.sin_family = AF_INET;
    int port = node.port;
    unsigned seed = static_cast<unsigned>(time(NULL) + port);
    for (int i = 0; i < max_retry + 1; ++i) {
      addr.sin_port = htons(port);
      if (rdma_bind_addr(listener_,
                         reinterpret_cast<struct sockaddr *>(&addr)) == 0) {
        break;
      }
      if (i == max_retry) {
        port = -1;
      } else {
        port = 10000 + rand_r(&seed) % 40000;
      }
    }
    PS_CHECK(rdma_listen(listener_, kRdmaListenBacklog) == 0)
        << "Listen RDMA connection failed: " << strerror(errno);
    return port;
  }

  void Connect(const Node &node) override {
    PS_VLOG(1) << "Connecting to Node " << node.id
               << ", My_Node=" << my_node_.id;
    PS_CHECK_NE(node.id, node.kEmpty);
    PS_CHECK_NE(node.port, node.kEmpty);
    PS_CHECK(node.hostname.size());

    // worker doesn't need to connect to the other workers. same for server
    if ((node.role == my_node_.role) && (node.id != my_node_.id)) {
      return;
    }

    if (node.id != Node::kEmpty) {
      endpoints_mu_.lock();
      auto it = endpoints_.find(node.id);

      // if there is an endpoint with pending connection
      if (it != endpoints_.end()) {
        endpoints_.erase(it);
      }

      Endpoint *endpoint;
      endpoints_[node.id] = std::make_unique<Endpoint>();
      endpoint = endpoints_[node.id].get();
      endpoints_mu_.unlock();

      endpoint->SetNodeID(node.id);

      struct addrinfo *remote_addr;
      PS_CHECK_EQ(
          getaddrinfo(node.hostname.c_str(), std::to_string(node.port).c_str(),
                      nullptr, &remote_addr),
          0);

      while (!endpoint->GetAllStatus(Endpoint::CONNECTED)) {
        std::unique_lock<std::mutex> lk(endpoint->connect_mu);
        endpoint->SetAllStatus(Endpoint::CONNECTING);

        FOR_QPS {
          if (endpoint->cm_ids[qpIndex] != nullptr) {
            rdma_destroy_qp(endpoint->cm_ids[qpIndex]);
            PS_CHECK_EQ(rdma_destroy_id(endpoint->cm_ids[qpIndex]), 0)
                << strerror(errno);
            endpoint->cm_ids[qpIndex] = nullptr;
          }
          PS_CHECK_EQ(rdma_create_id(event_channel_, &endpoint->cm_ids[qpIndex],
                                     nullptr, RDMA_PS_TCP),
                      0)
              << "Create RDMA connection identifier failed";
          endpoint->cm_ids[qpIndex]->context = endpoint;
        }
        auto val = Environment::Get()->find("DMLC_NODE_HOST");
        if (val) {
          struct addrinfo *addr;
          auto rc = getaddrinfo(val, "", NULL, &addr);
          PS_CHECK_EQ(rc, 0) << "getaddrinfo failed: " << gai_strerror(rc);
          FOR_QPS {
            PS_CHECK_EQ(
                rdma_resolve_addr(endpoint->cm_ids[qpIndex], addr->ai_addr,
                                  remote_addr->ai_addr, kTimeoutms),
                0)
                << "Resolve RDMA address failed with errno: "
                << strerror(errno);
          }
        } else {
          FOR_QPS {
            PS_CHECK_EQ(rdma_resolve_addr(endpoint->cm_ids[qpIndex], nullptr,
                                          remote_addr->ai_addr, kTimeoutms),
                        0)
                << "Resolve RDMA address failed with errno: "
                << strerror(errno);
          }
        }

        endpoint->cv.wait(lk, [endpoint] {
          // return endpoint->status != Endpoint::CONNECTING;
          return !endpoint->GetAllStatus(Endpoint::CONNECTING);
        });

        if (endpoint->GetAllStatus(Endpoint::CONNECTED)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }

      bool is_local_node =
          disable_ipc_ ? false
                       : (node.hostname == my_node_.hostname ? true : false);
      {
        std::lock_guard<std::mutex> lk(local_mu_);
        is_local_[node.id] = is_local_node;
      }

      LOG(INFO) << "Connect to Node " << node.id << " (" << node.hostname << ")"
                << " with Transport=" << (is_local_node ? "IPC" : "RDMA");

      std::shared_ptr<Transport> t =
          is_local_node ? std::make_shared<IPCTransport>(
                              endpoint, mem_allocator_.get(), postoffice_)
                        : std::make_shared<RDMATransport>(
                              endpoint, mem_allocator_.get(), postoffice_);
      endpoint->SetTransport(t);

      freeaddrinfo(remote_addr);
    }
  }

  void RegisterRecvBuffer(Message &msg) override {
    RegisterMemory(msg);
    std::unique_lock<std::mutex> lock(registered_recv_buffers_mu_);
    uint64_t key = DecodeKey(msg.data[0]);
    int node_id = msg.meta.sender;
    registered_recv_buffers_[key].push_back(
        std::make_pair(node_id, msg.data[1]));
    LOG(INFO) << "register recv buffer: key=" << key << ", node" << node_id
              << ", size=" << msg.data[1].size();
  }

  void QueryRecvBuffer(uint64_t key, int node_id, void **buffer, size_t *size,
                       uint32_t *rkey) override {
    std::unique_lock<std::mutex> lock(registered_recv_buffers_mu_);
    auto itr = registered_recv_buffers_.find(key);
    if (itr != registered_recv_buffers_.end()) {
      for (auto &t : itr->second) {
        if (t.first == node_id) {
          *buffer = t.second.data();
          *size = t.second.size();
          std::unique_lock<std::mutex> mem_lock(map_mu_);
          PS_CHECK(mem_mr_.find(t.second.data()) != mem_mr_.end());
          *rkey = mem_mr_[t.second.data()]->rkey;
          return;
        }
      }
    }

    *buffer = nullptr;
    *size = 0;
  }

  int SendMsg(Message &msg) override {
    int remote_id = msg.meta.recver;
    PS_CHECK_NE(remote_id, Meta::kEmpty);
    Endpoint *endpoint = nullptr;
    {
      std::unique_lock<std::mutex> lock(endpoints_mu_);
      auto itr = endpoints_.find(remote_id);
      PS_CHECK_NE(itr, endpoints_.end())
          << "failed to found enpoints " << remote_id;
      endpoint = itr->second.get();
    }

    int meta_len = GetPackMetaLen(msg.meta);
    size_t data_len = msg.meta.data_size;
    size_t total_len = meta_len + data_len;
    PS_CHECK(meta_len);

    RegisterMemory(msg);
    // pack meta info
    if (IsValidPushpull(msg)) {
      AddMeta(msg);
    }
    auto trans = PS_CHECK_NOTNULL(endpoint->GetTransport());

    // start rendezvous if no remote info
    if (!IsValidPushpull(msg)) {
      MessageBuffer *msg_buf = PrepareNewMsgBuf(msg);
      StoreMsgBuf(msg_buf, msg);
      trans->SendRendezvousBegin(msg, msg_buf);
      return total_len;
    }

    if (!HasRemoteInfo(msg, msg.meta.key, msg.meta.push, remote_id)) {
      MessageBuffer *msg_buf = PrepareNewMsgBuf(msg);
      StoreMsgBuf(msg_buf, msg);
      PrepareData(msg, msg_buf);
      trans->SendRendezvousBegin(msg, msg_buf);
      return total_len;
    }

    auto addr_tuple =
        GetRemoteAndLocalInfo(msg.meta.key, msg.meta.push, remote_id);
#ifdef STEPMESH_USE_GDR
    MessageBuffer *msg_buf = std::get<5>(addr_tuple);  // local message buffer
#else
    MessageBuffer *msg_buf = std::get<3>(addr_tuple);  // local message buffer
#endif

    // prepare new meta and data
    PS_CHECK_EQ(msg_buf->inline_len, (size_t)meta_len);
    PS_CHECK(msg_buf->inline_buf);
    msg_buf->data = msg.data;  // may not need this

    if (msg.meta.tensor_ev != nullptr) {
      msg.meta.tensor_ev->Sync();
      msg.meta.tensor_ev->Release();
      msg.meta.tensor_ev = nullptr;
    }

#ifdef STEPMESH_ENABLE_TRACE
    if (msg.meta.request) {
      msg.meta.request_trace.postsend = GetNanosecond();
    } else {
      msg.meta.response_trace.postsend = GetNanosecond();
    }
#endif  // STEPMESH_ENABLE_TRACE
    PackMeta(msg.meta, &(msg_buf->inline_buf), &meta_len);
#ifdef STEPMESH_ENABLE_TRACE
    PrintSendLog(msg, msg_buf, addr_tuple);
#endif  // STEPMESH_ENABLE_TRACE
    if (msg.meta.push && msg.meta.request) {
      // worker, push request
      trans->SendPushRequest(msg, msg_buf, addr_tuple);
    } else if (msg.meta.push && !msg.meta.request) {
      // server, push response
      trans->SendPushResponse(msg, msg_buf, addr_tuple);
    } else if (!msg.meta.push && msg.meta.request) {
      // worker, pull request
      trans->SendPullRequest(msg, msg_buf, addr_tuple);
    } else if (!msg.meta.push && !msg.meta.request) {
      // server, pull response
      map_mu_.lock();
      auto temp_mr = mem_mr_.find(msg_buf->data[1].data());
      PS_CHECK_NE(temp_mr, mem_mr_.end());
      map_mu_.unlock();
      trans->SendPullResponse(msg, msg_buf, addr_tuple, temp_mr->second->lkey);
    } else {
      PS_CHECK(0) << "unexpected message type";
    }
    return total_len;
  }

  int RecvMsg(Message *msg) override {
    msg->data.clear();
    std::tuple<Endpoint *, BufferContext *, uint64_t, int> notification;
    recv_buffers_.WaitAndPop(&notification);
    int cmd = std::get<int>(notification);
    Endpoint *endpoint = std::get<Endpoint *>(notification);
    BufferContext *buffer_ctx = std::get<BufferContext *>(notification);
    auto trans = PS_CHECK_NOTNULL(endpoint->GetTransport());
    msg->meta.recver = my_node_.id;
    msg->meta.sender = endpoint->node_id;

    // the second argument is actually deprecated,
    // we keep it as is in order to be compatible
#ifdef STEPMESH_USE_GDR
    bool is_server = static_cast<RDMATransport *>(trans.get())->is_server_;
    char *meta_buf = is_server ? buffer_ctx->meta_buffer : buffer_ctx->buffer;
    PS_CHECK(meta_buf);
#else
    char *meta_buf = buffer_ctx->buffer;
    PS_CHECK(meta_buf);
#endif

    RawMeta *raw = reinterpret_cast<RawMeta *>(meta_buf);

    auto counters = raw->slave_qp_counter;
    if (raw->slave_qp_num > 0) {
      while (1) {
        bool matched = true;
        for (int i = 1; i < raw->slave_qp_num + 1; i++) {
          matched = matched && (counters[i] <= endpoint->qp_pkt_count[i]);
        }
        if (matched) {
          break;
        }
      }
      PS_VLOG(3) << "Recv Msg: qp: " << endpoint->cm_ids[1]->qp->qp_num
                 << " , rCounter: " << counters[1]
                 << "lCounter: " << endpoint->qp_pkt_count[1];
    }

    UnpackMeta(meta_buf, buffer_ctx->meta_len, &msg->meta);
    int meta_len = GetPackMetaLen(msg->meta);

#ifdef STEPMESH_ENABLE_TRACE
    if (msg->meta.request) {
      msg->meta.request_trace.postrecv = std::get<uint64_t>(notification);
    } else {
      msg->meta.response_trace.postrecv = std::get<uint64_t>(notification);
    }
#endif  // STEPMESH_ENABLE_TRACE

    int total_len = 0;
    total_len += meta_len;

#ifdef STEPMESH_ENABLE_TRACE
    PrintRecvLog(msg, buffer_ctx, meta_len);
#endif  // STEPMESH_ENABLE_TRACE
    if (!IsValidPushpull(*msg)) {
      return total_len;
    }

    if (cmd != 0) {
      msg->meta.head = cmd & 0xFF;
    }

    // valid data message
    if (msg->meta.push && msg->meta.request) {
      // push request
      total_len += trans->RecvPushRequest(msg, buffer_ctx, meta_len);
    } else if (!msg->meta.push && msg->meta.request) {
      // pull request
      total_len += trans->RecvPullRequest(msg, buffer_ctx, meta_len);
    } else if (msg->meta.push && !msg->meta.request) {
      // push response
      total_len += trans->RecvPushResponse(msg, buffer_ctx, meta_len);
    } else if (!msg->meta.push && !msg->meta.request) {
      // pull response
      total_len += trans->RecvPullResponse(msg, buffer_ctx, meta_len);
    } else {
      PS_CHECK(0) << "unknown msg type";
    }

    return total_len;
  }

 private:
  void PrintSendLog(Message &msg, MessageBuffer *msg_buf,
                    RemoteTuple remote_tuple) {
    if (!enable_log_) return;
    std::lock_guard<std::mutex> lock(log_mu_);

    if (!IsValidPushpull(msg)) {
      LOG(INFO) << "Send Control Message" << std::flush;
    } else if (msg.meta.push && msg.meta.request) {
      // worker, push request
      LOG(INFO) << "Send Push Request: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t tensor_len=" << msg_buf->mrs[0].second
                << "\t remote_idx=" << std::get<2>(remote_tuple)
                << "\t remote_addr="
                << reinterpret_cast<void *>(std::get<0>(remote_tuple))
                << std::flush;
    } else if (msg.meta.push && !msg.meta.request) {
      // server, push response
      LOG(INFO) << "Send Push Response: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t remote_idx=" << std::get<2>(remote_tuple)
                << "\t remote_addr="
                << reinterpret_cast<void *>(std::get<0>(remote_tuple))
                << std::flush;
    } else if (!msg.meta.push && msg.meta.request) {
      // worker, pull request
      LOG(INFO) << "Send Pull Request: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t remote_idx=" << std::get<2>(remote_tuple)
                << "\t remote_addr="
                << reinterpret_cast<void *>(std::get<0>(remote_tuple))
                << std::flush;
    } else if (!msg.meta.push && !msg.meta.request) {
      // server, pull response
      LOG(INFO) << "Send Pull Response: key=" << msg.meta.key
                << "\t timestamp=" << msg.meta.timestamp
                << "\t recver=" << msg.meta.recver
                << "\t tensor_len=" << msg.meta.val_len << "\t idx="
                << "none"
                << "\t remote_addr="
                << reinterpret_cast<void *>(std::get<0>(remote_tuple))
                << std::flush;
    }
  }

  void PrintRecvLog(Message *msg, BufferContext *buffer_ctx, int meta_len) {
    if (!enable_log_) return;
    std::lock_guard<std::mutex> lock(log_mu_);

    if (!IsValidPushpull(*msg)) {
      LOG(INFO) << "Recv Control Message" << std::flush;
    } else if (msg->meta.push && msg->meta.request) {
      // push request
      LOG(INFO) << "Recv Push Request: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender
                << "\t tensor_len=" << buffer_ctx->data_len[1] << std::flush;
    } else if (!msg->meta.push && msg->meta.request) {
      // pull request
      LOG(INFO) << "Recv Pull Request: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender << std::flush;
    } else if (msg->meta.push && !msg->meta.request) {
      // push response
      LOG(INFO) << "Recv Push Response: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender << std::flush;
    } else if (!msg->meta.push && !msg->meta.request) {
      // pull response
      LOG(INFO) << "Recv Pull Response: key=" << msg->meta.key
                << "\t timestamp=" << msg->meta.timestamp
                << "\t sender=" << msg->meta.sender
                << "\t tensor_len=" << msg->meta.val_len;
    }
  }

  bool HasRemoteInfo(Message &msg, uint64_t key, bool is_push, int recver) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    if (is_push && (push_addr_.find(key) != push_addr_.end()) &&
        (push_addr_[key].find(recver) != push_addr_[key].end())) {
      return true;
    }
    if (!is_push && (pull_addr_.find(key) != pull_addr_.end()) &&
        (pull_addr_[key].find(recver) != pull_addr_[key].end())) {
      return true;
    }

    return false;
  }

  void StoreMsgBuf(MessageBuffer *msg_buf, Message &msg) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    PS_CHECK_EQ(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());
    msgbuf_cache_[msg_buf] = msg;
  }

  Message *GetFirstMsg(MessageBuffer *msg_buf) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    PS_CHECK_NE(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());
    return &msgbuf_cache_[msg_buf];
  }

  void ReleaseFirstMsg(MessageBuffer *msg_buf) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    PS_CHECK_NE(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());
    msgbuf_cache_.erase(msg_buf);
  }

#ifdef STEPMESH_USE_GDR
  void StoreRemoteAndLocalInfo(MessageBuffer *msg_buf, uint64_t meta_addr,
                               uint32_t meta_rkey, uint64_t data_addr,
                               uint32_t data_rkey, uint32_t idx) {
    std::lock_guard<std::mutex> lk(addr_mu_);

    PS_CHECK_NE(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());

    auto &msg = msgbuf_cache_[msg_buf];

    auto key = msg.meta.key;
    auto is_push = msg.meta.push;
    auto recver = msg.meta.recver;

    auto t = std::make_tuple(meta_addr, meta_rkey, data_addr, data_rkey, idx,
                             msg_buf);
    if (is_push) {
      push_addr_[key][recver] = t;
    } else {
      pull_addr_[key][recver] = t;
    }
  }
#endif

  void StoreRemoteAndLocalInfo(MessageBuffer *msg_buf, uint64_t remote_addr,
                               uint32_t rkey, uint32_t idx) {
    std::lock_guard<std::mutex> lk(addr_mu_);

    PS_CHECK_NE(msgbuf_cache_.find(msg_buf), msgbuf_cache_.end());

    auto &msg = msgbuf_cache_[msg_buf];

    auto key = msg.meta.key;
    auto is_push = msg.meta.push;
    auto recver = msg.meta.recver;

#ifdef STEPMESH_USE_GDR
    // When GDR is enabled, this function is only used for non-push/pull,
    // where data_addr and data_rkey are not used.
    auto t = std::make_tuple(remote_addr, rkey, 0, 0, idx, msg_buf);
#else
    auto t = std::make_tuple(remote_addr, rkey, idx, msg_buf);
#endif
    if (is_push) {
      push_addr_[key][recver] = t;
    } else {
      pull_addr_[key][recver] = t;
    }
  }

  RemoteTuple GetRemoteAndLocalInfo(uint64_t key, bool is_push, int recver) {
    std::lock_guard<std::mutex> lk(addr_mu_);
    return (is_push ? push_addr_[key][recver] : pull_addr_[key][recver]);
  }

  MessageBuffer *PrepareNewMsgBuf(Message &msg) {
    MessageBuffer *msg_buf = new MessageBuffer();
    auto meta_len = GetPackMetaLen(msg.meta);
    msg_buf->inline_len = meta_len;
    msg_buf->inline_buf = mem_allocator_->Alloc(meta_len);
    msg_buf->data = msg.data;
    PackMeta(msg.meta, &(msg_buf->inline_buf), &meta_len);
    return msg_buf;
  }

  void RegisterMemory(Message &msg) {
    size_t sa_cnt = 0;
    for (auto &sa : msg.data) {
      if (sa.size() == 0) continue;
      std::lock_guard<std::mutex> lock(map_mu_);
      if ((mem_mr_.find(sa.data()) == mem_mr_.end()) &&
          (sa_cnt == 1)) {  // only vals register memory
        struct ibv_mr *temp_mr;
        temp_mr = ibv_reg_mr(mem_allocator_->GetPD(), sa.data(), sa.size(),
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (temp_mr == nullptr) {
          LOG(WARNING) << "Failed to register the memory region: "
                       << strerror(errno) << ", sa.size()=" << sa.size();
          PS_CHECK(0);
        }

        mem_mr_[sa.data()] = temp_mr;
      }
      ++sa_cnt;
    }
    // register for tensor address of pull request
    if (IsValidPushpull(msg) && !msg.meta.push && msg.meta.request) {
      PS_CHECK_GT(msg.meta.val_len, 0) << msg.meta.val_len;
      auto addr = reinterpret_cast<char *>(msg.meta.addr);
      std::lock_guard<std::mutex> lock(map_mu_);
      if (mem_mr_.find(addr) == mem_mr_.end()) {
        struct ibv_mr *temp_mr;
        temp_mr = ibv_reg_mr(mem_allocator_->GetPD(), addr, msg.meta.val_len,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (temp_mr == nullptr) {
          LOG(WARNING) << "Failed to register the memory region: "
                       << strerror(errno);
          PS_CHECK(0);
        }
        mem_mr_[addr] = temp_mr;
      }
    }
  }

  void PrepareData(Message &msg, MessageBuffer *msg_buf) {
    if (!(msg.meta.push && msg.meta.request)) return;  // only push request
    auto &sa = msg_buf->data[1];
    if (sa.size() == 0) return;
    std::lock_guard<std::mutex> lock(map_mu_);
    auto it = mem_mr_.find(sa.data());
    PS_CHECK_NE(it, mem_mr_.end());
    MRPtr ptr(it->second, [](struct ibv_mr *mr) {});
    PS_CHECK(ptr.get()) << strerror(errno);
    msg_buf->mrs.push_back(std::make_pair(std::move(ptr), sa.size()));
  }

  void AddMeta(Message &msg) {
    if (msg.meta.request) {
      msg.meta.key = DecodeKey(msg.data[0]);
    }
    if (!msg.meta.push && msg.meta.request) {
      // pull request
      std::lock_guard<std::mutex> lock(map_mu_);
      auto val_addr = reinterpret_cast<char *>(msg.meta.addr);
      msg.meta.option = mem_mr_[val_addr]->rkey;
    }
  }

  void InitContext(struct ibv_context *context) {
    context_ = context;
    PS_CHECK(context_) << "ibv_context* empty";

    pd_ = ibv_alloc_pd(context_);
    PS_CHECK(pd_) << "Failed to allocate protection domain";

    mem_allocator_.reset(new MemoryAllocator(pd_));

    // TODO(clan): Replace the rough estimate here
    cq_ = ibv_create_cq(context_, kMaxConcurrentWorkRequest * 2, NULL, nullptr,
                        0);
    PS_CHECK(cq_) << "Failed to create completion queue";
  }

  void ReleaseWorkRequestContext(WRContext *context, Endpoint *endpoint,
                                 int qpIndex = 0) {
    switch (context->type) {
      case kRendezvousStartContext:
        endpoint->free_start_ctx.Push(context);
        break;
      case kRendezvousReplyContext:
        endpoint->free_reply_ctx.Push(context);
        break;
      case kReceiveContext: {
        auto id = endpoint->cm_ids[qpIndex];
        endpoint->PostRecv(context, id);
        break;
      }
      default:
        PS_CHECK(0);
    }
  }

  void PollCQ() {
    // Pre-allocated work completions array used for polling
    struct ibv_wc wc[kMaxConcurrentWorkRequest];
    BindCpuCore(1, 1);

    while (!should_stop_.load()) {
      int ne = ibv_poll_cq(cq_, kMaxConcurrentWorkRequest, wc);

      PS_CHECK_GE(ne, 0);
      for (int i = 0; i < ne; ++i) {
        PS_CHECK(wc[i].status == IBV_WC_SUCCESS)
            << "Failed status \n"
            << ibv_wc_status_str(wc[i].status) << " " << wc[i].status << " "
            << static_cast<uint64_t>(wc[i].wr_id) << " " << wc[i].vendor_err
            << " " << wc[i].opcode << " "
            << (wc[i].opcode == IBV_WC_RECV ? "RECV" : "OTHER")
            << " postoffice ptr: " << reinterpret_cast<void *>(postoffice_);

        // IBV_WC_RDMA_WRITE use msg_buf as the wr_id
        // so there won't be context and endpoint for this op
        if (wc[i].opcode == IBV_WC_RDMA_WRITE) {
          continue;
        }
        WRContext *context = reinterpret_cast<WRContext *>(wc[i].wr_id);
        Endpoint *endpoint =
            reinterpret_cast<Endpoint *>(context->private_data);

        // IBV_WC_RDMA_WRITE use msg_buf as the wr_id
        // so there won't be context and endpoint for this op
        switch (wc[i].opcode) {
          case IBV_WC_SEND: {
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          case IBV_WC_RECV_RDMA_WITH_IMM: {
            int cmIdInde = 0;
            FOR_QPS {
              if (wc[i].qp_num == endpoint->cm_ids[qpIndex]->qp->qp_num) {
                cmIdInde = qpIndex;
              }
            }
            uint32_t addr_idx = wc[i].imm_data;
            if ((addr_idx & 0x8000) == 0) {
              uint32_t cmd = ((addr_idx & 0xFFFF) >> 16) & 0xFFFF;

              endpoint->master_id = endpoint->cm_ids[cmIdInde];
              BufferContext *buf_ctx = addr_pool_.GetAddress(addr_idx);
              recv_buffers_.Push(
                  std::make_tuple(endpoint, buf_ctx, GetNanosecond(), cmd));
            } else {
              // slave qp
              endpoint->qp_pkt_count[1]++;
            }
            ReleaseWorkRequestContext(context, endpoint, cmIdInde);
          } break;
          case IBV_WC_RECV: {
            PS_CHECK(wc[i].wc_flags & IBV_WC_WITH_IMM);
            uint32_t imm = wc[i].imm_data;
            struct ibv_mr *mr = context->buffer;

            if (imm == kRendezvousStart) {
              RendezvousStart *req =
                  reinterpret_cast<RendezvousStart *>(mr->addr);
              auto trans = PS_CHECK_NOTNULL(endpoint->GetTransport());
              trans->SendRendezvousReply(req, addr_pool_);
            } else if (imm == kRendezvousReply) {
              RendezvousReply *resp =
                  reinterpret_cast<RendezvousReply *>(mr->addr);

              uint64_t origin_addr = resp->origin_addr;
              uint32_t idx = resp->idx;

              MessageBuffer *msg_buf =
                  reinterpret_cast<MessageBuffer *>(origin_addr);
              // Before RDMA write, store the remote info so that
              // subsequent write does not need repeated rendezvous
#ifdef STEPMESH_USE_GDR
              StoreRemoteAndLocalInfo(msg_buf, resp->meta_addr, resp->meta_rkey,
                                      resp->data_addr, resp->data_rkey, idx);
#else
              StoreRemoteAndLocalInfo(msg_buf, resp->addr, resp->rkey, idx);
#endif

              Message *msg = GetFirstMsg(msg_buf);
              auto addr_tuple = GetRemoteAndLocalInfo(
                  msg->meta.key, msg->meta.push, msg->meta.recver);
              auto trans = PS_CHECK_NOTNULL(endpoint->GetTransport());
              if (!IsValidPushpull(*msg)) {
#ifdef STEPMESH_USE_GDR
                // control message. meta_addr contains the full buffer.
                trans->RDMAWriteWithImm(msg_buf, resp->meta_addr,
                                        resp->meta_rkey, idx);
#else
                // control message
                trans->RDMAWriteWithImm(msg_buf, resp->addr, resp->rkey, idx);
#endif
              } else if (msg->meta.push && msg->meta.request) {
                // worker, push request
                trans->SendPushRequest(*msg, msg_buf, addr_tuple);
              } else if (msg->meta.push && !msg->meta.request) {
                // server, push response
                trans->SendPushResponse(*msg, msg_buf, addr_tuple);
              } else if (!msg->meta.push && msg->meta.request) {
                // worker, pull request
                trans->SendPullRequest(*msg, msg_buf, addr_tuple);
              } else if (!msg->meta.push && !msg->meta.request) {
                // server, pull response
                map_mu_.lock();
                auto temp_mr = mem_mr_.find(msg_buf->data[1].data());
                PS_CHECK_NE(temp_mr, mem_mr_.end());
                map_mu_.unlock();
                trans->SendPullResponse(*msg, msg_buf, addr_tuple,
                                        temp_mr->second->lkey);
              }

              // release the msg_buf from msgbuf_cache_
              ReleaseFirstMsg(msg_buf);
            } else {
              PS_CHECK(0);
            }
            ReleaseWorkRequestContext(context, endpoint);
          } break;
          default:
            PS_CHECK(0) << "Unexpected opcode: " << wc[i].opcode;
        }
      }
    }
  }

  void PollEvents() {
    int flags = fcntl(event_channel_->fd, F_GETFL);
    int rc = fcntl(event_channel_->fd, F_SETFL, flags | O_NONBLOCK);
    PS_CHECK_GE(rc, 0);
    int error_flags = POLLERR | POLLHUP | POLLNVAL;

    while (!should_stop_.load()) {
      struct pollfd pfd = {
          .fd = event_channel_->fd, .events = POLLIN, .revents = 0};
      int ret = poll(&pfd, 1, 10);

      PS_CHECK_GE(ret, 0) << strerror(errno);
      PS_CHECK_EQ(pfd.revents & error_flags, 0);

      if (!(pfd.revents & POLLIN)) {
        continue;
      }

      struct rdma_cm_event *event;
      PS_CHECK_EQ(rdma_get_cm_event(event_channel_, &event), 0);
      // TODO(clan): Reorder the list according to the event frequency
      switch (event->event) {
        case RDMA_CM_EVENT_CONNECT_REQUEST:
          OnConnectRequest(event);
          break;
        case RDMA_CM_EVENT_ADDR_RESOLVED:
          OnAddrResolved(event);
          break;
        case RDMA_CM_EVENT_ROUTE_RESOLVED:
          OnRouteResolved(event);
          break;
        case RDMA_CM_EVENT_ESTABLISHED:
          OnConnected(event);
          break;
        case RDMA_CM_EVENT_DISCONNECTED:
          OnDisconnected(event);
          break;
        case RDMA_CM_EVENT_REJECTED:
          OnRejected(event);
          break;
        default:
          PS_CHECK(0) << "OnEvent: unknown event " << event->event << " ("
                      << rdma_event_str(event->event) << ")";
      }
      rdma_ack_cm_event(event);
    }
  }

  void OnRejected(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);

    endpoints_mu_.lock();
    auto it = endpoints_.find(endpoint->node_id);
    PS_CHECK(it != endpoints_.end()) << "Connection not ready.";
    endpoints_mu_.unlock();

    PS_VLOG(1) << my_node_.id << " to " << endpoint->node_id
               << " connection rejected, retrying...";
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->SetAllStatus(Endpoint::REJECTED);
    }
    endpoint->cv.notify_all();
  }

  // Server Side
  void OnConnectRequest(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    PS_CHECK_NOTNULL(id);
    PS_CHECK_LE(sizeof(RequestContext), event->param.conn.private_data_len)
        << "RequestContext size mismatch. Actual: "
        << event->param.conn.private_data_len
        << ", Expected: " << sizeof(RequestContext);
    PS_CHECK_NOTNULL(event->param.conn.private_data);

    const RequestContext *remote_ctx = reinterpret_cast<const RequestContext *>(
        event->param.conn.private_data);

    Endpoint *endpoint = nullptr;
    std::string rem_host = std::string(remote_ctx->hostname) + "," +
                           std::to_string(remote_ctx->node) + "," +
                           std::to_string(remote_ctx->port);
    auto itr = incoming_.find(rem_host);
    if (itr != incoming_.end()) {
      endpoint = itr->second.get();
    } else {
      incoming_[rem_host] = std::make_unique<Endpoint>();
      endpoint = incoming_[rem_host].get();
    }
    endpoint->SetNodeID(remote_ctx->node);
    endpoint->cm_ids[endpoint->inComingCount] = id;
    endpoint->inComingCount++;
    id->context = endpoint;

    if (context_ == nullptr) {
      InitContext(id->verbs);
    }

    endpoint->Init(cq_, pd_, id);

    bool is_local_node =
        disable_ipc_
            ? false
            : (std::string(remote_ctx->hostname) == my_node_.hostname ? true
                                                                      : false);
    {
      std::lock_guard<std::mutex> lk(local_mu_);
      is_local_[remote_ctx->node] = is_local_node;
    }

    PS_LOG(INFO) << my_node_.id << " OnConnect to " << remote_ctx->node
                 << " with Transport=" << (is_local_node ? "IPC" : "RDMA")
                 << " QP_NUM " << id->qp->qp_num;

    std::shared_ptr<Transport> t =
        is_local_node ? std::make_shared<IPCTransport>(
                            endpoint, mem_allocator_.get(), postoffice_)
                      : std::make_shared<RDMATransport>(
                            endpoint, mem_allocator_.get(), postoffice_);
    endpoint->SetTransport(t);

    RequestContext ctx;
    ctx.node = static_cast<uint32_t>(my_node_.id);
    ctx.port = static_cast<uint16_t>(my_node_.port);
    snprintf(ctx.hostname, kMaxHostnameLength, "%s", my_node_.hostname.c_str());
    struct rdma_conn_param cm_params;
    memset(&cm_params, 0, sizeof(cm_params));
    cm_params.retry_count = 7;
    cm_params.rnr_retry_count = 7;
    cm_params.private_data = &ctx;
    cm_params.private_data_len = sizeof(RequestContext);

    PS_CHECK_EQ(rdma_accept(id, &cm_params), 0)
        << "Accept RDMA connection failed: " << strerror(errno);
    if (endpoint->inComingCount == QP_NUM) endpoint->inComingCount = 0;
  }

  // Resolve a route after address is resolved
  void OnAddrResolved(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    PS_CHECK_EQ(rdma_resolve_route(id, kTimeoutms), 0)
        << "Resolve RDMA route failed";
  }

  // Make a connection after route is resolved
  void OnRouteResolved(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);

    if (context_ == nullptr) {
      InitContext(id->verbs);
    }
    endpoint->Init(cq_, pd_, id);
    endpoint->inComingCount++;
    RequestContext ctx;
    ctx.node = static_cast<uint32_t>(my_node_.id);
    ctx.port = static_cast<uint16_t>(my_node_.port);
    snprintf(ctx.hostname, kMaxHostnameLength, "%s", my_node_.hostname.c_str());

    struct rdma_conn_param cm_params;
    memset(&cm_params, 0, sizeof(cm_params));
    cm_params.retry_count = 7;
    cm_params.rnr_retry_count = 7;
    cm_params.private_data = &ctx;
    cm_params.private_data_len = sizeof(RequestContext);
    PS_CHECK_EQ(rdma_connect(id, &cm_params), 0)
        << "RDMA connect failed" << strerror(errno);
    LOG(INFO) << " QP NUM " << id->qp->qp_num;
    if (endpoint->inComingCount == QP_NUM) endpoint->inComingCount = 0;
  }

  void OnConnected(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    PS_CHECK(id) << "rdma_cm_id not found.";
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);
    PS_CHECK(endpoint) << "Endpoint not found.";
    if (cq_polling_thread_ == nullptr) {
      cq_polling_thread_.reset(new std::thread(&RDMAVan::PollCQ, this));
      int gpu = -1;
      Environment::Get()->find("STEPMESH_GPU", &gpu, gpu);
    }

    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      endpoint->status_list[endpoint->inComingCount] = Endpoint::CONNECTED;
    }

    // endpoint->cm_ids[endpoint->inComingCount] = id;
    endpoint->inComingCount++;
    if (endpoint->GetAllStatus(Endpoint::CONNECTED)) {
      endpoint->cv.notify_all();
    }
    if (endpoint->node_id != my_node_.id) {
      PS_VLOG(1) << my_node_.id << " OnConnected to " << endpoint->node_id;
    }
    if (endpoint->inComingCount == QP_NUM) {
      LOG(INFO) << "ONConnected: " << QP_NUM;
      endpoint->SetQPLag(my_node_.id, endpoint->node_id);
      endpoint->inComingCount = 0;
    }
  }

  void OnDisconnected(struct rdma_cm_event *event) {
    struct rdma_cm_id *id = event->id;
    Endpoint *endpoint = reinterpret_cast<Endpoint *>(id->context);
    {
      std::lock_guard<std::mutex> lk(endpoint->connect_mu);
      // endpoint->status = Endpoint::IDLE;
      endpoint->SetAllStatus(Endpoint::IDLE);
    }
    endpoint->cv.notify_all();
    LOG(INFO) << my_node_.id << " OnDisconnected from " << endpoint->node_id;
  }

  AddressPool<BufferContext> addr_pool_;
  std::unique_ptr<MemoryAllocator> mem_allocator_;

  std::unique_ptr<RDMATransport> rdma_trans_;
  std::unique_ptr<IPCTransport> ipc_trans_;

  struct rdma_cm_id *listener_ = nullptr;
  std::atomic<bool> should_stop_;

  std::mutex endpoints_mu_;
  std::unordered_map<int, std::unique_ptr<Endpoint>> endpoints_;
  std::unordered_map<std::string, std::unique_ptr<Endpoint>> incoming_;

  struct rdma_event_channel *event_channel_ = nullptr;
  struct ibv_context *context_ = nullptr;

  // ibverbs protection domain
  struct ibv_pd *pd_ = nullptr;
  // Completion queue, to poll on work completions
  struct ibv_cq *cq_ = nullptr;
  // cq thread
  std::unique_ptr<std::thread> cq_polling_thread_ = nullptr;
  // event thread
  std::unique_ptr<std::thread> cm_event_polling_thread_ = nullptr;
  // Recv buffer queue
  ThreadsafeQueue<std::tuple<Endpoint *, BufferContext *, uint64_t, int>>
      recv_buffers_;

  // local IPC related
  bool disable_ipc_ = false;
  std::mutex local_mu_;
  std::unordered_map<int, bool> is_local_;

  std::mutex addr_mu_;
  // <key, recver>, (<remote_addr, rkey, idx, local_addr>)
  std::unordered_map<uint64_t, RemoteAndLocalAddress> push_addr_;
  std::unordered_map<uint64_t, RemoteAndLocalAddress> pull_addr_;
  std::unordered_map<MessageBuffer *, Message> msgbuf_cache_;  // msg_buf, msg

  std::mutex map_mu_;
  std::unordered_map<char *, struct ibv_mr *>
      mem_mr_;  // (memory address, ibv_mr)

  // logging
  bool enable_log_;
  std::mutex log_mu_;

  int kMaxConcurrentWorkRequest = 4224;  // 128 + 2048 * 2

  std::mutex registered_recv_buffers_mu_;
  std::unordered_map<uint64_t, std::vector<std::pair<int, SArray<char>>>>
      registered_recv_buffers_;
};  // class RDMAVan

};  // namespace ps

#endif  // DMLC_USE_RDMA
#endif  // RDMA_VAN_H_
