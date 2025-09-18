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

#ifndef RDMA_TRANSPORT_H_
#define RDMA_TRANSPORT_H_

#ifdef DMLC_USE_RDMA

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "./ibvwarp.h"
#include "./rdma_utils.h"
#include "dmlc/logging.h"
#include "ps/internal/multi_qp.h"

namespace ps {

class Postoffice;
class Transport;

struct Endpoint {
  enum ConnectionStatus { IDLE, CONNECTING, CONNECTED, REJECTED };

  ConnectionStatus status_list[QP_MAX_NUM];
  uint64_t qp_pkt_count[QP_MAX_NUM];
  int node_id;
  std::condition_variable cv;
  std::mutex connect_mu;
  struct rdma_cm_id *cm_ids[QP_MAX_NUM];
  struct rdma_cm_id *master_id;
  std::shared_ptr<Transport> trans;
  bool multi_qp_ = false;

  int inComingCount = 0;
  int kStartDepth = 128;
  int kRxDepth = 256;
  int kReplyDepth = kRxDepth;
  WRContext *rx_ctx;
  WRContext *start_ctx;
  WRContext *reply_ctx;

  ThreadsafeQueue<WRContext *> free_start_ctx;
  ThreadsafeQueue<WRContext *> free_reply_ctx;

  uint8_t inited = 0;

  Endpoint() : node_id(Node::kEmpty), rx_ctx() {
    if (wrap_ibv_symbols() != 1) {
      PS_LOG(WARNING) << "Load mlx5 symbols fails.";
    }
    FOR_QPS {
      cm_ids[qpIndex] = nullptr;
      status_list[qpIndex] = IDLE;
      qp_pkt_count[qpIndex] = 0;
    }
    auto byteps_rx_depth = Environment::Get()->find("BYTEPS_RDMA_RX_DEPTH");
    auto byteps_start_depth =
        Environment::Get()->find("BYTEPS_RDMA_START_DEPTH");
    const char *role_val =
        PS_CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
    std::string role_str(role_val);
    // for joint mode with large number of workers, the default value of rx/tx
    // depth is reduced for less memory consumption.
    if (role_str == "scheduler") {
      kStartDepth = 256;
      kRxDepth = 16;
    }
    kStartDepth = byteps_start_depth ? atoi(byteps_start_depth) : kStartDepth;
    kRxDepth = byteps_rx_depth ? atoi(byteps_rx_depth) : kRxDepth;
    kReplyDepth = kRxDepth;

    start_ctx = new WRContext[kStartDepth];
    reply_ctx = new WRContext[kReplyDepth];
    rx_ctx = new WRContext[kRxDepth];
  }

  ~Endpoint() {
    for (int i = 0; i < kRxDepth; ++i) {
      if (!(rx_ctx[i].buffer)) {
        continue;
      }
      free(rx_ctx[i].buffer->addr);
      PS_CHECK_EQ(ibv_dereg_mr(rx_ctx[i].buffer), 0);
    }

    for (int i = 0; i < kStartDepth; ++i) {
      if (start_ctx[i].buffer) {
        free(start_ctx[i].buffer->addr);
        PS_CHECK_EQ(ibv_dereg_mr(start_ctx[i].buffer), 0);
      }
    }

    for (int i = 0; i < kReplyDepth; ++i) {
      if (reply_ctx[i].buffer) {
        free(reply_ctx[i].buffer->addr);
        PS_CHECK_EQ(ibv_dereg_mr(reply_ctx[i].buffer), 0);
      }
    }
    FOR_QPS {
      rdma_destroy_qp(cm_ids[qpIndex]);
      PS_CHECK_EQ(rdma_destroy_id(cm_ids[qpIndex]), 0) << strerror(errno);
    }
  }

  void SetTransport(std::shared_ptr<Transport> t) { trans = t; }

  std::shared_ptr<Transport> GetTransport() { return trans; }

  inline bool GetAllStatus(ConnectionStatus s, bool eq = true) {
    bool ret = true;
    if (eq) {
      FOR_QPS { ret &= (s == status_list[qpIndex]); }
    } else {
      FOR_QPS { ret &= (s != status_list[qpIndex]); }
    }
    return ret;
  }

  inline void SetStatus(struct rdma_cm_id *id, ConnectionStatus s) {
    FOR_QPS {
      if (cm_ids[qpIndex] == id) {
        status_list[qpIndex] = s;
      }
    }
  }

  void Disconnect() {
    std::unique_lock<std::mutex> lk(connect_mu);
    FOR_QPS {
      PS_CHECK_EQ(rdma_disconnect(cm_ids[qpIndex]), 0) << strerror(errno);
    }
    cv.wait(lk, [this] {
      bool ret = true;
      FOR_QPS { ret &= (status_list[qpIndex] == IDLE); }
      return ret;
    });
    trans.reset();
  }

  void SetNodeID(int id) { node_id = id; }

  void InitSendContextHelper(struct ibv_pd *pd, WRContext *ctx,
                             ThreadsafeQueue<WRContext *> *queue, size_t num,
                             WRContextType type) {
    for (size_t i = 0; i < num; ++i) {
      void *buf;
      aligned_malloc(reinterpret_cast<void **>(&buf), kMempoolChunkSize);
      PS_CHECK(buf);
      struct ibv_mr *mr = ibv_reg_mr(pd, buf, kMempoolChunkSize, 0);
      PS_CHECK(mr)
          << "ibv_reg_mr failed: " << strerror(errno)
          << "\nYou can try to reduce BYTEPS_RDMA_START_DEPTH (current "
          << kStartDepth << ") or BYTEPS_RDMA_RX_DEPTH (current " << kRxDepth
          << ").";

      ctx[i].type = type;
      ctx[i].buffer = mr;
      ctx[i].private_data = this;
      queue->Push(&ctx[i]);
    }
  }

  void Init(struct ibv_cq *cq, struct ibv_pd *pd, rdma_cm_id *id = nullptr) {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_init_attr));
    attr.send_cq = cq;
    attr.recv_cq = cq;
    attr.cap.max_send_wr = kStartDepth + kReplyDepth;
    attr.cap.max_recv_wr = kRxDepth;
    attr.cap.max_send_sge = kSGEntry;
    attr.cap.max_recv_sge = kSGEntry;
    attr.cap.max_inline_data = 256;
    attr.qp_type = IBV_QPT_RC;
    attr.sq_sig_all = 0;
    PS_CHECK_EQ(rdma_create_qp(id, pd, &attr), 0)
        << "Create RDMA queue pair failed: " << strerror(errno);
    id->pd = pd;

    PS_LOG(TRACE) << "qp created: pd=" << pd << " , cq=" << cq
                  << ", qp=" << id->qp->qp_num;
    if (inited == 0) {
      InitSendContextHelper(pd, start_ctx, &free_start_ctx, kStartDepth,
                            kRendezvousStartContext);
      InitSendContextHelper(pd, reply_ctx, &free_reply_ctx, kReplyDepth,
                            kRendezvousReplyContext);
    }

    for (int i = 0; i < kRxDepth; ++i) {
      if (inited == 0) {
        void *buf;
        aligned_malloc(reinterpret_cast<void **>(&buf), kMempoolChunkSize);
        PS_CHECK(buf);
        struct ibv_mr *mr =
            ibv_reg_mr(pd, buf, kMempoolChunkSize, IBV_ACCESS_LOCAL_WRITE);
        PS_CHECK(mr)
            << "ibv_reg_mr failed: " << strerror(errno)
            << "\nYou can try to reduce BYTEPS_RDMA_START_DEPTH (default 128)"
            << " or BYTEPS_RDMA_RX_DEPTH (default 2048)";

        rx_ctx[i].type = kReceiveContext;
        rx_ctx[i].buffer = mr;
        rx_ctx[i].private_data = this;
      }
    }
    for (int i = 0; i < kRxDepth / QP_NUM; ++i) {
      if (inited < QP_NUM) {
        PostRecv(&rx_ctx[i + inited * QP_NUM], id);
      }
    }
    inited++;
  }

  void SetQPLag(int local_id, int rem_id) {
    auto qp_split_val = Environment::Get()->find("STEPMESH_SPLIT_QP_LAG");
    int val = qp_split_val ? atoi(qp_split_val) : -1;

    if (val == 1) {
      multi_qp_ = true;
      FOR_QPS {
        int lag = 1 + qpIndex % 2;
        int ret = wrap_mlx5dv_modify_qp_lag_port(cm_ids[qpIndex]->qp, lag);
        if (ret != 1) {
          PS_LOG(INFO) << "Failed to mlx5dv_modify_qp_lag_port qp ["
                       << cm_ids[qpIndex]->qp->qp_num << "] to port: " << lag
                       << ", qp type: " << cm_ids[qpIndex]->qp->qp_type;
        } else {
          uint8_t set_port = 0xff, act_port = 0xff;
          wrap_mlx5dv_query_qp_lag_port(cm_ids[qpIndex]->qp, &set_port,
                                        &act_port);
          PS_LOG(INFO) << "QP LAG Port: QP: " << cm_ids[qpIndex]->qp->qp_num
                       << ", Modify Port: " << lag
                       << ", Set to Port: " << static_cast<int>(set_port)
                       << ", Active Port: " << static_cast<int>(act_port);
        }
      }
    }
  }

  void PostRecv(WRContext *ctx, rdma_cm_id *id) {
    struct ibv_recv_wr wr, *bad_wr = nullptr;
    memset(&wr, 0, sizeof(wr));

    struct ibv_sge sge;
    sge.addr = reinterpret_cast<uint64_t>(ctx->buffer->addr);
    sge.length = kMempoolChunkSize;
    sge.lkey = ctx->buffer->lkey;

    wr.wr_id = reinterpret_cast<uint64_t>(ctx);
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    PS_CHECK_EQ(ibv_post_recv(id->qp, &wr, &bad_wr), 0)
        << "ibv_post_recv failed. qp_num:" << id->qp->qp_num
        << " err=" << strerror(errno);
  }
};

class Transport {
 public:
  virtual void RDMAWriteWithImm(MessageBuffer *msg_buf, uint64_t remote_addr,
                                uint32_t rkey, uint32_t idx,
                                bool inline_write = false,
                                struct ibv_send_wr *prev_wr = nullptr) = 0;

  virtual int RecvPushRequest(Message *msg, BufferContext *buffer_ctx,
                              int meta_len) = 0;
  virtual int RecvPullRequest(Message *msg, BufferContext *buffer_ctx,
                              int meta_len) = 0;
  virtual int RecvPushResponse(Message *msg, BufferContext *buffer_ctx,
                               int meta_len) = 0;
  virtual int RecvPullResponse(Message *msg, BufferContext *buffer_ctx,
                               int meta_len) = 0;

  virtual void Send(Message &msg, MessageBuffer *msg_buf,
                    RemoteTuple remote_tuple) = 0;
  virtual void SendPullRequest(Message &msg, MessageBuffer *msg_buf,
                               RemoteTuple remote_tuple) = 0;
  virtual void SendPushRequest(Message &msg, MessageBuffer *msg_buf,
                               RemoteTuple remote_tuple) = 0;
  virtual void SendPushResponse(Message &msg, MessageBuffer *msg_buf,
                                RemoteTuple remote_tuple) = 0;
  virtual void SendPullResponse(Message &msg, MessageBuffer *msg_buf,
                                RemoteTuple remote_tuple, size_t lkey) = 0;
  virtual void SendRendezvousBegin(Message &msg, MessageBuffer *msg_buf) = 0;
  virtual void SendRendezvousReply(RendezvousStart *req,
                                   AddressPool<BufferContext> &pool) = 0;
  virtual SArray<char> CreateFunctionalSarray(void *value, size_t size) = 0;

  virtual void RegisterRecvBuffer(Message &msg) {
    PS_CHECK(0) << "RegisterRecvBuffer is not implemented";
  }
};  // class Transport

class RDMATransport : public Transport {
 public:
  explicit RDMATransport(Endpoint *endpoint, MemoryAllocator *allocator,
                         Postoffice *postoffice) {
    endpoint_ = PS_CHECK_NOTNULL(endpoint);
    allocator_ = PS_CHECK_NOTNULL(allocator);
    pagesize_ = sysconf(_SC_PAGESIZE);

    postoffice_ = postoffice;
    is_server_ = postoffice_->is_server();
#ifdef STEPMESH_USE_GDR
    if (is_server_) {
      // Get the current GPU device ID directly from the CUDA runtime.
      // This relies on the launcher (e.g., mpirun) setting
      // CUDA_VISIBLE_DEVICES correctly for process-to-GPU affinity.
      int gpu_id = -1;
      Environment::Get()->find("STEPMESH_GPU", &gpu_id, gpu_id);
      PS_CHECK_GE(gpu_id, 0) << "failed to get gpu id, please set STEPMESH_GPU";
      Backend::Get()->SetDevice(gpu_id);
      mem_allocator_ =
          new BackendMemoryAllocator(endpoint_->cm_ids[0]->pd, gpu_id);
    } else {
      mem_allocator_ = nullptr;
    }
#else
    mem_allocator_ = nullptr;
#endif  // STEPMESH_USE_GDR
  }

  ~RDMATransport() {
    if (mem_allocator_) {
      delete mem_allocator_;
      mem_allocator_ = nullptr;
    }
  }

  virtual void RDMAWriteWithImm(MessageBuffer *msg_buf, uint64_t remote_addr,
                                uint32_t rkey, uint32_t idx,
                                bool inline_write = false,
                                struct ibv_send_wr *prev_wr = nullptr) {
    struct ibv_sge sge;
    sge.addr = reinterpret_cast<uint64_t>(msg_buf->inline_buf);
    sge.length = msg_buf->inline_len;
    sge.lkey = allocator_->LocalKey(msg_buf->inline_buf);

    struct ibv_send_wr wr = {}, *bad_wr = nullptr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = reinterpret_cast<uint64_t>(msg_buf);
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.next = nullptr;
    wr.imm_data = idx;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;

    if (inline_write) {
      wr.send_flags |= IBV_SEND_INLINE;
    }

    if (prev_wr == nullptr) {
      PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, &wr, &bad_wr), 0)
          << "ibv_post_send failed.";
    } else {
      prev_wr->next = &wr;
      PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, prev_wr, &bad_wr), 0)
          << "ibv_post_send failed.";
    }
  }

  void SendRendezvousBegin(Message &msg, MessageBuffer *msg_buf) {
    WRContext *context = nullptr;
    endpoint_->free_start_ctx.WaitAndPop(&context);

    RendezvousStart *req =
        reinterpret_cast<RendezvousStart *>(context->buffer->addr);
    req->meta_len = msg_buf->inline_len;
    req->origin_addr = reinterpret_cast<uint64_t>(msg_buf);
    req->data_num = msg_buf->data.size();
    req->key = msg.meta.key;
    for (size_t i = 0; i < req->data_num; ++i) {
      req->data_len[i] = msg.data[i].size();
    }

    struct ibv_sge sge;
    sge.addr = reinterpret_cast<uint64_t>(req);
    sge.lkey = context->buffer->lkey;
    sge.length = sizeof(RendezvousStart);

    struct ibv_send_wr wr, *bad_wr = nullptr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = reinterpret_cast<uint64_t>(context);
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    wr.next = nullptr;
    wr.imm_data = kRendezvousStart;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, &wr, &bad_wr), 0)
        << strerror(errno);
  }

  void SendRendezvousReply(RendezvousStart *req,
                           AddressPool<BufferContext> &addrpool) {
    BufferContext *buf_ctx = new BufferContext();
    buf_ctx->meta_len = req->meta_len;
    buf_ctx->data_num = req->data_num;

    size_t data_len = 0;

    for (size_t i = 0; i < req->data_num; ++i) {
      buf_ctx->data_len[i] = req->data_len[i];
      data_len += req->data_len[i];
    }

#ifdef STEPMESH_USE_GDR
    if (is_server_) {
      // Server-side GDR: Allocate separate CPU meta buffer and GPU data buffer
      PS_VLOG(2) << "SendRendezvousReply (GDR Server): meta_len="
                 << req->meta_len << ", data_len=" << data_len;
      // 1. Allocate CPU buffer for metadata
      buf_ctx->meta_buffer = allocator_->Alloc(req->meta_len);
      PS_CHECK(buf_ctx->meta_buffer)
          << "Failed to allocate CPU metadata buffer. Size: " << req->meta_len;

      uint32_t rkey = 0;
      if (data_len > 0) {
        // 2. Allocate GPU buffer for data
        size_t size = 0;
        postoffice_->van()->QueryRecvBuffer(req->key, this->endpoint_->node_id,
                                            &buf_ctx->gpu_data_buffer, &size,
                                            &rkey);
        if (buf_ctx->gpu_data_buffer == nullptr) {
          PS_LOG(INFO) << "Alloc new gpu buffer: key=" << req->key
                       << ", size=" << req->data_len[1];
          buf_ctx->gpu_data_buffer =
              mem_allocator_->Alloc(req->key, req->data_len[1]);
          rkey = mem_allocator_->GetRemoteKey(req->key);
        } else {
          PS_LOG(INFO) << "Using pre-registered gpu buffer: key=" << req->key
                       << ", size=" << req->data_len[1];
          PS_CHECK(size >= req->data_len[1]);
          PS_CHECK(rkey != 0);
        }

        PS_CHECK(buf_ctx->gpu_data_buffer)
            << "Failed to allocate GPU data buffer. Size: " << data_len;
      } else {
        buf_ctx->gpu_data_buffer = nullptr;
      }

      WRContext *reply_ctx_ptr = nullptr;
      endpoint_->free_reply_ctx.WaitAndPop(&reply_ctx_ptr);
      auto *resp =
          reinterpret_cast<RendezvousReply *>(reply_ctx_ptr->buffer->addr);

      // Populate reply with addresses and rkeys for both buffers
      resp->meta_addr = reinterpret_cast<uint64_t>(buf_ctx->meta_buffer);
      resp->meta_rkey = allocator_->RemoteKey(buf_ctx->meta_buffer);

      if (data_len > 0) {
        resp->data_addr = reinterpret_cast<uint64_t>(buf_ctx->gpu_data_buffer);
        resp->data_rkey = rkey;
      } else {
        resp->data_addr = 0;
        resp->data_rkey = 0;
      }
      resp->origin_addr = req->origin_addr;
      resp->idx = addrpool.StoreAddress(buf_ctx);

      PS_VLOG(2) << "GDR Server Reply: meta_addr=" << resp->meta_addr
                 << ", meta_rkey=" << resp->meta_rkey
                 << ", data_addr=" << resp->data_addr
                 << ", data_rkey=" << resp->data_rkey;

      // Send the reply
      struct ibv_sge sge;
      sge.addr = reinterpret_cast<uint64_t>(resp);
      sge.length = sizeof(RendezvousReply);
      sge.lkey = reply_ctx_ptr->buffer->lkey;
      struct ibv_send_wr wr, *bad_wr = nullptr;
      memset(&wr, 0, sizeof(wr));
      wr.wr_id = reinterpret_cast<uint64_t>(reply_ctx_ptr);
      wr.opcode = IBV_WR_SEND_WITH_IMM;
      wr.next = nullptr;
      wr.imm_data = kRendezvousReply;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, &wr, &bad_wr), 0)
          << "ibv_post_send failed.";

      return;  // Early return for server-side GDR path
    }
#endif  // STEPMESH_USE_GDR

    // Original logic for non-GDR or for client-side GDR (only receives meta)
    size_t buffer_size = is_server_
                             ? (align_ceil(req->meta_len, pagesize_) + data_len)
                             : req->meta_len;
    char *buffer = allocator_->Alloc(buffer_size);
    PS_CHECK(buffer);
    buf_ctx->buffer = buffer;

    WRContext *reply_ctx_ptr = nullptr;
    endpoint_->free_reply_ctx.WaitAndPop(&reply_ctx_ptr);
    RendezvousReply *resp =
        reinterpret_cast<RendezvousReply *>(reply_ctx_ptr->buffer->addr);

    // In GDR mode, client still uses single buffer logic,
    // so we populate the single addr/rkey
#ifdef STEPMESH_USE_GDR
    resp->meta_addr = reinterpret_cast<uint64_t>(buffer);
    resp->meta_rkey = allocator_->RemoteKey(buffer);
    resp->data_addr = 0;  // Not used by client
    resp->data_rkey = 0;  // Not used by client
#else
    resp->addr = reinterpret_cast<uint64_t>(buffer);
    resp->rkey = allocator_->RemoteKey(buffer);
#endif
    resp->origin_addr = req->origin_addr;
    resp->idx = addrpool.StoreAddress(buf_ctx);

    struct ibv_sge sge;
    sge.addr = reinterpret_cast<uint64_t>(resp);
    sge.length = sizeof(RendezvousReply);
    sge.lkey = reply_ctx_ptr->buffer->lkey;
    struct ibv_send_wr wr, *bad_wr = nullptr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = reinterpret_cast<uint64_t>(reply_ctx_ptr);
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    wr.next = nullptr;
    wr.imm_data = kRendezvousReply;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, &wr, &bad_wr), 0)
        << "ibv_post_send failed.";
  }

  void Send(Message &msg, MessageBuffer *msg_buf, RemoteTuple remote_tuple) {
#ifdef STEPMESH_USE_GDR
    auto raddr = std::get<0>(remote_tuple);
    auto rkey = std::get<1>(remote_tuple);
    auto idx = std::get<4>(remote_tuple);
#else
    auto raddr = std::get<0>(remote_tuple);
    auto rkey = std::get<1>(remote_tuple);
    auto idx = std::get<2>(remote_tuple);
#endif

    RDMAWriteWithImm(msg_buf, raddr, rkey, idx, true, nullptr);
  }

  void SendPushRequest(Message &msg, MessageBuffer *msg_buf,
                       RemoteTuple remote_tuple) {
    PS_CHECK_EQ(msg_buf->mrs.size(), 1);
    PS_CHECK(msg.data.size() >= 2);
    PS_CHECK(msg.data[1].size() <= msg_buf->mrs[0].second);

#ifdef STEPMESH_USE_GDR
    auto meta_raddr = std::get<0>(remote_tuple);
    auto meta_rkey = std::get<1>(remote_tuple);
    auto data_raddr = std::get<2>(remote_tuple);
    auto data_rkey = std::get<3>(remote_tuple);
    auto idx = std::get<4>(remote_tuple);
#else
    auto meta_raddr = std::get<0>(remote_tuple);
    auto meta_rkey = std::get<1>(remote_tuple);
    auto idx = std::get<2>(remote_tuple);
    auto data_raddr = meta_raddr + align_ceil(msg_buf->inline_len, pagesize_);
    auto data_rkey = meta_rkey;
#endif

    struct ibv_sge data_sge;
    struct ibv_send_wr data_wr, *bad_wr = nullptr;

    data_sge.addr = reinterpret_cast<uint64_t>(msg_buf->mrs[0].first->addr);
    data_sge.length = msg.data[1].size();
    data_sge.lkey = msg_buf->mrs[0].first->lkey;

    memset(&data_wr, 0, sizeof(data_wr));
    data_wr.wr_id = 0;  // No completion notification for this
    data_wr.opcode = IBV_WR_RDMA_WRITE;
    data_wr.next = nullptr;
    data_wr.sg_list = &data_sge;
    data_wr.num_sge = 1;
    data_wr.wr.rdma.rkey = data_rkey;
    data_wr.wr.rdma.remote_addr = data_raddr;

    if (endpoint_->multi_qp_) {
      uint32_t chunk_size = msg.data[1].size() / QP_NUM;
      data_sge.length = chunk_size + msg.data[1].size() % QP_NUM;

      // QP 0 Meta
      RawMeta *meta = reinterpret_cast<RawMeta *>(msg_buf->inline_buf);
      meta->slave_qp_num = QP_NUM - 1;
      FOR_QPS {
        endpoint_->qp_pkt_count[qpIndex]++;
        meta->slave_qp_counter[qpIndex] = endpoint_->qp_pkt_count[qpIndex];
        PS_LOG(ALL) << "Push Request, QP: "
                    << endpoint_->cm_ids[qpIndex]->qp->qp_num << "Counter"
                    << endpoint_->qp_pkt_count[qpIndex];
      }
      PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, &data_wr, &bad_wr), 0)
          << "ibv_post_send failed.";
      RDMAWriteWithImm(msg_buf, meta_raddr, meta_rkey, idx, true, nullptr);

      data_wr.wr.rdma.remote_addr += data_sge.length;
      data_sge.addr += data_sge.length;
      data_sge.length = chunk_size;
      uint32_t slave_idx = idx | 0x8000;

      for (int qpIndex = 1; qpIndex < QP_NUM; qpIndex++) {
        data_wr.wr_id = reinterpret_cast<uint64_t>(msg_buf);
        data_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        data_wr.imm_data = slave_idx;
        data_wr.send_flags = IBV_SEND_SIGNALED;

        PS_CHECK_EQ(
            ibv_post_send(endpoint_->cm_ids[qpIndex]->qp, &data_wr, &bad_wr), 0)
            << "ibv_post_send failed.";

        data_wr.wr.rdma.remote_addr += data_sge.length;
        data_sge.addr += data_sge.length;
      }
    } else {
      RDMAWriteWithImm(msg_buf, meta_raddr, meta_rkey, idx, true, &data_wr);
    }
  }

  void SendPullRequest(Message &msg, MessageBuffer *msg_buf,
                       RemoteTuple remote_tuple) {
    PS_CHECK_EQ(msg_buf->mrs.size(), 0);
    Send(msg, msg_buf, remote_tuple);
  }

  virtual void SendPushResponse(Message &msg, MessageBuffer *msg_buf,
                                RemoteTuple remote_tuple) {
    PS_CHECK_EQ(msg_buf->mrs.size(), 0);
    Send(msg, msg_buf, remote_tuple);
  }

  virtual void SendPullResponse(Message &msg, MessageBuffer *msg_buf,
                                RemoteTuple remote_tuple, size_t lkey) {
    PS_CHECK_EQ(msg_buf->mrs.size(), 0);

    auto data_raddr = msg.meta.addr;
    auto data_rkey = msg.meta.option;
    auto data_len = msg.meta.val_len;
    PS_CHECK_EQ((size_t)msg.meta.val_len, msg_buf->data[1].size());

    struct ibv_sge data_sge;
    data_sge.addr = reinterpret_cast<uint64_t>(msg_buf->data[1].data());
    data_sge.length = data_len;
    data_sge.lkey = lkey;
#ifdef STEPMESH_USE_GDR
    auto meta_raddr = std::get<0>(remote_tuple);
    auto meta_rkey = std::get<1>(remote_tuple);
    auto idx = std::get<4>(remote_tuple);
#else
    auto meta_raddr = std::get<0>(remote_tuple);
    auto meta_rkey = std::get<1>(remote_tuple);
    auto idx = std::get<2>(remote_tuple);
#endif

    // this rdma-write will not trigger any signal both remotely and locally
    struct ibv_send_wr data_wr, *bad_wr = nullptr;
    memset(&data_wr, 0, sizeof(data_wr));
    data_wr.wr_id = reinterpret_cast<uint64_t>(data_raddr);
    data_wr.opcode = IBV_WR_RDMA_WRITE;
    data_wr.next = nullptr;
    data_wr.sg_list = &data_sge;
    data_wr.num_sge = 1;
    data_wr.wr.rdma.remote_addr = data_raddr;
    data_wr.wr.rdma.rkey = data_rkey;

    if (endpoint_->multi_qp_) {
      uint32_t chunk_size = data_len / QP_NUM;
      data_sge.length = chunk_size + data_len % QP_NUM;
      RawMeta *meta = reinterpret_cast<RawMeta *>(msg_buf->inline_buf);
      meta->slave_qp_num = QP_NUM - 1;
      FOR_QPS {
        endpoint_->qp_pkt_count[qpIndex]++;
        meta->slave_qp_counter[qpIndex] = endpoint_->qp_pkt_count[qpIndex];
        PS_LOG(ALL) << "Push Request, QP: "
                    << endpoint_->cm_ids[qpIndex]->qp->qp_num << "Counter"
                    << endpoint_->qp_pkt_count[qpIndex];
      }
      PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, &data_wr, &bad_wr), 0)
          << "ibv_post_send failed.";
      RDMAWriteWithImm(msg_buf, meta_raddr, meta_rkey, idx, true, nullptr);

      data_wr.wr.rdma.remote_addr += data_sge.length;
      data_sge.addr += data_sge.length;
      data_sge.length = chunk_size;
      uint32_t slave_idx = idx | 0x8000;

      for (int qpIndex = 1; qpIndex < QP_NUM; qpIndex++) {
        data_wr.wr_id = reinterpret_cast<uint64_t>(msg_buf);
        data_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        data_wr.imm_data = slave_idx;
        data_wr.send_flags = IBV_SEND_SIGNALED;

        PS_CHECK_EQ(
            ibv_post_send(endpoint_->cm_ids[qpIndex]->qp, &data_wr, &bad_wr), 0)
            << "ibv_post_send failed.";

        data_wr.wr.rdma.remote_addr += data_sge.length;
        data_sge.addr += data_sge.length;
      }
    } else {
      PS_CHECK_EQ(ibv_post_send(endpoint_->cm_ids[0]->qp, &data_wr, &bad_wr), 0)
          << "ibv_post_send failed.";

      // after write keys/vals/lens (no imm), write the meta (with imm)
      RDMAWriteWithImm(msg_buf, meta_raddr, meta_rkey, idx, true, nullptr);
    }
  }

  virtual int RecvPushResponse(Message *msg, BufferContext *buffer_ctx,
                               int meta_len) {
    PS_CHECK_EQ(buffer_ctx->data_num, 0);
    return 0;
  }

  virtual int RecvPushResponseFast(Message *msg, BufferContext *buffer_ctx,
                                   MessageBuffer *msg_buf) {
    PS_CHECK_EQ(buffer_ctx->data_num, 0);
    RawMeta *raw = reinterpret_cast<RawMeta *>(msg_buf->inline_buf);
    // restore timestamp from local buffer
    msg->meta.timestamp = raw->timestamp;
    return 0;
  }

  virtual int RecvPullRequest(Message *msg, BufferContext *buffer_ctx,
                              int meta_len) {
    SArray<char> keys = CreateFunctionalSarray(&msg->meta.key, sizeof(Key));

    SArray<char> vals;  // add an empty sarray to pass kvapp check

    msg->data.push_back(keys);
    msg->data.push_back(vals);

    return keys.size() + vals.size();
  }

  virtual int RecvPushRequest(Message *msg, BufferContext *buffer_ctx,
                              int meta_len) {
    PS_CHECK(msg->meta.push && msg->meta.request);
#ifdef STEPMESH_USE_GDR
    if (is_server_) {
      PS_CHECK(buffer_ctx->data_num >= 2);
      // Data is in a separate GPU buffer
      char *cur_meta = buffer_ctx->meta_buffer;
      void *cur_data = buffer_ctx->gpu_data_buffer;
      PS_CHECK(cur_meta) << "Meta buffer is null in RecvPushRequest";
      PS_CHECK(cur_data) << "GPU data buffer is null in RecvPushRequest";

      // Meta is parsed from the original message, data from buffers
      SArray<char> keys = CreateFunctionalSarray(&msg->meta.key, sizeof(Key));

      uint32_t len = msg->meta.val_len;
      SArray<char> vals;

      vals.reset(reinterpret_cast<char *>(cur_data), len, [](void *){});

      SArray<char> lens =
          CreateFunctionalSarray(&msg->meta.val_len, sizeof(int));

      msg->data.push_back(keys);
      msg->data.push_back(vals);
      msg->data.push_back(lens);

      return keys.size() + vals.size() + lens.size();
    }
#endif  // STEPMESH_USE_GDR

    // Original non-GDR or client-side logic
    PS_CHECK_EQ(buffer_ctx->data_num, 3);
    char *cur = buffer_ctx->buffer +
                align_ceil(static_cast<size_t>(meta_len), pagesize_);

    SArray<char> keys = CreateFunctionalSarray(&msg->meta.key, sizeof(Key));
    SArray<char> lens = CreateFunctionalSarray(&msg->meta.val_len, sizeof(int));

    uint32_t len = msg->meta.val_len;
    SArray<char> vals;
    vals.reset(cur, len, [](void *) {});  // no need to delete

    msg->data.push_back(keys);
    msg->data.push_back(vals);
    msg->data.push_back(lens);

    return keys.size() + vals.size() + lens.size();
  }

  virtual int RecvPullResponse(Message *msg, BufferContext *buffer_ctx,
                               int meta_len) {
    SArray<char> keys = CreateFunctionalSarray(&msg->meta.key, sizeof(Key));
    SArray<char> lens = CreateFunctionalSarray(&msg->meta.val_len, sizeof(int));

    SArray<char> vals;
    auto addr = msg->meta.addr;
    vals.reset(reinterpret_cast<char *>(addr), msg->meta.val_len,
               [](void *) {});

    msg->data.push_back(keys);
    msg->data.push_back(vals);
    msg->data.push_back(lens);

    return keys.size() + vals.size() + lens.size();
  }

  virtual int RecvPullResponseFast(Message *msg, BufferContext *buffer_ctx,
                                   int meta_len, MessageBuffer *msg_buf) {
    RawMeta *raw = reinterpret_cast<RawMeta *>(msg_buf->inline_buf);
    // restore timestamp from local buffer
    msg->meta.timestamp = raw->timestamp;
    return RecvPullResponse(msg, buffer_ctx, meta_len);
  }

  SArray<char> CreateFunctionalSarray(void *value, size_t size) {
    SArray<char> sarr;
    void *p = malloc(size);
    memcpy(p, value, size);
    sarr.reset(reinterpret_cast<char *>(p), size, [p](void *) { free(p); });
    return sarr;
  }

  bool is_server_;

 protected:
  size_t pagesize_ = 4096;
  Endpoint *endpoint_;
  MemoryAllocator *allocator_;
  BackendMemoryAllocator *mem_allocator_;

  Postoffice *postoffice_;
};  // class Transport

class IPCTransport : public RDMATransport {
 public:
  explicit IPCTransport(Endpoint *endpoint, MemoryAllocator *allocator,
                        Postoffice *postoffice)
      : RDMATransport(endpoint, allocator, postoffice) {
    auto val = Environment::Get()->find("BYTEPS_IPC_COPY_NUM_THREADS");
    ipc_copy_nthreads_ = val ? atoi(val) : 4;
    for (int i = 0; i < ipc_copy_nthreads_; ++i) {
      auto q = new ThreadsafeQueue<AsyncCopy>;
      async_copy_queue_.push_back(q);
    }
    for (int i = 0; i < ipc_copy_nthreads_; ++i) {
      auto t = new std::thread(&IPCTransport::AsyncCopyThread, this, i);
      ipc_copy_thread_list_.push_back(t);
    }
    val = Environment::Get()->find("BYTEPS_PARTITION_BYTES");
    byteps_partition_bytes_ = val ? atoi(val) : 4096000;

    val = Environment::Get()->find("BYTEPS_ENCODING_SCHEME_VERSION");
    encoding_scheme_version_ = val ? atoi(val) : 0;

    val = Environment::Get()->find("BYTEPS_LOCAL_SIZE");
    auto byteps_local_size = val ? atoi(val) : 8;
    byteps_partition_bytes_ = RoundUp(
        byteps_partition_bytes_, byteps_local_size * sysconf(_SC_PAGESIZE));

    val = Environment::Get()->find("BYTEPS_IPC_ENABLE_ASYNC_COPY");
    enable_async_copy_ = val ? atoi(val) : 1;  // default enabled
    if (!enable_async_copy_)
      PS_LOG(INFO)
          << "Async copy has been disabled, this could affect the performance";

    val = Environment::Get()->find("BYTEPS_PCIE_SWITCH_SIZE");
    auto byteps_nccl_pcie_size = val ? atoi(val) : 8;
    if (byteps_local_size % byteps_nccl_pcie_size != 0) {
      // local_size < pcie_size or unbalance PCIe switches
      byteps_nccl_pcie_size = byteps_local_size;
    }
    // ensure this name corresponds with that in
    // BytePSSharedMemory::openPcieSharedMemory()
    if (byteps_local_size > byteps_nccl_pcie_size) {
      // cross pcie switch, use the last pcie cpu buffer
      auto byteps_pcie_num = byteps_local_size / byteps_nccl_pcie_size;
      shm_prefix_ =
          kShmPciePrefix + std::to_string(byteps_pcie_num - 1) + "_Shm_";
    } else {
      shm_prefix_ = kShmPrefix;
    }
    val = Environment::Get()->find("BYTEPS_JOB_ID");
    std::string _job_id = val ? std::string(val) : "0";
    shm_prefix_ = shm_prefix_ + _job_id + "_";
  }

  ~IPCTransport() {
    for (size_t i = 0; i < ipc_copy_thread_list_.size(); ++i) {
      AsyncCopy m;
      m.shutdown = true;
      async_copy_queue_[i]->Push(m);
      ipc_copy_thread_list_[i]->join();
    }
  }

  void SendPushRequest(Message &msg, MessageBuffer *msg_buf,
                       RemoteTuple remote_tuple) {
    Send(msg, msg_buf, remote_tuple);
  }

  void SendPullResponse(Message &msg, MessageBuffer *msg_buf,
                        RemoteTuple remote_tuple, size_t lkey) {
    auto addr = reinterpret_cast<void *>(PS_CHECK_NOTNULL(msg.data[1].data()));
    void *shm_addr =
        PS_CHECK_NOTNULL(GetSharedMemory(shm_prefix_, msg.meta.key));

    if (enable_async_copy_) {
      // async copy with a simple load-balancing strategy
      AsyncCopy m = {msg_buf, remote_tuple,     shm_addr,
                     addr,    msg.meta.val_len, false};
      auto cnt = cpy_counter_.fetch_add(1);
      async_copy_queue_[cnt % ipc_copy_nthreads_]->Push(m);
    } else {
      // synchronous copy
      memcpy(shm_addr, addr, msg.meta.val_len);
      Send(msg, msg_buf, remote_tuple);
    }
  }

  int RecvPushRequest(Message *msg, BufferContext *buffer_ctx, int meta_len) {
    // get data message from local shared memory
    auto key = msg->meta.key;
    auto len = msg->meta.val_len;

    SArray<char> keys = CreateFunctionalSarray(&msg->meta.key, sizeof(Key));

    SArray<char> vals;
    void *addr = GetSharedMemory(shm_prefix_, key);
    vals.reset(reinterpret_cast<char *>(addr), len, [](void *) {});

    SArray<char> lens = CreateFunctionalSarray(&msg->meta.val_len, sizeof(int));

    msg->data.push_back(keys);
    msg->data.push_back(vals);
    msg->data.push_back(lens);

    return keys.size() + vals.size() + lens.size();
  }

 private:
  struct AsyncCopy {
    MessageBuffer *msg_buf;
    RemoteTuple remote_tuple;
    void *dst;
    void *src;
    int len;
    bool shutdown;
  };

  void AsyncCopyThread(int i) {
    auto &q = async_copy_queue_[i];
    while (true) {
      AsyncCopy m;
      q->WaitAndPop(&m);
      if (m.shutdown) break;
      if (m.len == 0) continue;

      // TODO(none): use parallel copy
      PS_CHECK(m.dst);
      PS_CHECK(m.src);
      memcpy(m.dst, m.src, m.len);

      auto raddr = std::get<0>(m.remote_tuple);
      auto rkey = std::get<1>(m.remote_tuple);
      auto idx = std::get<2>(m.remote_tuple);

      RDMAWriteWithImm(m.msg_buf, raddr, rkey, idx);
    }
  }

  void *GetSharedMemory(const std::string &prefix, uint64_t key) {
    std::lock_guard<std::mutex> lock(shm_mu_);
    auto worker_key = DecodeWorkerKey(key);
    auto seq_num = worker_key % (1 << 16);
    // Total key space is [0, 2^64 - 1]
    // It will be divided to N PS servers, for now we assume N <= 2^16
    // Then we have 2^48 key space left.
    // Encoding scheme version 0:
    //   Then we have 2^48 key space left (top 16 bits for different servers)
    //   MXNet server has a bug dealing with keys larger than 2^32
    //   Below we support up to 2^16 tensors, and up to 2^16 partitions per
    //   tensor
    // Encoding scheme version 1:
    //   Top 16 bits out of the 48 bits encodes the sender rank
    //   Mid 16 bits out of the 48 bits encodes the tensor id
    //   The next 6 bits encodes request types (pushpull, send, etc)
    //   The last 10 bits encodes the partition id
    //   Therefore, we support up to 2^16 tensors, and up to 2^10 partitions per
    //   tensor
    if (encoding_scheme_version_ == 1) {
      seq_num = worker_key % (1 << 10);
    }
    auto base_key = worker_key - seq_num;
    uint64_t offset = byteps_partition_bytes_ * seq_num;
    if (key_shm_addr_.find(base_key) != key_shm_addr_.end()) {
      return reinterpret_cast<void *>(
          reinterpret_cast<char *>(key_shm_addr_[base_key]) + offset);
    }
    std::string shm_name(prefix);
    std::stringstream stream;
    stream << std::hex << base_key;

    shm_name += stream.str();
    int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
    PS_CHECK_GE(shm_fd, 0) << "shm_open failed for " << shm_name << ", "
                           << strerror(errno);

    struct stat sb;
    PS_CHECK_EQ(0, fstat(shm_fd, &sb)) << strerror(errno);
    auto total_shm_size = sb.st_size;

    void *base_ptr =
        mmap(0, total_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    PS_CHECK_NE(base_ptr, (void *)-1) << strerror(errno);
    key_shm_addr_[base_key] = base_ptr;

    PS_VLOG(1) << "open Shared Memory: " << shm_name << " offset=" << offset
               << " (in bytes) size=" << total_shm_size;
    return reinterpret_cast<void *>(
        reinterpret_cast<char *>(key_shm_addr_[base_key]) + offset);
  }

  int ipc_copy_nthreads_;
  std::vector<std::thread *> ipc_copy_thread_list_;
  std::vector<ThreadsafeQueue<AsyncCopy> *> async_copy_queue_;
  std::atomic<uint64_t> cpy_counter_{0};

  int byteps_partition_bytes_ = 4096000;

  std::string shm_prefix_;

  std::mutex shm_mu_;
  std::unordered_map<uint64_t, void *> key_shm_addr_;

  bool enable_async_copy_;
  int encoding_scheme_version_ = 0;
};  // class IPCTransport

}  // namespace ps

#endif  // DMLC_USE_RDMA
#endif  // RDMA_TRANSPORT_H_
