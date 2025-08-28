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

#ifndef RDMA_UTILS_H_
#define RDMA_UTILS_H_

#ifdef DMLC_USE_RDMA

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <rdma/rdma_cma.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef DMLC_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "./van_common.h"
#include "ps/internal/threadsafe_queue.h"
#include "ps/internal/van.h"

namespace ps {

#define DIVUP(x, y) (((x) + (y)-1) / (y))
#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

static const int kSGEntry = 1;
static const int kTimeoutms = 1000;
static const int kRdmaListenBacklog = 128;
static const int kMaxHostnameLength = 16;
static const int kRdmaMaxWRs = 12;

// should have the same prefix with BytePS shared memory
// for pcie reduce:  BytePS_Pcie_{pcie_id}_ShM_{JOB_ID}_{BYTEPS_KEY}
// otherwise:        BytePS_ShM_{JOB_ID}_{BYTEPS_KEY}
static const std::string kShmPrefix("BytePS_ShM_");       // NOLINT
static const std::string kShmPciePrefix("BytePS_Pcie_");  // NOLINT

enum WRContextType {
  kRendezvousStartContext,
  kRendezvousReplyContext,
  kWriteContext,
  kReceiveContext
};

class MemoryAllocator {
 public:
  explicit MemoryAllocator(struct ibv_pd *pd) {
    std::lock_guard<std::mutex> lk(mu_);
    pd_ = pd;
  }

  ~MemoryAllocator() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &it : mr_) {
      PS_CHECK_EQ(ibv_dereg_mr(it.second), 0);
      free(it.first);
    }
  }

  char *Alloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }

    // align to page size (usually 4KB)
    size = align_ceil(size, pagesize_);

    char *p;
    aligned_malloc(reinterpret_cast<void **>(&p), size);
    PS_CHECK(p);

    struct ibv_mr *mr = nullptr;
    mr = ibv_reg_mr(pd_, p, size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    PS_CHECK(mr != nullptr);

    std::lock_guard<std::mutex> lk(mu_);
    mr_[p] = mr;
    used_list.emplace(p, size);

    return p;
  }

  uint32_t LocalKey(char *addr) { return Addr2MR(addr)->lkey; }

  uint32_t RemoteKey(char *addr) { return Addr2MR(addr)->rkey; }

  struct ibv_pd *GetPD() {
    return pd_;
  }

 private:
  // convert the memory address to its associated RDMA memory region
  inline struct ibv_mr *Addr2MR(char *addr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = mr_.find(addr);
    PS_CHECK_NE(it, mr_.end()) << "cannot find the associated memory region";

    return it->second;
  }

  std::mutex mu_;
  struct ibv_pd *pd_;
  size_t pagesize_ = sysconf(_SC_PAGESIZE);
  std::unordered_map<char *, size_t> used_list;
  std::unordered_map<char *, struct ibv_mr *> mr_;
};

class BackendMemoryAllocator {
 public:
  explicit BackendMemoryAllocator(struct ibv_pd *pd, int gpu_id)
      : pd_(pd), associated_gpu_id_(gpu_id) {
    PS_CHECK(pd_) << "Protection Domain (pd_) is null.";
    PS_LOG(INFO) << "Initialized BackendMemoryAllocator for GPU " << gpu_id
                 << " with pd " << pd_;
  }

  ~BackendMemoryAllocator() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto const &it : key_to_mr_) {
      auto mr = it.second;
      if (mr) {
        if (ibv_dereg_mr(mr) != 0) {
          PS_LOG(WARNING) << "ibv_dereg_mr failed for GPU MR " << mr->handle
                          << " on GPU " << associated_gpu_id_ << ": "
                          << strerror(errno);
        }
        if (mr->addr) {
          Backend::Get()->Free(mr->addr);
        }
      }
    }
    key_to_mr_.clear();
  }

  void *Alloc(uint64_t key, size_t requested_size) {
    std::lock_guard<std::mutex> lock(mu_);
    Backend::Get()->SetDevice(associated_gpu_id_);
    auto it = key_to_mr_.find(key);
    if (it != key_to_mr_.end()) {
      PS_CHECK_GE(it->second->length, requested_size)
          << "Existing buffer for key " << key << " is too small. "
          << "Existing size: " << it->second->length
          << ", requested size: " << requested_size;
      return it->second->addr;
    }

    void *ptr = Backend::Get()->Alloc(requested_size);

    /*cudaError_t cuda_err = cudaMalloc(&ptr, requested_size);
    if (cuda_err != cudaSuccess) {
      PS_LOG(FATAL) << "cudaMalloc failed for GPU " << associated_gpu_id_
                 << " size " << requested_size << " for key " << key << ": "
                 << cudaGetErrorString(cuda_err);
    }*/

    struct ibv_mr *mr =
        ibv_reg_mr(pd_, ptr, requested_size,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
      Backend::Get()->Free(ptr);
      PS_LOG(FATAL) << "ibv_reg_mr failed for memory on GPU "
                    << associated_gpu_id_ << " for key " << key << " size "
                    << requested_size << ": " << strerror(errno);
    }

    key_to_mr_.emplace(key, mr);
    PS_VLOG(2) << "Allocated new GPU buffer for key=" << key << " addr=" << ptr
               << " size=" << requested_size;

    return ptr;
  }

  uint32_t GetRemoteKey(uint64_t key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = key_to_mr_.find(key);
    if (it == key_to_mr_.end()) {
      PS_LOG(ERROR) << "Key " << key
                    << " not found in allocator for GetRemoteKey.";
      return 0;
    }
    PS_CHECK(it->second) << "MR is null for GPU buffer with key " << key;
    return it->second->rkey;
  }

 private:
  struct ibv_pd *pd_ = nullptr;
  std::mutex mu_;
  int associated_gpu_id_ = -1;

  std::unordered_map<uint64_t, struct ibv_mr *> key_to_mr_;
};

struct WRContext {
  WRContextType type;
  void *buffer;
  struct ibv_mr *ref_mr;
  void *private_data;
};

struct RendezvousStart {
  uint64_t meta_len;
  uint64_t data_num;
  uint64_t data_len[kMaxDataFields];
  uint64_t origin_addr;
  uint64_t key;
};

struct RendezvousReply {
#ifdef STEPMESH_USE_GDR
  uint64_t meta_addr;
  uint32_t meta_rkey;
  uint64_t data_addr;
  uint32_t data_rkey;
#else
  uint64_t addr;
  uint32_t rkey;
#endif  // STEPMESH_USE_GDR
  uint64_t origin_addr;
  uint32_t idx;
};

struct BufferContext {
  char *buffer = nullptr;  // Original buffer, for non-GDR or client-side GDR
#ifdef STEPMESH_USE_GDR
  char *meta_buffer = nullptr;      // For server-side GDR meta
  void *gpu_data_buffer = nullptr;  // For server-side GDR data
#endif
  int meta_len = 0;
  size_t data_num = 0;
  size_t data_len[kMaxDataFields] = {0};
};

typedef std::unique_ptr<struct ibv_mr, std::function<void(struct ibv_mr *)>>
    MRPtr;

struct MessageBuffer {
  size_t inline_len;
  char *inline_buf;
  std::vector<SArray<char>> data;
  std::vector<std::pair<MRPtr, size_t>> mrs;
};

struct RequestContext {
  uint32_t node;
  uint16_t port;
  char hostname[kMaxHostnameLength];
};

// <remote_addr, rkey, idx, local_addr>
#ifdef STEPMESH_USE_GDR
// <meta_addr, meta_rkey, data_addr, data_rkey, idx, local_addr>
typedef std::tuple<uint64_t, uint32_t, uint64_t, uint32_t, uint32_t,
                   MessageBuffer *>
    RemoteTuple;
#else
// <remote_addr, rkey, idx, local_addr>
typedef std::tuple<uint64_t, uint32_t, uint32_t, MessageBuffer *> RemoteTuple;
#endif  // STEPMESH_USE_GDR

// recver, <remote_addr, rkey, idx>
typedef std::unordered_map<int, RemoteTuple> RemoteAndLocalAddress;

static_assert(std::is_pod<RendezvousStart>::value,
              "RendezvousStart must be a POD type.");
static_assert(std::is_pod<RendezvousReply>::value,
              "RendezvousReply must be a POD type.");
static_assert(std::is_pod<RequestContext>::value,
              "RequestContext must be a POD type.");

static const size_t kMempoolChunkSize =
    std::max({sizeof(RendezvousStart), sizeof(RendezvousReply)});

uint64_t DecodeWorkerKey(uint64_t key) {
  auto kr = ps::Postoffice::GetServer()
                ->GetServerKeyRanges()[ps::Postoffice::GetServer()->my_rank()];
  return key - kr.begin();
}

int AlignTo(int input, int alignment) { return input / alignment * alignment; }
int DivUp(int x, int y) { return (x + y - 1) / y; }
int RoundUp(int x, int y) { return DivUp(x, y) * y; }

};  // namespace ps

#endif  // DMLC_USE_RDMA
#endif  // RDMA_UTILS_H_
