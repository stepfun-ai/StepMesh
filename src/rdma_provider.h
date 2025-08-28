// Copyright 2025 Alibaba Group. or its affiliates. All Rights Reserved.
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

#ifndef RDMA_PROVIDER_H_
#define RDMA_PROVIDER_H_

#ifdef DMLC_USE_RDMA

#include <rdma/rdma_cma.h>

#include <string>

#include "./ibvwarp.h"
#include "dmlc/logging.h"

namespace ps {

class RDMAProvider {
 public:
  static RDMAProvider *GetProvider(struct ibv_context *context);

  virtual int InlineSize() = 0;
  virtual int SetQPLag(struct ibv_qp *qp, int port_num) = 0;
  virtual int ErrIgnore(struct ibv_wc *wc) = 0;

 protected:
  RDMAProvider() {}
};

class eRDMAProvider : public RDMAProvider {
 public:
  static inline RDMAProvider *Get() {
    static RDMAProvider *inst_ptr = new eRDMAProvider();
    return inst_ptr;
  }

  static inline const char *DevPrefix() { return "erdma"; }
  int InlineSize() override { return 96; }
  int SetQPLag(struct ibv_qp *qp, int port_num) override { return 1; }
  int ErrIgnore(struct ibv_wc *wc) override {
    // When destroy qp, qp will modify to error and then destroyed.
    // A cqe with IBV_WC_WR_FLUSH_ERR can be polled because of the qp in
    // error state with posted rqes to notify release resources of rqes.
    // Because the resources of rqes has been released before destroy qp,
    // so the IBV_WC_WR_FLUSH_ERR can be ignored.
    if (wc->status == IBV_WC_SUCCESS ||
        (wc->status == IBV_WC_WR_FLUSH_ERR && !wc->vendor_err)) {
      return 1;
    }
    return 0;
  }

 private:
  eRDMAProvider() {}
};

class Mlx5Provider : public RDMAProvider {
 public:
  static inline RDMAProvider *Get() {
    static RDMAProvider *inst_ptr = new Mlx5Provider();
    return inst_ptr;
  }

  static inline const char *DevPrefix() { return "mlx5"; }
  int InlineSize() override { return 512; }

  int SetQPLag(struct ibv_qp *qp, int port_num) override {
    int ret = wrap_mlx5dv_modify_qp_lag_port(qp, port_num);
    if (ret == 1) {
      uint8_t set_port = 0xff, act_port = 0xff;
      wrap_mlx5dv_query_qp_lag_port(qp, &set_port, &act_port);
      PS_LOG(INFO) << "QP LAG Port: QP: " << qp->qp_num
                   << ", Modify Port: " << port_num
                   << ", Set to Port: " << static_cast<int>(set_port)
                   << ", Active Port: " << static_cast<int>(act_port);
    }
    return ret;
  }

  int ErrIgnore(struct ibv_wc *wc) override {
    return wc->status == IBV_WC_SUCCESS;
  }

 private:
  Mlx5Provider() {
    if (wrap_ibv_symbols() != 1) {
      PS_LOG(WARNING) << "Load mlx5 symbols fails.";
    }
  }
};

RDMAProvider *RDMAProvider::GetProvider(struct ibv_context *context) {
  const char *dev_name = ibv_get_device_name(context->device);
  if (strstr(dev_name, eRDMAProvider::DevPrefix())) {
    return eRDMAProvider::Get();
  } else {
    if (!strstr(dev_name, Mlx5Provider::DevPrefix())) {
      PS_LOG(WARNING) << "rdma device(" << dev_name
                      << ") with unknow provider, use mlx5 as default, maybe "
                         "not compatible.";
    }
    return Mlx5Provider::Get();
  }
}

}  // namespace ps

#endif  // DMLC_USE_RDMA
#endif  // RDMA_PROVIDER_H_
