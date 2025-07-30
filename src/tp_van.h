#ifndef PS_TP_VAN_H_
#define PS_TP_VAN_H_

#ifdef DMLC_USE_TP

#include <stdio.h>
#include <stdlib.h>
#include <tensorpipe/tensorpipe.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "ps/internal/threadsafe_queue.h"
#include "ps/internal/van.h"

namespace ps {
/**
 * \brief TensorPipe based implementation,
 *   mostly adapted from the RPC implementation of PyTorch and DGL.
 *   Ref:
 * https://github.com/pytorch/pytorch/tree/master/torch/csrc/distributed/rpc
 *        https://github.com/dmlc/dgl/tree/master/src/rpc
 */
class TPVan : public Van {
 public:
  TPVan(Postoffice *postoffice) : Van(postoffice) {}
  virtual ~TPVan() {}
  virtual std::string GetType() const { return std::string("tensorpipe"); }

 public:
  void Start(int customer_id, bool standalone) {
    start_mu_.lock();
    PS_CHECK(!standalone);
    queue_ = std::make_shared<ThreadsafeQueue<Message>>();
    start_mu_.unlock();
    Van::Start(customer_id, standalone);
  }

  void Stop() override {
    PS_VLOG(1) << my_node_.ShortDebugString() << " is stopping";
    Van::Stop();
    if (listener_) {
      listener_->close();
      listener_.reset();
    }
    for (auto it : send_pipes_) {
      if (it.second) {
        it.second->close();
      }
    }
    send_pipes_.clear();
    if (send_ctx_) {
      send_ctx_->join();
      send_ctx_.reset();
    }
    if (recv_ctx_) {
      recv_ctx_->join();
      recv_ctx_.reset();
    }
  }

  int Bind(Node &node, int max_retry) override {
    PS_CHECK(!node.hostname.empty()) << "Empty hostname";
    PS_CHECK(!send_ctx_ && !recv_ctx_);
    send_ctx_ = InitContext(node);
    recv_ctx_ = InitContext(node);
    std::string addr =
        "tcp://" + node.hostname + ":" + std::to_string(node.port);
    auto use_recv_ctx = Environment::Get()->find("DMLC_USE_RECVCTX");
    if (use_recv_ctx) {
      recv_ctx_ = InitContext(node);
      listener_ = recv_ctx_->listen({addr});
    } else {
      listener_ = send_ctx_->listen({addr});
    }
    listener_->accept([this](const tensorpipe::Error &error,
                             std::shared_ptr<tensorpipe::Pipe> pipe) {
      OnAccepted(error, pipe);
    });
    return node.port;
  }

  void Connect(const Node &node) override {
    PS_CHECK_NE(node.id, Node::kEmpty);
    PS_CHECK_NE(node.port, Node::kEmpty);
    PS_CHECK(node.hostname.size());
    int recv_id = node.id;
    if (send_pipes_.find(recv_id) != send_pipes_.end()) {
      return;
    }
    // worker doesn't need to connect to the other workers. same for server
    if ((node.role == my_node_.role) && (node.id != my_node_.id)) {
      return;
    }
    // connect
    std::string addr =
        "tcp://" + node.hostname + ":" + std::to_string(node.port);
    bool connected = false;
    while (!connected) {
      std::shared_ptr<tensorpipe::Pipe> pipe = send_ctx_->connect(addr);
      tensorpipe::Message tpmsg;
      tpmsg.metadata = "ps-lite";
      auto done = std::make_shared<std::promise<bool>>();
      pipe->write(tpmsg, [done](const tensorpipe::Error &error) {
        done->set_value(!error);
      });
      connected = done->get_future().get();
      if (connected) {
        send_pipes_[recv_id] = pipe;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    }
    return;
  }

  int SendMsg(Message &msg) override {
    std::lock_guard<std::mutex> lk(mu_);
    int recv_id = msg.meta.recver;
    PS_CHECK_NE(recv_id, Node::kEmpty);
    auto it = send_pipes_.find(recv_id);
    if (it == send_pipes_.end()) {
      return -1;
    }
    auto pipe = it->second;
    // send meta
    tensorpipe::Message tp_msg;
    tp_msg.metadata.append(reinterpret_cast<char *>(&my_node_.id), sizeof(int));
    int meta_size;
    char *meta_buf = nullptr;
    PackMeta(msg.meta, &meta_buf, &meta_size);
    tp_msg.metadata.append(reinterpret_cast<char *>(&meta_size), sizeof(int));
    tp_msg.metadata.append(meta_buf, meta_size);
    delete[] meta_buf;
    int data_size = msg.data.size();
    tp_msg.metadata.append(reinterpret_cast<char *>(&data_size), sizeof(int));
    int send_bytes = tp_msg.metadata.size();
    auto sarray_holder = std::make_shared<std::vector<SArray<char>>>();
    if (data_size) {
      int small_data_size = 0;
      for (int i = 0; i < msg.data.size(); i++) {
        if (msg.data[i].size() <= 16) small_data_size += 1;
      }
      tp_msg.metadata.append(reinterpret_cast<char *>(&small_data_size),
                             sizeof(int));
      for (int i = 0; i < msg.data.size(); i++) {
        auto sarray = msg.data[i];
        sarray_holder->push_back(sarray);
        int size = sarray.size();
        if (size <= 16) {
          tp_msg.metadata.append(reinterpret_cast<char *>(&i), sizeof(int));
          tp_msg.metadata.append(reinterpret_cast<char *>(&size), sizeof(int));
          if (size > 0) tp_msg.metadata.append(msg.data[i].data(), size);
          continue;
        }
        tensorpipe::CpuBuffer cpu_buffer;
        cpu_buffer.ptr = static_cast<void *>(sarray.data());
        tensorpipe::Message::Tensor tensor{.buffer = cpu_buffer,
                                           .length = sarray.size()};
        tp_msg.tensors.push_back(tensor);
        send_bytes += sarray.size();
      }
    }
    pipe->write(tp_msg,
                [sarray_holder, recv_id](const tensorpipe::Error &error) {
                  if (error) {
                    PS_VLOG(1) << "Failed to send message to " << recv_id
                               << ". Details: " << error.what();
                  }
                });
    return send_bytes;
  }

  int RecvMsg(Message *msg) override {
    queue_->WaitAndPop(msg);
    return msg->meta.data_size;
  }

 private:
  void OnAccepted(const tensorpipe::Error &error,
                  std::shared_ptr<tensorpipe::Pipe> pipe) {
    if (error) {
      if (error.isOfType<tensorpipe::ListenerClosedError>()) {
        // Expected.
      } else {
        // EOF error is also expected
        PS_VLOG(1) << "An error when reading from a pipe: " << error.what();
      }
      return;
    }
    // Accept the next connection request
    listener_->accept([this](const tensorpipe::Error &error,
                             std::shared_ptr<tensorpipe::Pipe> pipe) {
      OnAccepted(error, pipe);
    });

    pipe->readDescriptor([pipe, this](const tensorpipe::Error &error,
                                      tensorpipe::Descriptor descriptor) {
      if (error) {
        if (error.isOfType<tensorpipe::PipeClosedError>()) {
          // Expected.
        } else {
          // EOF error is also expected
          PS_VLOG(1) << "An error when reading from a pipe: " << error.what();
        }
        return;
      }
      PS_CHECK(descriptor.metadata == "ps-lite") << "Invalid connect message.";
      tensorpipe::Allocation allocation;
      pipe->read(allocation, [](const tensorpipe::Error &error) {});
      ReceiveFromPipe(pipe);
    });
  }

  void ReceiveFromPipe(std::shared_ptr<tensorpipe::Pipe> pipe) {
    pipe->readDescriptor([pipe, this](const tensorpipe::Error &error,
                                      tensorpipe::Descriptor descriptor) {
      if (error) {
        if (error.isOfType<tensorpipe::PipeClosedError>()) {
          // Expected.
        } else {
          // EOF error is also expected
          PS_VLOG(1) << "An error when reading from a pipe: " << error.what();
        }
        return;
      }
      tensorpipe::Allocation allocation;
      PS_CHECK_EQ(descriptor.payloads.size(), 0) << "Invalid Message";
      int num_sarray = descriptor.tensors.size();
      if (num_sarray > 0) {
        allocation.tensors.resize(num_sarray);
        for (size_t i = 0; i < descriptor.tensors.size(); i++) {
          tensorpipe::CpuBuffer cpu_buffer;
          cpu_buffer.ptr = new char[descriptor.tensors[i].length];
          allocation.tensors[i].buffer = cpu_buffer;
        }
      }
      pipe->read(allocation, [allocation, descriptor = std::move(descriptor),
                              pipe, this](const tensorpipe::Error &error) {
        if (error) {
          // EOF error is expected
          PS_VLOG(1) << "An error when reading from a pipe: " << error.what();
          return;
        }
        Message msg;
        msg.meta.recver = my_node_.id;
        char *ptr = const_cast<char *>(descriptor.metadata.data());
        msg.meta.sender = *reinterpret_cast<int *>(ptr);
        ptr += sizeof(int);
        int meta_size = *reinterpret_cast<int *>(ptr);
        ptr += sizeof(int);
        UnpackMeta(ptr, meta_size, &(msg.meta));
        ptr += meta_size;
        int data_size = *reinterpret_cast<int *>(ptr);
        ptr += sizeof(int);
        if (data_size > 0) {
          msg.data.resize(data_size);
          int small_data_size = *reinterpret_cast<int *>(ptr);
          ptr += sizeof(int);
          PS_CHECK_EQ(small_data_size + descriptor.tensors.size(), data_size);
          std::vector<bool> is_small(data_size, false);
          // fill small tensors
          for (int i = 0; i < small_data_size; i++) {
            int idx = *reinterpret_cast<int *>(ptr);
            ptr += sizeof(int);
            int size = *reinterpret_cast<int *>(ptr);
            ptr += sizeof(int);
            SArray<char> data;
            if (size) data.CopyFrom(ptr, size);
            ptr += size;
            msg.data.at(idx) = data;
            is_small.at(idx) = true;
          }
          // fill large tensors
          int idx = 0;
          for (int i = 0; i < data_size; i++) {
            if (is_small[i]) continue;
            void *buf = allocation.tensors[idx]
                            .buffer.unwrap<tensorpipe::CpuBuffer>()
                            .ptr;
            SArray<char> data;
            data.reset(
                static_cast<char *>(buf), descriptor.tensors[idx].length,
                [](char *data_ptr) { delete[] data_ptr; },
                msg.meta.src_dev_type, msg.meta.src_dev_id,
                msg.meta.dst_dev_type, msg.meta.dst_dev_id);
            msg.data[i] = data;
            idx++;
          }
        }
        queue_->Push(msg);
        ReceiveFromPipe(pipe);
      });
    });
  }

  std::shared_ptr<tensorpipe::Context> InitContext(Node &node) {
    auto context = std::make_shared<tensorpipe::Context>();
    auto transportContext = tensorpipe::transport::uv::create();
    context->registerTransport(0, "tcp", transportContext);
    auto shmtransport = tensorpipe::transport::shm::create();
    context->registerTransport(10, "shm", shmtransport);
    auto basicChannel = tensorpipe::channel::basic::create();
    context->registerChannel(100, "basic", basicChannel);
    auto uv_nthreads_str = Environment::Get()->find("DMLC_PS_UV_NTHREADS");
    int uv_nthreads = uv_nthreads_str ? atoi(uv_nthreads_str) : 0;
    if (uv_nthreads > 1) {
      std::vector<std::shared_ptr<tensorpipe::transport::Context>> contexts;
      std::vector<std::shared_ptr<tensorpipe::transport::Listener>> listeners;
      std::string address = node.hostname + ":0";
      for (int i = 0; i < uv_nthreads; i++) {
        auto context = tensorpipe::transport::uv::create();
        contexts.push_back(std::move(context));
        listeners.push_back(contexts.back()->listen(address));
      }
      auto mptChannel = tensorpipe::channel::mpt::create(std::move(contexts),
                                                         std::move(listeners));
      context->registerChannel(120, "mpt", mptChannel);
    }
    // cross-process memory channel for intra-machine communication
    auto cmaChannel = tensorpipe::channel::cma::create();
    context->registerChannel(130, "cma", cmaChannel);
    return context;
  }

  std::shared_ptr<tensorpipe::Context> send_ctx_{nullptr};

  std::shared_ptr<tensorpipe::Context> recv_ctx_{nullptr};

  std::shared_ptr<tensorpipe::Listener> listener_{nullptr};

  std::unordered_map<int, std::shared_ptr<tensorpipe::Pipe>> send_pipes_;

  std::mutex mu_;

  std::shared_ptr<ThreadsafeQueue<Message>> queue_;
};
}  // namespace ps

#endif  // DMLC_USE_TP
#endif  // PS_TP_VAN_H_
