/* Copyright (c) 2025, StepFun Authors. All rights reserved. */

#include "./util.h"
#include "./public.hpp"
#include <future>
#ifdef DMLC_USE_CUDA
  #include <ATen/cuda/CUDAEvent.h>
  #include <ATen/cuda/CUDAContext.h>
#endif

#ifndef PRIAVET_OPS_
#define PRIVATE_OPS_

using namespace ps;
#ifdef DMLC_USE_CUDA
class SimpleNotify{
private:
    int notify_cnt = 1;
    CUdeviceptr dflag;
    uint32_t* hflag;
    std::thread th_;
    std::future<std::vector<ServerDataBatch>> fut;
public:
    void init() {
        cudaHostAlloc(&hflag, sizeof(uint32_t), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&dflag, (void*)hflag, 0);
    }

    // for worker
    void wait_event_done(){
      if (th_.joinable()) {
        th_.join();  
      }
    }

    // for worker
    void stream_wait_event(int handler) {
      auto stream = at::cuda::getCurrentCUDAStream();
      cuStreamWaitValue32((CUstream)stream, dflag, notify_cnt, CU_STREAM_WAIT_VALUE_EQ);
      th_ = std::thread([handler, this]{
         fworker_->Wait(handler);
         *(this->hflag) = this->notify_cnt;
         ++(this->notify_cnt);      
      });
    }

    void block_now_stream() {
      auto stream = at::cuda::getCurrentCUDAStream();
      cuStreamWaitValue32((CUstream)stream, dflag, notify_cnt, CU_STREAM_WAIT_VALUE_EQ);
    }

    // for server
    void block_now_stream_and_get_batch() {
      auto stream = at::cuda::getCurrentCUDAStream();
      cuStreamWaitValue32((CUstream)stream, dflag, notify_cnt, CU_STREAM_WAIT_VALUE_EQ);
      fut = std::async(std::launch::async, [this]{
           auto ret = get_batch();
           *(this->hflag) = this->notify_cnt;
           ++(this->notify_cnt);
           return ret;
        });
    }

    // for server
    std::vector<ServerDataBatch> get_future_batch_data(){
      return fut.get();
    }
};

void pybind_private(py::module &m){
  py::class_<SimpleNotify>(m, "SimpleNotify")
      .def(py::init<>())
      .def("init", &SimpleNotify::init)
      .def("block_now_stream_and_get_batch", &SimpleNotify::block_now_stream_and_get_batch)
      .def("get_future_batch_data", &SimpleNotify::get_future_batch_data)
      .def("block_now_stream", &SimpleNotify::block_now_stream)
      .def("wait_event_done", &SimpleNotify::wait_event_done)
      .def("stream_wait_event", &SimpleNotify::stream_wait_event);
}
#else
void pybind_private(py::module &m){}
#endif //DMLC_USE_CUDA

#endif //PRIVATE_OPS_
