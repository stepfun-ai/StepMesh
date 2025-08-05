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
void pybind_private(py::module &m){}
#else
void pybind_private(py::module &m){}
#endif //DMLC_USE_CUDA

#endif //PRIVATE_OPS_