/* Copyright (c) 2025, StepFun Authors. All rights reserved. */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/extension.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <pybind11/stl.h>

#include <unordered_map>
#include <vector>
#include <cassert>
#include <iostream>
#include <mutex>

#include "ps/ps.h"

#ifndef UTIL_H_
#define  UTIL_H_
#endif  // UTIL_H_
