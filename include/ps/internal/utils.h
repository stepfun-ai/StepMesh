/**
 *  Copyright (c) 2015 by Contributors
 *  Modifications Copyright (C) by StepAI Contributors. 2025.
 */
#ifndef PS_INTERNAL_UTILS_H_
#define PS_INTERNAL_UTILS_H_

#include <ctype.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "dmlc/logging.h"
#include "ps/internal/env.h"

namespace ps {

#ifdef _MSC_VER
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <inttypes.h>
#endif

/*!
 * \brief Get environment variable as int with default.
 * \param key the name of environment variable.
 * \param default_val the default value of environment vriable.
 * \return The value received
 */
template <typename V>
inline V GetEnv(const char *key, V default_val) {
  const char *val = Environment::Get()->find(key);
  if (val == nullptr) {
    return default_val;
  } else {
    return V(val);
  }
}

inline int GetEnv(const char *key, int default_val) {
  const char *val = Environment::Get()->find(key);
  if (val == nullptr) {
    return default_val;
  } else {
    return atoi(val);
  }
}

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &);              \
  void operator=(const TypeName &)
#endif

#define LL PS_LOG(ERROR)

/*
 * Get the current cycle count. This is useful for performance measurement.
 */
inline static uint64_t _GetCurrentCycle() {
  uint64_t cycle = 0;
  uint32_t *t = reinterpret_cast<uint32_t *>(&cycle);
  __asm__ volatile("rdtsc" : "=a"(t[0]), "=d"(t[1]));
  return cycle;
}

/*
 * Get the current nanosecond count. Only for initialization.
 */
static inline uint64_t _GetNanosecond() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  uint64_t nsec = ts.tv_sec * 1000000000LL + ts.tv_nsec;
  return nsec;
}

/*
 * Convert cycle count to 32 * nanosecond count.
 */
static uint64_t CycleToNs() {
  uint64_t start_ns = _GetNanosecond();
  uint64_t start_cycle = _GetCurrentCycle();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  uint64_t stop_ns = _GetNanosecond();
  uint64_t stop_cycle = _GetCurrentCycle();
  return static_cast<uint64_t>(((stop_cycle - start_cycle) << 5) /
                               (stop_ns - start_ns));
}

/*
 * Get the cycle to ns paramater.
 */
static uint64_t norm = CycleToNs();

/*!
 * \brief Get the current nanocount.
 */
static inline uint64_t GetNanosecond() {
#ifdef STEPMESH_ENABLE_TRACE
  if (norm == 0) {
    norm = CycleToNs();
  }
  return static_cast<uint64_t>((_GetCurrentCycle() << 5) / norm);
#else
  return 0;
#endif
}

static int PS_VERBOSE = ps::GetEnv("PS_VERBOSE", 0);

/**
 * \brief Bind current thread to a specific CPU core.
 * \param offset is the start of the core id
 * \param core_count is the number of cores the thread need.
 */
static inline void BindCpuCore(int offset, int core_count = 1) {
  int gpu = -1;
  Environment::Get()->find("STEPMESH_GPU", &gpu, gpu);
  int bind_enable = 0;
  Environment::Get()->find("STEPMESH_BIND_CPU_CORE", &bind_enable, bind_enable);

  if (gpu <= -1 || bind_enable == 0) {
    return;
  }

  int cores_per_gpu = 5;
  int cpu_core = 0;
  int cores_per_socket = 48;
  int start_offset = 10;

  Environment::Get()->find("STEPMESH_CPU_CORES_PER_SOCKET", &cores_per_socket,
                           cores_per_socket);
  Environment::Get()->find("STEPMESH_CPU_CORES_PER_GPU", &cores_per_gpu,
                           cores_per_gpu);
  Environment::Get()->find("STEPMESH_CPU_START_OFFSET", &start_offset,
                           start_offset);

  if (offset >= cores_per_gpu) {
    offset = cores_per_gpu - 1;
    std::cout << "Bind Core idx is larger than cores count for each GPU, reset "
                 "idx as "
              << offset;
  }
  int basic_core_id = gpu * cores_per_gpu + start_offset + offset;
  if (gpu < 2) {
    cpu_core = basic_core_id;
  } else if (gpu < 4) {
    cpu_core = basic_core_id + cores_per_socket / 2 - cores_per_gpu * 2;
  } else if (gpu < 6) {
    cpu_core = basic_core_id + cores_per_socket - cores_per_gpu * 4;
  } else {
    cpu_core = basic_core_id + cores_per_socket / 2 * 3 - cores_per_gpu * 6;
  }

  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int i = 0; i < core_count; i++) {
    CPU_SET(cpu_core + i, &mask);
  }

  if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
    std::cerr << "could not set CPU affinity: gpu " << gpu << "-> cpu"
              << cpu_core << std::endl;
  } else if (PS_VERBOSE >= 1) {
    std::cout << "BindToCpuCore: gpu " << gpu << " -> cpu " << cpu_core
              << std::endl;
  }
}

}  // namespace ps

#endif  // PS_INTERNAL_UTILS_H_
