/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef IBVWARP_H_
#define IBVWARP_H_
namespace ps {

// Attempt to load a specific symbol version - fail silently
#define LOAD_SYM_VERSION(handle, symbol, funcptr, version) \
  do {                                                     \
    cast = reinterpret_cast<void**>(&funcptr);             \
    *cast = dlvsym(handle, symbol, version);               \
  } while (0)

#define IBV_INT_PS_CHECK_RET_ERRNO(container, internal_name, call,            \
                                   success_retval)                            \
  PS_CHECK_NOT_NULL(container, internal_name);                                \
  int ret = container.call;                                                   \
  if (ret != success_retval) {                                                \
    PS_LOG(WARNING) << "call to " << #internal_name << " failed with error (" \
                    << strerror(errno) << ")";                                \
    return -1;                                                                \
  }                                                                           \
  return 1;

/* PS_CHECK_NOT_NULL: helper macro to check for NULL symbol */
#define PS_CHECK_NOT_NULL(container, internal_name)    \
  if (container.internal_name == NULL) {               \
    PS_LOG(WARNING) << "lib wrapper not initialized."; \
    return -1;                                         \
  }

struct dmlcMlx5dvSymbols {
  int (*mlx5dv_internal_query_qp_lag_port)(struct ibv_qp* qp, uint8_t* port_num,
                                           uint8_t* active_port_num);
  int (*mlx5dv_internal_modify_qp_lag_port)(struct ibv_qp* qp,
                                            uint8_t port_num);
} mlx5dvSymbols;

int buildMlx5dvSymbols(struct dmlcMlx5dvSymbols* mlx5dvSymbols) {
  static void* mlx5dvhandle = NULL;
  void** cast;

  mlx5dvhandle = dlopen("libmlx5.so", RTLD_NOW);
  if (!mlx5dvhandle) {
    mlx5dvhandle = dlopen("libmlx5.so.1", RTLD_NOW);
    if (!mlx5dvhandle) {
      printf("Failed to open libmlx5.so[.1]");
      goto teardown;
    }
  }

  LOAD_SYM_VERSION(mlx5dvhandle, "mlx5dv_query_qp_lag_port",
                   mlx5dvSymbols->mlx5dv_internal_query_qp_lag_port,
                   "MLX5_1.14");
  LOAD_SYM_VERSION(mlx5dvhandle, "mlx5dv_modify_qp_lag_port",
                   mlx5dvSymbols->mlx5dv_internal_modify_qp_lag_port,
                   "MLX5_1.14");
  return 1;

teardown:
  mlx5dvSymbols->mlx5dv_internal_query_qp_lag_port = NULL;
  mlx5dvSymbols->mlx5dv_internal_modify_qp_lag_port = NULL;
  if (mlx5dvhandle != NULL) dlclose(mlx5dvhandle);
  return -1;
}

int wrap_mlx5dv_query_qp_lag_port(struct ibv_qp* qp, uint8_t* port_num,
                                  uint8_t* active_port_num) {
  IBV_INT_PS_CHECK_RET_ERRNO(
      mlx5dvSymbols, mlx5dv_internal_query_qp_lag_port,
      mlx5dv_internal_query_qp_lag_port(qp, port_num, active_port_num), 0);
}

int wrap_mlx5dv_modify_qp_lag_port(struct ibv_qp* qp, uint8_t port_num) {
  IBV_INT_PS_CHECK_RET_ERRNO(mlx5dvSymbols, mlx5dv_internal_modify_qp_lag_port,
                             mlx5dv_internal_modify_qp_lag_port(qp, port_num),
                             0);
}

static pthread_once_t initOnceControl = PTHREAD_ONCE_INIT;
int initResult = -1;
int wrap_ibv_symbols(void) {
  pthread_once(&initOnceControl,
               []() { initResult = buildMlx5dvSymbols(&mlx5dvSymbols); });
  return initResult;
}

}  // namespace ps

#endif  // IBVWARP_H_
