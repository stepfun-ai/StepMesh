# StepMesh Tests

> All test cases have been rigorously evaluated within our internal environments. While it may be challenging for certain cases to perform consistently across every platform, we are committed to providing the best possible support to ensure broad compatibility.

## 1. Fserver Tests

The tests cases shown in root directory README.

### A. Single-Node Example

- Single GPU Example: Suppose you want to run with 1 worker and 1 server on same server.

```bash
# ROLE: joinly run scheduler, worker and server; RNIC: your first rdma nic; 
ROLE=joint RNIC=brainpf_bond0 bash tests/fserver/run_single_gpu.sh
```
- Multiple GPU Exmaple: Suppose you want to run with 8 workers and 8 servers on different GPUs of the same server.
```bash
# ROLE: joinly run scheduler, worker and server; RNIC: your first rdma nic; 
ROLE=joint RNIC=brainpf_bond0 bash tests/fserver/run_multi_gpu.sh
```

### B. Two-Node Example
- Run scheduler and servers
```bash
# Server
ROLE=server bash tests/fserver/run_multi_gpu.sh
# the first line prints scheduler ip
```

- Run workers
```bash
# worker
ROLE=worker RNIC=brainpf_bond0 bash tests/fserver/run_multi_gpu.sh ${scheduler ip}
```


## 2. Unit tests

These scripts should run in the root directory of the project. Uint tests are designed to be run on a single machine.

### A. Single GPU Example

``` bash
bash tests/utest/run_single_gpu_ut.sh
```

And the output looks like:

```bash
-> # bash tests/utests/run_single_gpu_ut.sh 

+ trap cleanup EXIT
+ export DMLC_NUM_WORKER=1
+ DMLC_NUM_WORKER=1
+ export DMLC_NUM_SERVER=1
+ DMLC_NUM_SERVER=1
+ export DMLC_INTERFACE=brainpf_bond0
+ DMLC_INTERFACE=brainpf_bond0
++ ip -o -4 addr
++ grep brainpf_bond0
++ awk '{print $4}'
++ cut -d/ -f1
+ export DMLC_PS_ROOT_URI=10.XX.XX.XX
+ DMLC_PS_ROOT_URI=10.XX.XX.XX
+ export DMLC_PS_ROOT_PORT=12278
+ DMLC_PS_ROOT_PORT=12278
+ export STEPMESH_SPLIT_QP_LAG=1
+ STEPMESH_SPLIT_QP_LAG=1
+ export DMLC_ENABLE_RDMA=ibverbs
+ DMLC_ENABLE_RDMA=ibverbs
+ echo 'SCHEDULER_IP is 10.XX.XX.XX'
SCHEDULER_IP is 10.XX.XX.XX
+ export DMLC_NODE_HOST=10.XX.XX.XX
+ DMLC_NODE_HOST=10.XX.XX.XX
+ cleanup
+ echo 'kill all testing process of ps lite for user root'
kill all testing process of ps lite for user root
+ pkill -9 -f ./cmake_build/tests/utests/ut_server
+ pkill -9 -f ./cmake_build/tests/utests/ut_scheduler
+ pkill -9 -f ./cmake_build/tests/utests/ut_tensor_worker
+ sleep 1
+ export STEPMESH_GPU=0
+ STEPMESH_GPU=0
+ DMLC_ROLE=scheduler
+ ./cmake_build/tests/utests/ut_scheduler
+ sleep 1
+ DMLC_ROLE=server
+ ./cmake_build/tests/utests/ut_server
+ export STEPMESH_GPU=0
+ STEPMESH_GPU=0
+ export DMLC_INTERFACE=auto
+ DMLC_INTERFACE=auto
+ DMLC_ROLE=worker
+ ./cmake_build/tests/utests/ut_tensor_worker
GPU 0 Batch 1: ALL PASS duration=124160254ns
GPU 0 Batch 2: ALL PASS duration=31193456ns
GPU 0 Batch 3: ALL PASS duration=28984183ns
GPU 0 Batch 4: ALL PASS duration=30030704ns
```

### B. Multi-GPU Example

```bash
bash tests/utests/run_multi_gpu_ut.sh
```

And the output looks like:

```bash
GPU 5 Batch 2: ALL PASS duration=169728103ns
GPU 4 Batch 2: ALL PASS duration=180842319ns
GPU 6 Batch 2: ALL PASS duration=169240452ns
GPU 0 Batch 2: ALL PASS duration=161503372ns
GPU 2 Batch 2: ALL PASS duration=178849940ns
GPU 1 Batch 2: ALL PASS duration=175771874ns
GPU 7 Batch 3: ALL PASS duration=163752633ns
GPU 3 Batch 3: ALL PASS duration=181710437ns
GPU 5 Batch 3: ALL PASS duration=176533632ns
GPU 0 Batch 3: ALL PASS duration=171963753ns
```


### C. Function Examples


| Function  | ENV |
| --- | --- |
| `echo` | BIN=./cmake_build/tests/stepmesh_echo_test |
| `pull` | BIN=./cmake_build/tests/stepmesh_pull_test |
| `push` | BIN=./cmake_build/tests/stepmesh_push_test |
| `register` | BIN=./cmake_build/tests/stepmesh_register_test |

This examples can be run with the unit tests scripts with only additional environment variables set. For example the following commqnd can run the single GPU test or 8 GPUs test for stepmesh_echo. Note that don't run this test with STEPMESH_BIND_CPU_CORE=1.

```bash
BIN=./cmake_build/tests/stepmesh_push_test SCHEDULER_BIN=$BIN SERVER_BIN=$BIN WORKER_BIN=$BIN bash tests/utests/run_single_gpu_ut.sh
```


## 3. Benchmark test


### 1 Server , 1 Worker

```bash
ROLE=server BIN=../benchmark/bmk_comm_latency_multiserver bash tests/fserver/run_multi_gpu.sh
# the first line prints scheduler ip
```

```bash
ROLE=worker BIN=../benchmark/bmk_comm_latency_multiserver bash tests/fserver/run_multi_gpu.sh ${scheduler ip}
```

Note that the output is all zero. It is required to build the library with the following edits and rebuild to show the real latency.

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 48bc802..9fe8411 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -23,7 +23,7 @@ set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
 list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
 list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
 
-set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_ZMQ -DDMLC_USE_CUDA -DSTEPMESH_USE_GDR -DDMLC_USE_RDMA")
+set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_ZMQ -DDMLC_USE_CUDA -DSTEPMESH_USE_GDR -DDMLC_USE_RDMA -DSTEPMESH_ENABLE_TRACE")
 
diff --git a/setup.py b/setup.py
index 0c4011a..0eee2f6 100644
--- a/setup.py
+++ b/setup.py
@@ -60,6 +60,7 @@ if __name__ == "__main__":
                         '-DSTEPMESH_USE_GDR',
                         '-DDMLC_USE_RDMA', 
                         '-DSTEPMESH_USE_TORCH',
+                        '-DSTEPMESH_ENABLE_TRACE'
                     ],
                     'nvcc': ['-O3', '-gencode', 'arch=compute_70,code=sm_70', 
                                 '--use_fast_math'] + cc_flag,

```

```bash
comm bmk gpu=6: mean=0.000ms, p50=0.000ms, p99=0.000ms, max=0.000ms
         gpu=6 push 1:
                python_req      0.000   0.000   0.000   0.000 0.000 0.000
                req_send        0.000   0.000   0.000   0.000 0.000 0.000
                req_recv        0.000   0.000   0.000   0.000 0.000 0.000
                process         0.000   0.000   0.000   0.000 0.000 0.000
                rsp_send        0.000   0.000   0.000   0.000 0.000 0.000
                rsp_recv        0.000   0.000   0.000   0.000 0.000 0.000
                python_rsp      0.000   0.000   0.000   0.000 0.000 0.000
                net_cost        0.000   0.000   0.000   0.000 0.000 0.000
         gpu=6 push 2:
                python_req      0.000   0.000   0.000   0.000 0.000 0.000
                req_send        0.000   0.000   0.000   0.000 0.000 0.000
                req_recv        0.000   0.000   0.000   0.000 0.000 0.000
                process         0.000   0.000   0.000   0.000 0.000 0.000
                rsp_send        0.000   0.000   0.000   0.000 0.000 0.000
                rsp_recv        0.000   0.000   0.000   0.000 0.000 0.000
                python_rsp      0.000   0.000   0.000   0.000 0.000 0.000
                net_cost        0.000   0.000   0.000   0.000 0.000 0.000
         gpu=6 pull 1:
                python_req      0.000   0.000   0.000   0.000 0.000 0.000
                req_send        0.000   0.000   0.000   0.000 0.000 0.000
                req_recv        0.000   0.000   0.000   0.000 0.000 0.000
                process         0.000   0.000   0.000   0.000 0.000 0.000
                rsp_send        0.000   0.000   0.000   0.000 0.000 0.000
                rsp_recv        0.000   0.000   0.000   0.000 0.000 0.000
                python_rsp      0.000   0.000   0.000   0.000 0.000 0.000
                net_cost        0.000   0.000   0.000   0.000 0.000 0.000
```


```
comm bmk gpu=7: mean=0.082ms, p50=0.081ms, p99=0.090ms, max=0.396ms
         gpu=7 push 1:
                python_req      0.009   0.009   0.013   0.021 0.021 0.020
                req_send        0.000   0.000   0.001   0.002 0.001 0.001
                req_recv        0.001   0.001   0.001   0.007 0.002 0.002
                process         0.000   0.000   0.001   0.002 0.001 0.001
                rsp_send        0.000   0.000   0.000   0.001 0.001 0.001
                rsp_recv        0.001   0.001   0.001   0.005 0.005 0.002
                python_rsp      0.040   0.040   0.049   0.355 0.343 0.069
                net_cost        0.028   0.028   0.029   0.033 0.031 0.030
```                

### M Server , N Worker [Repairment required]


```bash
ROLE=server NUM_SERVER=2 NUM_WORKER=2 NODE_RANK=0 BIN=../benchmark/bmk_comm_latency_multiserver bash tests/fserver/run_multi_gpu.sh
# the first line prints scheduler ip
```

```bash
ROLE=server-slave NUM_SERVER=2 NUM_WORKER=2 NODE_RANK=1 BIN=../benchmark/bmk_comm_latency_multiserver bash tests/fserver/run_multi_gpu.sh ${scheduler ip}
```


```bash
ROLE=worker NUM_SERVER=2 NUM_WORKER=2 NODE_RANK=0 BIN=../benchmark/bmk_comm_latency_multiserver bash tests/fserver/run_multi_gpu.sh ${scheduler ip}
```

```bash
ROLE=worker NUM_SERVER=2 NUM_WORKER=2 NODE_RANK=1 BIN=../benchmark/bmk_comm_latency_multiserver bash tests/fserver/run_multi_gpu.sh ${scheduler ip}
```