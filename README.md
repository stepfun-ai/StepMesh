# StepAF

StepAF is the communication library for Attention-FFN Disaggregation of Step3.
It is designed for ultra-low-latency communication based on GPUDirect RDMA.

## Build

```bash
git clone -b byteps https://github.com/bytedance/ps-lite
cd ps-lite 
make -j USE_RDMA=1
```

- Remove `USE_RDMA=1` if you don't want to build with RDMA ibverbs support.
- Add `USE_FABRIC=1` if you want to build with RDMA libfabric support for AWS Elastic Fabric Adaptor.

To build ps-lite with UCX:
```
# dependencies
sudo apt install -y build-essential libtool autoconf automake libnuma-dev unzip pkg-config

# build ucx
wget https://github.com/openucx/ucx/archive/refs/tags/v1.11.1.tar.gz
tar -xf v1.11.1.tar.gz
cd ucx-1.11.1
(./autogen.sh || ./autogen.sh) && ./configure --enable-logging --enable-mt --with-verbs --with-rdmacm --with-cuda=/usr/local/cuda
make clean && make -j && sudo make install -j

# build ps-lite
cd ..
make clean; USE_UCX=1 CUDA_HOME=/usr/local/cuda USE_CUDA=1 make -j
```

BytePS relies on UCXVan for GPU related communication, such as intra-node cuda-IPC, inter-node GPU-to-GPU / GPU-to-CPU communication with GPU-direct RDMA.
For the list of transports UCX supports, see [link](https://openucx.readthedocs.io/en/master/faq.html?highlight=UCX_TLS#list-of-main-transports-and-aliases).


## Concepts

In ps-lite, there are three roles: worker, server and scheduler. Each role is an independent process.

The scheduler is responsible for setting up the connections between workers and servers at initialization. There should be only 1 scheduler process.

A worker process only communicates with server processes, and vice versa. There won't be any traffic between worker-to-worker, and server-to-server.


## Tutorial

After build, you will have two testing applications under `tests/` dir, namely `test_benchmark` and `test_ipc_benchmark`. 
Below we elaborate how you can run with them. 

To debug, set `PS_VERBOSE=1` to see important logs during connection setup, and `PS_VERBOSE=2` to see each message log.

### 1. Basic benchmark

Suppose you want to run with 1 worker and 1 server on different machines. Therefore, we need to launch 3 processes in total (including the scheduler). You can launch the scheduler process at any machine as it does not affect the performance.

For the scheduler:

```
# common setup
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=`ip addr show brainpf0 | awk '/inet / {print $2}' | cut -d/ -f1`
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_INTERFACE=brainpf0        # my RDMA interface 
export DMLC_NODE_HOST=`ip addr show $DMLC_INTERFACE | awk '/inet / {print $2}' | cut -d/ -f1`
# launch scheduler
DMLC_ROLE=scheduler ./tests/test_benchmark
```


For the server:
```
# common setup
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=`ip addr show brainpf0 | awk '/inet / {print $2}' | cut -d/ -f1`
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_INTERFACE=brainpf1        # my RDMA interface 
export DMLC_NODE_HOST=`ip addr show $DMLC_INTERFACE | awk '/inet / {print $2}' | cut -d/ -f1`
# launch server
DMLC_ROLE=server ./tests/test_benchmark
```

For the worker:
```
# common setup
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=`ip addr show brainpf0 | awk '/inet / {print $2}' | cut -d/ -f1`
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_INTERFACE=brainpf2        # my RDMA interface 
export DMLC_NODE_HOST=`ip addr show $DMLC_INTERFACE | awk '/inet / {print $2}' | cut -d/ -f1`
# launch worker
DMLC_ROLE=worker ./tests/test_benchmark
```

If you want to use libfabric with Amazon Elastic Fabric Adaptor, make sure to set `DMLC_ENABLE_RDMA=fabric` for all processes. If you are
using libfabric < 1.10, please also set `FI_EFA_ENABLE_SHM_TRANSFER=0` to avoid a bug in the EFA shm provider.

If you just want to use TCP, make sure to unset `DMLC_ENABLE_RDMA` for all processes.

### 2. Benchmark with IPC support

The `test_ipc_benchmark` demonstrates how inter-process communication (IPC) helps improve RDMA performance when the server is co-located with the worker.

Suppose you have two machines. Each machine should launch a worker and a server process. 

For the scheduler: 
(you can launch it on either machine-0 or machine-1)
```
# common setup
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=2 
export DMLC_PS_ROOT_URI=10.0.0.2  # scheduler's RDMA interface IP 
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_INTERFACE=eth5        # my RDMA interface 

# launch scheduler
DMLC_ROLE=scheduler ./tests/test_ipc_benchmark
```

For machine-0 and machine-1:

```
# common setup
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=2 
export DMLC_PS_ROOT_URI=10.0.0.2  # scheduler's RDMA interface IP 
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_INTERFACE=eth5        # my RDMA interface 

# launch server and worker
DMLC_ROLE=server ./tests/test_ipc_benchmark &
DMLC_ROLE=worker ./tests/test_ipc_benchmark 
```

Note: This benchmark is only valid for RDMA. 

### 3. Other GPU-related benchmarks


```
cd tests;
NODE_ONE_IP=xxx NODE_TWO_IP=yyy bash test.sh (local|remote|joint) bytes_per_msg msg_count (push_only|pull_only|push_pull) (cpu2cpu|cpu2gpu|gpu2gpu|gpu2cpu)
```

### 4. AF benchmarks

Build the AF test
```
make -j USE_RDMA=1 CUDA_HOME=/usr/local/cuda USE_CUDA=1
```

Run the benchmark
- On FFN Server & Scheduler Node 
```
bash tests/test_benchmark_af.sh
```

- On Worker Node
```
bash tests/test_benchmark_af.sh x.x.x.x  
```

### 5. Bonding test

Use env DMLC_SPLIT_QP_LAG for LAG configuration. 0 for auto, 1/2 for configuration while others for disable.

Run Scheduler

```bash
 # script            role      nic           rank
 ./tests/test-mqp.sh scheduler brainpf_bond0 0
 ```

Run Server

```bash
 # script            role      nic           rank
 ./tests/test-mqp.sh server brainpf_bond0 0
 ```


Run Worker

```bash
 # script            role      nic           rank    lag: 0 as auto, 1/2 for config, other for disable
 ./tests/test-mqp.sh worker brainpf_bond0    0       1
 ```

### 6. Tensor Based GDR test

Build GDR Tensor Test
```bash
make  tensor
 ```

Run Server & Scheduler 
```bash

export BINARY=./cmake_build/tests/test_benchmark_af_tensor
# export  BINARY=./cmake_build/tests/test_benchmark_af_verify
 
bash ./tests/test_benchmark_af.sh 
 ```

Run Worker
```bash

export BINARY=./cmake_build/tests/test_benchmark_af_tensor
# export  BINARY=./cmake_build/tests/test_benchmark_af_verify

bash ./tests/test_benchmark_af.sh xx.xx.xx.xx # scheduler ip
 ```

### 7. Run UT

Run utests in a GPU node with RNICs
```bash
bash tests/utests/run_ut.sh
 ```
