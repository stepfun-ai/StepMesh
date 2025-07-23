# StepAF: High-Performance, Low-Latency Communication Library for Attention-FFN Decoupling

StepAF is a communication library designed to provide high-performance and low-latency
communication for Attention-FFN decoupling architectures.

The codebase is developed upon [BytePS](https://github.com/bytedance/ps-lite).

## Overview

- **Unified and Flexible ZeroCopy PushPull Interface**: StepAF provides a single, efficient interface (*ZBatchPushPull*) for tensor transfer. 
*ZBatchPushPull* interface allows users to tailor communication strategies to specific application needs.
- **Communication Across Multiple Accelerators**: Designed to work with different computing Accelerators, 
StepAF leverages RDMA to achieve high-speed, low-latency communication
and is architected to easily incorporate support for new computing chips.
- **Low Latency**: StepAF is engineered to provide the ultra-low latency communication essential for Attention-FFN decoupling architectures.
Its performance is on par with the limits imposed by hardware components.

## Getting Started

### Prerequisites

- Ubuntu OS
- Servers with RDMA NICs and GPU cards.
- PyTorch
- CUDA NVCC or other compilers

## Build

Download code and install dependencies
```bash
git clone https://github.com/stepfun-ai/stepaf
cd stepaf
bash tools/install_deps.sh # just once
```

- Remove `USE_RDMA=1` if you don't want to build with RDMA ibverbs support.
- Add `USE_FABRIC=1` if you want to build with RDMA libfabric support for AWS Elastic Fabric Adaptor.

Build StepAF
```bash
# Please ensure your CUDA_HOME is correct

# Build AF library
make af

# Build and install Fserver （AF's Python SDK）
pip3 install -v -e .
```

## Concepts

In StepAF, there are three roles: worker, server and scheduler. Each role is an independent process.

The scheduler is responsible for setting up the connections between workers and servers at initialization. There should be only 1 scheduler process.

A worker process only communicates with server processes, and vice versa.
There won't be any traffic between worker-to-worker, and server-to-server.

Workers can push or pull tensors to servers, and all operations are async.
For pushing tensors, the tensors are broadcasted to servers.
For pulling tensors, each worker receives an independent tensor from each server.

## Tutorial

After build, you will have two testing applications under `tests/` dir. 
Below we elaborate how you can run with them. 

To debug, set `PS_VERBOSE=1` to see important logs during connection setup, and `PS_VERBOSE=2` to see each message log.

### 1. Single-Node Example

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

### 2. Two-Node Example
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

For more test cases and examples, please refer to [tests](./tests).
