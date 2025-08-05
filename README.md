# StepMesh: A High-Performance, Low-Latency Communication Library for Attention-FFN Disaggregation

StepMesh is a communication library designed to provide high-performance and low-latency
communication for Attention-FFN decoupling architectures.

The codebase is developed upon [BytePS](https://github.com/bytedance/ps-lite).

## Overview

The following diagram illustrates the architecture of the **StepMesh** communication framework with the follow highlights.

- **Unified and Flexible ZeroCopy PushPull Interface**: StepMesh provides a single, efficient interface (*ZBatchPushPull*) for tensor transfer.
  *ZBatchPushPull* interface allows users to tailor communication strategies to specific application needs.
- **Communication Across Multiple Accelerators**: Designed to work with different computing Accelerators,
  StepMesh leverages RDMA to achieve high-speed, low-latency communication
  and is architected to easily incorporate support for new computing chips.
- **Low Latency**: StepMesh is engineered to provide the ultra-low latency communication essential for Attention-FFN decoupling architectures.
  Its performance is on par with the limits imposed by hardware components.

<!-- ![StepMesh Framework](./docs/images/framework.png) -->


- **AFTensorWorker API**: This API is responsible for managing the communication operations performed by attention nodes. It provides methods such as `Wait` and `PushPull` for synchronization and data transfer.
- **AFTensorServer API**: This API handles the communication operations performed by FFN nodes. It includes methods like `GetBatch` and `Respond` for retrieving and responding to requests from workers.

#### StepMesh Core

- **NetSend/NetRecv Threads**: These threads are part of the **StepMesh Core** and are responsible for handling network send and receive operations. They interact with various backends to manage data transfers efficiently.

#### Backends

- **RDMATransport**: This backend is responsible for managing **Remote Direct Memory Access (RDMA)** operations. It interacts with the **RDMA NIC (Network Interface Card)** to provide high-speed, low-latency communication between nodes.
- **CPUBackend**: This backend handles computations and data transfers on the **CPU**. It is designed to offload communication tasks from the GPU to the CPU, reducing GPU resource usage.
- **GPUBackend**: This backend manages computations and data transfers on the **GPU**. It leverages the GPU's parallel processing capabilities for efficient data handling.
- **xPUBackend**: This is a placeholder for future backends that may support other types of processing units (e.g., NPU).

#### Communication Flow

- **Worker-to-Server Communication**: Workers use the `PushPull` method to send and receive data to/from servers. The `Wait` method is used for synchronization, ensuring that data transfers are completed before proceeding with further computations.
- **Server-to-Worker Communication**: Servers use the `GetBatch` method to retrieve data from workers and the `Respond` method to send data back to workers. These methods facilitate efficient data exchange between servers and workers.

#### Integration with Accelerators

- The **StepMesh Core** interacts with different backends (CPU, GPU, xPU) to manage data transfers and computations efficiently. This modular design allows the framework to support various types of accelerators and adapt to different hardware configurations.

## Getting Started

### Prerequisites

- Ubuntu OS
- Servers with RDMA NICs and GPU cards.
- PyTorch
- CUDA NVCC or other compilers

## Build

Download code and install dependencies
```bash
git clone https://github.com/stepfun-ai/StepMesh
cd StepMesh
bash tools/install_deps.sh # only once
```

Build StepMesh
```bash
# Please check that your CUDA_HOME is correct

# Build AF library
make af

# Build and install Fserver （AF's Python SDK）
pip3 install -v -e .
```

Build without cuda

```bash

# Build AF library
USE_CUDA=0 make af

# Build and install Fserver （AF's Python SDK）
USE_CUDA=0 pip3 install -v -e .

```


## Concepts

In StepMesh, there are three roles: worker, server and scheduler. Each role is an independent process.

The scheduler is responsible for setting up the connections between workers and servers at initialization. There should be only 1 scheduler process.

A worker process only communicates with server processes, and vice versa.
There won't be any traffic between worker-to-worker, and server-to-server.

Workers can push or pull tensors to servers, and all operations are async.
For pushing tensors, the tensors are broadcasted to servers
## Tutorial

After build, you will have testing applications under `tests/` dir. 
Below we elaborate how you can run with them. 

To debug, set `PS_VERBOSE=1` to see important logs during connection setup, and `PS_VERBOSE=2` to see each message log.

### 1. Single-Node Example

- Single GPU Example: Suppose you want to run with 1 worker and 1 server on same server.

```bash
# ROLE: jointly run scheduler, worker and server; RNIC: your first rdma nic; 
ROLE=joint RNIC=brainpf_bond0 bash tests/fserver/run_single_gpu.sh
```
- Multiple GPU Example: Suppose you want to run with 8 workers and 8 servers on different GPUs of the same server.
```bash
# ROLE: jointly run scheduler, worker and server; RNIC: your first rdma nic; 
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

For more documents, please refer to [docs](./docs).

For more details, please refer to the [**Step-3 system technical report**](https://arxiv.org/abs/2507.19427) and our [**Introduction**](Introduction.md)([Chinese Version](Introduction_cn.md)).
