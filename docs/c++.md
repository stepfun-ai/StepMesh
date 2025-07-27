## StepMesh C++ API

### Overview

This document provides an overview of the API interfaces available in the provided C++ code, which is part of an **Attention-FFN Disaggregation (AFD)** system. The system is designed for tensor-based communication between workers and servers, with a focus on efficient and scalable data processing.

### Enum Definitions

- **AF_FLAG_BATCH_START**: Flag indicating the start of a batch operation.
- **AF_FLAG_BATCH_MIDDLE**: Flag indicating an intermediate step within a batch operation.
- **AF_FLAG_BATCH_END**: Flag indicating the end of a batch operation.

### Struct Definitions

- **KeyTensor**: A struct containing a key and a tensor.
- **AFTensorRequest**: A struct representing an AF request, including push and pull tensor batches, timestamps, and an event pointer.
- **AFTensorMeta**: A struct containing metadata about a KV request, including sender information, push and pull metadata, and tensors.
- **AFTensorResponse**: A struct representing an AF response, including KV metadata, KV pairs, an event pointer, and response start time.

### Class Definitions

#### AFTensorWorker

**Description**: The `AFTensorWorker` class represents a worker node in the AFD system. It is responsible for performing tensor-based communication operations, such as pushing and pulling tensors to/from servers.

##### Public Methods

- **AFTensorWorker(int instance_idx = 0)**
    - **Description**: Constructor for the `AFTensorWorker` class. Initializes the worker with the given instance index.
    - **Input**:
        - `instance_idx`: The instance index within a group.

- **~AFTensorWorker()**
    - **Description**: Destructor for the `AFTensorWorker` class. Stops the push-pull worker thread and cleans up resources.

- **int ZBatchPushPull(KeyTensorBatch& push_tensors, KeyTensorBatch& pull_tensors)**
    - **Description**: Performs a batch operation of pushing and pulling tensors to/from servers.
    - **Input**:
        - `push_tensors`: A reference to the KeyTensorBatch object containing the tensors to be pushed and their associated keys.
        - `pull_tensors`: A reference to the KeyTensorBatch object where the pulled tensors and their associated keys will be stored.
    - **Output**: An integer indicating the result of the operation.

- **void Wait(int timestamp)**
    - **Description**: Waits for the operation associated with the given timestamp to complete.
    - **Input**:
        - `timestamp`: The timestamp returned by push, pull, or push-pull operations.

- **std::vector<int> GetAllHandlers(int timestamp)**
    - **Description**: Returns all handlers for the batch push-pull operation associated with the given timestamp.
    - **Input**:
        - `timestamp`: The timestamp returned by push, pull, or push-pull operations.
    - **Output**: A vector of integer handlers.

- **std::pair<struct Trace, struct Trace> FetchTrace(int timestamp)**
    - **Description**: Fetches the performance trace for the operation associated with the given timestamp.
    - **Input**:
        - `timestamp`: The timestamp returned by push, pull, or push-pull operations.
    - **Output**: A pair of `Trace` structs representing the performance trace.

#### AFTensorServer

**Description**: The `AFTensorServer` class represents a server node in the AFD system. It is responsible for handling requests from workers, processing tensor data, and responding to worker requests.

##### Public Methods

- **AFTensorServer(int gpu)**
    - **Description**: Constructor for the `AFTensorServer` class. Initializes the server with the given GPU index.
    - **Input**:
        - `gpu`: The local GPU rank.

- **~AFTensorServer()**
    - **Description**: Destructor for the `AFTensorServer` class. Stops the response worker thread and cleans up resources.

- **void Response(const AFTensorMeta& meta, KeyTensorBatch tensors = {}, bool need_event = true)**
    - **Description**: Responds to a push-pull operation with the given metadata and tensors.
    - **Input**:
        - `meta`: The handler metadata.
        - `tensors`: The pull tensors to respond.
        - `need_event`: A boolean indicating whether an event is needed for synchronization.

- **void SetRequestHandle(const AFServerRequestHandle& request_handle)**
    - **Description**: Sets the request handle for processing AF requests.
    - **Input**:
        - `request_handle`: The user-defined handle for processing AF requests.

- **void RegisterRecvTensor(const at::Tensor& tensor, std::vector<int>& worker_ranks, std::vector<Key>& keys)**
    - **Description**: Registers a tensor with local RDMA devices for communication with workers.
    - **Input**:
        - `tensor`: The tensor to register.
        - `worker_ranks`: The worker ranks to register, and the tensor will be sliced to register for different ranks.
        - `keys`: The keys to register.

### Conclusion

The provided API interfaces enable efficient and scalable communication between worker and server nodes in an AFD system. The `AFTensorWorker` class facilitates tensor-based push and pull operations, while the `AFTensorServer` class handles request processing and response generation. These interfaces are designed to support high-performance data processing and synchronization in distributed computing environments.