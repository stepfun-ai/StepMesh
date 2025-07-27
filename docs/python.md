## StepMesh Python API

### Overview

This document provides an overview of the API interfaces available in the provided C++ code. These interfaces are designed to facilitate communication and synchronization between worker and server processes in a distributed computing environment.

### Function Summary

1. **get_batch**: Retrieves a batch of server data.
2. **respond**: Responds to a request with a vector of tensors.
3. **respond_vec**: Responds to multiple requests with a vector of tensors.
4. **push_pull**: Performs a push-pull operation for tensors.
5. **wait**: Waits for a specific handler to complete.
6. **barrier**: Implements a barrier synchronization.
7. **init**: Initializes the system.
8. **register_recv_buffer**: Registers a receive buffer for tensors.
9. **stop**: Stops the system.
10. **get_all_handlers**: Retrieves all handlers for a specific handler.
11. **fetch_trace**: Fetches the trace for a specific handler.

### Detailed Function Descriptions

#### 1. get_batch

```cpp
def get_batch()
```

**Description**:
This function retrieves a batch of server data. It spins until the `q_signal_` matches the `worker_mask_`, then locks a mutex, and processes the data queues to construct and return a vector of `ServerDataBatch`.

**Return**:
A vector of `ServerDataBatch` objects.

#### 2. respond

```python
void respond(tensors, handler, need_event)
```

**Description**:
This function responds to a request identified by `handler` with a vector of tensors. It retrieves the request metadata, checks the tensor sizes, and then constructs and sends a response.

**Parameters**:
- `tensors`: A vector of `torch::Tensor` objects to be sent in the response.
- `handler`: The identifier for the request to be responded to.
- `need_event`: A boolean indicating whether an event is needed.

#### 3. respond_vec

```python
def respond_vec(ret_buffer, tensors_vec, handler_vec)
```

**Description**:
This function responds to multiple requests with a vector of tensors. It iterates over the `handler_vec` and slices the `ret_buffer` to create individual responses for each handler.

**Parameters**:
- `ret_buffer`: A `torch::Tensor` to be sliced and sent in the responses.
- `tensors_vec`: A vector of `torch::Tensor` objects to be sent in the responses.
- `handler_vec`: A vector of handler identifiers for the requests to be responded to.

#### 4. push_pull

```python
def push_pull(push_tensors, push_keys, pull_tensors, pull_keys)
```

**Description**:
This function performs a push-pull operation for tensors. It constructs `KeyTensorBatch` objects for both push and pull operations, and then calls the `ZBatchPushPull` method of the `fworker_` object.

**Parameters**:
- `push_tensors`: A vector of `torch::Tensor` objects to be pushed.
- `push_keys`: A vector of keys for the push tensors.
- `pull_tensors`: A vector of `torch::Tensor` objects to be pulled.
- `pull_keys`: A vector of keys for the pull tensors.

**Return**:
An integer handler for the push-pull operation.

#### 5. wait

```cpp
def wait(handler)
```

**Description**:
This function waits for a specific handler to complete. It calls the `Wait` method of the `fworker_` object with the given handler.

**Parameters**:
- `handler`: The identifier for the handler to wait for.

#### 6. barrier

```cpp
def barrier(include_server, include_worker)
```

**Description**:
This function implements a barrier synchronization. It determines the node group based on the `include_server` and `include_worker` flags, and then calls the `Barrier` method of the appropriate `Postoffice` object.

**Parameters**:
- `include_server`: A boolean indicating whether servers should be included in the barrier.
- `include_worker`: A boolean indicating whether workers should be included in the barrier.

#### 7. init

```python
def init()
```

**Description**:
This function initializes the system. It sets up the role, GPU, group size, node rank, instance ID, and number of workers. It also initializes the data queues and signals, starts the parameter server, and creates the appropriate `AFTensorWorker` or `AFTensorServer` object based on the role.

#### 8. register_recv_buffer

```python
def register_recv_buffer(tensor, worker_ranks, push_keys)
```

**Description**:
This function registers a receive buffer for tensors. It calls the `RegisterRecvTensor` method of the `fserver_` object with the given tensor, worker ranks, and push keys.

**Parameters**:
- `tensor`: A `torch::Tensor` to be registered as a receive buffer.
- `worker_ranks`: A vector of worker ranks.
- `push_keys`: A vector of push keys.

#### 9. stop

```python
def stop()
```

**Description**:
This function stops the system. It performs a barrier synchronization across all nodes and then calls the `Finalize` method of the parameter server.

#### 10. get_all_handlers

```python
def get_all_handlers(handler)
```

**Description**:
This function retrieves all handlers for a specific handler. It calls the `GetAllHandlers` method of the `fworker_` object with the given handler.

**Parameters**:
- `handler`: The identifier for the handler to retrieve all handlers for.

**Return**:
A vector of integer handlers.

#### 11. fetch_trace

```cpp
def fetch_trace(handler)
```

**Description**:
This function fetches the trace for a specific handler. It calls the `FetchTrace` method of the `fworker_` object with the given handler, and then constructs and returns a vector of timestamps.

**Parameters**:
- `handler`: The identifier for the handler to fetch the trace for.

**Return**:
A vector of `uint64_t` timestamps representing the trace.
