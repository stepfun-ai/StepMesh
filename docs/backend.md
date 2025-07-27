# StepMesh Backend: An Abstract Class for Accelerators 

> We assume that all the backends can use standard RDMA NICs and RDMA APIs.

## Backend

The `Backend` class is an abstract interface that provides a set of methods for managing device resources, such as setting device indices, allocating and freeing memory, creating and managing events, and synchronizing operations across devices. This API allows developers to write device-agnostic code that can be executed on various backends (e.g., CPU, GPU) without needing to know the specific details of each device.

### Class Methods

#### `SetDevice(int dev)`

Sets the device index for the current thread.

- **Parameters:**
    - `dev`: The device index to set.
- **Returns:**
    - `BACKEND_OK` if the device is successfully set.
    - `BACKEND_FAILED` if an error occurs.

#### `GetDeviceId()`

Retrieves the device index for the current thread.

- **Returns:**
    - The device index.

#### `GetDevice()`

Gets the Torch device associated with the current device.

- **Returns:**
    - The Torch device.

#### `Alloc(uint64_t size)`

Allocates memory on the device.

- **Parameters:**
    - `size`: The size of memory to allocate in bytes.
- **Returns:**
    - A pointer to the allocated memory if successful.
    - `nullptr` if memory allocation fails.

#### `Free(void* m)`

Frees memory allocated on the device.

- **Parameters:**
    - `m`: A pointer to the memory to be freed.

#### `CreateEvent()`

Creates a stream event.

- **Returns:**
    - A pointer to the created event if successful.
    - `nullptr` if event creation fails.

#### `FreeEvent(void* event)`

Frees a previously created event.

- **Parameters:**
    - `event`: A pointer to the event to be freed.
- **Returns:**
    - `BACKEND_OK` if the event is successfully freed.
    - `BACKEND_FAILED` if an error occurs.

#### `RecordEvent(void* event, void* stream)`

Records an event on the specified stream.

- **Parameters:**
    - `event`: A pointer to the event to be recorded.
    - `stream`: A pointer to the user-designated stream (can be `nullptr` to use the default stream).
- **Returns:**
    - `BACKEND_OK` if the event is successfully recorded.
    - `BACKEND_FAILED` if an error occurs.

#### `SyncEvent(void* event)`

Synchronizes and waits for the specified event.

- **Parameters:**
    - `event`: A pointer to the event to be synchronized.
- **Returns:**
    - `BACKEND_OK` if the event is successfully synchronized.
    - `BACKEND_FAILED` if an error occurs.

#### `Get()`

Retrieves the backend implementation.

- **Returns:**
    - A pointer to the backend implementation.

#### `Register(const std::string& name, Backend* backend)`

Registers a backend implementation with a specified name.

- **Parameters:**
    - `name`: The name of the backend.
    - `backend`: A pointer to the backend implementation.

### Protected Constructor

#### `Backend()`

The protected default constructor ensures that the `Backend` class cannot be instantiated directly. Instead, derived classes must implement the abstract methods defined in the `Backend` interface.

### Private Static Variables and Methods

#### `backends_mutex_`

A static mutex used to synchronize access to the `backends_` map.

#### `backends_`

A static unordered map that stores registered backend implementations, keyed by their names.

#### `GetImpl()`

A private static method that retrieves the backend implementation based on the environment variable `STEPMESH_BAKCEND`. If no backend is specified, it defaults to "GPU".

#### `RegisterImpl(const std::string& name, Backend* backend)`

A private static method that registers a backend implementation in the `backends_` map.

This example demonstrates setting the device index, allocating memory, creating and recording an event, synchronizing the event, and finally freeing the allocated resources. The `Backend::Get()` method is used to retrieve the backend implementation, allowing the code to be device-agnostic.

## Add Your Backend

### Step 1: Implement Your Backend

Begin by implementing all the required interfaces of the `Backend` class. You can use the [GPU Backend](../src/backend/gpu_backend.cc) as a reference for structuring your implementation. Ensure that each method is correctly overridden to provide the necessary functionality for your custom backend.

### Step 2: Register Your Backend

Once your backend implementation is complete, register it using the `Backend::Register` function. This step is crucial for making your backend available within the system. For example:

```cpp
Backend::Register("MY_BACKEND", new MyBackend());
```

This registration should be performed during the initialization phase, typically within the `StartPS` function located in [ps.h](../include/ps/ps.h).

### Step 3: Configure and Run StepMesh with Your Backend

To utilize your newly created backend when running StepMesh, set the appropriate environment variable:

```bash
export STEPMESH_BACKEND=MY_BACKEND
```

This configuration ensures that StepMesh will use your custom backend for its operations. Verify that the environment variable is correctly set before launching StepMesh to guarantee that your backend is properly loaded and utilized.

By following these steps, you can seamlessly integrate your custom backend into the StepMesh framework, enabling it to leverage your specialized functionalities and optimizations.
