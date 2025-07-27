# Environment Variables

## BytePS Environment Variables
The variables must be set for starting

- `DMLC_NUM_WORKER` : The number of workers
- `DMLC_NUM_SERVER` : The number of servers
- `DMLC_GROUP_SIZE` : The number of processes per worker or server
- `DMLC_NODE_RANK` : The node rank for servers and workers
- `DMLC_ROLE` : The role of the current node, can be `worker`, `server`, or `scheduler`
- `DMLC_PS_ROOT_URI` : The ip or hostname of the scheduler node
- `DMLC_PS_ROOT_PORT` : The port that the scheduler node is listening
- `DMLC_ENABLE_RDMA` : Enable to use RDMA Van

additional variables:

- `DMLC_INTERFACE` : The network interface a node should use.
`auto` can be used for automatically detection of mappings between RDMA NIC ports and GPU cards.
- `DMLC_LOCAL` : Runs in local machines, no network is needed

## StepMesh Environment Variables

Besides the above, StepMesh introduces some independent environment variables.

- `STEPMESH_BAKCEND` : The backend to be is used,
currently we only support `CPU` backend and `GPU` (default) backend.
We are working on supporting more backends
- `STEPMESH_GPU` : Set the device id used by the GPU backend
- `STEPMESH_SPLIT_QP_LAG` : Enable QP traffic balance over bonding RDNA NICs, STEPMESH_SPLIT_QP_LAG=0 bydefault
- `STEPMESH_MEM_SYNC` : Enable synchronize GPU kernel with CPU memory instead of cudaEvent,
STEPMESH_MEM_SYNC=1 by default
- `STEPMESH_BIND_CPU_CORE` : Enable CPU core binding, STEPMESH_BIND_CPU_CORE=1 by default
- `STEPMESH_CPU_CORES_PER_SOCKET` : The count of cpu cores of each CPU socket, default is 48.
- `STEPMESH_CPU_CORES_PER_GPU` : The count of cpu cores should used by each GPU, the default value is 5.
- `STEPMESH_CPU_START_OFFSET` : The first idx of cpu core used by StepMesh.
