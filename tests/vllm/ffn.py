import os
from vllm.config import AFDConfig, VllmConfig,ParallelConfig
from stepmesh_connector import StepMeshAFDConnector
import torch
import torch.profiler
import time
from bind_pid import set_numa_affinity, bind_pid
from cycle import get_cycles_per_ms
from stepmesh_connector import StepMeshAFDConnector,AFDConnectorMetadata
# from vllm.distributed.afd_transfer.afd_connector.stepmesh_connector import StepMeshAFDConnector,AFDConnectorMetadata

import numpy as np

os.environ['STEPMESH_BIND_CPU_CORE']='1'
os.environ['STEPMESH_CONNECTOR_DEBUG']='true'
os.environ['STEPMESH_SPLIT_QP_LAG']='1'

ip="10.203.8.15"

cycle_per_ms = get_cycles_per_ms()

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
node_rank = rank // local_world_size

# bind_pid(os.getpid(), local_rank)
set_numa_affinity(local_rank)
afd_config = AFDConfig(
    afd_connector="stepmesh",
    afd_role="ffn",
    afd_port=1239,
    afd_host=f"{ip}",
    num_afd_stages=3,
    num_attention_servers=1,
    num_ffn_servers=1,
    afd_server_rank=node_rank,
)
parallel_config = ParallelConfig(
    tensor_parallel_size=8,
    pipeline_parallel_size=1,
    data_parallel_size=1,
)
vllm_config = VllmConfig(
    afd_config=afd_config,
    parallel_config=parallel_config,
)
connector = StepMeshAFDConnector(
    rank=rank,
    local_rank=local_rank,
    config=vllm_config
)
torch.cuda.set_device(local_rank)
time.sleep(5)
connector.init_afd_connector()
# set_numa_affinity(local_rank)
import fserver_lib as ps
ret_buffer = torch.rand([65535, 7168], dtype=torch.bfloat16, device='cuda')


s = torch.cuda.Stream()

if __name__ == "__main__":
    counter = 0
    profiler = None
    while True:
        counter += 1
        if counter % 1000 == 0:
            print(f"Respond {rank} counter {counter}")
        
        # 在counter为10000~11100时启用torch profiler，包含100轮warmup + 1000轮active记录
        if counter == 10000:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=100, active=1000),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_logs/rank_{rank}', use_gzip=True),
                record_shapes=True,
                with_stack=True,
                experimental_config=torch.profiler._ExperimentalConfig(
                    verbose=True,     # 启用详细日志
                    enable_cuda_sync_events=True  # 启用CUDA同步事件跟踪
                )
            )
            profiler.start()
            print(f"Rank {rank}: Started profiler at counter {counter}, will warmup 100 steps then record 1000 steps with gzip compression")
        
        if counter >= 10000 and counter <= 11099:
            profiler.step()
        
        if counter == 11099:
            # profiler会在active阶段结束时自动停止并保存，无需手动stop()
            print(f"Rank {rank}: Profiler completed at counter {counter}, recorded 100 warmup + 1000 active steps")
            profiler = None
        
        with torch.cuda.stream(s):
            batches = ps.get_batch()
            if len(batches) != 0:
                recv_tensor_list = [batches[i][1][0] for i in range(1)]
                comm_id_list = [batches[i][0] for i in range(1)]
                torch.cuda._sleep(int(cycle_per_ms * 0.20))

                ps.respond_vec(ret_buffer, recv_tensor_list, comm_id_list)
                # if counter % (1830*5) == 0:
                #     connector.print_trace()