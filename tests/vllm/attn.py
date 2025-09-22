import os
from vllm.config import AFDConfig, VllmConfig,ParallelConfig
from stepmesh_connector import StepMeshAFDConnector,AFDConnectorMetadata
import torch
import time
import numpy as np
import fserver_lib as ps
from cycle import get_cycles_per_ms
from bind_pid import set_numa_affinity
import torch.profiler

os.environ['STEPMESH_BIND_CPU_CORE']='1'
os.environ['STEPMESH_CONNECTOR_DEBUG']='true'
os.environ['STEPMESH_SPLIT_QP_LAG']='1'

ip="10.203.8.15"

cycle_per_ms = get_cycles_per_ms()

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
afd_config = AFDConfig(
    afd_connector="stepmesh",
    afd_role="attention",
    afd_port=1239,
    afd_host=f"{ip}",
    num_afd_stages=3,
    num_attention_servers=1,
    num_ffn_servers=1,
)
parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=8,
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
set_numa_affinity(local_rank)
time.sleep(5)
connector.init_afd_connector()
print(f"--------- rank {rank} local_rank {local_rank} init ---------")



if __name__ == "__main__":
    counter = 0
    s = torch.cuda.Stream()
    torch.cuda.set_stream(s)
    profiler = None
    while True:
        if counter % (1830*2) == 0:
            connector.print_trace()
            torch.cuda.synchronize()

        hidden_states = torch.randn(4, 7168, dtype=torch.bfloat16, device="cuda")    
        for layer_idx in range(61):
            for stage_idx in range(afd_config.num_afd_stages):
                counter += 1
                if layer_idx > 0:
                    connector.recv_ffn_output()
                
                torch.cuda._sleep(int(cycle_per_ms * 0.20))
                # cpu sleep 100us 
                time.sleep(0.0001)
                connector.send_attn_output(
                    hidden_states,
                    AFDConnectorMetadata.create_attention_metadata(
                        layer_idx,
                        stage_idx,
                        hidden_states.shape[0],
                        hidden_states.dtype,
                        hidden_states.device,
                    )
                )


                if counter == 14000:
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
                
                if counter >= 14000 and counter <= 15099:
                    profiler.step()
                
                if counter == 15099:
                    # profiler会在active阶段结束时自动停止并保存，无需手动stop()
                    print(f"Rank {rank}: Profiler completed at counter {counter}, recorded 100 warmup + 1000 active steps")
                    profiler = None

        for i in range(afd_config.num_afd_stages):
            connector.recv_ffn_output()
        torch.cuda.synchronize()
        time.sleep(0.02)