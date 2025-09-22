import os
from vllm.config import AFDConfig, VllmConfig,ParallelConfig
from stepmesh_connector import StepMeshAFDConnector,AFDConnectorMetadata
import torch
import time
import numpy as np
import fserver_lib as ps
from cycle import get_cycles_per_ms

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
time.sleep(5)
connector.init_afd_connector()
print(f"--------- rank {rank} local_rank {local_rank} init ---------")
def execute():
    for layer_idx in range(61):
        for stage_idx in range(afd_config.num_afd_stages):
            hidden_states = torch.randn(4, 7168, dtype=torch.bfloat16, device="cuda")
            e = connector.send_attn_output(
                hidden_states,
                AFDConnectorMetadata.create_attention_metadata(
                    layer_idx,
                    stage_idx,
                    hidden_states.shape[0],
                    hidden_states.dtype,
                    hidden_states.device,
                )
            )
            torch.cuda._sleep(int(cycle_per_ms * 0.29))
            connector.recv_ffn_output()
if __name__ == "__main__":
    counter = 0
    while True:
        if counter % 10 == 0:
            torch.cuda.synchronize()
        execute()