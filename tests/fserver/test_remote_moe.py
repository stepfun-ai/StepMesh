import torch, os
import time
import fserver_lib as f
import optimus
from optimus.ops import silu_dot as optimus_silu_dot
from test_utils import *
from torch.profiler import ProfilerActivity
from contextlib import nullcontext
import argparse

def setup_seed(seed=42):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Benchmark settings
parser = argparse.ArgumentParser(description='Fserver Remote MOE Test')
parser.add_argument('--num_iters', '-n', type=int, default=100)
parser.add_argument('--profile', '-p', action='store_true', default=False)
parser.add_argument('--nccl', action='store_true', default=False,)
parser.add_argument('--sccl', action='store_true', default=False,)

args = parser.parse_args()

is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'
gpu = os.environ.get('STEPAF_GPU', '0')
local_rank_num = int(os.environ.get('DMLC_GROUP_SIZE', '1'))
node_rank = int(os.environ.get('DMLC_NODE_RANK', '0'))
rank = node_rank * local_rank_num + int(gpu)

if is_server:
    if args.nccl:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:10091',
            world_size=8,
            rank=rank,
        )
        torch.distributed.barrier()
    elif args.sccl:
        from stepccl import tensor_parallel_comm as tp_comm
        tp_comm.init(rank, 8, 0, 1024*1024*1024)

world_size = 8
bsz, num_token, dim = 1, 1024, 7168
num_iters = args.num_iters
expert_dist_key = 1 << 16

setup_seed(42)
w1 = [
    torch.rand([dim, 2*dim], dtype=torch.bfloat16, device=f'cuda:{gpu}') for _ in range(bsz)
]
w2 = [
    torch.rand([dim, dim], dtype=torch.bfloat16, device=f'cuda:{gpu}') for _ in range(bsz)
]

allgather_buffer = torch.empty([num_token * world_size, 1, dim], dtype=torch.bfloat16, device= f'cuda:{gpu}')

torch.cuda.set_device(int(gpu))

f.init()

def moe_computation(input_tensor, w1, w2, expert_token_cnt):
    """
    Perform the MOE computation.
    """
    if is_server:
        if args.nccl:
            out_buffers = [torch.empty_like(input_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(out_buffers, input_tensor)
            out = torch.cat(out_buffers)
        elif args.sccl:
            tp_comm.allgather_async(allgather_buffer, input_tensor.unsqueeze(1), world_size)
            out = allgather_buffer.squeeze(1)
            tp_comm.sync()

    out = torch.ops.OptimusMoe.gemm_grouped(input_tensor, w1, expert_token_cnt)
    out = optimus_silu_dot(out)
    out = torch.ops.OptimusMoe.gemm_grouped(out, w2, expert_token_cnt)
    return out


if is_worker:
    # prepare initial buffers (each key should have a dedicated buffer)
    expert_token_cnt_buffer = torch.tensor([num_token for _ in range(bsz)], dtype=torch.int32, device=f'cuda:{gpu}')
    tokens_buffers = [
        torch.rand([num_token, dim], dtype=torch.bfloat16, device=f'cuda:{gpu}') for _ in range(bsz)
    ]
    tokens_buffers.append(expert_token_cnt_buffer)
    inp_tensors_buffers = tokens_buffers
    inp_tensors_keys = [gen_push_key(i) for i in range(len(tokens_buffers))]
    out_tensors_buffers = [
        torch.rand([num_token, dim], dtype=torch.bfloat16, device=f'cuda:{gpu}') for _ in range(bsz)
    ]
    out_tensors_keys = [gen_pull_key(i) for i in range(len(out_tensors_buffers))]

    f.barrier(True, True)

    def worker():
        expert_token_cnt = torch.tensor([num_token for _ in range(bsz)], dtype=torch.int32, device=f'cuda:{gpu}')
        expert_token_cnt_buffer.copy_(expert_token_cnt)
        tokens = [
            torch.rand([num_token, dim], dtype=torch.bfloat16, device=f'cuda:{gpu}') for _ in range(bsz)
        ]

        for x in range(len(tokens)):
            tokens_buffers[x].copy_(tokens[x])

        # clear output buffer
        for t in out_tensors_buffers:
            t.zero_()

        handler = f.push_pull(
            inp_tensors_buffers,
            inp_tensors_keys, 
            out_tensors_buffers,
            out_tensors_keys,
        )

        f.wait(handler)

        # do check 
        gt = moe_computation(torch.cat(tokens), w1, w2, expert_token_cnt)
        gt = gt.split(expert_token_cnt.tolist())
        for i in range(len(out_tensors_buffers)):
            assert torch.allclose(gt[i], out_tensors_buffers[i]), \
                f"check failed gpu={gpu}, index={i}, local={gt[i]}, remote={out_tensors_buffers[i]}"
        print(f"worker check done: gpu={gpu}")
    
    for itr in range(num_iters):
        print(f"iter start: gpu={gpu}, itr={itr}")
        worker()

elif is_server:
    registered_tensor_buffer = torch.rand([num_token * bsz, dim], dtype=torch.bfloat16, device=f'cuda:{gpu}')
    f.register_recv_buffer(registered_tensor_buffer,
                           [0 for _ in range(bsz)],
                           [gen_push_key(i, 0, 0) for i in range(bsz)])
    # barrier for registration ops
    f.barrier(True, True)
    print("barrier done")
    if args.profile and rank == 0:
        profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(".trace", worker_name="fserver_trace"),
            record_shapes=True, profile_memory=True, with_stack=True, with_flops=True, with_modules=True
        )
    else:
        profiler = nullcontext()

    def server():
        res_list = []
        while len(res_list) == 0:
            time.sleep(0.1)
            res_list = f.get_batch()
        # assert len(res) == 1, "Expected exactly one batch"

        for res in res_list:
            comm_id, batch, keys = res
            expert_token_cnt = batch[-1]

            if get_worker_rank(keys[0]) == 0:
                global_ptr = registered_tensor_buffer.data_ptr()
                for i in range(bsz):
                    assert global_ptr == batch[i].data_ptr()
                    global_ptr += batch[i].numel() * batch[i].element_size()
                out = moe_computation(registered_tensor_buffer, w1, w2, expert_token_cnt)
            else:
                out = moe_computation(torch.cat(batch[:-1]), w1, w2, expert_token_cnt)

            f.respond(out.split(expert_token_cnt.tolist()), comm_id)
    
    with profiler as prof:
        for _ in range(num_iters):
            server()
            if hasattr(prof, 'step'): 
                prof.step()


f.stop()