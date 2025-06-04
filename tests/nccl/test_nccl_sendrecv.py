import sys
import numpy
import torch
import torch.distributed as dist
import time


def run_benchmark(rank):
    dist.init_process_group(backend="nccl", rank=rank, world_size=2)
    torch.cuda.set_device("cuda:0")

    a2f_tensor_size = 64 * 7168
    a2f_tensor = torch.ones(a2f_tensor_size, dtype=torch.int8, device="cuda:0")

    f2a_tensor_size = 64 * 7168
    f2a_tensor = torch.ones(f2a_tensor_size, dtype=torch.bfloat16, device="cuda:0")

    print("Barriering & Warmup Starts ", rank)
    for _ in range(10):
        if rank == 0:
            dist.recv(a2f_tensor, src=1)
            dist.send(f2a_tensor, dst=1)
        else:
            dist.send(a2f_tensor, dst=0)
            dist.recv(f2a_tensor, src=0)
    print("Barrier & Warmup Done")

    iterations = 1000
    push_costs = []
    pull_costs = []
    overall_costs = []

    for _ in range(iterations):
        if rank == 0:
            dist.recv(a2f_tensor, src=1)
            dist.send(f2a_tensor, dst=1)
        else:
            start = torch.Event(enable_timing=True)
            pull_start = torch.Event(enable_timing=True)
            start.record()
            dist.send(a2f_tensor, dst=0)
            pull_start.record()
            dist.recv(f2a_tensor, src=0)
            end = torch.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            push_costs.append(start.elapsed_time(pull_start))
            pull_costs.append(pull_start.elapsed_time(end))
            overall_costs.append(start.elapsed_time(end))

    if rank != 0:
        print(f"Rank {rank}: Push time per iteration: Mean={numpy.mean(push_costs)} P50={numpy.percentile(push_costs, 50)} P99={numpy.percentile(push_costs, 99)}")
        print(f"Rank {rank}: Pull time per iteration: Mean={numpy.mean(pull_costs)} P50={numpy.percentile(pull_costs, 50)} P99={numpy.percentile(pull_costs, 99)}")
        print(f"Rank {rank}: PushPull time per iteration: Mean={numpy.mean(overall_costs)} P50={numpy.percentile(overall_costs, 50)} P99={numpy.percentile(overall_costs, 99)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    import os
    os.environ['MASTER_ADDR'] = sys.argv[1]
    os.environ['MASTER_PORT'] = '12355'
    run_benchmark(int(sys.argv[2]))
