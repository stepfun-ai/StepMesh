
import threading
from queue import Queue

import torch, os
import fserver_lib as f
import optimus
import numpy as np


def gen_push_key(private_key, microbatch=0, worker_rank=-1):
    """
    Generate a key for push tensors based on microbatch, worker rank, and private key.
    :param private_key: your own key, ranging from 0-255, can be used for identify different tensors
    :param microbatch: microbatch id
    :param worker_rank: current worker rank, otherwise retrieving it from environ
    :return: the key for fserver
    """
    assert 0 <= private_key < 256, f"illegal private key: {private_key}"
    if worker_rank == -1:
        if "DMLC_NODE_RANK" in os.environ:
            worker_rank = int(os.environ["DMLC_NODE_RANK"])
        else:
            worker_rank = 0
    return private_key + microbatch * (1 << 8) + worker_rank * (1 << 16)


def gen_pull_key(private_key, microbatch=0, worker_rank=-1):
    """
    Generate a key for pull tensors based on microbatch, worker rank, and private key.
    :param private_key: your own key, ranging from 0-255, can be used for identify different tensors
    :param microbatch: microbatch id
    :param worker_rank: current worker rank, otherwise retrieving it from environ
    :return: the key for fserver
    """
    assert 0 <= private_key < 256, f"illegal private key: {private_key}"
    if worker_rank == -1:
        if "DMLC_NODE_RANK" in os.environ:
            worker_rank = int(os.environ["DMLC_NODE_RANK"])
        else:
            worker_rank = 0
    return private_key + microbatch * (1 << 8) + worker_rank * (1 << 16) + (1 << 24)


def setup_seed(seed=42):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'
server_count = int(os.environ.get('DMLC_NUM_SERVER','1'))
worker_count = int(os.environ.get('DMLC_NUM_WORKER','1'))
gpu = os.environ.get('STEPMESH_GPU', '0')
local_rank_num = int(os.environ.get('DMLC_GROUP_SIZE', '1'))
node_rank = int(os.environ.get('DMLC_NODE_RANK', '0'))
rank = node_rank * local_rank_num + int(gpu)
bsz, num_token, dim = 1, 8, 8
num_iters = 24
setup_seed(42)
torch.cuda.set_device(int(gpu))

f.init()

# prepare initial buffers (each key should have a dedicated buffer)

inp_tensors_buffers = []
inp_tensors_keys = []
out_tensors_buffers = []
out_tensors_keys = []

for mb in range(3):
    expert_token_cnt_buffer = torch.tensor([num_token for _ in range(bsz)], dtype=torch.int32, device=f'cuda:{gpu}')
    tokens_buffers = [
        torch.rand([num_token, dim], dtype=torch.bfloat16, device=f'cuda:{gpu}') for _ in range(bsz)
    ]
    tokens_buffers.append(expert_token_cnt_buffer)
    inp_tensors_buffers.append(tokens_buffers)
    inp_tensors_keys.append([gen_push_key(i, mb) for i in range(len(tokens_buffers))])

    o_tensors = []
    for _ in range(server_count ):
        o_tensors += [torch.rand([num_token, dim], dtype=torch.bfloat16, device=f'cuda:{gpu}') for _ in range(bsz)]
    out_tensors_buffers.append(o_tensors)

    out_tensors_keys.append([gen_pull_key(i, mb) for i in range(len(o_tensors))])


print_queue = Queue()

if is_worker:
    f.barrier(False, True)
    q = Queue()
    time_list = []
    net_cost_list = [[] for _ in range(bsz + 1 + bsz)]
    idx = 0
    device = torch.cuda.device(int(gpu))
    signal_flag_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
    ack_flag_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)

    signal_flag_dev = f.map_pinned_tensor(signal_flag_host, int(gpu))
    ack_flag_dev = f.map_pinned_tensor(ack_flag_host, int(gpu))
    
    sequence_tensor = torch.zeros(1, dtype=torch.int64, device=f'cuda:{gpu}')

    expected_sequence = 1
    # 创建 sequence tensor 用于 CUDA Graph
    # Graph 中不能使用动态的 Python int 值，需要使用 tensor
    profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=0, warmup=0, active=24),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_logs/rank_{rank}', use_gzip=True),
                        record_shapes=True,
                        with_stack=True,
                    )    

    thread_stop = threading.Event()
    def cpu_handle_thread():
        global  expected_sequence, idx, signal_flag_dev, ack_flag_dev, stream
        while True:
            signal_value = signal_flag_host.item()
            if signal_value < expected_sequence:
                if thread_stop.is_set():
                    print("Cpu handle thread stop")
                    break
                continue
            handler = f.push_pull(
                inp_tensors_buffers[idx % 3],
                inp_tensors_keys[idx % 3],
                out_tensors_buffers[idx % 3],
                out_tensors_keys[idx % 3],
                need_event=False,
            )
            f.wait(handler)
            expected_sequence += 1
            idx += 1
            print(f"wait done signal_value:{signal_value} expected_seq:{expected_sequence} ")
            ack_flag_host.fill_(signal_value)


    th = threading.Thread(target=cpu_handle_thread)

    print(f"start to run {num_iters}")

    graph = torch.cuda.CUDAGraph()

    ack_flag_host.fill_(num_iters)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        for i in range(num_iters):
            f.seq_add_one(sequence_tensor)  # 更新 tensor 值
            f.write_flag(signal_flag_dev, sequence_tensor)
            torch.cuda._sleep(1000)
            f.wait_flag(ack_flag_dev, sequence_tensor)
        graph.capture_end()
    torch.cuda.synchronize()

    def reset_control():
        sequence_tensor.fill_(0)
        ack_flag_host.fill_(0)
        expected_sequence = 1
        torch.cuda.synchronize()
    reset_control()

    th.start()
    profiler.start()
    # big graph replay
    graph.replay()
    profiler.step()
    
    # small graph replay
    # for itr in range(num_iters):
    #     graph.replay()
    #     profiler.step()
    torch.cuda.synchronize()
    print(f"Worker stop")
    profiler.stop()
    thread_stop.set()
    th.join()

elif is_server:
    ret_buffer = torch.rand([65535, dim], dtype=torch.bfloat16, device='cuda')
    count = 0
    f.barrier(True, False)
    def server():
        global count
        iter_count = 0
        while True:
            batches = f.get_batch()
            # print(f"Server get batch: {batches}")
            if len(batches) != 0:
                iter_count += 1
                recv_tensor_list = [batches[i][1][0] for i in range(worker_count)]
                comm_id_list = [batches[i][0] for i in range(worker_count)]
                # torch.cuda._sleep(10000)
                f.respond_vec(ret_buffer, recv_tensor_list, comm_id_list)
                print(f"Server iter: {iter_count}/{num_iters}")
                if iter_count == num_iters:
                    break
    server()
    torch.cuda.synchronize()
    print(f"Server stop")

print("Fserver Stop")
f.stop()
