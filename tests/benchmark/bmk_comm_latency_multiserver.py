import threading
from queue import Queue

import torch, os
import fserver_lib as f
import optimus
from optimus.ops import silu_dot as optimus_silu_dot
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
bsz, num_token, dim = 1, 64, 7168
num_iters = 100000000

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

    out_tensors_keys.append([gen_pull_key(i, mb) for i in range(len(out_tensors_buffers))])


print_queue = Queue()


def print_thread():
    time_list = []
    cost_list = []

    while True:
        start, end, costs = print_queue.get()
        time_list.append((start, end))
        cost_list.append(costs)

        if len(time_list) == 500:
            overall_list = []
            python_req_costs = [[] for _ in range(3)]
            req_send_costs = [[] for _ in range(3)]
            net_costs = [[] for _ in range(3)]
            req_recv_costs = [[] for _ in range(3)]
            process_costs = [[] for _ in range(3)]
            rsp_send_costs = [[] for _ in range(3)]
            rsp_recv_costs = [[] for _ in range(3)]
            python_rsp_costs = [[] for _ in range(3)]

            for i in range(len(time_list)):
                start, end = time_list[i]

                overall_list.append((end - start)/1e6)
                for j in range(3):
                    python_req_costs[j].append(cost_list[i][j][0] / 1000000.0 - start/1e6)
                    python_rsp_costs[j].append(end/1e6 - cost_list[i][j][7] / 1000000.0)
                    req_send_costs[j].append((cost_list[i][j][1] - cost_list[i][j][0]) / 1000000.0)
                    req_recv_costs[j].append((cost_list[i][j][3] - cost_list[i][j][2]) / 1000000.0)
                    process_costs[j].append((cost_list[i][j][4] - cost_list[i][j][3]) / 1000000.0)
                    rsp_send_costs[j].append((cost_list[i][j][5] - cost_list[i][j][4]) / 1000000.0)
                    rsp_recv_costs[j].append((cost_list[i][j][7] - cost_list[i][j][6]) / 1000000.0)
                    net_costs[j].append(
                        (cost_list[i][j][6] - cost_list[i][j][1] - (cost_list[i][j][5] - cost_list[i][j][2])) / 1000000.0)

            def _str_formater(data):
                sorted_data = np.sort(data)
                return f"{np.mean(data):0.3f}\t{np.percentile(data, 50):0.3f}\t{np.percentile(data, 99):0.3f}\t{sorted_data[-1]:0.3f} {sorted_data[-2]:0.3f} {sorted_data[-3]:0.3f}"

            print(f"comm bmk gpu={gpu}: mean={np.mean(overall_list):0.3f}ms, "
                  f"p50={np.percentile(overall_list, 50):0.3f}ms, "
                  f"p99={np.percentile(overall_list, 99):0.3f}ms, "
                  f"max={np.max(overall_list):0.3f}ms\n"
                  f"\t gpu={gpu} push 1:\n"
                  f"\t\tpython_req\t{_str_formater(python_req_costs[0])}\n"
                  f"\t\treq_send\t{_str_formater(req_send_costs[0])}\n"
                  f"\t\treq_recv\t{_str_formater(req_recv_costs[0])}\n"
                  f"\t\tprocess  \t{_str_formater(process_costs[0])}\n"
                  f"\t\trsp_send\t{_str_formater(rsp_send_costs[0])}\n"
                  f"\t\trsp_recv\t{_str_formater(rsp_recv_costs[0])}\n"
                  f"\t\tpython_rsp\t{_str_formater(python_rsp_costs[0])}\n"
                  f"\t\tnet_cost\t{_str_formater(net_costs[0])}\n"
                  f"\t gpu={gpu} push 2:\n"
                  f"\t\tpython_req\t{_str_formater(python_req_costs[1])}\n"
                  f"\t\treq_send\t{_str_formater(req_send_costs[1])}\n"
                  f"\t\treq_recv\t{_str_formater(req_recv_costs[1])}\n"
                  f"\t\tprocess  \t{_str_formater(process_costs[1])}\n"
                  f"\t\trsp_send\t{_str_formater(rsp_send_costs[1])}\n"
                  f"\t\trsp_recv\t{_str_formater(rsp_recv_costs[1])}\n"
                  f"\t\tpython_rsp\t{_str_formater(python_rsp_costs[1])}\n"
                  f"\t\tnet_cost\t{_str_formater(net_costs[1])}\n"
                  f"\t gpu={gpu} pull 1:\n"
                  f"\t\tpython_req\t{_str_formater(python_req_costs[2])}\n"
                  f"\t\treq_send\t{_str_formater(req_send_costs[2])}\n"
                  f"\t\treq_recv\t{_str_formater(req_recv_costs[2])}\n"
                  f"\t\tprocess  \t{_str_formater(process_costs[2])}\n"
                  f"\t\trsp_send\t{_str_formater(rsp_send_costs[2])}\n"
                  f"\t\trsp_recv\t{_str_formater(rsp_recv_costs[2])}\n"
                  f"\t\tpython_rsp\t{_str_formater(python_rsp_costs[2])}\n"
                  f"\t\tnet_cost\t{_str_formater(net_costs[2])}\n")

            time_list = []
            cost_list = []


if is_worker:
    th = threading.Thread(target=print_thread)
    th.start()
    f.barrier(True, True)
    q = Queue()
    time_list = []
    net_cost_list = [[] for _ in range(bsz + 1 + bsz)]
    idx = 0

    def worker():
        global idx
        start = f.get_nanosecond()
        handler = f.push_pull(
            inp_tensors_buffers[idx % 3],
            inp_tensors_keys[idx % 3],
            out_tensors_buffers[idx % 3],
            out_tensors_keys[idx % 3],
        )
        idx += 1

        handlers = f.get_all_handlers(handler)
        f.wait(handler)
        end = f.get_nanosecond()

        costs = [f.fetch_trace(handler) for handler in handlers]
        print_queue.put((start, end, costs))

    print(f"start to run {num_iters}")
    for itr in range(num_iters):
        worker()

elif is_server:
    ret_buffer = torch.rand([65535, dim], dtype=torch.bfloat16, device='cuda')
    count = 0

    def server():
        global count
        iter_count = 0
        while True:
            batches = f.get_batch()
            if len(batches) != 0:
                iter_count += 1
                recv_tensor_list = [batches[i][1][0] for i in range(worker_count)]
                comm_id_list = [batches[i][0] for i in range(worker_count)]

                f.respond_vec(ret_buffer, recv_tensor_list, comm_id_list)
                if iter_count == num_iters:
                    break
    server()

f.stop()
