import torch, os
import time
import fserver_lib as f
import random

is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'
server_count = int(os.environ.get('DMLC_NUM_SERVER','1'))
worker_count = int(os.environ.get('DMLC_NUM_WORKER','1'))

iter_count = 2000

max_num_tokens = 128

if is_worker:
    f.init()
    f.barrier(False, True)
    gpu = os.environ.get('STEPMESH_GPU')
    rank = int(os.environ.get('DMLC_NODE_RANK', '0'))
    print(f"worker gpu: {gpu}")

    recv_buffer: list[list[torch.Tensor]] = [
        [
                    torch.empty((max_num_tokens ,7196), dtype=torch.bfloat16, device=torch.device('cuda')).contiguous() for _ in range(server_count)
                    ] for _ in range(3)
        ]
    send_buffer: list[torch.Tensor] = [
                    torch.empty((max_num_tokens ,7196), dtype=torch.bfloat16, device=torch.device('cuda')).contiguous()
                    for _ in range(3)
        ]

    for i in range(iter_count):
        h = []
        pslist_cost = []
        recv = []
        cnt = i%1000
        for stage in range(3):
            send_buffer[stage].random_()
            if i == 0:
                rand_width = max_num_tokens
            else:   
                rand_width  = random.randint(1, max_num_tokens)
            data = torch.rand([rand_width, 7196], dtype=torch.bfloat16, device=torch.device('cuda'))
                
            push_tensors = [send_buffer[stage][:rand_width,:]]
            
            push_tensors[0].copy_(data)
             
            pull_tensors = [recv_buffer[stage][i][:rand_width,:] for i in range(server_count)]
            print(f"buffer 0x{send_buffer[stage].data_ptr():x}, tensor 0x{push_tensors[0].data_ptr():x}")
            key = stage + int(1e6)
            handler =f.push_pull(
                push_tensors,
                [key],
                pull_tensors,
                [1000 + stage + i for i in range(server_count)],
            )

            beg = time.perf_counter_ns()
            f.wait(handler, 1000 * 500)
            end = time.perf_counter_ns()
            costs = f.fetch_trace(handler)
            pslist_cost.append(costs)
            recv.append((end-beg))

            # print(f"wait time: {wait_time}")
            # print(f"Client, iter: {i}, stage: {stage}, handler: {handler}; shape {push_tensors[0].shape}, buffer 0x{send_buffer[stage].data_ptr():x}, tensor 0x{push_tensors[0].data_ptr():x},  Match : {torch.allclose(push_tensors[0], pull_tensors[0])}")
        print("\n")
    print("worker test done")
    f.stop()

elif is_server:
    f.init()
    gpu = os.environ.get('STEPMESH_GPU')
    ret_buffer = torch.rand([8 * max_num_tokens,7196], dtype=torch.bfloat16, device=f'cuda:{gpu}')
    f.barrier(True, False)
    batches = []
    for i in range(iter_count * 3):
        batches = f.get_batch()
        if len(batches) != 0:
        # recv_tensor_list = batches[0][1]
        # comm_id_list = [batches[0][0]]
        # f.respond_vec(ret_buffer, recv_tensor_list, comm_id_list)  
            # buff = ret_buffer[:batches[0][1][0].size(0),:].contiguous()
            # buff.copy_(batches[0][1][0])
            # print(batches)
            # print(f"Server iter: {i//3}, stage: {i%3}; buff 0x{ret_buffer.data_ptr():x}, tensor 0x{buff.data_ptr():x},")
            for j in range(worker_count):
                f.respond(batches[j][1], batches[j][0], need_event=True)
    f.stop()        

else:
    f.init()
    f.stop()


