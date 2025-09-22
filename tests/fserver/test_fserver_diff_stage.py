import torch, os
import time
import fserver_lib as f
import random

is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'
server_count = int(os.environ.get('DMLC_NUM_SERVER','1'))
worker_count = int(os.environ.get('DMLC_NUM_WORKER','1'))

iter_count = 100


if is_worker:
    f.init()
    f.barrier(False, True)
    gpu = os.environ.get('STEPMESH_GPU')
    print(f"worker gpu: {gpu}")
    max_num_tokens = 128
    recv_buffer: list[list[torch.Tensor]] = [
        [
                    torch.empty((max_num_tokens, 4), dtype=torch.bfloat16, device=torch.device('cuda')) for _ in range(1)
                    ] for _ in range(3)
        ]
    send_buffer: list[torch.Tensor] = [
                    torch.empty((max_num_tokens, 4), dtype=torch.bfloat16, device=torch.device('cuda'))
                    for _ in range(3)
        ]

    for i in range(iter_count):
        for stage in range(3):
            send_buffer[stage].random_()
            if i == 0:
                rand_width = 128
            else:   
                rand_width = 128 #random.randint(1, 128)
            push_tensors = [send_buffer[stage][:]]
            print(f"{send_buffer[stage].data_ptr()}, {push_tensors[0].data_ptr()}")
            pull_tensors = [recv_buffer[stage][0]]
            handler =f.push_pull(
                push_tensors,
                [stage],
                pull_tensors,
                [stage + i for i in range(server_count)],
            )
            print(f"iter: {i}, stage: {stage}, handler: {handler}")
            f.wait(handler)
            print(f"match :{pull_tensors}, {push_tensors}")

        print(f"iter: {i}")
    print("worker test done")

elif is_server:
    f.init()
    gpu = os.environ.get('STEPMESH_GPU')
    ret_buffer = torch.rand([1, 2048], dtype=torch.float32, device=f'cuda:{gpu}')
    f.barrier(True, False)
    batches = []
    for i in range(iter_count * 3):
        batches = f.get_batch()
        if len(batches) != 0:
        # recv_tensor_list = batches[0][1]
        # comm_id_list = [batches[0][0]]
        # f.respond_vec(ret_buffer, recv_tensor_list, comm_id_list)  
        # buff = ret_buffer[:,:batches[0][1][0].size(1)]
            print(batches)
        # time.sleep(1)
            # recv_tensor_list = [batches[i][1][0] for i in range(worker_count)]
            # comm_id_list = [batches[i][0] for i in range(worker_count)]

            # f.respond_vec(ret_buffer, recv_tensor_list, comm_id_list)
            for i in range(worker_count):
                f.respond(batches[i][1], batches[i][0], True)
            print(f"iter: {i//3}, stage: {i%3}, server ")

else:
    f.init()
f.stop()
