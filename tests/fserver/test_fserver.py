import torch, os
import time
import fserver_lib as f
is_worker = os.environ.get('DMLC_ROLE') == 'worker'
is_server = os.environ.get('DMLC_ROLE') == 'server'

f.init()

if is_worker:
    gpu = os.environ.get('STEPMESH_GPU')
    push_tensors = [
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}'),
    ]
    pull_tensors = [
        torch.rand([1, 8192], dtype=torch.float32, device=f'cuda:{gpu}')
    ]
    handler = f.push_pull(
        push_tensors, 
        [i for i in range(len(push_tensors))], 
        pull_tensors, 
        [i for i in range(len(pull_tensors))]
    )
    f.wait(handler)
    assert torch.allclose(sum(push_tensors), pull_tensors[0])
    print("worker test done")

elif is_server:
    gpu = os.environ.get('STEPMESH_GPU')
    torch.set_default_device('cuda:{}'.format(gpu))
    res = []
    while len(res) == 0:
        time.sleep(1)
        res = f.get_batch()
    print(res)
    for r in res:
        comm_id, batch, _ = r
        f.respond([sum(batch)], comm_id)

f.stop()
