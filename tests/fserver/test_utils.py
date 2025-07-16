import os

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

def get_worker_rank(key : int):
    return (key % (1 << 24)) / (1 << 16)