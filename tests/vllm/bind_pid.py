import os
import sys
import torch
import multiprocessing

def get_system_cpu_count():
    try:
        import subprocess
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'CPU(s):' in line:
                    cpu_count = int(line.split(':')[1].strip())
                    return cpu_count
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return 192


def local_rank_to_gpu_id(local_rank):

    cuda_visible_devices = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")] if "CUDA_VISIBLE_DEVICES" in os.environ else None
    ascend_rt_visible_devices = [
        int(x)
        for x in os.environ.get(
            "ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
        ).split(",")
    ]
    if os.environ.get("GPU_VENDOR") == "ASCEND":
        return ascend_rt_visible_devices[local_rank]
    else:
        return cuda_visible_devices[local_rank] if cuda_visible_devices else local_rank
# %%
def bind_pid(pid, local_rank):
    if pid == 0:
        pid = os.getpid()
    gpu_id = local_rank_to_gpu_id(local_rank)
    core_count = get_system_cpu_count()    
    core_list  = list(range(core_count))
    core_per_socket = core_count//4
    # mask 10-20,34-44,58-68,82-92,106-116,130-140,154-164,178-188
    mask = list(range(10,21)) + list(range(34,45)) + list(range(58,69)) + list(range(82,93)) + list(range(106,117)) + list(range(130,141)) + list(range(154,165)) + list(range(178,189))

    core_list = []
    core_per_gpu = core_per_socket // 2
    gpu_offset_id = gpu_id // 2
    for v in list(range(core_per_gpu * gpu_offset_id, core_per_gpu * (gpu_offset_id + 1))) + list(range(core_per_gpu * gpu_offset_id + core_per_socket * 2, core_per_gpu * (gpu_offset_id + 1) + core_per_socket * 2)):
        if v not in mask:
            core_list.append(v)

    try:
        os.sched_setaffinity(pid, core_list)
        assgined = os.sched_getaffinity(pid)
        print(f"Set GPU {gpu_id} pid {pid} affinity to {core_list}, result as {assgined}")
    except Exception as e:
        print(f"Error set GPU {gpu_id} pid {pid} affinity to {core_list}: {e}")


def _get_numa_node_by_gpu_uuid(target_uuid):
    import subprocess

    """
    根据GPU的UUID获取对应的NUMA节点ID。
  
    参数:
        target_uuid (str): GPU的UUID。
  
    返回:
        int: NUMA节点ID。
  
    异常:
        RuntimeError: 如果执行命令失败或读取文件出错。
        ValueError: 如果UUID不存在或PCI格式无效。
    """
    # 获取GPU的UUID和PCI总线ID列表
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv"],
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"执行nvidia-smi失败: {e}") from e
    except FileNotFoundError:
        raise RuntimeError("未找到nvidia-smi命令，请确保NVIDIA驱动已安装。")

    # 解析输出，建立UUID到PCI总线ID的映射
    uuid_pci_map = {}
    for line in output.strip().split("\n")[1:]:  # 跳过标题行
        if line.strip():
            uuid, pci_bus_id = [s.strip() for s in line.split(",", 1)]
            uuid_pci_map[uuid] = pci_bus_id

    if target_uuid not in uuid_pci_map:
        raise ValueError(f"未找到UUID为 {target_uuid} 的GPU")

    pci_bus_id = uuid_pci_map[target_uuid]

    # 将PCI总线ID格式化为sysfs路径格式（如0000:00:00.0）
    parts = pci_bus_id.split(":")
    if len(parts) != 3:
        raise ValueError(f"无效的PCI总线ID格式: {pci_bus_id}")
    domain = parts[0][-4:].zfill(4)  # 提取后四位作为domain，补零
    formatted_pci = f"{domain}:{parts[1]}:{parts[2]}".lower()

    # 读取NUMA节点信息
    sysfs_path = f"/sys/bus/pci/devices/{formatted_pci}/numa_node"
    if not os.path.exists(sysfs_path):
        raise RuntimeError(f"PCI设备 {formatted_pci} 不存在")

    try:
        with open(sysfs_path, "r") as f:
            numa_node = f.read().strip()
    except IOError as e:
        raise RuntimeError(f"无法读取 {sysfs_path}: {e}") from e

    # 解析NUMA节点ID
    try:
        numa_node_id = int(numa_node)
    except ValueError:
        raise RuntimeError(f"无效的NUMA节点值: {numa_node}")

    return numa_node_id



def get_current_gpu_numa_node_id():
    device = torch.cuda.current_device()
    nvml_device_hanlder = torch.cuda._get_pynvml_handler(device)
    import pynvml

    if hasattr(pynvml, "nvmlDeviceGetNumaNodeId"):
        try:
            return pynvml.nvmlDeviceGetNumaNodeId(nvml_device_hanlder)
        except pynvml.NVMLError:
            # fall back to get from uuid
            pass
    uuid = pynvml.nvmlDeviceGetUUID(nvml_device_hanlder)
    return _get_numa_node_by_gpu_uuid(uuid)


def set_numa_affinity(i, numa_node_id: int = None, strict_bind=False):
    import ctypes as ct
    from ctypes.util import find_library

    class bitmask_t(ct.Structure):
        _fields_ = [
            ("size", ct.c_ulong),
            ("maskp", ct.POINTER(ct.c_ulong)),
        ]

    LIBNUMA = ct.CDLL(find_library("numa"))
    LIBNUMA.numa_parse_nodestring.argtypes = [ct.c_char_p]
    LIBNUMA.numa_parse_nodestring.restype = ct.POINTER(bitmask_t)
    LIBNUMA.numa_run_on_node_mask.argtypes = [ct.POINTER(bitmask_t)]
    LIBNUMA.numa_run_on_node_mask.restype = ct.c_int
    if strict_bind:
        LIBNUMA.numa_set_membind.argtypes = [ct.POINTER(bitmask_t)]
        LIBNUMA.numa_set_membind.restype = ct.c_void_p
    else:
        LIBNUMA.numa_set_preferred.argtypes = [ct.POINTER(bitmask_t)]
        LIBNUMA.numa_set_preferred.restype = ct.c_void_p
    LIBNUMA.numa_num_configured_nodes.argtypes = []
    LIBNUMA.numa_num_configured_nodes.restype = ct.c_int

    def numa_bind(nid: int):
        bitmask = LIBNUMA.numa_parse_nodestring(bytes(str(nid), "ascii"))
        LIBNUMA.numa_run_on_node_mask(bitmask)
        if strict_bind:
            LIBNUMA.numa_set_membind(bitmask)
        else:
            LIBNUMA.numa_set_preferred(bitmask)

    if numa_node_id is None:
        try:
            numa_node_id = get_current_gpu_numa_node_id()
        except Exception as e:
            numa_node_id = int(i / 4)
            print(f"Failed to get NUMA node ID, {e} {numa_node_id}")
            return

    try:
        numa_bind(numa_node_id)
        print("BIND_NUMA: success")
    except Exception as e:
        print(f"BIND_NUMA: {e}")
    bind_pid(os.getpid(),i)
    _SET_AFFINITY = True


if __name__ == "__main__":
    bind_pid(0,0)