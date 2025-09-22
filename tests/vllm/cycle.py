import torch
import numpy as np
def get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch.cuda._sleep
    """

    def measure() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # Get 10 values and remove the 2 max and 2 min and return the avg.
    # This is to avoid system disturbance that skew the results, e.g.
    # the very first cuda call likely does a bunch of init, which takes
    # much longer than subsequent calls.
    #
    # Tested on both Tesla V100, Quadro GP100, Titan RTX, RTX 3090 GPUs
    # and seems to return stable values. Therefore, we enable caching
    # using lru_cache decorator above.
    num = 10
    vals = [measure() for _ in range(num)]
    vals = sorted(vals)
    return np.mean(vals[2 : num - 2])