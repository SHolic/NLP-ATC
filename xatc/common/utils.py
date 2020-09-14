from functools import wraps
import random
import sys
import time
import torch
import numpy as np


def my_timer(func):
    @wraps(func)
    def wrapper(*args, **kw):
        try:
            func_name = func.__qualname__
        except:
            func_name = func.__name__
        sys.stdout.write(f"[{func_name}]: running ...")
        start_time = time.time()
        ret = func(*args, **kw)
        sys.stdout.flush()
        sys.stdout.write(f"\r[{func_name}]: run time is {round(time.time() - start_time, 3)} s\n")
        return ret

    return wrapper


def batch_print(*s, flag="batch"):
    sys.stdout.flush()
    if flag == "batch":
        sys.stdout.write("\r" + " ".join(s))
    else:
        sys.stdout.write("\r" + " ".join(s) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import time

    for i in range(10):
        time.sleep(1)
        batch_print(str(time.time()), flag="batch")
