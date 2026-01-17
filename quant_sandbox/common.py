import numpy as np
import os
import random
import torch


def set_seed(seed = int(os.getenv("SEED", "42"))):
    # 1. Set seed for Python's built-in random number generator
    random.seed(seed)

    # 2. Set seed for NumPy (used by Polars/Scikit-Learn internally sometimes)
    np.random.seed(seed)

    # 3. Set seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # 4. Ensure deterministic behavior in cuDNN (optional, strictly for GPUs)
    # This might slightly slow down training but ensures exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_dir():
    if os.getenv("QUANT_SANDBOX_BASE_DIR"):
        quant_sandbox_dir = os.getenv("QUANT_SANDBOX_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        quant_sandbox_dir = os.path.join(cache_dir, "quant_sandbox")
    os.makedirs(quant_sandbox_dir, exist_ok=True)
    return quant_sandbox_dir
