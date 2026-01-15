import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    """Ensure reproducibility across torch, numpy, and python."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
