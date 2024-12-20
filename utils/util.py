import torch
import random
import numpy as np

def set_new_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)