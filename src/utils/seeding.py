"""
MSc AI Dissertation — Seeding Utility
File: src/utils/seeding.py
"""

import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Sets random seeds across all underlying libraries for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic behaviors behind the scenes
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
