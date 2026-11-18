from __future__ import annotations

import numpy as np
import torch

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device):
    starts = torch.randint(len(dataset) - context_length, (batch_size,)).tolist()
    x = np.stack([dataset[start: start + context_length] for start in starts])
    y = np.stack([dataset[start + 1: start + context_length + 1] for start in starts])
    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )