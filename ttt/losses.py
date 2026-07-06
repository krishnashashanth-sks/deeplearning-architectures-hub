import torch.nn.functional as F
import torch

#  Define entropy_loss_fn
def entropy_loss_fn(logits):
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)
    # Compute entropy: -(p * log(p)).sum()
    # Add a small epsilon to log to prevent log(0)
    entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=1).mean()
    return entropy