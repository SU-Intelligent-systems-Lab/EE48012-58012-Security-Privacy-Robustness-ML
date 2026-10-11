"""Evaluation: loss and accuracy on a data split."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model on a data split.

    Args:
        model (nn.Module): Network to evaluate. It is switched to eval mode.
        loader (DataLoader): Yields mini-batches (x, y) of the split.
        device (str): "cpu" or "cuda".

    Returns:
        tuple[float, float]: Mean loss and accuracy (between 0 and 1).
    """
    model.eval()
    loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return loss / n, correct / n
