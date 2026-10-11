"""Task A: two-layer MLP for MNIST."""

import torch.nn as nn


def build_mlp():
    """Return the Task A network, Eq. (21): 784 -> 256 -> 10 (about 204k parameters).

    It is built in a function, so that the seed of main.py decides the weights.

    Returns:
        nn.Sequential: Maps (B, 1, 28, 28) to logits (B, 10); mlp[:3] gives z (B, 256).
    """
    mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    return mlp
