"""One file per architecture, and build_model() to select one."""

from models.cnn import SimpleCNN
from models.mlp import build_mlp
from models.resnet_cifar import ResNet18


def build_model(task):
    """Return the untrained model of a task.

    Args:
        task (str): "mnist" (Task A), "cifar10" (Task B) or "cifar100" (Task C).

    Returns:
        nn.Module: The two-layer MLP, the simple CNN with 10 classes, or
            ResNet-18 with 100 classes, on the CPU.
    """
    if task == "mnist":
        return build_mlp()
    if task == "cifar10":
        return SimpleCNN(num_classes=10)
    if task == "cifar100":
        return ResNet18(num_classes=100)
    raise ValueError(task)
