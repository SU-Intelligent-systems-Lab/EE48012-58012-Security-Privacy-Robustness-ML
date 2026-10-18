"""build_model(), the Normalize layer, VGG-16 and ResNet-18.

MEAN and STD are those of the CIFAR-10 training images, per colour channel.
"""

import torch
import torch.nn as nn

from models.resnet import ResNet18
from models.vgg import VGG16

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    """Standardise every colour channel: (x - mean) / std.

    It is the first layer of every model, so the attacks see images in [0, 1].
    """

    def __init__(self, mean, std):
        """Store the per-channel mean and standard deviation as buffers.

        Args:
            mean (tuple[float]): Mean of each channel.
            std (tuple[float]): Standard deviation of each channel.
        """
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        """Return the standardised images.

        Args:
            x (Tensor): Images in [0, 1], shape (B, C, H, W).

        Returns:
            Tensor: Standardised images, same shape.
        """
        return (x - self.mean) / self.std


def build_model(arch):
    """Return the untrained model of an architecture, with the Normalize layer in front.

    Args:
        arch (str): "vgg16" or "resnet18".

    Returns:
        nn.Sequential: Normalize followed by the network; model[1] is the
            network itself, with features() for the 512-dimensional
            penultimate feature and fc for the last layer.
    """
    if arch == "vgg16":
        net = VGG16(num_classes=10)
    elif arch == "resnet18":
        net = ResNet18(num_classes=10)
    else:
        raise ValueError(arch)
    return nn.Sequential(Normalize(MEAN, STD), net)
