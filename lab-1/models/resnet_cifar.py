"""Task C: ResNet-18 with the CIFAR stem for CIFAR-100."""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 conv/BatchNorm layers with an identity or 1x1 shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        """Create the residual branch and, when shapes change, a 1x1 projection.

        Args:
            in_channels (int): Number of channels of the input.
            out_channels (int): Number of channels of the output.
            stride (int): Stride of the first convolution; 2 halves H and W.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """Return ReLU(branch(x) + shortcut(x)): add first, ReLU after.

        Args:
            x (Tensor): Feature maps, shape (B, in_channels, H, W).

        Returns:
            Tensor: Feature maps, shape (B, out_channels, H / stride, W / stride).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 whose 7x7 stride-2 stem and max-pool are one 3x3 stride-1 conv.

    Stages of 2 basic blocks with 64, 128, 256, 512 channels at 32, 16, 8, 4 pixels;
    GAP gives z in R^512, followed by the linear classifier.
    """

    def __init__(self, num_classes=100):
        """Create the CIFAR stem, the four stages, and the classifier fc.

        Args:
            num_classes (int): Number of output logits.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.stage(64, 64, stride=1)
        self.layer2 = self.stage(64, 128, stride=2)
        self.layer3 = self.stage(128, 256, stride=2)
        self.layer4 = self.stage(256, 512, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def stage(in_channels, out_channels, stride):
        """Return two basic blocks; the first one may downsample.

        Args:
            in_channels (int): Number of channels of the input.
            out_channels (int): Number of channels of both blocks.
            stride (int): Stride of the first block.

        Returns:
            nn.Sequential: One stage of the network.
        """
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def features(self, x):
        """Return the penultimate feature z = phi(x), the GAP output.

        The maps go from (B, 64, 32, 32) to (B, 512, 4, 4); stride 2 halves H and W.

        Args:
            x (Tensor): Images, shape (B, 3, 32, 32).

        Returns:
            Tensor: Features z, shape (B, 512).
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return x.flatten(1)

    def forward(self, x):
        """Return the class logits.

        Args:
            x (Tensor): Images, shape (B, 3, 32, 32).

        Returns:
            Tensor: Logits, shape (B, num_classes).
        """
        return self.fc(self.features(x))
