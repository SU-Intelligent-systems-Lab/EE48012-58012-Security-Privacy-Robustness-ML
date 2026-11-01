"""VGG-16 with BatchNorm for 32x32 images.

STAGES holds the channels of the 3x3 convolutions; each stage ends with a max-pooling.
"""

import torch.nn as nn

STAGES = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]


class VGG16(nn.Module):
    """Thirteen (conv 3x3 -> BatchNorm -> ReLU) layers, five max-poolings, FC 512 -> 10.

    The poolings reduce 32x32 to 1x1, so z is the 512 channels of the last block.
    """

    def __init__(self, num_classes=10):
        """Create the convolutional layers and the classifier fc.

        Args:
            num_classes (int): Number of output logits.
        """
        super().__init__()
        layers, in_channels = [], 3
        for stage in STAGES:
            for channels in stage:
                layers += [
                    nn.Conv2d(
                        in_channels, channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                ]
                in_channels = channels
            layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def features(self, x):
        """Return the penultimate feature z = phi(x).

        Args:
            x (Tensor): Standardised images, shape (B, 3, 32, 32).

        Returns:
            Tensor: Features z, shape (B, 512).
        """
        return self.layers(x).flatten(1)

    def forward(self, x):
        """Return the class logits.

        Args:
            x (Tensor): Standardised images, shape (B, 3, 32, 32).

        Returns:
            Tensor: Logits, shape (B, num_classes).
        """
        return self.fc(self.features(x))
