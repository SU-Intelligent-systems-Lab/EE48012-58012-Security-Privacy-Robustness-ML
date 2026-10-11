"""Datasets, transforms and build_loaders().

CIFAR is downloaded from a mirror, since cs.toronto.edu serves at a few KB/s;
torchvision still checks the MD5.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

datasets.CIFAR10.url = (
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
)
datasets.CIFAR100.url = (
    "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"
)

DATASETS = {
    "mnist": {
        "torchvision_class": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "cifar10": {
        "torchvision_class": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "torchvision_class": datasets.CIFAR100,
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
    },
}


def get_splits(task):
    """Return the training, validation and test datasets of a task.

    Images are normalised; CIFAR training images are also cropped and flipped. A fixed
    permutation (seed 0, not args.seed) holds out the same 5,000 images in every run.

    Args:
        task (str): "mnist", "cifar10" or "cifar100".

    Returns:
        dict[str, Dataset]: The datasets "train" (augmented), "val" and
            "test". Each item is an image tensor of shape (C, H, W) and an
            integer label.
    """
    info = DATASETS[task]
    plain = [transforms.ToTensor(), transforms.Normalize(info["mean"], info["std"])]
    augment = []
    if task != "mnist":
        augment = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

    dataset_class = info["torchvision_class"]
    train_aug = dataset_class(
        "data", train=True, download=True, transform=transforms.Compose(augment + plain)
    )
    train = dataset_class(
        "data", train=True, download=True, transform=transforms.Compose(plain)
    )
    test = dataset_class(
        "data", train=False, download=True, transform=transforms.Compose(plain)
    )

    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    val_idx, train_idx = perm[:5000].tolist(), perm[5000:].tolist()
    return {
        "train": Subset(train_aug, train_idx),
        "val": Subset(train, val_idx),
        "test": test,
    }


def build_loaders(args):
    """Return the training, validation and test loaders of a task.

    Args:
        args (argparse.Namespace): Uses args.task and args.batch_size.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: The training loader
            (shuffled, augmented, batches of args.batch_size) and the
            validation and test loaders (in order, batches of 1,000).
    """
    splits = get_splits(args.task)
    train_loader = DataLoader(
        splits["train"], batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(splits["val"], batch_size=1000)
    test_loader = DataLoader(splits["test"], batch_size=1000)
    return train_loader, val_loader, test_loader
