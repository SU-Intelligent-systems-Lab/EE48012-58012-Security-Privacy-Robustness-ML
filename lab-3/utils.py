"""Helpers: data loaders, seeding, checkpoint saving/loading, plotting.

CIFAR-10 is downloaded from a mirror, since cs.toronto.edu serves at a few KB/s;
torchvision still checks the MD5.
"""

import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

datasets.CIFAR10.url = (
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
)

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def make_train_val_loaders(args):
    """Return the training loader and the held-out validation loader of CIFAR-10.

    A fixed permutation (seed 0) holds out --val-size training images, so the split is
    the same in every run. Two datasets on the same files give the training images crop
    and flip, and the validation images ToTensor() only, which scales pixels to [0, 1].

    Args:
        args (argparse.Namespace): Uses args.val_size and args.batch_size.

    Returns:
        tuple[DataLoader, DataLoader]: The training loader (shuffled, augmented)
            and the validation loader (no augmentation, no shuffling).
    """
    augment = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_aug = datasets.CIFAR10("data", train=True, download=True, transform=augment)
    train = datasets.CIFAR10(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    val_idx, train_idx = perm[: args.val_size].tolist(), perm[args.val_size :].tolist()
    train_loader = DataLoader(
        Subset(train_aug, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(Subset(train, val_idx), batch_size=args.batch_size)
    return train_loader, val_loader


def make_test_loader(args):
    """Return the loader of the CIFAR-10 test set, with ToTensor() only, not shuffled.

    Args:
        args (argparse.Namespace): Uses args.batch_size.

    Returns:
        DataLoader: The 10,000 test images in [0, 1] and their labels.
    """
    test = datasets.CIFAR10(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )
    return DataLoader(test, batch_size=args.batch_size)


def set_seed(seed):
    """Seed Python, NumPy and PyTorch, including the GPU, so that a run can be repeated.

    Args:
        seed (int): The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checkpoint_path(arch, seed):
    """Return the file of the clean model of an architecture.

    Args:
        arch (str): "vgg16" or "resnet18".
        seed (int): Seed of the training run.

    Returns:
        str: For example "checkpoints/vgg16_clean_seed0.pt".
    """
    return f"checkpoints/{arch}_clean_seed{seed}.pt"


def model_name(args):
    """Return the name of the evaluated checkpoint, used in the result files.

    Args:
        args (argparse.Namespace): Uses args.checkpoint, or else args.arch and
            args.seed for the clean model.

    Returns:
        str: The file name without ".pt", for example "vgg16_pgdat_seed0_best".
    """
    path = args.checkpoint or checkpoint_path(args.arch, args.seed)
    return os.path.basename(path).removesuffix(".pt")


def save_log(path, rows):
    """Write rows as a CSV file, for example the per-epoch log of a run.

    Args:
        path (str): File to write, for example "results/vgg16_pgdat_seed0.csv".
        rows (list[dict]): One dict per row, all with the same keys; the keys
            become the header line.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, optimizer, epoch):
    """Save the model weights and the training state to a file.

    Args:
        path (str): File to write, for example "checkpoints/vgg16_pgdat_seed0_best.pt".
        model (nn.Module): Network whose state_dict is saved.
        optimizer (torch.optim.Optimizer): Optimiser whose state is saved,
            needed only to resume training.
        epoch (int): Number of epochs trained so far.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load a file written by save_checkpoint into an existing model.

    map_location lets the file load on a machine without GPU.

    Args:
        path (str): File to read.
        model (nn.Module): Network of the same architecture; its weights are
            replaced in place.
        optimizer (torch.optim.Optimizer, optional): If given, its state is
            restored as well.
        device (str): Device the tensors are loaded to, "cpu" or "cuda".

    Returns:
        int: The epoch stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]


def to_image(x):
    """Return an image tensor (C, H, W) as an array (H, W, C) that matplotlib can show.

    Args:
        x (Tensor): Image in [0, 1], shape (3, H, W).

    Returns:
        ndarray: The image, shape (H, W, 3).
    """
    return x.detach().cpu().permute(1, 2, 0).numpy()


def save_adversarial_example(x, x_adv, pred, adv_pred, path):
    """Save a clean image, its adversarial version and the perturbation, side by side.

    The perturbation delta = x_adv - x is rescaled to [0, 1] for display.

    Args:
        x (Tensor): Clean image in [0, 1], shape (3, H, W).
        x_adv (Tensor): Adversarial image, same shape.
        pred (int): Predicted class of the clean image.
        adv_pred (int): Predicted class of the adversarial image.
        path (str): File the figure is saved to.
    """
    delta = x_adv - x
    delta = (delta - delta.min()) / (delta.max() - delta.min() + 1e-12)
    titles = [
        f"clean: {CLASSES[pred]}",
        f"adversarial: {CLASSES[adv_pred]}",
        "perturbation (rescaled)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for ax, img, title in zip(axes, (x, x_adv, delta), titles):
        ax.imshow(to_image(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
