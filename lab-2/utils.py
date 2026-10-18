"""Helpers: data loaders, seeding, checkpoint saving/loading, plotting.

CIFAR-10 is downloaded from a mirror, since cs.toronto.edu serves at a few KB/s;
torchvision still checks the MD5.
"""

import csv
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


def get_splits():
    """Return the training, validation and test datasets of CIFAR-10.

    ToTensor() scales pixels to [0, 1]; the standardisation is the model's Normalize
    layer. Training images are also cropped and flipped, and a fixed permutation
    (seed 0) holds out 5,000 of them for validation.

    Returns:
        dict[str, Dataset]: The datasets "train" (augmented), "val" and
            "test". Each item is an image tensor of shape (3, 32, 32) in
            [0, 1] and an integer label.
    """
    plain = transforms.ToTensor()
    augment = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_aug = datasets.CIFAR10("data", train=True, download=True, transform=augment)
    train = datasets.CIFAR10("data", train=True, download=True, transform=plain)
    test = datasets.CIFAR10("data", train=False, download=True, transform=plain)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    val_idx, train_idx = perm[:5000].tolist(), perm[5000:].tolist()
    return {
        "train": Subset(train_aug, train_idx),
        "val": Subset(train, val_idx),
        "test": test,
    }


def build_loaders(args):
    """Return the training, validation and test loaders.

    Args:
        args (argparse.Namespace): Uses args.batch_size.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: The training loader
            (shuffled, augmented) and the validation and test loaders (no
            augmentation, no shuffling), all with batches of args.batch_size.
    """
    splits = get_splits()
    train_loader = DataLoader(
        splits["train"], batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(splits["val"], batch_size=args.batch_size)
    test_loader = DataLoader(splits["test"], batch_size=args.batch_size)
    return train_loader, val_loader, test_loader


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


def save_log(path, rows):
    """Write the per-epoch log of a run as a CSV file.

    Args:
        path (str): File to write, for example "results/vgg16_clean_seed0.csv".
        rows (list[dict]): One dict per epoch, all with the same keys; the
            keys become the header line.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, optimizer, epoch):
    """Save the model weights and the training state to a file.

    Args:
        path (str): File to write, for example "checkpoints/vgg16_clean_seed0.pt".
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


def save_gradcam_figure(rows, path):
    """Save one row per image: images and Grad-CAM maps overlaid on them.

    Args:
        rows (list[list[tuple]]): For every image, the panels of its row as
            (image, map, title); map is None for an image shown alone, else a
            Tensor of shape (H, W) in [0, 1].
        path (str): File the figure is saved to.
    """
    ncols = len(rows[0])
    fig, axes = plt.subplots(
        len(rows), ncols, figsize=(2.2 * ncols, 2.3 * len(rows)), squeeze=False
    )
    for row, panels in zip(axes, rows):
        for ax, (image, cam, title) in zip(row, panels):
            ax.imshow(to_image(image))
            if cam is not None:
                ax.imshow(cam.cpu().numpy(), cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tsne_figure(emb, groups, success, path):
    """Save the joint t-SNE embedding twice: by set, and with the attack outcome.

    Args:
        emb (ndarray): Embedding, shape (N, 2).
        groups (ndarray): Set of every point, shape (N,): 0 clean cats,
            1 adversarial cats, 2 clean dogs.
        success (ndarray): For every point, True if it is an adversarial cat
            classified as dog, shape (N,).
        path (str): File the figure is saved to.
    """
    names = ["clean cat", "adversarial cat", "clean dog"]
    colours = ["tab:blue", "tab:red", "tab:green"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for g in range(3):
        axes[0].scatter(*emb[groups == g].T, s=4, color=colours[g], label=names[g])
    axes[0].set_title("the three sets")
    adv = groups == 1
    axes[1].scatter(*emb[~adv].T, s=4, color="lightgray", label="clean cats and dogs")
    axes[1].scatter(
        *emb[adv & success].T,
        s=4,
        color="tab:red",
        label="adversarial, classified as dog",
    )
    axes[1].scatter(
        *emb[adv & ~success].T, s=4, color="black", label="adversarial, not dog"
    )
    axes[1].set_title("adversarial cats by attack success")
    for ax in axes:
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
