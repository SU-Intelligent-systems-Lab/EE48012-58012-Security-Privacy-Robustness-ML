"""CIFAR-10-C: the CIFAR10C dataset and evaluate_corruptions().

CATEGORY gives the category of each of the 15 corruptions. The corrupted images are
classified as they are, with no attack and no gradient.
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import model_name, save_log

CATEGORY = {
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "impulse_noise": "noise",
    "defocus_blur": "blur",
    "glass_blur": "blur",
    "motion_blur": "blur",
    "zoom_blur": "blur",
    "snow": "weather",
    "frost": "weather",
    "fog": "weather",
    "brightness": "weather",
    "contrast": "digital",
    "elastic_transform": "digital",
    "pixelate": "digital",
    "jpeg_compression": "digital",
}


class CIFAR10C(Dataset):
    """The 10,000 CIFAR-10 test images under one corruption at one severity."""

    def __init__(self, root, corruption, severity):
        """Memory-map <corruption>.npy and keep the rows of the severity.

        Rows 10,000 (s - 1) to 10,000 s - 1 of a file hold severity s; labels.npy has
        the same layout.

        Args:
            root (str): Folder with the 15 corruption files and labels.npy.
            corruption (str): Name of the corruption, for example "fog".
            severity (int): Severity, from 1 to 5.
        """
        rows = slice(10000 * (severity - 1), 10000 * severity)
        self.images = np.load(f"{root}/{corruption}.npy", mmap_mode="r")[rows]
        self.labels = np.load(f"{root}/labels.npy")[rows]
        self.transform = transforms.ToTensor()

    def __len__(self):
        """Return the number of images, 10,000."""
        return len(self.labels)

    def __getitem__(self, i):
        """Return image i in [0, 1], shape (3, 32, 32), and its label.

        np.array reads the image from the file into memory; ToTensor() is the same as
        for the clean test set.
        """
        return self.transform(np.array(self.images[i])), int(self.labels[i])


@torch.no_grad()
def accuracy(model, loader, device):
    """Return the accuracy of the model on the images of a loader.

    Args:
        model (nn.Module): Classifier in eval mode, taking images in [0, 1].
        loader (DataLoader): Yields mini-batches (x, y).
        device (str): "cpu" or "cuda".

    Returns:
        float: Accuracy, between 0 and 1.
    """
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)


def evaluate_corruptions(model, test_loader, args, device):
    """Measure the accuracy on the clean test set and on every corruption and severity.

    The rows are written to results/corruptions.csv, which holds every evaluated model;
    the rows of the same model are replaced. The clean test set is the row with
    corruption "clean" and severity 0. training is the name of the checkpoint without
    the architecture, for example "pgdat_seed0_best".

    Args:
        model (nn.Module): Classifier taking images in [0, 1]. It is switched to
            eval mode.
        test_loader (DataLoader): The clean CIFAR-10 test set.
        args (argparse.Namespace): Uses args.cifar10c_root, args.corruptions,
            args.severities and args.batch_size.
        device (str): "cpu" or "cuda".

    Returns:
        dict[tuple[str, int], float]: Accuracy of every (corruption, severity).
    """
    model.eval()
    training = model_name(args).removeprefix(f"{args.arch}_")
    acc = accuracy(model, test_loader, device)
    print(f"clean test set: accuracy {acc:.2%}")
    accuracies = {("clean", 0): acc}
    rows = [
        {
            "arch": args.arch,
            "training": training,
            "corruption": "clean",
            "category": "clean",
            "severity": 0,
            "accuracy": acc,
        }
    ]
    for corruption in args.corruptions or list(CATEGORY):
        for severity in args.severities:
            data = CIFAR10C(args.cifar10c_root, corruption, severity)
            acc = accuracy(model, DataLoader(data, batch_size=args.batch_size), device)
            print(f"{corruption}, severity {severity}: accuracy {acc:.2%}")
            accuracies[corruption, severity] = acc
            rows.append(
                {
                    "arch": args.arch,
                    "training": training,
                    "corruption": corruption,
                    "category": CATEGORY[corruption],
                    "severity": severity,
                    "accuracy": acc,
                }
            )

    path = "results/corruptions.csv"
    if os.path.exists(path):
        with open(path, newline="") as f:
            others = [
                row
                for row in csv.DictReader(f)
                if (row["arch"], row["training"]) != (args.arch, training)
            ]
        rows = others + rows
    os.makedirs("results", exist_ok=True)
    save_log(path, rows)
    return accuracies
