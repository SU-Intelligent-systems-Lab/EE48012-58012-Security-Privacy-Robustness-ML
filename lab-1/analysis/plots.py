"""Learning curves and accuracy tables of the 3x3 grid (Exercises 1 and 4).

python -m analysis.plots --task cifar10

Reads results/<run>.csv of the nine runs; writes figures and tables to results/.
"""

import os
import pickle

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import DATASETS, get_splits
from models import build_model
from parameters import get_parameters
from utils import load_checkpoint, load_log, run_name

OPTIMIZERS = ["sgd", "momentum", "adam"]
SCHEDULERS = ["constant", "step", "poly"]


def load_history(task, seed):
    """Return the logs of the runs of the 3x3 grid of a task.

    Args:
        task (str): "mnist", "cifar10" or "cifar100".
        seed (int): Seed of the runs.

    Returns:
        dict: Maps (optimizer, scheduler) to the log of that run, a dict of
            lists with one value per epoch; runs without a log are left out.
    """
    history = {}
    for optimizer in OPTIMIZERS:
        for scheduler in SCHEDULERS:
            path = f"results/{run_name(task, optimizer, scheduler, seed)}.csv"
            if os.path.exists(path):
                history[optimizer, scheduler] = load_log(path)
    return history


def plot_curves(history, path="curves.pdf"):
    """Plot test loss and test accuracy against the epoch for several runs.

    Args:
        history (dict): Maps a run name to a dict with the lists "test_loss"
            and "test_acc" (one value per epoch).
        path (str): File the figure is saved to.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, h in history.items():
        epochs = range(1, len(h["test_loss"]) + 1)
        axes[0].plot(epochs, h["test_loss"], label=name)
        axes[1].plot(epochs, h["test_acc"], label=name)
    axes[0].set_ylabel("test loss")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("test accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_per_optimizer(history, key, ylabel, path):
    """Plot one logged quantity against the epoch, one panel per optimiser.

    Args:
        history (dict): Maps (optimizer, scheduler) to the log of that run.
        key (str): Column of the log to plot, for example "train_loss".
        ylabel (str): Label of the y axis.
        path (str): File the figure is saved to.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, optimizer in zip(axes, OPTIMIZERS):
        for colour, scheduler in enumerate(SCHEDULERS):
            if (optimizer, scheduler) in history:
                h = history[optimizer, scheduler]
                ax.plot(h["epoch"], h[key], color=f"C{colour}", label=scheduler)
        ax.set_title(optimizer)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def markdown_table(title, values):
    """Return a 3x3 Markdown table, one row per optimiser, one column per schedule.

    Args:
        title (str): Heading of the table.
        values (dict): Maps (optimizer, scheduler) to a number.

    Returns:
        str: The table; a configuration without a value is shown as "-".
    """
    lines = [f"### {title}", "", "| | " + " | ".join(SCHEDULERS) + " |"]
    lines.append("| --- | --- | --- | --- |")
    for optimizer in OPTIMIZERS:
        cells = []
        for scheduler in SCHEDULERS:
            value = values.get((optimizer, scheduler))
            cells.append("-" if value is None else f"{value:.4f}")
        lines.append(f"| {optimizer} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n"


@torch.no_grad()
def predict(model, loader, device):
    """Return the logits of the model for every image of a loader.

    Args:
        model (nn.Module): Network on device. It is switched to eval mode.
        loader (DataLoader): Yields mini-batches (x, y), in a fixed order.
        device (str): "cpu" or "cuda".

    Returns:
        tuple[Tensor, Tensor]: The logits, shape (N, K), and the labels,
            shape (N,), both on the CPU.
    """
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def topk_accuracy(logits, y, k):
    """Return the top-k accuracy: the true label is among the k largest logits.

    Args:
        logits (Tensor): Shape (N, K).
        y (Tensor): True labels, shape (N,).
        k (int): Number of classes a prediction may name.

    Returns:
        float: Top-k accuracy (between 0 and 1).
    """
    top = logits.topk(k, dim=1).indices
    return (top == y[:, None]).any(dim=1).float().mean().item()


def show_images(images, titles, mean, std, ncols=8, path="images.png"):
    """Save a grid of images, with Normalize undone.

    Args:
        images (Tensor): Shape (N, C, H, W), normalised with mean and std.
        titles (list[str]): One title per image, for example the class name.
        mean, std (list[float]): Per-channel values used in Normalize.
        ncols (int): Number of images per row.
        path (str): File the figure is saved to.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    images = (images.cpu() * std + mean).clamp(0, 1)
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False, figsize=(1.5 * ncols, 1.7 * nrows)
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, title in zip(axes.flat, images, titles):
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def superclasses():
    """Return the superclass of each of the 100 CIFAR-100 classes.

    torchvision gives the class labels only; the superclasses are in the dataset files.

    Returns:
        Tensor: Superclass index (0 to 19) of every class, shape (100,).
    """
    with open("data/cifar-100-python/test", "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    coarse = torch.zeros(100, dtype=torch.long)
    coarse[torch.tensor(raw["fine_labels"])] = torch.tensor(raw["coarse_labels"])
    return coarse


def top1_top5(task, seed, history, device):
    """Write the top-1 and top-5 tables of CIFAR-100 and look at the top-5-only hits.

    The best model is chosen on validation accuracy, never on the test set. Its images
    wrong at top-1 but right at top-5 are counted, by superclass, and 16 are saved.

    Args:
        task (str): "cifar100".
        seed (int): Seed of the nine runs.
        history (dict): Maps (optimizer, scheduler) to the log of that run.
        device (str): "cpu" or "cuda".
    """
    test = get_splits(task)["test"]
    loader = DataLoader(test, batch_size=1000)
    top1, top5, outputs = {}, {}, {}
    for optimizer, scheduler in history:
        path = f"checkpoints/{run_name(task, optimizer, scheduler, seed)}.pt"
        if not os.path.exists(path):
            continue
        model = build_model(task).to(device)
        load_checkpoint(path, model, device=device)
        logits, y = predict(model, loader, device)
        top1[optimizer, scheduler] = topk_accuracy(logits, y, k=1)
        top5[optimizer, scheduler] = topk_accuracy(logits, y, k=5)
        outputs[optimizer, scheduler] = logits
    if not outputs:
        raise SystemExit(f"no checkpoint of a finished {task} run in checkpoints/")
    text = markdown_table("top-1 test accuracy", top1)
    text += markdown_table("top-5 test accuracy", top5)
    print(text)
    with open(f"results/table_top5_{task}.md", "w") as f:
        f.write(text)

    best = max(outputs, key=lambda config: history[config]["val_acc"][-1])
    top = outputs[best].topk(5, dim=1).indices
    pred = top[:, 0]
    hits = (pred != y) & (top == y[:, None]).any(dim=1)
    coarse = superclasses()
    same = (coarse[pred[hits]] == coarse[y[hits]]).float().mean().item()
    print(f"best model: {best[0]} + {best[1]}")
    print(f"{hits.sum().item()} test images are wrong at top-1 but correct at top-5;")
    print(
        f"for {same:.1%} of them the predicted class is in the superclass of the true class"
    )

    idx = hits.nonzero().flatten()[:16]
    images = torch.stack([test[int(i)][0] for i in idx])
    titles = [f"{test.classes[y[i]]} -> {test.classes[pred[i]]}" for i in idx]
    info = DATASETS[task]
    show_images(
        images, titles, info["mean"], info["std"], path=f"results/top5_{task}.png"
    )


def main():
    """Make the learning curves and the accuracy tables of the grid of --task."""
    args = get_parameters()
    task = args.task
    history = load_history(task, args.seed)
    if not history:
        raise SystemExit(f"no log of a {task} run in results/")

    for optimizer in OPTIMIZERS:
        runs = {sch: h for (opt, sch), h in history.items() if opt == optimizer}
        if runs:
            plot_curves(runs, f"results/curves_{task}_{optimizer}.pdf")
    plot_per_optimizer(
        history, "train_loss", "training loss", f"results/train_loss_{task}.pdf"
    )
    plot_per_optimizer(history, "lr", "learning rate", f"results/lr_{task}.pdf")

    final = {config: h["test_acc"][-1] for config, h in history.items()}
    text = markdown_table("final test accuracy", final)
    best = max(history, key=lambda config: history[config]["val_acc"][-1])
    text += f"best run (validation accuracy): {run_name(task, *best, args.seed)}\n"
    print(text)
    with open(f"results/table_{task}.md", "w") as f:
        f.write(text)

    if task == "cifar100":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        top1_top5(task, args.seed, history, device)


if __name__ == "__main__":
    main()
