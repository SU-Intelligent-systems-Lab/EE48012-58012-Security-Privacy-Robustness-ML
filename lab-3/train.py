"""Training: train_one_epoch(), train_epoch_adv() and fit() with model selection.

WEIGHT_DECAY is fixed by the training recipe of this assignment.
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from attacks import pgd_linf
from robust import clean_and_pgd_accuracy
from utils import save_checkpoint, save_log

WEIGHT_DECAY = 5e-4


def train_one_epoch(model, loader, optimizer, device):
    """Train the model for one epoch on clean images.

    Args:
        model (nn.Module): Network to train. It is switched to train mode.
        loader (DataLoader): Yields mini-batches (x, y); x has shape (B, C, H, W).
        optimizer (torch.optim.Optimizer): Optimiser over model.parameters().
        device (str): "cpu" or "cuda".

    Returns:
        tuple[float, float]: Mean training loss and accuracy over the epoch.
    """
    model.train()
    total, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total / n, correct / n


def train_epoch_adv(model, loader, optimizer, args, device):
    """Train the model for one epoch of PGD adversarial training.

    Every mini-batch is attacked with pgd_linf in eval mode, so the attack does not
    change the BatchNorm statistics; the weights are then updated in train mode on the
    adversarial examples only.

    Args:
        model (nn.Module): Network to train.
        loader (DataLoader): Yields mini-batches (x, y); x has shape (B, C, H, W).
        optimizer (torch.optim.Optimizer): Optimiser over model.parameters().
        args (argparse.Namespace): Uses args.train_eps, args.train_alpha and
            args.train_steps.
        device (str): "cpu" or "cuda".

    Returns:
        tuple[float, float]: Mean loss and accuracy on the adversarial examples.
    """
    total, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.eval()
        x_adv = pgd_linf(
            model, x, y, args.train_eps, args.train_alpha, args.train_steps
        )
        model.train()
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total / n, correct / n


def fit(model, train_loader, val_loader, args, device):
    """Train for args.epochs epochs and keep the checkpoint with the best PGD accuracy.

    SGD with momentum 0.9; the learning rate is divided by 10 at args.milestones. After
    every epoch the clean and the PGD accuracy on the validation set, with the training
    attack, are logged to results/<run>.csv. checkpoints/<run>_best.pt is saved whenever
    the PGD accuracy is the best so far, checkpoints/<run>_last.pt at the end. run is
    vgg16_pgdat_seed0 for --train-mode pgd_at and vgg16_clean_seed0 for standard.

    Args:
        model (nn.Module): Network to train, already on device.
        train_loader (DataLoader): Shuffled, augmented training batches.
        val_loader (DataLoader): Held-out validation batches.
        args (argparse.Namespace): The arguments of parameters.py.
        device (str): "cpu" or "cuda".
    """
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    training = "pgdat" if args.train_mode == "pgd_at" else "clean"
    run = f"{args.arch}_{training}_seed{args.seed}"
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    best, log, start = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        if args.train_mode == "pgd_at":
            train_loss, train_acc = train_epoch_adv(
                model, train_loader, optimizer, args, device
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device
            )
        scheduler.step()
        val_acc, val_pgd_acc = clean_and_pgd_accuracy(
            model,
            val_loader,
            args.train_eps,
            args.train_alpha,
            args.train_steps,
            device,
        )
        log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_pgd_acc": val_pgd_acc,
                "lr": lr,
                "time": time.time() - start,
            }
        )
        print(", ".join(f"{k} {v:.4g}" for k, v in log[-1].items()))
        save_log(f"results/{run}.csv", log)
        if val_pgd_acc > best:
            best = val_pgd_acc
            save_checkpoint(f"checkpoints/{run}_best.pt", model, optimizer, epoch)
    save_checkpoint(f"checkpoints/{run}_last.pt", model, optimizer, args.epochs)
