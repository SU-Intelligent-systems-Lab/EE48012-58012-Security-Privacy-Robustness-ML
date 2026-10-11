"""All command-line arguments and their defaults."""

import argparse


def get_parameters():
    """Parse and return the command-line arguments of one experiment."""
    p = argparse.ArgumentParser(description="Self-study assignment 1")
    p.add_argument("--task", required=True, choices=["mnist", "cifar10", "cifar100"])
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "momentum", "adam"])
    p.add_argument(
        "--scheduler", default="constant", choices=["constant", "step", "poly"]
    )
    p.add_argument("--lr", type=float, default=0.1, help="base learning rate")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--checkpoint", default=None, help="weights to load")
    return p.parse_args()
