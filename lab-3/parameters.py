"""All command-line arguments and their defaults.

get_parameters() is Listing 1 of the handout on top of the arguments of Assignment 2.
"""

import argparse


def fraction(s):
    """Parse a number given as a decimal or a fraction, e.g. "0.5" or "8/255"."""
    num, _, den = s.partition("/")
    return float(num) / float(den or 1)


def get_parameters():
    """Parse and return the command-line arguments of one experiment."""
    p = argparse.ArgumentParser(description="Self-study assignment 3")
    p.add_argument("--arch", required=True, choices=["vgg16", "resnet18"])
    p.add_argument(
        "--mode",
        default="train",
        choices=["train", "evaluate_robustness", "evaluate_corruptions"],
    )
    p.add_argument("--attack", default="pgd", choices=["fgsm", "pgd"])
    p.add_argument("--norm", default="linf", choices=["linf", "l2"])
    p.add_argument("--eps", type=fraction, default=8 / 255)
    p.add_argument(
        "--alpha",
        type=fraction,
        default=None,
        help="step size; default 2.5 * eps / steps",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument(
        "--targeted", action="store_true", help="targeted attack; untargeted if omitted"
    )
    p.add_argument(
        "--target-mode",
        default="next",
        choices=["next", "random", "least"],
        help="how the target class is chosen",
    )
    p.add_argument(
        "--n-eval",
        type=int,
        default=1000,
        help="number of test images that are attacked",
    )
    p.add_argument("--checkpoint", default=None, help="weights to load")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.05, help="base learning rate")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--train-mode",
        default="standard",
        choices=["standard", "pgd_at"],
        help="clean training or PGD adversarial training",
    )
    p.add_argument("--train-eps", type=fraction, default=8 / 255)
    p.add_argument("--train-alpha", type=fraction, default=2 / 255)
    p.add_argument("--train-steps", type=int, default=10)
    p.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="training images held out for model selection",
    )
    p.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[50, 75],
        help="epochs at which the learning rate is divided by 10",
    )
    p.add_argument("--cifar10c-root", default="data/CIFAR-10-C")
    p.add_argument(
        "--corruptions",
        nargs="+",
        default=None,
        help="subset of the 15 corruptions; all if omitted",
    )
    p.add_argument("--severities", type=int, nargs="+", default=[3, 5])
    return p.parse_args()
