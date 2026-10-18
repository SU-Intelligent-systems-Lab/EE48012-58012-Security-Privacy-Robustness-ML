"""All command-line arguments and their defaults.

get_parameters() is Listing 1 of the handout.
"""

import argparse


def fraction(s):
    """Parse a number given as a decimal or a fraction, e.g. "0.5" or "8/255"."""
    num, _, den = s.partition("/")
    return float(num) / float(den or 1)


def get_parameters():
    """Parse and return the command-line arguments of one experiment."""
    p = argparse.ArgumentParser(description="Self-study assignment 2")
    p.add_argument("--arch", required=True, choices=["vgg16", "resnet18"])
    p.add_argument(
        "--mode",
        default="evaluate_robustness",
        choices=["train", "evaluate_robustness", "gradcam", "tsne", "transfer"],
    )
    p.add_argument(
        "--source-arch",
        choices=["vgg16", "resnet18"],
        help="model the attack is computed on (transfer mode)",
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
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "momentum", "adam"])
    p.add_argument(
        "--scheduler",
        default="constant",
        choices=["constant", "step", "poly", "cosine"],
    )
    p.add_argument("--lr", type=float, default=0.1, help="base learning rate")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()
