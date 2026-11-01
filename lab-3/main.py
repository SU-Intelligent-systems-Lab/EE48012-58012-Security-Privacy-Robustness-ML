"""Entry point: reads the arguments and runs the selected --mode."""

import torch

from corruptions import evaluate_corruptions
from models import build_model
from parameters import get_parameters
from robust import evaluate_robustness
from train import fit
from utils import (
    checkpoint_path,
    load_checkpoint,
    make_test_loader,
    make_train_val_loaders,
    set_seed,
)


def load_model(arch, path, device):
    """Return a trained model in eval mode.

    Args:
        arch (str): "vgg16" or "resnet18".
        path (str): Checkpoint written by the train mode.
        device (str): "cpu" or "cuda".

    Returns:
        nn.Module: The model with the stored weights, on device, in eval mode.
    """
    model = build_model(arch).to(device)
    load_checkpoint(path, model, device=device)
    return model.eval()


def main():
    """Parse the arguments, then train or evaluate as --mode says."""
    args = get_parameters()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "train":
        train_loader, val_loader = make_train_val_loaders(args)
        model = build_model(args.arch).to(device)
        fit(model, train_loader, val_loader, args, device)
        return

    test_loader = make_test_loader(args)
    path = args.checkpoint or checkpoint_path(args.arch, args.seed)
    model = load_model(args.arch, path, device)
    if args.mode == "evaluate_robustness":
        evaluate_robustness(model, test_loader, args, device)
    elif args.mode == "evaluate_corruptions":
        evaluate_corruptions(model, test_loader, args, device)


if __name__ == "__main__":
    main()
