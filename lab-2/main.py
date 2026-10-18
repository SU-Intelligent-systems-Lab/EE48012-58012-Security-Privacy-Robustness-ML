"""Entry point: reads the arguments and runs the selected --mode."""

import torch

from features import tsne
from gradcam import gradcam
from models import build_model
from parameters import get_parameters
from robust import evaluate_robustness, evaluate_transfer
from train import fit
from utils import build_loaders, checkpoint_path, load_checkpoint, set_seed


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
    """Parse the arguments, then train, attack or analyse as --mode says."""
    args = get_parameters()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = build_loaders(args)
    if args.mode == "train":
        model = build_model(args.arch).to(device)
        fit(model, train_loader, val_loader, test_loader, args, device)
        return

    path = args.checkpoint or checkpoint_path(args.arch, args.seed)
    model = load_model(args.arch, path, device)
    if args.mode == "evaluate_robustness":
        evaluate_robustness(model, test_loader, args, device)
    elif args.mode == "gradcam":
        gradcam(model, test_loader, args, device)
    elif args.mode == "tsne":
        tsne(model, test_loader, args, device)
    elif args.mode == "transfer":
        if args.source_arch is None:
            raise SystemExit("--mode transfer needs --source-arch")
        source = load_model(
            args.source_arch, checkpoint_path(args.source_arch, args.seed), device
        )
        evaluate_transfer(model, source, test_loader, args, device)


if __name__ == "__main__":
    main()
