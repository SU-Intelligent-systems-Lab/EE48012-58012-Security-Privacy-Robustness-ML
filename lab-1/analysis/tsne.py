"""t-SNE of the hidden features of the MNIST MLP (Exercise 2).

python -m analysis.tsne --task mnist --checkpoint checkpoints/<best run>.pt
"""

import os

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import get_splits
from models import build_model
from parameters import get_parameters
from utils import load_checkpoint


def test_subset(test):
    """Return the first 200 test images of each digit.

    Args:
        test (Dataset): The MNIST test set.

    Returns:
        tuple[Tensor, Tensor]: The 2,000 normalised images, shape
            (2000, 1, 28, 28), and their labels, shape (2000,).
    """
    labels = torch.as_tensor(test.targets)
    per_digit = [(labels == c).nonzero().flatten()[:200] for c in range(10)]
    idx = torch.cat(per_digit)
    images = torch.stack([test[int(i)][0] for i in idx])
    return images, labels[idx]


@torch.no_grad()
def hidden_features(model, images, device):
    """Return the hidden features z of the MLP and its predictions.

    model[:3] gives z = ReLU(W1 x + b1); model[3] maps z to the logits.

    Args:
        model (nn.Sequential): The Task A network, in eval mode, on device.
        images (Tensor): Normalised images, shape (N, 1, 28, 28).
        device (str): "cpu" or "cuda".

    Returns:
        tuple[ndarray, ndarray]: The features z, shape (N, 256), and the
            predicted digits, shape (N,).
    """
    z = model[:3](images.to(device))
    pred = model[3](z).argmax(dim=1)
    return z.cpu().numpy(), pred.cpu().numpy()


def embed(z):
    """Reduce features to 50 dimensions with PCA, then embed them in 2D with t-SNE.

    Args:
        z (ndarray): Features, shape (N, D) with D >= 50.

    Returns:
        ndarray: The 2D embedding (perplexity 30, seed 0), shape (N, 2).
    """
    z50 = PCA(n_components=50, random_state=0).fit_transform(z)
    return TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(z50)


def plot_embedding(emb, y, pred, path):
    """Plot the embedding twice, coloured by the true and by the predicted digit.

    Misclassified points are marked with black crosses.

    Args:
        emb (ndarray): The 2D embedding, shape (N, 2).
        y (ndarray): True digits, shape (N,).
        pred (ndarray): Predicted digits, shape (N,).
        path (str): File the figure is saved to.
    """
    wrong = y != pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, digits, title in zip(axes, (y, pred), ("true digit", "predicted digit")):
        sc = ax.scatter(
            emb[:, 0], emb[:, 1], c=digits, cmap="tab10", s=6, vmin=-0.5, vmax=9.5
        )
        ax.scatter(
            emb[wrong, 0],
            emb[wrong, 1],
            c="black",
            marker="x",
            s=25,
            label="misclassified",
        )
        fig.colorbar(sc, ax=ax, ticks=range(10), label=title)
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Embed the hidden features of 2,000 test images of the MLP in --checkpoint."""
    args = get_parameters()
    if args.task != "mnist" or args.checkpoint is None:
        raise SystemExit(
            "t-SNE is the analysis of Task A: pass --task mnist and --checkpoint"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.task).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    run = os.path.basename(args.checkpoint).removesuffix(".pt")

    images, y = test_subset(get_splits(args.task)["test"])
    z, pred = hidden_features(model, images, device)
    emb = embed(z)
    os.makedirs("results", exist_ok=True)
    plot_embedding(emb, y.numpy(), pred, f"results/tsne_{run}.png")
    print(f"{(y.numpy() != pred).sum()} of {len(y)} images misclassified")
    print(f"saved results/tsne_{run}.png")


if __name__ == "__main__":
    main()
