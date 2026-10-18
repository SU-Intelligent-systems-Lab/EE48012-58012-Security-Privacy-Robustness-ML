"""Latent features and joint t-SNE of clean and adversarial samples (Exercise 4).

CAT and DOG are the CIFAR-10 class indices.
"""

import json
import os

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from robust import attack
from utils import save_tsne_figure

CAT, DOG = 3, 5


@torch.no_grad()
def penultimate(model, x):
    """Return the 512-dimensional penultimate features z = phi(x).

    Args:
        model (nn.Sequential): Normalize layer followed by the network.
        x (Tensor): Images in [0, 1], shape (B, C, H, W).

    Returns:
        Tensor: Features, shape (B, 512).
    """
    normalize, net = model[0], model[1]
    return net.features(normalize(x))


def tsne(model, loader, args, device):
    """Embed clean cats, adversarial cats (targeted to dog) and clean dogs in one t-SNE.

    Only correctly classified test images are used. Prints, in the 512-dimensional
    space, the distances to the cat and dog centroids and the 10-NN dog fraction; saves
    results/tsne_<arch>.json and results/tsne_<arch>.png.

    Args:
        model (nn.Module): Classifier in eval mode, Normalize layer included.
        loader (DataLoader): Test loader without augmentation or shuffling.
        args (argparse.Namespace): The arguments of parameters.py.
        device (str): "cpu" or "cuda".
    """
    cats, dogs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        cats.append(x[(y == CAT) & (pred == CAT)])
        dogs.append(x[(y == DOG) & (pred == DOG)])
    cats, dogs = torch.cat(cats), torch.cat(dogs)

    adv_cats, adv_pred = [], []
    for x in cats.split(args.batch_size):
        y = torch.full((len(x),), CAT, device=device)
        target = torch.full((len(x),), DOG, device=device)
        x_adv = attack(model, x, y, args, target)
        adv_cats.append(x_adv)
        with torch.no_grad():
            adv_pred.append(model(x_adv).argmax(dim=1))
    adv_cats, success = torch.cat(adv_cats), torch.cat(adv_pred) == DOG

    z_cat = torch.cat([penultimate(model, x) for x in cats.split(args.batch_size)])
    z_adv = torch.cat([penultimate(model, x) for x in adv_cats.split(args.batch_size)])
    z_dog = torch.cat([penultimate(model, x) for x in dogs.split(args.batch_size)])

    def summary(z):
        """Return the distances to the two centroids and the 10-NN dog fraction."""
        neighbours = NearestNeighbors(n_neighbors=10).fit(clean.cpu().numpy())
        _, idx = neighbours.kneighbors(z.cpu().numpy())
        dog_votes = torch.from_numpy(is_dog[idx]).sum(dim=1)
        return {
            "n": len(z),
            "distance_to_cat_centroid": (z - z_cat.mean(dim=0))
            .norm(dim=1)
            .mean()
            .item(),
            "distance_to_dog_centroid": (z - z_dog.mean(dim=0))
            .norm(dim=1)
            .mean()
            .item(),
            "fraction_mostly_dog_neighbours": (dog_votes > 5).float().mean().item(),
        }

    clean = torch.cat([z_cat, z_dog])
    is_dog = torch.cat([torch.zeros(len(z_cat)), torch.ones(len(z_dog))]).bool().numpy()
    numbers = {
        "targeted_success_rate": success.float().mean().item(),
        "adversarial_cats": summary(z_adv),
        "adversarial_cats_classified_as_dog": (
            summary(z_adv[success]) if success.any() else None
        ),
        "adversarial_cats_not_dog": (
            summary(z_adv[~success]) if (~success).any() else None
        ),
    }
    print(
        f"{args.arch}: {len(z_cat)} cats, {len(z_dog)} dogs; "
        f"{numbers['targeted_success_rate']:.2%} of the adversarial cats are classified as dog"
    )
    for key in (
        "adversarial_cats",
        "adversarial_cats_classified_as_dog",
        "adversarial_cats_not_dog",
    ):
        s = numbers[key]
        if s is not None:
            print(
                f"  {key} ({s['n']}): distance to the cat centroid {s['distance_to_cat_centroid']:.2f}, "
                f"to the dog centroid {s['distance_to_dog_centroid']:.2f}, "
                f"mostly dog neighbours {s['fraction_mostly_dog_neighbours']:.2%}"
            )

    z = torch.cat([z_cat, z_adv, z_dog]).cpu().numpy()
    z50 = PCA(n_components=50, random_state=0).fit_transform(z)
    emb = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(z50)
    groups = torch.cat(
        [torch.zeros(len(z_cat)), torch.ones(len(z_adv)), torch.full((len(z_dog),), 2)]
    )
    point_success = torch.cat(
        [
            torch.zeros(len(z_cat), dtype=torch.bool),
            success.cpu(),
            torch.zeros(len(z_dog), dtype=torch.bool),
        ]
    )
    os.makedirs("results", exist_ok=True)
    with open(f"results/tsne_{args.arch}.json", "w") as f:
        json.dump(numbers, f, indent=2)
    save_tsne_figure(
        emb, groups.numpy(), point_success.numpy(), f"results/tsne_{args.arch}.png"
    )
