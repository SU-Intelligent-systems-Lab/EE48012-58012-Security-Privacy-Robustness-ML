"""Grad-CAM maps for VGG-16 on clean and adversarial images (Exercise 3).

N_FIGURE images are shown; the map statistics are averaged over N_STATS images.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from robust import attack
from utils import CLASSES, save_gradcam_figure

N_FIGURE = 6
N_STATS = 100


def last_conv(model):
    """Return the last convolutional layer of a model.

    Args:
        model (nn.Module): The model, Normalize layer included.

    Returns:
        nn.Conv2d: Its last convolution; in VGG-16 it works on 2x2 maps.
    """
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]


def grad_cam(model, layer, x, cls):
    """Return the Grad-CAM maps of the classes cls, upsampled to the image size.

    Hooks keep A^k and dS_c/dA^k; L^c = ReLU(sum_k beta_k^c A^k), with beta_k^c the
    mean of the gradient over the positions. Eval mode keeps the images of a batch
    independent.

    Args:
        model (nn.Module): Classifier in eval mode, taking images in [0, 1].
        layer (nn.Module): The layer whose activations are explained.
        x (Tensor): Images, shape (B, C, H, W).
        cls (Tensor): The class c of every image, shape (B,).

    Returns:
        Tensor: Maps min-max normalised to [0, 1], shape (B, H, W).
    """
    stored = {}

    def keep_activations(module, inputs, output):
        """Forward hook: keep A, the output of the layer."""
        stored["A"] = output

    def keep_gradients(module, grad_input, grad_output):
        """Backward hook: keep dS_c/dA, the gradient of the layer's output."""
        stored["dA"] = grad_output[0]

    forward_hook = layer.register_forward_hook(keep_activations)
    backward_hook = layer.register_full_backward_hook(keep_gradients)
    x = x.clone().requires_grad_(True)
    score = model(x).gather(1, cls[:, None]).sum()
    torch.autograd.grad(score, x)
    forward_hook.remove()
    backward_hook.remove()

    A, dA = stored["A"].detach(), stored["dA"].detach()
    beta = dA.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((beta * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear")[:, 0]
    lo, hi = cam.amin((1, 2), keepdim=True), cam.amax((1, 2), keepdim=True)
    return (cam - lo) / (hi - lo + 1e-12)


def correlation(a, b):
    """Return the Pearson correlation of every pair of maps.

    Args:
        a (Tensor): Maps, shape (B, H, W).
        b (Tensor): Maps, shape (B, H, W).

    Returns:
        Tensor: Correlations, shape (B,).
    """
    a = a.flatten(1) - a.flatten(1).mean(dim=1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(dim=1, keepdim=True)
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def top_iou(a, b, fraction=0.2):
    """Return the overlap (IoU) of the top regions of every pair of maps.

    Args:
        a (Tensor): Maps, shape (B, H, W).
        b (Tensor): Maps, shape (B, H, W).
        fraction (float): Share of the pixels with the largest values that
            forms the top region of a map.

    Returns:
        Tensor: Intersection over union of the two top regions, shape (B,).
    """

    def top(m):
        """Return the mask of the top pixels of every map; topk avoids ties."""
        m = m.flatten(1)
        k = round(fraction * m.size(1))
        mask = torch.zeros_like(m, dtype=torch.bool)
        return mask.scatter_(1, m.topk(k, dim=1).indices, True)

    ta, tb = top(a), top(b)
    return (ta & tb).sum(dim=1) / (ta | tb).sum(dim=1)


def gradcam(model, loader, args, device):
    """Compare Grad-CAM maps of correct test images and of their adversarial examples.

    The figure shows, for 6 images: clean, map of y, adversarial, map of its prediction
    y', map of y. The correlation and the top-20% IoU with the clean map are averaged
    over 100 images. Saves results/gradcam_<arch>.json and results/gradcam_<arch>.png.

    Args:
        model (nn.Module): VGG-16 in eval mode, Normalize layer included.
        loader (DataLoader): Test loader without augmentation or shuffling.
        args (argparse.Namespace): The arguments of parameters.py.
        device (str): "cpu" or "cuda".
    """
    if args.arch != "vgg16":
        raise SystemExit(
            "Grad-CAM is computed for VGG-16 (Exercise 3): use --arch vgg16"
        )
    layer = last_conv(model)
    rows, stats, count = (
        [],
        {"corr_pred": [], "iou_pred": [], "corr_y": [], "iou_y": []},
        0,
    )
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            correct = model(x).argmax(dim=1) == y
        x, y = x[correct][: N_STATS - count], y[correct][: N_STATS - count]
        x_adv = attack(model, x, y, args)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1)
        cam_clean = grad_cam(model, layer, x, y)
        cam_adv = grad_cam(model, layer, x_adv, adv_pred)
        cam_adv_y = grad_cam(model, layer, x_adv, y)
        stats["corr_pred"] += correlation(cam_clean, cam_adv).tolist()
        stats["iou_pred"] += top_iou(cam_clean, cam_adv).tolist()
        stats["corr_y"] += correlation(cam_clean, cam_adv_y).tolist()
        stats["iou_y"] += top_iou(cam_clean, cam_adv_y).tolist()
        for i in range(min(len(y), N_FIGURE - len(rows))):
            true, new = CLASSES[y[i]], CLASSES[adv_pred[i]]
            rows.append(
                [
                    (x[i], None, f"clean: {true}"),
                    (x[i], cam_clean[i], f"Grad-CAM of {true}"),
                    (x_adv[i], None, f"adversarial: {new}"),
                    (x_adv[i], cam_adv[i], f"Grad-CAM of {new}"),
                    (x_adv[i], cam_adv_y[i], f"Grad-CAM of {true}"),
                ]
            )
        count += len(y)
        if count >= N_STATS:
            break

    means = {key: sum(v) / len(v) for key, v in stats.items()}
    means["n_images"] = count
    print(f"Grad-CAM, mean over {count} correctly classified test images:")
    print(
        f"  clean map of y vs adversarial map of y': correlation {means['corr_pred']:.3f}, "
        f"top-20% IoU {means['iou_pred']:.3f}"
    )
    print(
        f"  clean map of y vs adversarial map of y:  correlation {means['corr_y']:.3f}, "
        f"top-20% IoU {means['iou_y']:.3f}"
    )
    os.makedirs("results", exist_ok=True)
    with open(f"results/gradcam_{args.arch}.json", "w") as f:
        json.dump(means, f, indent=2)
    save_gradcam_figure(rows, f"results/gradcam_{args.arch}.png")
