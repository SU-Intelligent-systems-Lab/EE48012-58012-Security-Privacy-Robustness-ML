"""PGD attacks in the l_inf and l_2 balls, and choose_target() for the targeted mode.

pgd_linf() and pgd_l2() are Listings 2 and 3 of the handout.
"""

import torch
import torch.nn.functional as F


def pgd_linf(
    model, x, y, eps, alpha, steps, random_start=True, targeted=False, target=None
):
    """Untargeted or targeted PGD in the l_inf ball.

    Lines of the algorithm: 1-2 random start; 4 gradient of J = loss(f(x'), y), or
    -loss(f(x'), t) if targeted; 5 sign step; 6 limit perturbation; 7 clamp to [0, 1].

    Args:
        model (nn.Module): Classifier in eval mode, taking images in [0, 1].
        x (Tensor): Clean images, shape (B, C, H, W), values in [0, 1].
        y (Tensor): True labels, shape (B,).
        eps (float): Radius of the l_inf ball, e.g. 8/255.
        alpha (float): Step size.
        steps (int): Number of iterations K.
        random_start (bool): Start from a uniform point in the ball.
        targeted (bool): If True, move towards `target` instead of away from y.
        target (Tensor, optional): Target labels, shape (B,).

    Returns:
        Tensor: Adversarial images, shape (B, C, H, W), in [0, 1].
    """
    x = x.clone().detach()
    x_adv = x.clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        if targeted:
            cost = -F.cross_entropy(logits, target, reduction="sum")
        else:
            cost = F.cross_entropy(logits, y, reduction="sum")
        (grad,) = torch.autograd.grad(cost, x_adv)
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = (x_adv - x).clamp(-eps, eps)
            x_adv = (x + delta).clamp(0, 1)
    return x_adv.detach()


def pgd_l2(
    model, x, y, eps, alpha, steps, random_start=True, targeted=False, target=None
):
    """Untargeted or targeted PGD in the l_2 ball (arguments as pgd_linf).

    Eq. (12) is the random start. Lines of the algorithm: 5 normalised step;
    6 projection onto the ball; 7 clamp to [0, 1].
    """
    x = x.clone().detach()
    x_adv = x.clone()
    B = x.size(0)
    if random_start:
        z = torch.randn_like(x)
        z = z / z.flatten(1).norm(dim=1).view(B, 1, 1, 1)
        u = torch.rand(B, 1, 1, 1, device=x.device)
        x_adv = (x + eps * u ** (1 / x[0].numel()) * z).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        if targeted:
            cost = -F.cross_entropy(logits, target, reduction="sum")
        else:
            cost = F.cross_entropy(logits, y, reduction="sum")
        (grad,) = torch.autograd.grad(cost, x_adv)
        with torch.no_grad():
            g_norm = grad.flatten(1).norm(dim=1).view(B, 1, 1, 1)
            x_adv = x_adv + alpha * grad / (g_norm + 1e-12)
            delta = x_adv - x
            d_norm = delta.flatten(1).norm(dim=1).view(B, 1, 1, 1)
            delta = delta * torch.clamp(eps / (d_norm + 1e-12), max=1.0)
            x_adv = (x + delta).clamp(0, 1)
    return x_adv.detach()


@torch.no_grad()
def choose_target(model, x, y, mode):
    """Choose the target class of every image for the targeted mode.

    Args:
        model (nn.Module): Classifier in eval mode, taking images in [0, 1].
        x (Tensor): Clean images, shape (B, C, H, W), values in [0, 1].
        y (Tensor): True labels, shape (B,).
        mode (str): "next" for (y + 1) mod K, "random" for a uniformly drawn
            class other than y, "least" for the least likely class of the
            clean image.

    Returns:
        Tensor: Target labels, shape (B,); different from y for every
            correctly classified image.
    """
    logits = model(x)
    K = logits.size(1)
    if mode == "next":
        return (y + 1) % K
    if mode == "random":
        shift = torch.randint(1, K, y.shape, device=y.device)
        return (y + shift) % K
    if mode == "least":
        return logits.argmin(dim=1)
    raise ValueError(mode)
