"""Clean and adversarial accuracy: evaluate_robustness() and clean_and_pgd_accuracy()."""

import csv
import json
import os
from collections import Counter

import torch

from attacks import choose_target, pgd_l2, pgd_linf
from utils import CLASSES, model_name, save_adversarial_example


def step_size(args):
    """Return the PGD step size: --alpha, or 2.5 * eps / steps when it is not given.

    Args:
        args (argparse.Namespace): Uses args.alpha, args.eps and args.steps.

    Returns:
        float: The step size alpha.
    """
    return args.alpha if args.alpha is not None else 2.5 * args.eps / args.steps


def attack(model, x, y, args, target=None):
    """Return adversarial images of a batch, with the attack chosen by the arguments.

    --attack fgsm is one step of size eps without random start; otherwise PGD from a
    random start.

    Args:
        model (nn.Module): Classifier in eval mode, taking images in [0, 1].
        x (Tensor): Clean images, shape (B, C, H, W), values in [0, 1].
        y (Tensor): True labels, shape (B,).
        args (argparse.Namespace): Uses args.norm, args.attack, args.eps,
            args.alpha and args.steps.
        target (Tensor, optional): Target labels, shape (B,); given, the
            attack is targeted.

    Returns:
        Tensor: Adversarial images, shape (B, C, H, W), in [0, 1].
    """
    pgd = pgd_linf if args.norm == "linf" else pgd_l2
    targeted = target is not None
    if args.attack == "fgsm":
        return pgd(
            model,
            x,
            y,
            args.eps,
            alpha=args.eps,
            steps=1,
            random_start=False,
            targeted=targeted,
            target=target,
        )
    return pgd(
        model,
        x,
        y,
        args.eps,
        step_size(args),
        args.steps,
        targeted=targeted,
        target=target,
    )


def attack_name(args):
    """Return a short name of the attack, used in the result files.

    Args:
        args (argparse.Namespace): The arguments of parameters.py.

    Returns:
        str: For example "pgd20_linf" or "pgd20_linf_targeted_next".
    """
    name = (
        f"{args.attack}_{args.norm}"
        if args.attack == "fgsm"
        else f"pgd{args.steps}_{args.norm}"
    )
    if args.targeted:
        name += f"_targeted_{args.target_mode}"
    if args.restarts > 1:
        name += f"_restarts{args.restarts}"
    return name


def evaluate_robustness(model, loader, args, device):
    """Measure the clean and the adversarial accuracy on the first --n-eval test images.

    With --restarts R > 1 an image is robust only if it survives every restart, and a
    targeted attack succeeds if one restart reaches the target. Prints the results and
    saves results/<model>_<attack>.json, .csv (every prediction) and .png (one example),
    with <model> the name of the checkpoint.

    Args:
        model (nn.Module): Classifier in eval mode, taking images in [0, 1].
        loader (DataLoader): Test loader without augmentation or shuffling.
        args (argparse.Namespace): The arguments of parameters.py.
        device (str): "cpu" or "cuda".

    Returns:
        dict: Clean accuracy, adversarial accuracy, attack success rate and,
            for a targeted attack, targeted success rate.
    """
    name = f"{model_name(args)}_{attack_name(args)}"
    n = clean_correct = robust = target_hits = 0
    records, example = [], None
    for x, y in loader:
        x, y = x[: args.n_eval - n].to(device), y[: args.n_eval - n].to(device)
        if len(y) == 0:
            break
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        target = choose_target(model, x, y, args.target_mode) if args.targeted else None
        robust_mask = torch.ones_like(y, dtype=torch.bool)
        hit_mask = torch.zeros_like(y, dtype=torch.bool)
        for r in range(args.restarts):
            x_r = attack(model, x, y, args, target)
            with torch.no_grad():
                pred_r = model(x_r).argmax(dim=1)
            robust_mask &= pred_r == y
            if args.targeted:
                hit_mask |= pred_r == target
            ok_r = (pred_r == target) if args.targeted else (pred_r != y)
            if r == 0:
                x_adv, adv_pred, ok = x_r, pred_r, ok_r
            else:
                take = ok_r & ~ok
                x_adv[take], adv_pred[take], ok = x_r[take], pred_r[take], ok | ok_r
        correct = pred == y
        clean_correct += correct.sum().item()
        robust += robust_mask.sum().item()
        target_hits += (hit_mask & correct).sum().item()
        for i in range(len(y)):
            records.append(
                {
                    "index": n + i,
                    "label": y[i].item(),
                    "clean_pred": pred[i].item(),
                    "adv_pred": adv_pred[i].item(),
                    "target": target[i].item() if args.targeted else "",
                }
            )
        if example is None and correct.any():
            i = correct.nonzero()[0, 0]
            example = (x[i], x_adv[i], pred[i].item(), adv_pred[i].item())
        n += len(y)

    acc_clean, acc_rob = clean_correct / n, robust / n
    results = {
        "arch": args.arch,
        "attack": args.attack,
        "norm": args.norm,
        "eps": args.eps,
        "alpha": args.eps if args.attack == "fgsm" else step_size(args),
        "steps": 1 if args.attack == "fgsm" else args.steps,
        "restarts": args.restarts,
        "targeted": args.targeted,
        "target_mode": args.target_mode if args.targeted else None,
        "n_eval": n,
        "clean_accuracy": acc_clean,
        "adversarial_accuracy": acc_rob,
        "attack_success_rate": 1 - acc_rob / acc_clean,
    }
    if args.targeted:
        results["targeted_success_rate"] = target_hits / clean_correct

    print(
        f"{name} on {n} test images: clean accuracy {acc_clean:.2%}, "
        f"adversarial accuracy {acc_rob:.2%}, "
        f"attack success rate {results['attack_success_rate']:.2%}"
    )
    if args.targeted:
        print(f"targeted success rate {results['targeted_success_rate']:.2%}")
    moves = Counter(
        (r["label"], r["adv_pred"])
        for r in records
        if r["clean_pred"] == r["label"] != r["adv_pred"]
    )
    print(
        "most frequent adversarial classes:",
        ", ".join(
            f"{CLASSES[a]} -> {CLASSES[b]} {c}" for (a, b), c in moves.most_common(5)
        ),
    )

    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"results/{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, records[0])
        writer.writeheader()
        writer.writerows(records)
    if example is not None:
        save_adversarial_example(*example, f"results/{name}.png")
    return results


def clean_and_pgd_accuracy(model, loader, eps, alpha, steps, device):
    """Return the clean accuracy and the accuracy under l_inf PGD from a random start.

    fit() calls it on the validation set after every epoch.

    Args:
        model (nn.Module): Classifier taking images in [0, 1]. It is switched to
            eval mode.
        loader (DataLoader): Yields mini-batches (x, y).
        eps (float): Radius of the l_inf ball.
        alpha (float): Step size.
        steps (int): Number of iterations K.
        device (str): "cpu" or "cuda".

    Returns:
        tuple[float, float]: Clean and PGD accuracy, between 0 and 1.
    """
    model.eval()
    clean = robust = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model, x, y, eps, alpha, steps)
        with torch.no_grad():
            clean += (model(x).argmax(dim=1) == y).sum().item()
            robust += (model(x_adv).argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return clean / n, robust / n
