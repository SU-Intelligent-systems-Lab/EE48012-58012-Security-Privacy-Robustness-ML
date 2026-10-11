# Self-Study Assignment 1 — Introduction to Training Deep Neural Networks

Code of the assignment: a two-layer MLP on MNIST (Task A), a simple CNN on CIFAR-10
(Task B) and ResNet-18 on CIFAR-100 (Task C), each trained with SGD, SGD with momentum
and Adam(W) under a constant, a step and a poly learning-rate schedule.

```text
main.py           entry point: reads the parameters, then calls training or testing
parameters.py     all command-line arguments and their defaults (argparse)
train.py          train_one_epoch() and fit(), the loop over epochs
test.py           evaluate(): loss and accuracy on a data split
models/           one file per architecture, and build_model() to select one
data.py           datasets, transforms and build_loaders()
utils.py          logging, saving and loading checkpoints
checkpoints/      saved weights
results/          logs and figures
analysis/         plots.py (curves and tables), tsne.py, gradcam.py
requirements.txt  package versions
```

## Setup

```bash
pip install -r requirements.txt
```

Run every command from this folder. Datasets are downloaded to `data/` (CIFAR from a
mirror, see `data.py`); a GPU is used when available. All arguments are defined in `parameters.py`.

## Training

A run is named `<task>_<optimizer>_<scheduler>_seed<seed>`, for example
`cifar10_adam_step_seed0`. It writes `results/<run>.csv` (per epoch: training loss,
validation and test loss and accuracy, learning rate at the start of the epoch, mean
gradient norm, wall-clock time) and, after the last epoch, `checkpoints/<run>.pt`.

The weight decay and the warm-up are fixed within a task and set by `--task`: no weight
decay for Task A and 5e-4 for Tasks B and C; a warm-up of one epoch for Task C only.
ResNet-18 is written from scratch in `models/resnet_cifar.py` (3x3 stride-1 stem, no
max-pool); `torchvision.models.resnet18` is not used.

**Task A (MLP / MNIST), 20 epochs.**

```bash
for s in constant step poly; do
  python main.py --task mnist --optimizer sgd      --scheduler $s --lr 0.1
  python main.py --task mnist --optimizer momentum --scheduler $s --lr 0.01
  python main.py --task mnist --optimizer adam     --scheduler $s --lr 0.001
done
```

**Task B (CNN / CIFAR-10), 50 epochs.**

```bash
for s in constant step poly; do
  python main.py --task cifar10 --optimizer sgd      --scheduler $s --lr 0.05  --epochs 50
  python main.py --task cifar10 --optimizer momentum --scheduler $s --lr 0.05  --epochs 50
  python main.py --task cifar10 --optimizer adam     --scheduler $s --lr 0.001 --epochs 50
done
```

**Task C (ResNet-18 / CIFAR-100), 100 epochs.**

```bash
for s in constant step poly; do
  python main.py --task cifar100 --optimizer sgd      --scheduler $s --lr 0.1   --epochs 100
  python main.py --task cifar100 --optimizer momentum --scheduler $s --lr 0.1   --epochs 100
  python main.py --task cifar100 --optimizer adam     --scheduler $s --lr 0.001 --epochs 100
done
```

**Testing saved weights.**

```bash
python main.py --task cifar10 --mode test --checkpoint checkpoints/cifar10_adam_step_seed0.pt
```

## Exercises

The analysis scripts read the same arguments as `main.py`.

**Exercise 1 — training and convergence.** For each task, after its nine runs:

```bash
python -m analysis.plots --task mnist      # also cifar10, cifar100
```

| Output | File |
| --- | --- |
| Test loss and test accuracy against the epoch, one figure per optimiser, one line per schedule | `results/curves_<task>_<optimizer>.pdf` |
| Training loss against the epoch, one panel per optimiser | `results/train_loss_<task>.pdf` |
| Logged learning rate against the epoch | `results/lr_<task>.pdf` |
| 3x3 table of the final test accuracy, and the best run (validation accuracy) | `results/table_<task>.md` |

**Exercise 2 — MNIST: t-SNE.** Pass the checkpoint of the best MLP:

```bash
python -m analysis.tsne --task mnist --checkpoint checkpoints/mnist_<optimizer>_<scheduler>_seed0.pt
```

It embeds the hidden features of 2,000 test images (200 per digit) and saves
`results/tsne_<run>.png`: the embedding coloured by the true and by the predicted digit,
with the misclassified points marked.

**Exercise 3 — CIFAR-10: Grad-CAM.** Pass the checkpoint of the best CNN:

```bash
python -m analysis.gradcam --task cifar10 --checkpoint checkpoints/cifar10_<optimizer>_<scheduler>_seed0.pt
```

It saves `results/gradcam_correct_<run>.png` (8 correctly classified test images) and
`results/gradcam_misclassified_<run>.png` (8 misclassified ones, with the map of the
true class next to the map of the predicted class).

**Exercise 4 — CIFAR-100: top-1 and top-5 accuracy.**

```bash
python -m analysis.plots --task cifar100
```

Besides the outputs of Exercise 1 it writes the top-1 and top-5 tables
(`results/table_top5_cifar100.md`), prints how many test images of the best model are
wrong at top-1 but correct at top-5 and how often the confusion stays within the
superclass, and saves 16 of these images (`results/top5_cifar100.png`).
