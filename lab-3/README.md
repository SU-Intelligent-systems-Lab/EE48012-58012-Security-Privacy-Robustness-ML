# Self-Study Assignment 3 — Adversarial Training: PGD-10 and Common Corruptions

Code of the assignment: PGD-10 adversarial training (Madry et al., ICLR 2018) of VGG-16 (BN)
on CIFAR-10, with model selection on a held-out validation set, then the clean and the
adversarially trained model compared under PGD and on the 15 corruptions of CIFAR-10-C
(Hendrycks and Dietterich, ICLR 2019).

```text
main.py           entry point: reads the arguments and runs the selected --mode
parameters.py     all command-line arguments
utils.py          helpers: data loaders and validation split, seeding, checkpoints, plotting
models/           build_model(), the Normalize layer, VGG-16 (vgg.py), ResNet-18 (resnet.py)
train.py          train_one_epoch(), train_epoch_adv() and fit() with model selection
attacks.py        pgd_linf(), pgd_l2(), choose_target(), unchanged from Assignment 2
robust.py         evaluate_robustness(): clean and adversarial accuracy; clean_and_pgd_accuracy()
corruptions.py    the CIFAR10C dataset and evaluate_corruptions()
checkpoints/      saved weights
results/          logs, results and figures
requirements.txt  package versions
```

## Setup

```bash
pip install -r requirements.txt
```

Run every command from this folder. CIFAR-10 is downloaded to `data/` (from a mirror, see
`utils.py`); after Assignment 1 it can be reused with `ln -s ../lab-1/data data`. A GPU is
used when available. The data loaders apply `ToTensor()` only (and the augmentation for
training), so images are in [0, 1]; the standardisation is the `Normalize` layer in front of
every model, and the attacks never see standardised values.

CIFAR-10-C (about 3 GB) is not downloaded by the code. Put its 15 corruption files and
`labels.npy` in `data/CIFAR-10-C`:

```bash
mkdir -p data
wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
tar -xf CIFAR-10-C.tar -C data
```

The baseline is the clean VGG-16 of Assignment 2:

```bash
cp ../lab-2/checkpoints/vgg16_clean_seed0.pt checkpoints/
```

## Training

`--train-mode pgd_at` replaces every training batch by its PGD-10 adversarial examples (ℓ∞,
ε = 8/255, α = 2/255, random start), computed with the model in eval mode; the weights are
then updated in train mode. The recipe is SGD with momentum 0.9, learning rate 0.05 divided
by 10 at epochs 50 and 75, 100 epochs, batch 128, weight decay 5e-4, random crop (padding 4)
and flip. 1,000 training images are held out for validation (`--val-size`).

```bash
python main.py --arch vgg16 --mode train --train-mode pgd_at
```

After every epoch the run measures the clean and the PGD-10 accuracy on the validation set.
It writes its per-epoch log to `results/vgg16_pgdat_seed0.csv` (adversarial training loss
and accuracy, clean and PGD validation accuracy, learning rate, time),
`checkpoints/vgg16_pgdat_seed0_best.pt` whenever the PGD validation accuracy is the best so
far, and `checkpoints/vgg16_pgdat_seed0_last.pt` at the end. The reference for the best
checkpoint is about 78–81% clean test accuracy and 45–48% under PGD-20.

`--train-mode standard` (the default) trains on clean images with the same loop and saves
`vgg16_clean_seed0_best.pt` and `vgg16_clean_seed0_last.pt`.

## Modes

Both evaluation modes load `--checkpoint`, or the clean model
`checkpoints/<arch>_clean_seed0.pt` when it is not given.

- `evaluate_robustness` is the mode of Assignment 2: it attacks the first `--n-eval` test
  images and prints the clean accuracy, the adversarial accuracy and the attack success
  rate. `--norm`, `--eps`, `--steps` and `--alpha` (default 2.5 ε/K) choose the attack. It
  saves `results/<model>_<attack>.json`, `.csv` and `.png`, where `<model>` is the name of
  the checkpoint, for example `results/vgg16_pgdat_seed0_best_pgd20_linf.json`.
- `evaluate_corruptions` classifies the clean test set and CIFAR-10-C, with no attack, and
  prints the accuracy of every set. `--corruptions` chooses the corruptions (all 15 by
  default) and `--severities` the severities (3 and 5 by default). The accuracies go to
  `results/corruptions.csv` with the columns `arch, training, corruption, category,
  severity, accuracy`; the clean test set is the row with `corruption = clean` and
  `severity = 0`. `training` is the name of the checkpoint without the architecture, for
  example `clean_seed0` or `pgdat_seed0_best`. The file holds every evaluated model, and a
  new run of a model replaces its rows.

## Exercises

**Exercise 1 — PGD-10 adversarial training.**

```bash
python main.py --arch vgg16 --mode train --train-mode pgd_at
for ckpt in best last; do
  python main.py --arch vgg16 --mode evaluate_robustness --norm linf --eps 8/255 --steps 20 --n-eval 10000 --checkpoint checkpoints/vgg16_pgdat_seed0_$ckpt.pt
done
```

The curves are the columns of `results/vgg16_pgdat_seed0.csv`; the learning rate drops after
epochs 50 and 75. The log of the clean model, `../lab-2/results/vgg16_clean_seed0.csv`, has
the clean accuracy and the time of clean training.

**Exercise 2 — adversarial robustness in ℓ∞ and ℓ2.**

```bash
for model in vgg16_clean_seed0 vgg16_pgdat_seed0_best; do
  python main.py --arch vgg16 --mode evaluate_robustness --norm linf --eps 8/255 --steps 20 --n-eval 10000 --checkpoint checkpoints/$model.pt
  python main.py --arch vgg16 --mode evaluate_robustness --norm l2   --eps 0.5   --steps 20 --n-eval 10000 --checkpoint checkpoints/$model.pt
done
```

**Exercises 3 and 4 — CIFAR-10-C.**

```bash
python main.py --arch vgg16 --mode evaluate_corruptions
python main.py --arch vgg16 --mode evaluate_corruptions --checkpoint checkpoints/vgg16_pgdat_seed0_best.pt
```

`results/corruptions.csv` then holds, for both models, the clean test accuracy and the
accuracy on the 15 corruptions at severities 3 and 5, with the category of every
corruption.
