# Self-Study Assignment 2 — Adversarial Examples: PGD Attacks in ℓ∞ and ℓ2

Code of the assignment: untargeted and targeted PGD (Madry et al., ICLR 2018) in the ℓ∞ and
ℓ2 balls against VGG-16 (BN) and ResNet-18 (CIFAR variant) trained on CIFAR-10, then
Grad-CAM, latent features with t-SNE, and transferability from VGG-16 to ResNet-18.

```text
main.py           entry point: reads the arguments and runs the selected --mode
parameters.py     all command-line arguments
utils.py          helpers: data loaders, seeding, checkpoint saving/loading, plotting
models/           build_model(), the Normalize layer, VGG-16 (vgg.py), ResNet-18 (resnet.py)
train.py          training loop of Assignment 1
attacks.py        pgd_linf(), pgd_l2(), choose_target()
robust.py         evaluate_robustness(): clean and adversarial accuracy; evaluate_transfer()
features.py       latent features and joint t-SNE (Exercise 4)
gradcam.py        Grad-CAM maps for VGG-16 (Exercise 3)
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

## Training

Both models are trained on clean images with the pipeline of Assignment 1: SGD with
momentum 0.9, cosine schedule with one epoch of warm-up, 100 epochs, batch 128, weight decay
5e-4, random crop (padding 4) and flip; 5,000 training images are held out for validation.

```bash
python main.py --arch vgg16    --mode train --optimizer momentum --scheduler cosine --lr 0.05 --epochs 100
python main.py --arch resnet18 --mode train --optimizer momentum --scheduler cosine --lr 0.1  --epochs 100
```

A run writes its per-epoch log to `results/<arch>_clean_seed0.csv` and its weights to
`checkpoints/<arch>_clean_seed0.pt`. Every other mode loads these files; `--checkpoint`
loads another one. The reference clean test accuracy is about 93–94% for VGG-16 and 94–95%
for ResNet-18.

## Modes

- `evaluate_robustness` attacks the first `--n-eval` test images and prints the clean
  accuracy, the adversarial accuracy, the attack success rate and, with `--targeted`, the
  targeted success rate. `--norm` selects `pgd_linf` or `pgd_l2`, `--eps` the budget,
  `--steps` the number of steps K, `--alpha` the step size (default 2.5 ε/K);
  `--attack fgsm` takes one step of size ε without random start. With `--restarts R` an
  image counts as robust only if it survives all R random starts. It saves
  `results/<arch>_<attack>.json` (the numbers), `results/<arch>_<attack>.csv` (label, clean
  prediction, adversarial prediction and target of every image) and
  `results/<arch>_<attack>.png` (a clean image, its adversarial version and the
  perturbation), for example `results/vgg16_pgd20_linf.json`.
- `gradcam` (VGG-16 only) saves `results/gradcam_vgg16.png` and `.json`.
- `tsne` saves `results/tsne_<arch>.png` and `.json`.
- `transfer` attacks `--source-arch` and evaluates the adversarial examples on `--arch`; it
  saves `results/transfer_<source>_to_<arch>_<attack>.json`.

## Exercises

**Exercise 1 — robustness against PGD-10 and PGD-20.**

```bash
for arch in vgg16 resnet18; do
  for k in 10 20; do
    python main.py --arch $arch --mode evaluate_robustness --norm linf --eps 8/255 --steps $k --n-eval 10000
    python main.py --arch $arch --mode evaluate_robustness --norm l2   --eps 0.5   --steps $k --n-eval 10000
  done
done
```

The figures `results/<arch>_pgd20_linf.png` and `results/<arch>_pgd20_l2.png` show one
correctly classified test image, its PGD-20 adversarial version and the perturbation.

**Exercise 2 — untargeted versus targeted PGD-20.**

```bash
for arch in vgg16 resnet18; do
  python main.py --arch $arch --mode evaluate_robustness --norm linf --eps 8/255 --steps 20 --n-eval 10000
  python main.py --arch $arch --mode evaluate_robustness --norm linf --eps 8/255 --steps 20 --n-eval 10000 --targeted --target-mode next
done
```

The targeted success rate is measured on the correctly classified images. The run prints
the most frequent (true class → adversarial class) pairs; the `.csv` file has the
prediction of every image.

**Exercise 3 — Grad-CAM (VGG-16).**

```bash
python main.py --arch vgg16 --mode gradcam
```

The figure has one row per image (6 correctly classified test images): the clean image, the
map of the true class on it, the untargeted PGD-20 ℓ∞ adversarial image, the map of its
prediction, and the map of the true class on it. The correlation and the top-20% IoU of the
clean and the adversarial maps are averaged over 100 images. The maps come from a forward
and a backward hook on the last convolutional layer, which works on 2x2 maps.

**Exercise 4 — latent distribution of clean and adversarial samples.**

```bash
python main.py --arch vgg16    --mode tsne
python main.py --arch resnet18 --mode tsne
```

The correctly classified cats are attacked with targeted PGD-20 ℓ∞ towards dog; clean cats,
adversarial cats and clean dogs are embedded in one t-SNE run (PCA to 50 dimensions,
perplexity 30, seed 0). The run prints the mean distance of the adversarial cats to the
centroids of the clean cats and dogs and the fraction whose 10 nearest clean neighbours are
mostly dogs, computed on the 512-dimensional features, for all adversarial cats and
separately by attack success.

**Exercise 5 — transferability from VGG-16 to ResNet-18.**

```bash
python main.py --arch resnet18 --mode transfer --source-arch vgg16 --norm linf --eps 8/255 --steps 20 --n-eval 10000
python main.py --arch resnet18 --mode transfer --source-arch vgg16 --norm l2   --eps 0.5   --steps 20 --n-eval 10000
```

It prints the clean accuracy of ResNet-18, its accuracy on the examples transferred from
VGG-16, its accuracy under its own white-box PGD-20 attack, the transfer rate (of the
examples that fool VGG-16, the fraction that also fool ResNet-18, on the images both
classify correctly) and how often both are fooled into the same class.
