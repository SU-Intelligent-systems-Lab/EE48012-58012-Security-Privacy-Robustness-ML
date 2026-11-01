<div align="center">

# Security, Privacy & Robustness in Machine Learning

**EE 48012 / 58012 — Special Topics on EE · Sabancı University**

Hands-on labs

</div>

---

## The labs

| Lab | The question | What you will do |
| :--- | :--- | :--- |
| **[Lab 1: Training deep networks](lab-1/README.md)** | What do the optimiser and the learning-rate schedule change? | Train an MLP on MNIST, a CNN on CIFAR-10 and ResNet-18 on CIFAR-100. Look inside them with t-SNE and Grad-CAM. |
| **[Lab 2: Adversarial attacks](lab-2/README.md)** | How easily can a good network be fooled? | Attack VGG-16 and ResNet-18 with PGD, targeted and untargeted. See what changes in Grad-CAM and t-SNE, and test whether an attack transfers to another model. |
| **[Lab 3: Adversarial training](lab-3/README.md)** | Does a network trained against attacks also resist natural corruptions? | Train VGG-16 with PGD-10 adversarial training and watch robust overfitting. Compare it with the clean model under PGD and on the 15 corruptions of CIFAR-10-C, and weigh robustness against accuracy. |

Each lab starts from the code of the previous one, so it pays to follow them in order. The
README of each lab explains what to run and what every experiment produces.

## Getting started

All three labs use the same package versions, so one environment is enough.

```bash
git clone https://github.com/SU-Intelligent-systems-Lab/EE48012-58012-Security-Privacy-Robustness-ML.git
cd EE48012-58012-Security-Privacy-Robustness-ML

python -m venv .venv
source .venv/bin/activate
pip install -r lab-1/requirements.txt
```

Then train your first network:

```bash
cd lab-1
python main.py --task mnist --optimizer adam --scheduler step --lr 0.001
```

## Good to know

- **Run commands from inside the lab folder.** Results and checkpoints are written there.
- **Datasets download by themselves** on the first run, except CIFAR-10-C in Lab 3 (see its
  README). To reuse them in a later lab, run `ln -s ../lab-1/data data` inside its folder.
- **A GPU is strongly recommended.** Adversarial training with a K-step attack costs about
  K + 1 times as much as standard training.
- **Start reading at `main.py`.** `parameters.py` lists every command-line argument, and the
  docstrings explain the code.

---

<div align="center">
<sub>Course material of the Applied Intelligence Systems Lab (APIS), Sabancı University.</sub>
</div>
