# IAV — MS COCO multi-label classification

Repository for the MS COCO challenge in IAV, by Tayeb and Paul.

**Multi-label** classification over 80 MS COCO classes: one image can contain several object categories. The full statement is in [`sujet.ipynb`](sujet.ipynb).

| | |
| --- | --- |
| Data | 65,000 labeled training images, 4,952 test images |
| Outputs | 80 classes, multi-hot vector |
| Metric | F1 with precision and recall **weighted by the inverse class frequency** |
| Submission | [leaderboard](https://www.creatis.insa-lyon.fr/kechichian/ms-coco-classif-leaderboard.html), JSON format |

---

## What drives the score

The server metric weights each class by `1/frequency`. In practice:

- `hair drier` (102 images) is **13.6%** of the score, `toaster` (117) is **11.9%**;
- the **10 rarest classes are 43.8%** of the score;
- `person` (35,494 images) is **0.039%**, about 350 times less than `hair drier`.

Optimizing accuracy or micro-F1 therefore gives a poor server score. The main lever is **recall on rare classes**, through a loss that handles imbalance and **per-class threshold calibration**.

---

## Setup

```bash
# torch and torchvision must come from the same CUDA 12 build.
# cu128 needs an NVIDIA driver from the 570 series or newer.
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Older driver: same command with
# --index-url https://download.pytorch.org/whl/cu124
```

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: a `+cu128` (or `+cu124`) build, `True`, and `NVIDIA GeForce RTX 4060 Ti`.

The dataset is not in the repository. It should sit in `../ms-coco` relative to the repo root, or at the path given by `MSCOCO_ROOT`:

```bash
export MSCOCO_ROOT=/path/to/ms-coco
```

Expected layout: `images/train/`, `images/test/`, `labels/train/`.

---

## How to run an experiment

Experiment 1 notebooks are in [`notebooks/experiments/1`](notebooks/experiments/1): one architecture each, stratified 70/15/15 split, a frozen-backbone baseline, then fine-tuning. Experiment 2 notebooks are in [`notebooks/experiments/2`](notebooks/experiments/2): MaxViT-T, ConvNeXt-Tiny and EfficientNetV2-S, for 10 epochs then 10 epochs, with `num_workers = 0` and mixed precision on CUDA. The server F1 is the metric. See [`notebooks/experiments/README.md`](notebooks/experiments/README.md).

`FULL_TRAIN = False` limits a run to 512 images and one epoch, to check the pipeline. The finished experiment 1 notebooks, except VGG16, and the three experiment 2 notebooks are set to `True`. ImageNet weights (`Weights.DEFAULT`) are downloaded on the first run.

Submissions are written as JSON under `submissions/`.

---

## Repository layout

```
src/coco_mlc/          shared library used by the notebooks
├── config.py          paths, 80 classes, default hyperparameters
├── data.py            datasets, transforms, stratified split
├── models.py          torchvision factory, classification heads
├── losses.py          weighted BCE, focal loss, asymmetric loss
├── metrics.py         server metric, macro/micro F1, mAP
├── engine.py          train/eval loops, checkpoints
├── thresholds.py      decision-threshold calibration
├── diagnostics.py     error curves, model diagram, bias/variance report
└── utils.py           seeds, experiment log

notebooks/experiments/1/  experiment 1, one notebook per architecture
notebooks/experiments/2/  experiment 2, three architectures, 10 + 10 epochs
outputs/                  results, checkpoints, figures
submissions/              JSON files for the leaderboard
```

---

## Challenge rules

- A new submission with the same group name **overwrites** the previous one. Check the leaderboard before sending, and do not create a second entry.
- Using another MS COCO distribution is **not allowed**, in particular the torchvision dataset or the `COCO_V1` detection weights. Only **ImageNet** weights are used.
- The test set is never used to pick a hyperparameter or a threshold. Those choices are made on the validation set.
