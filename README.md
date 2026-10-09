# IAV — MS COCO multi-label classification

Repository for the MS COCO challenge in IAV, by Tayeb and Paul.

**Multi-label** classification over 80 MS COCO classes: one image can contain several object categories.

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

## Execution instructions

Python 3.10 or newer. A CUDA GPU is recommended; a short run also works on CPU.

1. Unzip this folder and open a terminal in it (the directory that contains `src/` and `requirements.txt`).
2. Create an environment and install PyTorch, then the rest of the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# torch and torchvision must come from the same CUDA 12 build.
# cu128 needs an NVIDIA driver from the 570 series or newer.
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Older driver: same command with
# --index-url https://download.pytorch.org/whl/cu124

# CPU-only machine:
# pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements.txt
```

3. Check that PyTorch loads (and that the GPU is visible, if you have one):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

4. Point the code to the course dataset. It is **not** in this archive. Either place it at `../ms-coco` relative to this folder, or set `MSCOCO_ROOT`:

```bash
# Linux/macOS
export MSCOCO_ROOT=/path/to/ms-coco

# Windows PowerShell
$env:MSCOCO_ROOT="C:\path\to\ms-coco"
```

Expected layout: `images/train/`, `images/test/`, `labels/train/`.

5. Open Jupyter from this folder and run a notebook **from the project root** (or with that folder as the working directory), so `src/coco_mlc` is found:

```bash
pip install notebook
jupyter notebook
```

- **Quick check (grading / smoke test):** open `notebooks/experiments/2/exp_maxvit_t.ipynb`, set `FULL_TRAIN = False` in the configuration cell (512 images, 1 epoch per stage), then **Run All**. The notebook writes a JSON file under `submissions/`.
- **Full training:** leave `FULL_TRAIN = True` (10 + 10 epochs on the 65,000 labeled images). This takes several hours on an RTX 4060 Ti.

Experiment 1 notebooks are in `notebooks/experiments/1/`. Experiment 2 notebooks are in `notebooks/experiments/2/`. ImageNet weights (`Weights.DEFAULT`) are downloaded on the first run.

This archive contains **no checkpoints or dataset images**. Training writes weights to `outputs/notebooks/…/*.pth`. The report notebook is `final_notebook.ipynb`; the ranking tables and validation curves it embeds are in `results/`.

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
notebooks/experiments/2/  experiment 2, MaxViT-T and ConvNeXt-Tiny, 10 + 10 epochs
outputs/                  results, checkpoints, figures
submissions/              JSON files for the leaderboard
```

---

## Challenge rules

- A new submission with the same group name **overwrites** the previous one. Check the leaderboard before sending, and do not create a second entry.
- Using another MS COCO distribution is **not allowed**, in particular the torchvision dataset or the `COCO_V1` detection weights. Only **ImageNet** weights are used.
- The test set is never used to pick a hyperparameter or a threshold. Those choices are made on the validation set.
