"""Datasets, transforms, and the train/validation/test split.

The real layout of the provided dataset is:

    ms-coco/images/train/<id>.jpg   (65,000)
    ms-coco/images/test/<id>.jpg    (4,952)
    ms-coco/labels/train/<id>.cls   (65,000)

Images were resized so the long side is 224, keeping the aspect ratio, so they
do not all have the same size (for example 224x149, 168x224).
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .config import NUM_CLASSES, PATHS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class PadToSquare:
    """Pad the image to a square with borders, without cropping anything.

    Dataset images have varied aspect ratios. A direct ``Resize((S, S))``
    stretches them, and a ``CenterCrop`` drops the borders where objects to
    predict may sit. Padding keeps the full content and the original ratio.
    """

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new(img.mode, (side, side), self.fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fill={self.fill})"


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    resize_mode: str = "pad",
    augment: str = "flip",
):
    """Build the transform pipeline.

    ``resize_mode`` controls the geometric reshape:
      - ``"pad"``    : pad to a square, then resize (nothing is dropped);
      - ``"squash"`` : resize straight to (S, S), which changes the ratio;
      - ``"crop"``   : resize the short side, then center-crop.

    ``augment`` (only if ``train=True``):
      - ``"none"``       : geometry only;
      - ``"flip"``       : horizontal flip (baseline);
      - ``"strong"``     : flip, light jitter, light affine;
      - ``"experiment"`` : flip, rotation / translation / scale, photometric jitter.
    """
    if resize_mode == "pad":
        geometry = [PadToSquare(), T.Resize((image_size, image_size))]
    elif resize_mode == "squash":
        geometry = [T.Resize((image_size, image_size))]
    elif resize_mode == "crop":
        geometry = [T.Resize(image_size), T.CenterCrop(image_size)]
    else:
        raise ValueError(f"unknown resize_mode: {resize_mode!r}")

    steps = list(geometry)
    if train:
        if augment == "flip":
            steps.append(T.RandomHorizontalFlip())
        elif augment == "strong":
            steps += [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            ]
        elif augment == "experiment":
            steps += [
                T.RandomHorizontalFlip(),
                T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.15)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            ]
        elif augment != "none":
            raise ValueError(f"unknown augment: {augment!r}")

    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class COCOTrainImageDataset(Dataset):
    """Labeled images from the train subset, with multi-hot targets (80).

    Same idea as the dataset in the assignment: the example list comes from
    the sorted ``.cls`` files, and the matching image is inferred from the
    file name. ``return_id`` adds the image id to the returned items, which
    is useful to trace predictions.
    """

    def __init__(
        self,
        img_dir: str | Path | None = None,
        annotations_dir: str | Path | None = None,
        max_images: int | None = None,
        transform=None,
        return_id: bool = False,
    ):
        self.img_dir = Path(img_dir) if img_dir is not None else PATHS.train_images
        self.annotations_dir = (
            Path(annotations_dir) if annotations_dir is not None else PATHS.train_labels
        )
        self.img_labels = sorted(glob("*.cls", root_dir=str(self.annotations_dir)))
        if not self.img_labels:
            raise RuntimeError(f"No .cls file in {self.annotations_dir}")
        if max_images:
            self.img_labels = self.img_labels[:max_images]
        self.transform = transform
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.img_labels)

    @property
    def ids(self) -> list[str]:
        return [Path(name).stem for name in self.img_labels]

    def __getitem__(self, idx: int):
        stem = Path(self.img_labels[idx]).stem
        image = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        with open(self.annotations_dir / self.img_labels[idx]) as f:
            class_ids = [int(line) for line in f.read().split()]
        if self.transform:
            image = self.transform(image)
        labels = torch.zeros(NUM_CLASSES)
        labels[class_ids] = 1.0
        if self.return_id:
            return image, labels, stem
        return image, labels


class COCOTestImageDataset(Dataset):
    """Images from the test subset, returned with their id."""

    def __init__(self, img_dir: str | Path | None = None, transform=None):
        self.img_dir = Path(img_dir) if img_dir is not None else PATHS.test_images
        self.img_list = sorted(glob("*.jpg", root_dir=str(self.img_dir)))
        if not self.img_list:
            raise RuntimeError(f"No .jpg image in {self.img_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_list)

    @property
    def ids(self) -> list[str]:
        return [Path(name).stem for name in self.img_list]

    def __getitem__(self, idx: int):
        path = self.img_dir / self.img_list[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path.stem


class TransformSubset(Dataset):
    """View of a dataset restricted to some indices, with its own transform.

    This is needed here: random augmentation must apply to train but not to
    validation, while both subsets come from the same base dataset.
    """

    def __init__(self, base: COCOTrainImageDataset, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        prev, self.base.transform = self.base.transform, self.transform
        try:
            return self.base[self.indices[idx]]
        finally:
            self.base.transform = prev


class TensorBatchLoader:
    """Iterate by slices over tensors that are already in memory.

    A classic ``DataLoader`` fetches examples one by one and then stacks them,
    which costs more than the computation itself when the "data" is a feature
    vector of 512 values and the model is a single linear layer. Slicing
    directly removes that overhead.

    The interface (``__iter__``, ``__len__``, ``dataset`` attribute) is the one
    expected by the functions in ``engine.py``, so they work the same way on
    features and on images.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1024,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ):
        self.features = torch.as_tensor(features).float()
        self.labels = torch.as_tensor(labels).float()
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels have different lengths")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self) -> int:
        return (len(self.features) + self.batch_size - 1) // self.batch_size

    @property
    def dataset(self):
        return self.features

    def __iter__(self):
        n = len(self.features)
        order = (
            torch.randperm(n, generator=self.generator)
            if self.shuffle
            else torch.arange(n)
        )
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            yield self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Fast annotation loading (without decoding images)
# ---------------------------------------------------------------------------
def load_label_matrix(annotations_dir: str | Path | None = None):
    """Return ``(ids, Y)`` with ``Y`` of shape (N, 80) and dtype uint8.

    This computes class statistics, metric weights, and the stratified split
    without ever decoding an image.
    """
    annotations_dir = Path(annotations_dir) if annotations_dir else PATHS.train_labels
    names = sorted(glob("*.cls", root_dir=str(annotations_dir)))
    ids = [Path(n).stem for n in names]
    Y = np.zeros((len(names), NUM_CLASSES), dtype=np.uint8)
    for i, name in enumerate(names):
        with open(annotations_dir / name) as f:
            Y[i, [int(x) for x in f.read().split()]] = 1
    return ids, Y


def class_frequencies(Y: np.ndarray) -> np.ndarray:
    return Y.sum(axis=0).astype(np.int64)


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------
def random_split_indices(n: int, val_fraction: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_fraction)))
    return perm[n_val:], perm[:n_val]


def iterative_stratified_split(Y: np.ndarray, val_fraction: float, seed: int):
    """Iterative multi-label stratification (Sechidis et al., 2011).

    A purely random draw leaves very few positives in validation for rare
    classes (``hair drier`` has only 102 positives out of 65,000 images),
    which makes the local metric noisy even though those classes dominate the
    server score. This algorithm assigns examples from the rarest class to
    the most frequent one.
    """
    Y = np.asarray(Y, dtype=np.int64)
    n, c = Y.shape
    rng = np.random.default_rng(seed)
    fractions = np.array([1.0 - val_fraction, val_fraction])

    desired = np.outer(fractions, Y.sum(axis=0)).astype(float)  # (2, C)
    desired_total = fractions * n
    assignment = np.full(n, -1, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)

    while True:
        rem_counts = Y[remaining].sum(axis=0)
        positive = rem_counts > 0
        if not positive.any():
            break
        cls = int(np.flatnonzero(positive)[np.argmin(rem_counts[positive])])
        idxs = np.flatnonzero(remaining & (Y[:, cls] == 1))
        rng.shuffle(idxs)
        for i in idxs:
            col = desired[:, cls]
            cand = np.flatnonzero(col == col.max())
            if cand.size > 1:
                totals = desired_total[cand]
                cand = cand[totals == totals.max()]
            s = int(cand[0]) if cand.size == 1 else int(rng.choice(cand))
            assignment[i] = s
            remaining[i] = False
            desired[s] -= Y[i]
            desired_total[s] -= 1

    # Examples with no label (absent from this dataset, but we stay robust).
    leftovers = np.flatnonzero(remaining)
    if leftovers.size:
        rng.shuffle(leftovers)
        n_val = int(round(leftovers.size * val_fraction))
        assignment[leftovers[:n_val]] = 1
        assignment[leftovers[n_val:]] = 0

    train_idx = np.flatnonzero(assignment == 0).tolist()
    val_idx = np.flatnonzero(assignment == 1).tolist()
    return train_idx, val_idx


def get_split(
    Y: np.ndarray,
    val_fraction: float,
    seed: int,
    strategy: str = "stratified",
    cache_path: str | Path | None = None,
):
    """Reproducible split, cached on disk.

    The cache makes sure every experiment (precomputed features, fine-tuning,
    threshold calibration) shares exactly the same split.
    """
    if cache_path is None:
        cache_path = PATHS.outputs / f"split_{strategy}_{val_fraction:g}_{seed}.json"
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        if payload.get("n") == len(Y):
            return payload["train"], payload["val"]

    if strategy == "stratified":
        train_idx, val_idx = iterative_stratified_split(Y, val_fraction, seed)
    elif strategy == "random":
        train_idx, val_idx = random_split_indices(len(Y), val_fraction, seed)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "n": len(Y),
                "strategy": strategy,
                "val_fraction": val_fraction,
                "seed": seed,
                "train": train_idx,
                "val": val_idx,
            }
        )
    )
    return train_idx, val_idx


def _fraction_token(value: float) -> str:
    """``0.15`` -> ``"0.15"``, ``0.7`` -> ``"0.7"`` (stable despite binary floats)."""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def iterative_stratified_split_three(
    Y: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
):
    """Iterative stratification into three sets: train, validation, test.

    Same algorithm as ``iterative_stratified_split`` (Sechidis et al., 2011),
    with three quotas. The local test is a subset of the labeled images: the
    official challenge test set has no labels. ``get_split`` (80/20) is not
    used and its cache is left untouched.
    """
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be strictly positive")
    train_fraction = 1.0 - float(val_fraction) - float(test_fraction)
    if train_fraction <= 1e-9:
        raise ValueError(
            f"the train fraction is zero or negative "
            f"(val={val_fraction}, test={test_fraction})"
        )

    Y = np.asarray(Y, dtype=np.int64)
    n, _ = Y.shape
    rng = np.random.default_rng(seed)
    fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    fractions = fractions / fractions.sum()

    desired = np.outer(fractions, Y.sum(axis=0)).astype(float)  # (3, C)
    desired_total = fractions * n
    assignment = np.full(n, -1, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)

    while True:
        rem_counts = Y[remaining].sum(axis=0)
        positive = rem_counts > 0
        if not positive.any():
            break
        cls = int(np.flatnonzero(positive)[np.argmin(rem_counts[positive])])
        idxs = np.flatnonzero(remaining & (Y[:, cls] == 1))
        rng.shuffle(idxs)
        for i in idxs:
            col = desired[:, cls]
            cand = np.flatnonzero(col == col.max())
            if cand.size > 1:
                totals = desired_total[cand]
                cand = cand[totals == totals.max()]
            s = int(cand[0]) if cand.size == 1 else int(rng.choice(cand))
            assignment[i] = s
            remaining[i] = False
            desired[s] -= Y[i]
            desired_total[s] -= 1

    leftovers = np.flatnonzero(remaining)
    if leftovers.size:
        rng.shuffle(leftovers)
        n_val = int(round(leftovers.size * float(val_fraction)))
        n_test = int(round(leftovers.size * float(test_fraction)))
        if n_val + n_test > leftovers.size:
            n_test = leftovers.size - n_val
        assignment[leftovers[:n_val]] = 1
        assignment[leftovers[n_val:n_val + n_test]] = 2
        assignment[leftovers[n_val + n_test:]] = 0

    if np.any(assignment < 0):
        raise RuntimeError("incomplete split: some indices were not assigned")

    train_idx = np.flatnonzero(assignment == 0).tolist()
    val_idx = np.flatnonzero(assignment == 1).tolist()
    test_idx = np.flatnonzero(assignment == 2).tolist()
    return train_idx, val_idx, test_idx


def random_split_three(n: int, val_fraction: float, test_fraction: float, seed: int):
    """Random draw into three disjoint sets."""
    train_fraction = 1.0 - float(val_fraction) - float(test_fraction)
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("the three fractions must be strictly positive")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * test_fraction)))
    if n_val + n_test >= n:
        raise ValueError("train set is too small for these fractions")
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return train_idx, val_idx, test_idx


def get_three_way_split(
    Y: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    strategy: str = "stratified",
    cache_path: str | Path | None = None,
):
    """Reproducible train / validation / test split, cached apart from ``get_split``.

    The default file is
    ``outputs/split_three_stratified_0.7_0.15_0.15_42.json`` for the full set.
    A call with another ``n`` (a ``MAX_IMAGES`` trial) does not overwrite that
    cache: it writes a file whose name is suffixed by the size.
    """
    train_fraction = 1.0 - float(val_fraction) - float(test_fraction)
    if cache_path is None:
        cache_path = PATHS.outputs / (
            f"split_three_{strategy}_{_fraction_token(train_fraction)}_"
            f"{_fraction_token(val_fraction)}_{_fraction_token(test_fraction)}_{seed}.json"
        )
    cache_path = Path(cache_path)

    def _read(path: Path):
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        if (
            payload.get("n") == len(Y)
            and payload.get("seed") == seed
            and payload.get("strategy") == strategy
            and abs(float(payload.get("val_fraction", -1)) - float(val_fraction)) < 1e-9
            and abs(float(payload.get("test_fraction", -1)) - float(test_fraction)) < 1e-9
        ):
            return payload["train"], payload["val"], payload["test"]
        return None

    found = _read(cache_path)
    if found is not None:
        return found

    write_path = cache_path
    if cache_path.exists():
        write_path = cache_path.with_name(f"{cache_path.stem}_{len(Y)}{cache_path.suffix}")
        found = _read(write_path)
        if found is not None:
            return found

    if strategy == "stratified":
        train_idx, val_idx, test_idx = iterative_stratified_split_three(
            Y, val_fraction, test_fraction, seed
        )
    elif strategy == "random":
        train_idx, val_idx, test_idx = random_split_three(
            len(Y), val_fraction, test_fraction, seed
        )
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(
        json.dumps(
            {
                "n": len(Y),
                "strategy": strategy,
                "train_fraction": train_fraction,
                "val_fraction": val_fraction,
                "test_fraction": test_fraction,
                "seed": seed,
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
        )
    )
    return train_idx, val_idx, test_idx
