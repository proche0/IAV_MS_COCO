"""Paths, classes, and default hyperparameters for the challenge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# The 80 classes, in the order required by the assignment. The index in this
# tuple is the class id written in the .cls files and expected by the
# evaluation server.
# ---------------------------------------------------------------------------
CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 80

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    """Root of the dataset provided for the course.

    Override it with the ``MSCOCO_ROOT`` environment variable so the same code
    works locally and on Colab without editing this file.
    """
    env = os.environ.get("MSCOCO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT.parent / "ms-coco").resolve()


@dataclass
class Paths:
    data_root: Path = field(default_factory=_default_data_root)
    outputs: Path = field(default_factory=lambda: REPO_ROOT / "outputs")
    features: Path = field(default_factory=lambda: REPO_ROOT / "features")
    runs: Path = field(default_factory=lambda: REPO_ROOT / "runs")
    submissions: Path = field(default_factory=lambda: REPO_ROOT / "submissions")

    @property
    def train_images(self) -> Path:
        return self.data_root / "images" / "train"

    @property
    def test_images(self) -> Path:
        return self.data_root / "images" / "test"

    @property
    def train_labels(self) -> Path:
        return self.data_root / "labels" / "train"

    def check(self) -> None:
        for name in ("train_images", "test_images", "train_labels"):
            p = getattr(self, name)
            if not p.is_dir():
                raise FileNotFoundError(
                    f"Folder not found: {p}\n"
                    "Check the dataset root (MSCOCO_ROOT environment variable). "
                    "Expected layout: <root>/images/train, <root>/images/test, <root>/labels/train"
                )

    def mkdirs(self) -> None:
        for p in (self.outputs, self.features, self.runs, self.submissions):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

# Reference hyperparameters. Notebooks can override them.
DEFAULTS = {
    "image_size": 224,
    "resize_mode": "pad",
    "batch_size": 64,
    "num_workers": max(1, (os.cpu_count() or 2) - 2),
    "val_fraction": 0.2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 10,
    "threshold": 0.5,
}
