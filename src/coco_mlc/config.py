"""Chemins, classes et hyper-parametres par defaut du challenge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Les 80 classes, dans l'ordre impose par le sujet. L'indice dans ce tuple est
# l'identifiant de classe ecrit dans les fichiers .cls et attendu par le
# serveur d'evaluation.
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
    """Racine du dataset fourni par le cours.

    Surchargeable par la variable d'environnement ``MSCOCO_ROOT``, ce qui
    permet d'utiliser le meme code en local et sur Colab sans edition.
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
                    f"Dossier introuvable : {p}\n"
                    "Verifiez la racine du dataset (variable d'environnement MSCOCO_ROOT). "
                    "Structure attendue : <racine>/images/train, <racine>/images/test, <racine>/labels/train"
                )

    def mkdirs(self) -> None:
        for p in (self.outputs, self.features, self.runs, self.submissions):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

# Hyper-parametres de reference, surcharges par les arguments de ligne de
# commande des scripts.
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
