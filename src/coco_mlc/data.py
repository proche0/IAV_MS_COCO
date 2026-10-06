"""Datasets, transformations et decoupage train/validation.

Le layout reel du dataset fourni est :

    ms-coco/images/train/<id>.jpg   (65 000)
    ms-coco/images/test/<id>.jpg    (4 952)
    ms-coco/labels/train/<id>.cls   (65 000)

Les images ont ete redimensionnees a grand cote = 224 en conservant le ratio,
elles n'ont donc pas toutes la meme taille (ex. 224x149, 168x224).
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
# Transformations
# ---------------------------------------------------------------------------
class PadToSquare:
    """Complete l'image en carre par des bordures, sans rien recadrer.

    Les images du dataset ont des ratios varies. Un ``Resize((S, S))`` direct
    les deforme, et un ``CenterCrop`` supprime les bords ou peuvent se trouver
    des objets a predire. Le padding conserve l'integralite du contenu et le
    ratio d'origine.
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
    """Construit le pipeline de transformations.

    ``resize_mode`` controle la mise en forme geometrique :
      - ``"pad"``    : padding en carre puis redimensionnement (aucune perte) ;
      - ``"squash"`` : redimensionnement direct en (S, S), deforme le ratio ;
      - ``"crop"``   : redimensionnement du petit cote puis recadrage central.
    """
    if resize_mode == "pad":
        geometry = [PadToSquare(), T.Resize((image_size, image_size))]
    elif resize_mode == "squash":
        geometry = [T.Resize((image_size, image_size))]
    elif resize_mode == "crop":
        geometry = [T.Resize(image_size), T.CenterCrop(image_size)]
    else:
        raise ValueError(f"resize_mode inconnu : {resize_mode!r}")

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
        elif augment != "none":
            raise ValueError(f"augment inconnu : {augment!r}")

    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class COCOTrainImageDataset(Dataset):
    """Images annotees du sous-ensemble train, avec cibles multi-hot (80).

    Reprend la logique du dataset fourni dans le sujet : la liste des exemples
    est derivee des fichiers ``.cls`` tries, l'image correspondante est deduite
    du nom de fichier. ``return_id`` ajoute l'identifiant de l'image aux
    elements retournes, utile pour tracer les predictions.
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
            raise RuntimeError(f"Aucun fichier .cls dans {self.annotations_dir}")
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
    """Images du sous-ensemble test, retournees avec leur identifiant."""

    def __init__(self, img_dir: str | Path | None = None, transform=None):
        self.img_dir = Path(img_dir) if img_dir is not None else PATHS.test_images
        self.img_list = sorted(glob("*.jpg", root_dir=str(self.img_dir)))
        if not self.img_list:
            raise RuntimeError(f"Aucune image .jpg dans {self.img_dir}")
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
    """Vue d'un dataset restreinte a des indices, avec sa propre transformation.

    Indispensable ici : l'augmentation aleatoire doit s'appliquer au train mais
    pas a la validation, alors que les deux sous-ensembles proviennent du meme
    dataset de base.
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
    """Itere par tranches sur des tenseurs deja en memoire.

    Un ``DataLoader`` classique collecte les exemples un par un puis les
    assemble, ce qui coute plus cher que le calcul lui-meme quand la "donnee"
    est un vecteur de features de 512 valeurs et le modele une seule couche
    lineaire. Le decoupage direct par tranches supprime ce surcout.

    L'interface (``__iter__``, ``__len__``, attribut ``dataset``) est celle
    attendue par les fonctions de ``engine.py``, qui fonctionnent donc sans
    modification sur des features comme sur des images.
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
            raise ValueError("features et labels de tailles differentes")
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
# Lecture rapide des annotations (sans charger les images)
# ---------------------------------------------------------------------------
def load_label_matrix(annotations_dir: str | Path | None = None):
    """Retourne ``(ids, Y)`` avec ``Y`` de forme (N, 80) en uint8.

    Permet de calculer les statistiques de classes, les poids de la metrique et
    le decoupage stratifie sans jamais decoder une image.
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
# Decoupage train / validation
# ---------------------------------------------------------------------------
def random_split_indices(n: int, val_fraction: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_fraction)))
    return perm[n_val:], perm[:n_val]


def iterative_stratified_split(Y: np.ndarray, val_fraction: float, seed: int):
    """Stratification iterative multi-label (Sechidis et al., 2011).

    Un tirage purement aleatoire laisse tres peu de positifs en validation pour
    les classes rares (``hair drier`` n'a que 102 positifs sur 65 000 images),
    ce qui rend la metrique locale bruitee alors que ces classes dominent le
    score du serveur. Cet algorithme repartit les exemples en traitant les
    classes de la plus rare a la plus frequente.
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

    # Exemples sans aucun label (absents de ce dataset, mais on reste robuste).
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
    """Decoupage reproductible, mis en cache sur disque.

    Le cache garantit que toutes les experiences (features pre-calculees,
    fine-tuning, calibration de seuils) partagent exactement le meme split.
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
        raise ValueError(f"strategy inconnue : {strategy!r}")

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
