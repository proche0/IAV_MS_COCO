"""Extraction et stockage des features d'un backbone gele.

Principe de la strategie "feature extraction" : le backbone pre-entraine est
fige, on ne reentraine que la tete de classification. Les features ne dependent
alors plus des parametres appris, donc un seul passage forward sur le dataset
suffit pour toutes les experiences de tete, de fonction de cout et de seuil.

Sur cette machine (CPU, 4 threads), c'est la difference entre 50 minutes par
epoque de fine-tuning ResNet18 et quelques secondes par experience de tete.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import PATHS


def cache_dir(model: str, image_size: int, resize_mode: str) -> Path:
    return PATHS.features / f"{model}_{image_size}_{resize_mode}"


def cache_path(model: str, image_size: int, resize_mode: str, subset: str) -> Path:
    return cache_dir(model, image_size, resize_mode) / f"{subset}.npz"


@torch.no_grad()
def extract_features(
    loader,
    backbone,
    device,
    with_labels: bool,
    progress: bool = True,
    desc: str = "features",
):
    """Parcourt un loader et renvoie ``(features, labels, ids)``.

    Les features sont stockees en float16 : la perte de precision est sans effet
    sur une tete lineaire et divise par deux la taille du cache.
    """
    backbone.eval()
    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ids: list[str] = []

    for batch in tqdm(loader, desc=desc, disable=not progress):
        images = batch[0].to(device, non_blocking=True)
        out = backbone(images)
        chunks.append(out.detach().float().cpu().numpy().astype(np.float16))
        if with_labels:
            labels.append(batch[1].detach().cpu().numpy().astype(np.uint8))
            ids.extend(batch[2])
        else:
            ids.extend(batch[1])

    features = np.concatenate(chunks)
    labels_arr = np.concatenate(labels) if with_labels else None
    return features, labels_arr, ids


def save_cache(path: str | Path, features: np.ndarray, ids: list[str], labels=None, meta=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"features": features, "ids": np.array(ids)}
    if labels is not None:
        payload["labels"] = labels
    if meta:
        payload["meta"] = np.array(repr(meta))
    np.savez(path, **payload)
    return path


def load_cache(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cache de features absent : {path}\n"
            "Lancez d'abord : python3 scripts/cache_features.py --model <nom>"
        )
    with np.load(path, allow_pickle=False) as data:
        out = {"features": data["features"], "ids": [str(x) for x in data["ids"]]}
        if "labels" in data:
            out["labels"] = data["labels"]
    return out


def load_train_cache(model: str, image_size: int, resize_mode: str) -> dict:
    data = load_cache(cache_path(model, image_size, resize_mode, "train"))
    if "labels" not in data:
        raise ValueError("Le cache train doit contenir les labels")
    return data


def load_test_cache(model: str, image_size: int, resize_mode: str) -> dict:
    return load_cache(cache_path(model, image_size, resize_mode, "test"))


def available_caches() -> list[Path]:
    if not PATHS.features.is_dir():
        return []
    return sorted(p for p in PATHS.features.iterdir() if (p / "train.npz").exists())


def standardize(train: np.ndarray, *others: np.ndarray):
    """Centre et reduit les features avec les statistiques du train.

    Les features d'un backbone ImageNet ont des echelles tres variables selon la
    dimension ; la normalisation accelere nettement la convergence d'une tete
    lineaire entrainee avec Adam.
    """
    train = train.astype(np.float32)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    scaled = [(train - mean) / std]
    scaled += [(o.astype(np.float32) - mean) / std for o in others]
    return (*scaled, mean, std)
