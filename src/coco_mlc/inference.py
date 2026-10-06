"""Reconstruction d'un modele a partir d'un checkpoint, pour l'inference.

Deux familles de checkpoints coexistent dans ce projet :

- ``kind="feature_head"`` : backbone gele + tete apprise sur features
  pre-calculees (``train_head.py``) ;
- ``kind="full_model"`` : reseau entierement affine (``train.py``).

Les deux sont ramenes ici a un unique ``nn.Module`` qui prend des images et
renvoie des logits, ce qui permet a ``predict.py`` et ``tune_thresholds.py`` de
traiter les deux cas sans distinction. Pour les tetes sur features, un chemin
rapide reste disponible via ``build_head_module`` quand le cache existe deja.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import NUM_CLASSES
from .engine import load_checkpoint
from .models import build_feature_extractor, build_head, build_model


class StandardizedHead(nn.Module):
    """Tete de classification precedee de la normalisation des features.

    Les statistiques de centrage/reduction font partie du modele : les oublier
    a l'inference produirait des predictions incoherentes avec l'entrainement.
    """

    def __init__(self, head: nn.Module, mean=None, std=None):
        super().__init__()
        self.head = head
        if mean is None or std is None:
            self.register_buffer("mean", None)
            self.register_buffer("std", None)
        else:
            self.register_buffer("mean", torch.as_tensor(np.asarray(mean)).float().reshape(1, -1))
            self.register_buffer("std", torch.as_tensor(np.asarray(std)).float().reshape(1, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is not None:
            features = (features - self.mean) / self.std
        return self.head(features)


class FeatureHeadModel(nn.Module):
    """Backbone gele suivi de la tete standardisee : images -> logits."""

    def __init__(self, backbone: nn.Module, head: StandardizedHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features.float())


def build_head_module(checkpoint: dict, device="cpu") -> StandardizedHead:
    """Reconstruit uniquement la tete, pour l'appliquer a des features en cache."""
    head = build_head(
        checkpoint["head"],
        checkpoint["feature_dim"],
        NUM_CLASSES,
        **checkpoint.get("head_kwargs", {}),
    )
    head.load_state_dict(checkpoint["model_state_dict"])
    module = StandardizedHead(head, checkpoint.get("standardize_mean"), checkpoint.get("standardize_std"))
    return module.to(device).eval()


def load_model(path: str | Path, device="cpu") -> tuple[nn.Module, dict]:
    """Charge un checkpoint et renvoie ``(modele sur images, metadonnees)``."""
    checkpoint = load_checkpoint(path, map_location=device)
    kind = checkpoint.get("kind", "full_model")

    if kind == "feature_head":
        backbone, feature_dim = build_feature_extractor(checkpoint["backbone"], pretrained=True)
        if feature_dim != checkpoint["feature_dim"]:
            raise ValueError(
                f"dimension de features incoherente : checkpoint {checkpoint['feature_dim']}, "
                f"backbone {feature_dim}"
            )
        net = FeatureHeadModel(backbone, build_head_module(checkpoint, device))
    elif kind == "full_model":
        net = build_model(
            checkpoint["backbone"],
            num_classes=NUM_CLASSES,
            pretrained=False,
            dropout=checkpoint.get("dropout", 0.0),
        )
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"type de checkpoint inconnu : {kind!r}")

    return net.to(device).eval(), checkpoint


def checkpoint_thresholds(checkpoint: dict, fallback: float = 0.5) -> torch.Tensor:
    th = checkpoint.get("thresholds")
    if th is None:
        return torch.full((NUM_CLASSES,), fallback)
    return torch.as_tensor(th).float().reshape(-1)


@torch.no_grad()
def score_features(head: StandardizedHead, features: np.ndarray, batch_size: int = 8192) -> torch.Tensor:
    """Probabilites par classe pour un tableau de features pre-calculees."""
    out = []
    for start in range(0, len(features), batch_size):
        chunk = torch.from_numpy(features[start : start + batch_size].astype(np.float32))
        out.append(torch.sigmoid(head(chunk)))
    return torch.cat(out)
