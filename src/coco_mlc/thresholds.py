"""Calibration des seuils de decision multi-label.

C'est le levier le plus rentable de ce challenge. La metrique du serveur pondere
chaque classe par l'inverse de sa frequence : abaisser le seuil des classes
rares augmente fortement leur rappel, pour un cout de precision reparti sur
l'ensemble du score. Un seuil unique a 0,5 est donc tres sous-optimal.

La calibration se fait exclusivement sur l'ensemble de validation.
"""

from __future__ import annotations

import numpy as np
import torch

from .metrics import (
    harmonic_f1,
    metric_class_weights,
    threshold_count_table,
)

DEFAULT_GRID = np.round(np.arange(0.02, 0.901, 0.02), 4)


def _aggregate_f1(tp: torch.Tensor, fp: torch.Tensor, total: torch.Tensor, weights: torch.Tensor):
    """F1 pondere du serveur a partir de compteurs par classe."""
    class_prec = torch.where(tp > 0, tp / (tp + fp).clamp_min(1e-12), torch.zeros_like(tp))
    class_recall = torch.where(tp > 0, tp / total.clamp_min(1e-12), torch.zeros_like(tp))
    precision = float((class_prec * weights).sum())
    recall = float((class_recall * weights).sum())
    return harmonic_f1(precision, recall), precision, recall


def tune_global_threshold(scores, targets, grid=None):
    """Meilleur seuil unique, et la courbe complete pour l'analyse."""
    grid = np.asarray(DEFAULT_GRID if grid is None else grid, dtype=np.float64)
    tp_table, fp_table, total = threshold_count_table(scores, targets, grid)
    weights = metric_class_weights(total)

    curve = []
    for t_idx, t in enumerate(grid):
        f1, precision, recall = _aggregate_f1(
            tp_table[:, t_idx], fp_table[:, t_idx], total, weights
        )
        curve.append({"threshold": float(t), "f1": f1, "precision": precision, "recall": recall})

    best = max(curve, key=lambda row: row["f1"])
    return best["threshold"], best["f1"], curve


def tune_per_class_thresholds(
    scores,
    targets,
    grid=None,
    rounds: int = 4,
    init_threshold: float | None = None,
    verbose: bool = True,
):
    """Seuils par classe par montee de coordonnees sur la metrique du serveur.

    Les seuils ne sont pas separables : la precision agregee melange toutes les
    classes, donc modifier le seuil d'une classe change le score des autres.
    On optimise donc classe par classe, en repetant plusieurs passes jusqu'a
    stabilisation. Le tableau de compteurs pre-calcule rend chaque evaluation
    quasi gratuite.
    """
    grid = np.asarray(DEFAULT_GRID if grid is None else grid, dtype=np.float64)
    tp_table, fp_table, total = threshold_count_table(scores, targets, grid)
    weights = metric_class_weights(total)
    n_classes, n_thresholds = tp_table.shape

    if init_threshold is None:
        init_threshold, _, _ = tune_global_threshold(scores, targets, grid)
    selection = np.full(n_classes, int(np.abs(grid - init_threshold).argmin()), dtype=np.int64)

    idx = torch.arange(n_classes)

    def score_of(sel: np.ndarray):
        sel_t = torch.as_tensor(sel)
        return _aggregate_f1(tp_table[idx, sel_t], fp_table[idx, sel_t], total, weights)

    best_f1 = score_of(selection)[0]
    history = [{"round": 0, "f1": best_f1}]

    # Les classes rares pesent le plus : les traiter d'abord fait converger
    # la montee de coordonnees plus vite.
    order = np.argsort(-weights.numpy())

    for r in range(1, rounds + 1):
        improved = False
        for c in order:
            current = selection[c]
            best_local, best_choice = best_f1, current
            for t_idx in range(n_thresholds):
                if t_idx == current:
                    continue
                selection[c] = t_idx
                f1 = score_of(selection)[0]
                if f1 > best_local:
                    best_local, best_choice = f1, t_idx
            selection[c] = best_choice
            if best_local > best_f1:
                best_f1 = best_local
                improved = True
        history.append({"round": r, "f1": best_f1})
        if verbose:
            print(f"  passe {r}: F1 validation = {best_f1:.4f}")
        if not improved:
            break

    thresholds = torch.tensor(grid[selection], dtype=torch.float32)
    return thresholds, best_f1, history


def apply_thresholds(scores, thresholds, min_labels: int = 1) -> torch.Tensor:
    """Decisions binaires, avec un minimum de classes predites par image.

    Le format de soumission attend une liste d'indices par image. Une liste
    vide ne peut etre qu'un faux negatif garanti, donc on retient au moins la
    classe la plus probable.
    """
    scores = torch.as_tensor(scores).float()
    thresholds = torch.as_tensor(thresholds).float()
    predictions = (scores > thresholds).float()

    if min_labels > 0:
        missing = predictions.sum(dim=1) < min_labels
        if missing.any():
            k = min_labels
            top = scores[missing].topk(k, dim=1).indices
            rows = torch.nonzero(missing, as_tuple=True)[0].unsqueeze(1).expand(-1, k)
            predictions[rows.reshape(-1), top.reshape(-1)] = 1.0
    return predictions
