"""Calibration of multi-label decision thresholds.

This is the most useful lever in this challenge. The server metric weights
each class by the inverse of its frequency: lowering the threshold of rare
classes raises their recall a lot, for a precision cost that is spread over
the whole score. A single threshold of 0.5 is therefore a poor default.

Calibration is done only on the validation set.
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
    """Weighted server F1 from per-class counts."""
    class_prec = torch.where(tp > 0, tp / (tp + fp).clamp_min(1e-12), torch.zeros_like(tp))
    class_recall = torch.where(tp > 0, tp / total.clamp_min(1e-12), torch.zeros_like(tp))
    precision = float((class_prec * weights).sum())
    recall = float((class_recall * weights).sum())
    return harmonic_f1(precision, recall), precision, recall


def tune_global_threshold(scores, targets, grid=None):
    """Best single threshold, plus the full curve for analysis."""
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
    """Per-class thresholds by coordinate ascent on the server metric.

    The thresholds are not independent: the aggregated precision mixes every
    class, so changing one class threshold changes the score of the others.
    We therefore optimize one class at a time, and repeat several passes until
    the score stops moving. The precomputed count table makes each evaluation
    almost free.
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

    # Rare classes weigh the most: handling them first makes coordinate
    # ascent converge faster.
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
            print(f"  pass {r}: validation F1 = {best_f1:.4f}")
        if not improved:
            break

    thresholds = torch.tensor(grid[selection], dtype=torch.float32)
    return thresholds, best_f1, history


def apply_thresholds(scores, thresholds, min_labels: int = 1) -> torch.Tensor:
    """Binary decisions, with a minimum number of predicted classes per image.

    The submission format expects a list of class indices per image. An empty
    list can only be a guaranteed false negative, so we keep at least the
    most likely class.
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
