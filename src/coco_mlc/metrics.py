"""Multi-label metrics, including a faithful copy of the server score.

The evaluation server (and the ``validation_loop`` from the assignment) does
not compute a classic macro-F1. Precision and recall of each class are weighted
by the inverse of that class frequency on the evaluated set, then F1 is the
harmonic mean of the two aggregates.

On the 65,000 training images this means: ``hair drier`` (102 positives) is
13.6% of the score and ``toaster`` (117) is 11.9%, while ``person`` (35,494)
is only 0.04%. The ten rarest classes are 43.8% of the score. Recall on rare
classes is the main lever, well ahead of overall accuracy.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_float_tensor(x) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x) if isinstance(x, np.ndarray) else x)
    return t.float()


def metric_class_weights(totals: torch.Tensor) -> torch.Tensor:
    """Normalized ``1/frequency`` weights, as in the assignment.

    Difference from the provided code: a class with no positive example in the
    evaluated set would get an infinite weight, then ``nan`` on every class.
    Here that class gets weight zero, and we renormalize over the classes
    that are present.
    """
    totals = _as_float_tensor(totals)
    present = totals > 0
    weights = torch.zeros_like(totals)
    if not present.any():
        return weights
    weights[present] = 1.0 / totals[present]
    return weights / weights.sum()


def harmonic_f1(precision, recall):
    """Harmonic mean. Returns 0 when either term is zero.

    The assignment writes ``2. / (1/prec + 1/recall)``, which divides by zero
    as soon as one of the two values is 0 (common in the first epochs).
    """
    precision = float(precision)
    recall = float(recall)
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 / (1.0 / precision + 1.0 / recall)


def counts_from_predictions(predictions, targets):
    """Count ``(tp, fp, total)`` per class from binary decisions."""
    predictions = _as_float_tensor(predictions)
    targets = _as_float_tensor(targets)
    tp = (predictions * targets).sum(dim=0)
    fp = (predictions - predictions * targets).sum(dim=0)
    total = targets.sum(dim=0)
    return tp, fp, total


def server_metrics_from_counts(tp, fp, total, class_metrics: bool = False):
    """Aggregate per-class counts with the server formula."""
    tp, fp, total = _as_float_tensor(tp), _as_float_tensor(fp), _as_float_tensor(total)

    class_prec = torch.where(tp > 0, tp / (tp + fp).clamp_min(1e-12), torch.zeros_like(tp))
    class_recall = torch.where(tp > 0, tp / total.clamp_min(1e-12), torch.zeros_like(tp))

    weights = metric_class_weights(total)
    precision = float((class_prec * weights).sum())
    recall = float((class_recall * weights).sum())

    results = {
        "f1": harmonic_f1(precision, recall),
        "precision": precision,
        "recall": recall,
        # The assignment calls "accuracy" the ratio sum(tp)/sum(positives),
        # which is actually micro recall. The name is kept so scores stay comparable.
        "accuracy": float(tp.sum() / total.sum().clamp_min(1e-12)),
    }

    if class_metrics:
        per_class = [
            {
                "f1": harmonic_f1(p, r),
                "precision": float(p),
                "recall": float(r),
                "support": int(s),
                "weight": float(w),
            }
            for p, r, s, w in zip(class_prec, class_recall, total, weights)
        ]
        return results, per_class
    return results


def server_metrics(predictions, targets, class_metrics: bool = False):
    """Server score from binary decisions of shape (N, C)."""
    tp, fp, total = counts_from_predictions(predictions, targets)
    return server_metrics_from_counts(tp, fp, total, class_metrics=class_metrics)


def standard_metrics(predictions, targets) -> dict[str, float]:
    """Macro and micro precision/recall/F1, to put the weighted score in context."""
    tp, fp, total = counts_from_predictions(predictions, targets)
    fn = total - tp

    class_prec = tp / (tp + fp).clamp_min(1e-12)
    class_recall = tp / (tp + fn).clamp_min(1e-12)
    denom = (class_prec + class_recall).clamp_min(1e-12)
    class_f1 = 2 * class_prec * class_recall / denom

    micro_p = float(tp.sum() / (tp.sum() + fp.sum()).clamp_min(1e-12))
    micro_r = float(tp.sum() / (tp.sum() + fn.sum()).clamp_min(1e-12))

    return {
        "macro_precision": float(class_prec.mean()),
        "macro_recall": float(class_recall.mean()),
        "macro_f1": float(class_f1.mean()),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": harmonic_f1(micro_p, micro_r),
    }


def average_precision_per_class(scores, targets) -> torch.Tensor:
    """Average precision per class (area under the P/R curve, VOC/COCO style).

    It does not depend on the decision threshold, so it compares the raw
    ranking of two models without mixing in the effect of calibration.
    """
    scores = _as_float_tensor(scores)
    targets = _as_float_tensor(targets)
    n_classes = scores.shape[1]
    aps = torch.zeros(n_classes)
    for c in range(n_classes):
        y = targets[:, c]
        n_pos = int(y.sum())
        if n_pos == 0:
            aps[c] = float("nan")
            continue
        order = torch.argsort(scores[:, c], descending=True)
        y_sorted = y[order]
        tp_cum = torch.cumsum(y_sorted, dim=0)
        ranks = torch.arange(1, len(y_sorted) + 1, dtype=torch.float32)
        precision_at_hits = (tp_cum / ranks)[y_sorted > 0]
        aps[c] = float(precision_at_hits.mean())
    return aps


def mean_average_precision(scores, targets) -> float:
    aps = average_precision_per_class(scores, targets)
    valid = ~torch.isnan(aps)
    return float(aps[valid].mean()) if valid.any() else 0.0


def all_metrics(scores, targets, thresholds=0.5, class_metrics: bool = False):
    """Full metric set for one batch of scores and targets.

    ``thresholds`` can be a scalar or a vector of size (C,), so per-class
    thresholds can be evaluated directly.
    """
    scores = _as_float_tensor(scores)
    targets = _as_float_tensor(targets)
    th = _as_float_tensor(thresholds)
    predictions = (scores > th).float()

    out = server_metrics(predictions, targets, class_metrics=class_metrics)
    if class_metrics:
        results, per_class = out
    else:
        results, per_class = out, None
    results = dict(results)
    results.update(standard_metrics(predictions, targets))
    results["mAP"] = mean_average_precision(scores, targets)
    return (results, per_class) if class_metrics else results


def threshold_count_table(scores, targets, grid):
    """Precompute ``tp`` and ``fp`` per (class, threshold) for calibration.

    Returns ``(tp, fp, total)`` with shapes (C, T), (C, T) and (C,). This table
    makes per-class threshold search almost free: scoring one combination is
    just an index lookup.
    """
    scores = _as_float_tensor(scores)
    targets = _as_float_tensor(targets)
    grid = _as_float_tensor(grid)
    n_classes = scores.shape[1]

    tp = torch.zeros(n_classes, len(grid))
    fp = torch.zeros(n_classes, len(grid))
    for t_idx, t in enumerate(grid):
        predictions = (scores > t).float()
        tp_c, fp_c, total = counts_from_predictions(predictions, targets)
        tp[:, t_idx] = tp_c
        fp[:, t_idx] = fp_c
    return tp, fp, targets.sum(dim=0)


def format_metrics(results: dict[str, float], keys=None) -> str:
    keys = keys or ("f1", "precision", "recall", "accuracy", "macro_f1", "micro_f1", "mAP")
    return " | ".join(f"{k}={results[k]:.4f}" for k in keys if k in results)
