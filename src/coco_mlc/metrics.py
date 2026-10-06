"""Metriques multi-label, dont la reproduction fidele du score du serveur.

Le serveur d'evaluation (et la fonction ``validation_loop`` fournie dans le
sujet) ne calcule pas un macro-F1 classique : la precision et le rappel de
chaque classe sont ponderes par l'inverse de la frequence de la classe dans
l'ensemble evalue, puis le F1 est la moyenne harmonique des deux agregats.

Consequence chiffree sur les 65 000 images d'entrainement : ``hair drier``
(102 positifs) pese 13,6 % du score et ``toaster`` (117) 11,9 %, alors que
``person`` (35 494) ne pese que 0,04 %. Les dix classes les plus rares
representent 43,8 % du score. Le rappel sur les classes rares est donc le
levier principal, bien avant la performance globale.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_float_tensor(x) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x) if isinstance(x, np.ndarray) else x)
    return t.float()


def metric_class_weights(totals: torch.Tensor) -> torch.Tensor:
    """Poids ``1/frequence`` normalises, comme dans le sujet.

    Difference assumee avec le code fourni : une classe sans aucun positif dans
    l'ensemble evalue y produirait un poids infini puis des ``nan`` sur toutes
    les classes. On lui attribue ici un poids nul et on renormalise sur les
    classes presentes.
    """
    totals = _as_float_tensor(totals)
    present = totals > 0
    weights = torch.zeros_like(totals)
    if not present.any():
        return weights
    weights[present] = 1.0 / totals[present]
    return weights / weights.sum()


def harmonic_f1(precision, recall):
    """Moyenne harmonique, avec 0 quand l'un des deux termes est nul.

    Le sujet ecrit ``2. / (1/prec + 1/recall)``, qui leve une division par zero
    des qu'une des deux grandeurs vaut 0 (cas courant aux premieres epoques).
    """
    precision = float(precision)
    recall = float(recall)
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 / (1.0 / precision + 1.0 / recall)


def counts_from_predictions(predictions, targets):
    """Compte ``(tp, fp, total)`` par classe a partir de decisions binaires."""
    predictions = _as_float_tensor(predictions)
    targets = _as_float_tensor(targets)
    tp = (predictions * targets).sum(dim=0)
    fp = (predictions - predictions * targets).sum(dim=0)
    total = targets.sum(dim=0)
    return tp, fp, total


def server_metrics_from_counts(tp, fp, total, class_metrics: bool = False):
    """Agrege les compteurs par classe selon la formule du serveur."""
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
        # Le sujet nomme "accuracy" le rapport sum(tp)/sum(positifs), qui est en
        # realite le rappel micro. On conserve le nom pour pouvoir comparer.
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
    """Score du serveur a partir de decisions binaires (N, C)."""
    tp, fp, total = counts_from_predictions(predictions, targets)
    return server_metrics_from_counts(tp, fp, total, class_metrics=class_metrics)


def standard_metrics(predictions, targets) -> dict[str, float]:
    """Macro et micro precision/rappel/F1, pour contextualiser le score pondere."""
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
    """Average precision par classe (aire sous la courbe P/R, style VOC/COCO).

    Independante du seuil de decision : utile pour comparer la qualite brute du
    classement de deux modeles sans melanger l'effet de la calibration.
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
    """Jeu complet de metriques pour un ensemble de scores et de cibles.

    ``thresholds`` accepte un scalaire ou un vecteur de taille (C,), ce qui
    permet d'evaluer directement des seuils par classe.
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
    """Pre-calcule ``tp`` et ``fp`` par (classe, seuil) pour la calibration.

    Retourne ``(tp, fp, total)`` de formes (C, T), (C, T) et (C,). Ce tableau
    rend l'optimisation des seuils par classe quasi instantanee : evaluer une
    combinaison de seuils devient une simple indexation.
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
