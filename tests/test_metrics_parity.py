"""Verifie que notre metrique reproduit celle du sujet.

Lancement : ``python3 tests/test_metrics_parity.py`` (ou ``pytest tests``).

La reference ci-dessous est la partie agregation de ``validation_loop`` copiee
telle quelle depuis le notebook du sujet. Elle n'est valide que lorsque chaque
classe a au moins un positif et que precision et rappel agreges sont non nuls ;
en dehors de ce domaine elle produit des ``inf``/``nan``, ce que notre version
corrige. Les tests couvrent donc le domaine commun, puis verifient le
comportement de notre version sur les cas degeneres.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from coco_mlc.metrics import (
    all_metrics,
    counts_from_predictions,
    server_metrics,
    threshold_count_table,
)
from coco_mlc.thresholds import apply_thresholds, tune_per_class_thresholds


def reference_metrics(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    """Agregation du sujet, recopiee sans modification."""
    class_total = {label: 0 for label in range(num_classes)}
    class_tp = {label: 0 for label in range(num_classes)}
    class_fp = {label: 0 for label in range(num_classes)}

    tps = predictions * labels
    fps = predictions - tps
    tps = tps.sum(dim=0)
    fps = fps.sum(dim=0)
    lbls = labels.sum(dim=0)
    for c in range(num_classes):
        class_tp[c] += tps[c]
        class_fp[c] += fps[c]
        class_total[c] += lbls[c]
    correct = tps.sum()

    class_prec, class_recall, freqs = [], [], []
    for c in range(num_classes):
        class_prec.append(0 if class_tp[c] == 0 else class_tp[c] / (class_tp[c] + class_fp[c]))
        class_recall.append(0 if class_tp[c] == 0 else class_tp[c] / class_total[c])
        freqs.append(class_total[c])

    freqs = torch.tensor(freqs)
    class_weights = 1.0 / freqs
    class_weights /= class_weights.sum()
    class_prec = torch.tensor(class_prec)
    class_recall = torch.tensor(class_recall)
    prec = (class_prec * class_weights).sum()
    recall = (class_recall * class_weights).sum()
    f1 = 2.0 / (1 / prec + 1 / recall)
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(recall),
        "accuracy": float(correct / freqs.sum()),
    }


def _random_case(n: int, c: int, seed: int, positive_rate: float, predict_rate: float):
    """Tire des cibles et des predictions, en garantissant un positif par classe."""
    g = torch.Generator().manual_seed(seed)
    labels = (torch.rand(n, c, generator=g) < positive_rate).float()
    labels[torch.randperm(n, generator=g)[:c], torch.arange(c)] = 1.0
    predictions = (torch.rand(n, c, generator=g) < predict_rate).float()
    # Au moins un vrai positif par classe, sinon la reference renvoie 0 partout
    # et la comparaison perd son interet.
    predictions[labels.argmax(dim=0), torch.arange(c)] = 1.0
    return predictions, labels


def test_parity_with_subject():
    for seed, positive_rate, predict_rate in [
        (0, 0.05, 0.08),
        (1, 0.15, 0.20),
        (2, 0.30, 0.50),
        (3, 0.02, 0.60),
    ]:
        predictions, labels = _random_case(400, 80, seed, positive_rate, predict_rate)
        ours = server_metrics(predictions, labels)
        ref = reference_metrics(predictions, labels, 80)
        for key in ("f1", "precision", "recall", "accuracy"):
            assert abs(ours[key] - ref[key]) < 1e-5, (
                f"seed={seed} {key}: {ours[key]:.8f} != {ref[key]:.8f}"
            )
    print("OK  parite avec validation_loop du sujet (4 configurations)")


def test_degenerate_cases():
    labels = torch.zeros(10, 5)
    labels[0, 0] = 1.0  # une seule classe a du support

    # Aucune prediction : precision et rappel nuls, donc F1 nul (et non une
    # division par zero comme dans le code du sujet).
    results = server_metrics(torch.zeros(10, 5), labels)
    assert results["f1"] == 0.0
    assert results["recall"] == 0.0

    # Les classes sans positif recoivent un poids nul au lieu d'un poids infini.
    results, per_class = server_metrics(torch.ones(10, 5), labels, class_metrics=True)
    assert per_class[0]["weight"] == 1.0
    assert all(per_class[c]["weight"] == 0.0 for c in range(1, 5))
    assert results["recall"] == 1.0
    print("OK  cas degeneres (support nul, precision nulle) sans nan ni inf")


def test_counts_consistency():
    predictions, labels = _random_case(200, 80, 7, 0.1, 0.15)
    tp, fp, total = counts_from_predictions(predictions, labels)
    assert torch.equal(total, labels.sum(dim=0))
    assert torch.equal(tp, (predictions.bool() & labels.bool()).sum(dim=0).float())
    assert torch.equal(fp, (predictions.bool() & ~labels.bool()).sum(dim=0).float())

    grid = [0.2, 0.5, 0.8]
    scores = torch.rand(200, 80)
    tp_table, fp_table, total_table = threshold_count_table(scores, labels, grid)
    for i, t in enumerate(grid):
        tp_i, fp_i, _ = counts_from_predictions((scores > t).float(), labels)
        assert torch.allclose(tp_table[:, i], tp_i)
        assert torch.allclose(fp_table[:, i], fp_i)
    print("OK  compteurs et tableau de seuils coherents")


def test_threshold_tuning_improves_score():
    g = torch.Generator().manual_seed(11)
    labels = (torch.rand(1500, 80, generator=g) < 0.04).float()
    labels[torch.randperm(1500, generator=g)[:80], torch.arange(80)] = 1.0
    # Scores informatifs mais mal calibres : le seuil 0.5 est trop severe.
    scores = torch.rand(1500, 80, generator=g) * 0.4 + labels * 0.35

    baseline = all_metrics(scores, labels, thresholds=0.5)["f1"]
    tuned, tuned_f1, _ = tune_per_class_thresholds(scores, labels, rounds=2, verbose=False)
    assert tuned_f1 >= baseline, f"{tuned_f1:.4f} < {baseline:.4f}"
    assert tuned.shape == (80,)
    print(f"OK  calibration par classe : F1 {baseline:.4f} -> {tuned_f1:.4f}")


def test_apply_thresholds_min_labels():
    scores = torch.tensor([[0.1, 0.2, 0.05], [0.9, 0.8, 0.1]])
    predictions = apply_thresholds(scores, torch.tensor([0.5, 0.5, 0.5]), min_labels=1)
    assert predictions.sum(dim=1).min() >= 1
    assert predictions[0].tolist() == [0.0, 1.0, 0.0]  # repli sur le plus probable
    assert predictions[1].tolist() == [1.0, 1.0, 0.0]
    print("OK  garde-fou : au moins une classe predite par image")


if __name__ == "__main__":
    test_parity_with_subject()
    test_degenerate_cases()
    test_counts_consistency()
    test_threshold_tuning_improves_score()
    test_apply_thresholds_min_labels()
    print("\nTous les tests passent.")
