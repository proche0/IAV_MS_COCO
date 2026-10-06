"""Split a trois voies, augmentation, diagnostic biais/variance.

Lancement : ``python tests/test_experiment_protocol.py`` (ou ``pytest tests``).
Aucun poids ImageNet ni image du dataset n'est charge.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from coco_mlc.data import (
    build_transforms,
    get_three_way_split,
    iterative_stratified_split_three,
)
from coco_mlc.diagnostics import diagnose_errors, plot_error_curves, plot_model_diagram
from coco_mlc.models import load_compatible_weights


def _toy_labels() -> np.ndarray:
    y = np.zeros((300, 4), dtype=np.int64)
    y[:10, 0] = 1
    y[:40, 1] = 1
    y[:120, 2] = 1
    y[:250, 3] = 1
    return y


def _assert_partition(n, train, val, test):
    assert len(train) + len(val) + len(test) == n
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(range(n))


def test_three_way_split_is_a_reproducible_partition():
    y = _toy_labels()
    first = iterative_stratified_split_three(y, 0.15, 0.15, seed=42)
    second = iterative_stratified_split_three(y, 0.15, 0.15, seed=42)
    assert first == second
    train, val, test = first
    _assert_partition(len(y), train, val, test)

    for idx, expected in (train, 0.70), (val, 0.15), (test, 0.15):
        fraction = len(idx) / len(y)
        assert abs(fraction - expected) < 0.08, fraction

    for split in (train, val, test):
        assert y[split, 0].sum() >= 1


def test_three_way_cache_does_not_clobber_another_size():
    y = _toy_labels()
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "split_three_stratified_0.7_0.15_0.15_42.json"
        train, val, test = get_three_way_split(
            y, 0.15, 0.15, seed=42, cache_path=cache,
        )
        _assert_partition(len(y), train, val, test)
        payload = json.loads(cache.read_text())
        assert payload["n"] == len(y)

        again = get_three_way_split(y, 0.15, 0.15, seed=42, cache_path=cache)
        assert again == (train, val, test)

        short = y[:80]
        get_three_way_split(short, 0.15, 0.15, seed=42, cache_path=cache)
        assert json.loads(cache.read_text())["n"] == len(y)
        sibling = cache.with_name(f"{cache.stem}_80{cache.suffix}")
        assert sibling.exists()
        assert json.loads(sibling.read_text())["n"] == 80


def test_experiment_augment_has_geometry_and_color():
    pipeline = build_transforms(224, train=True, augment="experiment")
    kinds = {type(step) for step in pipeline.transforms}
    assert T.RandomHorizontalFlip in kinds
    assert T.RandomAffine in kinds
    assert T.ColorJitter in kinds


def test_diagnosis_matches_the_four_regimes():
    under = diagnose_errors(10, 12)
    over = diagnose_errors(1, 10)
    both = diagnose_errors(10, 20)
    ideal = diagnose_errors(0.5, 1)
    middle = diagnose_errors(4, 7)
    leaked = diagnose_errors(0.5, 1, test_error=6)

    assert under["regime"] == "underfit"
    assert over["regime"] == "overfit"
    assert both["regime"] == "both"
    assert ideal["regime"] == "ideal"
    assert middle["regime"] == "intermediate"
    assert not ideal["poor_test_generalization"]
    assert leaked["regime"] == "ideal"
    assert leaked["poor_test_generalization"]
    assert any("validation plus grand" in action for action in leaked["actions"])
    assert any("réseau plus volumineux" in action for action in under["actions"])
    assert any("augmentation" in action for action in over["actions"])
    assert any("régularisation" in action for action in both["actions"])
    assert any("réseau plus volumineux" in action for action in both["actions"])


def test_dropout_head_receives_the_linear_weights():
    class Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

    class Drop(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(4, 3))

    torch.manual_seed(0)
    source = Bare()
    target = Drop()
    copied = load_compatible_weights(target, source)
    assert "fc.1.weight" in copied
    assert torch.allclose(target.fc[1].weight, source.fc.weight)
    assert torch.allclose(target.fc[1].bias, source.fc.bias)


def test_curves_and_diagram_build_figures():
    history = [
        {"epoch": 1, "train_error": 40, "val_error": 42, "train_loss": 0.5, "val_loss": 0.55},
        {"epoch": 2, "train_error": 20, "val_error": 25, "train_loss": 0.3, "val_loss": 0.4},
    ]
    fig = plot_error_curves(history, title="essai")
    assert len(fig.axes) == 2
    fig.clear()

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Conv2d(3, 4, 1)
            self.fc = nn.Linear(4, 80)
            for param in self.features.parameters():
                param.requires_grad = False

    diagram = plot_model_diagram(Tiny(), frozen=True)
    assert diagram.axes
    diagram.clear()


if __name__ == "__main__":
    test_three_way_split_is_a_reproducible_partition()
    test_three_way_cache_does_not_clobber_another_size()
    test_experiment_augment_has_geometry_and_color()
    test_diagnosis_matches_the_four_regimes()
    test_dropout_head_receives_the_linear_weights()
    test_curves_and_diagram_build_figures()
    print("\nTous les tests passent.")
