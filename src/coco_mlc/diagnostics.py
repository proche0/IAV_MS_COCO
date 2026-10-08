"""Error curves, model diagram, and bias/variance diagnosis.

The plotted error is the complement of the server F1, in percent:
``100 * (1 - F1)``. That is the challenge criterion, not a single-label
classification error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .models import classifier_in_features, count_parameters

# Thresholds in percentage points, set from the protocol examples:
# underfit 10/12, overfit 1/10, both 10/20, ideal 0.5/1.
HIGH_ERROR = 8.0
IDEAL_TRAIN = 2.0
IDEAL_VAL = 3.0
IDEAL_GAP = 2.0
CLOSE_GAP = 4.0
LARGE_GAP = 5.0
OVERFIT_TRAIN = 5.0
TEST_GAP = 3.0

UNDERFIT_ACTIONS = [
    "Use a larger network (more layers or more units).",
    "Train for longer.",
    "Change the optimizer.",
    "Run a hyperparameter search.",
]
OVERFIT_ACTIONS = [
    "Collect a larger and more varied training set.",
    "Use stronger data augmentation.",
    "Add regularization (L2 penalty, dropout).",
    "Run a hyperparameter search.",
]
TEST_ACTION = (
    "The local test is worse than the validation set: build a larger and more "
    "varied validation set, so the model does not overfit the evaluation data. "
    "Do not reuse this test set to choose hyperparameters."
)


def error_percent(f1: float) -> float:
    """Error percent matching the server F1."""
    return 100.0 * (1.0 - float(f1))


def save_history(history, path: str | Path) -> Path:
    """Write the history of one stage so the curves can be plotted again later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(history)).to_csv(path, index=False)
    return path


def _as_frame(history) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        frame = history.copy()
    else:
        frame = pd.DataFrame(list(history))
    if "train_error" not in frame.columns and "train_f1" in frame.columns:
        frame["train_error"] = frame["train_f1"].map(error_percent)
    if "val_error" not in frame.columns and "val_f1" in frame.columns:
        frame["val_error"] = frame["val_f1"].map(error_percent)
    return frame


def plot_error_curves(history, title: str | None = None):
    """Train/validation errors (%) and losses, per epoch."""
    frame = _as_frame(history)
    if frame.empty:
        raise ValueError("empty history")
    epochs = frame["epoch"] if "epoch" in frame.columns else range(1, len(frame) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(epochs, frame["train_error"], marker="o", label="Train")
    axes[0].plot(epochs, frame["val_error"], marker="o", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Error (%)  =  100 × (1 − F1)")
    axes[0].set_title("Errors")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, frame["train_loss"], marker="o", label="Train")
    axes[1].plot(epochs, frame["val_loss"], marker="o", label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Losses")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_model_diagram(model, frozen: bool):
    """Three-block diagram: preprocessing, backbone, head with 80 logits."""
    total, trainable = count_parameters(model)
    feature_dim = classifier_in_features(model)
    frozen_params = total - trainable

    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    if frozen:
        extractor_sub = f"frozen\n{frozen_params / 1e6:.1f} M"
        head_sub = f"trainable\n{trainable / 1e6:.2f} M"
    else:
        extractor_sub = "trainable"
        head_sub = "trainable"

    blocks = [
        (0.3, "Preprocessing", "ImageNet\n(fixed)"),
        (4.2, "Backbone", extractor_sub),
        (8.1, f"Head  {feature_dim} → 80", head_sub),
    ]
    for x, title, subtitle in blocks:
        patch = FancyBboxPatch(
            (x, 0.7), 3.2, 1.8,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.2,
            edgecolor="#1f4e79",
            facecolor="#e8f1fa",
        )
        ax.add_patch(patch)
        ax.text(x + 1.6, 1.85, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + 1.6, 1.15, subtitle, ha="center", va="center", fontsize=9)

    for x in (3.5, 7.4):
        ax.annotate(
            "",
            xy=(x + 0.65, 1.6),
            xytext=(x, 1.6),
            arrowprops={"arrowstyle": "->", "color": "#1f4e79", "lw": 1.4},
        )

    mode = "frozen backbone, head only" if frozen else "whole network trainable"
    ax.set_title(f"{total / 1e6:.1f} M parameters — {mode}", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def diagnose_errors(
    train_error: float,
    val_error: float,
    test_error: float | None = None,
) -> dict:
    """Compare train / validation / test errors and suggest a next step.

    Errors are percentages ``100 * (1 - F1)``. The train/validation case is
    one of underfit, overfit, both, ideal, or intermediate. A test that is
    clearly worse than validation adds a warning, without replacing that case.
    The dict is returned without printing: the notebook writes it to JSON and
    prints the text through ``format_diagnosis``.
    """
    train_error = float(train_error)
    val_error = float(val_error)
    gap = val_error - train_error

    if train_error >= HIGH_ERROR and val_error >= HIGH_ERROR and gap >= LARGE_GAP:
        regime = "both"
        title = "Underfitting and overfitting at the same time"
        actions = _unique(UNDERFIT_ACTIONS + OVERFIT_ACTIONS)
    elif (
        train_error < IDEAL_TRAIN
        and val_error < IDEAL_VAL
        and abs(gap) < IDEAL_GAP
    ):
        regime = "ideal"
        title = "Ideal model"
        actions = [
            "Errors are very low and close: the protocol can stay as it is."
        ]
    elif train_error >= HIGH_ERROR and val_error >= HIGH_ERROR and gap < CLOSE_GAP:
        regime = "underfit"
        title = "Underfitting (high bias)"
        actions = list(UNDERFIT_ACTIONS)
    elif train_error < OVERFIT_TRAIN and gap >= LARGE_GAP:
        regime = "overfit"
        title = "Overfitting (high variance)"
        actions = list(OVERFIT_ACTIONS)
    else:
        regime = "intermediate"
        title = "Intermediate case"
        actions = [
            "The errors do not match any of the four typical regimes. "
            "Read the absolute level and the gap before changing the model."
        ]

    test_gap = None
    poor_test = False
    if test_error is not None:
        test_error = float(test_error)
        test_gap = test_error - val_error
        poor_test = test_gap >= TEST_GAP
        if poor_test:
            actions = actions + [TEST_ACTION]

    return {
        "regime": regime,
        "title": title,
        "train_error": train_error,
        "val_error": val_error,
        "test_error": test_error,
        "gap": gap,
        "test_gap": test_gap,
        "poor_test_generalization": poor_test,
        "actions": actions,
    }


def _f1(error: float) -> float:
    return 1.0 - float(error) / 100.0


def _reading(result: dict) -> str:
    regime = result["regime"]
    if regime == "underfit":
        text = "Errors are high and close: the model has not learned enough yet."
    elif regime == "overfit":
        text = "Train error is low and validation drops: the model is memorizing the train set."
    elif regime == "both":
        text = (
            "Errors are high, and the gap is large: learning is incomplete "
            "and generalization is weak."
        )
    elif regime == "ideal":
        text = "Errors are low and close."
    else:
        text = "Neither the level nor the gap matches one of the four typical examples."
    if result.get("poor_test_generalization"):
        text += " The local test is clearly worse than validation."
    return text


def format_diagnosis(result: dict, heading: str | None = None) -> str:
    """Readable summary of one diagnosis, without the action list."""
    lines = []
    if heading:
        lines.append(heading)
    lines.append(result["title"])

    f1 = f"F1 train {_f1(result['train_error']):.3f} | validation {_f1(result['val_error']):.3f}"
    if result.get("test_error") is not None:
        f1 += f" | test {_f1(result['test_error']):.3f}"
    lines.append(f1)
    lines.append(
        f"Train error {result['train_error']:.1f}% | "
        f"validation {result['val_error']:.1f}% | "
        f"gap {result['gap']:+.1f} pts"
    )
    if result.get("test_error") is not None:
        lines.append(
            f"Test {result['test_error']:.1f}% | "
            f"gap with validation {result['test_gap']:+.1f} pts"
        )
    lines.append(_reading(result))
    return "\n".join(lines)


def format_actions(result: dict) -> str:
    """Grouped suggestions, printed once for the operating point."""
    lines = ["Suggested next steps"]
    regime = result["regime"]
    if regime == "both":
        lines.append("Learning")
        lines.extend(f"- {item}" for item in UNDERFIT_ACTIONS)
        lines.append("Generalization")
        lines.extend(f"- {item}" for item in OVERFIT_ACTIONS)
    elif regime == "ideal":
        lines.append("- The protocol can stay as it is.")
    elif regime == "intermediate":
        lines.append("- Compare the error level and the gap before changing the model.")
    else:
        kept = [item for item in result["actions"] if item != TEST_ACTION]
        lines.extend(f"- {item}" for item in kept)
    if result.get("poor_test_generalization"):
        lines.append("Test")
        lines.append(
            "- Make the validation set larger and more varied. "
            "Do not tune hyperparameters on this test set."
        )
    return "\n".join(lines)


def format_report(sections: list[tuple[str, dict]]) -> str:
    """Short summary of three stages, then the suggestions of the last case only."""
    blocks = [format_diagnosis(result, heading=heading) for heading, result in sections]
    if sections:
        blocks.append(format_actions(sections[-1][1]))
    return "\n\n".join(blocks)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
