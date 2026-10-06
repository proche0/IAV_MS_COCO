"""Courbes d'erreur, schema de modele et diagnostic biais / variance.

L'erreur affichee est le complement du F1 du serveur, en pourcentage :
``100 * (1 - F1)``. C'est le critere du challenge, pas une erreur de
classification mono-label.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .models import classifier_in_features, count_parameters

# Seuils en points de pourcentage, cales sur les exemples du protocole :
# sous-apprentissage 10/12, surapprentissage 1/10, les deux 10/20, ideal 0,5/1.
HIGH_ERROR = 8.0
IDEAL_TRAIN = 2.0
IDEAL_VAL = 3.0
IDEAL_GAP = 2.0
CLOSE_GAP = 4.0
LARGE_GAP = 5.0
OVERFIT_TRAIN = 5.0
TEST_GAP = 3.0

UNDERFIT_ACTIONS = [
    "Utiliser un réseau plus volumineux (plus de couches ou d'unités).",
    "Entraîner plus longtemps.",
    "Changer d'algorithme d'optimisation.",
    "Lancer une recherche d'hyperparamètres.",
]
OVERFIT_ACTIONS = [
    "Agrandir et diversifier le jeu d'entraînement.",
    "Intensifier l'augmentation de données.",
    "Ajouter de la régularisation (norme L2, dropout).",
    "Lancer une recherche d'hyperparamètres.",
]
TEST_ACTION = (
    "Le test local sous-performe par rapport à la validation : constituer un "
    "ensemble de validation plus grand et plus diversifié, pour éviter de "
    "surapprendre les données d'évaluation. Ne pas réutiliser ce test pour "
    "choisir les hyperparamètres."
)


def error_percent(f1: float) -> float:
    """Pourcentage d'erreur associe au F1 du serveur."""
    return 100.0 * (1.0 - float(f1))


def save_history(history, path: str | Path) -> Path:
    """Ecrit l'historique d'une etape pour reafficher les courbes plus tard."""
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
    """Erreurs train/validation (%) et pertes, par epoque."""
    frame = _as_frame(history)
    if frame.empty:
        raise ValueError("historique vide")
    epochs = frame["epoch"] if "epoch" in frame.columns else range(1, len(frame) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(epochs, frame["train_error"], marker="o", label="Entraînement")
    axes[0].plot(epochs, frame["val_error"], marker="o", label="Validation")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Erreur (%)  =  100 × (1 − F1)")
    axes[0].set_title("Erreurs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, frame["train_loss"], marker="o", label="Entraînement")
    axes[1].plot(epochs, frame["val_loss"], marker="o", label="Validation")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Perte")
    axes[1].set_title("Pertes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_model_diagram(model, frozen: bool):
    """Schema en trois blocs : prétraitement, extracteur, tête à 80 logits."""
    total, trainable = count_parameters(model)
    feature_dim = classifier_in_features(model)
    frozen_params = total - trainable

    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    if frozen:
        extractor_sub = f"gelé\n{frozen_params / 1e6:.1f} M"
        head_sub = f"entraînable\n{trainable / 1e6:.2f} M"
    else:
        extractor_sub = "entraînable"
        head_sub = "entraînable"

    blocks = [
        (0.3, "Prétraitement", "ImageNet\n(figé)"),
        (4.2, "Extracteur", extractor_sub),
        (8.1, f"Tête  {feature_dim} → 80", head_sub),
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

    mode = "extracteur gelé, tête seule" if frozen else "réseau entier entraînable"
    ax.set_title(f"{total / 1e6:.1f} M paramètres — {mode}", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def diagnose_errors(
    train_error: float,
    val_error: float,
    test_error: float | None = None,
) -> dict:
    """Confronte les erreurs train / validation / test et propose la suite.

    Les erreurs sont des pourcentages ``100 * (1 - F1)``. Le cas train/validation
    est choisi parmi sous-apprentissage, surapprentissage, les deux, idéal, ou
    intermédiaire. Un test nettement pire que la validation ajoute un avertissement,
    sans remplacer ce cas. Le dictionnaire est renvoyé sans affichage : le notebook
    l'écrit en JSON et n'imprime que le texte via ``format_diagnosis``.
    """
    train_error = float(train_error)
    val_error = float(val_error)
    gap = val_error - train_error

    if train_error >= HIGH_ERROR and val_error >= HIGH_ERROR and gap >= LARGE_GAP:
        regime = "both"
        title = "Sous-apprentissage et surapprentissage simultanés"
        actions = _unique(UNDERFIT_ACTIONS + OVERFIT_ACTIONS)
    elif (
        train_error < IDEAL_TRAIN
        and val_error < IDEAL_VAL
        and abs(gap) < IDEAL_GAP
    ):
        regime = "ideal"
        title = "Modèle idéal"
        actions = [
            "Les erreurs sont très basses et proches : le protocole peut être conservé."
        ]
    elif train_error >= HIGH_ERROR and val_error >= HIGH_ERROR and gap < CLOSE_GAP:
        regime = "underfit"
        title = "Sous-apprentissage (biais élevé)"
        actions = list(UNDERFIT_ACTIONS)
    elif train_error < OVERFIT_TRAIN and gap >= LARGE_GAP:
        regime = "overfit"
        title = "Surapprentissage (variance élevée)"
        actions = list(OVERFIT_ACTIONS)
    else:
        regime = "intermediate"
        title = "Cas intermédiaire"
        actions = [
            "Les erreurs ne correspondent à aucun des quatre régimes types. "
            "Lire le niveau absolu et l'écart avant de changer de modèle."
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
        text = "Erreurs hautes et proches : le modèle n'a pas encore assez appris."
    elif regime == "overfit":
        text = "L'entraînement est bas et la validation décroche : le modèle retient le train."
    elif regime == "both":
        text = (
            "Erreurs hautes, et un écart net : apprentissage incomplet "
            "et généralisation faible."
        )
    elif regime == "ideal":
        text = "Erreurs basses et proches."
    else:
        text = "Ni le niveau ni l'écart ne correspondent à un des quatre exemples types."
    if result.get("poor_test_generalization"):
        text += " Le test local est nettement pire que la validation."
    return text


def format_diagnosis(result: dict, heading: str | None = None) -> str:
    """Résumé lisible d'un diagnostic, sans la liste d'actions."""
    lines = []
    if heading:
        lines.append(heading)
    lines.append(result["title"])

    f1 = f"F1 train {_f1(result['train_error']):.3f} | validation {_f1(result['val_error']):.3f}"
    if result.get("test_error") is not None:
        f1 += f" | test {_f1(result['test_error']):.3f}"
    lines.append(f1)
    lines.append(
        f"Erreur train {result['train_error']:.1f} % | "
        f"validation {result['val_error']:.1f} % | "
        f"écart {result['gap']:+.1f} pts"
    )
    if result.get("test_error") is not None:
        lines.append(
            f"Test {result['test_error']:.1f} % | "
            f"écart avec la validation {result['test_gap']:+.1f} pts"
        )
    lines.append(_reading(result))
    return "\n".join(lines)


def format_actions(result: dict) -> str:
    """Pistes regroupées, une seule fois pour le point de fonctionnement."""
    lines = ["Suite proposée"]
    regime = result["regime"]
    if regime == "both":
        lines.append("Apprentissage")
        lines.extend(f"- {item}" for item in UNDERFIT_ACTIONS)
        lines.append("Généralisation")
        lines.extend(f"- {item}" for item in OVERFIT_ACTIONS)
    elif regime == "ideal":
        lines.append("- Le protocole peut être conservé.")
    elif regime == "intermediate":
        lines.append("- Comparer le niveau des erreurs et l'écart avant de changer de modèle.")
    else:
        kept = [item for item in result["actions"] if item != TEST_ACTION]
        lines.extend(f"- {item}" for item in kept)
    if result.get("poor_test_generalization"):
        lines.append("Test")
        lines.append(
            "- Élargir et diversifier la validation. "
            "Ne pas régler les hyperparamètres sur ce test."
        )
    return "\n".join(lines)


def format_report(sections: list[tuple[str, dict]]) -> str:
    """Trois étapes en résumé, puis les pistes du dernier cas seulement."""
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
