"""Emet un notebook d'experimentation par architecture du registre.

Usage, depuis la racine du depot :

    python notebooks/build_experiment_notebooks.py

Les cellules de demarche sont identiques. Seuls le titre, la fiche modele et
les hyperparametres de depart changent.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from coco_mlc.models import MODEL_REGISTRY

OUT_DIR = REPO / "notebooks" / "experiments"

# Batch et pas d'apprentissage de depart, pour une RTX 4060 Ti.
# La tete seule supporte un pas plus grand ; le fine-tuning reprend les ordres
# de grandeur du README (ResNet, ConvNeXt, Swin) et descend pour les modeles lourds.
RUN_CONFIG = {
    "mobilenet_v3_small": {"batch_size": 128, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "shufflenet_v2_x1_0": {"batch_size": 128, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "mobilenet_v3_large": {"batch_size": 96, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "efficientnet_b0": {"batch_size": 64, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "efficientnet_b1": {"batch_size": 48, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "efficientnet_b3": {"batch_size": 32, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "efficientnet_b4": {"batch_size": 16, "lr_head": "1e-3", "lr_finetune": "5e-5"},
    "resnet18": {"batch_size": 128, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "vgg16": {"batch_size": 8, "lr_head": "1e-3", "lr_finetune": "1e-5"},
    "resnet50": {"batch_size": 96, "lr_head": "1e-3", "lr_finetune": "1e-4"},
    "convnext_tiny": {"batch_size": 64, "lr_head": "1e-3", "lr_finetune": "5e-5"},
    "swin_t": {"batch_size": 64, "lr_head": "1e-3", "lr_finetune": "5e-5"},
    "maxvit_t": {"batch_size": 8, "lr_head": "1e-3", "lr_finetune": "5e-5"},
    "efficientnet_v2_s": {"batch_size": 16, "lr_head": "1e-3", "lr_finetune": "5e-5"},
}

DISPLAY = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "shufflenet_v2_x1_0": "ShuffleNet V2 x1.0",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b1": "EfficientNet-B1",
    "efficientnet_b3": "EfficientNet-B3",
    "efficientnet_b4": "EfficientNet-B4",
    "resnet18": "ResNet18",
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-T",
    "maxvit_t": "MaxViT-T",
    "efficientnet_v2_s": "EfficientNetV2-S",
}


def _fr(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _lines(source: str) -> list[str]:
    text = source.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def _cell(kind: str, source: str) -> dict:
    cell = {
        "cell_type": kind,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": _lines(source),
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _config_source(name: str, cfg: dict) -> str:
    return f"""MODEL_NAME = {name!r}
FULL_TRAIN = False

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
IMAGE_SIZE = 224
RESIZE_MODE = "pad"
LOSS = "asl"

BATCH_SIZE = {cfg["batch_size"]}
LR_HEAD = {cfg["lr_head"]}
LR_FINETUNE = {cfg["lr_finetune"]}
WEIGHT_DECAY_BASELINE = 0.0
WEIGHT_DECAY_FINETUNE = 1e-4
DROPOUT_FINETUNE = 0.3

# FULL_TRAIN lance la config complete (5 epoques tete gelee, puis 8 de fine-tuning).
# Sinon : 512 images et 1 epoque, pour verifier le pipeline.
EPOCHS_BASELINE = 5 if FULL_TRAIN else 1
EPOCHS_FINETUNE = 8 if FULL_TRAIN else 1
MAX_IMAGES = None if FULL_TRAIN else 512
NUM_WORKERS = 0
"""


SETUP_CODE = """
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

cwd = Path.cwd().resolve()
if cwd.name == "experiments" and cwd.parent.name == "notebooks":
    REPO = cwd.parent.parent
elif cwd.name == "notebooks":
    REPO = cwd.parent
else:
    REPO = cwd
sys.path.insert(0, str(REPO / "src"))

from coco_mlc.config import CLASSES, PATHS
from coco_mlc.data import (
    COCOTrainImageDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TransformSubset,
    build_transforms,
    get_three_way_split,
    load_label_matrix,
)
from coco_mlc.diagnostics import (
    diagnose_errors,
    error_percent,
    plot_error_curves,
    plot_model_diagram,
    save_history,
)
from coco_mlc.engine import describe_device, evaluate, fit_stage, pick_device, save_checkpoint
from coco_mlc.losses import build_criterion
from coco_mlc.metrics import all_metrics, format_metrics
from coco_mlc.models import (
    build_model,
    count_parameters,
    load_compatible_weights,
)
from coco_mlc.thresholds import tune_per_class_thresholds
from coco_mlc.utils import set_seed

set_seed(SEED)
device = pick_device()
USE_AMP = device.type == "cuda"
OUT_DIR = PATHS.outputs / "notebooks" / MODEL_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(describe_device(device), "| amp" if USE_AMP else "| fp32")
print("Résultats :", OUT_DIR)
"""

SPLIT_CODE = """
PATHS.check()
ids, Y = load_label_matrix()
if MAX_IMAGES is not None:
    ids = ids[:MAX_IMAGES]
    Y = Y[:MAX_IMAGES]

cache_path = PATHS.outputs / f"split_three_stratified_0.7_0.15_0.15_{SEED}.json"
if MAX_IMAGES is not None:
    cache_path = cache_path.with_name(f"{cache_path.stem}_{len(Y)}{cache_path.suffix}")
train_idx, val_idx, test_idx = get_three_way_split(
    Y,
    val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION,
    seed=SEED,
    cache_path=cache_path,
)
train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)
assert len(train_set) + len(val_set) + len(test_set) == len(Y)

n = len(Y)
print(f"{n} images annotées")
for label, idx in ("train", train_idx), ("validation", val_idx), ("test local", test_idx):
    print(f"  {label:12s} {len(idx):6d}  ({100 * len(idx) / n:.1f} %)")

counts = pd.DataFrame({
    "classe": list(CLASSES),
    "train": Y[train_idx].sum(axis=0),
    "validation": Y[val_idx].sum(axis=0),
    "test": Y[test_idx].sum(axis=0),
})
rare = counts[counts["classe"].isin(["hair drier", "toaster", "person"])]
print()
print(rare.to_string(index=False))
print()
print("Le test officiel (images/test) n'a pas d'étiquettes : il sert à la soumission, pas à ce F1.")
"""

AUGMENT_CODE = """
eval_tf = build_transforms(IMAGE_SIZE, train=False, resize_mode=RESIZE_MODE)
train_tf = build_transforms(IMAGE_SIZE, train=True, resize_mode=RESIZE_MODE, augment="flip")
experiment_tf = build_transforms(
    IMAGE_SIZE, train=True, resize_mode=RESIZE_MODE, augment="experiment",
)

preview = COCOTrainImageDataset(max_images=MAX_IMAGES)
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

def show_tensor(ax, tensor, title):
    image = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

n_show = 4
fig, axes = plt.subplots(n_show, 3, figsize=(8, 2.3 * n_show))
for row in range(n_show):
    image, _labels = preview[row]
    show_tensor(axes[row, 0], eval_tf(image), "évaluation (sans aléa)")
    show_tensor(axes[row, 1], train_tf(image), "baseline : flip")
    show_tensor(axes[row, 2], experiment_tf(image), "optimisé : géométrie + couleur")
fig.suptitle("Pad en carré, puis augmentation. L'évaluation n'applique pas l'aléa.")
fig.tight_layout()
plt.show()
"""

LOADERS_CODE = """
base = COCOTrainImageDataset(max_images=MAX_IMAGES)

def make_loader(indices, transform, shuffle):
    return DataLoader(
        TransformSubset(base, indices, transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=shuffle and len(indices) >= 2 * BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=device.type == "cuda",
    )

train_loader = make_loader(train_idx, train_tf, shuffle=True)
train_eval_loader = make_loader(train_idx, eval_tf, shuffle=False)
val_loader = make_loader(val_idx, eval_tf, shuffle=False)
test_loader = make_loader(test_idx, eval_tf, shuffle=False)
print(
    f"batches train {len(train_loader)} | "
    f"éval train {len(train_eval_loader)} | "
    f"validation {len(val_loader)} | test {len(test_loader)}"
)
"""

BASELINE_MODEL_CODE = """
prior = Y[train_idx].mean(axis=0)
model_baseline = build_model(
    MODEL_NAME,
    pretrained=True,
    freeze_backbone=True,
    dropout=0.0,
    prior=prior,
).to(device)

total, trainable = count_parameters(model_baseline)
print(f"{total / 1e6:.1f} M paramètres, dont {trainable / 1e6:.3f} M entraînables (la tête)")
fig = plot_model_diagram(model_baseline, frozen=True)
plt.show()
summary(model_baseline, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), depth=3)
"""

BASELINE_TRAIN_CODE = """
criterion = build_criterion(LOSS)
optimizer = torch.optim.AdamW(
    [p for p in model_baseline.parameters() if p.requires_grad],
    lr=LR_HEAD,
    weight_decay=WEIGHT_DECAY_BASELINE,
)
baseline_steps = EPOCHS_BASELINE * max(len(train_loader), 1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(baseline_steps, 1))

history_baseline = fit_stage(
    model_baseline,
    train_loader,
    train_eval_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    EPOCHS_BASELINE,
    scheduler=scheduler,
    amp=USE_AMP,
    desc="baseline",
)
save_history(history_baseline, OUT_DIR / "history_baseline.csv")
save_checkpoint(OUT_DIR / "baseline.pth", model_baseline, {
    "kind": "notebook_baseline",
    "backbone": MODEL_NAME,
    "freeze_backbone": True,
    "dropout": 0.0,
    "loss": LOSS,
    "image_size": IMAGE_SIZE,
    "resize_mode": RESIZE_MODE,
    "seed": SEED,
})

fig = plot_error_curves(history_baseline, title=f"{MODEL_NAME} — baseline, tête seule")
plt.show()
last_baseline = history_baseline[-1]
print(
    "Dernière époque (seuil 0,5) — "
    f"précision train/val {last_baseline['train_precision']:.3f}/{last_baseline['val_precision']:.3f} | "
    f"rappel {last_baseline['train_recall']:.3f}/{last_baseline['val_recall']:.3f} | "
    f"F1 {last_baseline['train_f1']:.3f}/{last_baseline['val_f1']:.3f}"
)
diagnose_errors(last_baseline["train_error"], last_baseline["val_error"])
"""

FINETUNE_CODE = """
train_loader_ft = make_loader(train_idx, experiment_tf, shuffle=True)
model_ft = build_model(
    MODEL_NAME,
    pretrained=True,
    freeze_backbone=False,
    dropout=DROPOUT_FINETUNE,
    prior=prior,
).to(device)
copied = load_compatible_weights(model_ft, model_baseline)
print(f"Tenseurs repris de la baseline : {len(copied)}")

total_ft, trainable_ft = count_parameters(model_ft)
print(f"{total_ft / 1e6:.1f} M paramètres, dont {trainable_ft / 1e6:.1f} M entraînables")
fig = plot_model_diagram(model_ft, frozen=False)
plt.show()

optimizer_ft = torch.optim.AdamW(
    model_ft.parameters(),
    lr=LR_FINETUNE,
    weight_decay=WEIGHT_DECAY_FINETUNE,
)
finetune_steps = EPOCHS_FINETUNE * max(len(train_loader_ft), 1)
scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, T_max=max(finetune_steps, 1),
)
history_ft = fit_stage(
    model_ft,
    train_loader_ft,
    train_eval_loader,
    val_loader,
    criterion,
    optimizer_ft,
    device,
    EPOCHS_FINETUNE,
    scheduler=scheduler_ft,
    amp=USE_AMP,
    desc="fine-tuning",
)
save_history(history_ft, OUT_DIR / "history_finetune.csv")
save_checkpoint(OUT_DIR / "finetune.pth", model_ft, {
    "kind": "notebook_finetune",
    "backbone": MODEL_NAME,
    "freeze_backbone": False,
    "dropout": DROPOUT_FINETUNE,
    "loss": LOSS,
    "image_size": IMAGE_SIZE,
    "resize_mode": RESIZE_MODE,
    "seed": SEED,
})

fig = plot_error_curves(history_ft, title=f"{MODEL_NAME} — fine-tuning")
plt.show()
last_ft = history_ft[-1]
print(
    "Dernière époque (seuil 0,5) — "
    f"précision train/val {last_ft['train_precision']:.3f}/{last_ft['val_precision']:.3f} | "
    f"rappel {last_ft['train_recall']:.3f}/{last_ft['val_recall']:.3f} | "
    f"F1 {last_ft['train_f1']:.3f}/{last_ft['val_f1']:.3f}"
)
diagnose_errors(last_ft["train_error"], last_ft["val_error"])
"""

TEST_CODE = """
def collect(loader, net, desc):
    results, _per_class, scores, targets = evaluate(
        loader, net, criterion, device,
        thresholds=0.5, return_scores=True, amp=USE_AMP, desc=desc,
    )
    return results, scores, targets

train_05, train_scores, train_targets = collect(train_eval_loader, model_ft, "train")
val_05, val_scores, val_targets = collect(val_loader, model_ft, "validation")
test_05, test_scores, test_targets = collect(test_loader, model_ft, "test local")

thresholds, _f1_val, _hist = tune_per_class_thresholds(val_scores, val_targets)

def at(scores, targets, th):
    return all_metrics(scores, targets, thresholds=th)

train_cal = at(train_scores, train_targets, thresholds)
val_cal = at(val_scores, val_targets, thresholds)
test_cal = at(test_scores, test_targets, thresholds)

rows = []
for split, raw, calibrated in (
    ("train", train_05, train_cal),
    ("validation", val_05, val_cal),
    ("test local", test_05, test_cal),
):
    rows.append({
        "ensemble": split,
        "précision @0,5": raw["precision"],
        "rappel @0,5": raw["recall"],
        "F1 @0,5": raw["f1"],
        "erreur @0,5 (%)": error_percent(raw["f1"]),
        "précision calibrée": calibrated["precision"],
        "rappel calibré": calibrated["recall"],
        "F1 calibré": calibrated["f1"],
        "erreur calibrée (%)": error_percent(calibrated["f1"]),
    })
report = pd.DataFrame(rows).set_index("ensemble")
print(report.round(4).to_string())
print()
print("Validation calibrée :", format_metrics(val_cal))
print("Test local calibré  :", format_metrics(test_cal))
report.to_csv(OUT_DIR / "generalization.csv")
"""

DIAG_CODE = """
print("Courbes, seuil 0,5, dernière époque de fine-tuning")
diagnose_errors(
    last_ft["train_error"],
    last_ft["val_error"],
    test_error=error_percent(test_05["f1"]),
)
print()
print("Point de fonctionnement : seuils calibrés sur la validation seulement")
diagnose_errors(
    error_percent(train_cal["f1"]),
    error_percent(val_cal["f1"]),
    test_error=error_percent(test_cal["f1"]),
)
"""


def _intro(name: str) -> str:
    info = MODEL_REGISTRY[name]
    title = DISPLAY[name]
    cfg = RUN_CONFIG[name]
    note = f"Note : {info.note}." if info.note else ""
    return f"""# {title} — expérience de classification multi-label

Notebook isolé pour **{title}** (`{name}`). La démarche est la même pour chaque
architecture : séparer les données, geler l'extracteur, augmenter, mesurer le
F1 du serveur, puis lire les courbes d'erreur avant de toucher au test.

| | |
| --- | --- |
| Paramètres | {_fr(info.params_m, 1)} M |
| Coût | {_fr(info.gflops, 2)} GFLOPS à 224 px |
| ImageNet top-1 | {_fr(info.imagenet_acc1, 1)} % |
| Batch de départ | {cfg["batch_size"]} |
| Pas, tête seule | {cfg["lr_head"]} |
| Pas, fine-tuning | {cfg["lr_finetune"]} |

{note.strip()}

Poids torchvision : `{info.weights}.DEFAULT`. Le réseau renvoie des logits ;
la sigmoïde n'est appliquée qu'au moment des métriques.

1. Découpage train / validation / test local
2. Augmentation géométrique et photométrique
3. Baseline : extracteur gelé, tête seule
4. Étape optimisée : augmentation plus forte, dropout, weight decay, dégel
5. Test local, une seule fois
6. Diagnostic biais / variance
"""


SHARED_MARKDOWN = {
    "config": """## Configuration

`FULL_TRAIN = False` limite l'exécution à 512 images et une époque, pour
vérifier que le notebook s'enchaîne. Le passer à `True` lance l'expérience
complète : 5 époques de tête gelée, puis 8 époques de fine-tuning, sur les
65 000 images annotées.

Le coût Asymmetric Loss (`asl`) est celui retenu pour le déséquilibre des
80 classes. L'erreur tracée plus bas vaut **100 × (1 − F1 serveur)**.
""",
    "setup": """## Environnement

Le notebook ajoute `src/` au chemin, fixe la graine, et choisit le GPU s'il
est visible. La précision mixte n'est activée que sur CUDA.
""",
    "split": """## Préparation et séparation des données

Les 65 000 images annotées sont coupées en trois ensembles **stratifiés** et
disjoints (graine 42) :

- **train (70 %)** — apprentissage des poids ;
- **validation (15 %)** — choix des hyperparamètres, arrêt, seuils ;
- **test local (15 %)** — généralisation, consulté une seule fois en fin de notebook.

Le cache est `outputs/split_three_stratified_0.7_0.15_0.15_42.json`. Il ne
remplace pas le découpage 80/20 des scripts. Un essai avec `MAX_IMAGES` écrit
un fichier à part, pour ne pas écraser le split complet.

Les classes rares (`hair drier`, `toaster`) dominent le F1 du serveur : le
tableau vérifie qu'elles sont présentes dans les trois ensembles.
""",
    "augment": """## Augmentation de données

La géométrie reste le mode `pad` : un recadrage central supprimerait des
objets dont l'étiquette resterait positive. Deux paliers ensuite :

- **baseline** — retournement horizontal ;
- **étape optimisée** — rotation, translation, échelle (`RandomAffine`) et
  jitter photométrique (`ColorJitter`).

Les loaders de validation, de test, et d'évaluation du train n'appliquent
aucune transformation aléatoire. L'écart train/validation mesure alors la
généralisation, pas le bruit de l'augmentation.
""",
    "loaders": """## Chargeurs

Le test local est construit ici pour ne pas être redéfini plus tard, mais
aucune métrique ne le lit avant la section de généralisation.
""",
    "baseline_model": """## Baseline — apprentissage par transfert

L'extracteur ImageNet est gelé. Seule la tête linéaire, réinitialisée à
80 logits, est apprise. Le biais de sortie part de la log-cote des
prévalences du **train** (pas de la validation, pas du test).

Le schéma distingue le prétraitement, l'extracteur gelé et la tête
entraînable. Le tableau `torchinfo` détaille les blocs du réseau fine-tuné.
""",
    "baseline_train": """## Entraînement de la tête

AdamW, pas `LR_HEAD`, sans weight decay. Le scheduler cosinus est calé sur
le nombre de mini-batches de cette étape. À chaque époque on enregistre
précision, rappel et F1, sur le train **sans** augmentation et sur la
validation, au seuil fixe 0,5.
""",
    "finetune": """## Étape optimisée

Trois leviers, dans cette cellule, pour pouvoir les modifier ensemble ou
les commenter un par un :

- augmentation `experiment` à la place du simple flip ;
- dropout sur la tête et weight decay (norme L2) ;
- dégel de tout le réseau, avec un pas plus faible.

Les poids de la baseline sont repris, y compris la tête déjà adaptée aux
80 classes. Le test local n'entre toujours pas dans les décisions.
""",
    "test": """## Généralisation — test local, une seule fois

Les seuils par classe sont calibrés **uniquement** sur la validation, puis
appliqués au train, à la validation et au test. Le test ne sert pas à choisir
le seuil. Les scores du réseau ne sont calculés qu'une fois par ensemble.
""",
    "guide": """## Diagnostic

L'erreur est **100 × (1 − F1 serveur)**. On compare d'abord l'entraînement et
la validation. Le test n'intervient qu'ensuite.

- **Sous-apprentissage (biais élevé).** Erreurs hautes et proches, par exemple
  10 % au train et 12 % en validation. Pistes : réseau plus volumineux,
  entraîner plus longtemps, changer d'optimiseur, recherche d'hyperparamètres.
- **Surapprentissage (variance élevée).** Écart fort, par exemple 1 % au train
  et 10 % en validation. Pistes : agrandir et diversifier le train, renforcer
  l'augmentation, régulariser (L2, dropout), recherche d'hyperparamètres.
- **Les deux.** Erreurs hautes et écart fort, par exemple 10 % et 20 %.
  Appliquer les deux listes d'actions.
- **Modèle idéal.** Erreurs très basses et proches, par exemple 0,5 % et 1 %.
- **Mauvaise généralisation au test.** Le test local est nettement pire que la
  validation. Constituer une validation plus grande et plus diversifiée : la
  validation a été surajustée. Ne pas retoucher les hyperparamètres sur ce test.

Un écart qui ne tombe dans aucun de ces régimes est signalé comme cas
intermédiaire. Le premier diagnostic reprend la dernière époque des courbes
(seuil 0,5). Le second reprend le point de fonctionnement après calibration
des seuils sur la validation.
""",
}


def build_notebook(name: str) -> dict:
    cfg = RUN_CONFIG[name]
    cells = [
        _cell("markdown", _intro(name)),
        _cell("markdown", SHARED_MARKDOWN["config"]),
        _cell("code", _config_source(name, cfg)),
        _cell("markdown", SHARED_MARKDOWN["setup"]),
        _cell("code", SETUP_CODE),
        _cell("markdown", SHARED_MARKDOWN["split"]),
        _cell("code", SPLIT_CODE),
        _cell("markdown", SHARED_MARKDOWN["augment"]),
        _cell("code", AUGMENT_CODE),
        _cell("markdown", SHARED_MARKDOWN["loaders"]),
        _cell("code", LOADERS_CODE),
        _cell("markdown", SHARED_MARKDOWN["baseline_model"]),
        _cell("code", BASELINE_MODEL_CODE),
        _cell("markdown", SHARED_MARKDOWN["baseline_train"]),
        _cell("code", BASELINE_TRAIN_CODE),
        _cell("markdown", SHARED_MARKDOWN["finetune"]),
        _cell("code", FINETUNE_CODE),
        _cell("markdown", SHARED_MARKDOWN["test"]),
        _cell("code", TEST_CODE),
        _cell("markdown", SHARED_MARKDOWN["guide"]),
        _cell("code", DIAG_CODE),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (IAV MS COCO)",
                "language": "python",
                "name": "iav-ms-coco",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "cells": cells,
    }


def main() -> None:
    missing = set(MODEL_REGISTRY) - set(RUN_CONFIG)
    extra = set(RUN_CONFIG) - set(MODEL_REGISTRY)
    if missing or extra:
        raise SystemExit(f"configs incohérentes : manquantes {sorted(missing)}, en trop {sorted(extra)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_REGISTRY:
        path = OUT_DIR / f"exp_{name}.ipynb"
        path.write_text(json.dumps(build_notebook(name), ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
