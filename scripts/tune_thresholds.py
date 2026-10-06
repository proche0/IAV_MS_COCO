"""Calibre les seuils de decision d'un checkpoint sur l'ensemble de validation.

La metrique du serveur ponderant chaque classe par l'inverse de sa frequence,
le seuil 0,5 est tres sous-optimal : il faut accepter beaucoup de faux positifs
sur les classes rares pour gagner du rappel la ou le score se joue.

Le script mesure aussi le sur-apprentissage de la calibration : les seuils sont
regles sur une moitie de la validation puis evalues sur l'autre moitie.

Usage :
    python3 scripts/tune_thresholds.py --checkpoint outputs/head_resnet18_linear_bce.pth
    python3 scripts/tune_thresholds.py --checkpoint ... --save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from coco_mlc.config import CLASSES, DEFAULTS, PATHS, SEED
from coco_mlc.data import (
    COCOTrainImageDataset,
    TransformSubset,
    build_transforms,
    get_split,
    load_label_matrix,
)
from coco_mlc.engine import collect_outputs, describe_device, load_checkpoint, pick_device
from coco_mlc.features import load_train_cache
from coco_mlc.inference import build_head_module, load_model, score_features
from coco_mlc.metrics import all_metrics, metric_class_weights
from coco_mlc.thresholds import tune_global_threshold, tune_per_class_thresholds
from coco_mlc.utils import save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--device", default=None)
    p.add_argument("--save", action="store_true", help="ecrit les seuils dans le checkpoint")
    p.add_argument("--no-figure", action="store_true")
    return p.parse_args()


def validation_scores(checkpoint_path, checkpoint, args, device):
    """Probabilites et cibles de validation, via le cache si disponible."""
    kind = checkpoint.get("kind", "full_model")
    split = checkpoint.get("split", {"strategy": "stratified", "val_fraction": DEFAULTS["val_fraction"]})

    if kind == "feature_head":
        cache = load_train_cache(
            checkpoint["backbone"], checkpoint["image_size"], checkpoint["resize_mode"]
        )
        _, val_idx = get_split(
            cache["labels"], split["val_fraction"], checkpoint.get("seed", SEED),
            strategy=split["strategy"],
        )
        head = build_head_module(checkpoint, device)
        scores = score_features(head, cache["features"][val_idx])
        targets = torch.from_numpy(cache["labels"][val_idx].astype(np.float32))
        print(f"Source des scores : cache de features ({len(val_idx):,} images de validation)")
        return scores, targets

    net, _ = load_model(checkpoint_path, device)
    base = COCOTrainImageDataset()
    _, labels = load_label_matrix()
    _, val_idx = get_split(
        labels, split["val_fraction"], checkpoint.get("seed", SEED), strategy=split["strategy"]
    )
    transform = build_transforms(
        checkpoint["image_size"], train=False, resize_mode=checkpoint["resize_mode"]
    )
    loader = DataLoader(
        TransformSubset(base, val_idx, transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    print(f"Source des scores : passage complet du reseau ({len(val_idx):,} images)")
    logits, targets = collect_outputs(loader, net, device, desc="validation")
    return torch.sigmoid(logits), targets


def main():
    args = parse_args()
    set_seed(SEED)
    PATHS.mkdirs()
    device = pick_device(args.device)
    print(f"Device     : {describe_device(device)}")
    print(f"Checkpoint : {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    print(f"Modele     : {checkpoint.get('backbone')} / {checkpoint.get('kind', 'full_model')}")
    print()

    scores, targets = validation_scores(args.checkpoint, checkpoint, args, device)

    baseline = all_metrics(scores, targets, thresholds=0.5)
    global_th, global_f1, curve = tune_global_threshold(scores, targets)

    print()
    print("=" * 78)
    print("SEUIL UNIQUE")
    print("=" * 78)
    print(f"  seuil 0.50 (defaut)   F1={baseline['f1']:.4f}  "
          f"precision={baseline['precision']:.4f}  rappel={baseline['recall']:.4f}")
    print(f"  seuil {global_th:.2f} (optimal)  F1={global_f1:.4f}")
    print(f"  gain : {100 * (global_f1 - baseline['f1']) / max(baseline['f1'], 1e-9):+.1f} %")

    per_class_th, per_class_f1, _ = tune_per_class_thresholds(
        scores, targets, rounds=args.rounds, init_threshold=global_th
    )
    print()
    print("=" * 78)
    print("SEUILS PAR CLASSE")
    print("=" * 78)
    print(f"  F1 validation = {per_class_f1:.4f} "
          f"({100 * (per_class_f1 - global_f1) / max(global_f1, 1e-9):+.1f} % vs seuil unique)")
    print(f"  seuils : min {per_class_th.min():.2f}, median {per_class_th.median():.2f}, "
          f"max {per_class_th.max():.2f}")

    # Le reglage de 80 seuils sur la validation peut la sur-apprendre. On le
    # quantifie en calibrant sur une moitie et en evaluant sur l'autre.
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(scores), generator=g)
    half = len(scores) // 2
    calib, holdout = perm[:half], perm[half:]

    th_calib, f1_calib, _ = tune_per_class_thresholds(
        scores[calib], targets[calib], rounds=args.rounds, verbose=False
    )
    global_calib, _, _ = tune_global_threshold(scores[calib], targets[calib])
    f1_holdout_per_class = all_metrics(scores[holdout], targets[holdout], thresholds=th_calib)["f1"]
    f1_holdout_global = all_metrics(scores[holdout], targets[holdout], thresholds=global_calib)["f1"]

    print()
    print("=" * 78)
    print("CONTROLE DE SUR-APPRENTISSAGE DE LA CALIBRATION")
    print("=" * 78)
    print(f"  calibre sur une moitie de la validation  F1={f1_calib:.4f}")
    print(f"  evalue sur l'autre moitie :")
    print(f"    seuil unique    F1={f1_holdout_global:.4f}")
    print(f"    seuils/classe   F1={f1_holdout_per_class:.4f}")
    verdict = ("les seuils par classe generalisent"
               if f1_holdout_per_class > f1_holdout_global
               else "le seuil unique est plus robuste ici")
    print(f"  -> {verdict}")

    weights = metric_class_weights(targets.sum(dim=0)).numpy()
    _, per_class = all_metrics(scores, targets, thresholds=per_class_th, class_metrics=True)
    table = pd.DataFrame({
        "class_name": CLASSES,
        "threshold": per_class_th.numpy(),
        "support": [c["support"] for c in per_class],
        "metric_weight_pct": 100 * weights,
        "f1": [c["f1"] for c in per_class],
        "precision": [c["precision"] for c in per_class],
        "recall": [c["recall"] for c in per_class],
    }).sort_values("metric_weight_pct", ascending=False)
    stem = Path(args.checkpoint).stem
    table.to_csv(PATHS.outputs / f"thresholds_{stem}.csv", index=False)
    table.to_csv(PATHS.outputs / "thresholds_per_class.csv", index=False)

    print()
    print("=" * 78)
    print("LES 12 CLASSES QUI PESENT LE PLUS DANS LE SCORE")
    print("=" * 78)
    print(table.head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.save:
        checkpoint["thresholds"] = per_class_th
        checkpoint["threshold_global"] = global_th
        checkpoint["val_f1_calibrated"] = per_class_f1
        torch.save(checkpoint, args.checkpoint)
        print(f"\nSeuils ecrits dans {args.checkpoint}")

    save_json(PATHS.outputs / "threshold_curve.json", curve)
    if not args.no_figure:
        write_figure(curve, per_class_th.numpy(), weights, baseline["f1"], global_f1, per_class_f1)
        print(f"Figure : {PATHS.outputs / 'figures' / 'threshold_tuning.png'}")
    print(f"CSV    : {PATHS.outputs / 'thresholds_per_class.csv'}")


def write_figure(curve, per_class_th, weights, f1_default, f1_global, f1_per_class):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = PATHS.outputs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(curve)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(df.threshold, df.f1, label="F1 (serveur)", color="crimson")
    axes[0].plot(df.threshold, df.precision, label="Precision", alpha=0.7)
    axes[0].plot(df.threshold, df.recall, label="Rappel", alpha=0.7)
    axes[0].axvline(0.5, ls=":", color="gray", label="seuil 0.5")
    axes[0].set_xlabel("Seuil unique")
    axes[0].set_title("Sensibilite au seuil")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(100 * weights, per_class_th, s=20, color="darkorange")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Poids de la classe dans le score en % (log)")
    axes[1].set_ylabel("Seuil calibre")
    axes[1].set_title("Les classes qui pesent recoivent un seuil plus bas")
    axes[1].grid(alpha=0.3)

    labels = ["seuil 0.5", f"seuil unique\noptimal", "seuils\npar classe"]
    axes[2].bar(labels, [f1_default, f1_global, f1_per_class],
                color=["gray", "steelblue", "seagreen"])
    for i, v in enumerate([f1_default, f1_global, f1_per_class]):
        axes[2].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[2].set_ylabel("F1 validation (metrique serveur)")
    axes[2].set_title("Apport de la calibration")

    fig.tight_layout()
    fig.savefig(figures / "threshold_tuning.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
