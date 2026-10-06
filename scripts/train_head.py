"""Entraine une tete de classification sur des features pre-calculees.

Strategie de transfer learning par extraction de features : le backbone
pre-entraine est gele, seule la tete est apprise. Comme les features sont
deja calculees (``cache_features.py``), chaque experience prend quelques
secondes, ce qui rend l'etude comparative realisable sans GPU.

Usage :
    python3 scripts/train_head.py --model resnet18
    python3 scripts/train_head.py --model resnet18 --loss asl --head mlp
    python3 scripts/train_head.py --model resnet18 --sweep        # etude comparative
"""

from __future__ import annotations

import argparse
import itertools
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from coco_mlc.config import CLASSES, DEFAULTS, NUM_CLASSES, PATHS, SEED
from coco_mlc.data import TensorBatchLoader, get_split
from coco_mlc.engine import evaluate, pick_device, save_checkpoint, train_one_epoch
from coco_mlc.features import available_caches, load_train_cache, standardize
from coco_mlc.losses import build_criterion, compute_pos_weight
from coco_mlc.metrics import format_metrics
from coco_mlc.models import build_head
from coco_mlc.thresholds import tune_global_threshold, tune_per_class_thresholds
from coco_mlc.utils import human_time, log_experiment, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet18", help="backbone dont le cache sera utilise")
    p.add_argument("--image-size", type=int, default=DEFAULTS["image_size"])
    p.add_argument("--resize-mode", default=DEFAULTS["resize_mode"])
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--loss", default="bce", choices=["bce", "bce_pos_weight", "focal", "asl"])
    p.add_argument("--pos-weight-cap", type=float, default=20.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=1024, help="taille cachee de la tete MLP")
    p.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"])
    p.add_argument("--split", default="stratified", choices=["stratified", "random"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--no-prior-bias", action="store_true",
                   help="biais de sortie a zero au lieu de la log-cote des prevalences")
    p.add_argument("--per-class-thresholds", action="store_true",
                   help="calibre un seuil par classe a la fin (sinon seuil global)")
    p.add_argument("--sweep", action="store_true", help="lance l'etude comparative predefinie")
    p.add_argument("--tag", default="", help="suffixe pour nommer le checkpoint")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def load_data(args):
    """Charge le cache de features et applique le decoupage train/validation."""
    try:
        cache = load_train_cache(args.model, args.image_size, args.resize_mode)
    except FileNotFoundError as exc:
        caches = available_caches()
        hint = "\n".join(f"  - {c.name}" for c in caches) or "  (aucun)"
        raise SystemExit(f"{exc}\n\nCaches disponibles :\n{hint}")

    X = cache["features"]
    Y = cache["labels"]
    train_idx, val_idx = get_split(Y, args.val_fraction, args.seed, strategy=args.split)

    X_train, X_val = X[train_idx], X[val_idx]
    if args.no_standardize:
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        mean = std = None
    else:
        X_train, X_val, mean, std = standardize(X_train, X_val)

    return {
        "X_train": torch.from_numpy(X_train),
        "Y_train": torch.from_numpy(Y[train_idx].astype(np.float32)),
        "X_val": torch.from_numpy(X_val),
        "Y_val": torch.from_numpy(Y[val_idx].astype(np.float32)),
        "Y_train_raw": Y[train_idx],
        "mean": mean,
        "std": std,
        "feature_dim": X.shape[1],
    }


def run_one(args, data, verbose=True):
    """Entraine une tete et retourne les resultats de validation."""
    set_seed(args.seed)
    device = pick_device("cpu")  # les features tiennent en memoire, le CPU suffit

    head_kwargs = {"dropout": args.dropout}
    if args.head == "mlp":
        head_kwargs["hidden"] = args.hidden
    prior = None if args.no_prior_bias else data["Y_train_raw"].mean(axis=0)
    net = build_head(
        args.head, data["feature_dim"], NUM_CLASSES, prior=prior, **head_kwargs
    ).to(device)

    pos_weight = None
    if args.loss == "bce_pos_weight":
        pos_weight = compute_pos_weight(data["Y_train_raw"], cap=args.pos_weight_cap).to(device)
    criterion = build_criterion(args.loss, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = TensorBatchLoader(
        data["X_train"], data["Y_train"], args.batch_size, shuffle=True, generator=generator
    )
    val_loader = TensorBatchLoader(data["X_val"], data["Y_val"], 4096)

    history = []
    best = {"f1": -1.0}
    start = time.perf_counter()

    for epoch in range(args.epochs):
        train_loss, _ = train_one_epoch(
            train_loader, net, criterion, optimizer, device, progress=False
        )
        val_results, _, val_scores, val_targets = evaluate(
            val_loader, net, criterion, device, thresholds=0.5,
            class_metrics=False, progress=False, return_scores=True,
        )
        # Le seuil 0,5 n'est presque jamais optimal pour cette metrique : on
        # suit aussi le score au meilleur seuil global, qui est la grandeur
        # reellement comparable entre configurations.
        best_th, f1_tuned, _ = tune_global_threshold(val_scores, val_targets)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_f1_th0.5": val_results["f1"],
            "val_f1_tuned": f1_tuned,
            "val_threshold": best_th,
            "val_mAP": val_results["mAP"],
            "val_macro_f1": val_results["macro_f1"],
        }
        history.append(row)

        if verbose:
            print(f"  epoque {epoch + 1:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_results['loss']:.4f}  "
                  f"F1@0.5={val_results['f1']:.4f}  F1*={f1_tuned:.4f} (th={best_th:.2f})  "
                  f"mAP={val_results['mAP']:.4f}")

        if f1_tuned > best["f1"]:
            best = {
                "f1": f1_tuned,
                "epoch": epoch + 1,
                "threshold": best_th,
                "state_dict": {k: v.clone() for k, v in net.state_dict().items()},
                "metrics": dict(val_results),
            }

    elapsed = time.perf_counter() - start
    net.load_state_dict(best["state_dict"])

    # Calibration finale sur la meilleure epoque.
    _, _, val_scores, val_targets = evaluate(
        val_loader, net, criterion, device, thresholds=0.5, progress=False, return_scores=True
    )
    thresholds = torch.full((NUM_CLASSES,), best["threshold"])
    f1_final = best["f1"]
    if args.per_class_thresholds:
        thresholds, f1_final, _ = tune_per_class_thresholds(
            val_scores, val_targets, init_threshold=best["threshold"], verbose=verbose
        )

    final_metrics, per_class = evaluate(
        val_loader, net, criterion, device, thresholds=thresholds,
        class_metrics=True, progress=False,
    )

    return {
        "net": net,
        "history": pd.DataFrame(history),
        "best_epoch": best["epoch"],
        "thresholds": thresholds,
        "f1": f1_final,
        "metrics": final_metrics,
        "per_class": per_class,
        "seconds": elapsed,
        "pos_weight": pos_weight,
    }


def describe(args, result) -> dict:
    m = result["metrics"]
    return {
        "backbone": args.model,
        "strategy": "feature_extraction",
        "head": args.head,
        "loss": args.loss,
        "image_size": args.image_size,
        "resize_mode": args.resize_mode,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "prior_bias": not args.no_prior_bias,
        "epochs": args.epochs,
        "best_epoch": result["best_epoch"],
        "per_class_th": args.per_class_thresholds,
        "val_f1": round(result["f1"], 4),
        "val_precision": round(m["precision"], 4),
        "val_recall": round(m["recall"], 4),
        "val_mAP": round(m["mAP"], 4),
        "val_macro_f1": round(m["macro_f1"], 4),
        "val_micro_f1": round(m["micro_f1"], 4),
        "seconds": round(result["seconds"], 1),
    }


def save_result(args, data, result, name: str):
    PATHS.mkdirs()
    checkpoint = PATHS.outputs / f"head_{name}.pth"
    save_checkpoint(checkpoint, result["net"], {
        "kind": "feature_head",
        "backbone": args.model,
        "head": args.head,
        "head_kwargs": {"dropout": args.dropout,
                        **({"hidden": args.hidden} if args.head == "mlp" else {})},
        "feature_dim": data["feature_dim"],
        "image_size": args.image_size,
        "resize_mode": args.resize_mode,
        "standardize_mean": data["mean"],
        "standardize_std": data["std"],
        "thresholds": result["thresholds"],
        "loss": args.loss,
        "best_epoch": result["best_epoch"],
        "val_f1": result["f1"],
        "seed": args.seed,
        "split": {"strategy": args.split, "val_fraction": args.val_fraction},
    })
    result["history"].to_csv(PATHS.outputs / f"history_{name}.csv", index=False)
    save_json(PATHS.outputs / f"per_class_{name}.json", {
        CLASSES[i]: result["per_class"][i] for i in range(NUM_CLASSES)
    })
    log_experiment(PATHS.outputs / "experiments.csv", describe(args, result))
    return checkpoint


SWEEP = [
    # (head, loss, lr, dropout) - une seule variable change a la fois par
    # rapport a la baseline (linear, bce), pour que les ecarts soient lisibles.
    ("linear", "bce", 1e-3, 0.0),
    ("linear", "bce_pos_weight", 1e-3, 0.0),
    ("linear", "focal", 1e-3, 0.0),
    ("linear", "asl", 1e-3, 0.0),
    ("mlp", "bce", 1e-3, 0.3),
    ("mlp", "bce_pos_weight", 1e-3, 0.3),
    ("mlp", "asl", 1e-3, 0.3),
]


def main():
    args = parse_args()
    data = load_data(args)
    print(f"Backbone      : {args.model} ({args.image_size}px, mode {args.resize_mode})")
    print(f"Features      : dimension {data['feature_dim']}, "
          f"train {len(data['X_train']):,} / validation {len(data['X_val']):,}")
    print(f"Decoupage     : {args.split}, {args.val_fraction:.0%} en validation, seed {args.seed}")
    print()

    if not args.sweep:
        name = args.tag or f"{args.model}_{args.head}_{args.loss}"
        print(f"Configuration : tete {args.head}, cout {args.loss}, lr {args.lr}")
        result = run_one(args, data, verbose=not args.quiet)
        checkpoint = save_result(args, data, result, name)
        print()
        print(f"Meilleure epoque : {result['best_epoch']}  en {human_time(result['seconds'])}")
        print(f"Validation       : {format_metrics(result['metrics'])}")
        print(f"Checkpoint       : {checkpoint}")
        return

    rows = []
    for head, loss, lr, dropout in SWEEP:
        args.head, args.loss, args.lr, args.dropout = head, loss, lr, dropout
        label = f"{head}+{loss}"
        print(f"--- {label} " + "-" * (60 - len(label)))
        result = run_one(args, data, verbose=False)
        save_result(args, data, result, f"{args.model}_{head}_{loss}")
        row = describe(args, result)
        rows.append(row)
        print(f"    F1={row['val_f1']:.4f}  precision={row['val_precision']:.4f}  "
              f"rappel={row['val_recall']:.4f}  mAP={row['val_mAP']:.4f}  "
              f"epoque {row['best_epoch']}  {human_time(result['seconds'])}")

    table = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    out = PATHS.outputs / f"study_{args.model}.csv"
    table.to_csv(out, index=False)
    print()
    print("=" * 78)
    print(f"ETUDE COMPARATIVE - {args.model}")
    print("=" * 78)
    print(table[["head", "loss", "val_f1", "val_precision", "val_recall",
                 "val_mAP", "val_macro_f1", "best_epoch", "seconds"]].to_string(index=False))
    print(f"\nCSV : {out}")


if __name__ == "__main__":
    main()
