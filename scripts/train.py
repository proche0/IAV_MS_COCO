"""Fine-tuning complet d'un reseau pre-entraine sur MS COCO multi-label.

Suit le squelette de programme decrit dans la partie 4 du sujet. A lancer en
local sur GPU : ``--amp`` active la precision mixte. Hors CUDA, le flag est
ignore avec un avertissement. ``--max-images`` sert a valider le pipeline sur
un sous-ensemble.

Les poids ImageNet (``Weights.DEFAULT``) sont telecharges automatiquement au
premier lancement, dans le cache torchvision.

Usage :
    python3 scripts/train.py --model resnet18 --epochs 8 --loss asl --amp
    python3 scripts/train.py --model resnet18 --max-images 2000 --epochs 1 --amp
    python3 scripts/train.py --model resnet50 --epochs 10 --amp --batch-size 96
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd
import torch
from torch.utils.data import DataLoader

from coco_mlc.config import CLASSES, DEFAULTS, NUM_CLASSES, PATHS, SEED
from coco_mlc.data import (
    COCOTrainImageDataset,
    TransformSubset,
    build_transforms,
    get_split,
    load_label_matrix,
)
from coco_mlc.engine import (
    describe_device,
    evaluate,
    pick_device,
    save_checkpoint,
    train_one_epoch,
    update_graphs,
)
from coco_mlc.losses import build_criterion, compute_pos_weight
from coco_mlc.metrics import format_metrics
from coco_mlc.models import MODEL_REGISTRY, build_model, count_parameters
from coco_mlc.thresholds import tune_global_threshold, tune_per_class_thresholds
from coco_mlc.utils import environment_summary, human_time, log_experiment, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet18", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--loss", default="bce", choices=["bce", "bce_pos_weight", "focal", "asl"])
    p.add_argument("--pos-weight-cap", type=float, default=20.0)
    p.add_argument("--image-size", type=int, default=DEFAULTS["image_size"])
    p.add_argument("--resize-mode", default=DEFAULTS["resize_mode"], choices=["pad", "squash", "crop"])
    p.add_argument("--augment", default="flip", choices=["none", "flip", "strong"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-prior-bias", action="store_true",
                   help="biais de sortie a zero au lieu de la log-cote des prevalences")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="n'entraine que la tete (equivalent de train_head.py, mais sans cache)")
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"])
    p.add_argument("--split", default="stratified", choices=["stratified", "random"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--max-images", type=int, default=None, help="restreint le dataset (test rapide)")
    p.add_argument("--eval-train", action="store_true",
                   help="evalue aussi sur le train (double le cout d'une epoque)")
    p.add_argument("--amp", action="store_true", help="precision mixte (GPU uniquement)")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--per-class-thresholds", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="")
    return p.parse_args()


def build_loaders(args):
    _, Y = load_label_matrix()
    if args.max_images:
        Y = Y[: args.max_images]
    train_idx, val_idx = get_split(
        Y, args.val_fraction, args.seed, strategy=args.split,
        cache_path=(PATHS.outputs /
                    f"split_{args.split}_{args.val_fraction:g}_{args.seed}_{len(Y)}.json"),
    )

    base = COCOTrainImageDataset(max_images=args.max_images)
    train_tf = build_transforms(args.image_size, train=True, resize_mode=args.resize_mode,
                                augment=args.augment)
    val_tf = build_transforms(args.image_size, train=False, resize_mode=args.resize_mode)

    pin = torch.cuda.is_available()
    common = {
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "pin_memory": pin,
    }
    train_loader = DataLoader(
        TransformSubset(base, train_idx, train_tf), batch_size=args.batch_size,
        shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        TransformSubset(base, val_idx, val_tf), batch_size=args.batch_size,
        shuffle=False, **common,
    )
    # Le train evalue sans augmentation, pour une comparaison honnete avec la
    # validation (sinon l'ecart train/validation melange deux effets).
    train_eval_loader = DataLoader(
        TransformSubset(base, train_idx, val_tf), batch_size=args.batch_size,
        shuffle=False, **common,
    )
    return train_loader, val_loader, train_eval_loader, Y[train_idx], len(train_idx), len(val_idx)


def main():
    args = parse_args()
    set_seed(args.seed)
    PATHS.check()
    PATHS.mkdirs()
    device = pick_device(args.device)
    if args.amp and device.type != "cuda":
        print("Avertissement : --amp ignore hors GPU.")
        args.amp = False

    name = args.tag or f"{args.model}_{args.loss}{'_frozen' if args.freeze_backbone else ''}"

    train_loader, val_loader, train_eval_loader, Y_train, n_train, n_val = build_loaders(args)
    net = build_model(
        args.model, NUM_CLASSES, pretrained=True,
        freeze_backbone=args.freeze_backbone, dropout=args.dropout,
        prior=None if args.no_prior_bias else Y_train.mean(axis=0),
    ).to(device)
    total_params, trainable_params = count_parameters(net)

    pos_weight = None
    if args.loss == "bce_pos_weight":
        pos_weight = compute_pos_weight(Y_train, cap=args.pos_weight_cap).to(device)
    criterion = build_criterion(args.loss, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * max(len(train_loader), 1)
        )
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    print("=" * 78)
    print(f"FINE-TUNING {args.model}")
    print("=" * 78)
    print(f"  environnement : {environment_summary()}")
    print(f"  device        : {describe_device(device)}")
    print(f"  donnees       : {n_train:,} train / {n_val:,} validation "
          f"({args.split}, seed {args.seed})")
    print(f"  pretraitement : {args.image_size}px, mode {args.resize_mode}, "
          f"augmentation {args.augment}")
    print(f"  parametres    : {total_params / 1e6:.1f}M dont {trainable_params / 1e6:.1f}M entrainables")
    print(f"  cout          : {args.loss}, optimiseur AdamW lr={args.lr}, "
          f"wd={args.weight_decay}, scheduler {args.scheduler}")
    print(f"  epoques       : {args.epochs}, batch {args.batch_size}, amp {args.amp}")
    print()

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(PATHS.runs / name)
        print(f"  tensorboard : tensorboard --logdir {PATHS.runs}")

    best = {"f1": -1.0, "epoch": None, "threshold": 0.5}
    checkpoint_path = PATHS.outputs / f"model_{name}.pth"
    history = []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss, _ = train_one_epoch(
            train_loader, net, criterion, optimizer, device,
            scheduler=scheduler, scaler=scaler, desc=f"epoque {epoch + 1}/{args.epochs}",
        )
        val_results, _, val_scores, val_targets = evaluate(
            val_loader, net, criterion, device, thresholds=0.5,
            desc="validation", return_scores=True, amp=args.amp,
        )
        best_th, f1_tuned, _ = tune_global_threshold(val_scores, val_targets)

        train_results = None
        if args.eval_train:
            train_results = evaluate(
                train_eval_loader, net, criterion, device, thresholds=best_th,
                desc="train (eval)", amp=args.amp,
            )

        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_f1_th0.5": val_results["f1"],
            "val_f1_tuned": f1_tuned,
            "val_threshold": best_th,
            "val_mAP": val_results["mAP"],
            "val_macro_f1": val_results["macro_f1"],
            "val_recall": val_results["recall"],
            "val_precision": val_results["precision"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(epoch_seconds, 1),
        }
        if train_results:
            row["train_f1_tuned"] = train_results["f1"]
            row["train_mAP"] = train_results["mAP"]
        history.append(row)

        marker = ""
        if f1_tuned > best["f1"]:
            best = {"f1": f1_tuned, "epoch": epoch + 1, "threshold": best_th}
            save_checkpoint(checkpoint_path, net, {
                "kind": "full_model",
                "backbone": args.model,
                "image_size": args.image_size,
                "resize_mode": args.resize_mode,
                "dropout": args.dropout,
                "loss": args.loss,
                "thresholds": torch.full((NUM_CLASSES,), best_th),
                "threshold_global": best_th,
                "best_epoch": epoch + 1,
                "val_f1": f1_tuned,
                "seed": args.seed,
                "split": {"strategy": args.split, "val_fraction": args.val_fraction},
            })
            marker = "  <- meilleur, sauvegarde"

        print(f"  epoque {epoch + 1:2d}/{args.epochs}  {human_time(epoch_seconds):>7s}  "
              f"train_loss={train_loss:.4f}  val_loss={val_results['loss']:.4f}  "
              f"F1@0.5={val_results['f1']:.4f}  F1*={f1_tuned:.4f} (th={best_th:.2f})  "
              f"mAP={val_results['mAP']:.4f}{marker}")

        if writer:
            update_graphs(
                writer, epoch,
                train_results or {"loss": train_loss},
                {**val_results, "f1": f1_tuned},
            )

        pd.DataFrame(history).to_csv(PATHS.outputs / f"history_{name}.csv", index=False)

    total_seconds = time.perf_counter() - start
    if writer:
        writer.close()

    print()
    print(f"Entrainement termine en {human_time(total_seconds)} "
          f"({human_time(total_seconds / max(args.epochs, 1))} par epoque)")
    print(f"Meilleure epoque : {best['epoch']} (F1 validation {best['f1']:.4f})")

    # Recharge le meilleur modele puis calibre les seuils par classe.
    from coco_mlc.engine import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    _, _, val_scores, val_targets = evaluate(
        val_loader, net, criterion, device, thresholds=0.5,
        desc="validation (final)", return_scores=True, amp=args.amp,
    )

    thresholds = torch.full((NUM_CLASSES,), best["threshold"])
    f1_final = best["f1"]
    if args.per_class_thresholds:
        print("\nCalibration des seuils par classe :")
        thresholds, f1_final, _ = tune_per_class_thresholds(
            val_scores, val_targets, init_threshold=best["threshold"]
        )

    final_metrics, per_class = evaluate(
        val_loader, net, criterion, device, thresholds=thresholds,
        class_metrics=True, desc="validation (calibre)", amp=args.amp,
    )
    checkpoint["thresholds"] = thresholds
    checkpoint["val_f1_calibrated"] = f1_final
    torch.save(checkpoint, checkpoint_path)

    save_json(PATHS.outputs / f"per_class_{name}.json",
              {CLASSES[i]: per_class[i] for i in range(NUM_CLASSES)})
    log_experiment(PATHS.outputs / "experiments.csv", {
        "backbone": args.model,
        "strategy": "frozen_backbone" if args.freeze_backbone else "fine_tuning",
        "head": "linear",
        "loss": args.loss,
        "image_size": args.image_size,
        "resize_mode": args.resize_mode,
        "augment": args.augment,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "per_class_th": args.per_class_thresholds,
        "val_f1": round(f1_final, 4),
        "val_precision": round(final_metrics["precision"], 4),
        "val_recall": round(final_metrics["recall"], 4),
        "val_mAP": round(final_metrics["mAP"], 4),
        "val_macro_f1": round(final_metrics["macro_f1"], 4),
        "val_micro_f1": round(final_metrics["micro_f1"], 4),
        "seconds": round(total_seconds, 1),
        "n_train": n_train,
        "device": describe_device(device),
    })

    print(f"\nValidation calibree : {format_metrics(final_metrics)}")
    print(f"Checkpoint          : {checkpoint_path}")
    print(f"Etape suivante      : python3 scripts/predict.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
