"""Produit le fichier JSON de soumission pour le leaderboard.

Format impose par le sujet : un dictionnaire associant le nom de chaque image de
test, sans extension, a la liste des indices de classes predits.

    {"000000000139": [56, 60, 62], "000000000285": [21], ...}

Le script verifie le fichier produit avant de le declarer pret : nombre
d'entrees egal au nombre d'images de test, identifiants conformes, indices dans
[0, 79], aucune liste vide.

Usage :
    python3 scripts/predict.py --checkpoint outputs/head_resnet18_linear_bce.pth
    python3 scripts/predict.py --checkpoint ... --threshold 0.3 --output outputs/pred.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from coco_mlc.config import CLASSES, DEFAULTS, NUM_CLASSES, PATHS
from coco_mlc.data import COCOTestImageDataset, build_transforms
from coco_mlc.engine import describe_device, load_checkpoint, pick_device
from coco_mlc.features import cache_path, load_cache
from coco_mlc.inference import build_head_module, checkpoint_thresholds, load_model, score_features
from coco_mlc.utils import human_time

ID_PATTERN = re.compile(r"^\d{12}$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None, help="chemin du JSON (defaut : outputs/predictions_<nom>.json)")
    p.add_argument("--threshold", type=float, default=None,
                   help="force un seuil unique au lieu des seuils du checkpoint")
    p.add_argument("--min-labels", type=int, default=1,
                   help="nombre minimal de classes predites par image")
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--device", default=None)
    p.add_argument("--no-cache", action="store_true", help="ignore le cache de features de test")
    p.add_argument("--submit-copy", action="store_true",
                   help="copie le JSON dans submissions/ pour le garder sous suivi git")
    return p.parse_args()


def test_scores(args, checkpoint, device):
    """Probabilites par classe sur l'ensemble de test, et identifiants d'images."""
    kind = checkpoint.get("kind", "full_model")

    if kind == "feature_head" and not args.no_cache:
        path = cache_path(
            checkpoint["backbone"], checkpoint["image_size"], checkpoint["resize_mode"], "test"
        )
        if path.exists():
            cache = load_cache(path)
            head = build_head_module(checkpoint, device)
            print(f"Source : cache de features de test ({len(cache['ids']):,} images)")
            return score_features(head, cache["features"]), cache["ids"]
        print(f"Cache de test absent ({path.name}), passage complet du reseau.")

    net, _ = load_model(args.checkpoint, device)
    transform = build_transforms(
        checkpoint.get("image_size", DEFAULTS["image_size"]),
        train=False,
        resize_mode=checkpoint.get("resize_mode", DEFAULTS["resize_mode"]),
    )
    dataset = COCOTestImageDataset(transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"Source : passage complet du reseau sur {len(dataset):,} images de test")

    # Le loader ne melange pas les images, donc l'ordre des identifiants du
    # dataset correspond a celui des logits accumules.
    logits = []
    net.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="test"):
            logits.append(net(images.to(device, non_blocking=True)).float().cpu())
    return torch.sigmoid(torch.cat(logits)), dataset.ids


def build_predictions(scores: torch.Tensor, ids, thresholds: torch.Tensor, min_labels: int):
    decisions = (scores > thresholds).numpy()
    scores_np = scores.numpy()

    predictions = {}
    n_rescued = 0
    for i, image_id in enumerate(ids):
        indices = np.flatnonzero(decisions[i])
        if len(indices) < min_labels:
            # Une liste vide est un faux negatif garanti : on retient au moins
            # les classes les plus probables.
            indices = np.argsort(-scores_np[i])[:min_labels]
            n_rescued += 1
        predictions[str(image_id)] = sorted(int(c) for c in indices)
    return predictions, n_rescued


def validate(predictions: dict, expected_n: int) -> list[str]:
    problems = []
    if len(predictions) != expected_n:
        problems.append(f"{len(predictions)} entrees au lieu de {expected_n}")
    bad_ids = [k for k in predictions if not ID_PATTERN.match(k)]
    if bad_ids:
        problems.append(f"identifiants non conformes (ex. {bad_ids[:3]})")
    empty = [k for k, v in predictions.items() if not v]
    if empty:
        problems.append(f"{len(empty)} listes vides")
    bad_values = [
        k for k, v in predictions.items()
        if not all(isinstance(c, int) and 0 <= c < NUM_CLASSES for c in v)
    ]
    if bad_values:
        problems.append(f"indices hors de [0, {NUM_CLASSES - 1}] (ex. {bad_values[:3]})")
    return problems


def main():
    args = parse_args()
    PATHS.check()
    PATHS.mkdirs()
    device = pick_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    print(f"Device     : {describe_device(device)}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Modele     : {checkpoint.get('backbone')} / {checkpoint.get('kind', 'full_model')}"
          f"  (F1 validation {checkpoint.get('val_f1_calibrated', checkpoint.get('val_f1', float('nan'))):.4f})")

    if args.threshold is not None:
        thresholds = torch.full((NUM_CLASSES,), args.threshold)
        print(f"Seuils     : unique, force a {args.threshold}")
    else:
        thresholds = checkpoint_thresholds(checkpoint)
        unique = thresholds.unique()
        if len(unique) == 1:
            print(f"Seuils     : unique, {float(unique[0]):.2f} (depuis le checkpoint)")
        else:
            print(f"Seuils     : par classe, min {thresholds.min():.2f} / "
                  f"median {thresholds.median():.2f} / max {thresholds.max():.2f}")
    print()

    start = time.perf_counter()
    scores, ids = test_scores(args, checkpoint, device)
    predictions, n_rescued = build_predictions(scores, ids, thresholds, args.min_labels)
    elapsed = time.perf_counter() - start

    expected = len(COCOTestImageDataset())
    problems = validate(predictions, expected)

    counts = np.array([len(v) for v in predictions.values()])
    print()
    print("=" * 78)
    print("PREDICTIONS")
    print("=" * 78)
    print(f"  images traitees        : {len(predictions):,} (attendu {expected:,})")
    print(f"  classes par image      : moyenne {counts.mean():.2f}, mediane {int(np.median(counts))}, "
          f"min {counts.min()}, max {counts.max()}")
    print(f"  reference train        : 2.93 classes par image")
    print(f"  images sans prediction : {n_rescued:,} rattrapees par le garde-fou "
          f"({100 * n_rescued / len(predictions):.1f} %)")
    print(f"  duree                  : {human_time(elapsed)}")

    predicted = np.zeros(NUM_CLASSES, dtype=np.int64)
    for v in predictions.values():
        predicted[v] += 1
    never = [CLASSES[c] for c in range(NUM_CLASSES) if predicted[c] == 0]
    print(f"  classes jamais predites: {len(never)}"
          + (f" ({', '.join(never[:8])}{'...' if len(never) > 8 else ''})" if never else ""))

    if problems:
        print()
        print("  PROBLEMES DETECTES :")
        for item in problems:
            print(f"    - {item}")
        raise SystemExit("Le fichier n'est pas conforme, soumission annulee.")

    name = args.output or (PATHS.outputs / f"predictions_{Path(args.checkpoint).stem}.json")
    out_path = Path(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, indent=2))

    print()
    print(f"  format verifie : OK")
    print(f"  JSON : {out_path}  ({out_path.stat().st_size / 1e6:.1f} Mo)")
    first = next(iter(predictions.items()))
    print(f"  exemple : \"{first[0]}\": {first[1]}")

    if args.submit_copy:
        copy = PATHS.submissions / out_path.name
        shutil.copy(out_path, copy)
        print(f"  copie : {copy}")

    print()
    print("Soumission : https://www.creatis.insa-lyon.fr/kechichian/ms-coco-classif-leaderboard.html")
    print("ATTENTION : une soumission avec le meme nom de groupe ecrase la precedente.")


if __name__ == "__main__":
    main()
