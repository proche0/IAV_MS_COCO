"""Mesure le debit reel des backbones sur cette machine.

Sert a decider ce qui est faisable localement : un passage forward sur les
70 000 images est acceptable, un entrainement complet avec retropropagation ne
l'est pas forcement. Les estimations sont extrapolees du debit mesure.

Usage :
    python3 scripts/benchmark_speed.py
    python3 scripts/benchmark_speed.py --models resnet18 resnet50 --batches 5
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd
import torch

from coco_mlc.config import DEFAULTS, NUM_CLASSES, PATHS, SEED
from coco_mlc.data import COCOTrainImageDataset, build_transforms
from coco_mlc.engine import describe_device, pick_device
from coco_mlc.models import MODEL_REGISTRY, build_model, count_parameters
from coco_mlc.utils import environment_summary, human_time, set_seed

N_TRAIN_IMAGES = 65_000
N_TEST_IMAGES = 4_952


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+",
                   default=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18", "resnet50"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=4, help="batches mesures (apres rodage)")
    p.add_argument("--image-size", type=int, default=DEFAULTS["image_size"])
    p.add_argument("--device", default=None)
    p.add_argument("--skip-backward", action="store_true")
    return p.parse_args()


def timed_forward(model, batch, n_batches):
    model.eval()
    with torch.no_grad():
        model(batch)  # rodage
        start = time.perf_counter()
        for _ in range(n_batches):
            model(batch)
        return (time.perf_counter() - start) / n_batches


def timed_train_step(model, batch, targets, n_batches):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()

    step()  # rodage
    start = time.perf_counter()
    for _ in range(n_batches):
        step()
    return (time.perf_counter() - start) / n_batches


def main():
    args = parse_args()
    set_seed(SEED)
    PATHS.check()
    PATHS.mkdirs()
    device = pick_device(args.device)

    print("Environnement :", environment_summary())
    print("Device        :", describe_device(device))
    print(f"Batch         : {args.batch_size} images de {args.image_size}x{args.image_size}")
    print()

    # Cout de lecture et de decodage des images, independant du modele : il
    # borne le debit atteignable quel que soit le backbone.
    dataset = COCOTrainImageDataset(transform=build_transforms(args.image_size))
    start = time.perf_counter()
    n_probe = 200
    for i in range(n_probe):
        dataset[i]
    io_per_image = (time.perf_counter() - start) / n_probe
    print(f"Lecture + decodage + transformation : {1 / io_per_image:.0f} images/s par worker "
          f"({1000 * io_per_image:.1f} ms/image)")
    print(f"  avec {DEFAULTS['num_workers']} workers : "
          f"~{DEFAULTS['num_workers'] / io_per_image:.0f} images/s en theorie")
    print()

    batch = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    targets = (torch.rand(args.batch_size, NUM_CLASSES, device=device) < 0.04).float()

    rows = []
    for name in args.models:
        info = MODEL_REGISTRY[name]
        model = build_model(name, pretrained=False).to(device)
        total, trainable = count_parameters(model)

        fwd = timed_forward(model, batch, args.batches)
        fwd_rate = args.batch_size / fwd

        if args.skip_backward:
            train_rate = float("nan")
        else:
            step = timed_train_step(model, batch, targets, args.batches)
            train_rate = args.batch_size / step

        del model

        cache_seconds = (N_TRAIN_IMAGES + N_TEST_IMAGES) / fwd_rate
        epoch_seconds = (0.8 * N_TRAIN_IMAGES) / train_rate if train_rate == train_rate else float("nan")

        rows.append({
            "model": name,
            "params_M": round(total / 1e6, 1),
            "gflops_doc": info.gflops,
            "imagenet_acc1": info.imagenet_acc1,
            "forward_img_per_s": round(fwd_rate, 1),
            "train_img_per_s": round(train_rate, 1),
            "cache_features_70k": human_time(cache_seconds),
            "one_epoch_52k": human_time(epoch_seconds) if epoch_seconds == epoch_seconds else "-",
        })
        print(f"{name:<22s} forward {fwd_rate:7.1f} img/s   "
              f"train {train_rate:7.1f} img/s   "
              f"cache 70k ~ {human_time(cache_seconds):>9s}   "
              f"1 epoque 52k ~ {human_time(epoch_seconds) if epoch_seconds == epoch_seconds else '-':>9s}")

    df = pd.DataFrame(rows)
    out = PATHS.outputs / "benchmark_speed.csv"
    df.to_csv(out, index=False)

    print()
    print("=" * 78)
    print("LECTURE")
    print("=" * 78)
    print("Le calcul du modele et la lecture des images se recouvrent grace aux workers :")
    print("le debit reel est proche du minimum des deux colonnes. Si le forward est plus")
    print("rapide que la lecture, c'est le decodage JPEG qui limite.")
    print()
    print("Un entrainement complet suppose plusieurs epoques et plusieurs modeles : si")
    print("'1 epoque' depasse la dizaine de minutes, passer par les features pre-calculees")
    print("(scripts/cache_features.py) pour l'etude comparative, et reserver le fine-tuning")
    print("au GPU.")
    print(f"\nCSV : {out}")


if __name__ == "__main__":
    main()
