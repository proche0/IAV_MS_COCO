"""Pre-calcule les features d'un backbone gele pour tout le dataset.

Un seul passage forward sur les 65 000 images d'entrainement et les 4 952 images
de test. Les features sont ensuite reutilisees par ``train_head.py`` pour
entrainer autant de tetes que necessaire en quelques secondes chacune.

Aucune augmentation de donnees n'est appliquee : les features doivent etre
deterministes pour pouvoir etre mises en cache.

Usage :
    python3 scripts/cache_features.py --model resnet18
    python3 scripts/cache_features.py --model mobilenet_v3_large --subsets train test
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from coco_mlc.config import DEFAULTS, PATHS, SEED
from coco_mlc.data import COCOTestImageDataset, COCOTrainImageDataset, build_transforms
from coco_mlc.engine import describe_device, pick_device
from coco_mlc.features import cache_path, extract_features, save_cache
from coco_mlc.models import MODEL_REGISTRY, build_feature_extractor
from coco_mlc.utils import human_time, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet18", choices=sorted(MODEL_REGISTRY))
    p.add_argument("--image-size", type=int, default=DEFAULTS["image_size"])
    p.add_argument("--resize-mode", default=DEFAULTS["resize_mode"], choices=["pad", "squash", "crop"])
    p.add_argument("--subsets", nargs="+", default=["train", "test"], choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--max-images", type=int, default=None, help="pour un test rapide du pipeline")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    PATHS.check()
    PATHS.mkdirs()
    device = pick_device(args.device)

    backbone, feature_dim = build_feature_extractor(args.model, pretrained=True)
    backbone = backbone.to(device)

    print(f"Backbone     : {args.model} (features de dimension {feature_dim})")
    print(f"Device       : {describe_device(device)}")
    print(f"Pretraitement: {args.image_size}px, mode '{args.resize_mode}', sans augmentation")
    print()

    # Les transformations d'evaluation sont volontairement deterministes.
    transform = build_transforms(args.image_size, train=False, resize_mode=args.resize_mode)

    for subset in args.subsets:
        out_path = cache_path(args.model, args.image_size, args.resize_mode, subset)
        if out_path.exists() and not args.overwrite:
            print(f"[{subset}] deja present, ignore : {out_path}  (--overwrite pour refaire)")
            continue

        if subset == "train":
            dataset = COCOTrainImageDataset(
                transform=transform, max_images=args.max_images, return_id=True
            )
        else:
            dataset = COCOTestImageDataset(transform=transform)
            if args.max_images:
                dataset.img_list = dataset.img_list[: args.max_images]

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        start = time.perf_counter()
        features, labels, ids = extract_features(
            loader, backbone, device, with_labels=(subset == "train"), desc=f"{subset}"
        )
        elapsed = time.perf_counter() - start

        meta = {
            "model": args.model,
            "feature_dim": feature_dim,
            "image_size": args.image_size,
            "resize_mode": args.resize_mode,
            "subset": subset,
            "n": len(ids),
            "seconds": round(elapsed, 1),
        }
        save_cache(out_path, features, ids, labels=labels, meta=meta)

        size_mb = out_path.stat().st_size / 1e6
        print(f"[{subset}] {len(ids):,} images en {human_time(elapsed)} "
              f"({len(ids) / elapsed:.1f} img/s) -> {out_path.name}, {size_mb:.0f} Mo")
        if labels is not None:
            print(f"         features {features.shape} {features.dtype}, "
                  f"labels {labels.shape}, {labels.sum() / len(labels):.2f} classes/image")

    print(f"\nCache : {cache_path(args.model, args.image_size, args.resize_mode, 'train').parent}")
    print("Etape suivante : python3 scripts/train_head.py --model", args.model)


if __name__ == "__main__":
    main()
