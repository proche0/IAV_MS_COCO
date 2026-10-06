"""Statistiques du dataset et poids reels de la metrique d'evaluation.

Produit :
  outputs/class_stats.csv        frequences, poids de la metrique, support par split
  outputs/cooccurrence_top.csv   paires de classes les plus co-occurrentes
  outputs/figures/*.png          figures pour le rapport

Usage :
    python3 scripts/explore_dataset.py
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from coco_mlc.config import CLASSES, DEFAULTS, NUM_CLASSES, PATHS, SEED
from coco_mlc.data import COCOTestImageDataset, get_split, load_label_matrix
from coco_mlc.metrics import metric_class_weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-figures", action="store_true", help="n'ecrit que les CSV")
    p.add_argument("--image-sample", type=int, default=2000,
                   help="nombre d'images echantillonnees pour les dimensions")
    return p.parse_args()


def image_dimension_stats(sample: int) -> pd.DataFrame:
    from PIL import Image

    rng = np.random.default_rng(SEED)
    rows = []
    for subset, directory in (("train", PATHS.train_images), ("test", PATHS.test_images)):
        names = sorted(p.name for p in directory.glob("*.jpg"))
        picks = rng.choice(len(names), size=min(sample, len(names)), replace=False)
        for i in picks:
            with Image.open(directory / names[i]) as img:
                w, h = img.size
            rows.append({"subset": subset, "width": w, "height": h,
                         "max_side": max(w, h), "ratio": w / h})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    PATHS.check()
    PATHS.mkdirs()
    figures_dir = PATHS.outputs / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ids, Y = load_label_matrix()
    n_images = len(ids)
    n_test = len(COCOTestImageDataset())

    print("=" * 78)
    print("DATASET")
    print("=" * 78)
    print(f"Images d'entrainement annotees : {n_images:,}")
    print(f"Images de test                 : {n_test:,}")
    print(f"Annotations totales            : {int(Y.sum()):,}")

    per_image = Y.sum(axis=1)
    print(f"Classes par image              : moyenne {per_image.mean():.2f}, "
          f"mediane {int(np.median(per_image))}, min {per_image.min()}, max {per_image.max()}")

    freqs = Y.sum(axis=0).astype(np.int64)
    weights = metric_class_weights(freqs).numpy()

    train_idx, val_idx = get_split(Y, args.val_fraction, args.seed, strategy="stratified")
    rnd_train_idx, rnd_val_idx = get_split(Y, args.val_fraction, args.seed, strategy="random")
    support_train = Y[train_idx].sum(axis=0)
    support_val = Y[val_idx].sum(axis=0)
    support_val_random = Y[rnd_val_idx].sum(axis=0)

    stats = pd.DataFrame({
        "class_id": range(NUM_CLASSES),
        "class_name": CLASSES,
        "frequency": freqs,
        "share_of_images_pct": 100.0 * freqs / n_images,
        "metric_weight_pct": 100.0 * weights,
        "support_train": support_train,
        "support_val_stratified": support_val,
        "support_val_random": support_val_random,
    }).sort_values("metric_weight_pct", ascending=False)

    stats.to_csv(PATHS.outputs / "class_stats.csv", index=False)

    print()
    print("=" * 78)
    print("POIDS REELS DE LA METRIQUE DU SERVEUR (ponderation 1/frequence)")
    print("=" * 78)
    head = stats.head(12)
    for _, row in head.iterrows():
        print(f"  {row.class_name:<16s} freq={int(row.frequency):6,d}  "
              f"poids={row.metric_weight_pct:6.2f} %  "
              f"val(strat)={int(row.support_val_stratified):4d}  "
              f"val(alea)={int(row.support_val_random):4d}")
    print("  ...")
    for _, row in stats.tail(3).iterrows():
        print(f"  {row.class_name:<16s} freq={int(row.frequency):6,d}  "
              f"poids={row.metric_weight_pct:6.3f} %")

    cumulative = stats.metric_weight_pct.cumsum()
    print()
    print(f"  Les 2 classes les plus rares pesent  {cumulative.iloc[1]:.1f} % du score")
    print(f"  Les 10 classes les plus rares pesent {cumulative.iloc[9]:.1f} % du score")
    print(f"  Les 40 classes les plus frequentes   {100 - cumulative.iloc[39]:.1f} % du score")
    print(f"  Rapport de frequence max/min         {freqs.max() / freqs.min():.0f}x")

    # Un modele qui ne predit jamais rien a un F1 de 0 ; un modele qui predit
    # tout a un rappel de 1 et une precision egale a la moyenne ponderee des
    # prevalences. Ces deux bornes situent les scores obtenus ensuite.
    prevalence = freqs / n_images
    trivial_precision = float((prevalence * weights).sum())
    print(f"  Reference 'tout predire' : precision={trivial_precision:.4f}, "
          f"rappel=1.0, F1={2 / (1 / trivial_precision + 1):.4f}")

    print()
    print("=" * 78)
    print("ECARTS DE SUPPORT EN VALIDATION : STRATIFIE VS ALEATOIRE")
    print("=" * 78)
    target = args.val_fraction * freqs
    err_strat = np.abs(support_val - target).sum() / target.sum()
    err_rand = np.abs(support_val_random - target).sum() / target.sum()
    print(f"  Erreur relative totale  stratifie={100 * err_strat:.2f} %  "
          f"aleatoire={100 * err_rand:.2f} %")
    print(f"  Support minimal en validation  stratifie={support_val.min()}  "
          f"aleatoire={support_val_random.min()}")

    # Co-occurrences : utile pour comprendre les confusions (une table a manger
    # vient presque toujours avec des chaises).
    cooc = Y.T.astype(np.int32) @ Y.astype(np.int32)
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            if cooc[i, j]:
                pairs.append({
                    "class_a": CLASSES[i],
                    "class_b": CLASSES[j],
                    "cooccurrences": int(cooc[i, j]),
                    "jaccard": float(cooc[i, j] / (freqs[i] + freqs[j] - cooc[i, j])),
                })
    cooc_df = pd.DataFrame(pairs).sort_values("cooccurrences", ascending=False)
    cooc_df.head(200).to_csv(PATHS.outputs / "cooccurrence_top.csv", index=False)
    print()
    print("=" * 78)
    print("PAIRES DE CLASSES LES PLUS CO-OCCURRENTES")
    print("=" * 78)
    for _, row in cooc_df.head(8).iterrows():
        print(f"  {row.class_a:<16s} + {row.class_b:<16s} {int(row.cooccurrences):6,d} images")

    dims = image_dimension_stats(args.image_sample)
    print()
    print("=" * 78)
    print(f"DIMENSIONS DES IMAGES (echantillon de {len(dims):,})")
    print("=" * 78)
    print(f"  grand cote toujours egal a 224 : {bool((dims.max_side == 224).all())}")
    print(f"  petit cote : min {min(dims.width.min(), dims.height.min())}, "
          f"max {max(dims.width.max(), dims.height.max())}")
    print(f"  ratio largeur/hauteur : min {dims.ratio.min():.2f}, max {dims.ratio.max():.2f}, "
          f"portrait {100 * (dims.ratio < 1).mean():.0f} %")
    print("  -> un Resize((224,224)) deforme l'image ; le mode 'pad' conserve le ratio")

    if not args.no_figures:
        write_figures(stats, per_image, dims, figures_dir)
        print(f"\nFigures : {figures_dir}")
    print(f"CSV     : {PATHS.outputs / 'class_stats.csv'}")


def write_figures(stats: pd.DataFrame, per_image: np.ndarray, dims: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_freq = stats.sort_values("frequency", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(by_freq)), by_freq.frequency, color="steelblue")
    ax.set_yscale("log")
    ax.set_xticks(range(len(by_freq)))
    ax.set_xticklabels(by_freq.class_name, rotation=90, fontsize=7)
    ax.set_ylabel("Nombre d'images (echelle log)")
    ax.set_title("Frequence des 80 classes dans le sous-ensemble train")
    fig.tight_layout()
    fig.savefig(out / "class_frequency.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(by_freq)), by_freq.metric_weight_pct, color="indianred")
    ax.set_xticks(range(len(by_freq)))
    ax.set_xticklabels(by_freq.class_name, rotation=90, fontsize=7)
    ax.set_ylabel("Poids dans le score (%)")
    ax.set_title("Poids de chaque classe dans la metrique du serveur (ponderation 1/frequence)")
    fig.tight_layout()
    fig.savefig(out / "metric_weights.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(stats.frequency, stats.metric_weight_pct, s=18, color="darkgreen")
    for _, row in stats.head(6).iterrows():
        ax.annotate(row.class_name, (row.frequency, row.metric_weight_pct),
                    fontsize=8, xytext=(6, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequence de la classe (echelle log)")
    ax.set_ylabel("Poids dans le score en % (echelle log)")
    ax.set_title("Moins une classe est frequente, plus elle pese")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "weight_vs_frequency.png", dpi=130)
    plt.close(fig)

    counter = collections.Counter(per_image.tolist())
    keys = sorted(counter)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(keys, [counter[k] for k in keys], color="slateblue")
    ax.set_xlabel("Nombre de classes presentes dans l'image")
    ax.set_ylabel("Nombre d'images")
    ax.set_title(f"Densite d'etiquettes (moyenne {per_image.mean():.2f} classes par image)")
    fig.tight_layout()
    fig.savefig(out / "labels_per_image.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(dims.width, dims.height, s=4, alpha=0.3)
    ax.set_xlabel("Largeur (px)")
    ax.set_ylabel("Hauteur (px)")
    ax.set_title("Dimensions des images : grand cote fixe a 224")
    fig.tight_layout()
    fig.savefig(out / "image_dimensions.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
