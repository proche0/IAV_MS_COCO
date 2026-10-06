"""Cree l'archive ZIP de rendu : code et instructions, sans poids ni images.

Le sujet penalise une archive qui contient des donnees binaires, des poids de
modele ou des images. Ce script n'embarque que des fichiers texte du depot
(README inclus) et refuse l'archive si un element interdit s'y glisse.

Usage :
    python scripts/package_submission.py
    python scripts/package_submission.py --name Tayeb_Paul
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "features",
    "runs",
    "ms-coco",
    "__pycache__",
    ".ipynb_checkpoints",
    "dist",
}

EXCLUDED_SUFFIXES = {
    ".pth", ".pt", ".bin", ".onnx", ".safetensors", ".ckpt",
    ".pkl", ".pickle", ".npy", ".npz",
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".ico",
    ".pyc", ".pyo", ".pyd", ".dll", ".exe", ".so", ".dylib",
    ".zip",
}

# Images integrees (notebooks Matplotlib, pieces jointes base64).
# On ne sonde que les notebooks : le source de ce script cite ces marqueurs.
NOTEBOOK_SUFFIXES = {".ipynb", ".html"}
EMBEDDED_IMAGE_MARKERS = (
    b"image/png",
    b"image/jpeg",
    b"image/webp",
    b"image/gif",
    b"iVBORw0KGgo",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="Tayeb_Paul",
                   help="nom du fichier ZIP, sans extension (noms du groupe)")
    p.add_argument("--output", type=Path, default=None,
                   help="chemin du ZIP (defaut : <racine du depot>/<nom>.zip)")
    return p.parse_args()


def _is_binary(path: Path) -> bool:
    chunk = path.read_bytes()[:8192]
    return b"\x00" in chunk


def _has_embedded_image(path: Path) -> bool:
    if path.suffix.lower() not in NOTEBOOK_SUFFIXES:
        return False
    data = path.read_bytes()
    return any(marker in data for marker in EMBEDDED_IMAGE_MARKERS)


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if _is_binary(path) or _has_embedded_image(path):
            continue
        yield path


def verify_archive(zf: zipfile.ZipFile) -> list[str]:
    problems = []
    names = zf.namelist()
    if "README.md" not in names:
        problems.append("README.md absent : les instructions d'execution manquent")
    for name in names:
        suffix = Path(name).suffix.lower()
        if suffix in EXCLUDED_SUFFIXES:
            problems.append(f"fichier interdit : {name}")
        parts = set(Path(name).parts)
        if parts & EXCLUDED_DIRS:
            problems.append(f"dossier interdit : {name}")
        data = zf.read(name)
        if b"\x00" in data[:8192]:
            problems.append(f"donnees binaires : {name}")
        if Path(name).suffix.lower() in NOTEBOOK_SUFFIXES and any(
            marker in data for marker in EMBEDDED_IMAGE_MARKERS
        ):
            problems.append(f"image integree : {name}")
    return problems


def main():
    args = parse_args()
    out = args.output or (REPO_ROOT / f"{args.name}.zip")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in iter_files(REPO_ROOT) if p.resolve() != out]
    if not any(p.name == "README.md" and p.parent == REPO_ROOT for p in files):
        raise SystemExit("README.md introuvable a la racine : archive non creee.")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            # ZipInfo garde les slashs POSIX. ZipFile.write sur Windows melange
            # antislashs du repertoire central et slashs de l'entete local.
            info = zipfile.ZipInfo(
                filename=path.relative_to(REPO_ROOT).as_posix(),
                date_time=time.localtime(path.stat().st_mtime)[:6],
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
        problems = verify_archive(zf)
        names = zf.namelist()

    if problems:
        out.unlink(missing_ok=True)
        raise SystemExit("Archive refusee :\n- " + "\n- ".join(problems))

    print(f"Archive : {out}")
    print(f"Fichiers : {len(names)}")
    print("README.md : present")
    print("Poids, images et donnees binaires : absents")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
