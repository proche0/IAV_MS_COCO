"""Reproductibilite et journalisation des experiences."""

from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def environment_summary() -> dict[str, str]:
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "threads": str(torch.get_num_threads()),
        "git": git_revision(),
    }


def log_experiment(csv_path: str | Path, row: dict) -> Path:
    """Ajoute une ligne au registre des experiences.

    Le fichier est reecrit si de nouvelles colonnes apparaissent, afin de
    pouvoir enrichir le suivi sans casser l'historique.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}

    existing: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing = list(csv.DictReader(f))

    fieldnames: list[str] = []
    for item in (*existing, row):
        for key in item:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in (*existing, row):
            writer.writerow({k: item.get(k, "") for k in fieldnames})
    return csv_path


def save_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
