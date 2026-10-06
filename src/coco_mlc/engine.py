"""Boucles d'entrainement et d'evaluation, checkpoints et logging Tensorboard.

Adapte des fonctions ``train_loop`` / ``validation_loop`` fournies dans le
sujet, avec deux differences assumees :

- ``evaluate`` accumule les sorties du reseau puis delegue le calcul a
  ``metrics.py``, ce qui permet de reutiliser exactement les memes scores pour
  la calibration des seuils et l'analyse par classe ;
- le seuillage est applique sur les probabilites ``sigmoid(logits)`` et non sur
  les sorties brutes. Le code du sujet ecrit ``outputs > th_multi_label`` avec
  ``th_multi_label=0.5`` par defaut, ce qui n'a de sens que si le reseau se
  termine par une sigmoide ; avec ``BCEWithLogitsLoss`` le seuil equivalent sur
  les logits serait 0.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .metrics import all_metrics


def _unpack(batch):
    """Supporte les batches ``(x, y)`` et ``(x, y, id)``."""
    return batch[0], batch[1]


def train_one_epoch(
    loader,
    net,
    criterion,
    optimizer,
    device,
    scheduler=None,
    mbatch_loss_group: int = -1,
    progress: bool = True,
    desc: str = "train",
    scaler=None,
):
    """Une epoque d'entrainement. Retourne la perte moyenne et l'historique.

    ``mbatch_loss_group`` > 0 enregistre la perte moyenne tous les N
    mini-batches, pour tracer une courbe plus fine que l'epoque.
    ``scaler`` active la precision mixte (utile seulement sur GPU).
    """
    net.train()
    total_loss = 0.0
    seen = 0
    running = 0.0
    mbatch_losses: list[float] = []
    use_amp = scaler is not None

    iterator = tqdm(loader, desc=desc, leave=False, disable=not progress)
    for i, batch in enumerate(iterator):
        inputs, labels = _unpack(batch)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        running += loss.item()
        if mbatch_loss_group > 0 and i % mbatch_loss_group == mbatch_loss_group - 1:
            mbatch_losses.append(running / mbatch_loss_group)
            running = 0.0
        if progress:
            iterator.set_postfix(loss=f"{total_loss / max(seen, 1):.4f}")

    return total_loss / max(seen, 1), mbatch_losses


@torch.no_grad()
def collect_outputs(loader, net, device, progress: bool = True, desc: str = "eval", amp: bool = False):
    """Concatene logits et cibles sur tout un loader.

    Les stocker une fois permet de calculer n'importe quel seuil ou metrique
    ensuite sans repasser dans le reseau.
    """
    net.eval()
    logits_all = []
    targets_all = []
    for batch in tqdm(loader, desc=desc, leave=False, disable=not progress):
        inputs, labels = _unpack(batch)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = net(inputs.to(device, non_blocking=True))
        logits_all.append(logits.detach().float().cpu())
        targets_all.append(labels.detach().float().cpu())
    return torch.cat(logits_all), torch.cat(targets_all)


def evaluate(
    loader,
    net,
    criterion,
    device,
    thresholds=0.5,
    class_metrics: bool = False,
    progress: bool = True,
    desc: str = "eval",
    return_scores: bool = False,
    amp: bool = False,
):
    """Perte et metriques completes sur un loader."""
    logits, targets = collect_outputs(loader, net, device, progress=progress, desc=desc, amp=amp)
    loss = float(criterion(logits, targets))
    scores = torch.sigmoid(logits)

    out = all_metrics(scores, targets, thresholds=thresholds, class_metrics=class_metrics)
    if class_metrics:
        results, per_class = out
    else:
        results, per_class = out, None
    results = dict(results)
    results["loss"] = loss

    if return_scores:
        return results, per_class, scores, targets
    return (results, per_class) if class_metrics else results


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: str | Path, model, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload["model_state_dict"] = model.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return f"cpu ({torch.get_num_threads()} threads)"


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


# ---------------------------------------------------------------------------
# Tensorboard (optionnel) - repris du sujet, simplifie
# ---------------------------------------------------------------------------
def update_graphs(
    summary_writer,
    epoch: int,
    train_results: dict,
    val_results: dict,
    train_class_results=None,
    val_class_results=None,
    class_names=None,
    mbatch_group: int = -1,
    mbatch_count: int = 0,
    mbatch_losses=None,
):
    """Ecrit pertes et metriques dans un ``SummaryWriter``."""
    if mbatch_group > 0 and mbatch_losses:
        for i, value in enumerate(mbatch_losses):
            summary_writer.add_scalar(
                "Losses/Train mini-batches", value, epoch * mbatch_count + (i + 1) * mbatch_group
            )

    step = (epoch + 1) * mbatch_count if mbatch_group > 0 else (epoch + 1)

    summary_writer.add_scalars(
        "Losses/Train vs Validation",
        {"Train": train_results["loss"], "Validation": val_results["loss"]},
        step,
    )
    for key, label in (
        ("f1", "F1 (serveur)"),
        ("precision", "Precision (serveur)"),
        ("recall", "Recall (serveur)"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("micro_f1", "Micro F1"),
        ("mAP", "mAP"),
    ):
        if key in train_results and key in val_results:
            summary_writer.add_scalars(
                f"Metrics/{label}",
                {"Train": train_results[key], "Validation": val_results[key]},
                step,
            )

    if train_class_results and val_class_results and class_names:
        for i, name in enumerate(class_names):
            for key in ("f1", "precision", "recall"):
                summary_writer.add_scalars(
                    f"Class Metrics/{name}/{key}",
                    {
                        "Train": train_class_results[i][key],
                        "Validation": val_class_results[i][key],
                    },
                    step,
                )
    summary_writer.flush()
