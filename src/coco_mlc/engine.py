"""Training and evaluation loops, checkpoints, and TensorBoard logging.

Adapted from the ``train_loop`` / ``validation_loop`` functions in the
assignment, with two deliberate differences:

- ``evaluate`` stores the network outputs, then delegates the scores to
  ``metrics.py``, so the same scores can be reused for threshold calibration
  and per-class analysis;
- the threshold is applied on the probabilities ``sigmoid(logits)``, not on
  the raw outputs. The assignment code writes ``outputs > th_multi_label`` with
  ``th_multi_label=0.5`` by default, which only makes sense if the network ends
  with a sigmoid. With ``BCEWithLogitsLoss`` the matching threshold on the
  logits would be 0.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .metrics import all_metrics


def _unpack(batch):
    """Supports batches ``(x, y)`` and ``(x, y, id)``."""
    return batch[0], batch[1]


def frozen_modules_eval(net) -> None:
    """Set to eval the modules whose own parameters are all frozen.

    ``net.train()`` would otherwise put frozen BatchNorm layers back in train
    mode, and their running stats would move even though the weights are fixed.
    """
    for module in net.modules():
        params = list(module.parameters(recurse=False))
        if params and all(not p.requires_grad for p in params):
            module.eval()


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
    frozen_eval: bool = False,
):
    """One training epoch. Returns the mean loss and the history.

    ``mbatch_loss_group`` > 0 records the mean loss every N mini-batches, for
    a curve that is finer than one point per epoch.
    ``scaler`` turns on mixed precision (useful only on GPU).
    """
    net.train()
    if frozen_eval:
        frozen_modules_eval(net)
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


def fit_stage(
    net,
    train_loader,
    train_eval_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs: int,
    scheduler=None,
    amp: bool = False,
    desc: str = "train",
) -> list[dict]:
    """Several epochs, with train and validation error at threshold 0.5.

    Train evaluation uses ``train_eval_loader`` (no augmentation), so the
    train/validation gap measures generalization and not the noise of random
    transforms. F1 is the server F1. The error in percent is ``100 * (1 - F1)``.
    """
    use_amp = bool(amp and getattr(device, "type", device) == "cuda")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, _ = train_one_epoch(
            train_loader, net, criterion, optimizer, device,
            scheduler=scheduler, scaler=scaler, frozen_eval=True,
            desc=f"{desc} {epoch}/{epochs}",
        )
        val_results = evaluate(
            val_loader, net, criterion, device, thresholds=0.5,
            desc="validation", amp=use_amp,
        )
        train_results = evaluate(
            train_eval_loader, net, criterion, device, thresholds=0.5,
            desc="train (eval)", amp=use_amp,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "train_f1": train_results["f1"],
            "val_f1": val_results["f1"],
            "train_precision": train_results["precision"],
            "val_precision": val_results["precision"],
            "train_recall": train_results["recall"],
            "val_recall": val_results["recall"],
            "train_error": 100.0 * (1.0 - train_results["f1"]),
            "val_error": 100.0 * (1.0 - val_results["f1"]),
        }
        history.append(row)
        print(
            f"  epoch {epoch:2d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_results['loss']:.4f}  "
            f"P/R/F1 train={train_results['precision']:.3f}/"
            f"{train_results['recall']:.3f}/{train_results['f1']:.3f}  "
            f"val={val_results['precision']:.3f}/"
            f"{val_results['recall']:.3f}/{val_results['f1']:.3f}  "
            f"error train={row['train_error']:.2f}%  val={row['val_error']:.2f}%"
        )
    return history


@torch.no_grad()
def collect_outputs(loader, net, device, progress: bool = True, desc: str = "eval", amp: bool = False):
    """Concatenate logits and targets over a whole loader.

    Storing them once lets us compute any threshold or metric afterwards
    without another forward pass.
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
    """Loss and full metrics on one loader."""
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
# TensorBoard (optional), simplified from the assignment
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
    """Write losses and metrics to a ``SummaryWriter``."""
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
        ("f1", "F1 (server)"),
        ("precision", "Precision (server)"),
        ("recall", "Recall (server)"),
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
