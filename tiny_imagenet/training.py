"""Fit from scratch and evaluate a frozen candidate on an isolated gate."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import Images
from .model import TinyCNN


class Scores(TypedDict):
    accuracy: float
    macro_f1: float


def source_identity() -> dict[str, str | None]:
    root = Path(__file__).resolve().parent.parent
    fingerprint = hashlib.sha256()
    for file in sorted((root / "tiny_imagenet").glob("*.py")):
        fingerprint.update(file.name.encode())
        fingerprint.update(file.read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "code_sha256": fingerprint.hexdigest()}


def device_for(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def normalization(data: Path, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for images, _ in DataLoader(Images(data), batch_size=batch_size):
        values = images.double()
        total += values.sum((0, 2, 3))
        squares += values.square().sum((0, 2, 3))
        count += images.shape[0] * 64 * 64
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt()
    if not torch.all(std > 0):
        raise ValueError("fit split has a constant channel")
    return mean.float().view(3, 1, 1), std.float().view(3, 1, 1)


def epoch_metrics(
    model: TinyCNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = int(targets.shape[0])
            total_loss += loss.item() * count
            correct += int((logits.argmax(1) == targets).sum().item())
            seen += count
    return total_loss / seen, correct / seen


def train(
    data: Path,
    output: Path,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[dict, list[dict]]:
    manifest = json.loads((data / "manifest.json").read_text())
    names = manifest["classes"]
    torch.manual_seed(seed)
    stats = normalization(data / "fit.parquet", batch_size)
    fit = DataLoader(
        Images(data / "fit.parquet", stats, np.random.default_rng(seed)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    tune = DataLoader(Images(data / "tune.parquet", stats), batch_size=batch_size)
    model = TinyCNN(len(names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy, best_epoch = -1.0, 0
    weights = {}
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = epoch_metrics(model, fit, device, optimizer)
        tune_loss, tune_acc = epoch_metrics(model, tune, device)
        schedule.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "tuning_loss": tune_loss,
            "tuning_accuracy": tune_acc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if tune_acc > best_accuracy:
            best_accuracy, best_epoch = tune_acc, epoch
            weights = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        if patience and epoch - best_epoch >= patience:
            break
    model.load_state_dict(weights)

    metadata = {
        "architecture": "tiny-cnn-v1",
        "data_version": manifest["data_version"],
        "profile": manifest["profile"],
        "classes": names,
        "selected_epoch": best_epoch,
        "mean": stats[0].flatten().tolist(),
        "std": stats[1].flatten().tolist(),
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        **source_identity(),
    }
    torch.save(weights, output)
    return metadata, history


def load_model(weights: Path, metadata: dict, device: torch.device) -> TinyCNN:
    if metadata["architecture"] != "tiny-cnn-v1":
        raise ValueError("unsupported model architecture")
    model = TinyCNN(len(metadata["classes"]))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval().to(device)


def evaluate(
    data: Path, weights: Path, metadata: dict, device: torch.device
) -> tuple[Scores, Scores]:
    model = load_model(weights, metadata, device)
    stats = tuple(torch.tensor(metadata[key]).view(3, 1, 1) for key in ("mean", "std"))
    loader = DataLoader(
        Images(data / "gate.parquet", (stats[0], stats[1])), batch_size=128
    )
    true, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
            true.extend(labels.tolist())

    fit_labels = pq.read_table(data / "fit.parquet", columns=["label"])[
        "label"
    ].to_pylist()
    majority = int(np.bincount(fit_labels).argmax())
    classes = list(range(len(metadata["classes"])))

    def metrics(predictions: list[int]) -> Scores:
        return {
            "accuracy": float(accuracy_score(true, predictions)),
            "macro_f1": float(
                f1_score(
                    true, predictions, labels=classes, average="macro", zero_division=0
                )
            ),
        }

    return metrics(predicted), metrics([majority] * len(true))
