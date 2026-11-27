"""Validate and partition the official training split."""

import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

DATASET = "zh-plus/tiny-imagenet"
REVISION = "5a77092c28e51558c5586e9c5eb71a7e17a5e43f"
SCHEMA = pa.schema([("id", pa.int64()), ("label", pa.int64()), ("image", pa.binary())])
PROFILES = {"full": (450, 30, 20), "smoke": (8, 2, 2)}


def read_image(raw: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw), formats=["JPEG", "PNG"]) as image:
        if image.size != (64, 64):
            raise ValueError("expected a 64 × 64 image")
        return image.convert("RGB")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(file.name, path)


def synthetic_rows() -> Iterator[tuple[int, int, bytes]]:
    rng = np.random.default_rng(21)
    for label in range(4):
        for index in range(24):
            pixels = rng.integers(5, 50, (64, 64, 3), dtype=np.uint8)
            channel = label % 3
            start = 8 if label < 3 else 32
            pixels[start : start + 24, start : start + 24, channel] = 230
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            yield label * 24 + index, label, buffer.getvalue()


def source(
    profile: str, arrow: Path | None
) -> tuple[list[str], Iterator[tuple[int, int, bytes]]]:
    if profile == "fixture":
        return [f"synthetic-{i}" for i in range(4)], synthetic_rows()

    from datasets import Dataset as ArrowDataset
    from datasets import Image as ArrowImage
    from datasets import load_dataset

    if arrow is None:
        rows = load_dataset(DATASET, revision=REVISION, split="train")
    else:
        if arrow.name != "tiny-imagenet-train.arrow":
            raise ValueError("--arrow must refer to the official training cache file")
        rows = ArrowDataset.from_file(str(arrow))
    names = list(rows.features["label"].names)
    if len(names) != 200 or len(rows) != 100_000:
        raise ValueError("expected the official 100,000-image training split")
    rows = rows.cast_column("image", ArrowImage(decode=False))
    if Counter(rows["label"]) != dict.fromkeys(range(200), 500):
        raise ValueError("expected 500 official training rows per class")
    return names, (
        (i, row["label"], row["image"]["bytes"]) for i, row in enumerate(rows)
    )


def prepare(root: Path, profile: str, seed: int, arrow: Path | None = None) -> Path:
    names, raw_rows = source(profile, arrow)
    targets = PROFILES.get(profile, (16, 4, 4))
    by_image: dict[str, list[tuple[int, int, bytes]]] = defaultdict(list)
    rejects = []
    seen = set()
    for row_id, label, raw in raw_rows:
        try:
            if not 0 <= label < len(names) or not isinstance(raw, bytes):
                raise ValueError("invalid label or missing image bytes")
            if row_id in seen:
                raise ValueError("duplicate source row ID")
            seen.add(row_id)
            read_image(raw)
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            rejects.append(
                {
                    "id": row_id,
                    "reason": str(exc)
                    if isinstance(exc, ValueError)
                    else type(exc).__name__,
                    "image_sha256": hashlib.sha256(raw).hexdigest()
                    if isinstance(raw, bytes)
                    else None,
                }
            )
            continue
        by_image[hashlib.sha256(raw).hexdigest()].append((row_id, label, raw))

    grouped: dict[int, list[tuple[int, int, bytes]]] = defaultdict(list)
    for image_sha256, rows in by_image.items():
        first, *copies = rows
        if any(row[1] != first[1] for row in copies):
            rejected = rows
            reason = "conflicting labels for identical image"
        else:
            grouped[first[1]].append(first)
            rejected = copies
            reason = "duplicate image with same label"
        rejects.extend(
            {"id": row[0], "reason": reason, "image_sha256": image_sha256}
            for row in rejected
        )

    subsets: dict[str, list[tuple[int, int, bytes]]] = {
        key: [] for key in ("fit", "tune", "gate")
    }
    rng = np.random.default_rng(seed)
    counts = []
    for label in range(len(names)):
        examples = grouped[label]
        if len(examples) < sum(targets) and profile != "full":
            raise ValueError(f"class {label} has too few valid examples")
        fit, tune, gate = targets if profile != "full" else (len(examples) - 50, 30, 20)
        if fit <= 0:
            raise ValueError(f"class {label} has too few valid examples")
        counts.append((fit, tune, gate))
        start = 0
        order = rng.permutation(len(examples))
        # Permute the full class first so smoke sampling still has a fixed seed.
        for name, count in zip(subsets, (fit, tune, gate), strict=True):
            subsets[name].extend(examples[int(i)] for i in order[start : start + count])
            start += count

    manifest = {
        "contract": 2,
        "source": "synthetic" if profile == "fixture" else DATASET,
        "revision": None if profile == "fixture" or arrow is not None else REVISION,
        "profile": profile,
        "seed": seed,
        "classes": names,
        "counts_by_class": counts,
        "counts": {name: len(items) for name, items in subsets.items()},
        "quarantined": len(rejects),
    }
    digest = hashlib.sha256(json.dumps([manifest, rejects], sort_keys=True).encode())
    for name, items in subsets.items():
        digest.update(name.encode())
        for row_id, label, raw in items:
            digest.update(f"{row_id}:{label}:{len(raw)}:".encode())
            digest.update(raw)
    version = digest.hexdigest()
    target = root / "data" / version
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / "version"
        staged.mkdir()
        for name, items in subsets.items():
            columns = dict(zip(SCHEMA.names, zip(*items, strict=True), strict=True))
            pq.write_table(
                pa.Table.from_pydict(columns, schema=SCHEMA), staged / f"{name}.parquet"
            )
        write_json(staged / "manifest.json", {**manifest, "data_version": version})
        write_json(staged / "quarantine.json", rejects)
        try:
            staged.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    return target


def tensor_image(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255


class Images(Dataset):
    def __init__(
        self,
        path: Path,
        normalization: tuple[torch.Tensor, torch.Tensor] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        table = pq.read_table(path, columns=["image", "label"])
        self.images = table["image"]
        self.labels = table["label"]
        self.normalization = normalization
        self.rng = rng

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = read_image(self.images[index].as_py())
        if self.rng is not None:
            image = ImageOps.expand(image, border=4, fill=0)
            left, top = map(int, self.rng.integers(0, 9, size=2))
            image = image.crop((left, top, left + 64, top + 64))
            if self.rng.random() < 0.5:
                image = ImageOps.mirror(image)
        values = tensor_image(image)
        if self.normalization is not None:
            mean, std = self.normalization
            values = (values - mean) / std
        return values, self.labels[index].as_py()
