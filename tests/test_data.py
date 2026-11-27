import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
from PIL import Image

from tiny_imagenet.data import prepare, source


class DuplicateImagesTest(unittest.TestCase):
    def test_quarantines_conflicts_and_keeps_first_same_label_copy(self):
        names, fixture = source("fixture", None)
        rows = list(fixture)
        conflicting = rows[0][2]
        same_label = rows[1][2]
        replacement = io.BytesIO()
        Image.new("RGB", (64, 64), (64, 120, 200)).save(replacement, format="PNG")
        rows.extend(
            [
                (96, 0, replacement.getvalue()),
                (97, 0, same_label),
                (98, 0, conflicting),
                (99, 1, conflicting),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "tiny_imagenet.data.source", side_effect=lambda *_: (names, iter(rows))
            ):
                first = prepare(root / "first", "fixture", 42)
                second = prepare(root / "second", "fixture", 42)
            self.assertEqual(first.name, second.name)
            for file in first.iterdir():
                self.assertEqual(file.read_bytes(), (second / file.name).read_bytes())

            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(manifest["contract"], 2)
            self.assertEqual(manifest["counts"], {"fit": 64, "tune": 16, "gate": 16})
            self.assertEqual(manifest["quarantined"], 4)

            kept_ids = set()
            kept_hashes = set()
            for split in ("fit", "tune", "gate"):
                for row in pq.read_table(first / f"{split}.parquet").to_pylist():
                    digest = hashlib.sha256(row["image"]).hexdigest()
                    self.assertNotIn(digest, kept_hashes)
                    kept_hashes.add(digest)
                    kept_ids.add(row["id"])
            self.assertIn(1, kept_ids)
            self.assertIn(96, kept_ids)
            self.assertFalse({0, 97, 98, 99} & kept_ids)

            quarantined = {
                row["id"]: (row["reason"], row["image_sha256"])
                for row in json.loads((first / "quarantine.json").read_text())
            }
            conflict_status = (
                "conflicting labels for identical image",
                hashlib.sha256(conflicting).hexdigest(),
            )
            self.assertEqual(
                quarantined,
                {
                    0: conflict_status,
                    97: (
                        "duplicate image with same label",
                        hashlib.sha256(same_label).hexdigest(),
                    ),
                    98: conflict_status,
                    99: conflict_status,
                },
            )

    def test_conflicting_images_cannot_fill_the_class_quota(self):
        names, fixture = source("fixture", None)
        rows = list(fixture)
        rows.append((96, 1, rows[0][2]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("tiny_imagenet.data.source", return_value=(names, iter(rows))),
                self.assertRaisesRegex(
                    ValueError, "class 0 has too few valid examples"
                ),
            ):
                prepare(root, "fixture", 42)
            self.assertFalse((root / "data").exists())
