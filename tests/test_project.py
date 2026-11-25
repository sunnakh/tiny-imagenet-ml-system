import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow.parquet as pq
import torch
from PIL import Image

from tiny_imagenet.__main__ import pipeline
from tiny_imagenet.data import prepare, source, write_json
from tiny_imagenet.model import TinyCNN
from tiny_imagenet.release import (
    REGISTRY_NAME,
    bundle_path,
    export,
    promote,
    register,
    releases,
    rollback,
)
from tiny_imagenet.service import App
from tiny_imagenet.training import evaluate


class ProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.metadata = pipeline(
            SimpleNamespace(
                state=cls.root,
                profile="fixture",
                epochs=6,
                arrow=None,
                seed=42,
                device="cpu",
                patience=10,
                shadow=False,
            )
        )
        assert cls.metadata["rejected"] is None, cls.metadata

    @staticmethod
    def image():
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color="white").save(buffer, format="PNG")
        return buffer.getvalue()

    def test_deterministic_partitions_and_quarantine(self):
        ids = set()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            names, rows = source("fixture", None)
            all_rows = list(rows) + [(999, 0, b"broken")]
            with patch(
                "tiny_imagenet.data.source",
                side_effect=lambda *a: (names, iter(all_rows)),
            ):
                first = prepare(path / "one", "fixture", 42)
                second = prepare(path / "two", "fixture", 42)
            self.assertEqual(first.name, second.name)
            self.assertEqual(
                json.loads((first / "quarantine.json").read_text())[0]["id"], 999
            )
            for file in first.iterdir():
                self.assertEqual(file.read_bytes(), (second / file.name).read_bytes())
            for name in ("fit", "tune", "gate"):
                split = set(pq.read_table(first / f"{name}.parquet")["id"].to_pylist())
                self.assertFalse(ids.intersection(split))
                ids.update(split)
            self.assertEqual(len(ids), 96)

    def test_registry_and_worse_model_rejected(self):
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=f"sqlite:///{self.root / 'mlflow.db'}")
        version = client.get_model_version(
            REGISTRY_NAME, self.metadata["registry_version"]
        )
        self.assertEqual(version.tags["data_version"], self.metadata["data_version"])
        active = releases(self.root)["active"]
        bundle = bundle_path(self.root, "643d69c7-6d0f-439a-8fbe-f731affad414")
        bundle.mkdir()
        # Zero logits predict class zero, the baseline tie-break class.
        weights = bundle / "weights.pt"
        model = TinyCNN(4)
        torch.save(
            {key: torch.zeros_like(value) for key, value in model.state_dict().items()},
            weights,
        )
        data = self.root / "data" / self.metadata["data_version"]
        scores, baseline = evaluate(data, weights, self.metadata, torch.device("cpu"))
        self.assertEqual(scores, baseline)
        metadata = {**self.metadata, "scores": scores, "baseline": baseline}
        result = register(self.root, bundle, metadata, [], shadow=False)
        self.assertIn("baseline", result["rejected"])
        self.assertEqual(releases(self.root)["active"], active)

    def test_candidate_beating_baseline_cannot_regress_against_champion(self):
        active = releases(self.root)["active"]
        bundle = bundle_path(self.root, "32942ebb-6d2d-4ea6-b625-d41ead783a1d")
        bundle.mkdir()
        weights = bundle_path(self.root, self.metadata["release_id"]) / "weights.pt"
        (bundle / "weights.pt").write_bytes(weights.read_bytes())

        baseline = self.metadata["baseline"]
        champion = self.metadata["scores"]
        scores = {
            "accuracy": champion["accuracy"],
            "macro_f1": (baseline["macro_f1"] + champion["macro_f1"]) / 2,
        }
        for metric, value in scores.items():
            self.assertGreater(value, baseline[metric])
        self.assertLess(scores["macro_f1"], champion["macro_f1"])

        candidate = {**self.metadata, "scores": scores}
        candidate.pop("registry_version")
        result = register(self.root, bundle, candidate, [], shadow=False)

        self.assertEqual(result["rejected"], "candidate regressed against the champion")
        self.assertNotIn("registry_version", result)
        self.assertEqual(releases(self.root)["active"], active)

    def test_inference_feedback_and_incident_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exported"
            active = export(self.root, root)
            app = App(root)
            self.addCleanup(app.close)
            raw = self.image()
            for _ in range(20):
                response = app.predict(raw)
                self.assertEqual(response["mode"], "active")
                app.feedback(response["request_id"], response["class_id"])
            alerts = app.report()
            self.assertIn("input_drift", alerts["alerts"])
            self.assertEqual(alerts["feedback_accuracy"], 1.0)
            self.assertLess(alerts["p95_ms"], 250)
            with self.assertRaises(OSError):
                app.predict(b"invalid")

            weights = bundle_path(root, active) / "weights.pt"
            restored = root / "saved-weights.pt"
            weights.rename(restored)
            with self.assertLogs("tiny_imagenet.service", "ERROR"):
                for _ in range(3):
                    self.assertEqual(app.predict(raw)["mode"], "fallback")
            self.assertIn("fallback_rate_high", app.report()["alerts"])
            restored.rename(weights)
            self.assertEqual(app.predict(raw)["mode"], "active")

    def test_shadow_promotion_and_rollback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exported"
            active = export(self.root, root)
            active_path = bundle_path(root, active)
            active_metadata = json.loads((active_path / "metadata.json").read_text())
            candidate = "7b029ab6-81c2-47b8-a4b6-428e6fb23ea7"
            candidate_path = bundle_path(root, candidate)
            candidate_path.mkdir()
            weights = (active_path / "weights.pt").read_bytes()
            (candidate_path / "weights.pt").write_bytes(weights)
            # Simulate two gate reports so a fully trained fixture can exercise release transitions.
            write_json(
                active_path / "metadata.json",
                {
                    **active_metadata,
                    "scores": {"accuracy": 0.60, "macro_f1": 0.60},
                },
            )
            write_json(
                candidate_path / "metadata.json",
                {
                    **active_metadata,
                    "release_id": candidate,
                    "scores": {"accuracy": 0.75, "macro_f1": 0.75},
                    "weights_sha256": hashlib.sha256(weights).hexdigest(),
                },
            )
            write_json(
                root / "releases.json",
                {"active": active, "previous": None, "shadow": candidate},
            )
            app = App(root)
            try:
                self.assertEqual(app.predict(self.image())["release_id"], active)
                for _ in range(100):
                    if app.report()["shadow"]["requests"]:
                        break
                    time.sleep(0.01)
                self.assertEqual(app.report()["shadow"]["requests"], 1)
                self.assertEqual(promote(root), candidate)
                self.assertEqual(app.predict(self.image())["release_id"], candidate)
                self.assertEqual(rollback(root), active)
                self.assertEqual(app.predict(self.image())["release_id"], active)
            finally:
                app.close()
