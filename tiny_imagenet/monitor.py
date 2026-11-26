"""Bounded-window service, drift and delayed-label measurements."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import time

from prometheus_client import CollectorRegistry, Gauge, generate_latest
from sklearn.metrics import f1_score

ALERT_NAMES = (
    "fallback_rate_high",
    "latency_high",
    "input_drift",
    "feedback_accuracy_low",
)


def prometheus_metrics(report: dict, ready: bool) -> bytes:
    registry = CollectorRegistry()
    p95_ms = report["p95_ms"]
    gauges = (
        ("ready", "Whether active model weights are available.", int(ready)),
        (
            "window_requests",
            "Prediction attempts in the active release window.",
            report["requests"],
        ),
        (
            "window_request_duration_p95_seconds",
            "Window p95 prediction processing time in seconds.",
            p95_ms / 1000 if p95_ms is not None else None,
        ),
        (
            "window_fallback_ratio",
            "Fraction of window attempts that abstained.",
            report["fallback_rate"],
        ),
        (
            "window_rgb_mean_shift",
            "Largest window RGB channel mean shift.",
            report["rgb_mean_shift"],
        ),
        (
            "window_labeled",
            "Labeled active predictions in the window.",
            report["labeled"],
        ),
        (
            "window_feedback_accuracy",
            "Accuracy on labeled window predictions.",
            report["feedback_accuracy"],
        ),
        (
            "window_feedback_macro_f1",
            "Macro F1 on labeled window predictions.",
            report["feedback_macro_f1"],
        ),
    )
    for name, help_text, value in gauges:
        if value is not None:
            Gauge(f"tiny_imagenet_{name}", help_text, registry=registry).set(value)
    alert = Gauge(
        "tiny_imagenet_alert",
        "Whether the service alert is active.",
        ["kind"],
        registry=registry,
    )
    for name in ALERT_NAMES:
        alert.labels(kind=name).set(int(name in report["alerts"]))
    return generate_latest(registry)


class Monitor:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "predictions.db"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, ts REAL NOT NULL, release TEXT, mode TEXT NOT NULL,
                    class_id INTEGER, rgb TEXT NOT NULL, ms REAL NOT NULL, label INTEGER
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS recent ON predictions (release, ts)")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=1)
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(
        self,
        request_id: str,
        release: str | None,
        mode: str,
        class_id: int | None,
        rgb: list[float],
        latency_ms: float,
    ) -> None:
        with self.connection() as db:
            db.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    request_id,
                    time(),
                    release,
                    mode,
                    class_id,
                    json.dumps(rgb),
                    latency_ms,
                ),
            )

    def feedback(self, request_id: str, label: int) -> None:
        with self.connection() as db:
            result = db.execute(
                "UPDATE predictions SET label = ? WHERE id = ? AND class_id IS NOT NULL "
                "AND (label IS NULL OR label = ?)",
                (label, request_id, label),
            )
            if result.rowcount != 1:
                raise ValueError("unknown prediction or conflicting feedback")

    def report(
        self,
        release: str | None,
        metadata: dict | None,
        target_ms: float,
        *,
        shadow: bool = False,
    ) -> dict:
        with self.connection() as db:
            rows = db.execute(
                "SELECT mode, class_id, rgb, ms, label FROM predictions "
                "WHERE release IS ? AND ts >= ? AND (mode LIKE 'shadow%') = ? ORDER BY ts DESC LIMIT 1000",
                (release, time() - 3600, shadow),
            ).fetchall()
        served = [row for row in rows if row[0] == ("shadow" if shadow else "active")]
        labeled = [row for row in served if row[4] is not None]
        latencies = sorted(row[3] for row in rows)
        p95 = latencies[(95 * len(latencies) - 1) // 100] if rows else None
        fallback_rate = (len(rows) - len(served)) / len(rows) if rows else 0.0
        drift, accuracy, macro_f1 = None, None, None
        alerts = []
        if fallback_rate > 0.10:
            alerts.append("fallback_rate_high")
        if p95 is not None and p95 > target_ms:
            alerts.append("latency_high")
        if metadata and len(rows) >= 20:
            channels = list(zip(*(json.loads(row[2]) for row in rows), strict=True))
            drift = max(
                abs(sum(values) / len(values) - reference)
                for values, reference in zip(channels, metadata["mean"], strict=True)
            )
            if drift > 0.15:
                alerts.append("input_drift")
        if len(labeled) >= 20:
            truth = [row[4] for row in labeled]
            predictions = [row[1] for row in labeled]
            accuracy = sum(
                a == b for a, b in zip(truth, predictions, strict=True)
            ) / len(labeled)
            macro_f1 = float(
                f1_score(truth, predictions, average="macro", zero_division=0)
            )
            if metadata and accuracy < metadata["scores"]["accuracy"] - 0.15:
                alerts.append("feedback_accuracy_low")
        return {
            "release_id": release,
            "requests": len(rows),
            "p95_ms": p95,
            "fallback_rate": fallback_rate,
            "rgb_mean_shift": drift,
            "labeled": len(labeled),
            "feedback_accuracy": accuracy,
            "feedback_macro_f1": macro_f1,
            "alerts": alerts,
        }
