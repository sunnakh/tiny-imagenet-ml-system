"""Serve an image classifier with a bounded inference path and abstention."""

import html
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter
from typing import Any

import torch
from PIL import Image

from .data import read_image, tensor_image
from .model import TinyCNN
from .monitor import Monitor
from .release import bundle_path, metadata_for, releases, validate_weights
from .training import load_model

MAX_BYTES = 64 * 1024
LOG = logging.getLogger(__name__)


class Inference:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.slot = BoundedSemaphore()
        self.cached: tuple[str, TinyCNN, dict] | None = None

    def predict(self, release_id: str, raw: torch.Tensor) -> tuple[int, str]:
        weights = bundle_path(self.root, release_id) / "weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if self.cached is None or self.cached[0] != release_id:
            metadata = metadata_for(self.root, release_id)
            validate_weights(weights, metadata)
            model = load_model(weights, metadata, torch.device("cpu"))
            self.cached = (release_id, model, metadata)
        _, model, metadata = self.cached
        mean, std = (
            torch.tensor(metadata[key]).view(3, 1, 1) for key in ("mean", "std")
        )
        with torch.inference_mode():
            prediction = int(model(((raw - mean) / std)[None]).argmax(1).item())
        return prediction, metadata["classes"][prediction]

    def submit(self, release_id: str, raw: torch.Tensor):
        if not self.slot.acquire(blocking=False):
            return None
        try:
            future = self.pool.submit(self.predict, release_id, raw)
        except RuntimeError:
            self.slot.release()
            raise
        future.add_done_callback(lambda _: self.slot.release())
        return future


class App:
    def __init__(self, root: Path, timeout_ms: float = 1000) -> None:
        self.root = root
        self.timeout_ms = timeout_ms
        self.monitor = Monitor(root)
        self.active = Inference(root)
        self.shadow = Inference(root)

    def predict(self, image: bytes) -> dict[str, Any]:
        start = perf_counter()
        pixels = tensor_image(read_image(image))
        channels = pixels.mean((1, 2)).tolist()
        pointers = releases(self.root)
        active, shadow = pointers["active"], pointers["shadow"]
        request_id = str(uuid.uuid4())
        mode, class_id, label = "fallback", None, None
        if active:
            future = self.active.submit(active, pixels)
            if future is not None:
                try:
                    class_id, label = future.result(self.timeout_ms / 1000)
                    mode = "active"
                except TimeoutError:
                    LOG.warning("inference timeout for %s", active)
                except (OSError, RuntimeError, ValueError, KeyError):
                    LOG.exception("inference failed for %s", active)
        self.monitor.record(
            request_id,
            active,
            mode,
            class_id,
            channels,
            (perf_counter() - start) * 1000,
        )
        if shadow:
            shadow_start = perf_counter()
            task = self.shadow.submit(shadow, pixels)
            if task is not None:

                def record(done):
                    try:
                        predicted, _ = done.result()
                        result = "shadow"
                    except Exception:
                        LOG.exception("shadow inference failed for %s", shadow)
                        predicted, result = None, "shadow_fallback"
                    self.monitor.record(
                        request_id + ":shadow",
                        shadow,
                        result,
                        predicted,
                        channels,
                        (perf_counter() - shadow_start) * 1000,
                    )

                task.add_done_callback(record)
        return {
            "request_id": request_id,
            "release_id": active,
            "mode": mode,
            "class_id": class_id,
            "label": label,
        }

    def report(self) -> dict:
        pointers = releases(self.root)
        release_id = pointers["active"]
        metadata = metadata_for(self.root, release_id) if release_id else None
        report = self.monitor.report(release_id, metadata, target_ms=250)
        if shadow := pointers["shadow"]:
            report["shadow"] = self.monitor.report(
                shadow, None, target_ms=250, shadow=True
            )
        return report

    def feedback(self, request_id: str, label: int) -> None:
        with self.monitor.connection() as db:
            row = db.execute(
                "SELECT release FROM predictions WHERE id = ?", (request_id,)
            ).fetchone()
        release_id = row[0] if row else None
        if not release_id or not 0 <= label < len(
            metadata_for(self.root, release_id)["classes"]
        ):
            raise ValueError("unknown request ID or invalid label")
        self.monitor.feedback(request_id, label)

    def close(self) -> None:
        for worker in (self.active, self.shadow):
            worker.pool.shutdown(wait=True)


def serve(root: Path, host: str, port: int, timeout_ms: float) -> None:
    app = App(root, timeout_ms)

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)

        def send(self, status: int, body: dict | str) -> None:
            data = (json.dumps(body) if isinstance(body, dict) else body).encode()
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json"
                if isinstance(body, dict)
                else "text/html; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                active = releases(root)["active"]
                ready = (
                    active is not None
                    and (bundle_path(root, active) / "weights.pt").is_file()
                )
                self.send(200 if ready else 503, {"ready": ready, "release_id": active})
            elif self.path == "/metrics":
                self.send(200, app.report())
            elif self.path == "/dashboard":
                report = html.escape(json.dumps(app.report(), indent=2))
                self.send(
                    200,
                    f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=10>"
                    f"<title>Tiny ImageNet service</title><h1>Tiny ImageNet service</h1><pre>{report}</pre>",
                )
            else:
                self.send(404, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > MAX_BYTES:
                    self.send(413, {"error": "body exceeds 64 KiB"})
                    return
                if size <= 0:
                    raise ValueError("a nonempty request body is required")
                content = self.rfile.read(size)
                if len(content) != size:
                    raise ValueError("incomplete request body")
                if self.path == "/predict":
                    predicted = app.predict(content)
                    self.send(200 if predicted["mode"] == "active" else 503, predicted)
                elif self.path == "/feedback":
                    payload = json.loads(content)
                    if (
                        not isinstance(payload, dict)
                        or type(payload.get("label")) is not int
                    ):
                        raise ValueError(
                            "feedback requires a request ID and integer class label"
                        )
                    request_id = str(uuid.UUID(payload["request_id"]))
                    app.feedback(request_id, payload["label"])
                    self.send(200, {"recorded": True})
                else:
                    self.send(404, {"error": "unknown endpoint"})
            except (
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                OSError,
                Image.DecompressionBombError,
            ):
                self.send(400, {"error": "invalid image or feedback"})

    with ThreadingHTTPServer((host, port), Handler) as server:
        print(
            f"Serving on http://{host}:{port} with dashboard at /dashboard", flush=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.close()
