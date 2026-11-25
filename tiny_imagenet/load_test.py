"""Measure the warmed inference endpoint under a bounded, serial workload."""

import argparse
import io
import json
import math
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("requests must be positive")
    if args.image:
        content = args.image.read_bytes()
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color="grey").save(buffer, format="PNG")
        content = buffer.getvalue()
    request = Request(
        args.url.rstrip("/") + "/predict",
        data=content,
        headers={"Content-Type": "application/octet-stream"},
    )

    def measure() -> float:
        start = perf_counter()
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        if result["mode"] != "active":
            raise RuntimeError("request did not use the active model")
        return (perf_counter() - start) * 1000

    measure()
    times = sorted(measure() for _ in range(args.requests))
    p95 = times[math.ceil(0.95 * len(times)) - 1]
    print(json.dumps({"requests": len(times), "p95_ms": p95, "target_ms": 250}))
    if p95 > 250:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
