"""Measure the warmed prediction endpoint with serial or concurrent requests."""

import argparse
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    if args.image:
        content = args.image.read_bytes()
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color="grey").save(buffer, format="PNG")
        content = buffer.getvalue()

    def measure(_: int) -> tuple[float, bool]:
        request = Request(
            args.url.rstrip("/") + "/predict",
            data=content,
            headers={"Content-Type": "application/octet-stream"},
        )
        start = perf_counter()
        try:
            with urlopen(request, timeout=5) as response:
                result = json.load(response)
                if result["mode"] != "active":
                    raise ValueError(
                        "successful prediction did not use the active model"
                    )
                active = True
        except HTTPError as error:
            with error:
                if error.code != 503 or json.load(error)["mode"] != "fallback":
                    raise
            active = False
        return (perf_counter() - start) * 1000, active

    _, active = measure(0)
    if not active:
        raise SystemExit("warmup did not use the active model")
    if args.concurrency == 1:
        results = [measure(index) for index in range(args.requests)]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(measure, range(args.requests)))
    served = [latency for latency, active in results if active]
    latency = p95([elapsed for elapsed, _ in results])
    target_ms = 250
    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "p95_ms": latency,
                "active_p95_ms": p95(served),
                "fallback_rate": (args.requests - len(served)) / args.requests,
                "target_ms": target_ms if args.concurrency == 1 else None,
            }
        )
    )
    if args.concurrency == 1 and (
        len(served) != args.requests or (latency is not None and latency > target_ms)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
