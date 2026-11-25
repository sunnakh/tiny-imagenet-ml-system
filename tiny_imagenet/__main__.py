"""Run the data, training, evaluation, registration and serving stages."""

import argparse
import json
import os
import uuid
from pathlib import Path

import torch

from .data import prepare
from .release import bundle_path, export, promote, register, rollback
from .service import serve
from .training import device_for, evaluate, train


def pipeline(args: argparse.Namespace) -> dict:
    epochs = (
        args.epochs
        if args.epochs is not None
        else {"fixture": 6, "smoke": 1, "full": 75}[args.profile]
    )
    if epochs < 1 or args.patience < 0:
        raise ValueError("epochs must be positive and patience must be nonnegative")
    root = args.state.resolve()
    data = prepare(root, args.profile, args.seed, args.arrow)
    print(f"Prepared data version {data.name}", flush=True)
    bundle = bundle_path(root, str(uuid.uuid4()))
    bundle.mkdir(parents=True)
    weights = bundle / "weights.pt"
    device = device_for(args.device)
    metadata, history = train(data, weights, epochs, args.patience, args.seed, device)
    metadata["scores"], metadata["baseline"] = evaluate(data, weights, metadata, device)
    return register(root, bundle, metadata, history, shadow=args.shadow)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path, default=Path(os.environ.get("TINY_STATE", ".state"))
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "demo"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument(
            "--profile", choices=("fixture", "smoke", "full"), default="full"
        )
        command_parser.add_argument("--arrow", type=Path)
        command_parser.add_argument("--seed", type=int, default=42)
        if command != "prepare":
            command_parser.add_argument("--epochs", type=int)
            command_parser.add_argument("--patience", type=int, default=10)
            command_parser.add_argument(
                "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
            )
            command_parser.add_argument("--shadow", action="store_true")
    commands.add_parser("promote")
    commands.add_parser("rollback")
    commands.add_parser("export").add_argument("output", type=Path)
    server = commands.add_parser("serve")
    for command_parser in (server, commands.choices["demo"]):
        command_parser.add_argument("--host", default="127.0.0.1")
        command_parser.add_argument(
            "--port", type=int, default=int(os.environ.get("PORT", "8080"))
        )
        command_parser.add_argument("--timeout-ms", type=float, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "prepare":
        print(prepare(args.state, args.profile, args.seed, args.arrow))
    elif args.command in ("promote", "rollback"):
        print((promote if args.command == "promote" else rollback)(args.state))
    elif args.command == "export":
        print(export(args.state, args.output))
    else:
        if args.command != "serve":
            metadata = pipeline(args)
            print(json.dumps(metadata, indent=2), flush=True)
            if metadata["rejected"] and args.command == "train":
                parser.exit(
                    2, f"Gate rejected candidate because {metadata['rejected']}\n"
                )
        if args.command in ("serve", "demo"):
            serve(args.state, args.host, args.port, args.timeout_ms)


if __name__ == "__main__":
    main()
