"""Persist immutable model bundles and switch local release pointers."""

import fcntl
import hashlib
import json
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .data import write_json
from .training import Scores

REGISTRY_NAME = "tiny-imagenet-cnn"


@contextmanager
def release_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".release.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def releases(root: Path) -> dict[str, str | None]:
    pointer = root / "releases.json"
    if not pointer.exists():
        return {"active": None, "previous": None, "shadow": None}
    return json.loads(pointer.read_text())


def bundle_path(root: Path, release_id: str) -> Path:
    if str(uuid.UUID(release_id)) != release_id:
        raise ValueError("invalid release ID")
    return root / "models" / release_id


def metadata_for(root: Path, release_id: str) -> dict:
    return json.loads((bundle_path(root, release_id) / "metadata.json").read_text())


def validate_weights(path: Path, metadata: dict) -> None:
    with path.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    if fingerprint != metadata["weights_sha256"]:
        raise ValueError("model weight digest mismatch")


def decision(
    scores: Scores, baseline: Scores, champion: dict | None, data_version: str
) -> str | None:
    if not 0 <= scores["accuracy"] <= 1 or not 0 <= scores["macro_f1"] <= 1:
        return "invalid candidate scores"
    if not all(scores[key] > baseline[key] for key in ("accuracy", "macro_f1")):
        return "candidate must exceed the baseline on accuracy and macro F1"
    if champion is not None:
        if champion["data_version"] != data_version:
            return "champion was evaluated on different data"
        incumbent = champion["scores"]
        if not all(scores[key] >= incumbent[key] for key in ("accuracy", "macro_f1")):
            return "candidate regressed against the champion"
        if not any(scores[key] > incumbent[key] for key in ("accuracy", "macro_f1")):
            return "candidate tied the champion"
    return None


def tracking(root: Path):
    import mlflow

    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{root.resolve() / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name("tiny-imagenet") is None:
        mlflow.create_experiment(
            "tiny-imagenet",
            artifact_location=(root / "mlflow-artifacts").resolve().as_uri(),
        )
    mlflow.set_experiment("tiny-imagenet")
    return mlflow


def register(
    root: Path,
    bundle: Path,
    metadata: dict,
    history: list[dict],
    *,
    shadow: bool,
) -> dict:
    with release_lock(root):
        return track_candidate(root, bundle, metadata, history, shadow=shadow)


def track_candidate(
    root: Path, bundle: Path, metadata: dict, history: list[dict], *, shadow: bool
) -> dict:
    mlflow = tracking(root)
    with mlflow.start_run(run_name=bundle.name) as run:
        pointers = releases(root)
        active = pointers["active"]
        champion = metadata_for(root, active) if active else None
        reason = decision(
            metadata["scores"], metadata["baseline"], champion, metadata["data_version"]
        )
        with (bundle / "weights.pt").open("rb") as stream:
            metadata["weights_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
        metadata.update(release_id=bundle.name, run_id=run.info.run_id, rejected=reason)
        write_json(bundle / "metadata.json", metadata)
        write_json(bundle / "history.json", history)
        mlflow.log_params(
            {key: metadata[key] for key in ("epochs", "patience", "seed", "profile")}
        )
        mlflow.set_tags(
            {
                "data_version": metadata["data_version"],
                "code_sha256": metadata["code_sha256"],
                "git_commit": metadata["git_commit"] or "unavailable",
                "gate": "rejected" if reason else "passed",
                "release_id": bundle.name,
            }
        )
        if reason:
            mlflow.set_tag("gate_reason", reason)
        mlflow.log_metrics(
            {f"gate_{key}": value for key, value in metadata["scores"].items()}
        )
        mlflow.log_metrics(
            {f"baseline_{key}": value for key, value in metadata["baseline"].items()}
        )
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
        if reason:
            return metadata

        from mlflow.exceptions import MlflowException
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        try:
            client.create_registered_model(REGISTRY_NAME)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        version = client.create_model_version(
            REGISTRY_NAME,
            f"{run.info.artifact_uri}/bundle",
            run.info.run_id,
            tags={
                "data_version": metadata["data_version"],
                "release_id": bundle.name,
                "code_sha256": metadata["code_sha256"],
            },
        )
        metadata["registry_version"] = version.version
        write_json(bundle / "metadata.json", metadata)
        if shadow:
            pointers["shadow"] = bundle.name
        else:
            pointers.update(active=bundle.name, previous=active, shadow=None)
        write_json(root / "releases.json", pointers)
        return metadata


def promote(root: Path) -> str:
    with release_lock(root):
        pointers = releases(root)
        candidate, active = pointers["shadow"], pointers["active"]
        if not candidate:
            raise ValueError("there is no shadow release")
        metadata = metadata_for(root, candidate)
        validate_weights(bundle_path(root, candidate) / "weights.pt", metadata)
        champion = metadata_for(root, active) if active else None
        reason = decision(
            metadata["scores"], metadata["baseline"], champion, metadata["data_version"]
        )
        if reason:
            raise ValueError(reason)
        write_json(
            root / "releases.json",
            {"active": candidate, "previous": active, "shadow": None},
        )
        return candidate


def rollback(root: Path) -> str:
    with release_lock(root):
        pointers = releases(root)
        previous = pointers["previous"]
        if not previous:
            raise ValueError("there is no previous release")
        metadata = metadata_for(root, previous)
        validate_weights(bundle_path(root, previous) / "weights.pt", metadata)
        write_json(
            root / "releases.json",
            {"active": previous, "previous": pointers["active"], "shadow": None},
        )
        return previous


def export(root: Path, output: Path) -> str:
    active = releases(root)["active"]
    if not active:
        raise ValueError("there is no active release")
    metadata = metadata_for(root, active)
    if metadata["profile"] == "fixture":
        print("Exporting a synthetic demonstration release.")
    bundle = bundle_path(root, active)
    validate_weights(bundle / "weights.pt", metadata)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "release"
        shutil.copytree(bundle, bundle_path(staged, active))
        write_json(
            staged / "releases.json",
            {"active": active, "previous": None, "shadow": None},
        )
        staged.rename(output)
    return active
