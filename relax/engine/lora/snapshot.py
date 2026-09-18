# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Snapshot completed HF adapter exports without importing training
dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DIGEST_PATTERN = re.compile(r"[a-f0-9]{64}\Z")


class AdapterConflictError(ValueError):
    """An immutable version ID was reused with different content."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_model(directory: str | Path) -> str:
    """Hash a frozen local HF base model, including config/tokenizer/weight
    files."""
    root = Path(directory).resolve()
    suffixes = {".json", ".safetensors", ".bin", ".model", ".tiktoken", ".txt", ".py"}
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix in suffixes)
    if not (root / "config.json").is_file() or not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError("base-model directory must contain config.json and model weights")
    return hashlib.sha256(_canonical_json({path.name: _file_digest(path) for path in files})).hexdigest()


def _snapshot_files(directory: Path) -> tuple[str, ...]:
    # Legacy byte-only snapshots remain readable by low-level tooling. Production
    # ModelContract validation requires this producer-signed-in-context manifest.
    provenance = "producer_manifest.json"
    return (*_ADAPTER_FILES, provenance) if (directory / provenance).exists() else _ADAPTER_FILES


def _manifest(directory: Path, version_id: str, base_model_digest: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "version_id": version_id,
        "base_model_digest": base_model_digest,
        "files": {name: _file_digest(directory / name) for name in _snapshot_files(directory)},
    }


def _content_digest(manifest: dict[str, Any]) -> str:
    # Version IDs identify publications, not content. Identical adapters may be
    # published under distinct IDs while retaining the same content checksum.
    content = {key: value for key, value in manifest.items() if key != "version_id"}
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _sync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class AdapterSnapshot:
    """A sealed export; engines must verify its bytes before acknowledging
    load."""

    version_id: str
    digest: str
    base_model_digest: str
    path: Path

    def __post_init__(self) -> None:
        if not _VERSION_PATTERN.fullmatch(self.version_id):
            raise ValueError("version_id must be a nonempty, path-safe identifier of at most 128 characters")
        if not _DIGEST_PATTERN.fullmatch(self.digest) or not _DIGEST_PATTERN.fullmatch(self.base_model_digest):
            raise ValueError("adapter and base-model digests must be lowercase SHA-256 hex strings")

    @property
    def lora_path(self) -> str:
        """Engine registration name, also used in every generate request."""
        return f"relax_policy@{self.version_id}"

    def verify(self) -> None:
        """Reject damaged or changed artifacts, including their manifest."""
        for name in (*_snapshot_files(self.path), "manifest.json"):
            path = self.path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"adapter snapshot requires a regular file: {name}")
        expected = _manifest(self.path, self.version_id, self.base_model_digest)
        actual = json.loads((self.path / "manifest.json").read_bytes())
        if actual != expected or _content_digest(expected) != self.digest:
            raise ValueError(f"adapter snapshot checksum mismatch: {self.version_id}")


def snapshot_adapter(
    export_dir: str | Path,
    store_dir: str | Path,
    *,
    version_id: str,
    base_model_digest: str,
) -> AdapterSnapshot:
    """Seal a completed export from ``write_hf_peft_adapter``.

    The producer must finish its consistent training-step export before calling
    this function and leave the source unchanged until it returns. Copying a live
    training directory cannot establish a consistent training-step snapshot.

    Identity covers the base model, canonical config and exact safetensors bytes.
    A temporary sibling directory is renamed into place without replacing any
    existing version. Repeating the same ID/content returns the existing snapshot.
    """
    # Validate identifiers before using them as paths.
    AdapterSnapshot(version_id, "0" * 64, base_model_digest, Path(store_dir))
    source = Path(export_dir)
    store = Path(store_dir).resolve()
    store.mkdir(parents=True, exist_ok=True)
    destination = store / version_id
    with tempfile.TemporaryDirectory(prefix=".adapter-stage-", dir=store) as temporary:
        stage = Path(temporary)
        for name in _snapshot_files(source):
            path = source / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"adapter export requires a regular file: {name}")
        config = json.loads((source / "adapter_config.json").read_bytes())
        if not isinstance(config, dict):
            raise ValueError("adapter_config.json must contain an object")
        (stage / "adapter_config.json").write_bytes(_canonical_json(config))
        shutil.copyfile(source / "adapter_model.safetensors", stage / "adapter_model.safetensors")
        if "producer_manifest.json" in _snapshot_files(source):
            shutil.copyfile(source / "producer_manifest.json", stage / "producer_manifest.json")
        manifest = _manifest(stage, version_id, base_model_digest)
        digest = _content_digest(manifest)
        (stage / "manifest.json").write_bytes(_canonical_json(manifest))
        snapshot = AdapterSnapshot(version_id, digest, base_model_digest, destination)
        for name in (*_snapshot_files(stage), "manifest.json"):
            (stage / name).chmod(0o444)
            with (stage / name).open("rb") as handle:
                os.fsync(handle.fileno())
        _sync_directory(stage)
        try:
            # A completed destination is nonempty, so rename cannot overwrite it,
            # including when another publisher won the race after our copy.
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(str(destination))
            os.rename(stage, destination)
        except OSError:
            if not destination.is_dir() or destination.is_symlink():
                raise
            existing_manifest = json.loads((destination / "manifest.json").read_bytes())
            if existing_manifest != manifest:
                raise AdapterConflictError(f"adapter version {version_id!r} already has different content") from None
            snapshot.verify()
        # Also repair durability on a retry whose predecessor renamed the
        # snapshot but failed before synchronizing its parent directory.
        _sync_directory(store)
        return snapshot
