# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from relax.engine.lora.snapshot import AdapterConflictError, snapshot_adapter


BASE_DIGEST = "a" * 64


def _export(path: Path, weights: bytes = b"fixture tensor bytes") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    # These CPU tests exercise artifact handling, not the safetensors parser.
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def test_snapshot_adapter_seals_completed_export_and_normalizes_config(tmp_path):
    source = _export(tmp_path / "export")
    first = snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    (source / "adapter_config.json").write_text('{"lora_alpha":8,"r":4}')
    assert snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST) == first
    second = snapshot_adapter(source, tmp_path / "store", version_id="B", base_model_digest=BASE_DIGEST)
    assert first.digest == second.digest
    assert first.lora_path != second.lora_path
    (source / "adapter_model.safetensors").write_bytes(b"training moved on")
    first.verify()
    assert (first.path / "adapter_model.safetensors").read_bytes() == b"fixture tensor bytes"
    assert not list((tmp_path / "store").glob(".adapter-stage-*"))


@pytest.mark.parametrize("change", ["weights", "config", "base"])
def test_snapshot_adapter_rejects_same_id_different_content(tmp_path, change):
    source = _export(tmp_path / "export")
    first = snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    base = BASE_DIGEST
    if change == "weights":
        (source / "adapter_model.safetensors").write_bytes(b"different")
    elif change == "config":
        (source / "adapter_config.json").write_text('{"r": 8}')
    else:
        base = "b" * 64
    with pytest.raises(AdapterConflictError):
        snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=base)
    first.verify()


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "adapter_config.json", "manifest.json"])
def test_snapshot_adapter_detects_tampering(tmp_path, name):
    artifact = snapshot_adapter(
        _export(tmp_path / "export"), tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST
    )
    target = artifact.path / name
    target.chmod(0o644)
    target.write_bytes(b"{}")
    with pytest.raises(ValueError, match="checksum"):
        artifact.verify()


@pytest.mark.parametrize("version_id", ["", "..", "../A", "/A", "a/b", "a" * 129])
def test_snapshot_adapter_rejects_invalid_version_id(tmp_path, version_id):
    with pytest.raises(ValueError, match="version_id"):
        snapshot_adapter(tmp_path, tmp_path / "store", version_id=version_id, base_model_digest=BASE_DIGEST)


def test_snapshot_adapter_duplicate_concurrent_writers_do_not_replace_artifact(tmp_path):
    source = _export(tmp_path / "export")

    def seal(_):
        return snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(seal, range(2)))
    assert first == second
    first.verify()
    assert not list((tmp_path / "store").glob(".adapter-stage-*"))


def test_snapshot_adapter_rejects_symlink_source(tmp_path):
    source = _export(tmp_path / "export")
    weights = source / "adapter_model.safetensors"
    weights.rename(source / "other")
    weights.symlink_to(source / "other")
    with pytest.raises(ValueError, match="regular file"):
        snapshot_adapter(source, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)


def test_snapshot_adapter_accepts_existing_relax_exporter(tmp_path):
    torch = pytest.importorskip("torch", reason="existing adapter exporter requires PyTorch")
    safetensors = pytest.importorskip("safetensors.torch")
    from relax.utils.megatron_peft_utils import write_hf_peft_adapter

    weights = {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(4, 8)}
    export_dir = write_hf_peft_adapter(
        weights, tmp_path / "export", lora_rank=4, lora_alpha=8, target_modules=["q_proj"], lora_dropout=0.0
    )
    artifact = snapshot_adapter(export_dir, tmp_path / "store", version_id="A", base_model_digest=BASE_DIGEST)
    artifact.verify()
    restored = safetensors.load_file(artifact.path / "adapter_model.safetensors")
    assert torch.equal(restored[next(iter(weights))], weights[next(iter(weights))])
