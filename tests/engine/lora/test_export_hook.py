# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from relax.engine.lora import cli


def test_export_hook_queues_bounded_work_and_flushes(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def publish(url, path, version):
        calls.append((url, path, version))
        started.set()
        assert release.wait(5)
        return {"version_id": version, "digest": "a" * 64}

    monkeypatch.setenv("RELAX_LORA_PUBLICATION_URL", "http://gateway")
    monkeypatch.setenv("RELAX_LORA_VERSION_PREFIX", "experiment")
    monkeypatch.setattr(cli, "publish_export", publish)
    monkeypatch.setattr(
        cli, "seal_export", lambda export, model, store, version, **kw: SimpleNamespace(path=store / version)
    )
    monkeypatch.setattr(cli, "_export_futures", [])
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(cli, "_export_pool", pool)
        try:
            cli.export_hook(SimpleNamespace(hf_checkpoint=str(tmp_path)), str(tmp_path), 1, dtype="bf16", is_lora=True)
            assert started.wait(2)
            cli.export_hook(SimpleNamespace(hf_checkpoint=str(tmp_path)), str(tmp_path), 2, dtype="bf16", is_lora=True)
            with pytest.raises(RuntimeError, match="already await publication"):
                cli.export_hook(
                    SimpleNamespace(hf_checkpoint=str(tmp_path)), str(tmp_path), 3, dtype="bf16", is_lora=True
                )
        finally:
            release.set()
        cli.flush()
    assert [item[2] for item in calls] == ["experiment-1", "experiment-2"]


def test_export_hook_rejects_full_model_export():
    with pytest.raises(ValueError, match="adapter export"):
        cli.export_hook(None, "unused", 1, dtype="bf16", is_lora=False)
