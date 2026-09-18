# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Publish existing Relax HF adapter exports and create numerical test
fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import httpx

from relax.engine.lora.artifact import PROVENANCE, ModelContract, canonical, provenance, read_object
from relax.engine.lora.snapshot import AdapterSnapshot, fingerprint_model, snapshot_adapter
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_export_pool: ThreadPoolExecutor | None = None
_export_futures: list[Future] = []
_export_lock = Lock()


def publish_export(gateway_url: str, export_dir: str | Path, version_id: str) -> dict[str, Any]:
    path = Path(export_dir).resolve()
    if (path / "lora_adapter").is_dir():
        path /= "lora_adapter"
    from relax.engine.lora.access import headers

    with httpx.Client(
        timeout=httpx.Timeout(300.0, connect=10.0), trust_env=False, headers=headers(control=True)
    ) as client:
        response = client.post(
            f"{gateway_url.rstrip('/')}/publish", json={"export_dir": str(path), "version_id": version_id}
        )
        response.raise_for_status()
        return response.json()


def export_hook(args: Any, hf_path: str, rollout_id: int, *, dtype: str, is_lora: bool) -> None:
    """Existing --save-hf-post-hook-path entry; publish a completed adapter
    export.

    The training job's normal rollout engines must be separate from the
    gateway's dedicated frozen-base engines. The existing save hook logs
    failures and keeps training; an unsuccessful publication leaves the gateway
    default unchanged.
    """
    if not is_lora:
        raise ValueError("immutable publication requires an adapter export")
    url = Envs.RELAX_LORA_PUBLICATION_URL
    prefix = Envs.RELAX_LORA_VERSION_PREFIX
    if not url or not prefix:
        raise ValueError("RELAX_LORA_PUBLICATION_URL and RELAX_LORA_VERSION_PREFIX are required")
    global _export_pool
    with _export_lock:
        _export_futures[:] = [future for future in _export_futures if not future.done()]
        if len(_export_futures) >= 2:
            raise RuntimeError("two adapter exports already await publication; retry this completed export explicitly")
        if _export_pool is None:
            _export_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lora-publish")
        # Seal before enqueue: the background publisher never borrows a mutable
        # training export directory. Backpressure is checked before the copy.
        path = Path(hf_path)
        if (path / "lora_adapter").is_dir():
            path /= "lora_adapter"
        version = f"{prefix}-{rollout_id}"
        sealed = seal_export(
            path,
            Path(args.hf_checkpoint),
            path.parent / ".publication-artifacts",
            version,
            producer="relax.save_hf_post_hook",
            export_step=rollout_id,
        )
        future = _export_pool.submit(publish_export, url, sealed.path, version)
        _export_futures.append(future)
    future.add_done_callback(_report_export_result)


def _report_export_result(future: Future) -> None:
    try:
        result = future.result()
        logger.info("Published immutable LoRA %s (%s)", result["version_id"], result["digest"])
    except Exception:
        logger.exception("Immutable LoRA publication failed; inspect gateway state before retrying the export")


def flush() -> None:
    """The existing final-save hook waits for pending publications here."""
    with _export_lock:
        pending = list(_export_futures)
    errors = []
    for future in pending:
        try:
            future.result()
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(f"{len(errors)} pending adapter publications failed") from errors[0]


@lru_cache(maxsize=4)
def producer_base_digest(model: Path) -> str:
    """The export job's explicitly frozen base is fingerprinted once."""
    return fingerprint_model(model)


def seal_export(
    export: Path, model: Path, store: Path, version: str, *, producer: str, export_step: int
) -> AdapterSnapshot:
    """Producer-owned synchronous handoff; source may be cleaned after
    return."""
    source_bytes = sum(path.stat().st_size for path in export.iterdir() if path.is_file())
    retained_bytes = sum(path.stat().st_size for path in store.glob("*/*") if path.is_file())
    if source_bytes + retained_bytes > 8 * 1024**3:
        raise RuntimeError("producer sealed-artifact quota exhausted; archive acknowledged exports before retrying")
    config = read_object(export / "adapter_config.json")
    contract = ModelContract(
        producer_base_digest(model.resolve()),
        read_object(model / "config.json"),
        config["r"],
        tuple(config["target_modules"]),
    )
    manifest = provenance(contract, export, producer=producer, export_step=export_step)
    (export / PROVENANCE).write_bytes(canonical(manifest))
    return snapshot_adapter(export, store, version_id=version, base_model_digest=contract.base_digest)


def make_fixtures(model_path: Path, output: Path, rank: int = 8) -> None:
    import torch

    from relax.utils.megatron_peft_utils import write_hf_peft_adapter

    config = json.loads((model_path / "config.json").read_text())
    if config.get("model_type") not in {"qwen2", "qwen3", "llama"}:
        raise ValueError("fixture generator supports dense Qwen2/Qwen3/Llama text models")
    if config.get("num_experts") or config.get("num_local_experts"):
        raise ValueError("use a dense model for the fixture experiment")
    hidden = config["hidden_size"]
    head_dim = config.get("head_dim", hidden // config["num_attention_heads"])
    widths = {
        "q_proj": config["num_attention_heads"] * head_dim,
        "v_proj": config.get("num_key_value_heads", config["num_attention_heads"]) * head_dim,
    }
    generator = torch.Generator().manual_seed(1707)
    weights = {}
    for layer in range(config["num_hidden_layers"]):
        for module, width in widths.items():
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden, generator=generator) * 0.1
            weights[f"{prefix}.lora_B.weight"] = torch.randn(width, rank, generator=generator) * 0.1
    for name, sign in (("A", 1), ("B", -1)):
        variant = {key: value * sign if ".lora_B." in key else value for key, value in weights.items()}
        write_hf_peft_adapter(
            variant,
            output / name,
            lora_rank=rank,
            lora_alpha=rank * 2,
            target_modules=list(widths),
            lora_dropout=0.0,
        )
        contract = ModelContract(fingerprint_model(model_path), config, rank, tuple(widths))
        (output / name / PROVENANCE).write_bytes(
            canonical(
                provenance(
                    contract,
                    output / name,
                    producer="relax.fixture",
                    export_step=0,
                )
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--gateway-url", required=True)
    publish.add_argument("--export-dir", type=Path, required=True)
    publish.add_argument("--version-id", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--export-dir", type=Path, required=True)
    seal.add_argument("--model-path", type=Path, required=True)
    seal.add_argument("--store", type=Path, required=True)
    seal.add_argument("--version-id", required=True)
    seal.add_argument("--export-step", type=int, required=True)
    seal.add_argument("--producer", required=True)
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("--model-path", type=Path, required=True)
    fixtures = commands.add_parser("fixtures")
    fixtures.add_argument("--model-path", type=Path, required=True)
    fixtures.add_argument("--output", type=Path, required=True)
    fixtures.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    if args.command == "publish":
        result = publish_export(args.gateway_url, args.export_dir, args.version_id)
    elif args.command == "seal":
        artifact = seal_export(
            args.export_dir,
            args.model_path,
            args.store,
            args.version_id,
            producer=args.producer,
            export_step=args.export_step,
        )
        result = {"path": str(artifact.path), "digest": artifact.digest}
    elif args.command == "fingerprint":
        result = {"base_model_digest": fingerprint_model(args.model_path)}
    else:
        make_fixtures(args.model_path, args.output, args.rank)
        result = {"fixtures": str(args.output.resolve())}
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
