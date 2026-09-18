# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Durable close responsibility shared by Agentic shards and the gateway.

The directory must reside on the explicitly configured shared persistent mount.
Files are pending work, not TTL leases: only a durable close ACK removes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from relax.engine.lora.snapshot import _sync_directory


class SessionCloseOutbox:
    def __init__(self, directory: Path, gateway_url: str) -> None:
        self.directory = directory
        self.gateway_url = gateway_url.rstrip("/")
        directory.mkdir(parents=True, exist_ok=True)
        identity = directory / ".spool-id"
        try:
            with identity.open("x") as stream:
                stream.write(uuid4().hex)
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(directory)
        except FileExistsError:
            pass
        self.identity = identity.read_text()
        if len(self.identity) != 32:
            raise RuntimeError("close spool identity is incomplete")

    def enqueue(self, session_id: str) -> Path:
        body = {"session_id": session_id, "gateway_url": self.gateway_url}
        destination = self.directory / (
            hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() + ".json"
        )
        fd, temporary = tempfile.mkstemp(prefix=".close-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(body, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            _sync_directory(self.directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return destination

    def acknowledge(self, path: Path) -> None:
        path.unlink(missing_ok=True)
        _sync_directory(self.directory)

    def pending(self) -> list[tuple[Path, str]]:
        result = []
        for path in self.directory.glob("*.json"):
            body = json.loads(path.read_bytes())
            if body["gateway_url"] == self.gateway_url:
                result.append((path, body["session_id"]))
        return result
