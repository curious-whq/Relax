# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-writer durable control journal; filesystem identity survives restart.

Writes are ordered on one dedicated thread. A barrier MUST precede external
side effects and successful acknowledgements. Reads are used at startup or for
indexed idempotency lookups, never to scan request history on the event loop.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4


class ControlJournal:
    def __init__(self, directory: Path, *, max_bytes: int = 512 * 1024**2, result_bytes: int = 64 * 1024**2) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = (directory / "owner.lock").open("a+")
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._db = sqlite3.connect(directory / "control.sqlite3", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        page_size = self._db.execute("PRAGMA page_size").fetchone()[0]
        if max_bytes < page_size * 32 or result_bytes < 1:
            raise ValueError("journal and result quotas must be positive and sufficient for metadata")
        self._db.execute(f"PRAGMA max_page_count={max_bytes // page_size}")
        self.result_bytes = result_bytes
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS records (kind TEXT, key TEXT, active INTEGER, body TEXT, PRIMARY KEY(kind,key))"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS active_records ON records(kind,active)")
        self._db.commit()
        self._result_bytes = self._db.execute(
            "SELECT COALESCE(SUM(length(body)),0) FROM records WHERE kind='result' AND active=1"
        ).fetchone()[0]
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lora-journal")
        self._tail: Future | None = None
        self._error: BaseException | None = None
        existing = self.get("meta", "owner")
        self.owner_id = existing["id"] if existing else uuid4().hex
        if existing is None:
            self.put("meta", "owner", {"id": self.owner_id})
            self._tail.result()

    def put(self, kind: str, key: str, body: dict[str, Any], *, active: bool = True) -> None:
        self.put_many([(kind, key, body, active)])

    def put_many(self, records: list[tuple[str, str, dict[str, Any], bool]]) -> None:
        encoded = [
            (kind, key, int(active), json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False))
            for kind, key, body, active in records
        ]

        def write() -> None:
            if self._error is not None:
                raise RuntimeError("control journal is unavailable") from self._error
            try:
                with self._db:
                    self._db.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?)", encoded)
            except BaseException as error:
                self._error = error
                raise

        self._tail = self._worker.submit(write)

    def put_result(
        self,
        key: str,
        digest: str,
        result: dict[str, Any],
        *,
        terminal_records: list[tuple[str, str, dict[str, Any], bool]] = (),
    ) -> None:
        body = json.dumps({"digest": digest, "result": result}, sort_keys=True, separators=(",", ":"), allow_nan=False)

        terminal_rows = [
            (kind, identity, int(active), json.dumps(value)) for kind, identity, value, active in terminal_records
        ]

        def write() -> None:
            if self._error is not None:
                raise RuntimeError("control journal is unavailable") from self._error
            try:
                with self._db:
                    self._db.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?)", terminal_rows)
                    previous = self._db.execute(
                        "SELECT length(body) FROM records WHERE kind='result' AND key=? AND active=1", (key,)
                    ).fetchone()
                    self._result_bytes -= previous[0] if previous else 0
                    self._db.execute("INSERT OR REPLACE INTO records VALUES ('result',?,1,?)", (key, body))
                    self._result_bytes += len(body)
                    while self._result_bytes > self.result_bytes:
                        row = self._db.execute(
                            "SELECT key,body FROM records WHERE kind='result' AND active=1 ORDER BY rowid LIMIT 1"
                        ).fetchone()
                        old = json.loads(row[1])
                        expired = json.dumps(
                            {"digest": old["digest"], "result": {"status": "expired", "code": "RESULT_EXPIRED"}}
                        )
                        self._db.execute(
                            "UPDATE records SET active=0,body=? WHERE kind='result' AND key=?", (expired, row[0])
                        )
                        self._result_bytes -= len(row[1])
            except BaseException as error:
                self._error = error
                raise

        self._tail = self._worker.submit(write)

    async def barrier(self) -> None:
        if self._tail is not None:
            await asyncio.shield(asyncio.wrap_future(self._tail))
        if self._error is not None:
            raise RuntimeError("control journal failed; admissions are fenced") from self._error

    def get(self, kind: str, key: str) -> dict[str, Any] | None:
        def read():
            row = self._db.execute("SELECT body FROM records WHERE kind=? AND key=?", (kind, key)).fetchone()
            return json.loads(row[0]) if row else None

        return self._worker.submit(read).result()

    def records(self, kind: str, *, active: bool = True) -> list[tuple[str, dict[str, Any]]]:
        def read():
            return [
                (key, json.loads(body))
                for key, body in self._db.execute(
                    "SELECT key,body FROM records WHERE kind=? AND active=?", (kind, int(active))
                )
            ]

        return self._worker.submit(read).result()

    def close(self) -> None:
        self._worker.shutdown(wait=True)
        self._db.close()
        self._lock.close()
