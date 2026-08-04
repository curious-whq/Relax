# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from collections.abc import Awaitable, Callable


class OrderedAsyncTaskChain:
    """Run submitted async operations in FIFO order without blocking
    submitters."""

    def __init__(self) -> None:
        self._tail: asyncio.Task[None] | None = None

    def submit(self, operation: Callable[[], Awaitable[None]]) -> asyncio.Task[None]:
        previous = self._tail

        async def run_in_order() -> None:
            if previous is not None:
                await previous
            await operation()

        task = asyncio.create_task(run_in_order())
        self._tail = task
        return task
