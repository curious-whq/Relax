# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

from relax.engine.rollout.ordered_publisher import OrderedAsyncTaskChain


async def test_ordered_async_task_chain_preserves_submission_order() -> None:
    chain = OrderedAsyncTaskChain()
    release_first = asyncio.Event()
    events = []

    async def first() -> None:
        events.append("first-start")
        await release_first.wait()
        events.append("first-end")

    async def second() -> None:
        events.append("second")

    first_task = chain.submit(first)
    second_task = chain.submit(second)
    await asyncio.sleep(0)

    assert events == ["first-start"]
    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert events == ["first-start", "first-end", "second"]


async def test_ordered_async_task_chain_propagates_previous_failure() -> None:
    chain = OrderedAsyncTaskChain()
    second_called = False

    async def first() -> None:
        raise RuntimeError("transfer failed")

    async def second() -> None:
        nonlocal second_called
        second_called = True

    first_task = chain.submit(first)
    second_task = chain.submit(second)
    results = await asyncio.gather(first_task, second_task, return_exceptions=True)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert second_called is False
