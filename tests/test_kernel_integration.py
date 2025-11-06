"""Integration tests for the Tessrax governance kernel background services (v18.5)."""

from __future__ import annotations

import asyncio

from tessrax.core.governance_kernel import GovernanceKernel


def test_background_services_initialise_and_shutdown() -> None:
    """Integrity monitor and repair engine launch without raising errors."""

    async def exercise() -> None:
        kernel = GovernanceKernel(auto_integrate=False)
        await kernel.start_background_services()
        assert kernel.background_tasks, "Background services did not start"
        await asyncio.sleep(0)
        await kernel.shutdown_background_services()
        assert not kernel.background_tasks

    asyncio.run(exercise())
