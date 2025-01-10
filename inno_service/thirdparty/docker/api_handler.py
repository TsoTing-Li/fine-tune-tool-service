from typing import AsyncGenerator

import httpx


async def get_container_log(
    aclient: httpx.AsyncClient, container_name_or_id: str
) -> AsyncGenerator[str, None]:
    params = {
        "id": container_name_or_id,
        "follow": True,
        "stdout": True,
        "stderr": True,
    }
    async with aclient.stream(
        "GET", f"http://docker/containers/{container_name_or_id}/logs", params=params
    ) as r:
        async for chunk in r.aiter_text():
            yield chunk
