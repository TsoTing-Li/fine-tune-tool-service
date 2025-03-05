import asyncio
import json
from collections.abc import AsyncGenerator

import httpx
from fastapi import status

from src.thirdparty.redis.handler import redis_async


async def listen_request_status(request_id: str) -> bool:
    sub = redis_async.client.pubsub()
    await sub.subscribe("chat_requests")

    async for message in sub.listen():
        if message["type"] == "message":
            req_id, request_status = message["data"].split(":")
            if req_id == request_id and request_status == "cancelled":
                return True


async def post_openai_chat(
    request_id: str,
    model_service: str,
    model_name: str,
    messages: list,
) -> AsyncGenerator[dict, None]:
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": msg} for msg in messages],
        "stream": True,
    }

    try:
        async with httpx.AsyncClient(timeout=None) as aclient:
            async with aclient.stream(
                "POST", f"{model_service}/v1/chat/completions", json=data
            ) as response:
                if response.status_code != status.HTTP_200_OK:
                    await redis_async.client.hdel("chat_requests", request_id)
                    raise RuntimeError(
                        f"Error: {response.status_code}, {response.text}"
                    ) from None

                listen_task = asyncio.create_task(
                    listen_request_status(request_id=request_id)
                )

                async for chunk in response.aiter_lines():
                    if listen_task.done() and listen_task.result():
                        break

                    if chunk:
                        data_chunk = json.loads(chunk[6:])

                        if data_chunk["choices"][0]["finish_reason"] == "stop":
                            break

                        yield json.dumps(
                            {
                                "id": request_id,
                                "content": data_chunk["choices"][0]["delta"]["content"],
                            }
                        )

    finally:
        await redis_async.client.hdel("chat_requests", request_id)
        listen_task.cancel()
