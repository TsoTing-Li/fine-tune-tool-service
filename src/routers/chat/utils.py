import json
from collections.abc import AsyncGenerator

import httpx
from fastapi import status


async def post_openai_chat(
    request_id: str,
    model_service: str,
    model_name: str,
    messages: list,
    active_requests: dict,
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
                    active_requests.pop(request_id, None)
                    raise RuntimeError(
                        f"Error: {response.status_code}, {response.text}"
                    ) from None

                async for chunk in response.aiter_lines():
                    if active_requests.get(request_id) == "cancelled":
                        print(f"Request {request_id} was cancelled")
                        break

                    if chunk:
                        data_chunk = json.loads(chunk[6:])

                        if data_chunk["choices"][0]["finish_reason"] == "stop":
                            print("DONE")
                            break

                        yield json.dumps(
                            {
                                "id": request_id,
                                "content": data_chunk["choices"][0]["delta"]["content"],
                            }
                        )

    finally:
        active_requests.pop(request_id, None)
