import anyio
from fastapi import APIRouter, WebSocket

from inno_service.thirdparty.redis.handler import AsyncRedisClient

router = APIRouter(prefix="/ws")


@router.websocket("/{sub_chan}")
async def ws_pubsub_log(websocket: WebSocket, sub_chan: str):
    await websocket.accept()
    async_redis = AsyncRedisClient()

    pubsub = async_redis.client.pubsub()
    await pubsub.subscribe(sub_chan)

    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=None
            )

            if message:
                log_info = message["data"].decode()

                await websocket.send_text(log_info)

                if log_info == "FINISHED":
                    await websocket.close(code=1000)
                    break

            await anyio.sleep(0.033)

    except BaseException as e:
        await websocket.send_text(str(e))
        await websocket.close(code=1006)

    finally:
        await pubsub.unsubscribe(sub_chan)
        await pubsub.aclose()
        await async_redis.client.aclose()
