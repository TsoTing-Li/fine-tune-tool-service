import os

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from inno_service.routers.ws import utils

router = APIRouter(prefix="/ws")


@router.websocket("/container/{id}/logs")
async def ws_docker_log(websocket: WebSocket, id: str):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    params = {"id": id, "follow": True, "stdout": True, "stderr": True}

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        async with aclient.stream(
            "GET", f"http://docker/containers/{id}/logs", params=params
        ) as r:
            skip_eval_process_bar = False
            async for chunk in r.aiter_text():
                for chunk_split in chunk.splitlines():
                    if chunk_split == "":
                        break
                    elif chunk_split[0] in ("\x01", "\x02"):
                        chunk_split = chunk_split[8:]

                    log_info = utils.parse_train_log(
                        stdout=chunk_split.strip(), exclude_flag=skip_eval_process_bar
                    )

                    if "[00:00<?, ?it/s]" in log_info["train_progress"]:
                        skip_eval_process_bar = True

                    await websocket.send_json(log_info)

    await websocket.close()


@router.websocket("/hw_info")
async def ws_hw_info_log(websocket: WebSocket):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    params = {
        "id": os.environ["HWINFO_CONTAINER_NAME"],
        "follow": True,
        "stdout": True,
        "stderr": True,
    }

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            async with aclient.stream(
                "GET",
                f"http://docker/containers/{os.environ['HWINFO_CONTAINER_NAME']}/logs",
                params=params,
            ) as r:
                async for chunk in r.aiter_text():
                    for chunk_split in chunk.splitlines():
                        if chunk_split == "":
                            break
                        elif chunk_split[0] in ("\x01", "\x02"):
                            chunk_split = chunk_split[8:]

                        await websocket.send_json(chunk_split.strip())

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        if websocket.client_state == "CONNECTED":
            await websocket.close()
        print("Websocket connection closed")
