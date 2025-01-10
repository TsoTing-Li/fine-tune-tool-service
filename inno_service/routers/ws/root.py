import json
import os

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from inno_service.routers.ws import utils
from inno_service.thirdparty.docker import api_handler
from inno_service.utils.logger import accel_logger

router = APIRouter(prefix="/ws")


@router.websocket("/trainLogs/{id}")
async def train_log(websocket: WebSocket, id: str):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            skip_eval_process_bar = False
            async for log in api_handler.get_container_log(
                aclient=aclient, container_name_or_id=id
            ):
                for log_split in log.splitlines():
                    if log_split == "":
                        break
                    elif log_split[0] in ("\x01", "\x02"):
                        log_split = log_split[8:]

                    train_log = utils.parse_train_log(
                        stdout=log_split.strip(),
                        exclude_flag=skip_eval_process_bar,
                    )

                    if "[00:00<?, ?it/s]" in train_log["train_progress"]:
                        skip_eval_process_bar = True

                    accel_logger.info(f"trainLog: {json.dumps(train_log)}")
                    await websocket.send_json({"trainLog": train_log})
        await websocket.send_json({"trainLog": "train finish"})

    except WebSocketDisconnect:
        accel_logger.info("trainLog: Client disconnected")

    except Exception as e:
        accel_logger.error(f"trainLog: Unexpected error: {e}")

    finally:
        await websocket.close()


@router.websocket("/quantizeLogs/{id}")
async def quantize_log(websocket: WebSocket, id: str):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            async for log in api_handler.get_container_log(
                aclient=aclient, container_name_or_id=id
            ):
                for log_split in log.splitlines():
                    if log_split == "":
                        break
                    elif log_split[0] in ("\x01", "\x02"):
                        log_split = log_split[8:]

                    accel_logger.info(f"mergeLog: {log_split}")
                    await websocket.send_json({"mergeLog": log_split})
        await websocket.send_json({"quantizeLog": "quantize finish"})

    except WebSocketDisconnect:
        accel_logger.info("quantizeLog: Client disconnected")

    except Exception as e:
        accel_logger.error(f"quantizeLog: Unexpected error: {e}")

    finally:
        await websocket.close()


@router.websocket("/mergeLogs/{id}")
async def merge_log(websocket: WebSocket, id: str):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            async for log in api_handler.get_container_log(
                aclient=aclient, container_name_or_id=id
            ):
                for log_split in log.splitlines():
                    if log_split == "":
                        break
                    elif log_split[0] in ("\x01", "\x02"):
                        log_split = log_split[8:]

                    accel_logger.info(f"mergeLog: {log_split}")
                    await websocket.send_json({"mergeLog": log_split})
        await websocket.send_json({"mergeLog": "merge finish"})

    except WebSocketDisconnect:
        accel_logger.info("mergeLog: Client disconnected")

    except Exception as e:
        accel_logger.error(f"mergeLog: Unexpected error: {e}")

    finally:
        await websocket.close()


@router.websocket("/hwInfo")
async def hw_info_log(websocket: WebSocket):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            async for log in api_handler.get_container_log(
                aclient=aclient,
                container_name_or_id=os.environ["HWINFO_CONTAINER_NAME"],
            ):
                for log_split in log.splitlines():
                    if log_split == "":
                        break
                    elif log_split[0] in ("\x01", "\x02"):
                        log_split = log_split[8:]

                    hw_info = utils.parse_hw_info_log(stdout=log_split)
                    await websocket.send_json(hw_info)

    except WebSocketDisconnect:
        accel_logger.info("hwInfo: Client disconnected")

    except Exception as e:
        accel_logger.error(f"hwInfo: Unexpected error: {e}")

    finally:
        if websocket.client_state == "CONNECTED":
            accel_logger.info(
                "hwInfo: WebSocket is still connected, automatically close"
            )
            await websocket.close()
