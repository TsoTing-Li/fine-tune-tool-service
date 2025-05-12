import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.config.params import HWINFO_CONFIG, STATUS_CONFIG
from src.routers.ws import schema
from src.thirdparty.docker.api_handler import get_container_log, wait_for_container
from src.utils.logger import accel_logger

router = APIRouter(prefix="/ws")


@router.websocket("/trainLogs/{id}")
async def train_log(websocket: WebSocket, id: str):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            last_train_progress = None
            total_steps = 0

            train_log = schema.TrainLogTemplate()

            async for log in get_container_log(
                aclient=aclient, container_name_or_id=id
            ):
                for log_split in log.splitlines():
                    if log_split == "":
                        break
                    elif log_split[0] in ("\x01", "\x02"):
                        log_split = log_split[8:]

                    if "Total optimization steps" in log_split:
                        total_steps = train_log.get_total_steps(log=log_split)

                    train_log.parse_train_log(
                        stdout=log_split.strip(),
                        last_train_progress=last_train_progress,
                        total_steps=total_steps,
                    )

                    last_train_progress = train_log.train_progress

                    await websocket.send_json({"trainLog": train_log.model_dump()})

            container_info = await wait_for_container(
                aclient=aclient, container_name=id
            )
            exit_status = container_info["StatusCode"]
            if exit_status == 0:
                train_status = STATUS_CONFIG.finish
            elif exit_status in {137, 143}:
                train_status = STATUS_CONFIG.stopped
            elif exit_status == 1:
                train_status = STATUS_CONFIG.failed

        await websocket.send_json({"trainLog": f"train {train_status}"})

    except WebSocketDisconnect:
        accel_logger.info("trainLog: Client disconnected")

    except ValueError as e:
        accel_logger.error(f"trainLog: {e}")
        await websocket.send_json({"trainLog": f"{e}"})

    except Exception as e:
        accel_logger.error(f"trainLog: Unexpected error {e}")
        await websocket.send_json({"trainLog": "Unexpected error"})

    finally:
        await websocket.close()


@router.websocket("/hwInfo")
async def hw_info_log(websocket: WebSocket):
    await websocket.accept()
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            hw_info = schema.HwInfoTemplate()

            async for log in get_container_log(
                aclient=aclient,
                container_name_or_id=HWINFO_CONFIG.container_name,
                tail=1,
            ):
                for log_split in log.splitlines():
                    if log_split == "":
                        break
                    elif log_split[0] in ("\x01", "\x02"):
                        log_split = log_split[8:]

                    hw_info.parse_hwinfo_log(stdout=log_split)
                    await websocket.send_json(hw_info)

    except WebSocketDisconnect:
        accel_logger.info("hwInfo: Client disconnected")

    except ValueError as e:
        accel_logger.error(f"hwInfo: {e}")
        await websocket.send_json({"hwInfo": f"{e}"})

    except Exception as e:
        accel_logger.error(f"hwInfo: Unexpected error: {e}")
        await websocket.send_json({"hwInfo": f"{e}"})

    finally:
        if websocket.client_state == "CONNECTED":
            accel_logger.info(
                "hwInfo: WebSocket is still connected, automatically close"
            )
            await websocket.close()
