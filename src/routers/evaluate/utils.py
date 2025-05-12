import itertools
import os
from typing import Dict, Literal, Union

import aiofiles
import httpx
import orjson

from src.config.params import COMMON_CONFIG, STATUS_CONFIG, TASK_CONFIG
from src.routers.evaluate import validator
from src.thirdparty.docker.api_handler import (
    create_container,
    start_container,
    stop_container,
    wait_for_container,
)
from src.thirdparty.redis.handler import redis_async
from src.utils.logger import accel_logger


async def run_lm_eval(
    image_name: str, cmd: list, docker_network_name: str, eval_name: str
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    env_var = [f"HF_HOME={COMMON_CONFIG.hf_home}"]
    data = {
        "User": "root",
        "Image": image_name,
        "HostConfig": {
            "IpcMode": "host",
            "Binds": [
                f"{COMMON_CONFIG.hf_home}:{COMMON_CONFIG.hf_home}:rw",
                f"{COMMON_CONFIG.root_path}/saves:{COMMON_CONFIG.save_path}:rw",
            ],
            "NetworkMode": docker_network_name,
        },
        "Tty": True,
        "Cmd": cmd,
        "Env": env_var,
    }

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            container_name_or_id = await create_container(
                aclient=aclient, name=f"eval-{eval_name}", data=data
            )

            started_container = await start_container(
                aclient=aclient, container_name_or_id=container_name_or_id
            )

            return started_container

    except Exception as e:
        raise RuntimeError(f"{e}") from None


def get_eval_result_path(root_path: str) -> Union[str, None]:
    with os.scandir(root_path) as entries:
        files = [
            entry.path
            for entry in entries
            if entry.is_file() and entry.name.endswith(".json")
        ]

    if len(files) == 0:
        return
    else:
        return files[0]


async def start_eval_background_task(eval_name: str, container_name_or_id: str) -> None:
    try:
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            container_info = await wait_for_container(
                aclient=aclient, container_name=container_name_or_id
            )
            exit_status = container_info["StatusCode"]
            if exit_status == 0:
                eval_status = STATUS_CONFIG.finish
            elif exit_status == 1:
                eval_status = STATUS_CONFIG.failed
            else:
                eval_status = None

    except ValueError as e:
        eval_status = STATUS_CONFIG.failed
        accel_logger.error(f"Docker error: {e}")

    except RuntimeError as e:
        eval_status = STATUS_CONFIG.failed
        accel_logger.error(f"Docker error: {e}")

    except Exception as e:
        eval_status = STATUS_CONFIG.failed
        accel_logger.error(f"Unexpected error: {e}")

    finally:
        try:
            if eval_status in {STATUS_CONFIG.finish, STATUS_CONFIG.failed}:
                info = await redis_async.client.hget(TASK_CONFIG.train, eval_name)
                info = orjson.loads(info)

                info["container"]["eval"]["status"] = eval_status
                info["container"]["eval"]["id"] = None

                eval_result_path = get_eval_result_path(
                    root_path=os.path.join(
                        os.path.dirname(info["train_args"]["output_dir"]),
                        "evaluate",
                        eval_name,
                    )
                )
                info["eval_result_path"] = eval_result_path

                await redis_async.client.hset(
                    TASK_CONFIG.train, eval_name, orjson.dumps(info)
                )
        except Exception as e:
            accel_logger.error(f"Database error: {e}")


async def stop_eval(
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        stopped_eval_container = await stop_container(
            aclient=aclient,
            container_name_or_id=container_name_or_id,
            signal=signal,
            wait_sec=wait_sec,
        )

    return stopped_eval_container


async def eval_finish_event(path: str) -> validator.EvalResult:
    output = validator.EvalResult()

    if os.path.exists(path):
        async with aiofiles.open(path) as f:
            content = await f.read()
        data = orjson.loads(content)
        eval_results: Dict[str, dict] = data["results"]
        eval_configs: Dict[str, dict] = data["configs"]

        for task, configs in eval_configs.items():
            filter_list = configs.get("filter_list", [{"name": "none"}])
            metric_list = configs.get("metric_list", [{"metric": "none"}])
            for filter_config, metric_config in itertools.product(
                filter_list, metric_list
            ):
                metric = metric_config["metric"]
                filter = filter_config["name"]
                task_info = validator.TaskInfo(
                    name=configs["task"],
                    filter=filter,
                    n_shot=configs["num_fewshot"],
                    metric=metric,
                    value=eval_results[task][f"{metric},{filter}"],
                    stderr=eval_results[task][f"{metric}_stderr,{filter}"],
                )
                output.task_info.append(task_info)

    validator.EvalResult.model_validate(output)

    return output


async def get_eval_result(info: dict) -> dict:
    status = info["container"]["eval"]["status"]
    if status == STATUS_CONFIG.finish:
        eval_result = await eval_finish_event(path=info["eval_result_path"])

    return eval_result.model_dump()
