import asyncio
import os
import re
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Union

import aiofiles
import aiofiles.os
import httpx
import orjson
import yaml
from fastapi import HTTPException, UploadFile, status

from src.config.params import (
    COMMON_CONFIG,
    MAINSERVICE_CONFIG,
    STATUS_CONFIG,
    TASK_CONFIG,
)
from src.routers.train import schema, validator
from src.thirdparty.docker.api_handler import (
    create_container,
    get_container_log,
    remove_container,
    start_container,
    stop_container,
    wait_for_container,
)
from src.thirdparty.redis.handler import redis_async
from src.utils.logger import accel_logger


def basemodel2dict(data) -> dict:
    train_args = {
        "base_model": data.base_model,
        **data.method.model_dump(),
        **data.dataset.model_dump(),
        **data.output.model_dump(),
        **data.params.model_dump(),
        **data.val.model_dump(),
    }

    return train_args


def file_train_args_process(
    train_name: str,
    train_args: dict,
    lora_args: Union[dict, None],
    save_path: str,
    dataset_path: str,
) -> dict:
    args = train_args.copy()
    args["model_name_or_path"] = args.pop("base_model")
    args[args.pop("compute_type")] = True
    args["output_dir"] = os.path.join(save_path, train_name, args["finetuning_type"])
    args["dataset"] = ", ".join(args["dataset"])
    args["dataset_dir"] = dataset_path
    args["eval_steps"] = args["save_steps"]
    args["do_train"] = True

    if args["finetuning_type"] == "lora" and isinstance(lora_args, dict):
        args["lora_alpha"] = lora_args["lora_alpha"]
        args["lora_dropout"] = lora_args["lora_dropout"]
        args["lora_rank"] = lora_args["lora_rank"]
        args["lora_target"] = (
            "all"
            if "all" in (targets := lora_args.get("lora_target", []))
            else ", ".join(targets)
        )

    return args


def redis_train_args_process(
    train_name: str,
    train_args: dict,
    lora_args: Union[dict, None],
    save_path: str,
    dataset_path: str,
) -> dict:
    args = train_args.copy()
    args["output_dir"] = os.path.join(save_path, train_name, args["finetuning_type"])
    args["dataset_dir"] = dataset_path
    args["eval_steps"] = args["save_steps"]
    args["do_train"] = True

    if args["finetuning_type"] == "lora" and isinstance(lora_args, dict):
        args["lora_alpha"] = lora_args["lora_alpha"]
        args["lora_dropout"] = lora_args["lora_dropout"]
        args["lora_rank"] = lora_args["lora_rank"]
        args["lora_target"] = (
            ["all"]
            if "all" in (targets := lora_args.get("lora_target", []))
            else targets
        )

    return args


def export_data_process(
    adapter_name_or_path: str,
    export_dir: str,
    model_name_or_path: str,
    template: str,
    finetuning_type: str,
) -> dict:
    export_data = {
        "adapter_name_or_path": adapter_name_or_path,
        "export_dir": export_dir,
        "export_size": 5,
        "export_device": "auto",
        "export_legacy_format": False,
        "model_name_or_path": model_name_or_path,
        "template": template,
        "finetuning_type": finetuning_type,
    }

    return export_data


def add_train_path(path: str) -> str:
    os.makedirs(path, exist_ok=False)
    return path


async def call_ds_api(
    name: str, ds_args: schema.DeepSpeedArgs, ds_file: Union[UploadFile, None] = None
) -> dict:
    base_url = f"http://127.0.0.1:{MAINSERVICE_CONFIG.port}/acceltune/deepspeed"
    async with httpx.AsyncClient(timeout=None) as aclient:
        if ds_args.src == "default":
            payload = {
                "json": {
                    "train_name": name,
                    "stage": ds_args.stage,
                    "enable_offload": ds_args.enable_offload,
                    "offload_device": ds_args.offload_device,
                }
            }
        elif ds_args.src == "file":
            payload = {
                "files": {
                    "ds_file": (
                        ds_file.filename,
                        await ds_file.read(),
                        ds_file.content_type,
                    ),
                    "train_name": (None, name),
                }
            }

        response = await aclient.post(f"{base_url}/{ds_args.src}/", **payload)

        if response.status_code != status.HTTP_200_OK:
            raise HTTPException(
                status_code=response.status_code, detail=response.json()
            )

        return response.json()


async def write_yaml(path: str, data: dict) -> None:
    try:
        yaml_content = yaml.dump(data, default_flow_style=False)

        async with aiofiles.open(path, "w") as af:
            await af.write(yaml_content)

    except Exception as e:
        raise OSError(f"Unexpected error: {e}") from None


async def del_train(path: str) -> None:
    await asyncio.to_thread(shutil.rmtree, path)


async def async_clear_last_checkpoint(train_path: str) -> None:
    for method in {"full", "lora", "merge", "quantize", "deploy"}:
        checkpoint_path = os.path.join(train_path, method)
        checkpoint_exists = await aiofiles.os.path.exists(checkpoint_path)

        if checkpoint_exists:
            if not await aiofiles.os.path.isdir(checkpoint_path):
                return

            entries = await asyncio.to_thread(lambda: list(os.scandir(checkpoint_path)))  # noqa: B023
            for entry in entries:
                item_path = entry.path

                if entry.is_file():
                    await aiofiles.os.remove(item_path)
                elif entry.is_dir():
                    await asyncio.to_thread(shutil.rmtree, item_path)


async def async_clear_file(paths: List[str]) -> None:
    async def delete_file(file_path: str) -> None:
        is_exists = await aiofiles.os.path.exists(file_path)
        if is_exists:
            await aiofiles.os.remove(file_path)

    await asyncio.gather(*(delete_file(path) for path in paths))


@asynccontextmanager
async def record_train_log(
    log_path: str,
) -> AsyncGenerator[aiofiles.threadpool.text.AsyncTextIndirectIOWrapper, None]:
    file = await aiofiles.open(log_path, "w")
    try:
        yield file
    finally:
        await file.close()


async def call_internal_merge_api(merge_name: str) -> str:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://127.0.0.1:{MAINSERVICE_CONFIG.port}/acceltune/merge/start/",
            json={"merge_name": merge_name},
        )
        container_name = response.json()["container_name"]

    if response.status_code == status.HTTP_200_OK:
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            container_info = await wait_for_container(
                aclient=aclient, container_name=container_name
            )
            exit_status = container_info["StatusCode"]
            if exit_status == 0:
                merge_status = STATUS_CONFIG.finish
            elif exit_status in {137, 143}:
                merge_status = STATUS_CONFIG.stopped
            elif exit_status == 1:
                merge_status = STATUS_CONFIG.failed
            else:
                merge_status = STATUS_CONFIG.failed

            await remove_container(aclient=aclient, container_name_or_id=container_name)

    return merge_status


async def merge_event(
    name: str,
    yaml_path: str,
    adapter_name_or_path: str,
    export_dir: str,
    model_name_or_path: str,
    template: str,
    finetuning_type: str,
) -> str:
    try:
        export_data = export_data_process(
            adapter_name_or_path=adapter_name_or_path,
            export_dir=export_dir,
            model_name_or_path=model_name_or_path,
            template=template,
            finetuning_type=finetuning_type,
        )
        await write_yaml(path=yaml_path, data=export_data)
        merge_status = await call_internal_merge_api(merge_name=name)
        return merge_status

    except Exception as e:
        accel_logger.error(f"Merge event error: {e}")
        raise RuntimeError("Merge event error") from None


def get_last_checkpoint(path: str) -> Union[str, None]:
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    checkpoint_dict = dict()

    if os.path.exists(path):
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    match = checkpoint_pattern.fullmatch(entry.name)

                    if match:
                        checkpoint_dict.update({entry.name: int(match.group(1))})

        return (
            os.path.join(path, max(checkpoint_dict, key=checkpoint_dict.get))
            if checkpoint_dict
            else None
        )
    else:
        return None


async def start_train_background_task(train_name: str, container_name_or_id: str):
    ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    try:
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            # 2025.02.26 by Manny
            async with record_train_log(
                log_path=os.path.join(COMMON_CONFIG.save_path, train_name, "train.log")
            ) as log_file:  # write all training log into file
                async for log in get_container_log(
                    aclient=aclient, container_name_or_id=container_name_or_id
                ):
                    for log_split in log.splitlines():
                        if log_split == "":
                            break
                        elif log_split[0] in ("\x01", "\x02"):
                            log_split = log_split[8:]

                        log_file: aiofiles.threadpool.text.AsyncTextIndirectIOWrapper
                        await log_file.write(f"{ANSI_ESCAPE.sub('', log_split)}\n")

            container_info = await wait_for_container(
                aclient=aclient, container_name=container_name_or_id
            )
            exit_status = container_info["StatusCode"]
            if exit_status == 0:
                train_status = STATUS_CONFIG.finish
            elif exit_status == 1:
                train_status = STATUS_CONFIG.failed
            else:
                train_status = None

            await remove_container(
                aclient=aclient, container_name_or_id=container_name_or_id
            )

    except ValueError as e:
        train_status = STATUS_CONFIG.failed
        accel_logger.error(f"Docker error: {e}")

    except RuntimeError as e:
        train_status = STATUS_CONFIG.failed
        accel_logger.error(f"Docker error: {e}")

    except Exception as e:
        train_status = STATUS_CONFIG.failed
        accel_logger.error(f"Unexpected error: {e}")

    finally:
        try:
            if train_status in {STATUS_CONFIG.finish, STATUS_CONFIG.failed}:
                info = await redis_async.client.hget(TASK_CONFIG.train, train_name)
                info = orjson.loads(info)
                output_dir = info["train_args"]["output_dir"]
                root_output_dir = os.path.dirname(output_dir)
                finetuning_type: Literal["full", "lora"] = info["train_args"][
                    "finetuning_type"
                ]
                last_model_path = get_last_checkpoint(output_dir)

                info["last_model_path"] = last_model_path
                info["container"]["train"]["status"] = train_status
                info["container"]["train"]["id"] = None

                if finetuning_type == "lora" and last_model_path is not None:
                    merge_path = os.path.join(root_output_dir, "merge")
                    try:
                        merge_status = await merge_event(
                            name=train_name,
                            yaml_path=os.path.join(root_output_dir, "export.yaml"),
                            adapter_name_or_path=last_model_path,
                            export_dir=merge_path,
                            model_name_or_path=info["train_args"]["base_model"],
                            template=info["train_args"]["template"],
                            finetuning_type=finetuning_type,
                        )

                        if merge_status == STATUS_CONFIG.finish:
                            info["last_model_path"] = merge_path
                        else:
                            info["last_model_path"] = None
                            info["container"]["train"]["status"] = STATUS_CONFIG.failed
                    except Exception as e:
                        info["container"]["train"]["status"] = STATUS_CONFIG.failed
                        accel_logger.error(f"Unexpected error: {e}")

                await redis_async.client.hset(
                    TASK_CONFIG.train, train_name, orjson.dumps(info)
                )
        except Exception as e:
            accel_logger.error(f"Database error: {e}")


async def run_train(
    image_name: str,
    cmd: list,
    docker_network_name: str,
    train_name: str,
    is_deepspeed: bool,
    use_nvme: bool,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    env_var = [f"HF_HOME={COMMON_CONFIG.hf_home}"]
    if is_deepspeed:
        env_var.append("FORCE_TORCHRUN=1")
    data = {
        "User": "root",
        "Image": image_name,
        "HostConfig": {
            "IpcMode": "host",
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
            ],
            "Binds": [
                f"{COMMON_CONFIG.hf_home}:{COMMON_CONFIG.hf_home}:rw",
                f"{COMMON_CONFIG.root_path}/data:{COMMON_CONFIG.data_path}:rw",
                f"{COMMON_CONFIG.root_path}/saves/{train_name}:{COMMON_CONFIG.save_path}/{train_name}:rw",
            ],
            "NetworkMode": docker_network_name,
        },
        "Cmd": cmd,
        "Env": env_var,
    }

    if use_nvme:
        data["HostConfig"]["Binds"].append(
            f"{COMMON_CONFIG.nvme_path}:{COMMON_CONFIG.nvme_path}:rw"
        )

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            container_name_or_id = await create_container(
                aclient=aclient, name=f"train-{train_name}", data=data
            )
            started_container = await start_container(
                aclient=aclient, container_name_or_id=container_name_or_id
            )

        return started_container

    except Exception as e:
        accel_logger.error(f"{e}")
        raise RuntimeError(f"{e}") from None


async def stop_train(
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            stopped_container = await stop_container(
                aclient=aclient,
                container_name_or_id=container_name_or_id,
                signal=signal,
                wait_sec=wait_sec,
            )

        return stopped_container

    except Exception as e:
        accel_logger.error(f"{e}")
        raise RuntimeError(f"{e}") from None


async def train_finish_event(path: str) -> validator.TrainResult:
    output = validator.TrainResult()

    if os.path.exists(path):
        async with aiofiles.open(path) as f:
            content = await f.read()
        train_results: List[dict] = orjson.loads(content)["log_history"]

        epoch_idx: Dict[int, int] = dict()
        last_eval_info = None

        for log in train_results:
            epoch: float = log.get("epoch")
            step: int = log.get("step")

            if "train_loss" in log:
                output.final_report.epoch = epoch
                output.final_report.step = step
                output.final_report.total_flos = log["total_flos"]
                output.final_report.train_runtime = log["train_runtime"]
                output.final_report.train_loss = log["train_loss"]
                output.final_report.train_samples_per_second = log[
                    "train_samples_per_second"
                ]
                output.final_report.train_steps_per_second = log[
                    "train_steps_per_second"
                ]
            elif "loss" in log:
                loss: float = log["loss"]
                log_history_info = validator.LogHistory(
                    epoch=epoch, step=step, loss=loss, eval_loss=0.0
                )
                epoch_idx[epoch] = len(output.log_history)
                output.log_history.append(log_history_info)
            elif "eval_loss" in log:
                if epoch in epoch_idx:
                    idx = epoch_idx[epoch]
                    output.log_history[idx].eval_loss = log["eval_loss"]
                    output.log_history[idx].step = step

                last_eval_info = {
                    "eval_loss": log["eval_loss"],
                    "eval_runtime": log.get("eval_runtime"),
                    "eval_samples_per_second": log.get("eval_samples_per_second"),
                    "eval_steps_per_second": log.get("eval_steps_per_second"),
                }

        if last_eval_info:
            output.final_report.eval_loss = last_eval_info["eval_loss"]
            output.final_report.eval_runtime = last_eval_info["eval_runtime"]
            output.final_report.eval_samples_per_second = last_eval_info[
                "eval_samples_per_second"
            ]
            output.final_report.eval_steps_per_second = last_eval_info[
                "eval_steps_per_second"
            ]

    validator.TrainResult.model_validate(output)

    return output


async def train_stop_failed_event(path: str) -> validator.TrainResult:
    output = validator.TrainResult()

    if os.path.exists(path):
        train_log: List[dict] = list()
        async with aiofiles.open(path) as f:
            async for line in f:
                if line.strip():
                    content = orjson.loads(line)
                    train_log.append(content)

        epoch_idx: Dict[int, int] = dict()

        for log in train_log:
            epoch: float = log.get("epoch")
            step: int = log.get("current_steps")

            if "loss" in log:
                loss: float = log["loss"]
                log_history_info = validator.LogHistory(
                    epoch=epoch,
                    step=step,
                    loss=loss,
                    eval_loss=0.0,
                )
                epoch_idx[epoch] = len(output.log_history)
                output.log_history.append(log_history_info)
            elif "eval_loss" in log:
                if epoch in epoch_idx:
                    idx = epoch_idx[epoch]
                    output.log_history[idx].eval_loss = log["eval_loss"]
                    output.log_history[idx].step = log["current_steps"]

    validator.TrainResult.model_validate(output)

    return output


async def get_train_result(info: dict) -> dict:
    status = info["container"]["train"]["status"]
    if status == STATUS_CONFIG.finish:
        train_result = await train_finish_event(
            path=os.path.join(info["train_args"]["output_dir"], "trainer_state.json")
        )
    elif status in {STATUS_CONFIG.stopped, STATUS_CONFIG.failed}:
        train_result = await train_stop_failed_event(
            path=os.path.join(info["train_args"]["output_dir"], "trainer_log.jsonl")
        )

    return train_result.model_dump()
