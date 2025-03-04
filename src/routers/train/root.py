import json
import os
from typing import Annotated, List, Union

import orjson
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from src.config.params import COMMON_CONFIG, FINETUNETOOL_CONFIG, TASK_CONFIG
from src.routers.train import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import (
    assemble_image_name,
    generate_uuid,
    get_current_time,
)

router = APIRouter(prefix="/train", tags=["Train"])


@router.post("/start/")
async def start_train(request_data: schema.PostStartTrain):
    validator.PostStartTrain(train_name=request_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.train_name)
        info = orjson.loads(info)

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg="Database error",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        commands = [
            f"llamafactory-cli train {os.path.join(COMMON_CONFIG.save_path, request_data.train_name, f'{request_data.train_name}.yaml')}"
        ]
        if info["train_args"]["finetuning_type"] == "lora":
            commands.append(
                f"llamafactory-cli export {os.path.join(COMMON_CONFIG.save_path, request_data.train_name, 'export.yaml')}"
            )

        await utils.async_clear_exists_path(train_path=info["train_args"]["output_dir"])
        container_name = await utils.run_train(
            image_name=assemble_image_name(
                username=COMMON_CONFIG.username,
                repository=COMMON_CONFIG.repository,
                tag=FINETUNETOOL_CONFIG.tag,
            ),
            cmd=["sh", "-c", " && ".join(commands)],
            train_name=request_data.train_name,
            is_deepspeed=True if info["train_args"].get("deepspeed", None) else False,
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        info["container"]["train"]["status"] = "active"
        info["container"]["train"]["id"] = container_name
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.train_name, orjson.dumps(info)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_train(request_data: schema.PostStopTrain):
    validator.PostStopTrain(train_name=request_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.train_name)
        info = orjson.loads(info)
        info["container"]["train"]["status"] = "stopped"
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.train_name, orjson.dumps(info)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        await utils.stop_train(container_name_or_id=info["container"]["train"]["id"])

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/")
async def add_train(
    train_name: str = Form(None),
    model_name_or_path: str = Form(...),
    finetuning_type: str = Form(...),
    lora_target: str = Form(None),
    dataset: List[str] = Form(...),
    template: str = Form(...),
    cutoff_len: int = Form(1024),
    max_samples: int = Form(10000),
    overwrite_cache: bool = Form(True),
    preprocessing_num_workers: int = Form(16),
    logging_steps: int = Form(5),
    save_steps: int = Form(5),
    per_device_train_batch_size: int = Form(1),
    gradient_accumulation_steps: int = Form(8),
    learning_rate: float = Form(0.0001),
    num_train_epochs: int = Form(3),
    lr_scheduler_type: str = Form("cosine"),
    warmup_ratio: float = Form(0.1),
    bf16: bool = Form(True),
    ddp_timeout: int = Form(180000000),
    val_size: float = Form(0.1),
    per_device_eval_batch_size: int = Form(1),
    deepspeed_src: str = Form(None),
    deepspeed_stage: str = Form(None),
    deepspeed_enable_offload: bool = Form(False),
    deepspeed_offload_device: str = Form(None),
    deepspeed_file: UploadFile = File(None),
):
    created_time = get_current_time(use_unix=True)
    train_args = {
        "model_name_or_path": model_name_or_path,
        "method": {
            "stage": "sft",
            "finetuning_type": finetuning_type,
            "lora_target": lora_target,
        },
        "dataset": {
            "dataset": dataset,
            "template": template,
            "cutoff_len": cutoff_len,
            "max_samples": max_samples,
            "overwrite_cache": overwrite_cache,
            "preprocessing_num_workers": preprocessing_num_workers,
        },
        "output": {
            "logging_steps": logging_steps,
            "save_steps": save_steps,
        },
        "params": {
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "bf16": bf16,
            "ddp_timeout": ddp_timeout,
        },
        "val": {
            "val_size": val_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "eval_strategy": "steps",
        },
    }

    deepspeed_args = (
        {
            "src": deepspeed_src,
            "stage": int(deepspeed_stage),
            "enable_offload": deepspeed_enable_offload,
            "offload_device": deepspeed_offload_device,
        }
        if deepspeed_src
        else None
    )

    request_data = schema.PostTrain(
        train_name=train_name
        if train_name
        else f"{get_current_time()}-{generate_uuid()}",
        train_args=train_args,
        deepspeed_args=deepspeed_args,
        deepspeed_file=deepspeed_file,
    )
    validator.PostTrain(train_name=request_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        train_args = utils.basemodel2dict(data=request_data.train_args)
        train_args["output_dir"] = os.path.join(
            COMMON_CONFIG.save_path,
            request_data.train_name,
            train_args["finetuning_type"],
        )
        train_args["dataset"] = ", ".join(train_args["dataset"])
        train_args["dataset_dir"] = COMMON_CONFIG.data_path
        train_args["eval_steps"] = train_args["save_steps"]
        train_args["do_train"] = True

        train_path = utils.add_train_path(
            path=os.path.join(COMMON_CONFIG.save_path, request_data.train_name)
        )

        if train_args["finetuning_type"] == "lora":
            export_data = {
                "adapter_name_or_path": train_args["output_dir"],
                "export_dir": os.path.join(
                    COMMON_CONFIG.save_path, request_data.train_name, "merge"
                ),
                "export_size": 5,
                "export_device": "auto",
                "export_legacy_format": False,
                "model_name_or_path": train_args["model_name_or_path"],
                "template": train_args["template"],
                "finetuning_type": train_args["finetuning_type"],
            }
            await utils.write_yaml(
                path=os.path.join(
                    COMMON_CONFIG.save_path, request_data.train_name, "export.yaml"
                ),
                data=export_data,
            )

        if request_data.deepspeed_args:
            ds_args = request_data.deepspeed_args.model_dump()
            ds_api_response = await utils.call_ds_api(
                name=request_data.train_name,
                ds_args=ds_args,
                ds_file=request_data.deepspeed_file,
            )
            train_args["deepspeed"] = ds_api_response["ds_path"]

        await utils.write_yaml(
            path=os.path.join(train_path, f"{request_data.train_name}.yaml"),
            data=train_args,
        )

    except HTTPException as e:
        accel_logger.error(f"DeepSpeed default error: {e.detail['detail']}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail["detail"],
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"train_name": request_data.train_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        train_info = {
            "name": request_data.train_name,
            "train_args": train_args,
            "container": {
                "train": {"status": "setup", "id": None},
                "eval": {"status": "setup", "id": None},
                "quantize": {"status": "setup", "id": None},
                "infer_backend": {"status": "setup", "id": None, "type": None},
            },
            "created_time": created_time,
            "modified_time": None,
        }
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.train_name, orjson.dumps(train_info)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input={"train_name": request_data.train_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps([train_info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/")
async def get_train(train_name: Annotated[Union[str, None], Query()] = None):
    query_data = schema.GetTrain(train_name=train_name)
    validator.GetTrain(train_name=query_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        if query_data.train_name:
            info = await redis_async.client.hget(
                TASK_CONFIG.train, query_data.train_name
            )
            train_info = [orjson.loads(info)]
        else:
            info = await redis_async.client.hgetall("TRAIN")
            train_info = (
                [orjson.loads(value) for value in info.values()]
                if len(info) != 0
                else list()
            )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Database error: {e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(train_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.put("/")
async def modify_train(
    train_name: str = Form(...),
    model_name_or_path: str = Form(...),
    finetuning_type: str = Form(...),
    lora_target: str = Form(None),
    dataset: List[str] = Form(...),
    template: str = Form(...),
    cutoff_len: int = Form(1024),
    max_samples: int = Form(10000),
    overwrite_cache: bool = Form(True),
    preprocessing_num_workers: int = Form(None),
    logging_steps: int = Form(None),
    save_steps: int = Form(None),
    per_device_train_batch_size: int = Form(None),
    gradient_accumulation_steps: int = Form(None),
    learning_rate: float = Form(None),
    num_train_epochs: int = Form(None),
    lr_scheduler_type: str = Form("cosine"),
    warmup_ratio: float = Form(None),
    bf16: bool = Form(True),
    ddp_timeout: int = Form(None),
    val_size: float = Form(None),
    per_device_eval_batch_size: int = Form(None),
    deepspeed_src: str = Form(None),
    deepspeed_stage: str = Form(None),
    deepspeed_enable_offload: bool = Form(False),
    deepspeed_offload_device: str = Form(None),
    deepspeed_file: UploadFile = File(None),
):
    modified_time = get_current_time(use_unix=True)
    train_args = {
        "model_name_or_path": model_name_or_path,
        "method": {
            "stage": "sft",
            "finetuning_type": finetuning_type,
            "lora_target": lora_target,
        },
        "dataset": {
            "dataset": dataset,
            "template": template,
            "cutoff_len": cutoff_len,
            "max_samples": max_samples,
            "overwrite_cache": overwrite_cache,
            "preprocessing_num_workers": preprocessing_num_workers,
        },
        "output": {
            "logging_steps": logging_steps,
            "save_steps": save_steps,
        },
        "params": {
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "bf16": bf16,
            "ddp_timeout": ddp_timeout,
        },
        "val": {
            "val_size": val_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "eval_strategy": "steps",
        },
    }

    deepspeed_args = (
        {
            "src": deepspeed_src,
            "stage": int(deepspeed_stage),
            "enable_offload": deepspeed_enable_offload,
            "offload_device": deepspeed_offload_device,
        }
        if deepspeed_src
        else None
    )

    request_data = schema.PutTrain(
        train_name=train_name,
        train_args=train_args,
        deepspeed_args=deepspeed_args,
        deepspeed_file=deepspeed_file,
    )
    validator.PutTrain(train_name=request_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        train_args = utils.basemodel2dict(data=request_data.train_args)
        train_args["output_dir"] = os.path.join(
            COMMON_CONFIG.save_path,
            request_data.train_name,
            train_args["finetuning_type"],
        )
        train_args["dataset"] = ", ".join(train_args["dataset"])
        train_args["dataset_dir"] = os.getenv("DATA_PATH", "/app/data")
        train_args["eval_steps"] = train_args["save_steps"]
        train_args["do_train"] = True

        await utils.async_clear_file(
            paths=[
                os.path.join(
                    COMMON_CONFIG.save_path,
                    request_data.train_name,
                    "export.yaml",
                ),
                os.path.join(
                    COMMON_CONFIG.save_path,
                    request_data.train_name,
                    f"ds_config_{request_data.train_name}.json",
                ),
            ]
        )

        if train_args["finetuning_type"] == "lora":
            export_data = {
                "adapter_name_or_path": train_args["output_dir"],
                "export_dir": os.path.join(
                    COMMON_CONFIG.save_path, request_data.train_name, "merge"
                ),
                "export_size": 5,
                "export_device": "auto",
                "export_legacy_format": False,
                "model_name_or_path": train_args["model_name_or_path"],
                "template": train_args["template"],
                "finetuning_type": train_args["finetuning_type"],
            }
            await utils.write_yaml(
                path=os.path.join(
                    COMMON_CONFIG.save_path, request_data.train_name, "export.yaml"
                ),
                data=export_data,
            )

        if request_data.deepspeed_args:
            ds_args = request_data.deepspeed_args.model_dump()
            ds_api_response = await utils.call_ds_api(
                name=request_data.train_name,
                ds_args=ds_args,
                ds_file=request_data.deepspeed_file,
            )
            train_args["deepspeed"] = ds_api_response["ds_path"]

        await utils.write_yaml(
            path=os.path.join(
                COMMON_CONFIG.save_path,
                request_data.train_name,
                f"{request_data.train_name}.yaml",
            ),
            data=train_args,
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"train_name": request_data.train_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.train_name)
        info = orjson.loads(info)
        info["train_args"] = train_args
        info["container"]["train"]["status"] = "setup"
        info["modified_time"] = modified_time
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.train_name, orjson.dumps(info)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input={"train_name": request_data.train_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps([info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.delete("/")
async def delete_train(train_name: Annotated[str, Query(...)]):
    query_data = schema.DelTrain(train_name=train_name)
    validator.DelTrain(train_name=query_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        del_info = await redis_async.client.hget(
            TASK_CONFIG.train, query_data.train_name
        )
        del_info = orjson.loads(del_info)
        await redis_async.client.hdel(TASK_CONFIG.train, query_data.train_name)

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input={"train_name": query_data.train_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        utils.del_train(
            path=os.path.join(COMMON_CONFIG.save_path, query_data.train_name)
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps([del_info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
