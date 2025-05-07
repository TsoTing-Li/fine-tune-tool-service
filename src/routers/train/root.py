import json
import os
from typing import Annotated, List, Literal, Union

import orjson
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from src.config.params import (
    COMMON_CONFIG,
    DOCKERNETWORK_CONFIG,
    FINETUNETOOL_CONFIG,
    STATUS_CONFIG,
    TASK_CONFIG,
)
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
async def start_train(
    background_tasks: BackgroundTasks, request_data: schema.PostStartTrain
):
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

        await utils.async_clear_last_checkpoint(
            train_path=os.path.dirname(info["train_args"]["output_dir"])
        )
        container_name = await utils.run_train(
            image_name=assemble_image_name(
                username=COMMON_CONFIG.username,
                repository=COMMON_CONFIG.repository,
                tag=FINETUNETOOL_CONFIG.tag,
            ),
            cmd=["sh", "-c", " && ".join(commands)],
            docker_network_name=DOCKERNETWORK_CONFIG.network_name,
            train_name=request_data.train_name,
            is_deepspeed=info["offloading"]["deepspeed"]["use"],
            use_nvme=info["offloading"]["nvme"]["use"],
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        info["container"]["train"]["status"] = STATUS_CONFIG.active
        info["container"]["train"]["id"] = container_name
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.train_name, orjson.dumps(info)
        )

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

    background_tasks.add_task(
        utils.start_train_background_task, request_data.train_name, container_name
    )

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
        await utils.stop_train(container_name_or_id=info["container"]["train"]["id"])

        output_dir = info["train_args"]["output_dir"]
        root_output_dir = os.path.dirname(output_dir)
        finetuning_type: Literal["full", "lora"] = info["train_args"]["finetuning_type"]
        last_model_path = utils.get_last_checkpoint(output_dir)
        train_status = STATUS_CONFIG.stopped

        info["last_model_path"] = last_model_path
        info["container"]["train"]["id"] = None

        if finetuning_type == "lora" and last_model_path is not None:
            merge_path = os.path.join(root_output_dir, "merge")
            merge_status = await utils.merge_event(
                name=request_data.train_name,
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
        train_status = STATUS_CONFIG.failed
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    finally:
        try:
            info["container"]["train"]["status"] = train_status
            await redis_async.client.hset(
                TASK_CONFIG.train, request_data.train_name, orjson.dumps(info)
            )

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

    return Response(
        content=json.dumps({"train_name": request_data.train_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/log/")
async def get_log(train_name: Annotated[str, Query(...)]):
    query_data = schema.GetTrainLog(train_name=train_name)
    validator.GetTrainLog(train_name=query_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, query_data.train_name)
        info = orjson.loads(info)

        log_path = os.path.join(
            os.path.dirname(info["train_args"]["output_dir"]), "train.log"
        )
        if not os.path.exists(log_path):
            raise FileNotFoundError("log file not found")

    except FileNotFoundError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return FileResponse(
        path=log_path,
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
        filename=f"{query_data.train_name}.log",
    )


@router.get("/result/")
async def get_result(train_name: Annotated[str, Query(...)]):
    query_data = schema.GetTrainResult(train_name=train_name)
    validator.GetTrainResult(train_name=query_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, query_data.train_name)
        info = orjson.loads(info)

        train_result = await utils.get_train_result(info=info)

    except FileNotFoundError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(train_result),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/")
async def add_train(
    train_name: str = Form(None),
    base_model: str = Form(...),
    finetuning_type: Literal["full", "lora"] = Form(...),
    dataset: List[str] = Form(...),
    template: str = Form(...),
    cutoff_len: int = Form(1024),
    max_samples: int = Form(10000),
    overwrite_cache: bool = Form(True),
    preprocessing_num_workers: int = Form(16),
    save_steps: int = Form(5),
    per_device_train_batch_size: int = Form(1),
    gradient_accumulation_steps: int = Form(8),
    learning_rate: float = Form(0.0001),
    num_train_epochs: int = Form(3),
    lr_scheduler_type: Literal[
        "cosine",
        "cosine_with_restarts",
        "cosine_with_min_lr",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "linear",
        "polynomial",
        "reduce_lr_on_plateau",
        "warmup_stable_decay",
    ] = Form("cosine"),
    warmup_ratio: float = Form(0.1),
    compute_type: Literal["bf16", "fp16"] = Form(...),
    ddp_timeout: int = Form(180000000),
    val_size: float = Form(0.1),
    lora_alpha: int = Form(None),
    lora_dropout: float = Form(0.0),
    lora_rank: int = Form(8),
    lora_target: List[str] = Form(["all"]),
    deepspeed_src: Literal["default", "file", None] = Form(None),
    deepspeed_stage: Literal["2", "3", None] = Form(None),
    deepspeed_enable_offload: bool = Form(False),
    deepspeed_offload_device: Literal["cpu", "nvme", None] = Form(None),
    deepspeed_file: UploadFile = File(None),
):
    unix_time, date_time = get_current_time()
    train_args = {
        "base_model": base_model,
        "method": {
            "stage": "sft",
            "finetuning_type": finetuning_type,
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
            "logging_steps": save_steps,
            "save_steps": save_steps,
            "plot_loss": False,
            "overwrite_output_dir": False,
            "log_level": "info",
            "logging_first_step": False,
        },
        "params": {
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "compute_type": compute_type,
            "ddp_timeout": ddp_timeout,
        },
        "val": {
            "val_size": val_size,
            "per_device_eval_batch_size": per_device_train_batch_size,
            "eval_strategy": "steps",
        },
        "lora": {
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_rank": lora_rank,
            "lora_target": lora_target,
        }
        if finetuning_type == "lora"
        else None,
    }

    deepspeed_args = (
        {
            "src": deepspeed_src,
            "stage": int(deepspeed_stage) if deepspeed_stage else None,
            "enable_offload": deepspeed_enable_offload,
            "offload_device": deepspeed_offload_device,
        }
        if deepspeed_src
        else None
    )

    request_data = schema.PostTrain(
        train_name=train_name if train_name else f"{date_time}-{generate_uuid()}",
        train_args=train_args,
        deepspeed_args=deepspeed_args,
        deepspeed_file=deepspeed_file,
    )
    validator.PostTrain(train_name=request_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        train_args = utils.basemodel2dict(data=request_data.train_args)
        redis_train_args = utils.redis_train_args_process(
            train_name=request_data.train_name,
            train_args=train_args,
            lora_args=request_data.train_args.lora.model_dump()
            if request_data.train_args.lora is not None
            else None,
            save_path=COMMON_CONFIG.save_path,
            dataset_path=COMMON_CONFIG.data_path,
        )
        file_train_args = utils.file_train_args_process(
            train_name=request_data.train_name,
            train_args=train_args,
            lora_args=request_data.train_args.lora.model_dump()
            if request_data.train_args.lora is not None
            else None,
            save_path=COMMON_CONFIG.save_path,
            dataset_path=COMMON_CONFIG.data_path,
        )

        train_path = utils.add_train_path(
            path=os.path.join(COMMON_CONFIG.save_path, request_data.train_name)
        )

        if request_data.deepspeed_args:
            ds_args = request_data.deepspeed_args
            ds_api_response = await utils.call_ds_api(
                name=request_data.train_name,
                ds_args=ds_args,
                ds_file=request_data.deepspeed_file,
            )
            file_train_args["deepspeed"] = ds_api_response["ds_path"]

            if ds_args.src == "default":
                redis_train_args["deepspeed_src"] = ds_args.src
                redis_train_args["deepspeed_stage"] = ds_args.stage
                redis_train_args["deepspeed_enable_offload"] = ds_args.enable_offload
                redis_train_args["deepspeed_offload_device"] = ds_args.offload_device
            elif ds_args.src == "file":
                redis_train_args["deepspeed_src"] = ds_args.src

        await utils.write_yaml(
            path=os.path.join(train_path, f"{request_data.train_name}.yaml"),
            data=file_train_args,
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
            "train_args": redis_train_args,
            "offloading": {
                "deepspeed": {
                    "use": True if request_data.deepspeed_args else False,
                    "path": file_train_args.get("deepspeed", None),
                },
                "nvme": {
                    "use": True
                    if request_data.deepspeed_args is not None
                    and request_data.deepspeed_args.offload_device == "nvme"
                    else False,
                    "path": COMMON_CONFIG.nvme_path
                    if request_data.deepspeed_args is not None
                    and request_data.deepspeed_args.offload_device == "nvme"
                    else None,
                },
            },
            "container": {
                "train": {"status": "setup", "id": None},
                "eval": {"status": "setup", "id": None},
                "quantize": {"status": "setup", "id": None},
                "infer_backend": {
                    "status": "setup",
                    "id": None,
                    "url": None,
                    "type": None,
                },
            },
            "last_model_path": None,
            "created_time": unix_time,
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
    base_model: str = Form(...),
    finetuning_type: Literal["full", "lora"] = Form(...),
    dataset: List[str] = Form(...),
    template: str = Form(...),
    cutoff_len: int = Form(...),
    max_samples: int = Form(...),
    overwrite_cache: bool = Form(...),
    preprocessing_num_workers: int = Form(...),
    save_steps: int = Form(...),
    per_device_train_batch_size: int = Form(...),
    gradient_accumulation_steps: int = Form(...),
    learning_rate: float = Form(...),
    num_train_epochs: int = Form(...),
    lr_scheduler_type: Literal[
        "cosine",
        "cosine_with_restarts",
        "cosine_with_min_lr",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "linear",
        "polynomial",
        "reduce_lr_on_plateau",
        "warmup_stable_decay",
    ] = Form(...),
    warmup_ratio: float = Form(...),
    compute_type: Literal["bf16", "fp16"] = Form(...),
    ddp_timeout: int = Form(...),
    val_size: float = Form(...),
    lora_alpha: int = Form(None),
    lora_dropout: float = Form(None),
    lora_rank: int = Form(None),
    lora_target: List[str] = Form(None),
    deepspeed_src: Literal["default", "file", None] = Form(None),
    deepspeed_stage: Literal["2", "3", None] = Form(None),
    deepspeed_enable_offload: bool = Form(False),
    deepspeed_offload_device: Literal["cpu", "nvme", None] = Form(None),
    deepspeed_file: UploadFile = File(None),
):
    unix_time, _ = get_current_time()
    train_args = {
        "base_model": base_model,
        "method": {
            "stage": "sft",
            "finetuning_type": finetuning_type,
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
            "logging_steps": save_steps,
            "save_steps": save_steps,
            "plot_loss": False,
            "overwrite_output_dir": False,
            "log_level": "info",
            "logging_first_step": False,
        },
        "params": {
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "lr_scheduler_type": lr_scheduler_type,
            "warmup_ratio": warmup_ratio,
            "compute_type": compute_type,
            "ddp_timeout": ddp_timeout,
        },
        "val": {
            "val_size": val_size,
            "per_device_eval_batch_size": per_device_train_batch_size,
            "eval_strategy": "steps",
        },
        "lora": {
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_rank": lora_rank,
            "lora_target": lora_target,
        }
        if finetuning_type == "lora"
        else None,
    }

    deepspeed_args = (
        {
            "src": deepspeed_src,
            "stage": int(deepspeed_stage) if deepspeed_stage else None,
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
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.train_name)
        info = orjson.loads(info)

    except Exception:
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
        train_args = utils.basemodel2dict(data=request_data.train_args)
        redis_train_args = utils.redis_train_args_process(
            train_name=request_data.train_name,
            train_args=train_args,
            lora_args=request_data.train_args.lora.model_dump()
            if request_data.train_args.lora is not None
            else None,
            save_path=COMMON_CONFIG.save_path,
            dataset_path=COMMON_CONFIG.data_path,
        )
        file_train_args = utils.file_train_args_process(
            train_name=request_data.train_name,
            train_args=train_args,
            lora_args=request_data.train_args.lora.model_dump()
            if request_data.train_args.lora is not None
            else None,
            save_path=COMMON_CONFIG.save_path,
            dataset_path=COMMON_CONFIG.data_path,
        )

        clear_up_list = [
            os.path.join(
                COMMON_CONFIG.save_path, request_data.train_name, "export.yaml"
            )
        ]

        if request_data.deepspeed_args:
            ds_args = request_data.deepspeed_args
            if (
                ds_args.src == "file" and request_data.deepspeed_file is not None
            ) or ds_args.src == "default":
                ds_api_response = await utils.call_ds_api(
                    name=request_data.train_name,
                    ds_args=ds_args,
                    ds_file=request_data.deepspeed_file,
                )
                file_train_args["deepspeed"] = ds_api_response["ds_path"]

                if info["offloading"]["deepspeed"]["path"] is not None:
                    clear_up_list.append(info["offloading"]["deepspeed"]["path"])
            else:
                if info["offloading"]["deepspeed"]["path"] is not None:
                    if not os.path.exists(info["offloading"]["deepspeed"]["path"]):
                        raise ValueError("no ds_file exists")

                    file_train_args["deepspeed"] = info["offloading"]["deepspeed"][
                        "path"
                    ]
                else:
                    raise ValueError("pls provide 'ds_file'")

            if ds_args.src == "default":
                redis_train_args["deepspeed_src"] = ds_args.src
                redis_train_args["deepspeed_stage"] = ds_args.stage
                redis_train_args["deepspeed_enable_offload"] = ds_args.enable_offload
                redis_train_args["deepspeed_offload_device"] = ds_args.offload_device
            elif ds_args.src == "file":
                redis_train_args["deepspeed_src"] = ds_args.src
        else:
            if info["offloading"]["deepspeed"]["path"] is not None:
                clear_up_list.append(info["offloading"]["deepspeed"]["path"])

        await utils.async_clear_file(paths=clear_up_list)

        await utils.write_yaml(
            path=os.path.join(
                COMMON_CONFIG.save_path,
                request_data.train_name,
                f"{request_data.train_name}.yaml",
            ),
            data=file_train_args,
        )

    except HTTPException as e:
        accel_logger.error(f"deepspeed api error: {e.detail['detail']}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail["detail"],
        ) from None

    except ValueError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_FORM],
            msg=f"{e}",
            input={
                "deepspeed_args.src": ds_args.src,
                "deepspeed_file": request_data.deepspeed_file,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_handler.errors,
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
        info["train_args"] = redis_train_args
        info["offloading"]["deepspeed"] = {
            "use": True if request_data.deepspeed_args else False,
            "path": file_train_args.get("deepspeed", None),
        }
        info["offloading"]["nvme"] = {
            "use": True
            if request_data.deepspeed_args is not None
            and request_data.deepspeed_args.offload_device == "nvme"
            else False,
            "path": COMMON_CONFIG.nvme_path
            if request_data.deepspeed_args is not None
            and request_data.deepspeed_args.offload_device == "nvme"
            else None,
        }
        info["container"] = {
            "train": {"status": "setup", "id": None},
            "eval": {"status": "setup", "id": None},
            "quantize": {"status": "setup", "id": None},
            "infer_backend": {
                "status": "setup",
                "id": None,
                "url": None,
                "type": None,
            },
        }
        info["last_model_path"] = None
        info["modified_time"] = unix_time
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
        await utils.del_train(
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
