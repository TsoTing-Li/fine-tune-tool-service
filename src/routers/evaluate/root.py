import json
import os
from typing import Annotated

import orjson
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status

from src.config.params import (
    COMMON_CONFIG,
    DOCKERNETWORK_CONFIG,
    EVAL_CONFIG,
    STATUS_CONFIG,
    TASK_CONFIG,
)
from src.routers.evaluate import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import assemble_image_name

router = APIRouter(prefix="/eval", tags=["Evaluate"])


@router.post("/start/")
async def start_lm_eval(
    background_tasks: BackgroundTasks, request_data: schema.PostStartEval
):
    validator.PostStartEval(eval_name=request_data.eval_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.eval_name)
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
        cmd = [
            "lm-eval",
            "--model",
            "local-completions",
            "--task",
            ",".join(request_data.tasks),
            "--batch_size",
            "auto",
            "--output_path",
            os.path.join(COMMON_CONFIG.save_path, request_data.eval_name, "evaluate"),
            "--use_cache",
            COMMON_CONFIG.cache_path,
        ]

        if any("humaneval" in task or "mbpp" in task for task in request_data.tasks):
            cmd += ["--confirm_run_unsafe_code"]

        model_args = (
            f"model={request_data.eval_name},"
            + f"base_url={request_data.model_service}/v1/completions,"
            + "num_concurrent=1,"
            + "max_retries=3,"
            + f"tokenizer={info['train_args']['base_model']}"
        )
        cmd += ["--model_args", model_args]

        eval_container = await utils.run_lm_eval(
            image_name=assemble_image_name(
                username=COMMON_CONFIG.username,
                repository=f"{COMMON_CONFIG.repository}-{EVAL_CONFIG.name}",
                tag=EVAL_CONFIG.tag,
            ),
            cmd=cmd,
            docker_network_name=DOCKERNETWORK_CONFIG.network_name,
            eval_name=request_data.eval_name,
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
        info["container"]["eval"]["status"] = STATUS_CONFIG.active
        info["container"]["eval"]["id"] = eval_container
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.eval_name, orjson.dumps(info)
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
        utils.start_eval_background_task,
        request_data.eval_name,
        eval_container,
        request_data.tasks,
    )

    return Response(
        content=json.dumps(info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_lm_eval(
    background_tasks: BackgroundTasks, request_data: schema.PostStopEval
):
    validator.PostStopEval(eval_name=request_data.eval_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.eval_name)
        info = orjson.loads(info)
        stop_container = info["container"]["eval"]["id"]
        info["container"]["eval"]["status"] = STATUS_CONFIG.stopped
        info["container"]["eval"]["id"] = None
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.eval_name, orjson.dumps(info)
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

    background_tasks.add_task(utils.stop_eval_background_task, stop_container)

    return Response(
        content=json.dumps({"eval_name": request_data.eval_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/result/")
async def get_eval_result(eval_name: Annotated[str, Query(...)]):
    query_data = schema.GetEvalResult(eval_name=eval_name)
    validator.GetEvalResult(eval_name=query_data.eval_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, query_data.eval_name)
        info = orjson.loads(info)

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg="Database error",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        eval_result = await utils.get_eval_result(info=info)

    except Exception as e:
        accel_logger.error(f"{e}")
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
        content=json.dumps(eval_result),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
