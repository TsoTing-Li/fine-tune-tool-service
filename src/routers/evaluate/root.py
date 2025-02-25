import json
import os

import orjson
from fastapi import APIRouter, HTTPException, Response, status

from src.config import params
from src.routers.evaluate import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import assemble_image_name

router = APIRouter(prefix="/eval", tags=["Evaluate"])


@router.post("/start/")
async def start_lm_eval(request_data: schema.PostStartEval):
    validator.PostStartEval(eval_name=request_data.eval_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(
            params.TASK_CONFIG.train, request_data.eval_name
        )
        model_params = orjson.loads(info)["train_args"]

        eval_container = await utils.run_lm_eval(
            image_name=assemble_image_name(
                username=params.COMMON_CONFIG.username,
                repository=params.COMMON_CONFIG.repository,
                tag=params.LMEVAL_CONFIG.tag,
            ),
            cmd=[
                "lm-eval",
                "--model",
                "local-completions"
                if request_data.eval_type == "generate"
                else "local-chat-completions",
                "--task",
                ",".join(list(dict.fromkeys(request_data.tasks))),
                "--batch_size",
                "auto",
                "--output_path",
                os.path.join(params.COMMON_CONFIG.save_path, request_data.eval_name),
                "--use_cache",
                params.COMMON_CONFIG.cache_path,
                "--model_args",
                f"model={request_data.eval_name},"
                f"base_url=http://{request_data.model_server_url}:8000/v1/completions,"
                f"num_concurrent={request_data.num_concurrent},"
                f"max_retires={request_data.max_retries},"
                f"tokenizer={model_params['model_name_or_path']}",
            ],
            eval_name=request_data.eval_name,
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
        info = await redis_async.client.hget(
            params.TASK_CONFIG.train, request_data.eval_name
        )
        info = orjson.loads(info)
        info["container"]["eval"] = "active"
        info["container"]["eval"]["id"] = eval_container
        await redis_async.client.hset(
            params.TASK_CONFIG.train, request_data.eval_name, orjson.dumps(info)
        )

    except Exception as e:
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
        content=json.dumps({"eval_container": eval_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_lm_eval(request_data: schema.PostStopEval):
    validator.PostStopEval(eval_name=request_data.eval_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(
            params.TASK_CONFIG.train, request_data.eval_name
        )
        info = orjson.loads(info)
        info["container"]["eval"]["status"] = "stopped"
        await redis_async.client.hset(
            params.TASK_CONFIG.train, request_data.eval_name, orjson.dumps(info)
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
        await utils.stop_eval(container_name_or_id=info["container"]["eval"]["id"])

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
        content=json.dumps({"eval_name": request_data.eval_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
