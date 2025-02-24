import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from src.config import params
from src.routers.evaluate import schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import assemble_image_name

router = APIRouter(prefix="/eval", tags=["Evaluate"])


@router.post("/start/")
async def start_lm_eval(request_data: schema.PostStartEval):
    validator.PostStartEval(eval_name=request_data.eval_name)
    error_handler = ResponseErrorHandler()

    try:
        model_params = await utils.get_model_params(
            path=os.path.join(
                params.COMMON_CONFIG.save_path,
                request_data.eval_name,
                f"{request_data.eval_name}.yaml",
            )
        )

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
                params.COMMON_CONFIG.save_path,
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

    return Response(
        content=json.dumps({"eval_container": eval_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_lm_eval(request_data: schema.PostStopEval):
    validator.PostStopEval(eval_container=request_data.eval_container)
    error_handler = ResponseErrorHandler()

    try:
        eval_container = await utils.stop_eval(
            container_name_or_id=request_data.eval_container
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

    return Response(
        content=json.dumps({"eval_container": eval_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
