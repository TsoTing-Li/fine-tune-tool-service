import json
import os

from fastapi import APIRouter, Response, status

from inno_service.routers.evaluate import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")

router = APIRouter(prefix="/eval", tags=["Evaluate"])


@router.post("/start/")
async def start_lm_eval(request_data: schema.PostStartEval):
    validator.PostStartEval(eval_name=request_data.eval_name)
    error_handler = ResponseErrorHandler()

    try:
        model_params = await utils.get_model_params(
            path=os.path.join(
                os.environ["WS"],
                os.environ["SAVE_PATH"],
                f"{request_data.eval_name}/{request_data.eval_name}.yaml",
            )
        )

        eval_container = await utils.run_lm_eval(
            image_name=f"{os.environ['USER_NAME']}/{os.environ['REPOSITORY']}:{os.environ['LM_EVAL_TAG']}",
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
                f"{os.environ['WS']}/{os.getenv('SAVE_PATH', 'saves')}",
                "--use_cache",
                f"{os.environ['WS']}/cache",
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
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input=request_data.model_dump(),
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

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
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input=request_data.model_dump(),
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({"eval_container": eval_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
