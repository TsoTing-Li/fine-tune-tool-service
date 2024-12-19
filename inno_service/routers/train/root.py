import json
import os

from fastapi import APIRouter, Response, status

from inno_service.routers.train import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler
from inno_service.utils.utils import get_current_time

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")
TRAIN_CONFIG_PATH = os.getenv("TRAIN_CONFIG_PATH", "/app/train_config")

router = APIRouter(prefix="/train", tags=["Train"])


@router.post("/start/")
async def start_train(request_data: schema.PostStartTrain):
    if not request_data.train_name:
        train_name = get_current_time()
    else:
        train_name = request_data.train_name

    train_name = validator.PostStartTrain(train_name=train_name).train_name
    error_handler = ResponseErrorHandler()

    try:
        train_args = utils.basemodel2dict(data=request_data.train_args)
        train_args["output_dir"] = os.path.join(
            SAVE_PATH, train_name, train_args["finetuning_type"]
        )
        train_args["eval_steps"] = train_args["save_steps"]
        yaml_path = os.path.join(SAVE_PATH, train_name, f"{train_name}.yaml")
        train_config_path = os.path.join(TRAIN_CONFIG_PATH, f"{train_name}.yaml")
        await utils.write_train_yaml_to_two_path(
            train_config_path=train_config_path, path=yaml_path, data=train_args
        )

        container_name = await utils.run_train(
            image_name=f"{os.environ['USER_NAME']}/{os.environ['REPOSITORY']}:{os.environ['FINE_TUNE_TOOL_TAG']}",
            cmd=["llamafactory-cli", "train", yaml_path],
            train_name=train_name,
        )

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"train_name": train_name},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps(
            {"train_name": train_name, "container_name": container_name}
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_train(request_data: schema.PostStopTrain):
    train_container = validator.PostStopTrain(
        train_container=request_data.train_container
    ).train_container
    error_handler = ResponseErrorHandler()

    try:
        train_container = await utils.stop_train(container_name_or_id=train_container)

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"train_container": train_container},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({"train_container": train_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
