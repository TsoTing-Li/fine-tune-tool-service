import json
import os

from fastapi import APIRouter, Response, status

from inno_service.routers.train import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler
from inno_service.utils.utils import generate_uuid, get_current_time

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")

router = APIRouter(prefix="/train", tags=["Train"])


@router.post("/start/")
async def start_train(request_data: schema.PostStartTrain):
    validator.PostStartTrain(train_name=request_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        container_name = await utils.run_train(
            image_name=f"{os.environ['USER_NAME']}/{os.environ['REPOSITORY']}:{os.environ['FINE_TUNE_TOOL_TAG']}",
            cmd=[
                "llamafactory-cli",
                "train",
                os.path.join(
                    SAVE_PATH,
                    request_data.train_name,
                    f"{request_data.train_name}.yaml",
                ),
            ],
            train_name=request_data.train_name,
        )

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"train_name": request_data.train_name},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps(
            {"train_name": request_data.train_name, "container_name": container_name}
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_train(request_data: schema.PostStopTrain):
    validator.PostStopTrain(train_container=request_data.train_container)
    error_handler = ResponseErrorHandler()

    try:
        train_container = await utils.stop_train(
            container_name_or_id=request_data.train_container
        )

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"train_container": request_data.train_container},
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


@router.post("/")
async def post_train(request_data: schema.PostTrain):
    if not request_data.train_name:
        train_name = f"{get_current_time()}-{generate_uuid()}"
    else:
        train_name = request_data.train_name
    validator.PostTrain(train_path=os.path.join(SAVE_PATH, train_name))

    error_handler = ResponseErrorHandler()

    try:
        train_args = utils.basemodel2dict(data=request_data.train_args)
        train_args["output_dir"] = os.path.join(
            SAVE_PATH, train_name, train_args["finetuning_type"]
        )
        train_args["eval_steps"] = train_args["save_steps"]
        train_args["do_train"] = True

        train_path = utils.add_train_path(path=os.path.join(SAVE_PATH, train_name))
        await utils.write_yaml(
            path=os.path.join(train_path, f"{train_name}.yaml"),
            data=train_args,
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
        content=json.dumps({"train_name": train_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
