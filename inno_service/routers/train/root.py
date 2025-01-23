import json
import os
from typing import Optional

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
from typing_extensions import Annotated

from inno_service.routers.train import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler
from inno_service.utils.utils import generate_uuid, get_current_time

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")
os.makedirs(SAVE_PATH, exist_ok=True)

router = APIRouter(prefix="/train", tags=["Train"])


@router.post("/start/")
async def start_train(request_data: schema.PostStartTrain):
    validator.PostStartTrain(train_name=request_data.train_name)
    error_handler = ResponseErrorHandler()

    try:
        await utils.async_clear_exists_path(train_name=request_data.train_name)
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
async def post_train(
    train_name: str = Form(None),
    train_args: str = Form(...),
    deepspeed_args: str = Form(None),
    deepspeed_file: UploadFile = File(None),
):
    error_handler = ResponseErrorHandler()
    try:
        train_args = orjson.loads(train_args)
        deepspeed_args = orjson.loads(deepspeed_args) if deepspeed_args else None
    except orjson.JSONDecodeError:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_FORM],
            msg="'train_args' or 'deepspeed_args' must be JSON format",
            input={},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json",
        )

    request_data = schema.PostTrain(
        train_name=train_name,
        train_args=train_args,
        deepspeed_args=deepspeed_args,
        deepspeed_file=deepspeed_file,
    )
    if not request_data.train_name:
        train_name = f"{get_current_time()}-{generate_uuid()}"
    else:
        train_name = request_data.train_name
    validator.PostTrain(train_path=os.path.join(SAVE_PATH, train_name))

    try:
        train_args = utils.basemodel2dict(data=request_data.train_args)
        train_args["output_dir"] = os.path.join(
            SAVE_PATH, train_name, train_args["finetuning_type"]
        )
        train_args["dataset"] = ", ".join(train_args["dataset"])
        train_args["dataset_dir"] = os.path.join(os.getenv("DATA_PATH", "/app/data"))
        train_args["eval_steps"] = train_args["save_steps"]
        train_args["do_train"] = True

        train_path = utils.add_train_path(path=os.path.join(SAVE_PATH, train_name))

        if request_data.deepspeed_args:
            ds_args = request_data.deepspeed_args.model_dump()
            ds_api_response = await utils.call_ds_api(
                name=train_name, ds_args=ds_args, ds_file=request_data.deepspeed_file
            )
            train_args["deepspeed"] = ds_api_response["ds_path"]

        await utils.write_yaml(
            path=os.path.join(train_path, f"{train_name}.yaml"),
            data=train_args,
        )

    except HTTPException as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_FORM],
            msg=f"{e.detail}",
            input={},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=e.status_code,
            media_type="application/json",
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


@router.get("/")
async def get_train(train_name: Optional[Annotated[str, Query("")]] = ""):
    query_data = schema.GetTrain(train_name=train_name)
    validator.GetTrain(train_path=os.path.join(SAVE_PATH, query_data.train_name))
    error_handler = ResponseErrorHandler()

    try:
        train_args_info = await utils.get_train_args(train_name=query_data.train_name)

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"train_name": query_data.train_name},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps(train_args_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.put("/")
async def modify_train(request_data: schema.PutTrain):
    validator.PutTrain(train_path=os.path.join(SAVE_PATH, request_data.train_name))
    error_handler = ResponseErrorHandler()

    try:
        train_args = utils.basemodel2dict(data=request_data.train_args)
        train_args["output_dir"] = os.path.join(
            SAVE_PATH, request_data.train_name, train_args["finetuning_type"]
        )
        train_args["eval_steps"] = train_args["save_steps"]
        train_args["do_train"] = True
        await utils.write_yaml(
            path=os.path.join(
                SAVE_PATH, request_data.train_name, f"{request_data.train_name}.yaml"
            ),
            data=train_args,
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
        content=json.dumps({"train_name": request_data.train_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.delete("/")
async def delete_train(train_name: Annotated[str, Query(...)]):
    query_data = schema.DelTrain(train_name=train_name)
    del_train_path = os.path.join(SAVE_PATH, query_data.train_name)
    validator.DelTrain(train_path=del_train_path)
    error_handler = ResponseErrorHandler()

    try:
        del_train_name = utils.del_train(path=del_train_path)

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"train_name": query_data.train_name},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({"train_name": del_train_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
