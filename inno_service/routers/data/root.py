import json
import os
from typing import Annotated, Literal, Union

import orjson
from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status
from fastapi.exceptions import HTTPException

from inno_service import thirdparty
from inno_service.config import params
from inno_service.routers.data import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler
from inno_service.utils.logger import accel_logger
from inno_service.utils.utils import generate_uuid, get_current_time

MAX_FILE_SIZE = 1024 * 1024 * 5
DATASET_INFO_FILE = "dataset_info.json"

router = APIRouter(prefix="/data", tags=["Data"])


@router.post("/")
async def add_dataset(
    dataset_name: str = Form(...),
    load_from: Annotated[Literal["file_name", "hf_hub_url"], Form(...)] = "file_name",
    dataset_src: str = Form(...),
    subset: str = Form(None),
    split: str = Form(None),
    num_samples: int = Form(None),
    formatting: Annotated[Literal["alpaca", "sharegpt"], Form(...)] = "alpaca",
    prompt: str = Form("instruction"),
    query: str = Form(None),
    response: str = Form("output"),
    history: str = Form(None),
    messages: str = Form("conversations"),
    system: str = Form(None),
    tools: str = Form(None),
    role_tag: str = Form("from"),
    content_tag: str = Form("value"),
    user_tag: str = Form("human"),
    assistant_tag: str = Form("gpt"),
    observation_tag: str = Form("observations"),
    function_tag: str = Form("function_call"),
    system_tag: str = Form("system"),
    dataset_file: Union[UploadFile, None] = File(None),
):
    created_time = get_current_time(use_unix=True)
    dataset_info = {
        "dataset_name": dataset_name,
        "load_from": load_from,
        "dataset_src": dataset_src,
        "subset": subset,
        "split": split,
        "num_samples": num_samples,
        "formatting": formatting,
    }

    if formatting == "alpaca":
        dataset_info["columns"] = {
            "prompt": prompt,
            "query": query,
            "response": response,
            "history": history,
            "system": system,
        }
    elif formatting == "sharegpt":
        dataset_info["columns"] = {
            "messages": messages,
            "system": system,
            "tools": tools,
        }
        dataset_info["tags"] = {
            "role_tag": role_tag,
            "content_tag": content_tag,
            "user_tag": user_tag,
            "assistant_tag": assistant_tag,
            "observation_tag": observation_tag,
            "function_tag": function_tag,
            "system_tag": system_tag,
        }

    request_body = schema.PostData(dataset_info=dataset_info, dataset_file=dataset_file)
    validator.PostData(dataset_name=request_body.dataset_info.dataset_name)
    error_handler = ResponseErrorHandler()

    try:
        if (
            request_body.dataset_info.load_from == "file_name"
            and request_body.dataset_file
        ):
            dataset_bytes = await request_body.dataset_file.read()
            dataset_content = await utils.async_load_bytes(content=dataset_bytes)

            utils.check_dataset_key_value(
                dataset_content=dataset_content,
                dataset_columns=request_body.dataset_info.columns,
                dataset_tags=request_body.dataset_info.tags,
                dataset_format=request_body.dataset_info.formatting,
            )

            request_body.dataset_info.dataset_src = os.path.join(
                params.COMMON_CONFIG.data_path,
                f"{generate_uuid()}-{request_body.dataset_info.dataset_src}",
            )
            await utils.async_write_file_chunk(
                file_content=dataset_bytes,
                file_path=request_body.dataset_info.dataset_src,
                chunk_size=MAX_FILE_SIZE,
            )
        else:
            utils.pull_dataset_from_hf(
                dataset_name=request_body.dataset_info.dataset_src,
                subset=request_body.dataset_info.subset,
                split=request_body.dataset_info.split,
            )

        add_content = await utils.async_add_dataset_info(
            dataset_info_file=os.path.join(
                params.COMMON_CONFIG.data_path, DATASET_INFO_FILE
            ),
            dataset_info=request_body.dataset_info,
        )

    except (TypeError, KeyError, ValueError) as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_FORM],
            msg=f"{e}",
            input={"dataset_src": request_body.dataset_info.dataset_src},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=error_handler.errors
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=request_body.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        data_info = {
            "name": request_body.dataset_info.dataset_name,
            "data_args": add_content[request_body.dataset_info.dataset_name],
            "is_used": False,
            "created_time": created_time,
            "modified_time": None,
        }
        await thirdparty.redis.handler.redis_async.client.hset(
            params.TASK_CONFIG.data,
            request_body.dataset_info.dataset_name,
            orjson.dumps(data_info),
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input={},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps([data_info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/")
async def get_dataset(dataset_name: Annotated[Union[str, None], Query()] = None):
    query_data = schema.GetData(dataset_name=dataset_name)
    validator.GetData(dataset_name=query_data.dataset_name)
    error_handler = ResponseErrorHandler()

    try:
        if query_data.dataset_name:
            info = await thirdparty.redis.handler.redis_async.client.hget(
                params.TASK_CONFIG.data, query_data.dataset_name
            )
            dataset_info = [orjson.loads(info)]
        else:
            info = await thirdparty.redis.handler.redis_async.client.hgetall(
                params.TASK_CONFIG.data
            )
            dataset_info = (
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
        content=json.dumps(dataset_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.put("/")
async def modify_dataset(request_data: schema.PutData):
    modified_time = get_current_time(use_unix=True)
    validator.PutData(
        dataset_name=request_data.dataset_name, new_name=request_data.new_name
    )
    error_handler = ResponseErrorHandler()

    try:
        await utils.modify_dataset_file(
            dataset_info_file=os.path.join(
                params.COMMON_CONFIG.data_path, DATASET_INFO_FILE
            ),
            ori_name=request_data.dataset_name,
            new_name=request_data.new_name,
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
        info = await thirdparty.redis.handler.redis_async.client.hget(
            params.TASK_CONFIG.data, request_data.dataset_name
        )
        dataset_info = orjson.loads(info)
        dataset_info["name"] = request_data.new_name
        dataset_info["modified_time"] = modified_time
        await thirdparty.redis.handler.redis_async.client.hset(
            params.TASK_CONFIG.data, request_data.new_name, orjson.dumps(dataset_info)
        )
        await thirdparty.redis.handler.redis_async.client.hdel(
            params.TASK_CONFIG.data, request_data.dataset_name
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
        content=json.dumps([dataset_info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.delete("/")
async def delete_dataset(dataset_name: Annotated[str, Query(...)]):
    query_data = schema.DeleteData(dataset_name=dataset_name)
    validator.DelData(dataset_name=query_data.dataset_name)
    error_handler = ResponseErrorHandler()

    try:
        del_dataset_info = await thirdparty.redis.handler.redis_async.client.hget(
            params.TASK_CONFIG.data, query_data.dataset_name
        )
        del_dataset_info = orjson.loads(del_dataset_info)
        await thirdparty.redis.handler.redis_async.client.hdel(
            params.TASK_CONFIG.data, query_data.dataset_name
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        await utils.async_del_dataset(
            dataset_info_file=os.path.join(
                params.COMMON_CONFIG.data_path, DATASET_INFO_FILE
            ),
            del_dataset_name=query_data.dataset_name,
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"dataset_name": query_data.dataset_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps([del_dataset_info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
