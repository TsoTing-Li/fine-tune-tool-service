import json
import os
from typing import Union

import orjson
from fastapi import APIRouter, Form, Query, Response, UploadFile, status
from typing_extensions import Annotated

from inno_service.routers.data import schema, utils
from inno_service.utils.error import ResponseErrorHandler
from inno_service.utils.utils import generate_uuid

MAX_FILE_SIZE = 1024 * 1024 * 5
DATASET_INFO_FILE = "data/dataset_info.json"
DATASET_PATH = os.getenv("DATA_PATH", "/app/data")
os.makedirs(DATASET_PATH, exist_ok=True)

router = APIRouter(prefix="/data", tags=["Data"])


@router.post("/")
async def add_data(
    dataset_info: Annotated[str, Form(...)],
    dataset_file: Union[UploadFile, None] = None,
):
    dataset_info = orjson.loads(dataset_info)
    request_body = schema.PostData(dataset_info=dataset_info, dataset_file=dataset_file)
    dataset_info = request_body.dataset_info

    error_handler = ResponseErrorHandler()

    try:
        if dataset_info.load_from == "file_name" and dataset_file:
            dataset_file = request_body.dataset_file
            dataset_file = await dataset_file.read()

            dataset_content = await utils.async_load_bytes(content=dataset_file)

            await utils.async_check_dataset_key_value(
                dataset_content=dataset_content,
                dataset_columns=dataset_info.columns,
                dataset_tags=dataset_info.tags,
                dataset_format=dataset_info.formatting,
            )

            dataset_info.dataset_src = os.path.join(
                DATASET_PATH, f"{generate_uuid()}-{dataset_info.dataset_src}"
            )
            await utils.async_write_file_chunk(
                file_content=dataset_file,
                file_path=dataset_info.dataset_src,
                chunk_size=MAX_FILE_SIZE,
            )

        await utils.async_add_dataset_info(
            dataset_info_file=DATASET_INFO_FILE, dataset_info=dataset_info
        )

    except (TypeError, KeyError, ValueError) as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_FORM],
            msg=f"{e}",
            input={"dataset_src": dataset_info.dataset_src},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json",
        )

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input=dict(),
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({"dataset_name": dataset_info.dataset_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.delete("/")
async def delete_data(dataset_name: Annotated[str, Query(...)]):
    dataset_name = schema.DeleteData(dataset_name=dataset_name).dataset_name

    error_handler = ResponseErrorHandler()

    try:
        await utils.async_del_dataset_info(
            dataset_info_file=DATASET_INFO_FILE, del_dataset_name=dataset_name
        )

    except (FileNotFoundError, ValueError) as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg=str(e),
            input={"dataset_name": dataset_name},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"dataset_name": dataset_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
