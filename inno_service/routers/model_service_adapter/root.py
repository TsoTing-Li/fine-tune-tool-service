import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from inno_service.routers.model_service_adapter import schema, utils
from inno_service.utils.error import ResponseErrorHandler

router = APIRouter(prefix="/model-service-adapter", tags=["Model-Service-Adapter"])

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


@router.post("/start/")
async def post_model_service_adapter_start(
    request_data: schema.PostModelServiceAdapterStart,
):
    error_handler = ResponseErrorHandler()

    try:
        model_service_info = await utils.startup_model_service(
            model_name=request_data.model_name
        )

    except Exception as e:
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
        content=json.dumps(model_service_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def post_model_service_adapter_stop(
    request_data: schema.PostModelServiceAdapterStop,
):
    error_handler = ResponseErrorHandler()
    try:
        stopped_container = await utils.stop_model_service(
            container_name=request_data.container_name
        )

    except Exception as e:
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
        content=json.dumps({"stopped_container": stopped_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
