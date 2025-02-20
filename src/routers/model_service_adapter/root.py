import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from src.routers.model_service_adapter import schema, utils
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/model-service-adapter", tags=["Model-Service-Adapter"])

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


@router.post("/start/")
async def start_model_service_adapter(
    request_data: schema.PostModelServiceAdapterStart,
):
    error_handler = ResponseErrorHandler()

    try:
        model_service_info = await utils.startup_model_service(
            model_name=request_data.model_name
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
        content=json.dumps(model_service_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_model_service_adapter(
    request_data: schema.PostModelServiceAdapterStop,
):
    error_handler = ResponseErrorHandler()
    try:
        stopped_container = await utils.stop_model_service(
            container_name=request_data.container_name
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
        content=json.dumps({"stopped_container": stopped_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
