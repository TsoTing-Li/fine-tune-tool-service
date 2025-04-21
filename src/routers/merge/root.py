import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from src.config.params import (
    COMMON_CONFIG,
    DOCKERNETWORK_CONFIG,
    FINETUNETOOL_CONFIG,
)
from src.routers.merge import schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import (
    assemble_image_name,
)

router = APIRouter(prefix="/merge", tags=["Merge"], include_in_schema=False)


@router.post("/start/")
async def post_start_merge(request_data: schema.PostStartMerge):
    validator.PostStartMerge(merge_name=request_data.merge_name)
    error_handler = ResponseErrorHandler()

    try:
        command = [
            f"llamafactory-cli export {os.path.join(COMMON_CONFIG.save_path, request_data.merge_name, 'export.yaml')}"
        ]

        container_name = await utils.run_merge(
            image_name=assemble_image_name(
                username=COMMON_CONFIG.username,
                repository=COMMON_CONFIG.repository,
                tag=FINETUNETOOL_CONFIG.tag,
            ),
            cmd=["sh", "-c", " && ".join(command)],
            docker_network_name=DOCKERNETWORK_CONFIG.network_name,
            merge_name=request_data.merge_name,
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(
            {"merge_name": request_data.merge_name, "container_name": container_name}
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def post_stop_merge(request_data: schema.PostStopMerge):
    error_handler = ResponseErrorHandler()

    try:
        stopped_container = await utils.stop_merge(
            container_name_or_id=request_data.merge_container_name
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
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
