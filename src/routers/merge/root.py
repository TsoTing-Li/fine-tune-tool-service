import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from src.routers.merge import schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/merge", tags=["Merge"])

MERGE_PATH = os.getenv("MERGE_PATH", "/app/merge")
SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


@router.post("/start/")
async def start_merge(request_data: schema.PostStartMerge):
    validator.PostStartMerge(merge_name=request_data.merge_name)
    error_handler = ResponseErrorHandler()

    try:
        model_args = await utils.get_model_args(
            path=os.path.join(
                SAVE_PATH, request_data.merge_name, f"{request_data.merge_name}.yaml"
            )
        )
        merge_path = os.path.join(
            MERGE_PATH, request_data.merge_name, f"{request_data.merge_name}.yaml"
        )
        update_data = {
            "model_name_or_path": model_args["model_name_or_path"],
            "adapter_name_or_path": model_args["output_dir"],
            "finetuning_type": model_args["finetuning_type"],
            "template": model_args["template"],
            "export_dir": f"{os.path.join(MERGE_PATH, request_data.merge_name)}",
            "export_size": request_data.export_size,
            "export_device": request_data.export_device,
            "export_legacy_format": request_data.export_legacy_format,
        }
        await utils.generate_merge_yaml(path=merge_path, update_data=update_data)
        merge_container_name = await utils.run_merge(
            image_name=f"{os.environ['USER_NAME']}/{os.environ['REPOSITORY']}:{os.environ['FINE_TUNE_TOOL_TAG']}",
            cmd=["llamafactory-cli", "export", merge_path],
            merge_name=request_data.merge_name,
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"merge_name": request_data.merge_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"merge_container_name": merge_container_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_merge(request_data: schema.PostStopMerge):
    validator.PostStopMerge(merge_container=request_data.merge_container)
    error_handler = ResponseErrorHandler()

    try:
        merge_container = await utils.stop_merge(
            container_name_or_id=request_data.merge_container
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"merge_container": request_data.merge_container},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"merge_container": merge_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
