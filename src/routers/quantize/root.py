import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from src.routers.quantize import schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")

router = APIRouter(prefix="/quantize", tags=["Quantize"])


@router.post("/start/")
async def start_quantize(request_data: schema.PostStartQuantize):
    validator.PostStartQuantize(
        checkpoint_path=os.path.join(SAVE_PATH, f"{request_data.quantize_name}")
    )
    error_handler = ResponseErrorHandler()

    try:
        checkpoint_path, finetune_type = await utils.get_quantize_args(
            os.path.join(
                SAVE_PATH,
                request_data.quantize_name,
                f"{request_data.quantize_name}.yaml",
            )
        )

        container_ids = await utils.quantize_as_gguf(
            quantize_service_url="http://127.0.0.1:8002/gguf",
            quantize_name=request_data.quantize_name,
            checkpoint_path=checkpoint_path,
            output_path=os.path.join(SAVE_PATH, request_data.quantize_name, "quantize"),
            finetune_type=finetune_type,
        )
        result = {
            "quantize_name": request_data.quantize_name,
            "finetune_type": finetune_type,
        }
        result.update(container_ids)

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
        content=json.dumps(result),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_quantize(request_data: schema.PostStopQuantize):
    quantize_container = validator.PostStopQuantize(
        quantize_container=request_data.quantize_container
    ).quantize_container
    error_handler = ResponseErrorHandler()

    try:
        quantize_container = await utils.stop_quantize(
            container_name_or_id=quantize_container
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
        content=json.dumps({"quantize_container": quantize_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
