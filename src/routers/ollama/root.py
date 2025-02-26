import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from src.config import params
from src.routers.ollama import schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/ollama", tags=["Ollama"])

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


@router.post("/start/")
async def start_ollama(request_data: schema.PostStartOllama):
    validator.PostStartOllama(model_name=f"{SAVE_PATH}/{request_data.model_name}")
    error_handler = ResponseErrorHandler()

    try:
        container_name = await utils.start_ollama_container(
            image_name=f"{os.environ['USER_NAME']}/{os.environ['OLLAMA_SERVICE_NAME']}:{os.environ['OLLAMA_SERVICE_TAG']}",
            model_name=request_data.model_name,
        )
        await utils.run_ollama_model(
            ollama_url=f"http://{container_name}:{params.OLLAMA_CONFIG.port}",
            model_name=request_data.model_name,
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
        content=json.dumps(
            {
                "ollama_service": f"http://{container_name}:{params.OLLAMA_CONFIG.port}",
                "container_name": container_name,
                "model_name": request_data.model_name,
            }
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_ollama(request_data: schema.PostStopOllama):
    validator.PostStopOllama(ollama_container=request_data.ollama_container)
    error_handler = ResponseErrorHandler()

    try:
        ollama_container = await utils.stop_ollama_container(
            container_name_or_id=request_data.ollama_container
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
        content=json.dumps({"ollama_container": ollama_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
