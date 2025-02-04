import json
import os
import traceback

from fastapi import APIRouter, HTTPException, Response, status

from inno_service.routers.ollama import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler

router = APIRouter(prefix="/ollama", tags=["Ollama"])

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")
OLLAMA_SERVICE_PORT = os.getenv("OLLAMA_SERVICE_PORT", 11434)


@router.post("/start/")
async def post_start_ollama(request_data: schema.PostStartOllama):
    validator.PostStartOllama(model_name=f"{SAVE_PATH}/{request_data.model_name}")
    error_handler = ResponseErrorHandler()

    try:
        container_name = await utils.start_ollama_container(
            image_name=f"{os.environ['USER_NAME']}/{os.environ['OLLAMA_SERVICE_NAME']}:{os.environ['OLLAMA_SERVICE_TAG']}",
            model_name=request_data.model_name,
        )
        await utils.run_ollama_model(
            ollama_url=f"http://127.0.0.1:{OLLAMA_SERVICE_PORT}",
            model_name=request_data.model_name,
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
        content=json.dumps(
            {
                "ollama_service": f"http://127.0.0.1:{OLLAMA_SERVICE_PORT}",
                "container_name": container_name,
                "model_name": request_data.model_name,
            }
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def post_stop_ollama(request_data: schema.PostStopOllama):
    validator.PostStopOllama(ollama_container=request_data.ollama_container)
    error_handler = ResponseErrorHandler()

    try:
        ollama_container = await utils.stop_ollama_container(
            container_name_or_id=request_data.ollama_container
        )

    except Exception as e:
        traceback.print_exc()
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
