import json
import os

from fastapi import APIRouter, Response, status

from inno_service.routers.quantize import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler

QUANTIZE_PATH = os.getenv("QUANTIZE_PATH", "/app/quantize")
TRAIN_CONFIG_PATH = os.getenv("TRAIN_CONFIG_PATH", "/app/train_config")

router = APIRouter(prefix="/quantize", tags=["Quantize"])


@router.post("/start/")
async def post_quantize(request_data: schema.PostStartQuantize):
    error_handler = ResponseErrorHandler()
    try:
        checkpoint_path, finetune_type = await utils.get_quantize_args(
            os.path.join(TRAIN_CONFIG_PATH, f"{request_data.quantize_name}.yaml")
        )
        print(
            f"quantize_name: {request_data.quantize_name}, finetune_type: {finetune_type}"
        )
        checkpoint_path = validator.PostStartQuantize(
            checkpoint_path=checkpoint_path
        ).checkpoint_path

        container_ids = await utils.quantize_as_gguf(
            quantize_service_url="http://127.0.0.1:8002/gguf",
            quantize_name=request_data.quantize_name,
            checkpoint_path=checkpoint_path,
            output_path=os.path.join(QUANTIZE_PATH, request_data.quantize_name),
            finetune_type=finetune_type,
        )
        result = {
            "quantize_name": request_data.quantize_name,
            "finetune_type": finetune_type,
        }
        result.update(container_ids)

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"quantize_name": request_data.quantize_name},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps(result),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
