import json
import os
from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from src.config.params import COMMON_CONFIG
from src.routers.deepspeed import adapter, schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

MAX_FILE_SIZE = 1024 * 1024 * 5

router = APIRouter(prefix="/deepspeed", tags=["DeepSpeed"])


@router.post("/default/", include_in_schema=False)
async def add_deepspeed_default(request_data: schema.PostDeepSpeedDefault):
    error_handler = ResponseErrorHandler()

    ds_config_adapter = adapter.PostDeepSpeedDefault(
        stage=request_data.stage,
        enable_offload=request_data.enable_offload,
        offload_device=request_data.offload_device,
        nvme_path=COMMON_CONFIG.nvme_path
        if request_data.offload_device == "nvme"
        else None,
    )
    target_model = ds_config_adapter.get_target_model()

    ds_config_content = {
        "train_batch_size": target_model.train_batch_size,
        "train_micro_batch_size_per_gpu": target_model.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": target_model.gradient_accumulation_steps,
        "gradient_clipping": target_model.gradient_clipping,
        "zero_allow_untested_optimizer": target_model.zero_allow_untested_optimizer,
        "fp16": target_model.fp16.model_dump(),
        "bf16": target_model.bf16.model_dump(),
        "zero_optimization": target_model.zero_optimization.model_dump(),
    }

    try:
        path = os.path.join(
            COMMON_CONFIG.save_path,
            request_data.train_name,
            f"ds_config_{request_data.train_name}.json",
        )
        await utils.async_write_ds_config(
            file_path=path, ds_config_content=ds_config_content
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
        content=json.dumps({"ds_path": path}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/file/", include_in_schema=False)
async def add_deepspeed_file(
    ds_file: UploadFile = File(...), train_name: str = Form(...)
):
    request_data = schema.PostDeepSpeedFile(train_name=train_name, ds_file=ds_file)
    error_handler = ResponseErrorHandler()

    try:
        ds_file = await request_data.ds_file.read()
        await utils.async_load_bytes(content=ds_file)

        ds_file_path = os.path.join(
            COMMON_CONFIG.save_path,
            request_data.train_name,
            f"ds_config_{request_data.ds_file.filename}",
        )

        await utils.async_write_file_chunk(
            file_content=ds_file,
            file_path=ds_file_path,
            chunk_size=MAX_FILE_SIZE,
        )

    except TypeError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_FORM],
            msg=f"{e}",
            input={"ds_file": request_data.ds_file.filename},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_handler.errors,
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={
                "ds_file": request_data.ds_file.filename,
                "train_name": request_data.train_name,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"ds_path": ds_file_path}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/preview/")
async def preview_deepspeed_config(ds_file_name: Annotated[str, Query(...)]):
    query_data = schema.GetDeepSpeedPreview(ds_file_name=ds_file_name)
    validator.GetDeepSpeedPreview(ds_file_name=query_data.ds_file_name)
    error_handler = ResponseErrorHandler()

    try:
        ds_config = await utils.async_preview_ds_config(
            path=os.path.join(query_data.ds_file_name)
        )

    except FileNotFoundError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg="ds_file not found",
            input={"ds_file_name": query_data.ds_file_name},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_handler.errors,
        ) from None

    except TypeError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg="ds_file is not JSON format",
            input={"ds_file_name": query_data.ds_file_name},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_handler.errors,
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(ds_config),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
