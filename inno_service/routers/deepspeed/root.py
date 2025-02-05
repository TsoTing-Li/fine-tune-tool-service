import json
import os

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
from typing_extensions import Annotated

from inno_service.routers.deepspeed import adapter, schema, utils
from inno_service.utils.error import ResponseErrorHandler

MAX_FILE_SIZE = 1024 * 1024 * 5
NVME_PATH = os.getenv("NVME_PATH", "/mnt/nvme")
SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")

router = APIRouter(prefix="/deepspeed", tags=["DeepSpeed"])


@router.post("/default/")
async def add_deepspeed_default(request_data: schema.PostDeepSpeedDefault):
    error_handler = ResponseErrorHandler()

    ds_config_adapter = adapter.PostDeepSpeedDefault(
        stage=request_data.stage,
        enable_offload=request_data.enable_offload,
        offload_device=request_data.offload_device,
        nvme_path=NVME_PATH if request_data.offload_device == "nvme" else None,
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
            SAVE_PATH, request_data.name, f"ds_config_{request_data.name}.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        await utils.async_write_ds_config(
            file_path=path, ds_config_content=ds_config_content
        )

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=dict(),
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


@router.post("/file/")
async def add_deepspeed_file(ds_file: UploadFile = File(...), name: str = Form(...)):
    request_data = schema.PostDeepSpeedFile(name=name, ds_file=ds_file)
    error_handler = ResponseErrorHandler()

    try:
        ds_file = await request_data.ds_file.read()
        await utils.async_load_bytes(content=ds_file)

        ds_file_path = os.path.join(
            SAVE_PATH,
            request_data.name,
            f"ds_config_{request_data.name}.json",
        )

        os.makedirs(os.path.dirname(ds_file_path), exist_ok=True)
        await utils.async_write_file_chunk(
            file_content=ds_file,
            file_path=ds_file_path,
            chunk_size=MAX_FILE_SIZE,
        )

    except TypeError as e:
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
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=dict(),
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


@router.get("/", tags=["DeepSpeed"])
async def get_all_deepspeed_config():
    error_handler = ResponseErrorHandler()
    try:
        all_ds_config = await utils.async_list_ds_config(SAVE_PATH)

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=dict(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(all_ds_config),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/preview/")
async def preview_ds_config(ds_file_name: Annotated[str, Query(...)]):
    ds_file_name = schema.GetDeepSpeedPreview(ds_file_name=ds_file_name).ds_file_name

    error_handler = ResponseErrorHandler()

    try:
        ds_config = await utils.async_preview_ds_config(
            path=os.path.join(SAVE_PATH, ds_file_name)
        )

    except (FileNotFoundError, TypeError) as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"ds_file_name": ds_file_name},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_handler.errors,
        ) from None

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"ds_file_name": ds_file_name},
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


@router.delete("/")
async def del_deepspeed(ds_file_name: Annotated[str, Query(...)]):
    ds_file_name = schema.DelDeepSpeed(ds_file_name=ds_file_name).ds_file_name

    error_handler = ResponseErrorHandler()

    try:
        await utils.async_delete_file(file_name=os.path.join(SAVE_PATH, ds_file_name))

    except FileNotFoundError as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"ds_file_name": ds_file_name},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_handler.errors,
        ) from None

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"ds_file_name": ds_file_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"ds_file_name": ds_file_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
