import os

import aiofiles
import aiofiles.os
import orjson


async def async_write_ds_config(file_path: str, ds_config_content: dict):
    ds_config_data = orjson.dumps(ds_config_content, option=orjson.OPT_INDENT_2)

    async with aiofiles.open(file_path, "wb") as af:
        await af.write(ds_config_data)


async def async_load_bytes(content: bytes) -> None:
    try:
        orjson.loads(content)
    except orjson.JSONDecodeError:
        raise TypeError("Invalid JSON format") from None


async def async_write_file_chunk(file_content: bytes, file_path: str, chunk_size: int):
    async with aiofiles.open(file_path, "wb") as af:
        for i in range(0, len(file_content), chunk_size):
            chunk = file_content[i : i + chunk_size]
            await af.write(chunk)


async def async_check_path_exists(file_name: str) -> bool:
    is_exists = await aiofiles.os.path.exists(file_name)
    return is_exists


async def async_list_ds_config(path: str) -> list:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


async def async_preview_ds_config(path: str) -> dict:
    try:
        async with aiofiles.open(path, "rb") as af:
            ds_config_content = await af.read()

        return orjson.loads(ds_config_content)

    except FileNotFoundError:
        raise FileNotFoundError(f"{path} does not exists") from None

    except orjson.JSONDecodeError:
        raise TypeError("Invalid JSON format") from None


async def async_delete_file(file_name: str) -> None:
    try:
        await aiofiles.os.remove(file_name)

    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} does not exists") from None
