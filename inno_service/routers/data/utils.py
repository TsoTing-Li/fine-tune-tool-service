import os
from typing import List, Union

import aiofiles
import aiofiles.os
import orjson
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from typing_extensions import Literal

from inno_service.routers.data.schema import Columns, DatasetInfo, Tags

MAX_JOBS = os.environ["MAX_JOBS"]


def get_json_decode_error_pos(
    ori_content: bytes, err_pos, context_range: int = 50
) -> str:
    err_start = max(err_pos - context_range, 0)
    err_end = min(err_pos + context_range, len(ori_content))
    return ori_content[err_start:err_end].decode()


async def async_check_path_exists(file_name: str) -> bool:
    is_exists = await aiofiles.os.path.exists(file_name)
    return is_exists


async def async_write_file_chunk(file_content: bytes, file_path: str, chunk_size: int):
    async with aiofiles.open(file_path, "wb") as af:
        for i in range(0, len(file_content), chunk_size):
            chunk = file_content[i : i + chunk_size]
            await af.write(chunk)


async def async_write_dataset_info_file(
    dataset_info_file: str, dataset_info_content: dict
) -> dict:
    try:
        async with aiofiles.open(dataset_info_file, "wb") as af:
            await af.write(
                orjson.dumps(dataset_info_content, option=orjson.OPT_INDENT_2)
            )

        return dataset_info_content

    except orjson.JSONEncodeError:
        raise TypeError("Failed to serialize data into JSON format") from None


async def async_get_dataset_info_file(dataset_info_file: str) -> dict:
    try:
        async with aiofiles.open(dataset_info_file) as af:
            content = await af.read()
            dataset_info_content = orjson.loads(content)
        return dataset_info_content

    except orjson.JSONDecodeError:
        raise TypeError("Invalid JSON format in dataset info file") from None


def check_sharegpt_format(
    dataset_content: List[dict],
    dataset_columns: Columns,
    dataset_tags: Union[Tags, None],
):
    columns_keys = {
        key
        for key in (
            dataset_columns.messages,
            dataset_columns.system,
            dataset_columns.tools,
        )
        if key is not None
    }

    tags_keys = {
        key
        for key in (
            dataset_tags.role_tag,
            dataset_tags.content_tag,
        )
        if key is not None
    }
    tags_values = {
        dataset_tags.user_tag,
        dataset_tags.assistant_tag,
        dataset_tags.observation_tag,
        dataset_tags.function_tag,
    }

    for column in dataset_content:
        if not columns_keys.issubset(column.keys()):
            raise KeyError("Invalid column key in required dataset")

        if not isinstance(column[dataset_columns.messages], list):
            raise TypeError(
                f"Value format in '{dataset_columns.messages}' must be list"
            )

        for tag in column[dataset_columns.messages]:
            if not tags_keys.issubset(tag.keys()):
                raise KeyError("Invalid tag key in required dataset")

            if tag[dataset_tags.user_tag] not in tags_values:
                raise ValueError(
                    f"Invalid '{dataset_tags.user_tag}' value: {tag[dataset_tags.user_tag]}"
                )

            if not isinstance(tag[dataset_tags.content_tag], str):
                raise TypeError(
                    f"Value format in '{dataset_tags.content_tag}' must be string"
                )


def check_alpaca_format(
    dataset_content: List[dict],
    dataset_columns: Columns,
):
    columns_keys = {
        key
        for key in (
            dataset_columns.prompt,
            dataset_columns.query,
            dataset_columns.response,
            dataset_columns.system,
            dataset_columns.history,
        )
        if key is not None
    }

    for column in dataset_content:
        if not columns_keys.issubset(column.keys()):
            raise KeyError("Invalid column key in required dataset")


def check_dataset_key_value(
    dataset_content: List[dict],
    dataset_columns: Columns,
    dataset_tags: Union[Tags, None],
    dataset_format: Literal["alpaca", "sharegpt"],
) -> None:
    if dataset_format == "alpaca":
        check_alpaca_format(
            dataset_content=dataset_content, dataset_columns=dataset_columns
        )

    elif dataset_format == "sharegpt":
        check_sharegpt_format(
            dataset_content=dataset_content,
            dataset_columns=dataset_columns,
            dataset_tags=dataset_tags,
        )


async def async_add_dataset_info(
    dataset_info_file: str, dataset_info: DatasetInfo, current_time: int
) -> dict:
    is_exists = await async_check_path_exists(file_name=dataset_info_file)
    dataset_info_content = dict()

    if is_exists:
        dataset_info_content = await async_get_dataset_info_file(
            dataset_info_file=dataset_info_file
        )

        if dataset_info.dataset_name in dataset_info_content.keys():
            raise ValueError(
                f"'dataset_name': '{dataset_info.dataset_name}' already in used"
            )

    update_data = {
        dataset_info.dataset_name: {
            dataset_info.load_from: dataset_info.dataset_src,
            "formatting": dataset_info.formatting,
            "num_samples": dataset_info.num_samples,
            "split": dataset_info.split,
            "columns": dataset_info.columns.model_dump(),
            "time": current_time,
        }
    }
    if dataset_info.formatting == "sharegpt":
        update_data[dataset_info.dataset_name]["tags"] = {
            "tags": dataset_info.tags.model_dump()
        }
    dataset_info_content.update(update_data)

    dataset_info_content = await async_write_dataset_info_file(
        dataset_info_file=dataset_info_file, dataset_info_content=dataset_info_content
    )

    return {dataset_info.dataset_name: dataset_info_content[dataset_info.dataset_name]}


async def async_load_bytes(content: bytes):
    try:
        json_content = orjson.loads(content)
        return json_content
    except orjson.JSONDecodeError as e:  # noqa: F841
        # raise TypeError(
        #     get_json_decode_error_pos(ori_content=content, err_pos=e.pos)
        # ) from None
        raise TypeError("Invalid JSON format") from None


def pull_dataset_from_hf(
    dataset_name: str, subset: Union[str, None] = None, split: str = "train"
):
    try:
        load_dataset(
            dataset_name,
            subset,
            split=split,
            num_proc=int(MAX_JOBS),
            trust_remote_code=True,
        )
    except DatasetNotFoundError as e:
        raise ValueError(f"{e}") from None


async def get_dataset_info(dataset_info_file: str, dataset_name: str) -> dict:
    is_exists = await async_check_path_exists(file_name=dataset_info_file)

    if not is_exists:
        raise FileNotFoundError("There are currently no dataset") from None

    dataset_info_content = await async_get_dataset_info_file(
        dataset_info_file=dataset_info_file
    )
    if dataset_name:
        if dataset_name not in dataset_info_content.keys():
            raise ValueError("dataset not found") from None
        return {dataset_name: dataset_info_content[dataset_name]}

    return dataset_info_content


async def modify_dataset_file(
    dataset_info_file: str, ori_name: str, new_name: str
) -> dict:
    is_exists = await async_check_path_exists(file_name=dataset_info_file)

    if not is_exists:
        raise FileNotFoundError("There are currently no dataset") from None

    dataset_info_content = await async_get_dataset_info_file(
        dataset_info_file=dataset_info_file
    )

    if ori_name not in dataset_info_content.keys():
        raise ValueError(f"'dataset_name': {ori_name} does not exists") from None

    if new_name in dataset_info_content.keys():
        raise KeyError(f"'new_name': {new_name} already in used") from None

    key_mapping = {ori_name: new_name}
    new_content = {key_mapping.get(k, k): v for k, v in dataset_info_content.items()}

    current_content = await async_write_dataset_info_file(
        dataset_info_file=dataset_info_file, dataset_info_content=new_content
    )

    return {new_name: current_content[new_name]}


async def async_delete_file(file_name: str) -> None:
    await aiofiles.os.remove(file_name)


async def async_del_dataset(dataset_info_file: str, del_dataset_name: str) -> dict:
    is_exists = await async_check_path_exists(file_name=dataset_info_file)

    if not is_exists:
        raise FileNotFoundError("There are currently no dataset") from None

    dataset_info_content = await async_get_dataset_info_file(
        dataset_info_file=dataset_info_file
    )

    if del_dataset_name not in dataset_info_content.keys():
        raise ValueError(
            f"'dataset_name': {del_dataset_name} does not exists"
        ) from None

    if "file_name" in dataset_info_content[del_dataset_name].keys():
        await async_delete_file(dataset_info_content[del_dataset_name]["file_name"])

    del_content = dataset_info_content[del_dataset_name]
    del dataset_info_content[del_dataset_name]

    await async_write_dataset_info_file(
        dataset_info_file=dataset_info_file, dataset_info_content=dataset_info_content
    )

    return del_content
