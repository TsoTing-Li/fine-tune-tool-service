import os
import time
import uuid
from datetime import datetime
from typing import Tuple

import aiofiles


def generate_uuid() -> str:
    """Generate uuid for each data, file, project, iteration, etc.

    Returns:
        new_uuid (str): uuid
    """
    return str(uuid.uuid4())


def get_current_time() -> Tuple[int, str]:
    return (int(time.time()), datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))


def assemble_image_name(username: str, repository: str, tag: str) -> str:
    if repository == "":
        return f"{username}:{tag}"
    else:
        return f"{username}/{repository}:{tag}"


async def check_dataset_info_file(file_path: str):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if not os.path.exists(file_path):
        async with aiofiles.open(file_path, mode="w") as f:
            await f.write("{}")
