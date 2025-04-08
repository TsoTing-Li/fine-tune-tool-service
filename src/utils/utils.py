import time
import uuid
from datetime import datetime
from typing import Tuple


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
