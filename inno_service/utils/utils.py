import time
import uuid
from datetime import datetime
from typing import Union


def generate_uuid() -> str:
    """Generate uuid for each data, file, project, iteration, etc.

    Returns:
        new_uuid (str): uuid
    """
    return str(uuid.uuid4())


def get_current_time(use_unix: bool) -> Union[int, str]:
    if use_unix:
        return int(time.time())
    else:
        now = datetime.now()
        return now.strftime("%Y-%m-%d-%H-%M-%S")
