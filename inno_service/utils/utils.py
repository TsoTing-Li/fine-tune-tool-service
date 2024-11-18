import uuid
from datetime import datetime


def generate_uuid() -> str:
    """Generate uuid for each data, file, project, iteration, etc.

    Returns:
        new_uuid (str): uuid
    """
    return str(uuid.uuid4())


def get_current_time() -> int:
    now = datetime.now()
    return int(now.timestamp())
