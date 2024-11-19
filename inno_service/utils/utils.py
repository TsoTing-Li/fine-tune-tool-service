import uuid
from datetime import datetime


def generate_uuid() -> str:
    """Generate uuid for each data, file, project, iteration, etc.

    Returns:
        new_uuid (str): uuid
    """
    return str(uuid.uuid4())


def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")
