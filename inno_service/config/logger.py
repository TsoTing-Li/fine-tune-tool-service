from pydantic import BaseModel


class LoggerConfig(BaseModel):
    log_folder: str
    log_name: str
    log_limit: int
    log_count: int
