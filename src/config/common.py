from pydantic import BaseModel


class CommonConfig(BaseModel):
    username: str
    repository: str
    hf_home: str
    root_path: str
    max_jobs: int
    workspace_path: str
    save_path: str
    nvme_path: str
    data_path: str
    cache_path: str
