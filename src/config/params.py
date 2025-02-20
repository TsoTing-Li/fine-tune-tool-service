import os

from src.config.common import CommonConfig
from src.config.docker_network import DockerNetworkConfig
from src.config.finetune_tool import FineTuneToolConfig
from src.config.hw_info import HwInfoConfig
from src.config.lm_eval import LmEvalConfig
from src.config.logger import LoggerConfig
from src.config.quantize_service import QuantizeServiceConfig
from src.config.redis import RedisConfig
from src.config.task import TaskConfig
from src.config.vllm import VllmConfig

PROJECT_NAME = os.getenv("PROJECT_NAME", "AccelTune")
ACCELTUNE_SETTING = {
    "common": {
        "username": os.environ["USER_NAME"],
        "repository": os.environ["REPOSITORY"],
        "hf_home": os.environ["HF_HOME"],
        "root_path": os.environ["ROOT_PATH"],
        "max_jobs": os.environ["MAX_JOBS"],
        "workspace_path": os.environ["WS"],
        "save_path": os.getenv("SAVE_PATH", "/app/saves"),
        "nvme_path": os.getenv("NVME_PATH", "/mnt/nvme"),
        "data_path": os.getenv("DATA_PATH", "/app/data"),
    },
    "task": {
        "data": "DATA",
        "train": "TRAIN",
        "merge": "MERGE",
        "eval": "EVAL",
        "chat": "CHAT",
        "quantize": "QUANTIZE",
        "accelbrain_device": "ACCELBRAIN_DEVICE",
    },
    "logger": {
        "log_folder": os.environ["ACCELTUNE_LOG_FOLDER"],
        "log_name": os.environ["ACCELTUNE_LOG_NAME"],
        "log_limit": os.environ["ACCELTUNE_LOG_LIMIT"],
        "log_count": os.environ["ACCELTUNE_LOG_COUNT"],
    },
    "redis": {
        "name": os.getenv("REDIS_NAME"),
        "username": os.getenv("REDIS_USERNAME"),
        "tag": os.getenv("REDIS_TAG"),
        "host": os.getenv("REDIS_HOST"),
        "port": os.getenv("REDIS_PORT"),
        "password": os.getenv("REDIS_PASSWORD"),
        "container_name": os.getenv("REDIS_CONTAINER_NAME"),
    },
    "vllm": {
        "name": os.getenv("VLLM_SERVICE_NAME"),
        "tag": os.getenv("VLLM_SERVICE_TAG"),
        "host": os.getenv("VLLM_SERVICE_HOST"),
        "port": os.getenv("VLLM_SERVICE_PORT"),
        "container_name": os.getenv("REDIS_CONTAINER_NAME"),
    },
    "lm_eval": {
        "name": os.getenv("LM_EVAL_NAME"),
        "tag": os.getenv("LM_EVAL_TAG"),
    },
    "hw_info": {
        "name": os.getenv("HWINFO_NAME"),
        "tag": os.getenv("HWINFO_TAG"),
        "container_name": os.getenv("HWINFO_CONTAINER_NAME"),
    },
    "quantize_service": {
        "name": os.getenv("QUANTIZE_SERVICE_NAME"),
        "tag": os.getenv("QUANTIZE_SERVICE_TAG"),
        "host": os.getenv("QUANTIZE_SERVICE_HOST"),
        "port": os.getenv("QUANTIZE_SERVICE_PORT"),
        "container_name": os.getenv("QUANTIZE_SERVICE_CONTAINER_NAME"),
        "gguf_tag": os.getenv("QUANTIZE_GGUF_TAG"),
    },
    "fine_tune_tool": {
        "name": os.getenv("FINE_TUNE_TOOL_NAME"),
        "tag": os.getenv("FINE_TUNE_TOOL_TAG"),
    },
}

COMMON_CONFIG = CommonConfig(**ACCELTUNE_SETTING["common"])
LOGGER_CONFIG = LoggerConfig(**ACCELTUNE_SETTING["logger"])
VLLM_CONFIG = VllmConfig(**ACCELTUNE_SETTING["vllm"])
REDIS_CONFIG = RedisConfig(**ACCELTUNE_SETTING["redis"])
HWINFO_CONFIG = HwInfoConfig(**ACCELTUNE_SETTING["hw_info"])
QUANTIZESERVICE_CONFIG = QuantizeServiceConfig(**ACCELTUNE_SETTING["quantize_service"])
FINETUNETOOL_CONFIG = FineTuneToolConfig(**ACCELTUNE_SETTING["fine_tune_tool"])
LMEVAL_CONFIG = LmEvalConfig(**ACCELTUNE_SETTING["lm_eval"])
DOCKERNETWORK_CONFIG = DockerNetworkConfig(network_name=PROJECT_NAME)
TASK_CONFIG = TaskConfig(**ACCELTUNE_SETTING["task"])
