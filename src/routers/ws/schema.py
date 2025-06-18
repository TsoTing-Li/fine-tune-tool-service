import re
from typing import List, Union

import orjson
from pydantic import BaseModel, Field

from src.routers.ws.thirdparty.hwinfo import validator


class TrainLogTemplate(BaseModel):
    convert_progress: Union[float, None] = None
    run_tokenizer_progress: Union[float, None] = None
    train_progress: Union[float, None] = None
    train_loss: Union[str, None] = None
    eval_loss: Union[str, None] = None
    ori: Union[str, None] = None

    _patterns = {
        "convert_progress": {
            "pattern": re.compile(
                r"^Converting format of dataset \(num_proc=\d+\):\s+\d+%\|.*?\|\s*(\d+)/(\d+)"
            ),
            "handler": lambda match: round(
                float(match.group(1)) / float(match.group(2)), 3
            ),
        },
        "run_tokenizer_progress": {
            "pattern": re.compile(
                r"^Running tokenizer on dataset \(num_proc=\d+\):\s+\d+%\|.*?\|\s*(\d+)/(\d+)"
            ),
            "handler": lambda match: round(
                float(match.group(1)) / float(match.group(2)), 3
            ),
        },
        "train_progress": {
            "pattern": re.compile(r"^\d+%\|.*?\|\s*(\d+)/(\d+)"),
        },
        "train_loss": {
            "pattern": re.compile(r"{'loss':.*?}"),
            "handler": lambda match: match.group(0).strip(),
        },
        "eval_loss": {
            "pattern": re.compile(r"{'eval_loss':.*?}"),
            "handler": lambda match: match.group(0).strip(),
        },
    }

    def get_total_steps(self, log: str) -> int:
        match = re.search(r"Total optimization steps\s*=\s*(\d+)", log)
        if match:
            return int(match.group(1))

    def get_train_progress(
        self,
        parse_current_step: int,
        parse_total_steps: int,
        last_train_progress: float,
        total_steps: int,
    ) -> float:
        if parse_total_steps == total_steps:
            return round(parse_current_step / parse_total_steps, 3)
        else:
            return last_train_progress

    def parse_train_log(
        self, stdout: str, last_train_progress: float, total_steps: int
    ):
        for key, value in self._patterns.items():
            match = value["pattern"].search(stdout)
            if match:
                if key == "train_progress":
                    result = self.get_train_progress(
                        parse_current_step=int(match.group(1)),
                        parse_total_steps=int(match.group(2)),
                        last_train_progress=last_train_progress,
                        total_steps=total_steps,
                    )
                else:
                    result = value["handler"](match)

                setattr(self, key, result)
        self.ori = stdout
        return self


class EvalLogTemplate(BaseModel):
    eval_progress: Union[float, None] = None
    current_task: Union[str, None] = None
    ori: Union[str, None] = None

    _patterns = {
        "eval_progress": {
            "pattern": re.compile(r"Requesting API:\s\s\d+%\s(\d+)/(\d+)"),
        },
        "current_task": {
            "pattern": re.compile(r"Building contexts for (\w+) on rank (\d+)"),
            "handler": lambda match: match.group(1).strip(),
        },
    }

    def get_total_requests(self, log: str) -> int:
        match = re.search(r"Cached requests: (\d+), Requests remaining: (\d+)", log)
        if match:
            return int(match.group(2))

    def get_eval_progress(self, parse_current_request: int, total_requests: int):
        return round(parse_current_request / total_requests, 3)

    def parse_eval_log(self, stdout: str, return_flag: bool, total_requests: int):
        for key, value in self._patterns.items():
            match = value["pattern"].search(stdout)
            if match:
                if key == "eval_progress":
                    result = self.get_eval_progress(
                        parse_current_request=int(match.group(1)),
                        total_requests=total_requests,
                    )
                else:
                    result = value["handler"](match)

                setattr(self, key, result)
        self.ori = stdout
        return self


class HwInfoTemplate(BaseModel):
    cpu: validator.CPUTemplate = Field(default_factory=validator.CPUTemplate)
    gpus: List[validator.GPUTemplate] = list()
    disk: validator.DiskTemplate = Field(default_factory=validator.DiskTemplate)
    memory: validator.MemoryTemplate = Field(default_factory=validator.MemoryTemplate)

    _patterns = {
        "hw_info": {
            "pattern": re.compile(r"INFO-(.*)"),
            "handler": lambda match: match.group(1).strip(),
        }
    }

    def parse_hwinfo_log(self, stdout: str):
        match = self._patterns["hw_info"]["pattern"].search(stdout)
        if match:
            result = self._patterns["hw_info"]["handler"](match)
            result = re.sub(
                r"(?<![a-zA-Z0-9_])'([^']*?)'(?![a-zA-Z0-9_])", r'"\1"', result
            )
            hw_info = orjson.loads(result)

            self.cpu = validator.CPUTemplate(**hw_info["cpu"])
            self.gpus = [validator.GPUTemplate(**gpu) for gpu in hw_info["gpus"]]
            self.disk = validator.DiskTemplate(**hw_info["disk"])
            self.memory = validator.MemoryTemplate(**hw_info["memory"])

        return self
