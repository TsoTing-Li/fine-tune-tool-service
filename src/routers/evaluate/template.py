import re
from typing import Union

from pydantic import BaseModel


class EvalLogTemplate(BaseModel):
    eval_progress: Union[float, None] = None
    current_task: Union[str, None] = None
    ori: Union[str, None] = None

    _attach_patterns = {
        "eval_progress": {
            "pattern": re.compile(r"Requesting API:\s+\d+%\s(\d+)/(\d+)"),
        },
        "current_task": {
            "pattern": re.compile(r"Building contexts for (\w+) on rank (\d+)"),
            "handler": lambda match: match.group(1).strip(),
        },
    }

    _log_patterns = {
        "current_task": {
            "pattern": re.compile(r"Building contexts for (\w+) on rank (\d+)"),
            "handler": lambda match: match.group(1).strip(),
        },
    }

    def set_first_task(self, first_task: str) -> None:
        self.current_task = first_task

    def _get_eval_progress(
        self, parse_current_request: int, total_requests: int
    ) -> float:
        if total_requests != 0:
            return round(parse_current_request / total_requests, 3)
        else:
            return 0.0

    def parse_eval_attach(self, stdout: str):
        for key, value in self._attach_patterns.items():
            match = self._attach_patterns[key]["pattern"].search(stdout)
            if match:
                if key == "eval_progress":
                    result = self._get_eval_progress(
                        parse_current_request=int(match.group(1)),
                        total_requests=int(match.group(2)),
                    )
                else:
                    result = value["handler"](match)

                setattr(self, key, result)
        self.ori = stdout
        return self

    def parse_eval_log(self, stdout: str):
        match = self._log_patterns["current_task"]["pattern"].search(stdout)
        if match:
            result = self._log_patterns["current_task"]["handler"](match)
            self.current_task = result

        self.ori = stdout
        return self
