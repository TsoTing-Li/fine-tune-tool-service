import re
from typing import Union

from pydantic import BaseModel


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
