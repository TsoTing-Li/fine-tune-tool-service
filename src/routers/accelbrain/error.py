from typing import Union

import orjson


class AccelBrainError(Exception):
    def __init__(
        self,
        status_code: int,
        action: str,
        progress: int,
        detail: dict = None,
    ) -> None:
        self.status_code = status_code
        self.action = action
        self.progress = progress
        self.detail = detail if detail is not None else {}

        self.error_data = {
            "AccelBrain": {
                "status": self.status_code,
                "message": {
                    "action": self.action,
                    "progress": self.progress,
                    "detail": self.detail,
                },
            }
        }
        super().__init__(orjson.dumps(self.error_data))

    def __str__(self):
        return orjson.dumps(self.error_data).decode()


class AccelTuneError(Exception):
    def __init__(
        self,
        status_code: int,
        action: str,
        progress: int,
        detail: Union[dict, None],
    ) -> None:
        self.status_code = status_code
        self.action = action
        self.progress = progress
        self.detail = detail

        self.error_data = {
            "AccelTune": {
                "status": self.status_code,
                "message": {
                    "action": self.action,
                    "progress": self.progress,
                    "detail": self.detail,
                },
            }
        }
        super().__init__(orjson.dumps(self.error_data))

    def __str__(self) -> str:
        return orjson.dumps(self.error_data).decode()
