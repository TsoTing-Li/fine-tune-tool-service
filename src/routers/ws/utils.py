import re

parse_train_log_patterns = {
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
        "pattern": re.compile(r"^\d+%\|.*?\|\s*(\d+/\d+)"),
        "handler": lambda match, is_eval, last_train_progress: (
            lambda s: round(int(s.split("/")[0]) / int(s.split("/")[1]), 3)
            if "/" in s
            else None
        )(match.group(1))
        if not is_eval
        else last_train_progress,
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


def parse_train_log(
    log_info: dict,
    stdout: str,
    is_eval: bool,
    last_train_progress: float,
) -> dict:
    for key, value in parse_train_log_patterns.items():
        match = value["pattern"].search(stdout)
        if match:
            if key == "train_progress":
                log_info[key] = value["handler"](match, is_eval, last_train_progress)
            else:
                log_info[key] = value["handler"](match)

    log_info["ori"] = stdout

    return log_info


parse_hw_info_log_patterns = {
    "hw_info": {
        "pattern": re.compile(r"INFO-(.*)"),
        "handler": lambda match: match.group(1).strip(),
    }
}


def parse_hw_info_log(stdout: str) -> str:
    for _, value in parse_hw_info_log_patterns.items():
        match = value["pattern"].search(stdout)
        if match:
            return value["handler"](match)
        else:
            return ""
