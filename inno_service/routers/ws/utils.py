import re

parse_train_log_patterns = {
    "convert_progress": {
        "pattern": re.compile(
            r"^Converting format of dataset \(num_proc=\d+\):\s+\d+%\|.*?\|\s*(\d+/\d+\s\[.*?\])"
        ),
        "handler": lambda match: match.group(1).strip(),
    },
    "run_tokenizer_progress": {
        "pattern": re.compile(
            r"^Running tokenizer on dataset \(num_proc=\d+\):\s+\d+%\|.*?\|\s*(\d+/\d+\s\[.*?\])"
        ),
        "handler": lambda match: match.group(1).strip(),
    },
    "train_progress": {
        "pattern": re.compile(r"^\d+%\|.*?\|\s*(\d+/\d+\s\[.*?\])"),
        "handler": lambda match, exclude_flag: (
            match.group(1).strip()
            if not exclude_flag
            or "[00:00<?, ?it/s]" not in match.group(1)
            and "[00:00<00:00," not in match.group(1)
            else ""
        ),
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


def parse_train_log(stdout: str, exclude_flag: bool) -> dict:
    log_info = dict()
    for key, value in parse_train_log_patterns.items():
        match = value["pattern"].search(stdout)
        if match:
            if key == "train_progress":
                log_info[key] = value["handler"](match, exclude_flag)
            else:
                log_info[key] = value["handler"](match)
        else:
            log_info[key] = ""

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
