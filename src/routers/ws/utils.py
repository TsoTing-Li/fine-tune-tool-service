import re

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
