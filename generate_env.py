from argparse import ArgumentParser, Namespace
from pathlib import Path

import tomllib


def build_parse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--redis_username", required=False, type=str, default="admin")
    parser.add_argument("--redis_password", required=False, type=str, default="admin")
    parser.add_argument("--mount_path", required=False, type=str, default="")
    return parser


def main(args: Namespace):
    cwd = Path.cwd()
    pyproject_file = cwd / "pyproject.toml"
    env_temp = cwd / ".env.template"
    env_file = cwd / ".env"

    if not pyproject_file.exists():
        raise FileNotFoundError("pyproject.toml not found")

    if not env_temp.exists():
        raise FileNotFoundError(".env.template not found")

    with pyproject_file.open("rb") as f:
        pyproject = tomllib.load(f)

    version = pyproject["project"]["version"]

    with env_temp.open() as f:
        env_temp_content = f.read()
        env_temp_content = env_temp_content.replace("{project_version}", version)
        env_temp_content = env_temp_content.replace(
            "{redis_username}", args.redis_username
        )
        env_temp_content = env_temp_content.replace(
            "{redis_password}", args.redis_password
        )
        env_temp_content = env_temp_content.replace("{mount_path}", args.mount_path)

    with env_file.open("w") as f:
        f.write(env_temp_content)

    print(f"Generated .env file with VERSION={version}")


if __name__ == "__main__":
    args = build_parse().parse_args()
    main(args=args)
