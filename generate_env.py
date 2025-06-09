from argparse import ArgumentParser, Namespace
from pathlib import Path

try:
    import tomllib

    _is_stdlib = True
except ModuleNotFoundError:
    import toml as tomllib

    _is_stdlib = False


def build_parse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--redis_username", required=False, type=str, default="admin")
    parser.add_argument("--redis_password", required=False, type=str, default="admin")
    parser.add_argument("--mount_path", required=False, type=str, default="")
    return parser


def main(args: Namespace):
    cwd = Path.cwd()
    pyproject_file = cwd / "pyproject.toml"
    version_file = cwd / "version.toml"
    env_temp = cwd / ".env.template"
    env_file = cwd / ".env"

    if not pyproject_file.exists():
        raise FileNotFoundError("pyproject.toml not found")

    if not version_file.exists():
        raise FileNotFoundError("version.toml not found")

    if not env_temp.exists():
        raise FileNotFoundError(".env.template not found")

    mode = "rb" if _is_stdlib else "r"
    with pyproject_file.open(mode) as f:
        pyproject_file_content = tomllib.load(f)

    project_version = pyproject_file_content["project"]["version"]

    with version_file.open(mode) as f:
        version_file_content = tomllib.load(f)

    assert project_version == version_file_content["project"]["version"], (
        "project version does not match env version"
    )

    placeholders = {
        "project_name": version_file_content["project"]["name"].lower(),
        "docker_username": version_file_content["docker"]["info"]["username"],
        "docker_base_repo": version_file_content["docker"]["info"]["base_repo"],
        "redis_username": args.redis_username,
        "redis_password": args.redis_password,
        "mount_path": args.mount_path,
    }

    for service_key, service_value in version_file_content["services"].items():
        placeholders[f"{service_key.replace('-', '_')}_name"] = service_value["name"]
        placeholders[f"{service_key.replace('-', '_')}_version"] = service_value[
            "version"
        ]

    for service_key, service_value in version_file_content["shared"].items():
        placeholders[f"{service_key.replace('-', '_')}_name"] = service_value["name"]
        placeholders[f"{service_key.replace('-', '_')}_version"] = service_value[
            "version"
        ]

    for thirdparty_key, thirdparty_value in version_file_content["thirdparty"].items():
        placeholders[f"{thirdparty_key.replace('-', '_')}_name"] = thirdparty_value[
            "name"
        ]
        placeholders[f"{thirdparty_key.replace('-', '_')}_version"] = thirdparty_value[
            "version"
        ]

    with env_temp.open() as f:
        env_temp_content = f.read()

        for key, value in placeholders.items():
            env_temp_content = env_temp_content.replace(f"{{{key}}}", value)

    with env_file.open("w") as f:
        f.write(env_temp_content)

    print(f"Generated .env file with VERSION={project_version}")


if __name__ == "__main__":
    args = build_parse().parse_args()

    try:
        main(args=args)
    except FileNotFoundError as e:
        print(f"{e}")
    except AssertionError as e:
        print(f"{e}")
    except Exception as e:
        print(f"{e}")
