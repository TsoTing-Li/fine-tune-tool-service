# fine-tune-tool-service
AccelTune is a finetuning tool in LLM.

## Pre-requirements
* Docker
  * [Docker 20.10 + ](https://docs.docker.com/engine/install/ubuntu/)
  * Setting docker to group
    ```bash
    sudo usermod -aG docker $USER
    ```

* Nvidia
  * [NVIDIA GPU Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
  * [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#step-1-install-nvidia-container-toolkit)

* generate `.env` file and enter monitor path `--mount_path`
    ```bash
    sudo ./generate_env --mount_path <your_want_to_monitor_path>
    ```

* pull all docker images
    ```bash
    sudo ./pull_all_images
    ```

* set your HF_TOKEN in environment
    ```bash
    export HF_TOKEN=<your_hf_token>
    ```

* download require files, please set your HF_TOKEN before execute.
    ```bash
    sudo chown -R $(whoami):$(whoami) /home/$(whoami)/.cache/huggingface
    ./download_require_files
    ```

## Startup
* run server in background
    ```bash
    docker compose -f compose.yaml up -d
    ```

* stop server
    ```bash
    docker compose -f compose.yaml down
    ```

## Website
* open in browser http://127.0.0.1:3001/