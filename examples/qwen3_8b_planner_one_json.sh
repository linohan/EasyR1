#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# 获取活跃网卡的IP
get_ip() {
    # 获取默认路由对应的网卡
    local interface=$(ip route | awk '/default/ {print $5}')
    # 获取该网卡的IP
    local ip=$(ip -4 addr show $interface | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    echo $ip
}

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen3-8B  # replace it with your local file path

ray job submit --address="http://$(get_ip):8265" \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
        config=examples/qwen3_8b_planner_config.yaml \
        worker.actor.model.model_path=${MODEL_PATH}
