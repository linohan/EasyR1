#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen3-8B  # replace it with your local file path

ray job submit --address="http://10.242.64.12:8265" \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
        config=examples/qwen3_8b_planner_config_with_thinking.yaml \
        worker.actor.model.model_path=${MODEL_PATH}
