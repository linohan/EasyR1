#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen3-30B-A3B  # replace it with your local file path

ray job submit --address="http://10.242.64.07:8265" \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
        config=examples/qwen3_30b_a3b_planner_config.yaml \
        worker.actor.model.model_path=${MODEL_PATH}
