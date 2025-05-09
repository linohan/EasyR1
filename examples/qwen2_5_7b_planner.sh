#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

ray job submit --address="http://10.242.64.07:8265" \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
        config=examples/qwen2_5_7b_planner_config.yaml \
        worker.actor.model.model_path=${MODEL_PATH}
