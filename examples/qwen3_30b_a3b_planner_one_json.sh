#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen3-30B-A3B  # replace it with your local file path

ray job submit --address="http://10.242.64.12:8265" \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
        config=examples/qwen3_30b_a3b_planner_one_json.yaml \
        data.train_files=/data_nvme3n1/dataset/manual_corrected-20250311@train \
        data.val_files=/data_nvme3n1/dataset/manual_corrected-20250311@test \
        worker.actor.model.model_path=${MODEL_PATH} \
        worker.actor.micro_batch_size_per_device_for_update=1 \
        worker.actor.micro_batch_size_per_device_for_experience=8 \
        worker.actor.fsdp.torch_dtype=bf16 \
        worker.actor.optim.strategy=adamw_bf16 \
        worker.rollout.tensor_parallel_size=8 \
        trainer.experiment_name=qwen3_30b_a3b_planner \
        trainer.n_gpus_per_node=8

