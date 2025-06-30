#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen3-14B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data_nvme3n1/dataset/manual_corrected-20250311@train \
    data.val_files=/data_nvme3n1/dataset/manual_corrected-20250311@test \
    data.format_prompt=./examples/format_prompt/planner_format.jinja \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.rollout_batch_size=512 \
    data.mini_rollout_batch_size=256 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.weight_decay=0.1 \
    worker.actor.optim.lr_warmup_steps=10 \
    worker.actor.global_batch_size=128 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.rollout.n=16 \
    worker.rollout.max_num_batched_tokens=22528 \
    worker.rollout.val_override_config='{"n":16,"temperature":1.0,"top_p":0.7}' \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.reward.reward_function=./verl/utils/reward_score/planner.py:compute_score \
    worker.reward.reward_function_kwargs='{"max_response_length":2048,"overlong_buffer_length":4096,"overlong_penalty_factor":1.0}' \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=accuracy_normalized \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.total_epochs=10 \
    trainer.max_try_make_batch=10 \
    trainer.experiment_name=qwen3_14b_planner_dapo \
    trainer.n_gpus_per_node=8
