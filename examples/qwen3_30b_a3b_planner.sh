set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen3-30B-A3B  # replace it with your local file path

#SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
# The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST follows immediately after the </think> tag, i.e.,
# <think> reasoning process here </think> answer here"""

SYSTEM_PROMPT="你是在线旅行公司同程艺龙专职酒店业务的平台客服机器人，是人工客服的助手，你叫程小艺。"

ray job submit --address="http://10.242.64.12:8265" \
    --runtime-env=examples/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
        config=examples/qwen3_30b_a3b_planner_config.yaml \
        data.system_prompt="${SYSTEM_PROMPT}" \
        worker.actor.model.model_path=${MODEL_PATH} \
        trainer.n_gpus_per_node=8
