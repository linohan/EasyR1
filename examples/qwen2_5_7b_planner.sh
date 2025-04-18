set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=/data_nvme3n1/model/Qwen/Qwen2.5-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST follows immediately after the </think> tag, i.e.,
 <think> reasoning process here </think> answer here"""

python3 -m verl.trainer.main \
    config=examples/qwen2_5_7b_planner_config.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=8
