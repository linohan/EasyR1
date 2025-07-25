import re
from verl.utils.reward_score.planner_utils import *
# from planner_utils import *
import json
import numpy as np
from typing import Dict, List, Any


def parse_thinking_content(predict_str: str) -> dict:
    """
    解析 Thought 字段内容。
    """
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    match = re.search(pattern, predict_str)
    if match:
        return {
            "format_right": True,
            "thought": match.group(1).strip()
        }
    return {
        "format_right": False,
        "thought": ""
    }

def planner_format_reward(predict_str: str) -> float:
    try:
        thought_parsed = parse_thinking_content(predict_str)
        format_right = thought_parsed["format_right"]
        # 尝试解析 JSON 字符串
        result = predict_str.split("</think>")[-1].strip()
        data = json.loads(result)
        # 检查是否为字典类型且包含 Thought 内容
        if isinstance(data, dict) and format_right:
            return 1.0
        else:
            return 0.0
    except json.JSONDecodeError:
        # 如果解析失败，说明不是有效的 JSON 字符串
        return 0.0


def planner_acc_reward(predict_str: str, ground_truth: str) -> float:
    reward = 0.0
    if len(ground_truth) != 0:
        reward = verify_ans(predict_str, ground_truth)
    else:
        reward = 0.0
    return reward


def calc_length_penalty(cur_length: int, opt_length: int = 25, opt_length_penalty: float = 0.05):
    """
    cur_length: 当前输出长度
    opt_length: 最佳输出长度
    opt_length_penalty: 调节参数（默认0.05），控制最佳长度处的惩罚强度(0~1,对应的惩罚值是0~-1)
    计算惩罚值，惩罚曲线为：
    1. 当cur_length <= opt_length时，使用二次函数曲线，        
        k = opt_length_penalty / (opt_length ** 2)
        penalty = -k * cur_length ** 2
    2. 当cur_length > opt_length时，使用指数函数曲线，
        k = opt_length_penalty / (opt_length ** 2)
        exponent = (cur_length - opt_length) / opt_length  # 标准化指数项
        penalty = k * opt_length ** 2 * (1 - 2 * np.exp(exponent))
    3. 截断到[-1, 0]区间
    返回：归一化到[0, -1]区间的惩罚值
    """
    if cur_length <= 0:
        return 0.0

    if cur_length <= opt_length:
        # 最佳长度前的二次函数
        k = opt_length_penalty / (opt_length ** 2)
        penalty = -k * cur_length ** 2
    else:
        # 最佳长度后的指数函数
        k = opt_length_penalty / (opt_length ** 2)
        exponent = (cur_length - opt_length) / opt_length  # 标准化指数项
        penalty = k * opt_length ** 2 * (1 - 2 * np.exp(exponent))

    # 截断到[-1, 0]区间
    return max(penalty, -1.0)


def planner_length_reward_optimal_length(predict_str: str, opt_length: int = 25, opt_length_penalty: float = 0.05) -> float:
    """
    指定最优长度和最优长度处的惩罚，构造惩罚曲线。
    """
    thought_parsed = parse_thinking_content(predict_str)
    thought = thought_parsed["thought"]
    length = len(thought)
    return calc_length_penalty(length, opt_length, opt_length_penalty)

def planner_length_reward_kimi1_5(predict_str: str, ground_truth: str, min_length: int, max_length: int) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599


    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    if min_length == max_length:
        return 0.0
    thought_parsed = parse_thinking_content(predict_str)
    thought = thought_parsed["thought"]
    length = len(thought)
    result_correctness = verify_ans(predict_str, ground_truth) >= 0.8
    reward = 0.0
    lambda_val = 0.5 - (length - min_length)/(max_length - min_length)
    if result_correctness:
        reward = lambda_val
    else:
        reward = min(0, lambda_val)
    return reward


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    thought_lengths = []
    for reward_input in reward_inputs:
        predict = reward_input["response"]
        thought_parsed = parse_thinking_content(predict)
        thought = thought_parsed["thought"]
        length = len(thought)
        thought_lengths.append(length)
    min_length = min(thought_lengths)
    max_length = max(thought_lengths)
    for reward_input in reward_inputs:
        predict = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        format_score = planner_format_reward(predict)
        accuracy_score = planner_acc_reward(predict, ground_truth)
        length_score = planner_length_reward_kimi1_5(predict, ground_truth, min_length, max_length)
        overall_score = 0.8 * accuracy_score + 0.1 * format_score + 0.1 * length_score

        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "length": length_score,
                "accuracy": accuracy_score,
            }
        )

    return scores


if __name__ == "__main__":
    _reward_inputs = [{"response": '''<think>
客户要求转人工
</think>
{
  "Action": "MANUAL_SERVICE",
  "ActionInput": {
    "forReason": "客户要求人工客服处理"
  }
}''',
"ground_truth": "{\"Action\": \"MANUAL_SERVICE\", \"ActionInput\": {}}"
},
{"response": '''<think>
</think>
{
  "Action": "MANUAL_SERVICE",
  "ActionInput": {
    "forReason": "客户要求人工客服处理"
  }
}''',
"ground_truth": "{\"Action\": \"MANUAL_SERVICE\", \"ActionInput\": {}}"
},
{"response": '''</think>
{
  "Action": "MANUAL_SERVICE",
  "ActionInput": {
    "forReason": "客户要求人工客服处理"
  }
}''',
"ground_truth": "{\"Action\": \"MANUAL_SERVICE\", \"ActionInput\": {}}"
}]
    print(f"compute_score: {compute_score(_reward_inputs)}")
