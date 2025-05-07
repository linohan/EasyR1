import re
from verl.utils.reward_score.planner_utils_one_json import *
# from planner_utils_one_json import *
import json
import numpy as np
from typing import Dict, List


def planner_format_reward(predict_str: str) -> float:
    try:
        # 尝试解析 JSON 字符串
        data = json.loads(predict_str)
        # 检查是否为字典类型且包含 Thought 字段
        if isinstance(data, dict) and 'Thought' in data:
            return 1.0
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


def planner_length_reward(predict_str: str, opt_length: int = 25, opt_length_penalty: float = 0.05) -> float:
    thinking_content = parse_answer(predict_str).get("Thought", "")
    length = len(thinking_content)
    return calc_length_penalty(length, opt_length, opt_length_penalty)


def planner_compute_score(predict_str: str, ground_truth: str) -> float:
    # return 0.9 * planner_acc_reward(predict_str, ground_truth) + 0.1 * planner_format_reward(predict_str)
    return 0.9 * planner_acc_reward(predict_str, ground_truth) + 0.1 * planner_format_reward(predict_str) + \
           1.0 * planner_length_reward(predict_str, opt_length=25, opt_length_penalty=0.05)


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        format_score = planner_format_reward(predict)
        accuracy_score = planner_acc_reward(predict, ground_truth)
        length_score = planner_length_reward(predict)
        overall_score = 0.9 * accuracy_score + 0.1 * format_score + 1.0 * length_score

        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

# def planner_compute_score_val(predict_str: str, ground_truth: str) -> float:
#     return planner_acc_reward(predict_str, ground_truth)


if __name__ == "__main__":
    _str = '''{
  "Thought": "客户要求转人工",
  "Action": "MANUAL_SERVICE",
  "ActionInput": {
    "forReason": "客户要求人工客服处理"
  }
}'''
    _ground_truth = "{\"Action\": \"MANUAL_SERVICE\", \"ActionInput\": {}}"
    print(f"length_reward: {planner_length_reward(_str, opt_length=25, opt_length_penalty=0.05)}")
    print(f"acc_reward: {planner_acc_reward(_str, _ground_truth)}")
    print(f"format_reward: {planner_format_reward(_str)}")
    print(f"planner_compute_score: {planner_compute_score(_str, _ground_truth)}")
    print(f"planner_compute_score_val: {planner_compute_score_val(_str, _ground_truth)}")
