import re
from verl.utils.reward_score.planner_utils import *


def planner_format_reward(predict_str: str) -> float:
    pattern = re.compile(r'<think>.*?</think>[^<]+', re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def planner_acc_reward(predict_str: str, ground_truth: str) -> float:
    reward = 0.0
    if len(ground_truth) != 0:
        answer_parsed = parse_answer(predict_str)[1]
        reward = verify_ans(answer_parsed, ground_truth)
    else:
        reward = 0.0
    return reward


def planner_length_reward(predict_str: str, max_length: int = 30, power: int = 8) -> float:
    thinking_content = extract_thinking_content(predict_str)
    length = len(thinking_content)
    if length <= 0:
        return 0.0
    if length >= max_length:
        return -1.0
    # 计算归一化的长度比例
    length_ratio = length / max_length
    # 幂指数，可调整以改变斜率变化速度，大于 1 时长度越长斜率越大
    reward = -length_ratio ** power
    return reward


def planner_compute_score(predict_str: str, ground_truth: str) -> float:
    return 0.9 * planner_acc_reward(predict_str, ground_truth) + 0.1 * planner_format_reward(predict_str)
    # return 0.7 * planner_acc_reward(predict_str, ground_truth) + 0.1 * planner_format_reward(predict_str) + 0.2 * planner_length_reward(predict_str, max_length=30, power=8)


if __name__ == "__main__":
    _str = '''{
  "Thought": "客户表达了想要人工客服处理的需求，根据客户需求转接人工服务。",
  "Action": "MANUAL_SERVICE",
  "ActionInput": {
    "forReason": "客户要求人工客服处理"
  }
}'''
    _ground_truth = '''{
  "Thought": "客户表达了想要人工客服处理的需求，根据客户需求转接人工服务。",
  "Action": "MANUAL_SERVICE",
  "ActionInput": {
    "forReason": "客户要求人工客服处理"
  }
}'''
    print(planner_compute_score(_str, _ground_truth))
