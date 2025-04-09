
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

def planner_length_reward(predict_str: str) -> float:
    thinking_content = extract_thinking_content(predict_str)
    length = len(thinking_content) 
    return length

def planner_compute_score(predict_str: str, ground_truth: str) -> float:
    return 0.9 * planner_acc_reward(predict_str, ground_truth) + 0.1 * planner_format_reward(predict_str)


if __name__ == "__main__":
    predict_str = "<think>1+1=2.</think>{'ans': 2}"
    print(planner_compute_score(predict_str, {'ans':2}))
    print(planner_compute_score(predict_str, "{'ans':2}"))
