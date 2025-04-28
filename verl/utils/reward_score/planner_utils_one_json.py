import json
import ast
from json_repair import repair_json
import copy


def extract_bracket_content(s):
    """提取文本中的json(通过{}匹配提取)"""
    stack = []
    start, end = None, None
    s = s.replace('”', '"') \
        .replace('“', '"') \
        .replace('‘', "'") \
        .replace('’', "'") \
        .replace('\\n\\n', ",") \
        .replace("'{", "{") \
        .replace("}'", "}") \
        .replace(':：', ':') \
        .replace('：', ':')
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if start is None:  # 记录最外侧的'{'位置
                start = i
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:  # 当堆栈为空时，记录'}'的位置
                    end = i
                    break  # 找到最外侧的'}'后退出循环
    if start is not None and end is not None:
        res = s[start: end + 1]
        return res
    else:
        return s


def json_parser_0(content):
    return json.loads(content)


def json_parser_1(content):
    return ast.literal_eval(content)


def json_parser_2(content):
    return repair_json(content, ensure_ascii=False)


def json_parser(s):
    """
    尝试多种方案解析json，如果解析失败则返回{}
    """
    content = s
    json_parsed = {}
    parser_list = [json_parser_0, json_parser_1, json_parser_2]
    success_func = ""
    json_parsed_success = True
    for i, parser in enumerate(parser_list):
        try:
            json_parsed = parser(content)
            success_func = f"json_parser_{i}"
            break
        except Exception as e:
            if i == 0:
                content = extract_bracket_content(content)
    if json_parsed == {} or not isinstance(json_parsed, dict):
        json_parsed_success = False

    if json_parsed_success:
        # logger.info(f'llm_res_parse_right by {success_func}: {s}')
        return json_parsed
    else:
        print(f'llm_res_parse_error: {s}')
        json_parsed = {}
        return json_parsed


def parse_answer(content):
    """Parse answer to reason and answer"""
    answer_parsed = {}
    try:
        answer_parsed = json_parser(content)
    except:
        answer_parsed = {}
    return answer_parsed

def remove_fields(data, fields_to_remove):
    """
    params:
        data: dict, 待删除字段的数据
        fields_to_remove: list, 要删除的字段列表
    递归地删除dict中所有的fields_to_remove字段。
    注意，会修改原data，所以请传入deepcopy
    """
    if isinstance(data, dict):
        for key in fields_to_remove:
            if key in data:
                del data[key]
        for key in data:
            remove_fields(data[key], fields_to_remove)
    elif isinstance(data, list):
        for item in data:
            remove_fields(item, fields_to_remove)

def remove_redundant_keys(obj):
    obj_c = copy.deepcopy(obj)
    obj_c = json_parser(obj_c) if isinstance(obj_c, str) else obj_c
    redundant_keys = ["Thought", "forReason"]
    remove_fields(obj_c, redundant_keys)
    return obj_c


def verify_ans(ans_parsed, gold_parsed):
    gold_parsed = remove_redundant_keys(gold_parsed)
    ans_parsed = remove_redundant_keys(ans_parsed)
    score = 0.0
    if gold_parsed.get("Action", "") == ans_parsed.get("Action", ""):
        score += 0.8
    if gold_parsed.get("ActionInput", "") == ans_parsed.get("ActionInput", ""):
        score += 0.2
    return score
