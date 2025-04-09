import json
import ast
from json_repair import repair_json
import copy


def extract_bracket_content(s):
    """提取文本中的json(通过{}匹配提取)"""
    stack = []
    start, end = None, None
    s = s.replace('”', '"')\
         .replace('“', '"')\
         .replace('‘', "'")\
         .replace('’', "'")\
         .replace('\\n\\n', ",")\
         .replace("'{", "{")\
         .replace("}'", "}")\
         .replace(':：', ':')\
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
        res = s[start : end + 1]
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
    json_parserd = {}
    parser_list = [json_parser_0, json_parser_1, json_parser_2]
    success_func = ""
    json_parsed_success = True
    for i, parser in enumerate(parser_list):
        try:
            json_parserd = parser(content)
            success_func = f"json_parser_{i}"
            break
        except Exception as e:
            if i == 0:
                content = extract_bracket_content(content)
    if json_parserd == {}:
        json_parsed_success = False

    if json_parsed_success:
        # logger.info(f'llm_res_parse_right by {success_func}: {s}')
        return json_parserd
    else:
        print(f'llm_res_parse_error: {s}')            
        return json_parserd

def extract_thinking_content(content):
    return content.split("</think>")[0]

def extract_answer_content(content):
    return content.split("</think>")[-1]

def parse_answer(content):
    """Parse answer to reason and answer"""
    reason_parsed = ""
    answer_parsed = {}
    try:
        reason_parsed = extract_thinking_content(content)
        answer_parsed = json_parser(extract_answer_content(content))
    except:
        answer_parsed = {}
    return reason_parsed, answer_parsed

def remove_redundant_keys(obj):
    obj_c = copy.deepcopy(obj)
    obj_c = json_parser(obj_c) if isinstance(obj_c, str) else obj_c
    redundant_keys = ["forReason"]
    for key in redundant_keys:
        if key in obj_c:
            del obj_c[key]
    return obj_c

def verify_ans(ans_parsed, gold_parsed):
    gold_parsed = remove_redundant_keys(gold_parsed)
    ans_parsed = remove_redundant_keys(ans_parsed)
    if gold_parsed == ans_parsed:
        return 1.0
    else:
        return 0.0

