#!/usr/bin/env python3
"""Extract Level2AnalysisResult from raw LLM response."""

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.labeling import Level2AnalysisResult


def extract_json_after_think(text):
    # 正则解释：
    # <\/think>   : 匹配结束标签
    # \s* : 匹配 0 个或多个空白字符（换行、空格、制表符）
    # (\{.*\})    : 捕获组，从第一个 { 匹配到最后一个 }
    # re.DOTALL   : 关键！让 . 能够匹配换行符，否则只能匹配单行
    match = re.search(r"<\/think>\s*(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


if __name__ == "__main__":
    json_path = Path("./tmp.json")
    with open(json_path) as f:
        data = json.load(f)

    raw_response = data["raw_response"]
    extracted_json = extract_json_after_think(raw_response)
    if extracted_json:
        extracted_dict = json.loads(extracted_json)
        level2_result = Level2AnalysisResult.model_validate(extracted_dict)
        print(level2_result)
    else:
        print("No valid JSON found after <think> tag.")
