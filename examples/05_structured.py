import json
import os
from typing import Literal

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

def build_model() -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        reasoning=False,
        temperature=0,
        num_ctx=8192,
    )

class WeatherReport(BaseModel):
    city: str = Field(description="城市名")
    condition: Literal["晴", "多云", "阴", "雨", "雪"] = Field(description="天气状况")
    temperature_c: int = Field(ge=-60, le=60, description="摄氏温度，整数")

TEST_INPUTS = [
    # Case 1: 干净输入 -> A 组有机会「侥幸成功」
    "北京今天晴，18度。",
    # Case 2: 非标天气词 + 带单位小数 -> A 组容易能 parse 但「字段/类型/枚举不对」
    "上海今天下着毛毛细雨，气温 24.5°C，湿度很大。",
    # Case 3: 中文负温 + 非标天气词 -> A 组容易把 condition 写成 "暴雪" 或温度带单位
    "哈尔滨现在零下五度，暴雪预警，风很大。",
    # Case 4: 格式与闲聊指令冲突 -> 诱导 A 组输出 ```json 围栏或提醒文字，触发「JSONDecodeError」
    "请用 Markdown 代码块整理深圳天气，顺便用一句话提醒我带伞：深圳大雨，26度。",
    # Case 5: 多时间干扰 + 要求对比 -> 诱导 A 组先输出解释文字或返回列表，触发「JSONDecodeError」或校验失败
    "昨天广州是阴天22度，今天转多云28度了，先帮我对比一下温差，再给出今天的天气。",
]

def run_naive(model: ChatOllama, text: str) -> tuple[str, str]:
    """A 组：天真做法（只交代字段 + 提示词要求只输出 JSON）"""
    prompt = (
        "从下面这段话中提取今天的天气信息，包含 city、condition、temperature_c 三个字段。\n"
        "只输出 JSON，不要任何解释。\n\n"
        f"输入：{text}"
    )
    response = model.invoke(prompt)
    raw = response.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        snippet = raw.replace("\n", "\\n")[:120]
        return "崩了(JSONDecodeError)", f"{e.msg} | 原始输出: {snippet}"

    try:
        report = WeatherReport.model_validate(data)
        return "侥幸成功", repr(report)
    except ValidationError as e:
        errs = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        return "崩了(字段/类型/业务约束不对)", f"{'; '.join(errs)} | 实际 JSON: {data}"

def run_structured(structured_model, text: str) -> tuple[str, str]:
    """B 组：结构化做法（with_structured_output）"""
    try:
        report: WeatherReport = structured_model.invoke(
            f"提取下面这段话中今天的天气信息：\n{text}"
        )
        return "成功", repr(report)
    except ValidationError as e:
        # 显式捕获业务/Schema 校验失败（例如模型把 24.5 吐成 245 被 ge/le 拦下）
        errs = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        return "校验拦截(ValidationError)", "; ".join(errs)
    except Exception as e:
        return "失败", f"{type(e).__name__}: {e}"

def main():
    model = build_model()
    structured_model = model.with_structured_output(WeatherReport)

    stats_a = {
        "崩了(JSONDecodeError)": 0,
        "崩了(字段/类型/业务约束不对)": 0,
        "侥幸成功": 0,
    }
    stats_b = {
        "成功": 0,
        "校验拦截(ValidationError)": 0,
        "失败": 0,
    }

    for i, text in enumerate(TEST_INPUTS, start=1):
        print(f"\n[Case {i}] 输入：{text}")
        print("-" * 70)

        status_a, detail_a = run_naive(model, text)
        stats_a[status_a] += 1
        print(f"  A 组（天真做法）  : {status_a}\n    -> {detail_a}")

        status_b, detail_b = run_structured(structured_model, text)
        stats_b[status_b] += 1
        print(f"  B 组（结构化做法）: {status_b}\n    -> {detail_b}")

    print("\n" + "=" * 70)
    print(f"汇总结果（共 {len(TEST_INPUTS)} 条）：")
    print(
        f"  A 组：JSONDecodeError {stats_a['崩了(JSONDecodeError)']} 条 | "
        f"字段/类型/业务约束不对 {stats_a['崩了(字段/类型/业务约束不对)']} 条 | "
        f"侥幸成功 {stats_a['侥幸成功']} 条"
    )
    print(
        f"  B 组：成功 {stats_b['成功']} 条 | "
        f"校验拦截 {stats_b['校验拦截(ValidationError)']} 条 | "
        f"其他失败 {stats_b['失败']} 条"
    )
    print("=" * 70)

if __name__ == "__main__":
    main()