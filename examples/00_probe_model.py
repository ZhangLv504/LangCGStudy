"""模型能力探针：判断一个本地模型能否胜任本项目的教学用途。

用法：
    python examples/00_probe_model.py                 # 默认 qwen3.5:4b
    python examples/00_probe_model.py qwen3.5:9b

为什么需要这个脚本：
    Ollama 的 capabilities 里写着 tools，不等于模型真能产出合规的 tool_calls。
    本项目已踩过一次：qwen2.5-coder-7b 标记支持 tools，实际把调用写成 content 里的 JSON，
    tool_calls 恒为 null。所以"选型"必须由 3 个最小用例验证，而不是看排行榜。

三个用例对应学习方案里的阶段 0 自检：
    1 单工具调用 —— tool_calls 非空、函数名与参数正确
    2 结果回填   —— 把执行结果作为 ToolMessage 追加后，模型能据此作答
    3 结构化输出 —— 能稳定填出 Pydantic 定义的字段
"""

import os
import sys
import time

# 本机 NO_PROXY 含 "[::1]"，httpx 0.28 解析代理列表会报 Invalid port；本进程只保留本机地址
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:4b"


def build_model() -> ChatOllama:
    """直接实例化 ChatOllama，而不是走 init_chat_model 的 provider 前缀。

    原因有二：模型名自带冒号（qwen3.5:4b），前缀解析容易歧义；
    这里需要显式暴露 num_ctx 与 reasoning 两个关键参数。
    """
    return ChatOllama(
        model=MODEL,
        temperature=0,
        reasoning=False,  # 关掉思考模式：教学要看清每一步，而不是等它想完
        num_ctx=8192,     # Ollama 默认约 4k，多轮/工具场景必须显式抬高
    )


def get_weather(city: str) -> str:
    """查询指定城市的当前天气。仅在用户明确询问某地天气时调用。

    Args:
        city: 城市名，使用中文，例如"北京"。
    """
    return f"{city}：晴，18°C，西北风 2 级"


class WeatherReport(BaseModel):
    """从一段文字中抽取出的天气信息。"""

    city: str = Field(description="城市名")
    condition: Literal["晴", "多云", "阴", "雨", "雪"] = Field(description="天气状况")
    temperature_c: int = Field(description="摄氏温度，整数")


def tok_s(ai) -> str:
    """从 response_metadata 粗略估算生成速度，用来判断是否溢出到了 CPU。"""
    meta = getattr(ai, "response_metadata", {}) or {}
    n = meta.get("eval_count")
    ns = meta.get("eval_duration")
    if not n or not ns:
        return "-"
    return f"{n / (ns / 1e9):.1f} tok/s"


def case1_tool_call(model: ChatOllama):
    print("\n[1] 单工具调用：tool_calls 是否真的非空")
    bound = model.bind_tools([get_weather])
    t0 = time.time()
    ai = bound.invoke([HumanMessage("北京现在天气怎么样？")])
    dt = time.time() - t0

    calls = ai.tool_calls or []
    print(f"    耗时 {dt:.1f}s | {tok_s(ai)}")
    print(f"    tool_calls = {calls}")
    print(f"    content    = {ai.content!r}")

    ok = bool(
        len(calls) == 1
        and calls[0]["name"] == "get_weather"
        and (calls[0]["args"] or {}).get("city")
    )
    if not ok and ai.content:
        # 历史上 qwen2.5-coder 就是这种失败模式：调用本身是对的，但写在正文里
        print('    ↑ 若 content 里出现 {"name": ...} 形状的 JSON，说明模型不产原生 tool_calls')
    print(f"    => {'PASS' if ok else 'FAIL'}")
    return ai, ok


def case2_full_loop(model: ChatOllama, first_ai, enabled: bool) -> bool:
    print("\n[2] 结果回填：ToolMessage 追加后能否据此作答")
    if not enabled:
        print("    => SKIP（用例 1 未通过，链路无法建立）")
        return False

    call = first_ai.tool_calls[0]
    # 代码侧执行：这一步不产生任何消息，只是拿到字符串
    result = get_weather(**call["args"])
    tool_msg = ToolMessage(content=result, tool_call_id=call["id"])

    t0 = time.time()
    ai = model.invoke([HumanMessage("北京现在天气怎么样？"), first_ai, tool_msg])
    dt = time.time() - t0
    text = str(ai.content)
    print(f"    耗时 {dt:.1f}s | {tok_s(ai)}")
    print(f"    工具返回 = {result}")
    print(f"    模型回答 = {text!r}")

    # 只要回答里带上真实工具结果的关键信息，就说明它确实读了 ToolMessage
    ok = ("18" in text) or ("晴" in text)
    print(f"    => {'PASS' if ok else 'FAIL'}")
    return ok


def case3_structured(model: ChatOllama) -> bool:
    print("\n[3] 结构化输出：能否稳定填出 Pydantic 字段")
    structured = model.with_structured_output(WeatherReport)
    prompt = "把这句话里的天气信息抽出来：今天上海多云，气温 22 摄氏度。"
    t0 = time.time()
    try:
        obj = structured.invoke([HumanMessage(prompt)])
    except Exception as e:  # 校验失败、JSON 解析失败都会走到这里
        print(f"    异常：{type(e).__name__}: {e}")
        print("    => FAIL")
        return False
    dt = time.time() - t0
    print(f"    耗时 {dt:.1f}s | 结果 = {obj!r}")

    ok = (
        isinstance(obj, WeatherReport)
        and "上海" in obj.city
        and obj.condition == "多云"
        and obj.temperature_c == 22
    )
    print(f"    => {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    print(f"模型：{MODEL}")
    print(f"参数：temperature=0, reasoning=False, num_ctx=8192")

    model = build_model()
    first_ai, c1 = case1_tool_call(model)
    c2 = case2_full_loop(model, first_ai, c1)
    c3 = case3_structured(model)

    print("\n" + "=" * 46)
    print(f"汇总：工具调用 {c1} | 结果回填 {c2} | 结构化 {c3}")
    verdict = "可作教学主力" if all([c1, c2, c3]) else "不可作教学主力"
    if all([c1, c2, c3]):
        pass
    elif c1 and not c3:
        verdict += "（工具链可用，结构化需换方式）"
    print(f"结论：{verdict}")
    print("=" * 46)


if __name__ == "__main__":
    main()
