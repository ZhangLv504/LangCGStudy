import json
import os
import time
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。"""
    temps = {
        "北京": "18°C",
        "上海": "24°C",
        "西安": "15°C",
    }
    temp = temps.get(city, "20°C")
    return f"{city}：晴，{temp}，西北风 2 级"

def exp1_latency(model: ChatOllama, prompt: str) -> None:
    print("=" * 50)
    print("实验 1：测首字延迟 (TTFT) vs 总耗时")
    print("=" * 50)

    model.invoke("hi")

    # 记录当前时刻的高精度时间戳，通常作为某段代码 / 操作的“起始时间”。
    t0 = time.perf_counter()
    res = model.invoke(prompt)
    t_invoke = time.perf_counter() - t0
    print(f"[invoke] 首字延迟: {t_invoke:.3f}s | 总耗时: {t_invoke:.3f}s")
    print(f"[invoke] 内容预览: {res.content[:40]}...\n")

    t0 = time.perf_counter()
    # TTFT = Time To First Token，中文通常叫首字延迟或首 token 延迟。
    ttft = None
    print("[stream] 输出: ", end="", flush=True)
    for chunk in model.stream(prompt):
        if chunk.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            print(chunk.content, end="", flush=True)
    t_stream = time.perf_counter() - t0
    print(f"\n[stream] 首字延迟: {ttft:.3f}s | 总耗时: {t_stream:.3f}s\n")

def exp2_to_4_tool_stream(model: ChatOllama, prompt: str) -> None:
    bound = model.bind_tools([get_weather])

    print("=" * 50)
    print("实验 2：观察 bind_tools 后的 stream 分片")
    print("=" * 50)

    full = None
    first_args_str = None


    for i, chunk in enumerate(bound.stream(prompt)):
        print(
            f"chunk[{i:02d}] | content={chunk.content!r} "
            f"| tool_calls={chunk.tool_calls} "
            f"| tool_call_chunks={chunk.tool_call_chunks}"
        )

        # 记录第 1 个非空的 tool_call_chunks 里的 args 字符串
        if first_args_str is None and chunk.tool_call_chunks:
            args_piece = chunk.tool_call_chunks[0].get("args")
            if args_piece:
                first_args_str = args_piece

        full = chunk if full is None else full + chunk

    print("\n" + "=" * 50)
    print("实验 3：累加还原 (full = full + chunk)")
    print("=" * 50)
    print(f"full.tool_calls = {full.tool_calls}")
    if full.tool_calls:
        print(f"args 类型: {type(full.tool_calls[0]['args'])}")

    print("\n" + "=" * 50)
    print("实验 4：不累加直接 json.loads 第 1 个非空 args 片段")
    print("=" * 50)
    if first_args_str is None:
        print("未捕获到任何非空 tool_call_chunks args（模型可能未触发工具调用）。")
        return

    print(f"第 1 个非空 args 片段: {first_args_str!r}")
    try:
        parsed = json.loads(first_args_str)
        print(f"居然解析成功了: {parsed}")
        print("(注：Ollama 底层有时会把整个 tool_call 攒完在一个 chunk 里一次性吐出，所以没有碎掉)")
    except json.JSONDecodeError as e:
        print(f"如期报错 -> {type(e).__name__}: {e}")

def main() -> None:
    model = ChatOllama(
        model="qwen3.5:4b",
        reasoning=False,
        temperature=0,
        num_ctx=8192,
    )

    prompt = "北京天气怎么样"

    exp1_latency(model, prompt)
    exp2_to_4_tool_stream(model, prompt)


if __name__ == "__main__":
    main()