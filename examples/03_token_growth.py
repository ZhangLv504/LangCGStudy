"""第 2 课配套实验：历史到底吃多少 token。

学员估算 20 轮对话约 800 字。这里用真实请求的 prompt_eval_count 校准：
模型每次请求收到的不是"字数"，而是 token，且 tool_calls 这类结构化内容 token 效率很低。

同时对比"删掉 tool 消息"能省多少——以及省下的这点空间值不值。
"""

import os
import time

# 本机 NO_PROXY 含 "[::1]"，httpx 0.28 解析代理列表会报 Invalid port；本进程只保留本机地址
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安", "南京", "重庆"]


def build_model() -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=0,
        reasoning=False,
        num_ctx=8192,
        num_predict=1,  # 只让它吐 1 个 token，我们只要 prompt 的计量，不要完整回答
    )


def build_history(rounds: int, keep_tools: bool = True) -> list:
    """构造 rounds 轮对话。keep_tools=False 时删掉 tool 消息与 ai 最终回答。"""
    msgs: list = []
    for i in range(rounds):
        city = CITIES[i % len(CITIES)]
        msgs.append(HumanMessage(f"{city}现在天气怎么样？"))

        call_id = f"call_{i}"
        application = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": city}, "id": call_id, "type": "tool_call"}
            ],
        )
        result = f"{city}：晴，18°C，西北风 2 级，湿度 45%，能见度 10 公里"
        answer = AIMessage(content=f"{city}现在晴，18°C，西北风 2 级，体感舒适。")

        if keep_tools:
            msgs.extend([application, ToolMessage(content=result, tool_call_id=call_id), answer])
        else:
            msgs.append(answer)
    return msgs


def measure(model: ChatOllama, msgs: list) -> tuple:
    chars = sum(len(m.content) if isinstance(m.content, str) else 0 for m in msgs)
    t0 = time.time()
    ai = model.invoke(msgs)
    dt = time.time() - t0
    meta = ai.response_metadata or {}
    return meta.get("prompt_eval_count", 0), chars, dt


def main() -> None:
    model = build_model()

    print("完整历史（保留 tool 结果与 ai 回答）")
    print(f"{'轮数':<6}{'消息数':<8}{'字符数':<10}{'prompt tokens':<16}{'耗时':<8}")
    for rounds in (5, 10, 20):
        msgs = build_history(rounds)
        tokens, chars, dt = measure(model, msgs)
        print(f"{rounds:<6}{len(msgs):<8}{chars:<10}{tokens:<16}{dt:.1f}s")

    print("\n删掉 tool 消息与 ai 回答（只留 human + 工具申请）")
    print(f"{'轮数':<6}{'消息数':<8}{'字符数':<10}{'prompt tokens':<16}{'耗时':<8}")
    for rounds in (10, 20):
        msgs = build_history(rounds, keep_tools=False)
        tokens, chars, dt = measure(model, msgs)
        print(f"{rounds:<6}{len(msgs):<8}{chars:<10}{tokens:<16}{dt:.1f}s")

    print("\n注：num_ctx=8192，prompt tokens 逼近该值时历史就会被截断，")
    print("    表现为模型开始忘记开头几轮——而它不会告诉你它忘了。")


if __name__ == "__main__":
    main()
