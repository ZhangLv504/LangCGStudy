"""第 2 课配套实验：消息列表里少了信息，模型会怎样。

同一个问题"北京和上海哪个温度高"，喂两组不同的历史：

  A 完整历史：保留了 tool 结果与每轮 ai 回答，模型见过 18°C 和 22°C。
  B 残缺历史：只有三条用户提问，一个温度数字都没有。

B 组模拟的是"裁剪时把工具结果和回答都删掉、只留用户消息"这种写法。
观察重点不是它答得好不好，而是它**有没有承认自己不知道**。
"""

import os

# 本机 NO_PROXY 含 "[::1]"，httpx 0.28 解析代理列表会报 Invalid port；本进程只保留本机地址
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama


def build_model() -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=0,
        reasoning=False,
        num_ctx=8192,
    )


def call(city: str, call_id: str, result: str) -> list:
    """构造一轮完整的工具调用：申请 → 结果 → 最终回答。"""
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": city}, "id": call_id, "type": "tool_call"}
            ],
        ),
        ToolMessage(content=f"{city}：{result}", tool_call_id=call_id),
        AIMessage(content=f"{city}现在{result}"),
    ]


QUESTION = "这两个城市哪个温度高？"

FULL = (
    [HumanMessage("北京现在天气怎么样？")]
    + call("北京", "c1", "晴，18°C，西北风 2 级")
    + [HumanMessage("那上海呢？")]
    + call("上海", "c2", "多云，22°C，东风 1 级")
    + [HumanMessage(QUESTION)]
)

BARE = [
    HumanMessage("北京现在天气怎么样？"),
    HumanMessage("那上海呢？"),
    HumanMessage(QUESTION),
]


def main() -> None:
    model = build_model()

    for label, msgs in (("A 完整历史", FULL), ("B 残缺历史", BARE)):
        print(f"\n{'=' * 52}")
        print(f"{label}（{len(msgs)} 条消息）")
        for m in msgs:
            kind = type(m).__name__.replace("Message", "")
            brief = m.content if isinstance(m.content, str) and m.content else "<工具申请>"
            print(f"  {kind:<4} | {brief}")
        print("-" * 52)
        ai = model.invoke(msgs)
        print(f"回答：{ai.content}")

    print(f"\n{'=' * 52}")
    print("对比两组回答里的数字：A 组应当复述 18 与 22；")
    print("B 组从未见过这两个数字，看它是否编造，还是明确说不知道。")


if __name__ == "__main__":
    main()
