import os
import time

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安", "南京", "重庆"]

def build_mode() -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        reasoning=False,
        temperature=0,
        num_ctx=8192,
    )

def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。"""
    temps = {
        "北京": "18°C",
        "上海": "24°C",
        "西安": "15°C",
    }
    temp = temps.get(city, "20°C")
    return f"{city}：晴，{temp}，西北风 2 级"

def get_time(city: str) -> str:
    """查询指定城市的实时时间。"""
    return f"{city}：下午5点整"

def trim_history(msgs: list, max_len: int = 20) -> list:
    if len(msgs) <= max_len:
        return msgs

    trimmed = msgs[-max_len:]

    for i, msg in enumerate(trimmed):
        if isinstance(msg, HumanMessage):
            return trimmed[i:]

    for i, msg in enumerate(trimmed):
        if not isinstance(msg, ToolMessage):
            return trimmed[i:]

    return trimmed

def main() -> None:
    model = build_mode()
    bound = model.bind_tools([get_weather, get_time])
    msgs = []

    tool_map = {
        "get_weather": get_weather,
        "get_time": get_time
    }

    print("开始对话 (输入 'quit' 退出)")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        msgs.append(HumanMessage(content=user_input))
        msgs = trim_history(msgs, max_len=20)

        response = bound.invoke(msgs)
        msgs.append(response)

        meta = response.response_metadata or {}
        print(f"\n当前消息数: {len(msgs)}, Prompt Tokens: {meta.get('prompt_eval_count', 0)}")

        max_tool_steps = 5
        step = 0
        while response.tool_calls and max_tool_steps > step:
            step += 1
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"模型正在调用工具: {tool_name}, 参数: {tool_args}")

                if tool_name in tool_map:
                    try:
                        tool_res = tool_map[tool_name](**tool_args)
                    except Exception as e:
                        # 捕获异常并回传给模型，触发自我修正
                        tool_res = f"Error executing {tool_name}: {type(e).__name__}: {e}"
                        print(f"[Debug] 工具执行异常，已回传: {tool_res}")
                else:
                    tool_res = f"Error: Tool {tool_name} not found."

                msgs.append(ToolMessage(content=str(tool_res), tool_call_id=tool_call["id"]))

            response = bound.invoke(msgs)
            msgs.append(response)

            meta = response.response_metadata or {}
            print(f"\n(工具执行后)当前消息数: {len(msgs)}, Prompt Tokens: {meta.get('prompt_eval_count', 0)}")

        # 达到步数上限时的收尾处理
        if response.tool_calls:
            print(f"[Warn] 工具调用达到最大步数 ({max_tool_steps})，已强制中止。")
            # 重要：把最后那条未兑现的 tool_calls 消息弹出，避免破坏下一轮的对话结构
            msgs.pop()
        else:
            print(f"AI: {response.content}")

if __name__ == "__main__":
    main()