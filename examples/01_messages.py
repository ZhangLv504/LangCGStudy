import os

# 本机 NO_PROXY 含 "[::1]"，httpx 0.28 解析代理列表会报 Invalid port；本进程只保留本机地址
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:qwen2.5-coder-7b:latest", temperature=0)

messages = [
    {"role": "system", "content": "你是一个简洁的中文助手，回答不超过两句话。"},
    {"role": "user", "content": "用一句话解释 LangGraph 是什么。"},
]

ai = model.invoke(messages)

print(type(ai).__name__)
print("content:", ai.content)
print("response_metadata:", ai.response_metadata)
