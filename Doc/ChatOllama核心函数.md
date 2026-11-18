# ChatOllama 核心函数

> 版本基线：`langchain-ollama==1.1.0`（`requirements.txt` 已锁定）
> 源码位置：`.venv/Lib/site-packages/langchain_ollama/chat_models.py`（1783 行）
> 本文所有结论标注了 `chat_models.py:行号`，可自行打开文件核对，不必相信记忆。

---

## 0. 先定位：它到底是个什么东西

`ChatOllama` 继承 `BaseChatModel`（`chat_models.py:261`），是一个**适配器**：

```
LangChain 的统一消息协议  ←→  ChatOllama  ←→  Ollama /api/chat （HTTP + JSON）
```

它本身不实现任何推理逻辑，只做四件事：

1. 把 `BaseMessage` 列表翻译成 Ollama 的 `{role, content, images, ...}` 字典；
2. 把构造参数翻译成 Ollama 的 `options` 字典；
3. 发 HTTP 请求；
4. 把返回的字典包装成 `AIMessage` / `AIMessageChunk`。

**理解这一点，后面所有签名都不用背**：你在文档里看到的每个方法，要么在做第 1、2 步的翻译，要么在做第 4 步的包装。

进入这个类的两种方式，最终落到同一个对象：

```python
from langchain_ollama import ChatOllama
model = ChatOllama(model="qwen3.5:4b", temperature=0)

# 等价写法（examples/01_messages.py 用的就是这个）
from langchain.chat_models import init_chat_model
model = init_chat_model("ollama:qwen3.5:4b", temperature=0)
```

`init_chat_model` 的 `ollama:` 前缀只是路由到 `ChatOllama`，参数原样透传。

---

## 1. 核心调用链：一次 `invoke` 到底走了几层

这是全篇最重要的一张图。`invoke` 是 `BaseChatModel` 的模板方法，真正干活的全是下划线开头的私有函数：

```
model.invoke(messages)                        # BaseChatModel，继承来的，不在本文件
  └─ _generate()                              chat_models.py:1193
       └─ _chat_stream_with_aggregation()     chat_models.py:1126   ← 把流拼回一条消息
            └─ _iterate_over_stream()         chat_models.py:1217   ← dict → ChatGenerationChunk
                 └─ _create_chat_stream()     chat_models.py:1107   ← 真正发 HTTP
                      └─ _chat_params()       chat_models.py:746    ← 装配所有参数
                      └─ _client.chat(**params)                    ← ollama 官方 SDK
```

而 `stream` 走的是同一根链条，只是**在 `_stream()`（1284）处就交给调用方**，不做聚合：

```
model.stream(messages)
  └─ _stream()                                chat_models.py:1284
       └─ _iterate_over_stream()              chat_models.py:1217
            └─ _create_chat_stream()          chat_models.py:1107
                 └─ _chat_params() / _client.chat()
```

### 1.1 一个反直觉的事实：`invoke` 内部也是流式的

打开 `_chat_params`，第 802 行：

```python
"stream": kwargs.pop("stream", True),
```

默认值是 `True`。也就是说 `model.invoke(...)` 打给 Ollama 的请求是 `"stream": true`，**回来后在本机用 `chunk += chunk` 一块块拼成完整回答**（`_chat_stream_with_aggregation`，1136-1139）。

这带来两个可以直接利用的后果：

- `invoke` 的延迟 = 首个 token 时间 + 全部生成时间，和 `stream` **完全一样**；`invoke` 并不比 `stream` 慢，只是你拿不到中间状态。
- `invoke` 时如果挂了回调，`on_llm_new_token` 依然会**逐 token 触发**（1140-1145 行调用了 `run_manager.on_llm_new_token`）。所以想在不改代码的前提下观察"模型吐字节奏"，挂回调即可，不必改成 `stream`。

> 这一条是从源码读出来的，不是实测。要验证：`invoke` 时传一个打印时间戳的 `BaseCallbackHandler`，看输出是否是一串递增时间。

### 1.2 流式的最后一块是空的

`_iterate_over_stream` 里，只有 `stream_resp.get("done") is True` 的那一块才会把整包统计写进 `generation_info`（1248-1253），而这一块的 `message.content` 通常是 `""`。

所以下面这段代码一定会多打印一个空串 + 一串元数据：

```python
for chunk in model.stream("你好"):
    print(repr(chunk.content))   # 最后一次是 ''
```

判断"流结束"要看 `chunk.response_metadata.get("done")`，不要用 `content == ""` 当停止条件。

---

## 2. 构造函数参数：分四组记，不要一条条背

**通用规则（最容易踩的坑）：所有采样类参数默认是 `None`，语义是"不下发"，而不是"等于 0"。**

证据在 `_chat_params` 的字典推导（773-793）：

```python
options_dict = {
    k: v for k, v in {...}.items() if v is not None
}
```

`v is None` 的项会被整个丢掉，请求里根本不出现这个字段，于是 **Ollama 服务端用它自己的默认值**。

结论：`ChatOllama(model="qwen3.5:4b")` 不写 `temperature`，拿到的是 Ollama 默认的 0.8（源码注释 620 行写明），**不是 0**。教学和评测场景必须显式写 `temperature=0`，否则同一问题两次结果不同，你会误判成"模型不稳定"。

### 2.1 连接与身份

| 参数 | 类型 / 默认 | 作用 | 教学建议 |
|---|---|---|---|
| `model` | `str`，必填 | 模型名，如 `qwen3.5:4b` | — |
| `base_url` | `str \| None` | Ollama 服务地址 | 默认 `http://localhost:11434`；远程/容器时必设 |
| `client_kwargs` | `dict`，默认 `{}` | 透传给 httpx 客户端 | 可放 `headers`；`{"trust_env": False}` 可绕开系统代理（替代脚本里的 `NO_PROXY` hack，待你实测确认） |
| `sync_client_kwargs` / `async_client_kwargs` | `dict` | 同步/异步各自覆盖 | 同步异步超时不同才需要 |
| `validate_model_on_init` | `bool`，默认 `False` | 初始化时检查模型是否已下载（0.3.4+） | **建议开**，把"模型名写错"从第一次调用时提前到构造时 |
| `keep_alive` | `int \| str \| None` | 模型在显存里驻留多久 | `"30m"` 可避免每轮重加载（你实测首次加载 5.2s） |

### 2.2 采样（影响输出内容）

| 参数 | 默认 | Ollama 侧默认 | 说明 |
|---|---|---|---|
| `temperature` | `None` | 0.8 | 0 = 确定；评测必须显式设 0 |
| `top_p` | `None` | 0.9 | 与 top_k 配合 |
| `top_k` | `None` | 40 | 越小越保守 |
| `seed` | `None` | 随机 | 固定种子 + temp=0 才能完全复现 |
| `repeat_penalty` | `None` | 1.1 | 中文长文出现复读时可抬到 1.2~1.3 |
| `repeat_last_n` | `None` | 64 | 回溯窗口；`-1` 表示 `num_ctx` |
| `tfs_z` | `None` | 1.0 | 基本不用动 |
| `mirostat` / `mirostat_eta` / `mirostat_tau` | `None` | 0 / 0.1 / 5.0 | 另一套采样算法，初学跳过 |

### 2.3 长度与显存（影响能不能跑起来）

| 参数 | 默认 | 说明 | 你的场景 |
|---|---|---|---|
| `num_ctx` | `None` | 上下文窗口。源码注释标 Ollama 默认 2048（576 行），实际随版本变动，**不要依赖默认值** | 你固定 `8192`，agent 阶段要抬到 32k |
| `num_predict` | `None` | 最大生成 token 数。注释标默认 128，`-1` 无限 | `03_token_growth.py` 用 `num_predict=1` 只取 prompt 计量，是个巧用 |
| `num_gpu` | `None` | 卸载到 GPU 的层数 | 8G 卡别乱调，交给自动分配 |
| `num_thread` | `None` | CPU 线程数 | 全驻显存时无意义 |

### 2.4 行为开关

| 参数 | 默认 | 说明 |
|---|---|---|
| `reasoning` | `None` | **映射到 Ollama 的 `think` 字段**（804 行）。`True` 时思考内容进 `additional_kwargs["reasoning_content"]`，不进 `content`；`False` 关闭思考；`None` 跟随模型默认（若模型默认思考，`<think>` 标签会**混在 content 里**）；也接受 `"low"/"medium"/"high"`（仅 gpt-oss） |
| `format` | `None` | `"json"` 或 JSON Schema 字典，走 Ollama 原生结构化输出 |
| `stop` | `None` | 停止词列表。**注意**：构造时设了 `stop`，调用时又传 `stop=`，会直接 `ValueError`（764-766） |
| `logprobs` / `top_logprobs` | `None` | 取 token 概率。只设 `top_logprobs` 会自动打开 `logprobs` 并告警（927-942） |

`reasoning=False` 是你已经在用的设置。它省掉的不只是 token：`qwen3.5:4b` 在思考模式下会先吐几百字"让我想想"，那部分对教学演示是噪音，且会挤占 `num_ctx`。

---

## 3. 六个真正的核心方法

### 3.1 `invoke` / `ainvoke` — 一次性拿完整结果

```python
def invoke(self, input: LanguageModelInput, config=None, **kwargs) -> AIMessage
```

继承自 `BaseChatModel`，本文件不重写。`input` 接受三种形式：

```python
model.invoke("一句话")                       # str → 自动包成 HumanMessage
model.invoke([("system", "..."), ("human", "...")])   # 元组列表
model.invoke([SystemMessage("..."), HumanMessage("...")])  # 消息对象列表
```

第三种是唯一能精确控制"每轮到底发了什么"的写法，调历史裁剪时必须用它。

实际落点是 `_generate`（1193），它把聚合后的 chunk 拆成 `AIMessage`，保留了 `usage_metadata`、`tool_calls`、`additional_kwargs` 三项（1205-1212）——**这三项是"流式聚合"的副产品，不是重新算的**。

### 3.2 `stream` / `astream` — 逐块拿

```python
for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
```

`chunk` 是 `AIMessageChunk`，支持 `+` 累加：

```python
stream = model.stream(messages)
full = next(stream)
for chunk in stream:
    full += chunk
# full 现在等价于 invoke 的返回（content 拼接完成，usage_metadata 也合上了）
```

这是官方推荐的"边打印边留完整结果"写法：既能看到流，又能拿到完整 `AIMessage` 去读 `tool_calls`。

### 3.3 `batch` / `abatch` — 并发多请求

```python
model.batch([[("human", "问题A")], [("human", "问题B")]])
```

Ollama 服务端默认单模型串行推理，`batch` 拿到的是"排队加速"而非真并行。8G 显存单卡上收益有限，别指望它提速。

### 3.4 `bind_tools` — 工具绑定的真实实现

```python
def bind_tools(
    self,
    tools: Sequence[dict | type | Callable | BaseTool],
    *,
    tool_choice: dict | str | Literal["auto", "any"] | bool | None = None,
    **kwargs,
) -> Runnable[LanguageModelInput, AIMessage]
```

实现只有两行（1430-1431）：

```python
formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
return super().bind(tools=formatted_tools, **kwargs)
```

也就是：**把你的函数/Pydantic 类转成 OpenAI 风格的 tool schema，挂进 `Runnable` 的 `kwargs`**，随后在 `_chat_params` 里被塞进请求的 `tools` 字段（818-819）。

两个必须知道的事实：

- **`tool_choice` 被完全忽略**。源码注释第 1426 行原话：*"This parameter is currently ignored as it is not supported by Ollama."* 你不能强制模型调某个工具，只能靠 docstring 引导。
- `bind_tools` 返回的是 `Runnable`，不是 `ChatOllama`。所以 `model.bind_tools([f]).bind_tools([g])` 不是"绑定两个工具"，正确写法是一次传列表：`model.bind_tools([f, g])`。

`ToolMessage` 回填时，Ollama 请求体里的 assistant 消息长这样（`_lc_tool_call_to_openai_tool_call`，229-238）：

```python
{"type": "function", "id": tc["id"], "function": {"name": ..., "arguments": tc["args"]}}
```

### 3.5 `with_structured_output` — 结构化输出

```python
def with_structured_output(
    self,
    schema: dict | type,
    *,
    method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
    include_raw: bool = False,
    **kwargs,
) -> Runnable[LanguageModelInput, dict | BaseModel]
```

三种方法的区别（这是面试级考点）：

| `method` | 走什么通道 | 特点 |
|---|---|---|
| `"json_schema"`（**1.x 默认**，0.3.0 起） | Ollama 的 `format=<JSON Schema>` | 服务端约束解码，最可靠；模型不支持时会退化 |
| `"function_calling"` | `tools` + 强制单工具 | 复用工具协议，多一次"选工具"开销 |
| `"json_mode"` | 只设 `format="json"` | **不保证字段结构**，必须在 prompt 里自己描述 schema |

`include_raw` 是调试开关：

```python
r = model.with_structured_output(Schema, include_raw=True).invoke("...")
# r = {'raw': AIMessage, 'parsed': Schema | None, 'parsing_error': Exception | None}
```

解析失败时它不会抛异常，而是把异常放进 `parsing_error` 并把 `parsed` 置 `None`（1480-1483）。**做评测必须用 `include_raw=True`**，否则失败率是隐形的——你的 `00_probe_model.py` 用例 3 只判断了"拿到对象"，没判断"原始输出是否合规"。

顺带：`format` 和 `response_format` 同时给会告警，且以 `format` 为准（840-848）；`response_format` 支持 OpenAI 风格的 `{"type": "json_schema", "json_schema": {"schema": {...}}}` 并自动拆包（855-922），这是给上层框架兼容用的，手写代码直接用 `format` 更清楚。

### 3.6 内部函数：`_chat_params` —— 所有参数的总闸门

```python
def _chat_params(self, messages, stop=None, **kwargs) -> dict[str, Any]   # 746
```

出参就是最终 HTTP body 的形状（800-811）：

```python
{
    "messages": [...],      # 来自 _convert_messages_to_ollama_messages
    "stream": True,         # 默认 True，见 1.1
    "model": "qwen3.5:4b",
    "think": None,          # ← reasoning 参数落在这里
    "format": None,         # ← with_structured_output / format 落在这里
    "logprobs": None,
    "top_logprobs": None,
    "options": {...},       # ← 所有采样参数落在这里
    "keep_alive": None,
    **kwargs,               # 其余全部透传
}
```

三个读代码才知道的点：

1. **采样参数进 `options` 嵌套字典，不在一层**。想在 `invoke(..., options={...})` 里覆盖时必须传完整的 `options`，一旦传了，构造时设的那些采样参数**全部失效**（770-773 用 `kwargs.pop("options", None)` 判断）。
2. `"think"` 不是 `"reasoning"`。Ollama 的字段名是 `think`，LangChain 侧叫 `reasoning`，804 行做了改名。
3. `strict` 会被主动删掉（815-816），因为上层框架（如结构化输出的 ProviderStrategy）会传它，而 Ollama 不认。

### 3.7 内部函数：`_convert_messages_to_ollama_messages` —— 消息翻译表

```python
def _convert_messages_to_ollama_messages(self, messages) -> Sequence[Message]   # 963
```

角色映射（997-1018）：

| LangChain 类型 | Ollama role | 附带字段 |
|---|---|---|
| `SystemMessage` | `system` | — |
| `HumanMessage` | `user` | — |
| `AIMessage` | `assistant` | `tool_calls`（有则带）、`thinking`（`additional_kwargs["reasoning_content"]` 有则带） |
| `ToolMessage` | `tool` | `tool_call_id` |
| `ChatMessage` | `message.role` 原样 | — |
| 其他 | **`TypeError`** | — |

这段代码解释了两个你已经在用的现象：

- **`ToolMessage` 必须带 `tool_call_id`**，否则回填的 `tool_call_id` 是空，语义就断了。而那个 id 是 **LangChain 自己用 `uuid4()` 造的**（`_get_tool_calls_from_response`，第 219 行 `id=str(uuid4())`）——因为 Ollama 的响应里根本不含 call id。所以"配对"完全是客户端责任，这正好解释第 2 课那个反复出错的点：模型侧没有 id，`AIMessage.tool_calls[i].id` 与 `ToolMessage.tool_call_id` 的对应关系全靠你自己维护。
- **`content` 为 list（content blocks）时会被拍平成字符串**（1020-1067）：多个 text 块用 `\n` 拼接，`image_url` 块被抽成 `images` 列表，其他类型（如 `tool_use`）直接 `continue` 丢弃。所以 **在 Ollama 上不要指望 block 级保真**，v1 的 content blocks 到这里已经退化。

另外注意 974 行 `messages = list(messages)` 是浅拷贝，它不会改你传进来的列表——这一点对调试友好，但反过来说，你也别指望"传给它的列表被就地规范化"。

### 3.8 内部函数：`_iterate_over_stream` —— 响应解析

```python
def _iterate_over_stream(self, messages, stop=None, **kwargs) -> Iterator[ChatGenerationChunk]   # 1217
```

它把 Ollama 的每一条 dict 变成 `ChatGenerationChunk`，规则如下：

| Ollama 响应字段 | 去向 |
|---|---|
| `message.content` | `AIMessageChunk.content` |
| `message.thinking` | `additional_kwargs["reasoning_content"]`（仅当 `reasoning` 为真，1263-1268） |
| `message.tool_calls` | `AIMessageChunk.tool_calls`（id 现造 uuid4） |
| `prompt_eval_count` / `eval_count` | `usage_metadata` 的 `input_tokens` / `output_tokens`（102-116） |
| `done=True` 那一块的整包 | `generation_info`，并补 `model_name` / `model_provider="ollama"`，再删掉 `message`（1248-1253） |

还有一段防御逻辑值得知道（1232-1246）：若某条响应 `done_reason == "load"` 且内容为空，会打一条 warning 并**跳过**。这是为"模型加载阶段返回空包"的上游行为打的补丁。你在日志里看到 `Ollama returned empty response with done_reason='load'` 时，不是你代码写错了。

### 3.9 工具调用参数解析：一个真实的坑

`_parse_arguments_from_tool_call`（171-205）里有这段注释：

> *"Band-aid fix for issue in Ollama with inconsistent tool call argument structure. Should be removed/changed if fixed upstream."*

具体做的两件事：

1. 若 `arguments` 是 dict，逐个 value 尝试 `json.loads`，失败再 `ast.literal_eval`（139-159，兜住模型用单引号的情况），只有解析出 dict/list 才替换，否则保留原字符串。
2. 若 `arguments` 是字符串，直接 `_parse_json_string(..., skip=False)`，解析失败抛 `OutputParserException`。
3. 顺手过滤掉 `functionName` 这种把函数名回显进参数的多余字段（189-190）。

这就是为什么 `qwen` 系偶尔"参数看着对但类型是字符串"——上游结构不稳，LangChain 尽力兜。知道这点后，工具函数入口做一次显式类型校验是有价值的，别指望框架全兜住。

---

## 4. `AIMessage` 字段从哪来（对照表）

调 `model.invoke()` 之后，别只会看 `.content`。完整映射：

| `AIMessage` 字段 | 来源 | 典型用途 |
|---|---|---|
| `content` | 拼接完的 `message.content` | 最终回答 |
| `tool_calls` | `message.tool_calls`，每条补 uuid4 id | 决定下一步调哪个工具 |
| `additional_kwargs["reasoning_content"]` | `message.thinking`（`reasoning=True` 时） | 看模型怎么想的；回传历史时会被原样送回 |
| `response_metadata` | `done=True` 那一块的全部字段 | `prompt_eval_count` / `eval_count` / `total_duration` / `load_duration` / `done_reason` |
| `usage_metadata` | 由 `prompt_eval_count` + `eval_count` 算出 | token 计费、上下文占用估算 |
| `id` | LangChain 的 run id | trace 定位 |

`response_metadata` 里的计时字段单位是**纳秒**，`total_duration / 1e9` 才是秒。算 tok/s 要用 `eval_count / (eval_duration / 1e9)`，不要用 `total_duration`——后者含加载和 prompt 处理时间，会把速度算低。

---

## 5. 落到 `examples/04_cli_chat.py`

这个文件目前是空的骨架（`build_model` 只有 `ChatOllama()`，`main` 里 `model =` 没有右侧）。按本文的结论，填充时按这个顺序：

```python
def build_mode() -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=0,            # 不写就是 0.8，会显得"不稳定"
        reasoning=False,          # 关思考，content 里不会有 <think>
        num_ctx=8192,             # 不写就是服务端默认，可能只有 2k~4k
        num_predict=-1,           # 不写默认 128，长回答会被截断
        keep_alive="30m",         # 避免每轮重加载
        validate_model_on_init=True,
    )
```

`num_predict=-1` 这一条特别重要：如果沿用默认的 128，第 3 课做长回答对比时你会看到回答"说到一半断掉"，很容易误判成模型能力问题。

`main()` 里要注意的三件事，全部来自本文上面的结论：

1. 历史列表自己维护 `list[BaseMessage]`，每轮 `append` `HumanMessage` 和返回的 `AIMessage`——不要 append 那个 `Runnable` 或 `chunk`。
2. 若用了工具：`AIMessage(tool_calls)` 与 `ToolMessage(tool_call_id=...)` **必须成对**。裁剪消息时按"轮"整体删，不要按条删（这条你已在工作记忆里踩过）。
3. 打印流式时，用 `stream=True` 的 `chunk.content` 逐块 `print(..., end="", flush=True)`，循环结束后再补一个换行；判断结束看 `response_metadata.get("done")`，别看 content 是否为空。

---

## 6. 十个坑（按"多久会踩到"排序）

| # | 坑 | 识别信号 | 对策 |
|---|---|---|---|
| 1 | 以为不写 `temperature` 就是 0 | 同问题同输入，两次答案不同 | 显式 `temperature=0`；彻底复现再加 `seed` |
| 2 | 以为 `invoke` 不是流式、所以更稳 | 无异常，只是误解了延迟构成 | 见 §1.1：`invoke` 就是本地聚合的流 |
| 3 | `num_predict` 默认 128 截断回答 | 回答在句中断掉、`done_reason` 不是 `stop` | 显式 `num_predict=-1` |
| 4 | 传了 `options={}` 导致采样参数全失效 | 设了 temperature=0 但输出仍有随机性 | 要么不传 `options`，要么在里面写全 |
| 5 | 流式循环多打印一个空串 | 末尾多一行空输出 | 用 `done` 判断结束，不用 `content` 判空 |
| 6 | `tool_choice` 不生效 | 想强制某工具，模型仍选别的 | Ollama 不支持（1426 行）；改 docstring 引导 |
| 7 | `ToolMessage` 缺 `tool_call_id` 或 id 对不上 | 模型答"没拿到结果"或忽略工具输出 | id 是客户端 uuid4 造的，位置自己维护 |
| 8 | 构造时设 `stop`，调用时又传 `stop=` | `ValueError: stop found in both...` | 只在一处设 |
| 9 | 依赖 `content` 的 block 结构 | 多模态/富文本块丢字段 | Ollama 侧会被拍平成字符串（1020-1067） |
| 10 | 结构化输出"看起来成功了" | `parsed` 有值但字段是猜的 | 用 `include_raw=True` 看 `parsing_error` |

---

## 7. 与其他模型的差异（顺手记）

同一套 `BaseChatModel` 接口下，`ChatOllama` 有几处**不具备**的能力，写跨模型代码时要注意：

- 没有 `tool_choice` 的强制约束；
- 没有 call id（每次现造 uuid4，所以跨进程/跨轮次无法靠 id 做幂等）；
- 没有 OpenAI 那样的 penalty/frequency 细分，只有 `repeat_penalty`；
- `parallel_tool_calls` 之类字段不存在，一次可以返回多个 tool_call 但要靠模型自己决定；
- 结构化输出靠 `format` 的服务端约束，不是 logits 层的严格 grammar（不同 Ollama 版本严格度不同）。

---

## 附：一页速查

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

m = ChatOllama(model="qwen3.5:4b", temperature=0, reasoning=False,
               num_ctx=8192, num_predict=-1, keep_alive="30m",
               validate_model_on_init=True)

ai = m.invoke("你好")                     # AIMessage
ai.content, ai.tool_calls, ai.usage_metadata, ai.response_metadata

for c in m.stream("你好"):                # AIMessageChunk
    print(c.content, end="", flush=True)

m.bind_tools([get_weather])               # → Runnable，tool_choice 无效
m.with_structured_output(Schema, include_raw=True)   # 默认 method="json_schema"
m.with_structured_output(Schema, method="json_mode") # 需自己在 prompt 里描述 schema
```

---

*本文依据 `langchain-ollama==1.1.0` 源码逐段核对写成；若升级版本，先看 `chat_models.py` 里 `bind_tools`、`with_structured_output`、`_chat_params` 三处是否变动。*
