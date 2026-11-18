# ChatOllama 核心函数详解

> 版本基线：`langchain-ollama==1.1.0`、`langchain-core`（随 langchain 1.4.10）、`ollama` SDK
> 源码位置：
> - `.venv/Lib/site-packages/langchain_ollama/chat_models.py`（1783 行，简称 `cm.py`）
> - `.venv/Lib/site-packages/langchain_core/language_models/chat_models.py`（简称 `core.py`）
> - `.venv/Lib/site-packages/ollama/_client.py`、`_types.py`（简称 `_client.py` / `_types.py`）
>
> 本文所有签名与默认值均从上述文件抄录，并标注行号，可自行核对。

---

## 0. 先定位：它到底是个什么东西

`ChatOllama` 继承 `BaseChatModel`（`cm.py:261`），是一个**适配器**：

```
LangChain 统一消息协议  ←→  ChatOllama  ←→  Ollama POST /api/chat
```

它不实现任何推理逻辑，只做四件事：

1. 把 `BaseMessage` 列表翻译成 Ollama 的 `{role, content, images, ...}` 字典；
2. 把构造参数翻译成 Ollama 的 `options` 字典；
3. 发 HTTP 请求；
4. 把响应字典包装成 `AIMessage` / `AIMessageChunk`。

**记住这个分层，每个参数你都能自己判断它属于哪一层**：构造参数 → `options`，调用参数 → 请求顶层字段，消息对象 → `messages`。

两种进入方式落到同一个对象：

```python
from langchain_ollama import ChatOllama
model = ChatOllama(model="qwen3.5:4b", temperature=0)

# 等价写法（examples/01_messages.py 用的就是这个）
from langchain.chat_models import init_chat_model
model = init_chat_model("ollama:qwen3.5:4b", temperature=0)
```

`init_chat_model` 的 `ollama:` 前缀只负责路由，其余参数原样透传。

---

## 1. 调用链：一次 `invoke` 走了几层

```
model.invoke(messages)                        # core.py:516，继承来的模板方法
  ├─ _convert_input(input)                    # str / tuple / dict 统一转消息列表
  ├─ generate_prompt(...)                     # core.py:1912，批处理外壳（单条也是一批）
  │    └─ _generate()                         # cm.py:1193   ← 真正干活
  │         └─ _chat_stream_with_aggregation()# cm.py:1126   ← 把流拼回一条
  │              └─ _iterate_over_stream()    # cm.py:1217   ← dict → ChatGenerationChunk
  │                   └─ _create_chat_stream()# cm.py:1107   ← 发 HTTP
  │                        ├─ _chat_params()  # cm.py:746    ← 装配所有参数
  │                        └─ _client.chat(**params)         # ollama SDK
  └─ .generations[0][0].message               # 取回 AIMessage
```

`stream` 走同一根链条，区别是**在 `_stream()`（`cm.py:1284`）处就交给你**，不做聚合：

```
model.stream(messages)                        # core.py:770
  └─ _should_stream(...)                      # 模型不支持流式则退化为 invoke
       └─ _stream()                           # cm.py:1284
            └─ _iterate_over_stream() → _create_chat_stream() → _client.chat()
```

`batch` 默认实现是**线程池并发跑 `invoke`**（`langchain_core/runnables/base.py:995`），不是批量接口。

### 1.1 反直觉事实：`invoke` 内部也是流式的

`_chat_params` 第 802 行：

```python
"stream": kwargs.pop("stream", True),
```

默认 `True`。也就是 `model.invoke(...)` 打给 Ollama 的是 `"stream": true`，**回来后在本机用 `chunk += chunk` 拼成完整回答**（`_chat_stream_with_aggregation`，1136-1139）。

两个可利用的后果：

- `invoke` 的延迟 = 首 token 时间 + 全部生成时间，和 `stream` **完全一样**；`invoke` 不是"非流式请求"，只是你看不到中间态。
- `invoke` 时若挂了回调，`on_llm_new_token` 依然**逐 token 触发**（1140-1145）。

> 此条由源码读出，非实测。验证：`invoke` 时传一个打印时间戳的 `BaseCallbackHandler`。

### 1.2 流式最后一块是空的

只有 `done=True` 的那一块会把整包统计写进 `generation_info`（1248-1253），而这一块的 `message.content` 通常是 `""`。

```python
for chunk in model.stream("你好"):
    print(repr(chunk.content))     # 最后一次是 ''
```

判结束看 `chunk.response_metadata.get("done")`，不要用 `content == ""`。

---

## 2. 构造函数：全部参数

```python
ChatOllama(
    model: str,                       # 唯一必填
    *,
    # —— 连接与客户端 ——
    base_url: str | None = None,
    client_kwargs: dict | None = {},
    sync_client_kwargs: dict | None = {},
    async_client_kwargs: dict | None = {},
    validate_model_on_init: bool = False,
    keep_alive: int | str | None = None,
    # —— 行为开关 ——
    reasoning: bool | str | None = None,
    format: Literal["", "json"] | JsonSchemaValue | None = None,
    stop: list[str] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    # —— 采样 ——
    temperature: float | None = None,     top_p: float | None = None,
    top_k: int | None = None,             seed: int | None = None,
    repeat_penalty: float | None = None,  repeat_last_n: int | None = None,
    tfs_z: float | None = None,
    mirostat: int | None = None,          mirostat_eta: float | None = None,
    mirostat_tau: float | None = None,
    # —— 长度与显存 ——
    num_ctx: int | None = None,           num_predict: int | None = None,
    num_gpu: int | None = None,           num_thread: int | None = None,
)
```

### 2.0 通用规则：`None` 表示"不下发"，不是"等于 0"

最容易误解的一点。`_chat_params` 用字典推导过滤（`cm.py:773-793`）：

```python
options_dict = {k: v for k, v in {...}.items() if v is not None}
```

`None` 的项整个从请求里消失 → **Ollama 用服务端默认值**。

所以 `ChatOllama(model="qwen3.5:4b")` 不写 `temperature`，实际是 0.8（注释 `cm.py:620`），**不是 0**。评测场景必须显式写 `temperature=0`，否则两次结果不同，会被误判为"模型不稳定"。

### 2.1 连接与客户端

| 参数 | 类型 / 默认 | 含义 | 注意 |
|---|---|---|---|
| `model` | `str`，必填 | 模型名，对应 `ollama list` 里的名字 | 带 tag，如 `qwen3.5:4b`；不写 tag 默认 `:latest` |
| `base_url` | `str \| None` | Ollama 服务地址 | 默认 `http://localhost:11434`（SDK host 默认值）。支持 `http://user:pass@host`，会被拆成 Authorization 头（946-947）；源码明确警告只适合本地/内网 |
| `client_kwargs` | `dict`，默认 `{}` | 透传给 httpx 客户端 | 放 `headers`、`timeout`、`trust_env` 等。**同步异步共用** |
| `sync_client_kwargs` | `dict`，默认 `{}` | 只覆盖同步客户端 | 与 `client_kwargs` 浅合并（951） |
| `async_client_kwargs` | `dict`，默认 `{}` | 只覆盖异步客户端 | 同上（955） |
| `validate_model_on_init` | `bool`，默认 `False` | 构造时调 `validate_model()` 检查模型是否存在（957-960） | **建议开**：把"模型名打错"从首次调用提前到构造时，代价是多一次 HTTP |
| `keep_alive` | `int \| str \| None` | 请求结束后模型在显存里驻留多久 | `"30m"`、`"1h"`、秒数；负数表示立即卸载。你实测首次加载 5.2s，设了能省掉重复加载 |

> `client_kwargs={"trust_env": False}` 可让 httpx 忽略系统代理，从而绕开脚本里的 `NO_PROXY` hack。这条路**我尚未实测**，只确认它是 httpx 的合法选项。

### 2.2 行为开关

| 参数 | 类型 / 默认 | 含义 | 取值影响 |
|---|---|---|---|
| `reasoning` | `bool \| str \| None`，默认 `None` | 思考模式开关，**映射到 Ollama 的 `think` 字段**（`cm.py:804`）——字段名不同，注意 | `True`：思考内容进 `additional_kwargs["reasoning_content"]`，`content` 干净；`False`：不思考；`None`：跟随模型默认（若模型默认思考，`<think>` 标签会**混在 `content` 里**，这是"回答前面一坨怪字"的最常见来源）；`"low"/"medium"/"high"`：思考强度，仅 gpt-oss 支持 |
| `format` | `None \| "" \| "json" \| dict` | 输出格式约束，即 Ollama 原生结构化输出 | `"json"` 只保证是合法 JSON，**不保证字段**；传 JSON Schema 字典由服务端约束解码。`with_structured_output` 内部就是往这里塞 |
| `stop` | `list[str] \| None` | 停止词，命中即截断 | 最终写进 `options["stop"]`（787）。**构造时设了、调用时再传 `stop=` 会直接 `ValueError`**（764-766） |
| `logprobs` | `bool \| None` | 是否返回每个 token 的对数概率 | 流式时每块带 `response_metadata["logprobs"]`，`invoke` 时聚合到最后一块 |
| `top_logprobs` | `int \| None` | 每个位置额外返回几个候选 token 及概率 | 必须 ≥1，否则 `ValueError`（649-653）；**Ollama 侧上限 20**（`_types.py:410`）。只设它不设 `logprobs` 会自动打开 `logprobs` 并告警（935-942）；显式 `logprobs=False` 同时设它会报错 |

### 2.3 采样参数（决定"说了什么"）

| 参数 | 默认 | Ollama 默认 | 含义 | 调高 / 调低 |
|---|---|---|---|---|
| `temperature` | `None` | 0.8 | 温度：对 logits 缩放后再采样 | 高＝更发散、更爱跑题；低＝更确定。**评测/工具调用用 0** |
| `top_p` | `None` | 0.9 | 核采样：只在累计概率达 p 的候选里采样 | 高＝候选更多（多样）；低＝更保守。通常与 `top_k` 二选一为主 |
| `top_k` | `None` | 40 | 只保留概率最高的 k 个候选 | 高（100）更发散；低（10）更保守 |
| `seed` | `None` | 随机 | 随机种子 | 固定它 + `temperature=0` 才谈复现；只固定 seed 不保证逐字节一致 |
| `repeat_penalty` | `None` | 1.1 | 重复惩罚：按已出现频率压低 token 概率 | 1.5 强惩罚（语句可能生硬）；0.9 宽容（可能复读）。中文长文复读时抬到 1.2~1.3 |
| `repeat_last_n` | `None` | 64 | 惩罚回溯窗口（看多少 token 内的重复） | `-1` = 整个 `num_ctx`；`0` = 关闭 |
| `tfs_z` | `None` | 1.0 | tail-free sampling：按概率曲线斜率截尾 | 2.0 削得更狠（更保守）；1.0 关闭。基本不用动 |
| `mirostat` | `None` | 0 | 另一套采样算法：0 关闭 / 1 Mirostat / 2 Mirostat 2.0 | 开了之后 `top_k`/`top_p`/`tfs_z` 失效 |
| `mirostat_tau` | `None` | 5.0 | Mirostat 的目标熵（连贯 vs 多样的平衡点） | 低＝更聚焦连贯；高＝更发散 |
| `mirostat_eta` | `None` | 0.1 | Mirostat 的学习率（调整响应速度） | 低＝调整慢而稳；高＝反应快但可能震荡 |

**关于"温度 0 是否等于确定性"**：理论上等价贪心解码，但 llama.cpp 在批处理/浮点上有细微差异，严格复现还要固定 `seed`、同版本、同机器。

### 2.4 长度与显存（决定"能不能跑、能跑多长"）

| 参数 | 默认 | 含义 | 关键事实 |
|---|---|---|---|
| `num_ctx` | `None` | 上下文窗口（token） | 注释标 Ollama 默认 2048（`cm.py:576`），**实际随版本变动，不要依赖**。这是"聊到一半模型忘了前面"的头号原因，显式设 8192/32768 |
| `num_predict` | `None` | 最多生成多少 token | 注释标默认 128；`-1` 无限，`-2` 填满上下文。**不显式设置时长回答会在句中断掉**，且 `done_reason` 不是 `stop`。你的 `03_token_growth.py` 用 `num_predict=1` 只取 prompt 计量，是巧用 |
| `num_gpu` | `None` | 卸载到 GPU 的层数 | 自动分配通常最优；8G 卡手动调容易把 KV cache 挤掉 |
| `num_thread` | `None` | CPU 推理线程数 | 建议物理核数；全驻显存时无意义 |

### 2.5 ChatOllama 没暴露、但能通过 `options=` 传的参数

Ollama `Options` 完整字段见 `_types.py:106-137`，`ChatOllama` 只暴露了 15 个（mirostat 三件套、num_ctx、num_gpu、num_thread、num_predict、repeat_last_n、repeat_penalty、temperature、seed、stop、tfs_z、top_k、top_p）。其余要自己塞：

```python
model.invoke("...", options={"presence_penalty": 0.4, "frequency_penalty": 0.3})
```

| 未暴露参数 | 含义 |
|---|---|
| `presence_penalty` | 惩罚"已出现过"的 token（管话题重复） |
| `frequency_penalty` | 按出现次数惩罚（管字面重复） |
| `typical_p` | 本地典型性采样，与 `top_p` 类似但更贴近"平均"分布 |
| `penalize_newline` | 是否惩罚换行符（避免频繁断行） |
| `num_keep` | 保留 prompt 开头多少 token 不参与压缩 |
| `num_batch` | 批处理大小，影响 prompt 处理速度与显存峰值 |
| `main_gpu` / `low_vram` / `use_mmap` / `use_mlock` / `numa` | 显存与内存装载策略，教学场景不用碰 |
| `vocab_only` / `logits_all` / `embedding_only` | 调试/嵌入式用途，正常对话不用 |

**代价**：`_chat_params` 用 `kwargs.pop("options", None)` 判断（770-772）——**只要你传了 `options`，构造时设的采样参数全部失效**，必须在这个字典里重新写全。

---

## 3. 公开方法逐个详解

### 3.1 `invoke` / `ainvoke`

```python
def invoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage
```
（`core.py:516`；`ainvoke` 同签名，`core.py:543`）

| 参数 | 类型 | 含义 |
|---|---|---|
| `input` | `LanguageModelInput` | 四种形态：`str`（自动包成 `HumanMessage`）、`list[tuple[str, str]]`（`("system", "…")`）、`list[BaseMessage]`、`PromptValue`。**只有第三种能精确控制"这轮发了什么"**，调历史裁剪必须用它 |
| `config` | `RunnableConfig \| None` | 运行配置：`callbacks`（观测）、`tags`、`metadata`、`run_name`、`run_id`、`recursion_limit`、`max_concurrency`、`configurable`。trace 里的分组靠 `metadata` |
| `stop` | `list[str] \| None` | 本次调用的停止词，**keyword-only**，不能按位置传。与构造函数的 `stop` 互斥 |
| `**kwargs` | — | 全部透传给 `_generate` → `_chat_params`，因此可临时覆盖 `model`、`options`、`format`、`reasoning`（落为 `think`）、`keep_alive`、`logprobs`、`tools` 等 |

返回 `AIMessage`。返回路径是 `generate_prompt(...).generations[0][0].message`（`core.py:527-537`）——**先拿 `ChatGeneration`，再取 `.message`**。

`_generate` 构造 `AIMessage` 时只取三样（1205-1212）：

```python
AIMessage(
    content=final_chunk.text,                                # 拼接后的文本
    usage_metadata=...usage_metadata,                        # 从流里合出来的
    tool_calls=...tool_calls,                                # 从流里合出来的
    additional_kwargs=final_chunk.message.additional_kwargs, # 含 reasoning_content
)
```

`response_metadata` 不在这里——它来自 `ChatGeneration.generation_info`（1203），由 BaseChatModel 包装时塞进 `AIMessage.response_metadata`。

### 3.2 `stream` / `astream`

```python
def stream(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[AIMessageChunk]
```
（`core.py:770`）

参数与 `invoke` 完全一致，返回从 `AIMessage` 变成迭代器。

**容易忽略的分支**（`core.py:777-783`）：进入前调 `_should_stream()`，若判定模型不支持流式，会**退化成 `yield self.invoke(...)`**，只产出一整块。"用了 stream 却只收到一块"不一定是 bug。

`AIMessageChunk` 支持 `+`：

```python
stream = model.stream(messages)
full = next(stream)
for chunk in stream:
    full += chunk                       # content 拼接，usage_metadata 累加
# full 等价于 invoke 的返回
```

官方推荐的"边打印边留完整结果"写法：既看到流，又拿到完整 `AIMessage` 读 `tool_calls`。

### 3.3 `batch` / `abatch`

```python
def batch(
    self,
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[Output]
```
（`langchain_core/runnables/base.py:995`）

| 参数 | 含义 |
|---|---|
| `inputs` | 一批输入，每个元素形状与 `invoke` 的 `input` 相同 |
| `config` | 单个配置（作用于全部），或**与 `inputs` 等长的列表**（逐个对应） |
| `return_exceptions` | `True` 时异常作为结果返回，而非整批中断。**批量评测必须开**，否则一条超时整批作废 |
| `**kwargs` | 透传给每条 `invoke` |

默认实现是线程池并发，但 Ollama 服务端单模型推理基本串行，`batch` 拿到的是"排队"而非真并行；8G 单卡收益有限。

### 3.4 `bind_tools`

```python
def bind_tools(
    self,
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
    *,
    tool_choice: dict | str | Literal["auto", "any"] | bool | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]
```
（`cm.py:1410`）

实现只有两行（1430-1431）：

```python
formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
return super().bind(tools=formatted_tools, **kwargs)
```

| 参数 | 类型 | 含义 | 实际行为 |
|---|---|---|---|
| `tools` | `Sequence` | 工具定义，四种形态：dict（裸 JSON Schema）、`type`（Pydantic 类）、`Callable`（带类型注解 + docstring）、`BaseTool` | 每个都过 `convert_to_openai_tool`，转成 `{"type": "function", "function": {name, description, parameters}}`。**docstring 与 `Field(description=...)` 就是模型看到的说明书** |
| `tool_choice` | 见签名 | 想强制/禁止调某个工具 | **完全被忽略**。源码注释原话（1426）：*"This parameter is currently ignored as it is not supported by Ollama."* |
| `**kwargs` | — | 直接进 `self.bind(**kwargs)` | 作为固定参数合并进后续每次调用 |

两个后果：

- 不能强制模型调工具，只能靠 **docstring 与工具命名** 引导；描述重叠时模型会乱选。
- 返回 `Runnable` 而非 `ChatOllama`。`model.bind_tools([f]).bind_tools([g])` **不是**"绑两个工具"，而是对已绑定的 Runnable 再包一层。正确写法 `model.bind_tools([f, g])`。

绑定的工具最终塞进请求的 `tools` 字段（818-819）：

```python
if tools := kwargs.get("tools"):
    params["tools"] = tools
```

回填时 assistant 消息里的 tool_call 形状（229-238）：

```python
{"type": "function", "id": tc["id"], "function": {"name": ..., "arguments": tc["args"]}}
```

注意 `arguments` 直接放 **dict**，不是 JSON 字符串——Ollama 接受对象，OpenAI 要求字符串。这是两个协议的真实差异。

### 3.5 `with_structured_output`

```python
def with_structured_output(
    self,
    schema: dict | type,
    *,
    method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, dict | BaseModel]
```
（`cm.py:1433`，实现体 1684 起）

| 参数 | 类型 / 默认 | 含义 |
|---|---|---|
| `schema` | `dict \| type` | 目标结构。四种形态：Pydantic 类、`TypedDict`、裸 JSON Schema、OpenAI function schema。**传 Pydantic 类返回实例并校验**；传其他返回 dict 且不校验（于是有"字段错却不报错"） |
| `method` | `"json_schema"` | 用哪条通道约束输出，见下表 |
| `include_raw` | `False` | `False`：只返回解析结果，解析失败**抛异常**；`True`：返回 `{'raw', 'parsed', 'parsing_error'}`，失败不抛，异常放进 `parsing_error`（1775-1781 的 fallback） |
| `**kwargs` | — | **几乎什么都传不进来**。实现体先 `kwargs.pop("strict", None)` 静默吞掉，其余非空即 `ValueError(f"Received unsupported arguments {kwargs}")`（1685-1688）。所以 `with_structured_output(Schema, temperature=0)` 直接报错，模型参数必须放构造函数 |

三种 `method` 的实现差异：

| `method` | 走哪个字段 | 内部实现 | 特点 |
|---|---|---|---|
| `"json_schema"`（默认） | `format=<JSON Schema>` | Pydantic → `model_json_schema()`；TypedDict → `convert_to_json_schema` 并**自动补 `required`**；裸 dict → 原样（1740-1760） | 服务端约束解码，最可靠。模型/版本不支持时会悄悄退化成普通 JSON 输出 |
| `"function_calling"` | `tools` | `convert_to_openai_tool(schema)` + `bind_tools([schema], tool_choice=tool_name)`（1696-1704）；`schema=None` 直接报错 | 复用工具协议，多一次"选工具"开销；由于 `tool_choice` 被忽略，它并不比 json_schema 更"强制" |
| `"json_mode"` | `format="json"` | 只设 `format="json"` | **不保证字段结构**，必须自己在 prompt 里描述 schema，否则拿到任何合法 JSON 都可能 |

解析器：Pydantic 用 `PydanticOutputParser`，其余用 `JsonOutputParser`（1745、1765），解析在本地做，与模型无关。

**评测时必须用 `include_raw=True`**，否则失败率是隐形的——你的 `00_probe_model.py` 用例 3 只判断了"拿到对象"，没看原始输出是否合规。

### 3.6 其他继承来的方法（知道存在即可）

| 方法 | 来源 | 说明 |
|---|---|---|
| `generate` / `agenerate` | `core.py:1635` | 直接接 `list[list[BaseMessage]]`，返回 `LLMResult`；`invoke` 就是它的单条包装 |
| `generate_prompt` | `core.py:1912` | `generate` 的 PromptValue 版 |
| `stream_events` / `astream_events` | `core.py:1330` | 事件级流式（LCEL/Agent 用），粒度比 `stream` 细，含 chain 起止事件 |
| `get_num_tokens` / `get_num_tokens_from_messages` | `BaseLanguageModel` | **基于启发式估算，不是真 tokenizer**。要准确数字用 `response_metadata["prompt_eval_count"]`（Ollama 真算的） |

---

## 4. 内部核心函数（真正决定行为的地方）

### 4.1 `_chat_params` —— 所有参数的总闸门

```python
def _chat_params(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]
```
（`cm.py:746`）

| 参数 | 含义 |
|---|---|
| `messages` | LangChain 消息列表，先经 `_convert_messages_to_ollama_messages` 翻译（762） |
| `stop` | 本次调用的停止词。与 `self.stop` 同时存在则 `ValueError`；只有 `self.stop` 时用它（764-768） |
| `**kwargs` | 调用时传的一切：`model`、`options`、`format`、`response_format`、`reasoning`、`logprobs`、`top_logprobs`、`keep_alive`、`stream`、`tools`，以及任何未识别的键——**未识别的会原样拼进请求体**（810 的 `**kwargs`） |

返回的 dict 就是 HTTP body 的形状（800-811）：

| 键 | 来源 | 说明 |
|---|---|---|
| `messages` | 翻译结果 | `[{role, content, images, tool_calls?, tool_call_id?, thinking?}]` |
| `stream` | `kwargs.pop("stream", True)` | 默认 `True`，见 §1.1 |
| `model` | `kwargs.pop("model", self.model)` | 可单次覆盖 |
| `think` | `kwargs.pop("reasoning", self.reasoning)` | **改名**：LangChain 叫 `reasoning`，Ollama 叫 `think` |
| `format` | `_resolve_format_param(...)` | `format` 优先于 `response_format`；同时给会告警（840-848） |
| `logprobs` / `top_logprobs` | 同名参数 | — |
| `options` | 采样参数字典 | 见 §2.0，`None` 项被剔除 |
| `keep_alive` | 同名参数 | — |
| `tools` | `kwargs.get("tools")` | 由 `bind_tools` 放入 |

三个读代码才知道的点：

1. **采样参数在 `options` 嵌套里，不在一层**。传 `options=` 会整体替换构造时的采样配置。
2. **`strict` 被主动删除**（815-816）。上层框架（结构化输出的 ProviderStrategy 等）会传它，Ollama 不认。
3. `response_format` 是 OpenAI 兼容入参，会被转成 Ollama 的 `format`（855-922）：`{"type":"json_object"}` → `"json"`；`{"type":"json_schema","json_schema":{"schema":{...}}}` → 拆包取内层 schema；无法识别则告警并忽略。

### 4.2 `_convert_messages_to_ollama_messages` —— 消息翻译

```python
def _convert_messages_to_ollama_messages(
    self, messages: list[BaseMessage]
) -> Sequence[Message]
```
（`cm.py:963`）

只有一个参数。第 974 行 `messages = list(messages)` 是**浅拷贝**，不会修改你传进来的列表——调试时别指望"它帮我规范化了"。

角色映射（997-1018）：

| LangChain 类型 | Ollama role | 附加字段 |
|---|---|---|
| `SystemMessage` | `system` | — |
| `HumanMessage` | `user` | — |
| `AIMessage` | `assistant` | `tool_calls`（有则带）、`thinking`（取 `additional_kwargs["reasoning_content"]`，1079-1082） |
| `ToolMessage` | `tool` | `tool_call_id` |
| `ChatMessage` | `message.role` 原样 | — |
| 其他类型 | — | 抛 `TypeError: Received unsupported message type for Ollama.` |

`content` 的处理（1020-1067）—— Ollama 与 OpenAI 的第二个大差异：

| content 形态 | 处理 |
|---|---|
| `str` | 直接作为 `content` |
| `list` 里的 `str` 元素 | `content += f"\n{part}"`（**前置换行**） |
| `list` 里的 `{"type": "text"}` | 同上，取 `["text"]` |
| `list` 里的 `{"type": "image_url"}` | 取 URL，按第一个逗号切分，逗号后的 base64 塞进 `images`；没有逗号就整体当 base64 |
| `list` 里的 `{"type": "image"}`（v1 格式） | 经 `_get_image_from_data_content_block` 取 base64 |
| `list` 里的 `{"type": "tool_use"}` | **`continue`，直接丢弃** |
| 其他 | 抛 `ValueError` |

结论：**在 Ollama 上 content blocks 会被拍平成字符串**，多块文本只剩 `\n` 边界，非 text/image 的块直接消失。不要指望 block 级保真。

v1 格式的 AIMessage（`response_metadata["output_version"] == "v1"`）会先经 `_convert_from_v1_to_ollama` 降级（977-990），这是为兼容新版 content blocks 协议。

### 4.3 `_iterate_over_stream` —— 响应解析

```python
def _iterate_over_stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]
```
（`cm.py:1217`）

参数与 `_chat_params` 一致（纯转发）。核心是字段映射：

| Ollama 响应字段 | 去向 |
|---|---|
| `message.content` | `AIMessageChunk.content` |
| `message.thinking` | `additional_kwargs["reasoning_content"]`，**仅当 `reasoning` 为真**（1263-1268） |
| `message.tool_calls` | `AIMessageChunk.tool_calls`（id 现造 uuid4） |
| `prompt_eval_count` / `eval_count` | `usage_metadata.input_tokens` / `output_tokens`（102-116） |
| `done=True` 那块的整包 | `generation_info`，补 `model_name`、`model_provider="ollama"`，再删掉 `message`（1248-1253） |
| `logprobs`（非终块） | `generation_info = {"logprobs": ...}` |

一段防御逻辑（1232-1246）：若某块 `done_reason == "load"` 且 `content` 为空，打 warning 并**跳过**。日志出现 `Ollama returned empty response with done_reason='load'` 时不是你写错了，是上游行为。

### 4.4 `_get_tool_calls_from_response` —— 为什么 tool_call 有 id

```python
def _get_tool_calls_from_response(response: Mapping[str, Any]) -> list[ToolCall]   # 208
```

```python
tool_calls.extend([
    tool_call(id=str(uuid4()), name=tc["function"]["name"],
              args=_parse_arguments_from_tool_call(tc) or {})
    for tc in raw_tool_calls
])
```

**那个 `id` 是 LangChain 现造的 uuid4，不是模型给的**——Ollama 的响应里根本没有 call id 字段。

这解释了你第 2 课反复漏掉的那个点：`AIMessage.tool_calls[i].id` 与 `ToolMessage.tool_call_id` 的对应关系**完全由客户端维护**，模型侧不存在这个配对。所以：

- `ToolMessage` 必须显式写 `tool_call_id`，否则翻译时该字段为空（1077-1078 只有非空才写）；
- 同一次调用跨进程/跨轮次**无法靠 id 做幂等**，每次响应都是新 uuid。

### 4.5 `_parse_arguments_from_tool_call` —— 参数兜底

```python
def _parse_arguments_from_tool_call(raw_tool_call: dict) -> dict | None   # 171
```

源码注释原话（176-179）：*"Band-aid fix for issue in Ollama with inconsistent tool call argument structure. Should be removed/changed if fixed upstream."* 指向 ollama/ollama#6155。

做三件事：

1. `arguments` 是 dict 时，逐个 value 尝试 `json.loads`，失败退回 `ast.literal_eval`（兜住模型写单引号的情况，139-159）；只有解析出 dict/list 才替换，否则保留原字符串。
2. `arguments` 是字符串时直接解析，失败抛 `OutputParserException`。
3. 过滤掉 `functionName` 这类把函数名回显进参数的多余字段（189-190）。

所以 `qwen` 系偶尔"参数看着对但类型是字符串"是上游结构不稳，框架在尽力兜。**工具函数入口做一次显式类型校验仍然有价值**，别指望框架全兜住。

### 4.6 `_get_usage_metadata_from_generation_info`

```python
def _get_usage_metadata_from_generation_info(generation_info) -> UsageMetadata | None   # 102
```

```python
input_tokens  = generation_info["prompt_eval_count"]
output_tokens = generation_info["eval_count"]
total_tokens  = input_tokens + output_tokens
```

两者任一缺失就返回 `None`（整块 `usage_metadata` 消失）。所以某些版本/模型下 `ai.usage_metadata` 是 `None` 不是 bug，是响应没给计数。

---

## 5. `AIMessage` 字段从哪来

| 字段 | 来源 | 用途 |
|---|---|---|
| `content` | 拼接完的 `message.content` | 最终回答 |
| `tool_calls` | `message.tool_calls`，每条补 uuid4 id | 决定下一步调哪个工具 |
| `additional_kwargs["reasoning_content"]` | `message.thinking`（`reasoning=True`） | 看模型怎么想；回传历史时被原样送回 |
| `response_metadata` | `done=True` 那块的 `generation_info` | `prompt_eval_count` / `eval_count` / `total_duration` / `load_duration` / `eval_duration` / `done_reason` / `model_name` / `model_provider` |
| `usage_metadata` | 由前两者算出 | token 计费、上下文占用估算 |
| `id` | LangChain run id | trace 定位 |

两个单位坑：

- `response_metadata` 的计时字段是**纳秒**，`total_duration / 1e9` 才是秒。
- 算 tok/s 用 `eval_count / (eval_duration / 1e9)`。用 `total_duration` 会把加载和 prompt 处理时间算进去，速度被算低——你的 76 tok/s 若是这么算的，实际还要更高。

---

## 6. 落到 `examples/04_cli_chat.py`

该文件目前是空骨架（`build_model` 里 `ChatOllama()` 无参，`main` 里 `model =` 右边空着）。按本文结论：

```python
def build_mode() -> ChatOllama:
    return ChatOllama(
        model="qwen3.5:4b",
        temperature=0,            # 不写 = 0.8，会显得"不稳定"
        reasoning=False,          # 关思考，content 里不会有 <think>
        num_ctx=8192,             # 不写 = 服务端默认（可能只有 2k~4k）
        num_predict=-1,           # 不写 = 128，长回答会被截断
        keep_alive="30m",         # 避免每轮重加载（首次加载 5.2s）
        validate_model_on_init=True,
    )
```

`num_predict=-1` 尤其重要：沿用默认 128 时，第 3 课做长回答对比会看到"说到一半断掉"，极易误判成模型能力问题。

`main()` 里三件事，全部来自上文：

1. 历史自己维护 `list[BaseMessage]`，每轮 `append` `HumanMessage` 与返回的 `AIMessage`；不要 append 那个 `Runnable` 或 `chunk`。
2. 用工具时 `AIMessage(tool_calls)` 与 `ToolMessage(tool_call_id=...)` **必须成对**；裁剪按"轮"整体删，不按条删。
3. 流式打印用 `chunk.content` + `end=""` + `flush=True`；循环后补换行；判结束看 `response_metadata.get("done")`。

---

## 7. 坑表（按踩到概率排序）

| # | 坑 | 识别信号 | 对策 |
|---|---|---|---|
| 1 | 以为不写 `temperature` 就是 0 | 同输入两次答案不同 | 显式 `temperature=0`，彻底复现再加 `seed` |
| 2 | 以为 `invoke` 不是流式 | 无异常，只是误解延迟构成 | 见 §1.1 |
| 3 | `num_predict` 默认 128 截断 | 回答句中断、`done_reason` 不是 `stop` | 显式 `-1` |
| 4 | 传 `options=` 导致采样参数全失效 | 设了 0 仍有随机性 | 不传，或在里面写全 |
| 5 | 流式循环多打印空串 | 末尾多一行空输出 | 用 `done` 判结束 |
| 6 | 想靠 `tool_choice` 强制工具 | 模型仍选别的 | Ollama 不支持（1426） |
| 7 | `ToolMessage` 缺 `tool_call_id` | 模型说"没拿到结果" | id 是客户端 uuid4 造的，自己维护 |
| 8 | 构造设 `stop`，调用又传 `stop=` | `ValueError: stop found in both...` | 只在一处设 |
| 9 | 依赖 content block 结构 | 多模态块丢字段 | Ollama 侧拍平（1020-1067） |
| 10 | `with_structured_output(Schema, temperature=0)` | `ValueError: Received unsupported arguments` | 参数放构造函数 |
| 11 | 结构化输出"看起来成功" | `parsed` 有值但字段是猜的 | `include_raw=True` 看 `parsing_error` |
| 12 | `top_logprobs` 设 >20 | 报错或被服务端忽略 | Ollama 上限 20 |
| 13 | `reasoning=None` 时 `<think>` 混进 content | 回答前一段怪字 | 显式 `reasoning=True/False` |
| 14 | 用 `get_num_tokens()` 估上下文 | 估出来和实际差很多 | 用 `prompt_eval_count` |

---

## 8. 与其他模型的协议差异（写跨模型代码时）

| 能力 | ChatOllama | OpenAI 风格 |
|---|---|---|
| 强制工具选择 | ✗（忽略 `tool_choice`） | ✓ |
| tool_call id | 客户端 uuid4 现造 | 服务端返回 |
| `arguments` 格式 | **dict** | **JSON 字符串** |
| content blocks | 拍平成字符串 | 保留结构 |
| 重复惩罚 | `repeat_penalty` 单参数 | `presence` / `frequency` 双参数 |
| 结构化输出 | 服务端 `format` 约束解码 | `response_format` + 严格 grammar |
| 并行工具调用 | 无显式开关，模型自己决定 | `parallel_tool_calls` |

---

## 附：一页速查

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

m = ChatOllama(model="qwen3.5:4b", temperature=0, reasoning=False,
               num_ctx=8192, num_predict=-1, keep_alive="30m",
               validate_model_on_init=True)

ai = m.invoke("你好")                      # AIMessage
ai.content | ai.tool_calls | ai.usage_metadata | ai.response_metadata

for c in m.stream("你好"):                 # AIMessageChunk
    print(c.content, end="", flush=True)

m.batch([["问题A"], ["问题B"]], return_exceptions=True)
m.bind_tools([get_weather])                # → Runnable，tool_choice 无效
m.with_structured_output(Schema, include_raw=True)      # 默认 json_schema
m.with_structured_output(Schema, method="json_mode")    # 需自己描述 schema
m.invoke("...", options={"presence_penalty": 0.4})      # 透传未暴露参数
```

---

*依据 `langchain-ollama==1.1.0` 及同环境 `langchain_core` / `ollama` 源码逐段核对。升级后先看 `bind_tools`、`with_structured_output`、`_chat_params` 三处是否变动。*
