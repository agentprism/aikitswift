# Manifold

多种 LLM wire format 进，一条归一化事件流出。Swift 写的，给 Swift 用。

[English](README.md)

Manifold 是进气歧管 —— 多个入口汇成一个出口。这个库做的就是这件事：Anthropic 的
Messages API、OpenAI 的 Completions 和 Responses、Google 的 Generative AI，描述的是同
一件事，但格式互不兼容。Manifold 把它们映射到同一条事件流上，让应用只写一遍，而不是每
接一家写一遍。

```swift
let client = try ManifoldClient(providerId: "deepseek", configuration: .init(apiKey: key))

for try await part in try client.stream(CallOptions(model: "deepseek-v4-flash", prompt: [
    .system("回答简洁。"),
    .user("巴黎天气怎么样？"),
])) {
    switch part {
    case .textDelta(_, let delta, _):      render(delta)
    case .reasoningDelta(_, let delta, _): renderThinking(delta)
    case .toolCall(let call):              try await execute(call)
    case .finish(let usage, _, _):         report(usage)
    default:                               break
    }
}
```

把 `"deepseek"` 换成 `"anthropic"`、`"google"` 或另外 46 家中的任意一家 —— 这个循环里
不用改任何东西。

## 核心思路：按协议切，不按厂商切

最该避免的错误是每家厂商写一套实现。内置的 catalog 里有 **49 家 provider、413 个模型，
但只有 5 种 wire protocol** —— 因为大多数厂商说的是别人的协议：

| 协议 | provider 数 |
|---|---|
| OpenAI Chat Completions | 38 |
| Anthropic Messages | 7 |
| OpenAI Responses | 2 |
| OpenAI Codex | 1 |
| Google Generative AI | 1 |

Manifold 就沿着这条缝切开：

```
Sources/Manifold/
  Spec/        归一化词汇表 —— 所有 provider 都映射到同一个 enum
  Wire/        每个协议一份实现   （5 个，真正的工作量在这）
  Providers/   catalog            （49 个 JSON 配置，纯数据）
  Tokens/      上下文分摊
  Client/      把它们连起来的管道
```

一个 provider 就是数据：base URL、认证方式、模型列表，以及标明它说哪种协议的 `adapter`
字段。加一家是写配置，不是写实现 —— 而且**不需要新的 wire 测试**，因为它指向的协议早就
被覆盖了。

这个切法借鉴自 [pi-ai][pi]，是各自独立得出的同一结论。反面教材也值得一提：某个知名的
Swift LLM 客户端把 provider 层塞在一个 6000 行的文件里，支持的格式反而更少。

## 没有 API key 也能测

做 provider 集成的人都会遇到同一个问题：调不通的东西没法测，而没人手里有 49 家的 key。

Manifold 绕开了它。测试套件回放 **97 组真实录制的流 + 98 个完整响应体**（从 [AI SDK][aisdk]
按 MIT 协议 vendored 过来），断言归一化后的输出结构正确。录制的字节进，期望的事件出。
不联网、无凭证、不需要账号。

```
$ swift test
Test run with 164 tests in 16 suites passed
```

Fixture 按**协议**分组而非按厂商，所以 Chat Completions 这一个 mapper 同时被 7 家厂商的
真实流量验证：

| 集合 | 录制数 |
|---|---|
| `anthropic` | 27 |
| `openai-responses` | 29 |
| `google` | 20 |
| `xai` | 7 |
| `deepseek`、`groq`、`mistral` | 各 3 |
| `openai-completions`、`openai-compatible` | 各 2 |
| `cerebras` | 1 |

每组录制都会检查这些不变量：

- text / reasoning / tool-input 三者都构成配平的 `start → delta* → end` 三元组
- 拼装出的 tool 参数能解析成 JSON —— 分片单独都是非法 JSON，错一个字符就废，而且这种错
  只会在真实调用时才暴露
- `finish` 恰好出现一次、在最后，且 usage 内部自洽
- 未识别的 chunk 走 `.raw` 而不是消失，这样 provider 上新事件类型时本库不会丢数据

用 `Scripts/sync-fixtures.sh` 刷新。这里出现 diff，是 provider 改了 wire format 的最早
信号。

除 fixture 外，[Osaurus][osaurus] 这类 Anthropic 兼容的本地服务器可以提供走真实 socket
的端到端覆盖，同样不需要 key。

## 上下文分摊

一次请求的 token 花在哪了，窗口还剩多少：

```swift
let usage = ContextReporter().report(
    options,
    contextWindow: ProviderCatalog.model("claude-opus-4-8")?.1.contextWindow,
    extras: [("Skills", 5_500), ("Memory files", 284)]
)

for entry in usage.entries {
    print(entry.segment.label, ContextUsage.format(entry.tokens),
          String(format: "%.1f%%", usage.share(of: entry) * 100))
}
// Messages       463.4k  46.3%
// System prompt    6.1k   0.6%
// Skills           5.5k   0.6%
// Memory files      284   0.0%
```

**分摊与 provider 无关，只有计数是 provider 专属的**，所以 tokenizer 是注入进来的。默认
的估算器按字符所属文字系统区分 —— 拉丁文约 4 字符 1 token，而中日韩接近 1 字符 1 token；
统一按 `字符数 / 4` 算会把中文少算约四倍。

要精确总数，别花两遍钱：

```swift
usage.calibrated(toTotal: lastResponse.inputTokens.total ?? 0)
```

上一次响应里的 usage 是权威值，而且已经付过费了。用它作锚点缩放，就能得到**精确的总数
和比例正确的分段，且零额外网络调用**。只有在发送**之前**就要知道数字时，才需要走 provider
的 `count_tokens` 端点。

## 安装

```swift
.package(url: "https://github.com/zjywill/manifold.git", branch: "main")
```

Swift 6，macOS 14+、iOS 17+。无依赖。

## 现状

早期，API 会变。五种协议的流式响应和请求编码都能用了，catalog 覆盖 49 家。

| | |
|---|---|
| 归一化事件规范 | ✅ |
| SSE 分帧 | ✅ |
| Anthropic Messages | ✅ 流式 + 请求 |
| OpenAI Chat Completions | ✅ 流式 + 请求 |
| OpenAI Responses | ✅ 流式 + 请求 |
| Google Generative AI | ✅ 流式 + 请求 |
| Provider catalog | ✅ 49 家、413 模型 |
| 上下文分摊 | ✅ |
| 非流式响应 | ✅ 全协议 |
| Server tool 结果（代码执行 / MCP / 联网搜索） | ✅ 全协议 |
| OAuth（PKCE + 回环监听） | ✅ |
| 各家方言差异 | ✅ 见下 |

**方言。** "38 家说 OpenAI 协议"是个有用的简化，不是事实 —— 它们说的是三十八种方言。
有的要 `max_completion_tokens`，有的不认 `strict`，有的要求 tool result 带 `name`；
光是 reasoning 一项，catalog 里就有七种互不兼容的请求形状。`CompletionsDialect` 把这些
差异编码成**数据而非分支**，所以一个编码器仍然服务所有人 —— 这是 [pi-ai][pi] 的做法，
也是 JS 生态不得不每家发一个包的原因。

其中 `supportsUsageInStreaming` 最值得注意：向不支持的 provider 发
`stream_options.include_usage` 会 400；向支持的 provider 漏发，则所有 token 计数**静默
消失**。两个方向都会出事。

**关于覆盖度的诚实说明。** Anthropic 和 OpenAI Responses 的 server tool 路径是拿真实录
制的流量测的。Gemini 那部分（代码执行、grounding）不是 —— 语料里没有任何一条录制触及它
们，所以那几个测试编码的是文档形状而非捕获流量。这是更弱的保证，我在测试套件里标注了，
没有蒙混过去。

## 设计取舍

**归一化对齐 AI SDK。** 事件词汇表镜像 AI SDK 的 `LanguageModelV4StreamPart` —— 这是整
个生态里对这个问题打磨得最充分的一套归一化。沿用它的形状意味着它的 fixture 可以直接当
conformance 套件用，它的设计评审也白送。

**什么都不丢。** 没有归一化归宿的 provider 专属细节，按 provider 命名空间放进
`providerMetadata`；未识别的 chunk 原样走 `.raw`；原始 usage 载荷完整保留，以便对账。

**三家的 usage 口径互相矛盾，每一种都单独编码。** 同一个概念，三种含义：

| | 输入含缓存吗？ | 输出含推理吗？ |
|---|---|---|
| Anthropic | 否 —— 要把缓存两条腿加回去 | 不适用 |
| OpenAI | 是 —— 要减掉才得到未缓存量 | 是 —— 要减掉才得到正文 |
| Google | 是 | **否** —— 要加上 thoughts 才是总量 |

拿一家的算法套另一家，**不会报错，只会算错钱**。每种口径都有独立测试。

**会导致 400 的参数会被丢弃并上报。** 较新的 Anthropic 模型直接拒收 `temperature`。
catalog 里记录了哪些模型如此，编码器据此丢弃该参数并在 `streamStart` 上发一条
`Warning`，而不是让请求失败。

**Tool 参数保持字符串。** `ToolCall.input` 是流进来的 JSON **文本**，不是解析后的对象 ——
因为重新编码一个已解析的值无法还原原始字节。要用的时候再解析。

**`nil` 表示"没报"。** 所有 usage 字段都是可选的，"provider 没说"和"值为零"是两回事。

**编码时对 key 排序。** Prompt 缓存是字节级前缀匹配，请求体里 key 顺序不稳定会悄悄摧毁
所有缓存命中。这类问题不会报错，只会安静地烧钱。

## 参照

- **[vercel/ai][aisdk]** —— 归一化事件规范，以及本库据以测试的 fixture 语料。MIT。
- **[pi-ai][pi]** —— 协议/provider 的二分法，以及"provider 层里有多大比例其实是配置而非
  代码"这个提醒。
- **[Osaurus][osaurus]** —— 原生 Swift，也是个好用的本地被测端。

## 许可

MIT。

`Tests/ManifoldTests/Fixtures/` 下的录制 fixture vendored 自 [vercel/ai][aisdk]，沿用其
MIT 许可，目录下有 `PROVENANCE.md`。`Sources/Manifold/Catalog/` 下的 provider catalog 是
vendored 数据，用 `Scripts/sync-catalog.sh` 刷新。

[aisdk]: https://github.com/vercel/ai
[pi]: https://github.com/earendil-works/pi
[osaurus]: https://github.com/osaurus-ai/osaurus
