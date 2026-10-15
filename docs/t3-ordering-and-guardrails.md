# T3 实施顺序 + 执行护栏

> 状态:建议稿 · 2026-10-12 · **范围已裁断为「全做,不裁剪」**
> 依据:[ADR-0003](decisions/0003-usable-harness-and-t3-scope.md) 裁断(全做 / 从零 Python 重写)
> 证据:DSH 源码实测依赖图 + 本机 Python 环境实测

## 0. 一句话结论

**范围:全做。** DSH 有的、社区在填的,都做 —— 不设「暂不内建」清单。本文只回答**先后与做法**,不回答「该不该做」。

顺序**不由重要性决定,由依赖深度决定** —— 实测 DSH 的内部依赖边后,`skill` 仅 3 条、`workflow` 4 条、`goal` 7 条、**`subagent` 22 条**。按依赖拓扑排:

```text
skill → 工作树 → plan-mode → goal → subagent → (schedule)
                                    ↑ 最重,必须最后
```

电脑操作 / 浏览器 / Docker 不在这条链上,是**独立能力接缝**,可插在任意位置。

**护栏**只保留与「怎么做才不出问题」有关的三条(顺序 / 垂直切片 / 一次一个);「该不该做」类判据已作废 —— 理由见 §2。

---

## 1. T3 顺序的实测依据

### 1.1 DSH 内部依赖边(源码实测)

用 `package.json` 的 `dependencies` + `peerDependencies` 解析内部依赖(只计 `@deepseek-ai/dsh-*`):

| 包 | 内部依赖数 | 依赖了什么 |
|---|---:|---|
| `skill` | **3** | llm, scope, util-values |
| `workflow` | **4** | agent, brand, llm, session |
| `tool-todo` | 4 | agent, session, session-projection, tools |
| `jobs` | 4 | agent, brand, session, workspace |
| `compaction` | 4 | brand, commands, llm, session |
| `spill` | 3 | brand, llm, session |
| `sandbox-policy` | 5 | agent, sandbox, session, session-projection, system-prompt |
| `goal` | **7** | agent, brand, llm, scope, session, session-projection, typert-protocol |
| `plan-mode` | **9** | agent, brand, commands, llm, session, session-projection, system-prompt, tools, user-questions |
| **`subagent`** | **22** | agent, agent-preset-registry, attachment, brand, chunked-list, jobs, llm, permission-presets, sandbox, sandbox-policy, scope, session, session-persistence, session-projection, session-query, system-prompt, tools, typert-protocol, user-approval, util-time, util-values, workspace |
| `schedule` | **12** | …含 **subagent** |

> ⚠️ **修正**:本节初稿称 `plan` 零内部依赖,实测**错误**。DSH 的 plan 包名为 `@deepseek-ai/dsh-plan-mode`,有 **9 条**内部依赖(含 commands、user-questions、tools)。因此它**不在**顺序的最前端。

**读法**:`subagent` 依赖 22 个包,意味着**它之前的一切都必须先能用**——包括 jobs、sandbox、permission、attachment、workspace、session-query。这不是「重要所以最后」,是「不先有那些就跑不起来」。

### 1.2 建议顺序

| 序 | 能力 | 前置条件 | 为什么在这个位置 |
|---|---|---|---|
| **1** | **Skill 系统** | 内核(agent/loop/session/tools/llm/prompt) | 依赖最轻(3 条),且是**能力扩展的入口**——后续能力可以做成 skill 先验证 |
| **2** | **工作树** | 内核 + shell | DSH 无原生实现,成本低;是 subagent 并行隔离的前置 |
| **3** | **Plan mode** | 内核 + session-projection + commands + tools | 9 条依赖,需 projection 与命令面已就绪 |
| **4** | **持久 Goal / 长任务恢复** | 内核 + session-projection | 是「长任务」的载体,依赖面小于 subagent |
| **5** | **子 Agent / Agent Teams** | **全部上述 + jobs + sandbox + permission + workspace** | 22 条依赖,必须最后 |
| — | 电脑操作 / 浏览器 | 内核 + 工具流水线 | **独立接缝**,任意位置可插 |
| — | Docker 沙箱后端 | sandbox 策略层 | **独立接缝**,实现 `sandbox` 的第二个 provider |

> **注**:DSH 的 `schedule` 依赖 `subagent`(12 条),若做定时任务则排在 subagent 之后。

### 1.3 与 ADR-0002 排序的关系

[ADR-0002](decisions/0002-design-alignment-and-native-builtins.md) 按「JD 命中度」把 Eval 排第一。**该判据已作废**(项目与求职解耦),但 **Eval 的位置反而更靠前**了 —— 理由变了:

- 旧理由:JD 第③条职责。
- 新理由:要判断**改动之后是变好还是变坏**,Eval 是让这件事**可观测**的手段,不是可选项。(注:原「行为判据」已按 ADR-0003 §6.3 撤回,此处理由不依赖它。)

**因此 Eval 建议与 M1 同期或紧随其后**,不排进 T3 链尾。

---

## 2. 范围护栏

> ⚠️ **本节初稿作废(2026-10-12)。** 初稿提出三条否决权,其中「使用判据」与「允许删除」两条**建立在本人无法执行的判断上** —— 我不知道你日常需要用什么,因此无法判断某能力「会不会被日常使用」。范围已按 [ADR-0003](decisions/0003-usable-harness-and-t3-scope.md) §6.3 改为**全做,不做裁剪**。
>
> 下面只保留**与「该不该做」无关**的护栏 —— 即那些回答「怎么做才不出问题」的,而不是「要不要做」的。

### 2.1 保留:顺序护栏

**全做 ≠ 同时做。** 「全做」回答范围,不回答先后。

> 按依赖拓扑推进:一个能力的**前置能力必须先能用**,才开下一个。

依据是 §1.1 的实测:`subagent` 依赖 22 个包。这不是纪律问题,是**物理约束**——前置没就绪,它跑不起来。

### 2.2 保留:垂直切片护栏

> 每个能力推进到**端到端可用**再开下一个;只能当 stub 的不许进。

**为什么保留**:它与「该不该做」无关,只约束「做的方式」。防止的是一堆半成品子系统同时挂着——那种状态下,任何一处都跑不通,也无法验证。

### 2.3 保留:一次一个

> 任一时刻,只能有一个能力处于「正在做」状态。

**为什么保留**:与 §2.1 同源。DSH 的 41 万行是多年、多人、有专职维护者的产物;单人并行多个子系统,结果是每个都半成品。

### 2.4 已作废的两条(留档,不再执行)

| 初稿护栏 | 为什么作废 |
|---|---|
| 「使用判据:它是否让『我日常真的用它』更可能?」 | **我无法执行** —— 不知道你日常需要什么,这道闸门会随机砍掉你真正需要的能力 |
| 「允许删除:连续两周没被日常使用就删」 | 同上。删除的依据必须是**你的实际使用反馈**,不能是我的猜测 |

> **留档的理由**:这两条是「用不可执行的判据当护栏」的实例。记下来,避免以后重犯。

### 2.5 真正的风险,以及它现在的形态

ADR-0001 把「范围失控」列为项目最大风险。范围改为全做之后,这条风险**不再由「裁剪」来防**(那是错的),而由 **§2.1–2.3 三条执行护栏**来防:

| | 旧护栏(已撤) | 新护栏 |
|---|---|---|
| 防的是什么 | 做太多能力 | 做的方式散乱 |
| 机制 | 裁剪能力清单 | 顺序 + 垂直切片 + 一次一个 |
| 谁能判断 | 我(错) | 依赖拓扑(客观) |

**剩余的真实风险**(无护栏可防,只能记录):

1. **收敛**:快速产出大量代码后,把它收敛成**连贯的**系统,比写出来难。
2. **可靠性**:压缩失真、工具失败恢复、会话损坏 —— 这些决定「能用/不能用」,但**不是靠加护栏能解决的**,只能靠测试与实测。
3. **我可能再次用不可执行的判据当护栏** —— 见 §2.4。

---

## 3. 本机 Python 环境实测(2026-10-12)

从零 Python 重写,先摸清可用底座:

| 项 | 状态 |
|---|---|
| Python | **3.11.14**(Homebrew `/opt/homebrew/opt/python@3.11`) |
| `uv` | ✅ 已装(`~/.local/bin/uv`)—— 建议作依赖与虚拟环境管理 |
| `pipx` / `poetry` | ❌ 未装 |
| `docker` CLI | ❌ **未安装** → **Docker 沙箱后端当前无验证条件** |
| Chrome | ✅ `/Applications/Google Chrome.app` → 可走 CDP,不依赖 Playwright |
| `playwright` | ⚠️ **二进制在 Python 3.10 下,3.11 的 `import playwright` 失败** —— 版本错位,需重建 |

已装 Python 包(实测):`openai` ✅、`pydantic` ✅;其余 `anthropic` / `httpx` / `typer` / `rich` / `pytest` 等**均未安装**。

PyPI 对应版本(实测,可作选型基线):

| 用途 | 包 | 最新版 |
|---|---|---|
| 模型适配 | `anthropic` / `openai` | 1.13.0 / 3.29.0 |
| HTTP | `httpx` | 0.28.1 |
| 数据模型 | `pydantic` | 2.14.1 |
| 浏览器 | `playwright` | 1.63.0 |
| Docker | `docker` | 7.2.0 |
| MCP | `mcp` | 2.3.0 |
| CLI | `typer` / `rich` | 0.27.3 / 15.0.0 |
| 测试 | `pytest` | 9.1.1 |

**两个需要先解决的问题**:

1. **Docker 未装** → Docker 沙箱后端无法验证。要么装 Docker,要么该能力后置(范围是全做,所以是「后置」不是「不做」)。
2. **Playwright 版本错位** → 若走 Playwright 路线需先修;或改用 **CDP 直连已装 Chrome**(Chrome 已存在,且能复用用户的登录态——这正是 DSH 内置浏览器的痛点,见 MEMORY)。

---

## 4. 状态(2026-10-12)

范围已裁断为**全做,不裁剪**([ADR-0003](decisions/0003-usable-harness-and-t3-scope.md) §6.3)。因此本节不再问「该不该做」,只剩**执行方式**:

| # | 问题 | 状态 |
|---|---|---|
| 1 | **起点**:先补内核主干(M1–M4),还是从 T3 的 Skill 开始 | **建议先内核** —— 内核是 T3 全部五项的公共前置(§1.1),按 §2.1 顺序护栏,前置必须先能用 |
| 2 | T3 内部顺序 | 建议见 §1.2,待确认 |
| 3 | Docker | 未装。**后置**,不取消 |
| 4 | 浏览器路线 | 倾向 **CDP 直连 Chrome**(可复用登录态,绕开 DSH 那个 Turnstile 痛点);Playwright 需先修版本错位 |
| 5 | 项目位置(是否移出 `02-projects/`) | 未决,与执行无关,可后置 |

## 相关

- [ADR-0003:可自用 harness 与 T3 范围](decisions/0003-usable-harness-and-t3-scope.md)
- [ADR-0002:设计对齐与原生内置](decisions/0002-design-alignment-and-native-builtins.md)
- [ADR-0001:范围、非目标与 clean-room](decisions/0001-scope-and-non-goals.md)
- [项目 README](../README.md)
