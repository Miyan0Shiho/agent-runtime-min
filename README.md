# agent-runtime-min

> ## 🚀 实施入口
> **[REQUIREMENTS.md](REQUIREMENTS.md) —— 需求书,自包含。** 换干净对话做实现时,先读这份,再按其中的索引读细节。
>
> **状态:L4 · 内核主干 + T3 全部 + MCP + hooks + 插件 + 浏览器 + 电脑操作 + Docker 沙箱完成 · 四份契约的「未解决」清单已全部关闭 · 已纳入 git** · 立项 2026-10-07 · 设计完成 2026-10-12 · 实现 2026-10-12
> **代码:28,389 行实现 / 23,102 行测试 / 1,240 个测试通过(另 12 条需真模型 key,无 key 自动 skip) / 81 个模块** · `make bench` / `make claims` / `make deadcode` / `make smoke` 全绿
> 决策链:[ADR-0001](docs/decisions/0001-scope-and-non-goals.md)(范围/clean-room)→ [ADR-0002](docs/decisions/0002-design-alignment-and-native-builtins.md)(设计对齐)→ [ADR-0003](docs/decisions/0003-usable-harness-and-t3-scope.md)(可自用 + 全做 + 顺序护栏)
> 设计产物:主链路 1 份 + 机制契约 4 份(**78 条不变量**)+ 决策记录 3 份 + 参考 2 份
> 实施产物:**内核主干 M1–M4 全部实现** + [契约修正记录](docs/contract-amendments.md)(86 条真实缺陷) + [方法论教训](docs/lessons.md) + [前置验证](spikes/)

## 现在能跑什么

```bash
make bench     # 一条命令:跑全部测试 + 输出四项验收指标
```

### 四个检查命令,回答四个不同的问题

| 命令 | 问的问题 | 什么时候会红 |
|---|---|---|
| `make test` | 每个零件对不对 | 某个单元的行为变了 |
| `make bench` | 验收指标达没达到 | M1–M4 的阈值破了 |
| `make smoke` | **装起来之后能不能用** | 零件都在,但没接上 |
| `make deadcode` | 有没有**代码没人调用** | 新增了未调用的定义 |
| `make claims` | 设计文档的断言**是否成立** | 实测推翻了文档的假设 |

**`make smoke` 是这五个里最重要的一个,也是最晚加的。** 它的由来值得说清楚:

这个项目里**所有靠「真跑一次」才发现的缺陷,形状都一样 —— 零件是对的,接线是错的**:

| 缺陷 | 零件 | 错在 |
|---|---|---|
| `developer` 角色被 provider 拒 | 内部词汇表正确 | 线上映射 |
| MCP 每次调用都失败 | 传输层、客户端都正确 | **CLI 把两个事件循环接错** |
| 子会话 schema 没告诉子 agent | 校验器正确 | 提示词构造 |
| plan mode 进得去出不来 | 模式实现正确 | CLI 没传审批通道 |
| hooks 拦了,日志说不出谁拦的 | hook 执行正确 | 身份没进日志 |

**这些没有一个是单元测试失败** —— 因为单元测试**自己搭零件**。它们共同的问题是:

> **按 CLI 的方式把 runtime 装起来之后,调用某个东西真的能用吗?**

`make smoke` 问的就是这个,对着真实的 `build_runtime` 跑。当前 **12 项离线检查**(其中两项逐个检查了 **26 个工具 / 9 个子系统**的连通性),`make check` 已包含它。

```bash
make smoke                          # 离线:不需要 provider
ARM_SMOKE_MODEL_KEY=... make smoke  # 加上真实 provider 往返 + 真实 MCP 调用
```

> **每一项都验证过「撤掉修复会失败」。** 一个不能失败的检查比没有检查更糟 —— 它占着「这里有覆盖」的位置。
>
> 而且它在开发中就抓住过自己:**smoke 检查器本身写出了它要检测的那个跨循环 bug**,只有真跑一次带 provider 才暴露。

实测输出(2026-10-12):

| 指标 | 结果 | 阈值 |
|---|---|---|
| M1 重放保真 | **100%**(6/6 请求逐字节重建) | 100% |
| M2 token 降低 | **79.1%**(36,490 → 7,638) | ≥ 40% |
| M2 质量代价 | **0.0 pp**(压缩前后成功率不变) | ≤ 5pp |
| M3 故障恢复 | **100%**(7/7 类故障) | ≥ 90% |
| M4 可复现性 | **100%**(3/3 重复运行一致) | — |
| M4 回归检出 | **100%**(3/3 注入回归) | 100% |

```bash
# 需要真实模型时(OpenAI 兼容端点)
export ARM_API_KEY=... ARM_MODEL=...
arm run "在这个仓库里找到那个失败的测试并修好"   # 跑一个任务
arm chat                                        # 交互
arm replay --session default                    # 从日志重建模型看到的每个字节
```

## 实现结构(与设计一致)

```text
src/arm/
  kernel/    container / waterfall / freeze / errors     可逆注册 + 瀑布链 + 深冻结
  session/   M1:envelope / surface / session / persistence / inbox
  llm/       seam / openai_adapter / mock                 模型接缝
  prompt/    assembler                                    有序段装配
  tools/     M3:definition / registry / pipeline / dispatch / drain / builtin
  context/   M2:meter / prune / spill / compact / manager
  eval/      M4:harness / judge / cases
  skills/    T3:registry / catalog / tool               渐进式披露的技能系统
  workspace/ T3:worktree / tools                        隔离检出(真实 git 仓库)
  plan/      T3:mode                                   计划模式(只改提示词)
  goal/      T3:store / tools                          持久目标(阶段持久/激活进程本地)
  approval/  安全:service                               审批(ask/never,唯一授权=allowed-once)
  sandbox/   安全:policy                                沙箱策略(拒绝为默认,拒绝带 code)
  todo/      任务:store / tools                        模型自用清单(整表替换,不进提示词)
  jobs/      任务:registry / ring                      后台作业(原子启动 + 双游标输出环)
  subagent/  T3:contracts / runtime / inprocess / tool 子 Agent(五步前置 + 会话隔离)
  team/      T3:journal / tools                        Agent Teams(一日志三投影 + DAG)
  schedule/  T3:runtime / tools                        定时提醒(独立存储 + 只补最新一次)
  mcp/       扩展:naming / client / stdio               MCP 客户端(远端工具=普通工具)
  hooks/     扩展:protocol / runner / bridge            外部 hook 桥到既有 waterfall
  plugins/   扩展:registry                              插件(单文件+列表,无三层叠加)
  browser/   接缝:cdp / tools                          浏览器(CDP 直连用户 Chrome)
  computer/  接缝:desktop / tools                      电脑操作(观察/动作分离,动作需审批)
  bench/     一条命令产出四项指标
  cli/       唯一入口
```

## T3 能力(按依赖拓扑推进)

| 序 | 能力 | 状态 |
|---|---|---|
| 1 | **Skill 系统** | ✅ 已实现(46 个单测 + 6 个集成测试) |
| 2 | **工作树** | ✅ 已实现(38 个测试,对真实 git 仓库) |
| 3 | **Plan mode** | ✅ 已实现(38 个测试) |
| 4 | **持久 Goal** | ✅ 已实现(57 个测试) |
| 5 | **子 Agent / Agent Teams** | ✅ 已实现(44 + 92 个测试) |
| 6 | **schedule** | ✅ 已实现(74 个测试) |
| — | **安全层(审批 + 沙箱策略)** | ✅ 已实现(61 个测试)——subagent 的前置 |
| — | **jobs(后台作业)** | ✅ 已实现(53 个测试)——subagent 的前置 |

## 实现期发现并修掉的真实缺陷

设计与实现之间总有缝。以下是实现**暴露出来的**、原契约没覆盖的问题:

- **完整清单**:[契约修正记录](docs/contract-amendments.md) —— 83 条,含复现步骤与实测输出
- **教训**(比缺陷更值得读):[方法论教训](docs/lessons.md) —— 同一个错误在项目里犯过两次以上才被记下来的那些

| # | 缺陷 | 为什么单测抓不到 |
|---|---|---|
| 1 | **只带 `tool_calls`、无文本的 assistant 消息被派生丢弃** | 单测全绿;只有真实 HTTP 端到端才暴露——真实模型 API 会拒绝这种对话 |
| 2 | `asyncio.shield` 单独**不足以**满足 M3.8 | 会通过原契约建议的那条测试,却在事件循环 teardown 时丢掉副作用 |
| 3 | schema 归一化器把 `$ref` 的兄弟键**覆盖**而非合取 | 静默**放宽**校验:非法参数能到达工具主体,而看起来一切正常 |
| 4 | 单分支 `anyOf` 折叠成 `anyOf: {...}`(**对象而非数组**) | 产出的是**无效 JSON Schema**,任何校验器都读不了 |
| 5 | 工具调用后未开新 step | 工具结果写进日志但模型永远看不到,任务半途而废 |

> 第 3、4 条来自对归一化器的**对抗性审查**(见 `spikes/q2_schema/`):审查者写了 3 个 `xfail(strict=True)` 测试,我修好实现后它们自动变成 XPASS 失败——一个修好的 bug 无法被悄悄改回去。


一个**以「模型可见即已记录」为唯一事实来源**的 Agent 运行时实现。

**定位(2026-10-12 定案,见 ADR-0003)**:

1. **目标是「一个能真正用的 harness」,不是简历交付物。** 简历项目另行落到具体场景中做自己的优势。
2. **「彻底复刻 DSH」= 设计对齐**,不是功能对齐——参考其分层设计(接缝 / 事件域 / 日志即事实)以避免绕弯路,实现要有自己的东西。
3. **范围:全做,不裁剪。** DSH 有的、社区在填的,都做。
4. **从零 clean-room 重写,语言 Python** —— 不用 DSH 作依赖或底座,与 TypeScript/Cordis 生态完全脱钩。
5. **spool 抛弃**:不读、不借鉴、不复用。
6. **起点:先补内核主干(M1–M4)**,它是全部能力的公共前置。

```text
模型能看见的输入
  → 必须成为一条已提交的 Session 事件
  → 必须能从事件日志重建
  → 上下文压缩、恢复、回放、评测全部从同一份日志派生
```

## 为什么做这个

- 想要一个**自己完全掌握、能改能修、长期自用**的 agent harness,而不是依赖一个不可控的上游产品。
- 本项目从 [sources/dsh-study](../../sources/dsh-study/README.md) 的源码拆解反向生长,**每个设计决策都有上游依据**,不重复踩 DSH 已经踩过的坑。
- 与 [paper-overclaim-harness](../paper-overclaim-harness/) 的分工:**先有 runtime,再在其上做场景**——检测器目前天花板被宿主 runtime 锁死。

## 能力范围(2026-10-12 重构)

> ⚠️ **下面「四个机制」是旧版收敛定位的核心。** ADR-0003 已把 T3 全部纳入,范围不再限于四个机制。保留本节是因为 M1–M4 仍是**内核主干与叙事地基**,但**不再是项目的全部内容**。完整能力清单与顺序待定(见 ADR-0003 §6)。

### 内核主干(四个机制)

M1 是地基;M2 与 M3 都是「向日志追加正确的事件」,M4 是「从日志读出证据」。**同一份日志的四种用途。**

### M1 · 日志即事实(Log-as-Truth)

| 项 | 内容 |
|---|---|
| 上游依据 | `packages/core/session`、`packages/session/session-projection` |
| 要回答的问题 | 模型看到的东西怎么保证可复现、可恢复、可回放? |
| 核心设计 | 只追加的事件日志;`durable fact` 与 `live control signal` 分离;上下文是日志的**派生投影**而非独立状态 |
| 讲得出的深度 | 为什么"日志即事实"是 Agent 可恢复性的前提;投影增量 fold 的失效场景 |

### M2 · 上下文压力处理(Compaction + Spill)

| 项 | 内容 |
|---|---|
| 上游依据 | `packages/compaction`、`packages/spill` |
| 要回答的问题 | 上下文超预算时,砍什么、怎么砍,以及砍掉的怎么还能找回来? |
| 核心设计 | 压力阈值触发 → 摘要替换历史区段 → 工具结果裁剪 → 超预算大结果落盘并返回 locator |
| 讲得出的深度 | 压缩为什么必须**追加事件**(不能就地改历史);摘要失真如何被发现 |

### M3 · 工具执行流水线 + 能力接缝

| 项 | 内容 |
|---|---|
| 上游依据 | `packages/core/tools`、`packages/sandbox/sandbox-policy`、`packages/interaction/user-approval` |
| 要回答的问题 | 工具调用失败、超时、越权、返回超大结果时,运行时怎么不崩、怎么留痕? |
| 核心设计 | `definition → provider → consumer` 三段式接缝;`pre-execute → guard → execute → post-execute → result` 流水线;结果归一化后**必落日志** |
| 讲得出的深度 | 为什么能力要拆成定义/提供/消费三段;权限与沙箱在流水线的哪一段介入 |

### M4 · Eval 基础设施(回归集 + Trace 评估 + LLM-as-Judge)

| 项 | 内容 |
|---|---|
| 上游依据 | **无——DSH 没有这个能力**(实测:`packages/` 下无 eval 包;市场 4,508 个插件中仅 20 个涉及) |
| 要回答的问题 | 改了提示词 / 换了模型 / 调了压缩阈值之后,任务成功率是涨了还是跌了? |
| 核心设计 | 从 M1 的事件日志派生评估输入;固定回归集;Trace 级评估;LLM-as-Judge 打分;A/B 对照 |
| 讲得出的深度 | 为什么 Eval 必须从**同一份日志**派生(而不是另建一套记录);judge 的偏差如何被检出 |

> **M4 的定位(2026-10-12 更新)**:原排序依据是「T1 岗位 JD 命中」,**该依据已作废**(项目与求职解耦)。但 M4 反而更靠前了——理由变了:要判断「改动之后是变好还是变坏」,Eval 是让这件事**可观测**的手段,不是可选项。依据见 [market-native-builtins.md](docs/market-native-builtins.md) §5 与 [t3-ordering-and-guardrails.md](docs/t3-ordering-and-guardrails.md) §1.3。

> M1 是地基,M2 和 M3 都是「向日志追加正确的事件」,M4 是「从日志读出证据」。这是**内核主干**的叙事主线;T3 各项能力挂在其上。

## 能力范围(2026-10-12 裁断:**全做,不裁剪**)

> **范围已定:DSH 有的、社区在填的,都做。** 不设「暂不内建」清单([ADR-0003](docs/decisions/0003-usable-harness-and-t3-scope.md) §6.3)。
> 顺序与执行护栏见 [t3-ordering-and-guardrails.md](docs/t3-ordering-and-guardrails.md) —— **全做 ≠ 同时做**,依赖拓扑决定先后(`subagent` 依赖 22 个包,必须最后)。

仍然明确不做的只有**方向性**约束(不是能力裁剪):

- ❌ **不追求与 DSH 行为一致**,也不以兼容 DSH 为目标
- ❌ 不抄 DSH 源码(见 [ADR-0001](docs/decisions/0001-scope-and-non-goals.md))
- ❌ 不读 / 不借鉴 / 不复用 `sources/_archive/spool/`(见 ADR-0003 §5)

**已纳入范围**:内核主干(M1–M4)、子 Agent / Agent Teams / mailbox / 共享任务 DAG、MCP、skill 系统、插件系统、bundle/patch 组合模型、完整沙箱后端(含 Docker)、电脑操作 / 浏览器、工作树、Plan mode、持久 Goal、jobs、schedule。

> ⚠️ **原「使用判据」护栏已作废**:曾用「会不会被我日常使用」来否决能力,但**我无法知道本人日常需要什么**,该判据不可执行。现只保留与「怎么做」有关的三条护栏(顺序 / 垂直切片 / 一次一个)。见 [t3-ordering-and-guardrails.md](docs/t3-ordering-and-guardrails.md) §2。

## 验收标准(2026-10-12 修订)

> ⚠️ 原 L3 指标是**为简历叙事设计的**。ADR-0003 已把项目与求职解耦,「L3」这个标签失去外部意义。下表降级为**「M1–M4 是否真的 work」的内部指标**,不再是项目的完成定义。

| 机制 | 量化验收 | 测法 |
|---|---|---|
| M1 日志即事实 | 从事件日志重放出的模型可见输入与原始请求**逐字节一致** | 自动化测试,diff 为空 |
| M2 上下文压力 | 长会话下 prompt token **降低 ≥ 40%**,同一评测集任务成功率下降 **≤ 5%** | 固定评测集 + 压缩前后对照 |
| M3 工具流水线 | 注入 6 类故障(超时 / 非法入参 / 权限拒绝 / 重复调用 / 超大结果 / 中途取消),**恢复成功率 ≥ 90%** 且均留下可解释的日志痕迹 | 故障注入测试 |
| M4 Eval | 固定回归集上任务成功率可重复测量(两次运行偏差 ≤ 5%);人为注入 3 个已知回归,检出率 **100%** | Eval 驱动 |
| 整体 | 一条命令跑完测试与评测,输出上述四项指标 | `make bench` 或等价 | 

达不到这四项就是「内核没做对」,需要继续修。**宁可缩小机制范围,也不要放宽指标。**

## 时间线(2026-10-12 修订)

> ⚠️ **原时间线作废。** 它建立在「设计期 8–16 小时 + 2027.01–02 集中实现 + 2027.02 写进简历」之上,其中两个前提已被 ADR-0003 推翻:①项目不再是简历交付物,「2027.02 写进简历」这个截止点消失;②范围纳入 T3。

实测等价规模(按 DSH 同类实现):

| 块 | LOC |
|---|---:|
| 内核主干(agent / loop / session / tools / prompt / llm / projection) | 25,713 |
| T1+T2(compaction / spill / sandbox / interaction / goal / persistence) | 24,122 |
| T3(subagent / agent-team / skill / browser-use / computer-use) | 22,912 |
| 「能真正用」还必需(fs / shell / subprocess / lsp / web / mcp) | 32,538 |
| **合计** | **≈105,285** |

参考:DSH 非测试代码 414,707 行(其中 client 139,016 / experimental 49,353 / api 20,627 / extensions 21,734 是产品化开销,自研不需要)。

> **⚠️ 这是等价规模,不是工作量。** 我曾据此外推「以年计」,**已被推翻**:`spool` 用 **31.4 小时**(首末提交 2026-09-19 16:52 → 09-21 00:13)产出 **27,819 行**自研 TypeScript,完成 P0+P1。按此校准,本项目是**以周计**。详见 [ADR-0003](docs/decisions/0003-usable-harness-and-t3-scope.md) §7。

**真正的瓶颈不是产出量,是可靠性** —— 压缩失真、工具失败恢复、会话损坏这些「能用/不能用」的分界,以及把快速产出的代码收敛成一个连贯系统。见 ADR-0003 §7.3。

**新时间线待定**,依赖 ADR-0003 §6.2 的待决项。

## 设计产物(2026-10-12 完成)

```text
agent-runtime-min/
  README.md                      本文件:立项、范围、验收
  docs/
    runtime-spine.md             ✅ 运行时主链路:turn/step 事件序、19 个事件、不变量 S1–S14
    contracts/
      M1-log-as-truth.md         ✅ 日志即事实:派生算法、surfaceOp、不变量 M1.1–M1.17
      M2-context-pressure.md     ✅ 上下文压力:两套价、阈值、事务时序、不变量 M2.1–M2.20
      M3-tool-pipeline.md        ✅ 工具流水线:七段、失败即结果、不变量 M3.1–M3.15
      M4-eval-infrastructure.md  ✅ Eval:确定性优先、judge 防偏、注入回归自证、M4.1–M4.12
    decisions/                   ADR-0001 / 0002 / 0003
    market-native-builtins.md    DSH 插件市场实测(4,508 插件 / 23 分类)
    t3-ordering-and-guardrails.md T3 顺序(依赖拓扑)+ 执行护栏
  src/arm/                                    实现(82 个模块 / 28,389 行)
  tests/                                      1,250 个测试 / 23,102 行
  spikes/                                     前置验证与对抗性审查(Q1 取消语义 / Q2 schema / Q3 瀑布)
  Makefile                                    make bench = 测试 + 四项验收指标
```

**四份契约的定位**:

| 契约 | 上游对齐 | 我们的偏离 |
|---|---|---|
| M1 | 强(32 条不变量可对照) | 裸 JSONL;无迁移链;19 事件;不做投影缓存 |
| M2 | 强(20 条不变量可对照) | **headroom 默认改小**(上游默认让它成为支配项);不做清扫 |
| M3 | 强(15 条不变量可对照) | 用 `pydantic` 而非自研 schema 编译器;不做工具呈现接缝 |
| M4 | **无**(DSH 零实现,市场 20/4,508) | 全部原创 |

> 每份契约都含:**不变量表 + 失败分支矩阵 + 「我们的选择 vs 上游」+ 未解决项**。未解决项不是缺陷,是**实现期必须实测的清单**。

## 规则

- **clean-room**:只读上游文档、类型定义与决策记录以理解契约,**不复制实现代码**。理由见 [ADR-0001](docs/decisions/0001-scope-and-non-goals.md)。可检验形式:任何设计文档**抽掉 DSH 的名字后仍须独立成立**(见 [ADR-0002](docs/decisions/0002-design-alignment-and-native-builtins.md) §5)。
- 设计期每份文档都要标注**上游源码依据**,让每个结论都可追溯。
- 新增能力用[排序工具](docs/decisions/0003-usable-harness-and-t3-scope.md) §3 决定先后,不再用于否决。
- **不读 `sources/_archive/spool/`**。

> ⚠️ **项目位置待定**:[02-projects](../../README.md) 的定义是「能拿出去给人看的简历项目」,本项目已不属于该定义(ADR-0003 §4)。是否移出待范围稳定后决定。

## 相关

- 素材来源:[sources/dsh-study](../../sources/dsh-study/README.md)
- 知识沉淀:机制契约文档将同步生长为 [03-knowledge](../../03-knowledge/README.md) 的条目
- 场景项目(简历向):[paper-overclaim-harness](../paper-overclaim-harness/)
