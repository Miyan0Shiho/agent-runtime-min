# M3 契约:工具执行流水线 + 能力接缝

> 状态:设计产物 · 2026-10-12 · 作者:本人
> 上游依据:[62-tools-deep](../../../../sources/dsh-study/kb/chapters/62-tools-deep.md)、[21-tool-system](../../../../sources/dsh-study/kb/chapters/21-tool-system.md)、[图册 02](../../../../sources/dsh-study/kb/diagrams/02-tool-pipeline.md)
> 前置:[M1 契约](M1-log-as-truth.md)(工具结果必须落日志并成对)
> **本文抽掉上游名字后必须能独立成立。**

---

## 0. 一句话结论

工具系统的本质是**「失败即结果」**:参数错、工具不存在、权限拒绝、主体抛错、输出不合法——全部收敛成一条 `isError: true` 的结果,**没有一条会中断步骤**。唯一能中断轮次的是工具**主动声明**「本轮到此结束」。

第二条本质是**参数一次冻结、永不可改**。所有策略(权限、审计、脱敏)因此只能**否决或替换结果**,不能改写参数。这不是限制,是安全设计:能改参数的守卫,等于能绕过参数校验。

---

## 1. 能力接缝:定义 / 提供 / 消费

一个可替换能力拆成三个角色,分属不同变化速率:

```text
定义(Service Definition)  ← 变化最慢:能力是什么
  → 提供(Provider)        ← 变化中等:怎么跑
  → 消费(Consumer)        ← 变化最快:模型面向什么编程
```

**为什么拆**:换一个执行后端(例如换沙箱)不应该搅动模型看到的工具 schema。三角色分离后,后端的替换不触碰任何 schema。

**纪律:不要预防性拆分**。只有一个可设想提供方和一个消费方时,**保持一个模块**。拆包本身有成本(package.json、配置、接线)。

| 能力 | 定义 | 提供(第一版) | 消费 |
|---|---|---|---|
| 文件系统 | `fs` 接缝 | 本地实现 | `read`/`write`/`edit` 工具 |
| 命令执行 | `shell` 接缝 | 本地 shell | `bash` 工具 |
| 沙箱 | `sandbox` 接缝 | 策略层 + 一个本地后端 | (被 fs/shell 消费) |
| 模型 | `llm` 接缝 | 一个适配器 | (被循环消费) |

---

## 2. 七段流水线

```text
模型返回 tool_call
  ① 物化 + 冻结参数          create_execution()
  ② pre-execute(waterfall)   策略层:可 deny / cancel / ask
  ③ guard 判定               单调:只能否决,不能放行
  ④ execute(环绕包装)        可换 signal(会被重新融合)
  ⑤ 工具主体                 tool.execute()
  ⑥ project_content()        投影出模型可见内容
  ⑦ post-execute(waterfall)  可改写 content / 替换 value / block
  → finalize_content()       每个结果恰好一次、必须 total
  → 提交 tool/result
```

**关键约束**:

1. **策略决策没有参数字段**——任何一段都不能改写工具参数。
2. **只有 `execute` 能替换 signal**,且注册表会在主体前**重新融合**调用方信号,防止「换 signal」等价于「摘掉取消」。
3. **「走不走 post-execute」由抛错发生在哪个块决定**:`execute` 包装器抛错 → 跳过 post-execute;工具主体抛错 / 参数校验失败 / 输出校验失败 → **post-execute 一定运行**。

---

## 3. 三条异常路径都变成结果

| 来源 | 模型可见文本 | 走 post-execute? | 中断步骤? |
|---|---|---|---|
| 参数非法 | `Error: invalid arguments: <violations>` | 是 | 否 |
| 未知工具 | `Error: unknown tool "<name>"`(且**仍走完 pre-execute**) | 是 | 否 |
| 权限拒绝 | `Error: <reason>` | 是 | 否 |
| 工具主体抛错 | `Error: <message>` | 是 | 否 |
| 输出不合法 | `Error: tool "<name>" returned invalid output: <violations>` | 是 | 否 |
| `execute` 包装器抛错 | `Error: <message>` | **否** | 否 |
| 调度器自身失败 | (由步骤记录恢复结果) | — | **是** |

**唯一能中断轮次的成功路径**是工具主动声明 `concludes_turn`。

---

## 4. 并发模型:一元 fail-closed + 模型序提交

```text
同一 assistant 消息里的 N 个 tool_call
  每个调用: is_concurrency_safe(args) === True ?
    是  → 可并行(有界滚动池)
    否  → 独占 + 构成屏障
  并行只发生在「派发 + 工具主体」阶段
  pre/post-execute 与结果提交严格按模型序
```

**判据是「显式 True 才并行」**:未声明或返回非 True 一律独占。这是 fail-closed——新工具默认安全。

**中止时**:为未派发的调用补 `ABORTED_BEFORE_DISPATCH` 合成结果,否则下一次请求会出现悬空 `tool_call`。

---

## 5. 超时与溢出是外挂策略,不是内核

```text
工具系统内核 = 注册表 + 七段流水线
   ├─ execute 包装 → 超时策略 → TOOL_TIMEOUT
   └─ post-execute 挂载 → 溢出策略 → 落盘
```

**这条最容易误读**:注册表本身**不做**超时与溢出。「是否走溢出 / 是否超时」取决于部署组合,不是默认行为。

**为什么这样设计**:内核保持最小且可预测;策略按需组合。

---

## 6. 不变量

| # | 不变量 | 违反了会怎样 |
|---|---|---|
| M3.1 | **单一视图派生**:「可见集合 / 已知名字 / 可过滤名字」三个结果从**一次遍历**产出 | 三处各自实现 shadowing,必然发散 |
| M3.2 | **限制只作用于继承面;自有层永久豁免** | 子 agent 会失去回报工具的能力 |
| M3.3 | **保留名不可注册、不可限制、不可遮蔽** | 两个模式可同时暴露同名工具,通告面 ≠ 可调用面 |
| M3.4 | **参数在策略之前一次性快照 + 深冻结** | 策略可改写参数 → 日志/UI/程序化调用三者不一致 |
| M3.5 | **只有 `execute` 包装器可换 signal,且主体前重新融合调用方信号** | 「换 signal」=「摘掉取消」 |
| M3.6 | **成功结果必须过输出 schema**:主体、post-execute 替换、包装器自造三条路径**都**重新校验 | 包装器绕过校验 |
| M3.7 | **`finalize_content` 每个结果恰好一次、必须 total**;抛错也被收敛成错误结果 | 结果丢失或重复 |
| M3.8 | **取消永不抛弃已启动的 promise** | 副作用发生了但日志说没发生 |
| M3.9 | **结果按模型序提交**;中止时补合成结果 | 悬空 tool_call |
| M3.10 | **结构化错误身份随结果持久化,但永不进模型内容** | 内部信息泄漏给模型;或丢失排障依据 |
| M3.11 | **`tool/result` 必须引用其 `tool/call` 的 seq**;跳过的调用也必须成对写 | 配对断裂 |
| M3.12 | **注册即可逆**:注册函数返回 disposer | 插件卸载后残留注册 |
| M3.13 | **限制拒绝空过滤器** | 空过滤器几乎总是「materialize 空配置」的 bug |
| M3.14 | **限制拒绝未知名字并列出已知工具** | 静默失效的配置 |
| M3.15 | **限制拒绝在全局上下文调用** | 全局限制会遮蔽所有 agent |

---

## 7. 失败分支矩阵

> 上游这张表有 30+ 行(见依据 §9)。以下是**我们第一版必须覆盖的**子集。

| 触发条件 | 模型可见文本 | 走 post? | 中断 step? |
|---|---|---|---|
| 参数不可无损快照 | `Error: tool execution arguments must be losslessly JSON-serializable` | 否 | 否 |
| 参数校验失败 | `Error: invalid arguments: <violations>` | 是 | 否 |
| 未知工具 | `Error: unknown tool "X"` | 是 | 否 |
| `pre-execute` 返回 deny | `Error: <reason>` | 是 | 否 |
| `pre-execute` 返回 cancel | `Error: tool call aborted before dispatch` | 是 | 否 |
| 用户拒绝审批 | `Error: the user rejected tool "X"` | 是 | 否 |
| 无审批通道 | `Error: tool "X" requires approval, but no approval channel is available` | 是 | 否 |
| 主体抛任意值 | `Error: <message>` | 是 | 否 |
| 返回值不满足输出 schema | `Error: tool "X" returned invalid output: <violations>` | 是 | 否 |
| `post-execute` 同时给 content 与 value | `Error: cannot replace both value and content` | 是 | 否 |
| `post-execute` 给失败结果换 value | `Error: cannot replace the value of a failed result` | 是 | 否 |
| `post-execute` 返回 block | feedback 文本;空则固定句 | 已在其内 | 否 |
| 超时 | `Error: tool call timed out after <n>ms`;`info.code = TOOL_TIMEOUT` | 是 | 否 |
| 主体启动后取消 | `Error: tool call aborted`;`info.code = ABORTED` | 是 | 否 |
| 主体前取消 | `Error: tool call aborted before dispatch` | 视路径 | 否 |
| 调度器内部失败 | — | — | **是** |

---

## 8. `exec` 对象的生命周期(可照抄的形态)

| 阶段 | 完整性 | 冻结 |
|---|---|---|
| 物化返回后 | 身份字段只读;`arguments` 深冻结 | 未冻结 |
| 策略流水线期间 | 包装器只能改 signal | 未冻结 |
| 提交结果前 | 结果字段仍归注册表所有 | **冻结** |
| 结果观察者 | 完全只读 | 已冻结 |

**两个互斥臂**:成功带执行局部的 `value`;失败带 `error{message, info?}`。`value` 在失败臂上不存在。

**关键**:持久事件里**没有** `value`——只把 `content` / `meta` / `additional_contexts` 当持久投影。这是「模型可见 ⟺ 可重建」的具体体现。

---

## 9. 我们的选择 vs 上游

| 维度 | 上游 | 我们 | 理由 |
|---|---|---|---|
| 语言 | TypeScript | **Python** | 本人主力栈 |
| schema 校验 | 自研 DSL 编译器 + 原始 JSON Schema 子集 | **`pydantic` 生成 schema** | 上游自研是因为要在浏览器侧跑同一套;我们不需要 |
| 参数快照 | 手写无损 JSON 检查 | **同构保留** | M3.4 是本质 |
| 并发池 | 有界滚动池,默认 10 | **同构保留** | 默认值可配 |
| 取消 | `AbortController` | **`asyncio` + 自定义 cause** | 需验证语义等价(见 §11) |
| 超时策略 | 独立插件包 | **同包内实现** | 只有一个提供方时保持一个模块 |
| 工具呈现 | 独立的呈现接缝 + 纯函数 presenter | **不做** | 第一版无 UI |
| PTC(程序化工具调用) | 完整实现 | **不做** | T3 阶段评估 |

**明确不抄的**:

- **自研 JSON Schema 编译器**。上游需要它是因为 schema 要同时服务浏览器与宿主;我们用 `pydantic` 即可,代价是失去「DSL 保证可推理类型」这一层。
- **工具呈现接缝**(`mode` 单值格子、惰性 getter)。第一版无 UI。
- **`markCanonical` 的 WeakMap 去重**。上游用它防止包装器伪造结果;我们用「所有成功结果都走同一校验入口」(M3.6)即可。

---

## 10. 依据(上游证据索引)

| 结论 | 证据 |
|---|---|
| 七段流水线与「谁能否决」 | [图册 02](../../../../sources/dsh-study/kb/diagrams/02-tool-pipeline.md) |
| 不变量 I1–I10 | [62 章](../../../../sources/dsh-study/kb/chapters/62-tools-deep.md) §4 |
| `exec` 生命周期表 | [62 章](../../../../sources/dsh-study/kb/chapters/62-tools-deep.md) §4 |
| 失败分支全枚举(30+ 行) | [62 章](../../../../sources/dsh-study/kb/chapters/62-tools-deep.md) §5 |
| 「本质十条 vs 细节十二条」 | [62 章](../../../../sources/dsh-study/kb/chapters/62-tools-deep.md) §6 |
| 注册表与 restrict 的错误设计 | [41-porting-playbook](../../../../sources/dsh-study/kb/chapters/41-porting-playbook.md) §3.2 契约 6 |
| 能力接缝三角色的决策原文 | [40-design-philosophy](../../../../sources/dsh-study/kb/chapters/40-design-philosophy.md) §2.1 |
| 上游源码位置 | `packages/core/tools/src/index.ts`、`packages/core/agent-loop/src/tool-calls.ts` |

---

## 11. 未解决 / 待验证

1. ~~**【未验证】Python `asyncio` 能否承载 M3.8（取消不抛弃已启动的 promise）。**~~ **已验证（2026-10-16）**：`tests/test_m3_cancellation.py` 有 12 条测试覆盖这个契约项，其中 `test_naive_await_loses_the_side_effect` 先**复现朴素写法的失败**，`test_drain_lets_a_started_body_finish` 与 `test_cancelled_step_still_records_a_result_for_a_started_body` 断言契约要求的结果。

   **结论：能承载。** Python 的 `CancelledError` 在 `await` 点注入确实与 JS 的 `AbortSignal` 传播模型不同——这正是 `DrainScope` 存在的原因：它把已启动的 body 放进一个**拥有自己的排空流程**的 scope，使取消不会把它连同结果一起抛弃。
2. ~~**【未验证】`pydantic` 生成的 schema 是否满足模型侧的兼容性。**~~ **已用真实模型验证（2026-10-16，deepseek-chat）**：**兼容**。

   **先纠正契约的一个前提**：这个风险比契约假设的低，因为工具参数是**手写的 JSON Schema**，不是从 pydantic 模型生成的。实测**没有任何 schema 出现 `$defs`/`anyOf`/`$ref`**——这正是不该出现的东西。已加测试固定这一点，这样「将来引入生成式 schema」是一个**可见的变更**而不是悄悄的。

   真实往返验证通过：模型返回工具调用、参数类型正确；**`usage` 与 `finish_reason` 都被填充**（这两者曾被本项目丢弃，此处是对真实 provider 的端到端确认，而 FakeServer 无论如何都会发送它们，所以只有真实调用能证明）。

   **一次测量方法的自我纠正**：我最初把「接受率」定义成「模型有没有调用我期望的那个工具」，测得 10/12，看起来像 16% 的 schema 失败。**它不是** ——`edit` 被跳过是因为模型**先读再改**，那是合理行为。当 `edit` 是唯一选项时它**每次都被选中**。

   > **把「模型偏好另一个工具」当成「schema 被拒绝」会产出一个毫无意义的数字。** 这个区分已写进测试文件的 docstring。

   测试在无 key 时 skip：`ARM_TEST_MODEL_KEY=... pytest tests/test_m32_real_model.py`。
3. ~~**【未验证】`pre-execute` 瀑布的 `next()` 语义。**~~ **已定义并实现（2026-10-16）**：语义采用契约自己的建议——**不调 `next()` 即短路，后续监听器不运行**。`tests/test_m3_tool_pipeline.py::test_first_hook_denying_without_next_short_circuits` 就是那条负面测试。
4. ~~**【推断】限制(restrict)与后续注册的时序缝。**~~ **已确认（2026-10-16）**，语义与契约推断一致，并已在代码里写明：`ToolRegistry.restrict` 的 docstring 说「`allow`/`deny` 针对**当下**存在的工具校验（M3.14），而**稍后注册的工具可以合法地被加进限制**」。

   理由：校验的是 **allow 条目**（一个什么都没指到的名字是静默的配置错误），而**准入是每次调用现算的**。实测：`restrict(allow=['read'])` 之后注册 `later`，`later` 可被取到。已加测试固定这个语义。
5. ~~**【未验证】超时与取消的优先级。**~~ **已查（2026-10-16）**：**不存在嵌套 deadline**，因此没有这层区分要做。

   契约自己写了条件是「取决于是否支持嵌套 deadline」。实测：`src/arm/tools/` 与 `agent/` 里**没有 deadline 概念**，超时是**工具 body 自己的**（`bash` 收 `timeout_ms`，默认 60 秒，超时返回 `TOOL_TIMEOUT`），**流水线本身没有超时策略**。

   所以「超时策略的计时器」与「外层嵌套 deadline」不会同时存在，二者优先级也就无从冲突。**若将来引入流水线级 deadline，这一条要重新打开** —— 这是一个有条件的关闭，不是「已解决」。
6. **【未验证】M3.10 的「结构化错误身份不进模型内容」在自用场景下的价值。** 上游需要它是因为有遥测与 UI;我们的第一版没有消费者。触发条件:引入 M4 的 trace 评估时,该信息会成为评估输入——届时必须实现。
