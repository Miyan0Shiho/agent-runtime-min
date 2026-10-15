# 前置验证汇总（§9 三项 + 对抗性审查）

> 建立 2026-10-12 · 实施期前置验证 · 每份 FINDINGS 都在各自目录下
> **用途**：REQUIREMENTS §9 要求「实现前必须先定清楚」的三项，以及一次对已实现代码的对抗性审查。这里是结论入口。

---

## 一句话结论

**三项前置验证全部完成，其中两项推翻了需求书自己的建议。**

| # | 问题 | 结论 | 是否推翻需求书 |
|---|---|---|---|
| **Q1** | `asyncio` 能否承载「取消不抛弃已启动的协程」？ | **能，但需要复合模式**，任何单一原语都不够 | ✅ **是**——§9「语义等价」的说法不成立；§11 建议的单条测试**不够** |
| **Q2** | `pydantic` 生成的 schema 是否满足模型侧兼容性？ | **需要归一化，且原归一化器有两个静默错误** | ✅ **是**——已发现并修掉 2 个真实 bug |
| **Q3** | `pre-execute` 瀑布不调 `next()` 的语义？ | **明确定义为「短路」**，并已实现 + 负面测试 | ➖ 需求书的建议被采纳 |

---

## Q1 · 取消语义 → 见 [q1_cancellation/FINDINGS.md](q1_cancellation/FINDINGS.md)

**命令**：`cd spikes && ../.venv/bin/python -m pytest q1_cancellation -v` → **30 passed**

**可实现的规则**（四部分缺一不可）：

1. 启动与登记**同步**完成（`create_task` 与登记之间不得有 `await`）
2. 用 `done_callback` 记录结果，**不依赖任何 awaiter**
3. 在 `finally` 里 **shield 下 drain**，用 `uncancel()` 吸收重复取消
4. **重抛恰好一次** `CancelledError`

**关键发现**：

- `await asyncio.create_task(body())` **不是 detach**——取消会被传播进内层 task，主体照死。这是从 `AbortSignal` 迁移过来时最常见的错误心智模型。
- **`shield` 单独不够**：主体活下来但**结果无人收集**，且 `asyncio.run` teardown 时 task 仍会被取消——**恰好在真实 turn 结束那一刻丢掉副作用**。
- **`TaskGroup` 与裸 `asyncio.timeout` 主动破坏**该契约：两者都取消它们包裹的工作。
- `Task.exception()` 在已取消的 task 上**会抛** `CancelledError`，必须**先判 `task.cancelled()`**。
- **Drain masking**：`finally` 里无保护的 `await` 会让主体的异常**顶替** `CancelledError`，把 abort 报成工具错误。

**过程可信度**：做了 7 个变异体测试，全部被杀。其中「硬编码 cause」那个**一开始存活**，暴露了真实覆盖缺口，补测后被杀。

**已落地**：`src/arm/tools/drain.py` + `tests/test_m3_cancellation.py`（14 条，含 1 条负对照）。

---

## Q2 · schema 兼容性 → 见 [q2_schema/FINDINGS.md](q2_schema/FINDINGS.md)

**命令**：`cd spikes && ../.venv/bin/python -m pytest q2_schema` → 15 passed / 10 xfailed（修复后其中 3 条转为 XPASS）

**发现的两个真实缺陷**（均已修复并加了永久回归测试）：

| 缺陷 | 症状 | 危害等级 |
|---|---|---|
| **`$ref` 兄弟键被覆盖而非合取** | 兄弟 `properties` 会**丢掉**目标的约束 → 接受原本被拒的载荷 | **静默放宽校验**：非法参数到达工具主体 |
| **单分支 `anyOf` 折叠成对象** | 产出 `"anyOf": {...}`（对象，而非数组） | **产出无效 JSON Schema**，任何校验器都读不了 |

**修法**：真合取（`properties` 合并 / `required` 求并 / `anyOf` 累加）；**无法合取时拒绝**，而不是二选一。折叠只在一处发生，且必须看到原始数组。

**过程可信度**：审查者用 `xfail(strict=True)` 标记缺陷——bug 存在时测试失败记为 xfail，修复后自动变成 **XPASS(strict) = 失败**。这是一个棘轮：**修好的 bug 无法被悄悄改回去**。

---

## Q3 · 瀑布语义 → 见 [q3_waterfall/](q3_waterfall/)

**命令**：`cd spikes && ../.venv/bin/python -m pytest q3_waterfall -q` → **17 passed**

**定义的语义**（需求书建议被采纳）：

> **不调 `next()` 即短路**，后续监听器不运行。既不继续也不返回决策 = 契约错误，不是静默放行。

**额外发现**：模糊实现（`None` + 未调 `next()` → 继续）**不仅多跑了监听器，还丢掉了终端的返回值**——比「多跑几个监听器」更糟。已写进负面测试。

**已落地**：`src/arm/kernel/waterfall.py`，语义与 `_collapse` 一致；`tests/test_m3_tool_pipeline.py` 有 4 条对应测试。

---

## 这些验证改变了什么

1. **需求书 §9.1 的表述不成立**——`AbortSignal` 是**被动标志**（被调用方观察），`CancelledError` 是**在 await 点注入的异常**。被动标志无法摧毁忽略它的工作；注入的异常会，而且确实会。
2. **§11 item 1 建议的单条测试不够**——它会在 shield-only 实现上**通过**。至少需要 4 条测试覆盖不同失效模式。
3. **M3.5 无法按字面实现**——Python 没有可替换的 signal 对象。它的真实意图须重述为「包装器不得移除取消能力」。
4. **M3 §7 与 runtime-spine §5 互相矛盾**——「取消」被混用于「这次调用被取消」（一个结果，step 继续）与「整个 turn 被取消」（step 死掉）。两者需要**相反**的代码路径。

全部四条已写入 [契约修正记录](../docs/contract-amendments.md)。
