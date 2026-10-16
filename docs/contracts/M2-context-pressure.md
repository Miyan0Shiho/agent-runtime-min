# M2 契约:上下文压力处理(计量 + 裁剪 + 摘要 + 落盘)

> 状态:设计产物 · 2026-10-12 · 作者:本人
> 上游依据:[64-compaction-spill-deep](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md)、[25-context-compaction-spill](../../../../sources/dsh-study/kb/chapters/25-context-compaction-spill.md)、[图册 03](../../../../sources/dsh-study/kb/diagrams/03-session-surface.md) §3
> 前置:[M1 契约](M1-log-as-truth.md)(压缩是「向日志追加 replace 事件」)
> **本文抽掉上游名字后必须能独立成立。**

---

## 0. 一句话结论

上下文压力处理的本质**不是「砍内容」,而是「向日志追加一次合法的替换」**。所有削减手段(裁剪、摘要、落盘)都只是**产生替换事件的不同方式**;而替换的合法性由 M1 的 `surfaceOp` 规则保证。

因此有一条反直觉但重要的推论:**压缩失败不应该损坏会话**。任何一步失败,日志里要么什么都没留下,要么留下一个可被识别的孤儿标记——不存在「压缩到一半」的状态。

---

## 1. 三种手段的分工

| 维度 | 落盘(spill) | 裁剪(prune) | 摘要(compact) |
|---|---|---|---|
| 触发时点 | 每次工具结果落地 | 压力达标后、摘要之前 | 压力仍超时 |
| 预算单位 | **token** | **字符 / code point** | **token** |
| 处置 | 保留首尾 + **落盘** + 回执 | 保留首尾 + 插入标记 | 用模型摘要替换整段 |
| 花模型调用 | 否 | 否 | **是** |
| 日志操作类型 | **append**(不破坏缓存) | replace(单点) | replace(区间) |
| 失败时 | 保留原内容(warn) | 抛,但**当轮已落地的替换保留** | 见 §6 |

**顺序关系**(一次压力触发的完整序列):

```text
spill(工具时点,独立发生)
  → 测压
  → prune(不花钱)
  → 重测
  → 若仍超 → 摘要(花钱,可多轮)
```

**为什么这个顺序**:先做不花钱的。裁剪不达标就不调模型——这条顺序本身就是成本优化。

---

## 2. 计量:两套价必须分离

这是 M2 最容易做错的地方,也是上游「本质」清单的第①条。

| 价 | 用途 | 特点 |
|---|---|---|
| **启发式价** | 供替换做「影子价」 | 与路由无关;必须**可加** |
| **路由价** | 判断真实压力 | 按当前路由的实际表示计价 |

**为什么必须两套**:

- 少了启发式价 → O(1) 的压力投影无法自洽。替换发生时,必须能立刻回答「削减了多少」,而这个数不能依赖「当前用哪个模型」。
- 少了路由价 → 阈值判的不是真压力(同一段文本在不同模型下 token 数不同)。

**最小实现**:同一节点上并存两个数——`heuristic_tokens`(可加)与`tokens`(可替换);**替换事件必须携带被遮蔽区间的启发式总价**。

### 2.1 影子价协议(把「削减了多少」压成一个相邻对)

```text
每个 replace 事件,必须有一条紧邻其前、覆盖完全相同区间的影子价事件
```

**为什么**:否则压力投影要么保留每节点价格(O(surface) 状态),要么只能漂移。相邻对让它保持 O(1)。

**约束**:影子价与替换**必须同步相邻、区间完全一致**。

---

## 3. 阈值公式

```python
message_budget   = context_window - reserved_completion
pressure_budget  = message_budget - headroom
threshold_tokens = floor(min(context_window * threshold_ratio, pressure_budget))
retain_tokens    = (retain_ratio is None
                    ? floor(message_budget * retain_ratio)
                    : retain_tokens)
```

| 变量 | 默认 | 校验 |
|---|---|---|
| `context_window` W | 无(必填) | 正整数 |
| `reserved_completion` O | 0 | 非负整数 |
| `headroom` | 65536 | 非负整数 |
| `threshold_ratio` | 0.8 | `(0, 1]` |
| `retain_ratio` | 0.16 | `(0, 1]`,且必须 `< 生效的 threshold_ratio` |
| `retain_tokens` | — | 与 `retain_ratio` **互斥**;`>= threshold` 则运行期抛 |

**五道运行期错误**(E1–E5):

| # | 条件 | 含义 |
|---|---|---|
| E1 | `context_window` 不是正整数 | 模型未声明容量 |
| E2 | `reserved_completion` 不是非负整数 | — |
| E3 | `W - O <= 0` | 保留的输出吃掉了全部窗口,没有消息预算 |
| E4 | `W - O - headroom <= 0` | 有消息预算但没有压力预算 |
| E5 | `retain_tokens >= threshold_tokens` | 保留量不小于阈值,压缩无意义 |

**一个有意的降级**:E1–E5 需要 `context_window`,所以**不是加载期失败**,而是可被抑制的运行期错误——按目标只 warn 一次然后继续本轮。理由:配置不全 → 不主动压缩,但**provider 报溢出时仍能救**(那条路径不查容量)。这是「配置缺失不该阻断工作」的体现。

**默认值下的实际数字**(W=200000):`threshold = floor(min(160000, 68928)) = 68928`,`retain = 21514`。

> ⚠️ **注意 headroom 是支配项**:`68928 < 160000`,所以 `threshold_ratio` 完全没生效。上游 README 自己承认这两个比值「没有语料依据」。**我们的实现应把 headroom 默认值调小或设为 0**,否则 `threshold_ratio` 是个摆设——这是我们要偏离上游的地方之一。

---

## 4. 压缩事务时序(锁 → 摘要 → 替换 → 解锁)

```text
1. 校验区间存在 + 双向工具配对平衡
2. 反向扫描:未匹配的 start / 最新 end-seed
3. 孤儿 start 检查(未陈旧则 busy)
4. owner 决定:手动要求无开放轮次;自动要求在开放轮次内
5. append compaction/start          ← 锁,同步,与第 3 步之间不 yield
6.   prepare(计量 + 影子价 + 路由价 + 构造输入)
7.   await 摘要调用                   ← 唯一的 await 长窗口
8.   断言区间稳定(整表或选定跨度)
9.   append compaction/summary
10.  append user/message{replace}    ← 三连,之间不 yield
11.  append compaction/end
12. flush(可选)
13. 失败分类处理
```

**为什么锁是日志标记而不是内存锁**:崩溃后能**从日志检测孤儿**。内存锁在进程死掉后不留痕迹。

**崩溃安全判定**:反向扫描取最新未匹配 `start` 与最新 `end-seed`,只有 `end_seed.seq > start.seq` 才判陈旧。

---

## 5. 裁剪算法

```python
total = sum(len(block.text) for block in blocks if block.type == 'text')
if total <= threshold_chars: return None        # 严格大于才动

removed_start = head_chars                       # 默认 4096
removed_end   = total - tail_chars               # 默认 total - 1024

for block in blocks:
    # 按 code point 切,不劈代理对
    merged = head_part + marker + tail_part
```

**四条要点**:

1. **区间退化为单点**——一次只替换一个 `tool/result` 节点,`shadowed_seqs` 长度必须为 1。
2. **影子价用启发式价**,不用路由价。
3. **循环用原始事件而非新 surface**:候选来自快照,同一次调用里的多条替换互不影响(不会「二次裁剪」)。
4. **非文本块原样保留**且在长度累计中计 0,所以富内容块顺序与数量不变。

**结构性保证**:`chars_after <= threshold && chars_after < total`,否则抛。这两条在加载期校验 `head + marker + tail <= threshold` 的前提下恒真。

---

## 6. 不变量

| # | 不变量 | 违反了会怎样 |
|---|---|---|
| M2.1 | **单一估算器**:所有消费者用同一套启发式;计量器不拥有阈值,后端不拥有计价 | 同一段内容两处算出不同价,阈值判断失真 |
| M2.2 | **阈值上界**:`threshold = floor(min(W×ratio, W−O−headroom))`,且必须 `> 0` | E3/E4 |
| M2.3 | **保留小于阈值**:`retain < threshold` | E5;压缩后仍超阈值 |
| M2.4 | **比值交叉**:用 `retain_ratio` 时必须 `< 生效的 threshold_ratio` | 保留量大于阈值 |
| M2.5 | **摘要必须更小**:`estimate(摘要) < 被遮蔽段的路由价之和`,否则抛、不写替换 | 压缩后反而更大 |
| M2.6 | **三连不 yield**:`summary → replace → end` 之间无 await | 中间崩溃留下半完成事务 |
| M2.7 | **影子价相邻**:每个 replace 必须有紧邻其前、覆盖完全相同区间的影子价事件 | O(1) 压力投影失效 |
| M2.8 | **区间一致性**:计量器表层与 session surface 的 seq 逐个相等,否则拒绝压缩 | 按过期区间压缩,遮蔽错内容 |
| M2.9 | **配对平衡**:压缩区间两端都必须工具配对平衡 | 切断 tool_call/tool_result 配对,下一次请求被模型拒绝 |
| M2.10 | **节点 0 保护**:节点 0 是系统提示时只接受同类型、恰好覆盖它的替换 | 系统提示被吃掉 |
| M2.11 | **`tool/result` 只能改 content**:除 message content 外必须深度相等 | 工具结果被偷换 |
| M2.12 | **裁剪预算自洽**:`head + marker + tail <= threshold`(加载期) | 裁剪后仍超阈值 |
| M2.13 | **裁剪必须更小**:`after <= threshold && after < total` | 无意义的替换 |
| M2.14 | **先落盘再替换**:落盘成功之后才构造新内容;失败保留原内容 | 内容被替换但原文丢失 |
| M2.15 | **图片不可切**:装不下的图片块直接终止该端拟合,剩余预算废弃 | 图片被截成损坏数据 |
| M2.16 | **不透明定位符**:落盘句柄是品牌类型,消费方不得解析,只能靠检索提示 | 消费方依赖句柄内部格式,换实现即崩 |
| M2.17 | **私有目录 + 排他写**:根 0700、文件 0600 且 `O_EXCL`;建议名永不当路径 | 重定向或目录穿越 |
| M2.18 | **清扫只删常规文件**:非 regular 一律跳过,符号链接不跟随 | 顺着链接删掉用户文件 |
| M2.19 | **配置 fail loud**:未知键、互斥字段、非法比值、重复策略一律加载期抛 | 静默用错配置 |
| M2.20 | **溢出恢复需持久进展**:只有 `replace_generation` 前进过才授权重试 | 无进展重试导致死循环 |

---

## 7. 失败分支

| 场景 | 结果 |
|---|---|
| 无路由目标 | 静默不压缩 |
| 容量未声明(E1) | 配置错误,按目标 warn 一次,继续本轮 |
| 压力未达阈值 | 返回 null,连裁剪都不碰 |
| 裁剪后已达标 | 返回 null,**不调模型** |
| 选不出可压缩区间 | 返回 null |
| 摘要重试 N 次后仍超阈值 | 抛「still above threshold after N attempts」 |
| 表层与计量器不一致 | 拒绝压缩 |
| 孤儿 start 未陈旧 | 手动:busy 错误;自动:warn |
| 手动路径有开放轮次 / 自动路径无开放轮次 | 各自的明确错误 |
| 端点不配对平衡 | 拒绝 |
| 准备期表层已变 | `SurfaceChanged` 错误 |
| 摘要空文本 / 含图片 / 被 max-tokens 截断 | 三种明确错误 |
| 摘要不比被遮蔽段小 | 抛,不写替换 |
| 摘要期间整表变化 | `SurfaceChanged` |
| `compaction/end` 也失败 | **保留孤儿 start**(不重试关闭),抛关闭错误 |
| 溢出恢复超重试上限 | 交回下游,不改写原错误 |
| 落盘:未配预算 | 插件完全不生效 |
| 落盘:清扫遇不可信根 / 符号链接 / 任何 fs 失败 | 全部 contain,**清扫永不 reject** |

---

## 8. 我们的选择 vs 上游

| 维度 | 上游 | 我们 | 理由 |
|---|---|---|---|
| 语言 | TypeScript | **Python** | 本人主力栈 |
| 启发式计量 | 4 字符/token + 块税/角色税 | **同构保留** | 「有启发式价可用」是本质;具体密度是细节 |
| `headroom` 默认 | 65536 | **0 或调小** | 上游默认让它成为支配项,`threshold_ratio` 变摆设。这是我们要修的地方 |
| 摘要指令 | 7 个小节名 + Rules | **同构保留**(措辞可改) | 上游 README 承认比值无语料依据,但小节结构是经验产物 |
| 图片处理 | `image/offload` 事件 + 不可切 | **不可切保留,offload 不做** | 第一版图片按普通内容处理 |
| 清扫 | 定时 + mtime 失效 | **第一版不做清扫** | 触发条件:落盘目录超过 1GB |
| 多轮摘要 | 支持 `compactionRetries` | **同构保留** | 一轮摘要常不够 |

**明确不抄的**:

- **`contextBreakdown` 等产品面投影**。第一版不做 UI。
- **缓存备份跳过(`backup-and-skip`)**。我们没有缓存。
- **`~XXXX` 的短名映射与 `session-<sha256 前 12 位>`** 的具体算法。需要一个不可预测名,但算法是细节。

---

## 9. 依据(上游证据索引)

| 结论 | 证据 |
|---|---|
| 阈值公式与每个变量 | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §3.2 |
| 压缩事务精确时序(13 步) | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §3.3 |
| 裁剪算法与「选谁/砍成什么」 | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §3.4 |
| 落盘全流程 | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §3.5 |
| 影子价协议与两个失效边界 | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §3.6 |
| 不变量 I1–I20 | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §4 |
| 失败分支全枚举 | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §5 |
| 「必须保住 vs 可以重写 vs 反模式」 | [64 章](../../../../sources/dsh-study/kb/chapters/64-compaction-spill-deep.md) §6 |
| compaction 事务时序图 | [图册 03](../../../../sources/dsh-study/kb/diagrams/03-session-surface.md) §3 |
| 上游源码位置 | `packages/compaction/`、`packages/spill/`、`packages/llm/token-meter/src/` |

---

## 10. 未解决 / 待验证

1. **【未验证】headroom 设为 0 后的行为。** 上游默认值让 `threshold_ratio` 失效。若我们设为 0,`threshold = min(W×0.8, W−O)`,在 W=200000、O=8192 时为 160000——**远大于上游的 68928**,压缩会更晚触发。这是否会导致 provider 报溢出?验证方法:M2 实现后跑长会话,统计溢出恢复的触发频率。
2. ~~**【未验证】裁剪标记在所有合法配置下恰好插入一次。**~~ **已做属性测试（2026-10-16）**：穷举 `head_chars ∈ {0,1,2,5,10,50,100,500,1000,4000}` × `tail_chars ∈ {0,1,10,100,1000}`，**实际跑过 49 种合法组合**，标记**恰好出现一次**，无异常。

   我**特意确认了它非空转**——一个跳过所有用例的属性测试会因为错误的理由通过。
3. ~~**【未验证】自动压缩真的会在开放轮次内触发。**~~ **已验证（2026-10-16）**：构造超阈值会话（12 轮预置历史，measured 18230 > threshold 16000），**自动路径确实触发**，`manual: False`，且**被单一轮次包住**。

   **补充一处与契约表述的差异**：契约说「确认 `compaction/start` 的 `turn` 是数字」。该事件**没有 `turn` 字段**——它携带 `start_seq`/`end_seq`/`shadowed`/`manual`，描述**操作本身**。轮次号记在 **`request/context`** 上，那个事件描述的是**实际发出去的内容**，因而也是「压缩产生了什么」的正确归属处。

   已加三条测试，并**撤掉自动路径验证它们会失败**。
4. **【推断】两套价在 Python 下的实现成本。** 上游用「启发式可加」这个性质换 O(1);Python 若用真实 tokenizer 会失去可加性。验证方法:M2 实现时先用启发式,记录换成真 tokenizer 的代价。
5. ~~**【未验证】`retain < threshold` 与 M2.3 的交互在 ratio 模式下。**~~ **已查并修复（2026-10-16）**：E5 **在加载期抛**，但**只对 `retain_ratio` 生效**——`retain_tokens` 这条路径**完全没检查**。

   **实测（修复前）**：`retain_tokens=200000` 对阈值 160000 **被接受**，而这个配置**永远压不到阈值以下**——压缩保留的比移除的还多，正是 E5 要拒绝的情形。

   两个写法是同一个设置的两种拼法（同时给会被判为歧义而拒绝），**只检查其中一个**意味着这条不变量成立与否**取决于调用方用哪种写法**。已补上，边界是 `>=`（恰好等于阈值 = 什么也没移除）。
6. **【未验证】不落盘时的降级行为。** 第一版不做清扫,落盘文件会持续累积。需要一条明确的容量上限策略,否则长期运行会填满磁盘。
