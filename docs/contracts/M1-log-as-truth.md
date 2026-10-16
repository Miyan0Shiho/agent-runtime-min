# M1 契约:日志即事实(Log-as-Truth)

> 状态:设计产物 · 2026-10-12 · 作者:本人
> 上游依据:[61-session-persistence-deep](../../../../sources/dsh-study/kb/chapters/61-session-persistence-deep.md)、[24-session-and-persistence](../../../../sources/dsh-study/kb/chapters/24-session-and-persistence.md)、[图册 03](../../../../sources/dsh-study/kb/diagrams/03-session-surface.md)
> 前置:[runtime-spine](../runtime-spine.md) §4 的不变量 S1–S14
> **本文抽掉上游名字后必须能独立成立**(见 [ADR-0002](../decisions/0002-design-alignment-and-native-builtins.md) §5)。

---

## 0. 一句话结论

日志是**唯一事实来源**,上下文是**它的派生投影**。派生的正确性判据是:给定日志,能重建出当时发给模型的每一个字节。

为了让「派生」可增量,必须把「追加」与「改写」拆成两种操作:**追加进日志,改写靠遮蔽**。遮蔽不删原事件,只让它们不进派生结果——这是「压缩后可回放、可审计」的全部机制。

---

## 1. 数据结构

```python
@dataclass(frozen=True)
class Envelope:
    seq: int                 # = len(log),定义即连续,不校验
    type: str                # 事件名,如 "tool/result"
    data: dict               # 事件体,深度冻结
    ts: str                  # ISO8601
    format_version: int      # 写方版本
    ignorable: bool = False  # 未知类型时是否可跳过

class Session:
    log: list[Envelope]           # 只追加
    surface: Surface              # 派生:有序节点
    _derived: list[Message]       # 派生缓存
    _derived_nodes: int           # 水位线
    _derived_generation: int      # 失效键
```

**`surface` 不是另一份数据**,它是从 `log` 折叠出来的节点序列。每个节点引用一个 `seq`。

```python
class Surface:
    nodes: list[int]              # seq 的有序列表
    content_generation: int       # append 不变,replace/投影才 +1
    replace_generation: int       # 只有 replace 才 +1
```

**为什么两个 generation**:`content_generation` 管「派生缓存是否要重建」,`replace_generation` 管「压缩是否真的削减了内容」。用错会让热路径每次 append 都重建整表。

---

## 2. 三种 surfaceOp

| 形态 | 合法性 | 语义 |
|---|---|---|
| `"append"` | surface 类事件必须显式写出 | 追加到 surface 尾部 |
| `{"op":"replace","start_seq":a,"end_seq":b}` | 恰好 3 个键;两值都是合法 seq;都 `< 本事件 seq` | 用本节点替换当前 surface 上 `[a,b]` **含端点**的区间 |
| 非 surface 事件携带 | **禁止** | — |

**替换的边界规则**(每一条都来自上游踩过的坑):

1. 端点用 `nodes.index(seq)` 定位**当前 surface 的位置**,不是日志位置。
2. `a > b` 非法;端点不在当前 surface 非法。
3. **新节点插在被遮蔽区间的起始位置**,不是尾部。
4. `source_event_seqs` 必须**完整覆盖**被遮蔽的节点;不得为空、不得重复、不得引用 `≥ 自身 seq`。
5. `assistant/message` **永远不能**带 `source_event_seqs`(它内嵌流)。

---

## 3. 派生算法(完整)

```python
def derive(self) -> list[Message]:
    nodes = self.surface.nodes
    gen = self.surface.content_generation
    if gen != self._derived_generation:
        self._derived = []
        self._derived_nodes = 0
        self._derived_generation = gen
    for seq in nodes[self._derived_nodes:]:
        msg = self._derive_one(self.log[seq])
        if msg is not None:
            self._derived.append(msg)
    self._derived_nodes = len(nodes)
    return list(self._derived)          # 新数组,共享冻结消息
```

**三个必须保住的性质**:

1. **失效键是 `content_generation`**。append 不 bump,所以尾部追加走 `nodes[watermark:]` 增量;replace 与投影才 bump,整表重建。用 `len(log)` 或 `replace_generation` 都会错。
2. **投影优先**:先查插件投影结果(按 seq);否则 `user/message` 返回其 `data`;`system`/`developer`/`assistant` 在 `content` 为空时返回 `None`(**占位不上 wire**);`tool/result` 返回其 `message`;其余一律 `None`。
3. **返回新数组、共享冻结消息**。调用方不能通过数组修改历史。

---

## 4. 不变量

| # | 不变量 | 违反了会怎样 |
|---|---|---|
| M1.1 | **`seq = len(log)`**:连续性不是校验出来的,是**定义**出来的;种子也必须从 0 连续 | 出现空洞,恢复时无法区分「缺失」与「截断」 |
| M1.2 | **先校验、后提交**:所有校验在 `log.append` 之前;失败不改 surface、不改 log | 部分写入,内存与磁盘发散 |
| M1.3 | **append 不可重入**:提交期间再次 append 直接抛错 | 监听器里写日志会造成无限递归 |
| M1.4 | **提交点只有一处**:监听器快照在提交前解析、提交后调用;监听器失败只记日志,不影响提交 | 多提交点必然产生顺序不一致 |
| M1.5 | **坏事件到不了磁盘**:append 阶段强制 JSON 无损性 | 磁盘格式与内存不同构,回放与迁移无法共用真源 |
| M1.6 | **surface 事件必须带 `surfaceOp`;非 surface 事件禁止带** | 改写变成删除,可审计性失效 |
| M1.7 | **replace 区间必须存在且有序**;新节点插在原位置 | 遮蔽区间错位,派生结果与预期不符 |
| M1.8 | **`source_event_seqs` 必须完整覆盖被遮蔽节点** | 有事件被静默丢弃 |
| M1.9 | **`tool/result` 替换只能改 `content`**(其余字段深度相等) | 工具结果被偷换成别的调用 |
| M1.10 | **节点 0 是系统提示时,只能被「恰好覆盖它的同类型替换」重写** | 系统提示被区间操作意外吃掉 |
| M1.11 | **未命中定义的投影事件必须拒绝**,不得静默丢弃 | 缺插件时静默丢内容 |
| M1.12 | **`derive` 的失效键是 `content_generation`** | 热路径全表重建,或替换后不生效 |
| M1.13 | **空内容的 system/developer/assistant 不进 wire**,但**保留 surface 位置** | 消息序列与位置语义分离,工具配对错位 |
| M1.14 | **append 尽力而为,`flush` 才是持久屏障** | 调用方误以为 append 返回即可抗崩溃 |
| M1.15 | **撕裂尾部永不返回给读者**;物理写入前先截断,再由写路径重写恢复记录 | 半个记录被当成新事件 |
| M1.16 | **未知事件类型默认必需**:没有 `ignorable:true` 就拒绝整条日志 | 静默恢复出残缺会话(**安全事故**,不是体验问题) |
| M1.17 | **追加失败必须回滚到写入前的 size** | 旧游标重试会产生重复 seq |

---

## 5. 写路径三步序(不可换)

```text
append(batch):
  ① 截掉撕裂尾(truncateTornTail)
  ② durably 重写恢复记录(recoveredTail)
  ③ 写新批次
  ── 任一步失败 → truncate(before) 回滚
                  否则旧游标会重试整批,产生重复 seq
```

**为什么必须先截断再写**:撕裂的尾部是「结构不完整的最后一帧」。若直接追加,它会与新批次拼成一个**看起来合法**的记录,把半个记录变成真事件。

---

## 6. 撕裂尾部处理三分法

| 情况 | 处理 |
|---|---|
| 完整帧但校验失败 | **算 corruption,拒绝** |
| 结构不完整的最后一帧 | **可恢复**,重写后再追加 |
| 裸文本、无换行尾 | **丢弃** |

**判据是「结构扫描」与「记录解码」两件事**:截断点来自结构扫描(帧起点),恢复集合来自记录解码(能解出完整行)。两者混用会把半个记录写成新事件。

---

## 7. 失败分支矩阵

| 场景 | 结果 |
|---|---|
| 非 JSON 数据 | 抛错,日志不变 |
| 重入 append | 抛「cannot reenter」 |
| 非法 replace(端点不存在 / 顺序反 / 未覆盖遮蔽节点) | plan 阶段抛错,surface 不变 |
| 缺投影定义 | 抛「requires a message projection」 |
| 定义被卸载后读 | 抛「was removed or replaced」 |
| 种子不连续 | 抛「seed must be contiguous from 0」 |
| 未知事件类型 | 无 `ignorable` → 拒绝整条日志 |
| 未来格式版本 | 拒绝,提示升级 |
| 追加部分写 | 关闭 + 截断回 before + 重抛 |
| 回滚也失败 | 聚合错误(两个都上报) |
| 撕裂帧 | 记录截断点,恢复已 flush 的完整记录 |
| 完整帧损坏 | 拒绝 |

---

## 8. 我们的选择 vs 上游

| 维度 | 上游 | 我们 | 理由 |
|---|---|---|---|
| 语言 | TypeScript | **Python** | 本人主力栈 |
| 容器 | zstd 分帧 + 200ms write-behind | **裸 JSONL** | 可读性优先,`cat` 即可调试;写屏障策略可后换 |
| 格式版本 | 4 代际 + 相邻迁移链 | **1 个版本,无迁移链** | 没有已发布数据,迁移链是「有用户之后」的成本 |
| 投影缓存 | 独立 domain + checkpoint 行 | **不做**,每次冷读重折 | 第一版日志量小;触发条件见 §10 |
| 事件数 | 59 | **19** | 见 [runtime-spine](../runtime-spine.md) §3 |
| 图片卸载 | `image/offload` 事件 | **不做** | 第一版图片按普通内容处理 |

**明确不抄的**(上游自己的决策文档也这么说):

- **四分真源**(写入器常量 / 定稿记录 / 发布记录 / schema 快照)。我们没有外部消费者,不需要这套发布纪律。
- **相邻迁移链**。理由同上。
- **`session_projcache` 的 `backup-and-skip`**。第一版不做缓存,就没有坏缓存要处理。

---

## 9. 依据(上游证据索引)

| 结论 | 证据 |
|---|---|
| 32 条持久化不变量 J1–J32 | [61 章](../../../../sources/dsh-study/kb/chapters/61-session-persistence-deep.md) §5.1 |
| `deriveMessages()` 完整算法与三个性质 | [61 章](../../../../sources/dsh-study/kb/chapters/61-session-persistence-deep.md) §4.2 |
| `surfaceOp` 三形态与边界规则 | [61 章](../../../../sources/dsh-study/kb/chapters/61-session-persistence-deep.md) §4.3 |
| 失败分支矩阵 | [61 章](../../../../sources/dsh-study/kb/chapters/61-session-persistence-deep.md) §5.2 |
| 「最小实现必须保住的不变量」十条 | [61 章](../../../../sources/dsh-study/kb/chapters/61-session-persistence-deep.md) §6.1 |
| 「最容易写错的三处」 | [61 章](../../../../sources/dsh-study/kb/chapters/61-session-persistence-deep.md) §6.3 |
| 写路径三步序 | [图册 03](../../../../sources/dsh-study/kb/diagrams/03-session-surface.md) §5 |
| 上游源码位置 | `packages/core/session/src/index.ts`、`packages/core/session/src/surface.ts`、`packages/session/session-persistence-jsonl/src/` |

---

## 10. 未解决 / 待验证

1. ~~**【未验证】19 个事件是否够用。**~~ **已实测（2026-10-16，`make claims`）**：**够用，而且定义数已超出当初的 19**。

   - 实际定义 **28 个事件类型**（契约第 3 节原本写「19 个」）；
   - **一个回合的每个阶段都有对应事件类型** —— 实测 11 个阶段，全部有覆盖；
   - **没有单类事件主导长会话**：最大的一类是 `step/start`，占 1400 个事件里的 **14.3%**（阈值 35%）。

   最后一条是「够用」的真正检验：若某一类事件吃掉大半日志，说明模型把多件事挤进了一个事件，那才是需要扩事件表的信号。
2. ~~**【未验证】裸 JSONL 在数万事件下的 `flush` 开销。**~~ **已压测（2026-10-16，`make claims`）**：**裸 JSONL 不需要 write-behind 窗口**。

   | 测量 | 结果 |
   |---|---|
   | 一次性追加 100,000 事件 | **1.677 s**（0.0168 ms/事件，56.4 MiB） |
   | **实时逐条追加**（2,000） | **0.105 ms/事件** —— 这是真实会话的路径 |
   | 扫描式逐条追加（500） | 4.620 ms/事件（`already_contiguous=False`，每次重读文件） |
   | 一次持久屏障（fsync） | **0.035 ms** |
   | 恢复 2,000 事件会话 | 40.8 ms（线性，0.0204 ms/事件） |

   上游用 200ms write-behind 是为了摊薄 flush 成本。**实测 fsync 只要 0.035 ms**，比上游的窗口所规避的代价小得多——所以「可读性优先，`cat` 即可调试」这个取舍**没有付出想象中的代价**。

   > 顺带发现的一处**我自己的测量错误**：最初报的 19 ms/事件取自**扫描式**路径（`already_contiguous=False`，每次重读整个文件），而**实时路径是 0.105 ms** —— 相差约 180 倍。用一个不是生产路径的测量去评价生产路径，会得出完全错误的结论。
3. ~~**【推断】不做投影缓存会不会成为瓶颈。**~~ **已实测（2026-10-16）**，结论：**暂不引入缓存**，理由是数字而不是偏好。

   契约的触发条件是「**超过 1 万事件** 且 **冷读 > 200ms**」。实测：

   | 事件数 | 冷读 |
   |---|---|
   | 5,000 | 95 ms |
   | **10,000** | **186 ms —— 在 200ms 线下** |
   | 20,000 | 384 ms（已触发） |
   | 50,000 | 992 ms |

   三个事实决定了这个取舍：

   1. **成本是线性的（约 0.02 ms/事件）**，不是平方的——所以它不会突然失控；
   2. 按实测的 **约 8 事件/工具调用**，达到 20,000 事件需要 **约 2,400 次工具调用**。作为参照，一个真实的长会话（本次 DSH 会话）是 1,617 步；
   3. **冷读发生在「打开会话」时，不是每轮一次**。所以代价是**恢复会话时的**，不是持续的。

   > 换句话说：**契约自己设的阈值在契约自己指定的规模上还没被触及**，而它要消除的代价是一次性的。

   已加一条测试固定这个测量（线性增长 + 1 万事件的上界），这样**换台慢机器、或某个改动让日志变大**时，它会**显眼地失败**，而不是悄悄让恢复变得难受。
4. **【未验证】Python 的 `dataclass(frozen=True)` 能否替代上游的 `deepFreeze` 语义。** Python 的冻结是浅层的;嵌套 dict 仍需手动处理。验证方法:写一条测试,尝试从 `derive()` 的返回值修改嵌套字段,必须失败。
5. **【未验证】`ignorable` 的默认值在自用场景下是否过严。** 上游选「默认拒绝」的理由是安全性不对称(过度拒绝是体验问题,静默恢复残缺会话是安全事故)。自用场景下我们是否接受这个取舍,需要在 M1 实现后按实际痛感复核。
6. ~~**【未验证】M1.13(空消息占位)与 M1.8 的交互。**~~ **已查（2026-10-16）**：实现**不采用占位机制**——空响应记 `assistant/attempt`，而它**不是 surface 事件**，所以不保留 surface 位置。

   **已验证它要保护的结果是完好的**：派生序列保持 `system, user, user, assistant`，**没有错位**，工具配对不受影响；且空节点不会作为无内容 assistant 消息送到 provider（provider 会拒绝）。差别在**机制**，不在**行为**——已加三条测试固定行为，并把「实现与契约的机制不同」这件事**写进测试 docstring 而不是含糊过去**。

   因此 M1.8 的交互问题**不存在**：既然空消息不进 surface，replace 区间就不可能覆盖它，`source_event_seqs` 也就没有「是否要包含它」的问题。
