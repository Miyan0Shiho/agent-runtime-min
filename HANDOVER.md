# 交接摘要

> 写于 2026-10-19 · `agent-runtime-min` 收尾时
> 面向"接着做的人"——包括几天后的我自己。

---

## 一句话

**一个从零写的 Python agent 运行时,已经能真的用**:接真实 provider 跑任务、
会话落盘与跨进程恢复、上下文压力自动压缩、子 agent / 团队 / 后台作业 / 计划模式
全部可达可跑。**约百个提交,87 条真实缺陷,五类检查全绿。**

---

## 1. 它是什么,不是什么

| 是 | 不是 |
|---|---|
| 一个能自用的 harness(DSH 那种通用骨架) | 简历项目 |
| clean-room,Python 从零写 | DSH 的移植或依赖 |
| 设计对齐 DSH | 功能对齐 DSH |

决策见 `docs/decisions/`(ADR-0001 范围 → 0002 设计对齐 → 0003 可自用 + 全做)。

---

## 2. 怎么跑

```bash
cd 02-projects/agent-runtime-min
make install                      # 建 venv

# 一个任务
export ARM_API_KEY=... ARM_MODEL=deepseek-chat ARM_BASE_URL=https://api.deepseek.com
arm run "在这个仓库里找到失败的测试并修好"

arm chat                          # 交互
arm replay --session default      # 从日志重建模型看到的每个字节
arm sessions                      # 列会话
```

**要记住的开关**(其余见 `grep -rhoE "ARM_[A-Z_]+" src/`):

| 开关 | 作用 | 默认 |
|---|---|---|
| `ARM_API_KEY` / `ARM_MODEL` / `ARM_BASE_URL` | provider | 无 |
| `ARM_APPROVAL` | `ask`(终端询问)/ `never` | `ask` |
| `ARM_SANDBOX` | `workspace-write` / `read-only` / `none` | `workspace-write` |
| `ARM_CONTEXT_WINDOW` | 上下文窗口,决定压缩阈值 | `200000` |
| `ARM_TEAM` | 团队开关 | **开**(设 `0` 关) |
| `ARM_DOCKER_SANDBOX` | bash 跑在容器里 | 关 |
| `ARM_BROWSER` | CDP 端点 | 关 |
| `ARM_COMPUTER` | `observe`(只看)/ `1`(可操作) | 关 |

---

## 3. 五个检查,各回答一个问题

```bash
make check      # 全部五项,一条命令,约 3 分钟
```

| 命令 | 问的问题 | 什么时候红 |
|---|---|---|
| `make test` | 每个零件对不对 | 单元行为变了 |
| `make smoke` | **装起来之后能不能用** | 零件都在,没接上 |
| `make deadcode` | 有没有代码没人调用 | 新增未调用定义 / 只被测试引用 |
| `make bench` | 验收指标达没达到 | M1–M4 阈值破了 |
| `make claims` | 设计文档的断言成立吗 | 实测推翻文档假设 |

**`make smoke` 和 `make bench` 进 `check` 都是用真实代价换来的** —— 见 §5。

`make claims` **刻意不在** `check` 里:它测的是设计假设,随设计变而不随提交变。

> **本文档的数字由 `make smoke` 核对**(「the handover states true counts」)。
> 理由和 README 一样:一个没人核对的说法,几轮之后就是错的 —— 而这份摘要的全部价值就在于它说的话是真的。

---

## 4. 当前状态(可核对)

```
1257 passed, 12 skipped        make test
19 smoke checks passed         make smoke
35 uncalled, 0 not in baseline make deadcode
M1 replay 100% · M2 79.068% / 0.0pp · M3 100% · M4 100%/100%   make bench
```

| 项 | 值 |
|---|---|
| 实现 | 28,827 行 / 83 模块 |
| 测试 | 23,276 行 / 29 个文件 / 1,257 条 |
| 提交 | 约 100(不精确 —— 写入它的那次提交本身会改变这个数,所以它不可校验) |
| 真实缺陷 | **87 条**(编号到 91,4 个跳号) |
| 仓库 | 39 M(含 .git 6.2 M) |

### 收尾判据(三条,都满足)

1. **REQUIREMENTS 每一项都可达** —— 30 个工具、10 个能力组、7 个子系统
2. **没有模块只被自己的测试引用** —— 只剩 `bench/claims.py`(`make claims` 的入口)
3. **设计文档未决项全有结论** —— 只剩 M1.5(见 §7)

---

## 5. 最重要的一件事:**测试全绿 ≠ 工具能用**

收尾后回头看,这个项目最大的教训只有一条,而它**推动了后面十轮的发现**:

> **我停止读文档、开始用真实 provider 真跑之后,12 个功能里有 5 个当场坏掉
> (subagent / plan / hooks / plugins / computer)—— 而它们的单元测试全部是绿的。**
>
> 另外 7 个是干净的:team、goal、todo、skill、schedule、worktree、browser。
> **「真跑」不保证发现缺陷,但它是唯一能发现的途径** —— 而 5/12 这个比例已经说明问题。

原因是**单元测试自己搭零件**,所以它看不见 `cli/main.py` 里怎么把它们装起来:

| 缺陷 | 零件 | 错在 |
|---|---|---|
| #78 | 内部词汇表 | `developer` 角色被 provider 拒 → **工具完全不可用** |
| #87 | 传输层、客户端 | CLI 把两个事件循环接错 → **每个 MCP 调用都失败** |
| #88 | M2 实现完整、bench 达标 | **CLI 从不构造 ContextManager** → 真实会话从不压缩 |
| #89 | 925 行、有测试 | **没有任何调用者** |
| #81/#82 | 模式实现 | 进得去出不来 |
| #83 | hook 执行 | 日志说不出谁拦的 |

**`make smoke` 和 `make deadcode` 的「只被测试引用」检查,是把这件事固化下来。**

完整的 18 条方法论教训见 **`docs/lessons.md`** —— 那份文档比这份摘要值得读。

---

## 6. 目录导航

```
REQUIREMENTS.md           ← 实施入口(自包含)
README.md                 ← 能跑什么 + 五个检查
docs/
  lessons.md              ← 方法论教训(18 组,按错误形状分类)
  contract-amendments.md  ← 87 条缺陷;#0 是权威编号表
  runtime-spine.md        ← 主链路 + S1–S14 不变量
  contracts/              ← M1–M4 四份机制契约(64 条不变量)
  decisions/              ← ADR-0001/0002/0003
  t3-ordering-and-guardrails.md
  market-native-builtins.md  ← 市场调研(结论有价值,数据已过期)
spikes/                   ← 前置验证
src/arm/
  kernel/ session/ llm/ prompt/ tools/ context/ eval/   ← 内核 M1–M4
  skills/ subagent/ team/ workspace/ jobs/ plan/ goal/ todo/ schedule/  ← T3
  mcp/ hooks/ plugins/ browser/ computer/ sandbox/      ← 扩展与安全
  cli/ bench/                                           ← 入口与检查
```

---

## 7. 唯一留着的一条

**M1.5:`ignorable` 默认「必需」是否过严。**

- **机制已实测**:日志里出现未知事件类型且未标 `ignorable:true` 时,读取**拒绝整条日志**。
- **为什么不能结**:契约写的验证方法是「**按实际痛感复核**」—— 痛感只在**真实升级后打开旧会话**时出现。
- **现在没有旧数据,没发生过跨版本升级**,所以**没有一个诚实的实测能替代这个判断**。
- **复核触发条件**:第一次出现「打开旧会话被拒」时,记录情形。若被拒的是无关紧要的新事件类型且可安全忽略,就把那种类型标成 `ignorable:true` 的**产出方**(而不是放宽默认值)。

---

## 8. 已知限制(不是缺陷,是接受了的代价)

| 限制 | 为什么接受 |
|---|---|
| **后台作业随进程结束** | CLI 一次调用一个进程;作业的粒度是会话内的多个步骤,不是跨 `arm run` |
| **Docker 沙箱需手动启用**,VM 无 autostart | 重启后需 `limactl start docker` |
| **浏览器附加到你自己的 Chrome** | 是设计(无头过不了 Cloudflare);`ARM_BROWSER` 让它成为刻意行为 |
| **电脑操作默认关闭** | 桌面不是沙箱;`ARM_COMPUTER=observe` 可只观察 |
| **`skills` 从运行目录解析** | 技能属于项目;从项目目录运行才会有项目级技能 |
| **环境无模型 key 时 12 条测试 skip** | 真模型契约只有真 provider 能验,skip 比 mock 诚实 |

---

## 9. 如果接着做,我建议的顺序

1. **先跑一遍 `make check`** —— 确认基线是绿的,再动手。
2. **改任何跨边界的东西后,跑 `make bench`** —— 它有两次抓到别人没抓到的回归。
3. **加功能时先问:它从哪个入口可达?** 这个项目最大的缺陷类是「写好了但没人调用」,而且**测试不会告诉你**。
4. **改动比较逻辑时,搜所有消费者** —— 两次缺陷都是「修了一个调用者,漏了另一个」。

---

## 10. 一句话总结

**这个项目的价值不在 28,000 行代码,而在那份 87 条缺陷的记录:它证明了一个测试全绿的系统可以几乎完全不能用,而唯一可靠的发现方式是——真的用它。**
