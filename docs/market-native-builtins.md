# DSH 插件市场实测 + 「原生内置」候选排序

> 状态:设计期调研产物 · 采集日期 2026-10-12
> 性质:本文是**市场证据 + 排序建议**,不是已接受的决策。决策见 [ADR-0002](decisions/0002-design-alignment-and-native-builtins.md)。
> 证据分级:【npm】= npm registry API 实测;【GitHub】= GitHub API 实测;【制品】= 本机 profile 实测;【源码】= `sources/deepseek-harness/` 源码;【推断】= 作者推理。

## 0. 一句话结论

DSH 的「插件市场」不是一个独立产品,而是 **npm 本身** —— 插件的分发机制就是 `dsh plugin add <npm 包名>`,注册表是 `registry.npmjs.org` 或 `registry.npmmirror.com`(`packages/boot/plugin-manager/src/registry.ts:14,17`)【源码】。截至采集日,社区维护的 `awesome-dsh-plugin` 清单收录 **4,508 个插件 / 23 个分类**,其中一个社区市场的月下载量达 **57 万**【npm】。

**但这个市场暴露的关键事实,与「DSH 靠插件实现某功能」的直觉相反:**

- 有一批能力是 **DSH 官方做成可选插件**的(skill / goal / plan / subagent / computer-use…)——内建它们属于**抄近路**。
- 另有一批能力 **DSH 根本没有,是社区在填空白**(记忆系统 / 上下文洞察 / 工作树 / Docker 沙箱 / Eval)——内建它们才是**原创空间**。

**排序结论(详见 §5):真正值得内建的第一名不是记忆或电脑操作,而是 `Eval 基础设施`** —— 它是 T1 岗位 JD 里唯一被写成**完整一条职责**、却在 DSH 源码中**完全不存在**、且市场 **4,508 个插件里只有 20 个**碰的能力。

---

## 1. 市场结构(实测)

### 1.1 规模与分发

| 项 | 值 | 证据 |
|---|---|---|
| 精选清单收录插件数 | **4,508** | 【GitHub】`awesome-dsh-plugin/count.json` |
| 分类数 | **23** | 【GitHub】清单 TOC |
| 分发方式 | `dsh plugin add` → pnpm → npm registry | 【源码】`packages/boot/plugin-manager/src/registry.ts` |
| 官方市场机制 | **不存在**。DSH 只有 npm 源选择器(官方源 / npmmirror / 自定义) | 【源码】`apps/web/tests/expected/plugin-install-registry/*.expected.md` |
| 清单维护者声明 | 「本清单**不排名、不评判质量**」 | 【GitHub】`awesome-dsh-plugin/README.md` |

> ⚠️ **对「插件市场很火」这一直觉的修正**:火的是**生态**,不是某个官方市场。DSH 官方只提供了「从 npm 装包」这一个原语,市场是被社区(第三方 `dshmarket` 等)补出来的。

### 1.2 分类分布(条目数)

| 分类 | 条目 | 分类 | 条目 |
|---|---:|---|---:|
| UI Enhancements | 786 | Themes & Appearance | 160 |
| Tools & Capabilities | 585 | Security & Permissions | 151 |
| Development & Runtime | 323 | Remote & Mobile | 139 |
| Sessions & Messages | 298 | Just for Fun | 137 |
| Workflow & Automation | 267 | Git & Code Review | 111 |
| Usage & Billing | 236 | Vision & Multimodal | 109 |
| Models & Providers | 215 | Browser & Web | 100 |
| **Memory** | **209** | Plugin Markets & Managers | 80 |
| Skills | 178 | WSL & Windows Interop | 77 |
| Notifications & Integrations | 173 | Docs & Rendering | 68 |
| | | Voice & Audio | 65 |

【GitHub】按 `###` 分节解析 `awesome-dsh-plugin/README.md`(`<!-- BEGIN PLUGINS -->` … `<!-- END PLUGINS -->`)。

### 1.3 按下载量排名(前 15,社区插件,月下载)

| # | 插件 | 月下载 | 一句话 |
|---|---|---:|---|
| 1 | `dshmarket` | 570,846 | DSH 内置的可视化插件市场 |
| 2 | `@michengai/dsh-skills-manager` | 202,149 | 统一加载并管理本机 Agent Skills |
| 3 | `@michengai/dsh-archive-manager` | 182,870 | 会话归档/恢复/收藏/删除 + 损坏日志诊断 |
| 4 | `dsh-context` | 147,041 | **上下文洞察与管理** |
| 5 | `dsh-plugin-model-proxy` | 131,719 | 按模型配置 http/socks5 代理 |
| 6 | `@michengai/dsh-automation` | 122,527 | 独立 Session 里按计划执行编码任务 |
| 7 | `dsh-cost-meter` | 108,645 | 会话费用统计与官方价格同步 |
| 8 | `@xmanrui/dsh-im` | 62,286 | 微信/飞书/企微/钉钉/Slack 等接入 |
| 9 | `@nanmicoder/dsh-agent-teams` | 55,140 | 多 Agent 团队协作 |
| 10 | `@dfy-plugins/dsh-image-generation` | 52,343 | 图像生成与编辑 |
| 11 | `dsh-univer-office` | 50,677 | 电子表格协作 |
| 12 | `@deepseek-harness-tui/dsh-tui` | 47,099 | 交互式终端界面 |
| 13 | `dsh-mnemon` | 45,540 | **可组合的视图式记忆系统** |
| 14 | `dsh-plugin-wallpaper-engine` | 43,299 | Web GUI 壁纸 |
| 15 | `dsh-free-search` | 42,419 | 免费网页搜索(24 引擎) |

【npm】`registry.npmjs.org` 搜索 API + `api.npmjs.org/downloads/point/last-month`。

> **读法警告**:npm 下载量会被 CI、镜像同步和目录包灌水(例如 `dsh-plugin-model-proxy` 周下载仅 542,与月下载 131,719 严重不符,说明是**批量拉取**而非真实使用)。**下载量只作趋势参考,不作为质量或热度排序的唯一依据。** 相比之下 `dsh-context`(周 38,580)、`dsh-mnemon`(周 11,541)、`@xmanrui/dsh-im`(周 13,752)的周/月比更健康。

### 1.4 本机已装插件(运行时实测)

本机 desktop profile 装了 **10 个第三方 bundle**:`dsh-hongqiye-image-gen`、`dsh-pet`、`@xmanrui/dsh-im`、`dsh-mnemon`、`dsh-office-tools`、`dsh-rewind-plugin`、`@anionex/dsh-computer-use`、`@anweat/dsh-browser`、`@banbolee/dsh-lsp-diagnostics`、`dsh-context`。【制品】`~/.dsh/profiles/desktop/package.json`,详见 [desktop-profile](../../../sources/dsh-study/kb/data/desktop-profile.md)。

**注意这 10 个里 7 个属于「DSH 没有、社区在填」那一类**(im / mnemon / rewind / context / image-gen / office / computer-use-provider)——本机 profile 本身就是 §3 那个结论的一份样本。

---

## 2. 关键区分:官方插件 vs 社区填坑

这是本次调研最有价值的发现。你原话是「原本 DSH 靠插件实现的功能」——实测下来**这句话只对一半**,必须拆成两类:

### 类别 A:DSH **官方**做成可选插件的能力

DSH 有对应的官方包,只是默认不挂在 profile 里。内建它们 = **把官方插件做的事重做一遍**。

| 能力 | DSH 官方归属 | 规模 |
|---|---|---|
| 持久 Goal | `packages/goal`(4 包) | 3,238 LOC |
| Plan mode | `packages/plan`(1 包) | 727 LOC |
| Todo | `packages/todo/tool-todo` | 355 LOC |
| 后台 Job | `packages/jobs`(3 包) | 3,076 LOC |
| 定时提醒 | `packages/schedule`(2 包) | 5,170 LOC |
| 工作流 / PTC | `packages/workflow` + `ptc-runtime`(6 包) | 6,831 LOC |
| 子 Agent / Agent Teams | `packages/subagent` + `experimental/agent-team` | 17,588 LOC |
| Skill 系统 | `packages/skill`(6 包) | 3,564 LOC |
| 电脑操作 | `packages/computer-use` + 4 个 experimental provider | — |
| 浏览器操作 | `packages/browser-use` + 3 个 experimental provider | — |
| 上下文压缩 / spill | `packages/compaction` + `packages/spill` | 5,362 LOC |
| 沙箱 + 审批 | `packages/sandbox` + `packages/interaction` | 9,456 LOC |
| MCP | `packages/mcp`(2 包) | 2,551 LOC |

【源码】按目录统计 `*.ts`(排除 `tests/`、`*.spec.ts`、`lib/`)。

### 类别 B:DSH **完全没有**,社区在填的空白

DSH 源码里**既没有官方包,也没有 `ctx` 服务或持久事件**。这些才是原创空间。

| 能力 | DSH 实测结论 | 社区规模 |
|---|---|---|
| **记忆系统** | 无 `packages/memory`;`ctx.<service>` 全表中**无 memory/recall**;59 个持久会话事件中**无 memory** | 209 个插件 |
| **上下文洞察** | 无对应包;`dsh-context` 是社区插件(本机已装) | — |
| **工作树 / worktree** | **无原生支持**。全仓 `worktree` 仅出现在:测试 fixture、`workflow-ptc` 的 `isolation:'worktree'` 选项、Windows ACL 注释 | `alpacachen/dsh-worktree` 等 |
| **Docker / 容器沙箱** | **无**。`sandbox-local` 的后端是 bwrap / Landlock / Seatbelt / Windows ACL restricted token,**不含 Docker** | — |
| **Eval 基础设施** | **无 `packages/eval`**;无 LLM-as-Judge 实现 | **仅 20 / 4,508** |

【源码】验证方式:`ls packages/`;`grep -rn "memory" kb/data/cordis-surface.md`(空);`grep -rn "worktree" packages/ --include="*.ts"`;`packages/sandbox/sandbox-local/src/index.ts:2-3`。

> **这条区分直接决定排序逻辑**:内建类别 A 是「复刻」(面试官问「你和 DSH 有什么不同」时答不上来);内建类别 B 是「补空白」(有天然叙事)。**你点名的五个能力里,记忆系统 / 上下文洞察 / 工作树 / Docker 全部属于 B,电脑操作属于 A。**

---

## 3. 你点名的五个能力,逐个核实

| 你的提名 | 实测归类 | 结论 |
|---|---|---|
| **记忆系统** | **B · DSH 无** | 判断正确。209 个社区插件在填,DSH 零原生支持。市场最热(`dsh-mnemon` 45k/月、`@openviking/dsh-memory-plugin` 38k/月) |
| **上下文洞察** | **B · DSH 无** | 判断正确。`dsh-context`(147k/月)是本机已装插件,非 DSH 官方能力 |
| **工作树** | **B · DSH 无** | 判断正确。DSH 无 worktree 管理;但**归档项目 `spool` 实现过**(P1:worktree task isolation),是现成参考 |
| **Docker** | **B · DSH 无** | 判断正确。DSH 沙箱只有 OS 原生后端。⚠️ 但 Docker 需要守护进程,与「无 UI、脚本驱动」的定位有摩擦 |
| **电脑操作** | **A · DSH 官方有** | ⚠️ 判断需修正。`packages/computer-use` 是**原生 seam**,官方有 4 个 provider(cua-driver native/MCP 等)。内建它属于类别 A |

**五个里四个判断正确,一个(电脑操作)属于官方已有。** 这个方向感是准的。

---

## 4. 与求职目标的交叉验证

排序不能只看市场热度,必须回到 T1 岗位的逐字用词。

### 4.1 e签宝「Agent 基础设施工程师」六条职责 → DSH 覆盖情况

| JD 职责 | DSH 是否有 | 市场插件数 | 我们的现状 |
|---|---|---|---|
| ① Harness 核心(Loop / 编排 / 会话状态 / **长任务恢复** / **中断恢复** / 审批 / 重试回滚) | ✅ 有 | — | M1 已覆盖一半 |
| ② Loop 工程化(Plan→Execute→**Evaluate**→Reflect→Retry) | ⚠️ 有 loop,无 Evaluate | — | — |
| ③ **Eval 基础设施**(回归测试集 / Trace 评估 / LLM-as-Judge / A-B / Eval 驱动迭代) | ❌ **无** | **20** | — |
| ④ Tool/MCP 治理(权限管控 / 调用审计 / Prompt Injection 防御 / 敏感操作审批 / 故障隔离) | ✅ 有 | 151(安全类) | M3 部分覆盖 |
| ⑤ 安全合规与可观测(Guardrails / 审计追溯 / **沙箱执行环境** / 链路追踪 / Token 成本) | ✅ 有 | — | M3 部分覆盖 |
| ⑥ SDK/CLI 工具链 + **Skill 资产** | ✅ 有 | 178(Skill 类) | — |

【文档】`05-job-search/jd/杭州天谷信息科技e签宝-Agent基础设施工程师-27届-2026-09-23.md`

### 4.2 排序时用的四个判据

1. **JD 命中度** —— 是否出现在 T1 岗位正文(尤其加分项逐字)。
2. **可量化度** —— 能否变成 L3 的量化指标(否则不写进简历)。
3. **DSH 空白度** —— 属类别 B 加分;属类别 A 减分(复刻感)。
4. **实现成本** —— 按 DSH 同类实现的 LOC 估算。

---

## 5. 原生内置候选排序(建议)

### T1 · 应当内建(直接落在 L3 验收标准上)

| 序 | 能力 | 归类 | JD 命中 | 量化指标 | 成本参考 | 为什么排这里 |
|---|---|---|---|---|---|---|
| **1** | **Eval 基础设施**(回归集 + Trace 评估 + LLM-as-Judge) | **B** | ③ 整条 + ② 的 Evaluate | 任务成功率 / 回归检出率 | 从零,~1,500 LOC | **唯一「JD 写成完整职责 + DSH 零实现 + 市场 20/4508」的三重交集**。且它让 M1/M2/M3 的指标**可自动产出**,不是额外负担而是验收工具 |
| **2** | **上下文压缩 + spill** | A | 加分项逐字「上下文压缩」 | token 降 ≥40% | 5,362 LOC(可压到 ~1,500) | 已是 M2;JD 逐字;量化最清晰。**类别 A 但成本最低、收益最直接** |
| **3** | **工具权限 + 沙箱策略** | A | ①④⑤ 三处 | 6 类故障恢复 ≥90% | 9,456 LOC(策略层 ~800) | 已是 M3;JD 出现频率最高(权限/审批/沙箱/审计/隔离) |

### T2 · 值得内建(差异化,但需先确认带宽)

| 序 | 能力 | 归类 | JD 命中 | 成本参考 | 取舍 |
|---|---|---|---|---|---|
| **4** | **记忆系统**(跨会话检索 + 可删除) | **B** | 加分项「长期记忆检索」 | ~1,200 LOC | 209 个插件在填 = 需求被验证,但**同质化严重**。差异化只能来自「与 M1 日志同源」(记忆从事件日志派生),而不是另做一套存储 |
| **5** | **持久 Goal / 长任务恢复** | A | ① 逐字「长任务恢复」「中断恢复」 | 3,238 LOC | JD 逐字命中,但属类别 A。且 [ADR-0001](decisions/0001-scope-and-non-goals.md) 现列「不做」——若内建需改 ADR |

### T3 · ~~暂不内建~~ → **已全部纳入(2026-10-12)**

> ⚠️ **本表作废。** [ADR-0003](decisions/0003-usable-harness-and-t3-scope.md) §2 已将范围改为**全做,不裁剪**,下列五项全部纳入。表内「理由」列多为**基于求职判据**的论证,而项目已与求职解耦,故该列不再有效。**保留本表只为记录当时的推理过程,不得据此排除任何能力。**
>
> 排序(而非排除)依据见 [t3-ordering-and-guardrails.md](t3-ordering-and-guardrails.md) §1。

| 能力 | 归类 | 当时排除理由(已失效) | 现状态 |
|---|---|---|---|
| 子 Agent / Agent Teams | A | 17,588 LOC,ADR-0001 明确非目标 | **纳入**,排在 T3 链末(22 条依赖) |
| 工作树 | B | 成本低(~300 LOC)但 JD 未点名 | **纳入** |
| Docker 沙箱后端 | B | 需守护进程,与「脚本驱动、无 UI」定位摩擦 | **纳入**(本机未装 Docker,先后置) |
| 电脑操作 / 浏览器 | A | DSH 官方已有 seam + 4 provider;内建纯属复刻 | **纳入** |
| Skill 系统 | A | 3,564 LOC;JD 仅提「Skill 资产」 | **纳入**,排在 T3 链首(3 条依赖) |

### 5.1 一句话建议

> **把 Eval 基础设施提到第一位,与 M1 合并为「日志即事实 → 日志即证据」一条线。**
>
> 理由:Eval 天然消费 M1 的事件日志(`从日志重建请求`正是 eval 的输入),两者**同源**;而它同时补齐 JD 第③条职责与第②条的 Evaluate 环节。这样三个机制扩成「M1 日志 → M2 压缩 → M3 工具 → M4 Eval」,仍是一条咬合的链,不是四个 demo。

---

## 6. 未解决问题 / 待验证

1. ~~**【未验证】4,508 这个数字的构成。**~~ **已复核（2026-10-16）**：**这个数字无法复现，本清单的引用方式不可靠——但结论不受影响**。

   **引用本身就回不去**。文档写的是「`awesome-dsh-plugin/count.json`」与「按 `###` 分节解析 `README.md`（`<!-- BEGIN PLUGINS -->` … `<!-- END PLUGINS -->`）」。实测**四个同名 fork 全部对不上**：

   | 检查 | 结果 |
   |---|---|
   | `count.json` | **404**（四个 fork 全部） |
   | `<!-- BEGIN PLUGINS -->` 标记 | **0 处**（四个 fork 全部） |
   | `###` 分节数 | **14**（文档写 23） |
   | 仓库名 | `awesome-dsh-plugin` 有 **≥4 个不同作者的同名仓库**，文档没写 owner |

   **当前真实数字**（`bruc3van/awesome-dsh-plugin`，`snapshot_date` 2026-10-15）：

   | 文件 | 值 |
   |---|---|
   | `data/stats.json` | `repositories: 16,836` |
   | `data/approved.json` | 16,836 条 |
   | `data/market.json` | `source_repo_count 19,477` / `pool_count 15,864` / `entries 600` |

   **所以「4,508」既不是当时的准确数、也不是现在的数。** 该清单几天内从 4,508 涨到 16,836（约 3.7 倍），**这本身就说明它是一个自动爬取的、快速变动的池子**，而不是经过评审的精选集——**恰好印证了文档原来的怀疑：「可能存在批量提交」**。

   **对排序结论的影响：无。** §5 的排序依据是**「DSH 源码里有没有」**（`packages/eval` 不存在），插件数只是佐证。**分母变大了，「只有 20 个碰 Eval」这个比例反而更强**——但它也同样是未复核的。

   > **教训**：引用一个**会变动的第三方爬取产物**时，必须同时记下 **owner / commit / 采集时刻**。只写仓库短名与文件名，**几天后就无法复核**——这不是「数字错了」，是**引用方式让人无法判断它对不对**。
2. ~~**【未验证】npm 下载量的真实基线。**~~ **已复核（2026-10-16，npm downloads API 逐日序列）**：**原判断成立——下载量被批量拉取污染，不能当使用量读**。

   逐日曲线是**机器人产物的教科书形态**：

   ```
   2026-09-13   9,973  #################################
   ...（连续 10 天，每天约 1 万）
   2026-09-22   7,960  ##########################
   2026-09-23      91   <- 断崖
   2026-09-24     190
   ...（之后稳定在 50–110/天）
   ```

   | 分组 | 天数 | 日均 |
   |---|---|---|
   | 高峰日（>1000） | 9 | **10,379** |
   | 正常日（≤1000） | 19 | **81** |
   | **比值** | | **128×** |

   **两条判据**：①**128 倍**的落差；②高峰期恰好是**连续 10 天**然后**整块停止**——**真实用户群不会像一块砖一样同时到来又同时离开**。

   （对比：文档当时记的「月 131,719 / 周 542」，现在实测是「月 94,950 / 周 413」——**周下载量比月下载量的 1/21 还低**，这本身就说明月总量主要是那段异常期贡献的。）

   **对这个文档结论的影响**：凡是**以下载量论证「社区在某方向上活跃」**的段落都要打折。本文的**排序结论不依赖下载量**（依据是「DSH 源码里有没有」），所以 §5 不受影响；但 §1 的市场规模描述应读作**「仓库数」而非「用户数」**——而仓库数本身也已证明在几天内翻了 3.7 倍（见上面第 1 条）。
3. ~~**【未验证】`dsh-context`(147k/月)的具体机制。**~~ **已读源码（2026-10-16，本机 `~/.dsh/profiles/desktop/node_modules/dsh-context@0.65.0`）**。

   **结论：它确实是「上下文洞察」，但洞察的对象是 DSH 自己已有的数据，它自己不产生任何测量。**

   | 事实 | 证据 |
   |---|---|
   | 依赖注入**只有一项** | `inject = ["sessionProjections"]` |
   | 消费的其他服务只有 logger/fiber | `ctx.logger.*`、`ctx.fiber.*` |
   | **不做 tokenization** | 全包 grep 无 tokenizer 依赖 |
   | 所谓 `tokenCountOf` **只是数值清洗** | `Math.max(0, Math.round(value))`，输入是 `usage.inputTokens` / `cacheReadTokens` / `outputTokens` |
   | 体积构成 | `lib/index.js` 4,000 行（宿主侧）、`lib/client.js` 14,312 行（**浏览器侧**）——**UI 是主体** |
   | 功能全是展示 | Insights 页、Context 标签页、右侧栏面板、`/context` 斜杠命令 |

   **「147k/月」这个下载量的机制**：它的价值在于**把 runtime 已有的投影数据可视化**（KPI 带、7 天 token/成本图、活动热力图、构成环图、可跳转的会话卡片）。

   > **对「是否值得内建」的判断**：这条**支持原结论**——上下文洞察是**消费** runtime 数据的能力，不是**产出**测量的能力。`dsh-context` 之所以能做，是因为 **DSH 已经记录了 usage 与投影**。所以它不构成「社区填了 DSH 的空白」这一类；它是**在 DSH 已有数据之上的展示层**。

   **一个与本项目直接相关的旁证**：`dsh-context` 读的是 `usage.inputTokens` / `cacheReadTokens` / `outputTokens`——**正是本项目曾经「请求了却丢弃」的那三个字段**（缺陷 #64）。**一个展示层依赖的数据，在 runtime 里被丢掉了，那么无论展示层多好都只能显示空白。**
4. ~~**【推断待证】「Eval 排第一」的结论依赖 JD 样本。**~~ **已复核（2026-10-16，JD 原件在本仓 `05-job-search/jd/`）**：**担心不成立，而且实际证据比文档记的更强。**

   文档说该结论「基于 17 份 JD 中匹配度最高的 **1 份**（e签宝）」。**核对 17 份原件后：不是 1 份，是 4 份**明确把评测写成职责：

   | JD | 原文用词 |
   |---|---|
   | 海康威视 · 研究院 · 大模型应用算法工程师 | **评测体系建设** / 评测与可… |
   | 云深处科技 · 智能体算法工程师 | **评测体系** |
   | 脑力波 Nolibox · LLM 应用产品工程师 | 评测（8 处提及） |
   | 阿里巴巴 · AI 应用开发工程师 · 27届实习 | 评测 |

   （另有 3 份**提到**评测但不是列为职责，已排除。）

   **所以 Eval 的权重不是押在一个样本上**——它是 17 份里 **4 份（约 24%）** 的共同要求，而且在**不同规模、不同细分方向**的公司里都出现（安防巨头 / 机器人 / 产品向 / 大厂）。**样本换人也不会让这个信号消失。**

   **仍然保留的部分**：文档建议的「2027-01 重新核验」**依然有意义**，但理由变了——不是「怕样本只有一个」，而是**岗位池本身会变**（春招与秋招的岗位构成不同）。这一条从「待证推断」降级为「**常规的季节性复核**」。
5. ~~**【未验证】社区插件与 DSH 官方契约的兼容成本。**~~ **本项目已经把这个实验做完了（2026-10-16）** —— 不是 2 天的小实验，而是**两种模式都实现并上线**，所以可以直接给出结论。

   问题原文是「从事件日志派生记忆是否真的比独立存储更省事」。本项目**两者都有**：

   | 状态 | 存放方式 | 代码里的理由 |
   |---|---|---|
   | `goal`（持久目标）、`todo` | **日志派生**（fold from the log） | `todo/store.py`：「The list is folded from the log like every other durable state here」 |
   | `schedule`（提醒） | **独立存储**（唯一的例外） | `schedule/runtime.py`：「This is the one durable store in the runtime that is **not** the session log, and the reason is worth stating because it looks like an inconsistency」 |

   **判据不是「省事」，是「这件事发生时，会话是否在场」**：

   > 「Everything else lives in a session log because a session is always the thing being worked on. A reminder is different: **it must fire at a wall-clock time whether or not its session is loaded**... Putting reminders in the session log would mean they only exist once the session is loaded -- **which is precisely the situation a reminder is supposed to escape.**」

   **所以对「从日志派生记忆」的直接回答**：

   - **日志派生的成本比想象的低**——`goal` 与 `todo` 都是这么做的，没有额外机制；折叠逻辑与 M1 的重放共用同一条路径。
   - **但它的适用条件是一条硬边界**：**状态必须在「会话在场」时才需要存在**。一旦某状态需要在**没有会话加载**时也生效（提醒、定时任务），日志派生就**结构性地**不成立——不是更贵，是**根本做不到**。
   - **记忆系统属于哪一类，取决于它的使用方式**：若记忆只在某会话里被读写，日志派生更省事；若记忆要**跨会话、且可能在无会话时被查询或写入**（这通常是「记忆」的意义所在），就必须独立存储。

   > 这条从「未验证的实验」变成了**已实现的结论**：本项目用两个真实模块给出了这个取舍的分界线。
