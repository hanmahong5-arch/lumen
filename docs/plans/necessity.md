# Lumen 存在必要性（产品否决器）

**裁决**：Lumen 有存在必要，当且仅当坚持卖「Agent 失败与烧钱的黑匣子」，并用
Forge/Kova 管道强制露出同一份 lifecycle；若改卖「全量观测平台」，则没有独立存在的必要。

口头一句（Triple）：**Forge 编排 · Kova 证明 · Lumen 看见。**
真源操作面：`2b-svc-kova/docs/kova/TRIPLE.md`。旁路裁决卡：Cursor canvas
`lumen-necessity.canvas.tsx`。

## 边界

| 组件 | 职责 | 不是 |
|---|---|---|
| Lumen（本仓） | 本地 trace → replay / cost / lifecycle / dashboard | Agent 运行时 |
| Kova | 持久执行、WAL、「证明跑过」；Connected 加深器 | 事故叙事读面 |
| Forge | 编排、HITL、demo；`/ops` 控制室 | 黑匣子本身 |
| Seal | 副作用围栏（opt-in）claims/replays | 成本 / 拒绝语义 |

Kova 是加深器，不是门槛。Core 在无 Kova 时必须完整可用。

## 没有它时的空洞

| 层 | 已有 | 缺 Lumen |
|---|---|---|
| Forge | 启停 / HITL / demo | 事故叙事不可演示、不可转发 |
| Kova | WAL / 可恢复 | 钱、拒绝语义、续跑不成一张图 |
| Seal | 围栏指标 | 不答哪步贵、哪步被拒 |
| 通用观测 | span / 调用可见 | 不绑诚实语义 + 续跑仪式 + 零 LLM replay |

编排与执行可以没有好看的读面；**卖得出去、排得了事故、控得住账单**不能。

## 三条硬条件（缺一可被替代）

1. **钱可归因** — 按步/按模型美元与 outlier（`cost` / `pricing`）。
2. **语义真** — lifecycle 诚实区分成功 vs CAN vs policy deny vs crash。
3. **活可续** — LangGraph checkpointer / `resume_hint`；Connected 时 Kova WAL 加深。

公式：`可安装 × 诚实事故叙事 × (本地完整 + Kova 选配加深)`。任一项为 0，退化为可有可无的
trace 查看器，存在必要性消失。

## 对谁必须有

- **独立 LangGraph 开发者** — 崩了或账单炸了：≤5 分钟可分享 lifecycle；不知 Kova 也能用 Core。
- **Forge/Kova 企业演示** — `FORGE_LUMEN_BIN` 时 `/demo` 自动打开 lifecycle 并卡住 Closed Beta；`/ops` 亦可导出。
- **事故值班** — 同一份 HTML/CLI replay，零二次 LLM 花费。

不为「只要通用 APM / 多框架全家桶」的人存在——那是刻意不做的红海。

## 何时不该存在 / 不该扩（停手条件）

- 做成第二个 Agent 运行时（与 Kova 打架）
- 做成通用 Observability 平台（与 Langfuse 正面拼功能表）
- 无 Kova 就不能给出任何 aha（违反 Core 完整）
- 公开表面有桩 API / 假绿联测（黑匣子靠信任，假一次就死）

## 与 `/ops` 的关系

`/ops` **不创造**必要性，只**放大分发**（canned demo → cancel → lifecycle → Seal metrics）。
否决仍看三条硬条件。

## 怎么用这份文件

功能取舍、PR 范围、文档叙事冲突时：先对照本页。通过后再看
[industrial-ux.md](./industrial-ux.md) 的下一刀候选（PyPI / Forge 默认嵌入 / replay 对齐）。
测量标准仍是：install → 5 分钟事故 HTML → 可分享产物——不是功能表长度。
