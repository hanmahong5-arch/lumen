中文 | [English](README.md)

# Lumen

面向 AI agent 的可观测性与崩溃恢复工具：回放任意一次运行、追踪每一笔 LLM 花费、故障后从检查点恢复。

Lumen 由两部分组成 —— 一次性 `instrument()` 接入的 Python SDK，以及指向落盘 trace 文件的 Rust
CLI/仪表盘。目标用户是用 LangGraph、CrewAI、AutoGen 或纯 Python 搭建 agent、需要知道「钱花在哪」「第
几步出的问题」「怎么不重跑就恢复」的开发者。项目目前处于早期 alpha 阶段
(`Development Status :: 3 - Alpha`，`lumen-sdk/pyproject.toml:22`)：核心的 trace、成本、replay 链路
已实现并有测试覆盖，LangGraph 是唯一有专用 tracer/checkpointer 的集成，CrewAI/AutoGen 尚未实现。Lumen
还没有发布到任何包仓库 —— 目前唯一受支持的安装方式是从源码构建。

## 核心能力

- **一行自动埋点** —— `instrument()` 给 OpenAI、Anthropic SDK 客户端和 LangGraph 打补丁，调用点无需
  改动即可全量记录 (`lumen-sdk/lumen/instrument.py`)。
- **确定性回放** —— 从 trace JSON 零 LLM 花费地重放任意一次历史运行，可用 `--from-step` 从中途某步开
  始 (`lumen-core/src/replay.rs`、`lumen-sdk/lumen/replay.py`)。
- **成本追踪与定价** —— 按 agent、按模型汇总 token 折算的美元花费，覆盖 30+ 模型，并标记单次运行的
  成本离群值 (`lumen-core/src/cost.rs`、`lumen-core/src/pricing.rs`、`lumen-sdk/lumen/cost.py`、
  `lumen-sdk/lumen/pricing.py`)。
- **崩溃安全检查点** —— 为 LangGraph 提供的 `LumenCheckpointer` 落盘持久化、零外部依赖，进程崩溃后可
  从最近一次检查点恢复 (`lumen-sdk/lumen/integrations/langgraph.py`)。
- **预算护栏与异常检测** —— 超预算可配置为直接中止，另有成本/指标异常检测，倍数阈值可配置
  (`lumen-sdk/lumen/_budget.py`、`lumen-sdk/lumen/_anomaly.py`)。
- **Web 仪表盘** —— trace 时间线；接上一个真实运行的 Kova + Netdata 后还有实时 Metrics 标签页
  (Prometheus 指标 + 逐指标 ML 异常带) 和 Terminal 标签页 (白名单式 Kova 控制台，不是真 shell)
  (`lumen-cli/src/dashboard.rs`、`lumen-cli/src/netdata.rs`、`lumen-cli/src/kova.rs`)。

## 快速开始

```bash
git clone https://github.com/hanmahong5-arch/lumen.git
cd lumen

# CLI (Rust，稳定版工具链需支持 Edition 2024；构建已在 1.96.0 验证)
cargo build --release
./target/release/lumen --version   # -> lumen 0.1.0

# Python SDK (Python 3.10+，可编辑安装)
pip install -e ./lumen-sdk                 # SDK 核心
pip install -e "./lumen-sdk[langgraph]"    # + LangGraph 集成
pip install -e "./lumen-sdk[all]"          # + 全部集成

# 测试
cargo test -p lumen-core -p lumen-cli
cd lumen-sdk && python -m pytest
```

完整构建/排障步骤见 [docs/INSTALL.md](docs/INSTALL.md)。

```python
from lumen import instrument, trace

instrument()  # 自动识别 OpenAI / Anthropic / LangGraph；此后所有调用自动记录

with trace("research-task") as t:
    r1 = client.chat.completions.create(model="your-model", messages=[...])
    print(f"Trace: {t.trace_id}, cost: ${t.total_cost_usd:.4f}")
```

手头没有 LangGraph 或 Kova？`lumen demo` 会启动一个临时 `kova-rest`、跑一次真实 agent、并打开生成的
lifecycle `.html` —— 需要 `KOVA_LLM_API_KEY`，以及 `PATH` 上的 `kova-rest` 二进制（或用 `--kova-url`
指向一个已在运行的 Kova）。

## 架构

```
lumen-core/   Rust 引擎（内嵌进 CLI，不发 HTTP）：trace 类型、replay、成本汇总、定价表、
              供仪表盘/导出使用的 lifecycle 组装。在本地 vendor 了 kova-types 的 trace
              格式，使这个 crate 可独立构建 (lumen-core/Cargo.toml:9-11)。
lumen-cli/    Rust CLI + Web 仪表盘：子命令 (main.rs)、仪表盘服务端 (dashboard.rs/.html)、
              Kova 控制台解释器 (kova.rs)、Netdata 指标代理 (netdata.rs)、trace 拉取
              (pull.rs)、demo 编排 (demo.rs)。
lumen-sdk/    Python 包 `lumen-ai`：instrument() 自动埋点、trace 读写器、成本/定价、预算
              与异常检测、脱敏、分层配置 (config.py)，以及 integrations/ (OpenAI、
              Anthropic、LangGraph tracer + checkpointer)。
```

数据流：你的 Python 代码 → **Lumen SDK** 把 trace JSON 写盘（或以类 OTLP 方式导出）→ **Lumen Core**
(内嵌在 CLI 里) 读取这些 JSON 做 replay/成本/lifecycle 渲染 → 你的 LLM 供应商只会被你自己的代码调用，
Lumen 从不代为发起。

## 配置

SDK 配置按此顺序解析：硬编码默认值 → `~/.lumen/config.toml` → `./lumen.toml`（从 cwd 向上查找）→
`LUMEN_*` 环境变量 → 显式 builder 覆盖 (`lumen-sdk/lumen/config.py:1-9`)。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LUMEN_ENABLED` | `true` | trace 采集总开关 (`config.py:351`) |
| `LUMEN_TRACE_DIR` | `./traces` | trace JSON 的读写目录 (`config.py:352`) |
| `LUMEN_SAMPLING_RATE` | `1.0` | 被记录调用的比例 (`config.py:353`) |
| `LUMEN_BUDGET_USD` | `0.0` | 预算上限；`0` = 不限 (`config.py:357`) |
| `LUMEN_KILL_ON_BUDGET` | `false` | 超预算是否中止 (`config.py:358`) |
| `LUMEN_ANOMALY_MULTIPLIER` | `2.0` | 成本离群阈值：超出单次均值的倍数 (`config.py:359`) |
| `LUMEN_REDACTION_ENABLED` | `false` | 写 trace 前先按规则脱敏 (`config.py:355`) |
| `LUMEN_OTLP_ENABLED` | `false` | 是否导出到 OTLP 端点 (`config.py:368`) |
| `LUMEN_NETDATA_URL` | — | 仪表盘 Metrics 标签页数据源，如 `http://localhost:19999` (`main.rs:333`) |
| `LUMEN_KOVA_URL` | — | `pull`/`export`/`kova`/Terminal 标签页用的 Kova 地址 (`main.rs:350`) |
| `LUMEN_KOVA_API_KEY` / `KOVA_API_KEY` | — | Kova 的 `X-API-Key`，只留在服务端，绝不下发到浏览器 (`main.rs:341-342`) |
| `KOVA_LLM_API_KEY` | — | `lumen demo` 的 agent 循环所必需 (`demo.rs:109`) |
| `KOVA_REST_BIN` | — | `lumen demo` 临时路径用的 `kova-rest` 二进制路径 (`demo.rs:143`) |

## CLI 概览

```
lumen replay <trace-id> [--from-step N]                 # 回放一次运行，零 LLM 花费
lumen cost --last 24h [--format json]                    # 成本报表 + 单次运行离群标记
lumen traces [--trace-dir ./traces]                       # 列出所有 agent 运行
lumen dashboard [--netdata-url …] [--kova-url … --api-key …]  # Web 界面
lumen metrics --last 10m [--format json]                  # 无界面 Netdata 快照
lumen pull --kova-url <url> [--deep]                       # 从在线 Kova 拉取 trace
lumen export <run> [--trace-dir …] [--kova-url …]          # 单次运行 -> 独立 .html
lumen kova "<verb>" [--kova-url …] [--yes]                 # 一次性 Kova 控制命令
lumen demo [--kova-url …] [--kova-bin …]                   # 零配置：跑一次并可视化
lumen tour                                                  # 组装多运行的 index.html
```

`lumen kova` / 仪表盘的 Terminal 标签页对 Kova REST API 说的是一套白名单动词 —— 只读动词
(`status`、`agents`、`workflows`、`schedules`、`tools`、`queues`、`traces`、`llm` 等)、安全变更
(`agent <id> run|stop|pause|resume|restart|approve|deny`、`workflow <id> cancel|resume`、
`schedule <id> pause|resume`)，以及需要 `--yes` 或二次确认的破坏性动词
(`agent <id> reset|terminate|delete`、`schedule <id> delete`、`trace <id> delete`) —— 见
`lumen-cli/src/kova.rs`。它是一个 REST 解释器，不是 shell：没有原始 `std::process` 执行，也没有任意
路径/方法/主机。

## 开发约定

- Rust workspace lint 禁用 `unwrap()`/`expect()`/`panic!()` 以及 unsafe-in-unsafe-fn
  (`Cargo.toml:11-17`)；CI (`.github/workflows/release-lumen-cli.yaml`) 对每个打 tag 的构建都跑
  `cargo test`、`cargo clippy -- -D warnings`、`cargo fmt --check` 三道闸。
- `lumen-core` 的 trace 类型是 `kova-types` 的本地 vendor 拷贝（手工保持同步，见
  `lumen-core/Cargo.toml` 里的注释），使这个仓库能不依赖私有姊妹仓独立构建。
- 发布 tag：推一个 `vX.Y.Z` 会同时触发 `publish-lumen-sdk.yaml` 和 `release-lumen-cli.yaml`；只想发
  一侧可用 `lumen-sdk-vX.Y.Z` / `lumen-cli-vX.Y.Z` 这两个按产物区分的 tag 前缀。
- 本仓自己的开发约定（品牌边界、目录布局、命令速查）写在仓库自带的开发约定文档里，该文档按设计不进公
  开树（见 `.gitignore`）。

## 相关项目

Lumen 是 **Kova** (`2b-svc-kova`，Lurus 的一个 agent 运行时) 的开放、独立品牌可观测性客户端 ——
`lumen-cli` 会调用 Kova 的 REST API (`pull`、`export --kova-url`、`kova`、Terminal 标签页)，
`lumen-core` vendor 了 Kova 的 trace schema，但 Lumen 的终端用户完全不需要知道 Kova 的存在。仪表盘的
lifecycle 渲染 (`lumen-cli/src/dashboard.rs`、`lumen-core/src/flow_types.rs`) 与 **Forge**
(`2b-bs-forge`) 共用同一套数据模型，Forge 自己的 lifecycle 导出功能就是嵌入 Lumen 实现的。

## 许可证与第三方声明

MIT —— 见 [LICENSE](LICENSE)。编译产物 `lumen` 二进制所含第三方 Rust crate 的许可证清单见
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)（由 `cargo about generate` 生成）；除上述
`kova-types` trace schema 拷贝外，本仓没有 fork 或 vendor 其他上游项目。
