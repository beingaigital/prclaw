# PRClaw

PRClaw 是一个独立的 CLI 智能体项目，目标是把传统公关传播工作流快速改造为可执行的 Agent 流程。

它基于 ReAct + LangChain 的工具调用机制，已在项目内本地化迁移 `All-in-One-PR-Solution` 的核心能力（GraphRAG / 方案生成 / 报告生成 / RLHF 反馈）。

## 核心能力

- 动态工具注册：`tools/*.py` 自动发现，无需手改核心文件。
- Skills Layer：`skills/*.md + config/skills.yaml` 按 query 自动激活。
- Profile Layer：`IDENTITY.md / USER.md / SOUL.md / AGENT.md / MEMORY.md / TOOLS.md` 注入系统上下文。
- PR 领域工具：
  - `pr_plan_schema`
  - `pr_plan_requirements`
  - `pr_query_knowledge`
  - `pr_generate_plan`
  - `pr_generate_report`
  - `pr_collect_feedback`
- 内置 UnifiedPRSystem：通过 `utils/unified_adapter.py` 懒加载（仅本仓库，无外部工程回退）。

## 目录结构

```text
prclaw/
├── agent/                    # ReAct Agent 封装
├── cli/                      # 交互式命令行
├── config/
│   ├── model.yaml            # Agent 模型配置
│   ├── tools.yaml            # 动态工具注册配置
│   ├── skills.yaml           # 技能自动激活配置
│   ├── profile.yaml          # Profile 注入配置
│   ├── prclaw.yaml           # PRClaw 运行配置（内置/外部 Unified 系统策略）
│   └── unified_config.yaml   # UnifiedPRSystem 配置
├── core/                     # 本地化迁移的 All-in-One 核心业务模块
├── prompt/system_prompt.txt  # 系统提示词
├── skills/                   # 技能文件（markdown）
├── tools/                    # PR 工具实现
├── utils/
│   ├── unified_adapter.py    # UnifiedPRSystem 适配层
│   ├── web_search.py         # 外部信息检索
│   └── ...
├── outputs/                  # 方案 markdown 输出（运行后自动创建）
└── memory/                   # 会话记忆（运行后自动创建）
```

## 安装与运行

1. 进入项目目录

```bash
cd /Users/biaowenhuang/Documents/All-in-One-PR-Solution/prclaw
```

2. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. 环境变量（推荐）

```bash
cp .env.example .env
# 编辑 .env：至少填写 APIKEY（或你在 model.yaml 中配置的 api_key_env）与 NEO4J_PASSWORD
```

4. 可选：启动前自检

```bash
python3 main.py --check
# 或
python3 -m cli.main --check
```

5. 配置模型与 Unified 系统

- `config/model.yaml`：设置 Agent 主模型（默认 `openai_compatible/qwen3.5-plus`）。
- `config/prclaw.yaml`：PRClaw 行为与默认检索开关等。
- `config/unified_config.yaml`：配置 Neo4j / LLM / RLHF 等核心参数（敏感项请用 `.env`）。

OpenAI-compatible 网关示例（如 DashScope Coding）：

```yaml
main:
  provider: openai_compatible
  model: qwen3.5-plus
  api_key_env: APIKEY
  base_url: https://coding.dashscope.aliyuncs.com/v1
```

并在环境变量中配置 `APIKEY`（或写入 `.env`）。

6. 启动 CLI

```bash
python3 main.py
```

或：

```bash
python3 -m cli.main
```

可用命令：
- `/new` 新建会话
- `/memory` 恢复历史会话
- `/tools` 查看工具
- `/tools reload` 重新扫描工具
- `/skills` 查看技能
- `/skills match <query>` 预览技能激活
- `/profile` 查看 profile 注入状态
- `/models` 查看模型配置
- `/clear` 清理记忆与沙箱
- `/exit` 退出

## 关键配置

### `config/prclaw.yaml`

- `unified_config_path`：Unified 系统配置文件路径（相对 prclaw 根目录）。
- `default_output_types`：默认方案输出类型（A-F）。
- `default_use_graph_rag` / `default_use_web_search`：默认检索开关。
- `web_search.provider`：`auto | serper | duckduckgo`。

### `config/tools.yaml`

- `auto_discover: true`：自动发现 `tools` 目录。
- `enabled_tools` / `disabled_tools`：按名称精细开关工具。

### `config/skills.yaml`

- `auto_activate: true`：按 query 自动激活技能。
- `manual_activation.always_on`：常驻技能。
- `skills.<id>.keywords`：触发关键词。

## Unified 能力迁移

PRClaw 已内置 UnifiedPRSystem 及核心业务模块，工具层仍通过统一适配接口调用：

- `UnifiedPRSystem.query_knowledge`
- `UnifiedPRSystem.generate_pr_plan`（以及 `plan_generator.generate_plan`）
- `PRReportGenerator.generate_report`
- `FeedbackCollector.collect_feedback`

## 后续扩展建议

- 新增 `tools/pr_export_ppt.py`：把方案结构化输出成 PPT。
- 新增 `tools/pr_copywriting.py`：按渠道自动生成文案资产。
- 新增 `tools/pr_creative_assets.py`：对接配图/海报生成能力。
- 把你原先 `all in one pr solution` 的高级流程逐步拆成可插拔 tools + skills。
