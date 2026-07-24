# AI Career Research Agent（AI 求职研究助手）

> 基于 LangGraph 多 Agent 工作流构建的智能 AI 求职研究助手，为求职者提供一站式求职准备服务。

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5%2B-green)](https://langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-orange)](https://langchain.com)

## 📖 项目简介

AI Career Research Agent 是一个面向 AI 岗位求职者的智能研究助手。它能够根据用户输入的目标岗位、目标城市以及个人技能情况，自动完成：

- **岗位市场分析** — 搜索并分析目标岗位的市场需求、薪资水平、行业趋势
- **技能差距评估** — 对比用户现有技能与岗位要求，识别差距并排序优先级
- **学习路线规划** — 生成个性化的分阶段学习计划和推荐资源
- **简历优化建议** — 提供关键词优化、项目描述改进、ATS 通过技巧
- **面试准备方案** — 生成高频技术面试题、行为面试题、系统设计题
- **GitHub 项目推荐** — 推荐相关开源项目，提升简历竞争力

本项目基于 LangGraph 官方 Open Deep Research 项目进行二次开发，将通用深度研究工作流扩展到 AI 求职场景，形成完整的求职研究闭环。

---

## ✨ 核心功能

### 1. 岗位市场研究
- 自动搜索目标岗位的招聘信息
- 分析岗位需求趋势和市场热度
- 调研薪资水平和福利待遇
- 识别行业发展趋势

### 2. 技能差距分析
- 提取岗位核心技能要求（硬技能 + 软技能）
- 对比用户现有技能，生成匹配度报告
- 按优先级排序技能差距
- 估算补齐核心技能所需时间

### 3. 学习路线规划
- 分阶段学习计划（3-5个阶段，每阶段2-4周）
- 推荐学习资源（免费课程、付费课程、官方文档）
- 设计里程碑项目，从简单到复杂
- 总体学习周期预估

### 4. 简历优化建议
- 关键词优化，通过 ATS 筛选
- STAR 法则项目描述模板
- 技能展示策略建议
- 简历格式和结构建议

### 5. 面试准备方案
- 技术面试题（5-8题）
- 行为面试题（5题）
- 系统设计题（2-3题）
- 编程题（3-5题）

### 6. GitHub 项目推荐
- 5-8个相关开源项目推荐
- 覆盖不同技能领域
- 包含贡献建议
- 展示技能匹配度

---

## 🚀 工作流架构

```text
用户输入求职需求
        │
        ▼
┌─ 信息澄清 (ClarifyUser) ─── 收集岗位/城市/技能/经验
│
├─ 岗位市场研究 (JobMarketResearch) ─── 并行搜索
│   ├── 岗位需求分析
│   ├── 薪资水平调研
│   └── 行业趋势分析
│
├─ 技能差距评估 (SkillGapAssessment) ─── LLM分析
│   ├── 用户现有技能 vs 岗位要求
│   ├── 差距优先级排序
│   └── 学习时间估算
│
├─ 学习路线规划 (LearningRoadmap) ─── LLM生成
│   ├── 分阶段学习计划
│   ├── 推荐学习资源
│   └── GitHub 项目推荐
│
├─ 简历优化建议 (ResumeOptimization) ─── LLM分析
│   ├── 关键词优化
│   ├── 项目经验包装
│   └── 格式建议
│
├─ 面试准备 (InterviewPrep) ─── LLM生成
│   ├── 高频面试题
│   ├── 行为面试题
│   └── 技术面试模拟
│
└─ 生成最终求职报告 (FinalReport)
```

---

## 🏗 技术架构

### 核心技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 开发语言 |
| LangGraph | 0.5.4+ | Agent Workflow 编排 |
| LangChain | 0.3+ | LLM 框架 |
| DeepSeek | - | 核心推理模型 |
| OpenAI | - | 备用推理模型 |
| Tavily | 0.5+ | 网络搜索 |
| GitHub API | - | 开源项目搜索 |
| Pydantic | 2.x | 数据模型 |
| LangSmith | 0.3+ | 调试与追踪 |

### 项目结构

```
AI_Career_Research_Agent/
│
├── src/open_deep_research/
│   ├── configuration.py        # 配置管理（含求职专用配置）
│   ├── deep_researcher.py      # LangGraph 主工作流
│   ├── career_prompts.py       # 求职专用 Prompt 模板
│   ├── prompts.py              # 通用 Prompt 模板
│   ├── state.py                # Agent 状态管理（含求职状态模型）
│   └── utils.py                # 工具函数（含 GitHub 搜索、技能分析）
│
├── tests/                      # 测试目录
├── .env.example                # 环境变量示例
├── pyproject.toml              # 项目配置
├── langgraph.json              # LangGraph 部署配置
└── README.md                   # 项目文档
```

---

## 📦 安装与运行

### 环境要求

- Python 3.11（推荐，与 `langgraph.json` 声明一致）
- [uv](https://github.com/astral-sh/uv)（推荐，提供 `uvx` 工具运行器）
- 或 pip / poetry

### 安装依赖

```bash
# 使用 pip
pip install -e .

# 或使用 poetry
poetry install
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

配置以下关键环境变量：

```env
# 模型 API Key（至少配置一个）
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# 搜索 API Key
TAVILY_API_KEY=your_tavily_api_key

# GitHub API Key（可选，用于项目搜索）
GITHUB_API_KEY=your_github_api_key

# LangSmith 追踪（推荐）
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
```

### 运行方式

推荐使用 `uvx` 以**免 Docker**的本地开发模式（`langgraph-cli[inmem]`）启动，它会自动构建隔离环境并加载当前项目：

```bash
# 推荐：使用 uvx 启动本地开发服务器（内存态，无需 Docker）
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

> 说明：
> - `--from "langgraph-cli[inmem]"` 启用内存态服务器，**不需要 Docker**；
> - `--with-editable .` 将当前项目以可编辑方式安装，使 `open_deep_research` 包可被导入；
> - `--python 3.11` 锁定解释器版本（与 `langgraph.json` 一致）；
> - `--allow-blocking` 允许 Agent 调用同步/阻塞型工具（如 Tavily、requests）；
> - `--refresh` 忽略缓存、强制刷新依赖。
>
> 启动后访问 http://localhost:8123 即可在 LangGraph Studio 中可视化调试。

如果你是先在本地用 pip 安装好依赖的，也可以直接运行（同样需 `langgraph-cli[inmem]`）：

```bash
# 已 pip install -e . 后，直接启动（仍需 [inmem] 以避免 Docker）
langgraph dev --allow-blocking
```

> ⚠️ 注意：旧文档中的 `langgraph up` 会启动完整 Platform 并**依赖 Docker**，本地无 Docker 环境会失败；顶层 `await` 的 `python -c` 写法在 `python -c` 下属语法错误，请勿使用。

---

## 📚 RAG 私有知识库（可选）

将**私有资料**（个人简历、目标公司 JD、面经、学习资源库等）构建为本地向量索引，研究员在调研时可优先检索这些资料，作为联网搜索的补充，从而给出更贴合你个人背景的回答。

### 1. 准备资料
把文档（`.md` / `.txt` / `.json` / `.html` / `.pdf`）放入 `data/` 目录。仓库已附带一个示例 `data/简历.md`。

### 2. 构建索引
```bash
# 默认使用 OpenAI embedding（需 OPENAI_API_KEY）
python -m open_deep_research.ingest --src ./data --index-path ./data/rag_index

# 也可使用本地离线 embedding（需先: pip install sentence-transformers）
python -m open_deep_research.ingest --src ./data --embedding BAAI/bge-small-zh-v1.5
```

### 3. 开启 RAG
在 `.env` 中设置：
```bash
RAG_ENABLED=true
RAG_EMBEDDING_MODEL=openai:text-embedding-3-small
RAG_INDEX_PATH=./data/rag_index
RAG_TOP_K=4
```
或在 LangGraph Studio 的配置面板里打开 `rag_enabled`。开启后，`retrieve_knowledge_base` 工具会自动注入到研究员工具集中。

### 4. 重新启动
```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

> 说明：RAG 索引建立在本地 Chroma，每次 `data/` 内容变化后需重新运行 `ingest` 脚本刷新。

### 5. （可选）使用 Supabase 远端向量库
默认 `RAG_VECTOR_STORE=chroma` 使用本地文件。若想让知识库上云、可多人共享，可切换为 Supabase（Postgres + pgvector）：
1. 在 Supabase 项目中启用 **pgvector** 扩展（`CREATE EXTENSION IF NOT EXISTS vector;`）；
2. 在 `.env` 设置 `SUPABASE_CONNECTION_STRING=postgresql://user:pass@host:5432/postgres`（项目的 Postgres 连接串）；
3. 设 `RAG_VECTOR_STORE=supabase`；
4. 构建索引时加 `--vector-store supabase`：
```bash
python -m open_deep_research.ingest --src ./data --vector-store supabase
```
> 这样就把"本地文件知识库"升级为"可远端访问的数据库向量库"，呼应本项目的可插拔存储设计。

## 🧑‍💼 求职背景资料（简历画像）

为了让调研真正"懂你"，Agent 会自动加载你的背景资料并注入到所有研究员的提示中（supervisor 与子研究员都会看到）：

- **默认来源**：`./data/简历.md`（仓库已附带示例）。把你的真实简历覆盖该文件即可。
- **覆盖方式**：在 LangGraph Studio 的 `CareerConfig` 配置面板里直接粘贴 `user_profile` 文本，优先级高于简历文件。
- 注入后，研究员在规划研究方向、撰写报告时会结合你的技能栈、经验与目标岗位，给出更贴合个人情况的建议。

> 结合上一项 RAG 能力：简历进知识库（RAG 检索）→ 简历进提示（画像注入），两套机制互补，分别解决"检索私有资料"与"理解你是谁"。

## 🎯 岗位匹配度分析（Gap Analysis）

当你在对话中给出某个具体岗位的 JD（职位描述）并想了解"我和这个岗位匹配吗 / 我还差什么"时，研究员会自动调用 `analyze_job_fit` 工具，结合你的简历/背景资料做**人岗匹配度分析**，输出：

- **综合匹配度评分（0-100）** 与一句话结论；
- **✅ 匹配的优势**：你已有的、与岗位吻合的技能/经验；
- **❌ 差距与缺失（Skill Gap）**：岗位要求但你欠缺的点（标注严重/中等/轻微）；
- **🚀 针对性提升路径**：最短路径补哪块短板、中期学什么、如何包装现有经历。

无需额外配置，默认开启（`CareerConfig.enable_gap_analysis`）。例如在 Studio 中直接提问：

> "这是我想投的 AI 工程师 JD：<粘贴 JD 文本>，分析一下我和它的匹配度与差距。"

## 📝 使用示例

### 输入示例

```text
我想应聘上海的 AI 工程师岗位，我有3年工作经验，本科毕业，会 Python、PyTorch 和机器学习。
```

### 输出报告结构

生成的求职分析报告包含以下章节：

1. **岗位市场分析** — 目标岗位的市场需求概况、薪资水平、行业趋势
2. **技能要求分析** — 核心技能要求清单、技能匹配度分析、差距优先级排序
3. **学习路线规划** — 分阶段学习计划、推荐学习资源、里程碑项目
4. **简历优化建议** — 关键词优化、项目描述改进、格式建议
5. **面试准备方案** — 高频面试题、行为面试题、编程题
6. **GitHub 项目推荐** — 推荐项目列表、贡献建议
7. **总结与行动建议** — 短期/中期/长期行动计划

---

## 🔧 配置说明

### 求职专用配置项

在 `.env` 或运行时配置中可以设置以下求职专用参数：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| DEFAULT_TARGET_ROLE | AI工程师 | 默认目标岗位 |
| DEFAULT_TARGET_CITY | 北京 | 默认目标城市 |
| DEFAULT_EDUCATION | 本科 | 默认学历 |
| DEFAULT_EXPERIENCE_YEARS | 3年 | 默认工作经验 |
| DEFAULT_SKILLS | Python, PyTorch, TensorFlow | 默认技能列表 |
| ENABLE_GITHUB_SEARCH | true | 是否启用 GitHub 搜索 |
| MAX_GITHUB_RESULTS | 10 | GitHub 项目最大返回数 |
| SKILL_MATCH_THRESHOLD | 0.6 | 技能匹配阈值 |

### LangGraph UI 配置

项目支持通过 LangGraph UI 进行可视化配置，所有配置项均已添加 `x_oap_ui_config` 元数据，可在 UI 中直接调整。

---

## 🧪 测试

```bash
# 运行全部测试
pytest tests/

# 运行特定测试
pytest tests/test_deep_researcher.py

# 运行求职功能专属测试（无需 API Key，LLM 调用已 mock）
pytest tests/test_career_features.py -q
```

### 求职功能测试覆盖

`tests/test_career_features.py` 对本次新增的求职能力做了单元测试与集成测试，**默认不依赖任何 API Key**：

| 测试对象 | 覆盖内容 |
| --- | --- |
| `profile.load_user_profile` | 内联画像优先于简历文件、缺失文件回退为空 |
| `utils.calculate_skill_match_score` | 满分 / 空输入 / 部分匹配（matched ÷ required） |
| `utils.extract_user_profile_from_messages` | 从用户消息抽取目标岗位、城市、技能 |
| `CareerConfig` / `Configuration` | 默认值、从 `configurable` 构建 |
| `utils.get_all_tools` | 按开关注入 `search_github_projects` / `analyze_job_fit` / `retrieve_knowledge_base` |
| `gap_analysis.run_gap_analysis` | 无画像时返回提示；有画像时用 mock LLM 验证四段式输出 |
| `rag.RAGStore` | Chroma 未构建 / Supabase 未配置连接串时的兜底提示 |
| Chroma 端到端检索 | 真实建索引 + 检索（需 `langchain_chroma`，缺失时自动跳过） |

### LangSmith 评估（可选）

`tests/evaluators_career.py` 提供了一组 LLM-as-judge / 启发式评估器，可接入 LangSmith 的 `client.evaluate(...)`，对「岗位匹配度分析」「求职调研」等输出做长期质量追踪：

```python
from langsmith import Client
from tests.evaluators_career import career_evaluators

client.evaluate(
    dataset_name="career-gap-analysis",   # 在 LangSmith 中创建的评估数据集
    evaluators=career_evaluators,
)
```

评估指标包括：匹配度分析结构完整度、是否给出量化匹配分、画像覆盖度、是否真正调用了私有知识库（RAG）。设置 `LANGSMITH_API_KEY` 与 `LANGSMITH_TRACING=true` 即可启用追踪。

---

## 📋 开发计划

- [x] 项目架构分析与设计
- [x] 求职专用 Prompt 重构
- [x] 求职 Workflow 重构（6个核心功能节点）
- [x] 岗位搜索模块（Tavily 搜索集成）
- [x] 技能差距分析模块
- [x] 学习路线规划模块
- [x] 简历优化建议模块
- [x] 面试准备方案模块
- [x] GitHub 项目推荐模块
- [x] 求职专用配置项
- [x] 项目文档完善
- [ ] Web 前端界面优化
- [ ] Docker 部署配置
- [ ] API 接口封装
- [ ] 用户记忆功能（Memory）

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南

1. Fork 本仓库
2. 创建功能分支
3. 提交代码
4. 创建 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

---

## 🙏 致谢

本项目基于 [LangGraph Open Deep Research](https://github.com/langchain-ai/open-deep-research) 项目进行二次开发，感谢 LangChain 团队的开源贡献。
