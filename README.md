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

- Python 3.10+
- pip 或 poetry

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

```bash
# 方式1：使用 langgraph-cli
langgraph up

# 方式2：Python 脚本运行
python -c "from open_deep_research.deep_researcher import deep_researcher; await deep_researcher.ainvoke({'messages': [('human', '我想找AI工程师岗位，在上海')]})"
```

---

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
# 运行测试
pytest tests/

# 运行特定测试
pytest tests/test_deep_researcher.py
```

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
