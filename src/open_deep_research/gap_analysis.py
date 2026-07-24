"""Job-fit / Gap Analysis for career research.

Compares the user's background (resume / inline profile) against a target
job description (JD) and returns a structured fit analysis: an overall match
score, matched strengths, skill gaps, and a concrete upskilling path.

Exposed as the ``analyze_job_fit`` tool so researchers can invoke it whenever
the user supplies a concrete JD and wants to know how they measure up.
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from open_deep_research.configuration import Configuration
from open_deep_research.profile import load_user_profile
from open_deep_research.utils import get_api_key_for_model

GAP_SYSTEM_PROMPT = """你是一位资深的招聘顾问与技术招聘官，擅长做「人岗匹配度分析」。

你会拿到两部分内容：
1. 候选人的背景资料（简历 / 自我介绍）
2. 一个目标岗位的 JD（职位描述）

请基于这两份材料，给出一份**结构化、可操作**的匹配度分析，严格按以下格式输出（使用中文）：

## 综合匹配度
- 评分：<0-100 的整数> / 100
- 一句话结论：<例如「整体匹配度较高，但在 X 方向存在明显短板」>

## ✅ 匹配的优势
- <列出候选人已有的、与岗位要求吻合的技能 / 经验 / 项目，每条一句>

## ❌ 差距与缺失（Skill Gap）
- <列出岗位要求但候选人明显欠缺或薄弱的点，每条一句，可标注「严重/中等/轻微」>

## 🚀 针对性提升路径
1. <最短路径：优先补哪个短板收益最大>
2. <中期：建议学习的技能 / 项目 / 证书>
3. <如何把现有经历包装得更贴合该岗位>

要求：结论必须基于两份材料的事实，不要凭空编造技能；差距要具体、可执行。
"""


async def run_gap_analysis(jd_text: str, config: RunnableConfig) -> str:
    """Run the fit analysis for ``jd_text`` against the user's profile."""
    configurable = Configuration.from_runnable_config(config)
    user_profile = load_user_profile(config)

    if not user_profile.strip():
        return (
            "⚠️ 未找到用户背景资料，无法进行匹配度分析。"
            "请在配置中设置 resume_path（指向简历文件）或在 CareerConfig 中粘贴 user_profile 文本。"
        )

    model_name = configurable.summarization_model
    api_key = get_api_key_for_model(model_name, config)
    model = init_chat_model(model_name, api_key=api_key, temperature=0)

    messages = [
        SystemMessage(content=GAP_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"# 候选人背景资料\n{user_profile}\n\n"
                f"# 目标岗位 JD\n{jd_text}\n\n"
                "请输出上述结构的匹配度分析。"
            )
        ),
    ]
    try:
        response = await model.ainvoke(messages)
    except Exception as e:  # noqa: BLE001 - surface a friendly error to the agent
        return f"⚠️ 匹配度分析调用模型失败：{e}"
    return response.content


@tool("analyze_job_fit")
async def analyze_job_fit(jd_description: str, config: RunnableConfig = None) -> str:
    """分析候选人与给定岗位的匹配度（Gap Analysis / 人岗匹配）。

    当用户提供了某个具体岗位的 JD（职位描述）文本，并想了解自己与该岗位的
    匹配程度、技能差距或提升路径时，调用此工具。工具会自动结合用户的简历 /
    背景资料（resume_path 或 user_profile），输出：综合匹配度评分（0-100）、
    匹配的优势技能、差距技能（Skill Gap），以及针对性提升路径。

    Args:
        jd_description: 目标岗位的 JD 描述文本（职位要求、职责等）。
    """
    return await run_gap_analysis(jd_description, config)
