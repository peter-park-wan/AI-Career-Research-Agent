"""End-to-end career job-search workflow orchestration.

This module ties the individual career capabilities (job discovery, research,
gap analysis, interview prep, resume optimization, cover letter) into one
coherent pipeline:

    岗位发现 -> 岗位/公司调研 -> 匹配度分析 -> 面试准备 -> 简历优化 -> 求职信

Two ways to use it:

1. **As agent tools** — :func:`discover_jobs`, :func:`prepare_interview`,
   :func:`optimize_resume`, :func:`write_cover_letter` and the one-shot
   :func:`run_career_workflow` are LangChain ``@tool``s. They are injected into
   the researcher's toolkit (behind ``CareerConfig.enable_career_workflow``) so
   the multi-agent research graph can drive the whole job-search flow itself.
2. **As a library / CLI entry** — call :func:`run_career_workflow` directly to
   produce a structured :class:`CareerWorkflowResult` (every artifact + a single
   markdown report) for surfacing to the user.

Every LLM call goes through :func:`_llm_completion`, which mirrors
``gap_analysis``'s pattern (``init_chat_model`` + ``get_api_key_for_model``), so
the whole module is trivially mockable in tests.
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.gap_analysis import run_gap_analysis
from open_deep_research.profile import load_user_profile
from open_deep_research.utils import get_api_key_for_model


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
DISCOVER_SYSTEM = """你是资深的职业顾问与技术招聘官。基于候选人的背景画像与目标城市，
推荐最适合优先投递的岗位方向与公司类型（无需真实职位链接，给出方向性建议即可）。
用中文，结构化输出："""
RESEARCH_SYSTEM = """你是行业研究分析师。针对一个目标岗位，产出该岗位的「硬技能 / 软技能 /
面试重点 / 行业趋势」洞察，帮助候选人有的放矢地准备。用中文，结构化输出："""
INTERVIEW_SYSTEM = """你是资深技术面试官与招聘顾问。基于候选人的目标岗位、背景与匹配度分析，
产出一份可直接用于备考的「面试准备清单」。用中文，结构化输出："""
RESUME_SYSTEM = """你是顶级简历顾问（简历优化专家）。基于候选人的背景与求职目标，
给出可落地的简历优化建议，并改写出 3-5 条更贴合目标岗位的核心经历 bullet。用中文，结构化输出："""
COVER_LETTER_SYSTEM = """你是专业的求职信撰写顾问。基于候选人的背景与目标岗位，
写一封真诚、具体、不套话的中文求职信（800 字以内）。用中文输出，直接给正文。"""


# --------------------------------------------------------------------------- #
# Shared LLM helper
# --------------------------------------------------------------------------- #
async def _llm_completion(
    system: str, user: str, config: RunnableConfig, temperature: float = 0.3
) -> str:
    """Call the configured chat model and return the textual completion."""
    configurable = Configuration.from_runnable_config(config)
    model_name = configurable.final_report_model
    api_key = get_api_key_for_model(model_name, config)
    model = init_chat_model(model_name, api_key=api_key, temperature=temperature)
    try:
        response = await model.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
    except Exception as e:  # noqa: BLE001 - surface a friendly error to the agent
        return f"⚠️ 调用模型失败：{e}"
    return response.content


# --------------------------------------------------------------------------- #
# Individual stages
# --------------------------------------------------------------------------- #
async def run_discover_jobs(
    profile: str, role: str, city: str, config: RunnableConfig
) -> str:
    """推荐优先投递的岗位方向与公司类型。"""
    user = (
        f"# 候选人背景\n{profile or '（未提供，请基于常识给通用建议）'}\n\n"
        f"# 求职目标\n岗位方向：{role}\n目标城市：{city}\n\n"
        "请给出：① 3-5 个最匹配的岗位方向；② 每类岗位推荐的公司类型/典型雇主；"
        "③ 优先级排序与理由。"
    )
    return await _llm_completion(DISCOVER_SYSTEM, user, config, temperature=0.3)


async def run_research_insights(
    role: str, city: str, config: RunnableConfig
) -> str:
    """产出目标岗位的技能/面试重点/趋势洞察。"""
    user = f"目标岗位：{role}\n目标城市：{city}\n\n请输出该岗位的硬技能、软技能、面试重点与行业趋势。"
    return await _llm_completion(RESEARCH_SYSTEM, user, config, temperature=0.2)


async def run_interview_prep(
    role: str, profile: str, gap_analysis: str, config: RunnableConfig
) -> str:
    """基于岗位、背景与差距分析，产出面试准备清单。"""
    user = (
        f"# 目标岗位\n{role}\n\n"
        f"# 候选人背景\n{profile or '（未提供）'}\n\n"
        f"# 匹配度分析\n{gap_analysis or '（未提供）'}\n\n"
        "请输出：① 高频技术面试题（含参考答案要点）；② 项目深挖清单（STAR 准备）；"
        "③ 行为/综合面试问题；④ 临门一脚复习路线。"
    )
    return await _llm_completion(INTERVIEW_SYSTEM, user, config, temperature=0.3)


async def run_resume_optimization(
    role: str, profile: str, gap_analysis: str, config: RunnableConfig
) -> str:
    """基于目标岗位与差距，产出简历优化建议 + 改写 bullet。"""
    user = (
        f"# 求职目标岗位\n{role}\n\n"
        f"# 候选人当前背景\n{profile or '（未提供）'}\n\n"
        f"# 匹配度分析\n{gap_analysis or '（未提供）'}\n\n"
        "请输出：① 整体简历优化策略；② 3-5 条更贴合目标岗位的核心经历 bullet（改写版）；"
        "③ 需要弱化/删除的内容建议。"
    )
    return await _llm_completion(RESUME_SYSTEM, user, config, temperature=0.3)


async def run_cover_letter(
    role: str, profile: str, config: RunnableConfig
) -> str:
    """写一封贴合目标岗位的中文求职信。"""
    user = (
        f"# 目标岗位\n{role}\n\n"
        f"# 候选人背景\n{profile or '（未提供，请基于常识撰写通用版）'}\n\n"
        "请写一封真诚、具体、不套话的中文求职信（800 字以内），直接给正文。"
    )
    return await _llm_completion(COVER_LETTER_SYSTEM, user, config, temperature=0.4)


# --------------------------------------------------------------------------- #
# Structured result
# --------------------------------------------------------------------------- #
class CareerWorkflowResult(BaseModel):
    """Artifacts produced by the end-to-end career workflow."""

    user_profile: str = ""
    target_role: str = ""
    target_city: str = ""
    discovery: str = ""
    research: str = ""
    gap_analysis: str = ""
    interview_prep: str = ""
    resume_optimization: str = ""
    cover_letter: str = ""
    report: str = ""


def _section(title: str, body: str) -> str:
    if not body:
        return ""
    return f"## {title}\n\n{body.strip()}\n\n"


async def run_career_workflow(
    target_role: str = "",
    target_city: str = "",
    jd_text: str = "",
    config: RunnableConfig = None,
) -> CareerWorkflowResult:
    """Run the full job-search pipeline and return every artifact.

    Args:
        target_role: Optional target role (falls back to ``CareerConfig``).
        target_city: Optional target city (falls back to ``CareerConfig``).
        jd_text: Optional concrete JD. When provided, gap analysis runs against
            it and the discovery/research stages are skipped.
        config: LangChain runnable config (carries ``career_config`` etc.).
    """
    configurable = Configuration.from_runnable_config(config)
    career = configurable.career_config
    profile = load_user_profile(config)

    role = target_role or (career.default_target_role if career else "") or "AI工程师"
    city = target_city or (career.default_target_city if career else "") or "北京"

    result = CareerWorkflowResult(
        user_profile=profile, target_role=role, target_city=city
    )

    # Stage 1 + 2: discovery & research (skipped when a concrete JD is supplied)
    if jd_text.strip():
        result.gap_analysis = await run_gap_analysis(jd_text, config)
    else:
        result.discovery = await run_discover_jobs(profile, role, city, config)
        result.research = await run_research_insights(role, city, config)
        # Gap analysis against the discovered role direction
        result.gap_analysis = await run_gap_analysis(
            f"目标岗位：{role}（{city}）\n{result.research}", config
        )

    # Stage 4-6: interview prep, resume optimization, cover letter
    result.interview_prep = await run_interview_prep(
        role, profile, result.gap_analysis, config
    )
    result.resume_optimization = await run_resume_optimization(
        role, profile, result.gap_analysis, config
    )
    result.cover_letter = await run_cover_letter(role, profile, config)

    result.report = (
        f"# 🧭 求职工作流报告：{role}（{city}）\n\n"
        f"> 基于以下候选人背景生成：\n\n{_section('候选人背景', profile)}"
        f"{_section('① 岗位发现', result.discovery)}"
        f"{_section('② 岗位调研洞察', result.research)}"
        f"{_section('③ 匹配度分析（Gap Analysis）', result.gap_analysis)}"
        f"{_section('④ 面试准备', result.interview_prep)}"
        f"{_section('⑤ 简历优化', result.resume_optimization)}"
        f"{_section('⑥ 求职信', result.cover_letter)}"
    ).strip()
    return result


# --------------------------------------------------------------------------- #
# Agent tools (injected when enable_career_workflow is True)
# --------------------------------------------------------------------------- #
@tool("discover_jobs")
async def discover_jobs(
    role: str = "", city: str = "", config: RunnableConfig = None
) -> str:
    """推荐优先投递的岗位方向与公司类型（岗位发现）。

    当用户在做求职规划、想了解「我适合投哪些岗位 / 哪些公司」时调用。
    可传入目标岗位 role 与城市 city；缺省时自动使用 CareerConfig 中的默认值。
    返回结构化的岗位方向、推荐公司类型与优先级建议。
    """
    configurable = Configuration.from_runnable_config(config)
    career = configurable.career_config
    role = role or (career.default_target_role if career else "") or "AI工程师"
    city = city or (career.default_target_city if career else "") or "北京"
    return await run_discover_jobs(load_user_profile(config), role, city, config)


@tool("prepare_interview")
async def prepare_interview(config: RunnableConfig = None) -> str:
    """基于目标岗位与匹配度分析，生成面试准备清单（高频面试题 / 项目深挖 / 复习路线）。

    当用户已进入求职准备阶段、需要针对性面试备战材料时调用。
    工具会自动结合用户的简历/背景资料与目标岗位（来自 CareerConfig），
    以及已有的匹配度分析结果，产出可直接使用的备考清单。
    """
    configurable = Configuration.from_runnable_config(config)
    career = configurable.career_config
    role = (career.default_target_role if career else "") or "AI工程师"
    profile = load_user_profile(config)
    gap = await run_gap_analysis(
        f"目标岗位：{role}\n（用于面试准备，请基于背景做匹配度判断）", config
    )
    return await run_interview_prep(role, profile, gap, config)


@tool("optimize_resume")
async def optimize_resume(config: RunnableConfig = None) -> str:
    """针对目标岗位优化简历，给出策略 + 改写后的核心经历 bullet。

    当用户希望让简历更贴合某个目标岗位（如 AI 工程师）时调用。
    工具结合用户简历/背景资料、目标岗位与匹配度分析，产出优化建议与改写示例。
    """
    configurable = Configuration.from_runnable_config(config)
    career = configurable.career_config
    role = (career.default_target_role if career else "") or "AI工程师"
    profile = load_user_profile(config)
    gap = await run_gap_analysis(
        f"目标岗位：{role}\n（用于简历优化，请基于背景做匹配度判断）", config
    )
    return await run_resume_optimization(role, profile, gap, config)


@tool("write_cover_letter")
async def write_cover_letter(config: RunnableConfig = None) -> str:
    """为目标岗位写一封真诚、具体的中文求职信。

    当用户需要针对目标岗位生成求职信时调用。工具结合用户简历/背景资料
    与目标岗位（来自 CareerConfig），产出 800 字以内的中文求职信正文。
    """
    configurable = Configuration.from_runnable_config(config)
    career = configurable.career_config
    role = (career.default_target_role if career else "") or "AI工程师"
    return await run_cover_letter(role, load_user_profile(config), config)


@tool("run_career_workflow")
async def run_career_workflow_tool(
    target_role: str = "",
    target_city: str = "",
    jd_text: str = "",
    config: RunnableConfig = None,
) -> str:
    """一键跑通端到端求职工作流：岗位发现 → 调研 → 匹配度 → 面试准备 → 简历优化 → 求职信。

    当用户希望一次性获得完整的求职作战方案（含各阶段产物）时调用。
    返回一份整合了全部阶段产物的 Markdown 报告。
    """
    result = await run_career_workflow(
        target_role=target_role,
        target_city=target_city,
        jd_text=jd_text,
        config=config,
    )
    return result.report
