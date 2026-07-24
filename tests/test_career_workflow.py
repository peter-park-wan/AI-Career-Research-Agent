"""Tests for the end-to-end career workflow orchestration.

All LLM calls are mocked via ``unittest.mock``, so the pipeline runs without
any API key. We verify: stage wiring, the JD short-circuit, the one-shot tool,
and tool injection into the researcher toolkit.

Run with:
    python -m pytest tests/test_career_workflow.py -q
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from langchain_core.messages import AIMessage  # noqa: E402

from open_deep_research import career_workflow  # noqa: E402
from open_deep_research.utils import get_all_tools  # noqa: E402


def _patch_llm(content="STAGE_OUTPUT"):
    """Patch both career_workflow and gap_analysis LLM entry points."""
    fake = AsyncMock()
    fake.ainvoke.return_value = AIMessage(content=content)
    return patch.multiple(
        "open_deep_research.career_workflow",
        init_chat_model=lambda *a, **k: fake,
        get_api_key_for_model=lambda *a, **k: "test-key",
    ), patch.multiple(
        "open_deep_research.gap_analysis",
        init_chat_model=lambda *a, **k: fake,
        get_api_key_for_model=lambda *a, **k: "test-key",
    ), fake


def test_run_career_workflow_orchestrates_all_stages():
    p1, p2, fake = _patch_llm()
    cfg = {"configurable": {"career_config": {"user_profile": "我精通 Python 与 RAG"}}}
    with p1, p2:
        result = asyncio.run(career_workflow.run_career_workflow(config=cfg))

    assert result.target_role and result.target_city
    # No JD supplied -> discovery + research run, all six artifacts populated.
    assert result.discovery and result.research
    assert result.gap_analysis and result.interview_prep
    assert result.resume_optimization and result.cover_letter
    # The combined report contains every section header.
    for header in ["① 岗位发现", "② 岗位调研洞察", "③ 匹配度分析",
                   "④ 面试准备", "⑤ 简历优化", "⑥ 求职信"]:
        assert header in result.report
    # 6 LLM stages (discovery, research, gap, interview, resume, cover letter).
    assert fake.ainvoke.call_count == 6


def test_run_career_workflow_with_jd_skips_discovery():
    p1, p2, fake = _patch_llm()
    cfg = {"configurable": {"career_config": {"user_profile": "我精通 Python"}}}
    with p1, p2:
        result = asyncio.run(
            career_workflow.run_career_workflow(
                jd_text="招聘 AI 工程师，要求 Python、RAG。", config=cfg
            )
        )
    # A concrete JD short-circuits discovery & research.
    assert result.discovery == ""
    assert result.research == ""
    assert result.gap_analysis  # gap runs directly against the JD
    assert result.interview_prep and result.resume_optimization and result.cover_letter


def test_workflow_tools_are_registered():
    names = [
        career_workflow.discover_jobs.name,
        career_workflow.prepare_interview.name,
        career_workflow.optimize_resume.name,
        career_workflow.write_cover_letter.name,
        career_workflow.run_career_workflow_tool.name,
    ]
    assert names == [
        "discover_jobs",
        "prepare_interview",
        "optimize_resume",
        "write_cover_letter",
        "run_career_workflow",
    ]


def test_one_shot_workflow_tool_returns_report():
    p1, p2, _ = _patch_llm()
    cfg = {"configurable": {"career_config": {"user_profile": "我精通 Python"}}}
    with p1, p2:
        # Async @tool must be invoked via .ainvoke (config is injected automatically).
        report = asyncio.run(career_workflow.run_career_workflow_tool.ainvoke({}, config=cfg))
    assert "求职工作流报告" in report


async def _tool_names(overrides):
    cfg = {"configurable": dict(overrides)}
    return {t.name for t in await get_all_tools(cfg)}


def test_get_all_tools_injects_workflow_tools():
    names = asyncio.run(
        _tool_names(
            {
                "search_api": "none",
                "rag_enabled": False,
                "career_config": {"enable_career_workflow": True},
            }
        )
    )
    for n in ["discover_jobs", "prepare_interview", "optimize_resume",
              "write_cover_letter", "run_career_workflow"]:
        assert n in names


def test_get_all_tools_respects_workflow_flag():
    names = asyncio.run(
        _tool_names(
            {
                "search_api": "none",
                "rag_enabled": False,
                "career_config": {"enable_career_workflow": False},
            }
        )
    )
    assert "discover_jobs" not in names
    assert "run_career_workflow" not in names
