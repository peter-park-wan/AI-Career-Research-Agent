"""Tests for the report reflection (Critique -> Revise) node.

No real LLM calls: the critic model and writer model are mocked. The node is
exercised directly via ``reflect_and_revise_report``.

Run with:
    python -m pytest tests/test_reflection.py -q
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from open_deep_research.configuration import Configuration  # noqa: E402
from open_deep_research.deep_researcher import (  # noqa: E402
    ReportCritique,
    reflect_and_revise_report,
)


BASE_STATE = {
    "research_brief": "研究字节跳动 AI 工程师岗位",
    "final_report": "字节跳动 AI 工程师要求 PyTorch。",
    "notes": ["字节跳动 AI 工程师岗位要求掌握 PyTorch 与分布式训练"],
}


def _cfg(reflection_rounds: int):
    return {
        "configurable": {
            "user_id": "u",
            "max_reflection_rounds": reflection_rounds,
            "final_report_model": "openai:gpt-4o",
            "final_report_model_max_tokens": 4096,
            "max_structured_output_retries": 1,
        }
    }


def _mock_models(needs_revision=True, revised_text="修订后的报告"):
    """Build a mock ``configurable_model`` reproducing production behavior.

    - ``.with_structured_output(ReportCritique)`` -> critic returning a critique
    - ``.with_config(...)`` -> writer whose ``.ainvoke`` returns revised text
    """
    critique = ReportCritique(
        needs_revision=needs_revision,
        coverage_gaps="缺少薪资信息" if needs_revision else "",
        factual_issues="",
        citation_issues="",
        revision_suggestions="补充薪资范围" if needs_revision else "",
    )
    writer = MagicMock()
    writer.ainvoke = AsyncMock(return_value=AIMessage(content=revised_text))

    # Mirror the node's call chain:
    #   model.with_structured_output(ReportCritique)   -> critic_runnable
    #   critic_runnable.with_retry(...).with_config(...) -> critique_model
    #   await critique_model.ainvoke(...)               -> critique
    model = MagicMock()
    model.with_structured_output.return_value.with_retry.return_value.with_config.return_value.ainvoke = AsyncMock(
        return_value=critique
    )
    model.with_config.return_value = writer
    return model


@patch("open_deep_research.deep_researcher.get_api_key_for_model", return_value="fake-key")
def test_reflection_disabled_passthrough(_):
    model = _mock_models()
    with patch("open_deep_research.deep_researcher.configurable_model", model):
        out = asyncio.run(reflect_and_revise_report(BASE_STATE, _cfg(0)))
    # Disabled -> report unchanged, no revision triggered
    assert out == {}
    model.with_structured_output.assert_not_called()


@patch("open_deep_research.deep_researcher.get_api_key_for_model", return_value="fake-key")
def test_reflection_revises_when_needed(_):
    model = _mock_models(needs_revision=True, revised_text="修订后的报告含薪资")
    with patch("open_deep_research.deep_researcher.configurable_model", model):
        out = asyncio.run(reflect_and_revise_report(BASE_STATE, _cfg(1)))
    assert "修订后的报告含薪资" in out["final_report"]
    assert out["reflection_summary"]  # critique recorded


@patch("open_deep_research.deep_researcher.get_api_key_for_model", return_value="fake-key")
def test_reflection_stops_when_satisfied(_):
    # Critic says no revision needed -> keep original, writer never invoked
    model = _mock_models(needs_revision=False)
    with patch("open_deep_research.deep_researcher.configurable_model", model):
        out = asyncio.run(reflect_and_revise_report(BASE_STATE, _cfg(3)))
    assert "要求 PyTorch" in out["final_report"]  # unchanged
    # The writer's ainvoke must NOT have been called when no revision is needed
    assert model.with_config.return_value.ainvoke.call_count == 0


@patch("open_deep_research.deep_researcher.get_api_key_for_model", return_value="fake-key")
def test_reflection_critic_failure_is_safe(_):
    # If critique raises, node should not crash and should keep the draft
    model = MagicMock()
    critique_model = MagicMock()
    critique_model.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    model.with_structured_output.return_value.with_retry.return_value = critique_model
    model.with_config.return_value = MagicMock()

    with patch("open_deep_research.deep_researcher.configurable_model", model):
        out = asyncio.run(reflect_and_revise_report(BASE_STATE, _cfg(1)))
    assert "要求 PyTorch" in out["final_report"]


def test_configuration_default_reflection_rounds():
    cfg = Configuration()
    assert cfg.max_reflection_rounds == 1
