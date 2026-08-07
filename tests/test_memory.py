"""Tests for the three-tier long-term memory layer.

All LangChain/LangGraph network calls are avoided; we use an in-process
``InMemoryStore`` as the backing store and patch ``langgraph.config.get_store``
so ``memory.py`` resolves the store deterministically. LLM calls inside
``extract_and_save_memory`` are mocked via ``unittest.mock``.

Run with:
    python -m pytest tests/test_memory.py -q
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

from langgraph.store.memory import InMemoryStore  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

import open_deep_research.memory as memory  # noqa: E402
from open_deep_research.configuration import Configuration  # noqa: E402


CONFIG = {"configurable": {"user_id": "test_user"}}


@pytest.fixture
def store():
    """Provide a deterministic in-process store and patch ``get_store``."""
    s = InMemoryStore(index=None)
    with patch.object(memory, "_fallback_store", return_value=s), patch(
        "langgraph.config.get_store", return_value=s
    ):
        yield s


async def _save_profile(store, cfg=CONFIG):
    await memory.save_user_memory(
        {
            "target_role": "AI 工程师",
            "skills": ["Python", "PyTorch"],
            "applied": ["字节跳动"],
        },
        cfg,
    )


def test_save_and_recall_user_profile(store):
    asyncio.run(_save_profile(store))
    recalled = memory.load_user_memory(CONFIG)
    assert "AI 工程师" in recalled
    assert "PyTorch" in recalled
    assert "字节跳动" in recalled


def test_recall_returns_empty_when_no_memory(store):
    # Store is fresh and nothing was written for this user
    recalled = memory.load_user_memory({"configurable": {"user_id": "nobody"}})
    assert recalled == ""


def test_save_research_memory_and_semantic_recall(store):
    async def _run():
        await memory.save_research_memory(
            "general", "字节跳动 AI 工程师岗位要求掌握 PyTorch 与分布式训练", CONFIG
        )
        return memory.recall_research_memory("general", "字节跳动 岗位要求", k=1, config=CONFIG)

    out = asyncio.run(_run())
    assert "PyTorch" in out


def test_recall_research_empty_when_missing(store):
    out = memory.recall_research_memory("general", "anything", config=CONFIG)
    assert out == ""


def test_user_id_fallback_to_anonymous(store):
    # No user_id in config -> "anonymous" namespace, should not raise
    async def _run():
        await memory.save_user_memory({"target_role": "后端工程师"}, {"configurable": {}})
        return memory.load_user_memory({"configurable": {}})

    out = asyncio.run(_run())
    assert "后端工程师" in out


def test_extract_and_save_memory_node(store):
    """``extract_and_save_memory`` persists profile facts without real LLM."""
    from open_deep_research.deep_researcher import (
        UserProfileFacts,
        extract_and_save_memory,
    )

    # Build a SYNC mock chain mirroring how a real (model-bound) ConfigurableModel
    # behaves in production: with_structured_output/with_retry/with_config are
    # synchronous, only the final ``.ainvoke`` is awaited.
    fake_facts = UserProfileFacts(
        target_role="AI 工程师",
        skills=["Python"],
        applied_companies=["字节跳动"],
    )
    configger = MagicMock()
    configger.ainvoke = AsyncMock(return_value=fake_facts)
    retrier = MagicMock()
    retrier.with_config.return_value = configger
    struct = MagicMock()
    struct.with_retry.return_value = retrier
    fake_model = MagicMock()
    fake_model.with_structured_output.return_value = struct

    state = {
        "messages": [
            HumanMessage(content="我想做 AI 工程师，会 Python，投过字节跳动"),
        ],
        "final_report": "字节跳动 AI 工程师要求 PyTorch。",
        "research_brief": "字节跳动 AI 工程师调研",
    }

    with patch(
        "open_deep_research.deep_researcher.configurable_model", fake_model
    ):
        asyncio.run(extract_and_save_memory(state, CONFIG))

    recalled = memory.load_user_memory(CONFIG)
    assert "AI 工程师" in recalled
    assert "字节跳动" in recalled


def test_configuration_rag_force_field():
    cfg = Configuration(rag_enabled=True, rag_force=True, rag_top_k=4)
    assert cfg.rag_enabled is True
    assert cfg.rag_force is True
    assert cfg.rag_top_k == 4


def test_rag_force_false_by_default():
    cfg = Configuration()
    assert cfg.rag_force is False
