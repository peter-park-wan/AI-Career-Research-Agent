"""Long-term memory layer for the AI Career Research Agent.

This module implements a three-tier memory architecture on top of LangGraph's
``Store`` abstraction (``langgraph.store``):

1. **User profile memory** (``(user_id, "profile")`` namespace)
   Cross-session, structured facts about the job seeker (skills, target role,
   applied companies, salary expectation, ...). Persisted so the agent "remembers"
   who you are across separate research runs.

2. **Research knowledge memory** (``("career-knowledge", role)`` namespace)
   Semantic index of past research findings / report chunks so the agent can
   recall prior conclusions instead of re-searching the web every time.

3. **RAG private knowledge base** (see ``rag.py`` + ``retrieve_knowledge_base``)
   Vector retrieval over the user's own resume / JDs / interview notes.

The store is obtained via ``langgraph.config.get_store()`` when the graph is
compiled with a ``store=`` argument (LangGraph platform or local ``compile(store=...)``).
A process-local fallback (``InMemoryStore``) is provided for standalone scripts
and tests that do not configure a platform store.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.store.base import BaseStore

from open_deep_research.configuration import Configuration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store acquisition
# ---------------------------------------------------------------------------
def get_memory_store(config: Optional[RunnableConfig] = None) -> Optional[BaseStore]:
    """Return the long-term ``Store`` for the current run.

    Priority:
        1. The platform/local store injected via ``langgraph.config.get_store()``
           (requires the graph to be compiled with ``store=...``).
        2. A module-level ``InMemoryStore`` fallback so memory works in scripts/tests.
    """
    try:
        store = get_store()
        if store is not None:
            return store
    except Exception:  # noqa: BLE001 - get_store raises if no store configured
        pass

    return _fallback_store()


_FALLBACK: Optional[BaseStore] = None


def _fallback_store() -> BaseStore:
    """Lazily build an in-process ``InMemoryStore`` (no embeddings by default)."""
    global _FALLBACK
    if _FALLBACK is None:
        from langgraph.store.memory import InMemoryStore

        _FALLBACK = InMemoryStore(index=None)
        logger.warning(
            "未检测到 LangGraph 平台 Store，使用进程内 InMemoryStore（重启即丢失）。"
            "生产环境请在 compile 时传入 store 或使用 PostgresStore。"
        )
    return _FALLBACK


def _user_id(config: Optional[RunnableConfig]) -> str:
    """Resolve a stable user id from runnable config."""
    if config:
        cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
        uid = cfg.get("user_id")
        if uid:
            return str(uid)
    return "anonymous"


# ---------------------------------------------------------------------------
# Tier 1: User profile memory
# ---------------------------------------------------------------------------
def load_user_memory(config: Optional[RunnableConfig]) -> str:
    """Recall the user's persisted profile as an injectable text block.

    Returns empty string when no memory exists (e.g. first run).
    """
    store = get_memory_store(config)
    if store is None:
        return ""

    uid = _user_id(config)
    try:
        items = store.search((uid, "profile"))
    except Exception as e:  # noqa: BLE001
        logger.warning("读取用户长期记忆失败: %s", e)
        return ""

    if not items:
        return ""

    blocks = []
    for it in items:
        val = it.value
        if isinstance(val, dict):
            # Pretty-print structured profile facts
            lines = "\n".join(f"- {k}: {v}" for k, v in val.items() if v)
            if lines:
                blocks.append(lines)
        elif isinstance(val, str) and val.strip():
            blocks.append(val.strip())
    return "\n".join(blocks)


async def save_user_memory(
    facts: dict,
    config: Optional[RunnableConfig] = None,
) -> None:
    """Persist structured user-profile facts into long-term memory.

    ``facts`` is a flat dict (e.g. ``{"target_role": "AI 工程师",
    "skills": ["Python", "PyTorch"], "applied": ["字节跳动"]}``). Values may be
    scalars or lists. Lists are joined into a readable string.
    """
    store = get_memory_store(config)
    if store is None or not facts:
        return

    # Normalize list values into strings for clean retrieval
    normalized = {}
    for k, v in facts.items():
        if isinstance(v, (list, tuple, set)):
            normalized[k] = ", ".join(str(x) for x in v) if v else ""
        else:
            normalized[k] = v

    uid = _user_id(config)
    try:
        await store.aput((uid, "profile"), "facts", normalized)
        logger.info("已保存用户长期记忆 (user=%s)", uid)
    except Exception as e:  # noqa: BLE001
        logger.warning("保存用户长期记忆失败: %s", e)


# ---------------------------------------------------------------------------
# Tier 2: Research knowledge memory (semantic recall)
# ---------------------------------------------------------------------------
async def save_research_memory(
    role: str,
    summary: str,
    config: Optional[RunnableConfig] = None,
) -> None:
    """Index a chunk of research findings under a role namespace for later recall."""
    store = get_memory_store(config)
    if store is None or not summary or not role:
        return
    try:
        await store.aput(
            ("career-knowledge", role),
            f"{role}-{abs(hash(summary))}",
            {"summary": summary, "role": role},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("保存研究知识记忆失败: %s", e)


def recall_research_memory(
    role: str,
    query: str,
    k: int = 3,
    config: Optional[RunnableConfig] = None,
) -> str:
    """Recall prior research findings for ``role`` relevant to ``query``."""
    store = get_memory_store(config)
    if store is None or not role:
        return ""
    try:
        items = store.search(("career-knowledge", role), query=query, limit=k)
    except Exception as e:  # noqa: BLE001
        logger.warning("召回研究知识记忆失败: %s", e)
        return ""

    if not items:
        return ""
    blocks = []
    for it in items:
        val = it.value
        text = val.get("summary") if isinstance(val, dict) else str(val)
        if text:
            blocks.append(text)
    return "\n\n".join(blocks)
