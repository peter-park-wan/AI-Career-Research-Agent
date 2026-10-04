"""Unit & integration tests for the AI Career Research Agent's job-seeking features.

These tests are intentionally dependency-light and API-key-free:
  * Pure logic (profile resolution, skill-score, tool wiring, RAG backend
    selection) runs anywhere `open_deep_research` imports.
  * The LLM-backed gap analysis is exercised with a mocked chat model.
  * The end-to-end Chroma retrieval test is gated behind
    `pytest.importorskip("langchain_chroma")` so it runs in a fully provisioned
    environment and is skipped cleanly otherwise.

Run with:
    python -m pytest tests/test_career_features.py -q
"""

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

# Make the local package importable when the project is not installed editable.
# tests/test_career_features.py -> project root is two levels up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from open_deep_research import configuration as configuration_module  # noqa: E402
from open_deep_research import gap_analysis, profile, rag, utils  # noqa: E402
from open_deep_research.configuration import CareerConfig, Configuration  # noqa: E402
from open_deep_research.utils import (  # noqa: E402
    calculate_skill_match_score,
    extract_user_profile_from_messages,
    get_all_tools,
)


# --------------------------------------------------------------------------- #
# 1. Resume / profile loading
# --------------------------------------------------------------------------- #
def test_load_user_profile_inline_priority():
    # Inline profile must win over the resume file on disk.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", encoding="utf-8", delete=False
    ) as f:
        f.write("# Resume on disk\nShould NOT be returned.")
        path = f.name
    try:
        cfg = {
            "configurable": {
                "career_config": {
                    "user_profile": "  我有 3 年 Python 后端经验  ",
                    "resume_path": path,
                }
            }
        }
        assert profile.load_user_profile(cfg) == "我有 3 年 Python 后端经验"
    finally:
        os.unlink(path)


def test_load_user_profile_resume_file():
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", encoding="utf-8", delete=False
    ) as f:
        f.write("  简历文件内容  ")
        path = f.name
    try:
        cfg = {"configurable": {"career_config": {"resume_path": path}}}
        assert profile.load_user_profile(cfg) == "简历文件内容"
    finally:
        os.unlink(path)


def test_load_user_profile_no_career_config(tmp_path, monkeypatch):
    """未配置 career_config 时回退到默认简历路径 ``./data/简历.md``。

    这里显式隔离工作目录：默认路径是相对进程 cwd 的 ``./data/简历.md``，
    原实现假设该文件"必然不存在"，于是开发者一旦按文档把简历放好，
    这个测试就会误报失败（测试结论依赖本机环境而不是代码行为）。
    """
    monkeypatch.chdir(tmp_path)
    # 默认路径不存在 → 继续回退到长期记忆 → 空串
    assert profile.load_user_profile({}) == ""
    assert profile.load_user_profile({"configurable": {}}) == ""

    # 按既定约定：把简历放到默认路径即自动生效
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "简历.md").write_text("默认路径简历", encoding="utf-8")
    assert profile.load_user_profile({}) == "默认路径简历"


def test_load_user_profile_missing_file_falls_back_to_empty():
    cfg = {
        "configurable": {
            "career_config": {"resume_path": "/no/such/file.md"}
        }
    }
    assert profile.load_user_profile(cfg) == ""


# --------------------------------------------------------------------------- #
# 2. Pure scoring / extraction helpers
# --------------------------------------------------------------------------- #
def test_calculate_skill_match_score_full_and_partial():
    # Signature takes comma-separated strings, not lists.
    assert calculate_skill_match_score("python", "python") == 1.0
    assert calculate_skill_match_score("", "python") == 0.0
    # score = matched / len(required): user covers 1 of 2 required skills -> 0.5
    assert calculate_skill_match_score("python", "python,go") == pytest.approx(0.5)


def test_extract_user_profile_from_messages():
    class _Msg:
        def __init__(self, content):
            self.content = content

    messages = [
        _Msg("我想找算法工程师岗位，base 在深圳。"),
        _Msg("技能：Java、SQL、Docker。"),
    ]
    result = extract_user_profile_from_messages(messages)
    # "算法工程师" is the only job title long enough to win the longest-match rule
    assert result["target_role"] == "算法工程师"
    assert result["target_city"] == "深圳"
    assert "Java" in result["current_skills"]


# --------------------------------------------------------------------------- #
# 3. CareerConfig defaults & wiring
# --------------------------------------------------------------------------- #
def test_career_config_defaults():
    c = CareerConfig()
    assert c.enable_github_search is True
    assert c.enable_gap_analysis is True
    assert c.resume_path == "./data/简历.md"


def test_from_runnable_config_builds_career_config():
    cfg = Configuration.from_runnable_config(
        {"configurable": {"career_config": {"enable_gap_analysis": False}}}
    )
    assert isinstance(cfg.career_config, CareerConfig)
    assert cfg.career_config.enable_gap_analysis is False
    # unset field keeps its default
    assert cfg.career_config.enable_github_search is True


# --------------------------------------------------------------------------- #
# 4. Tool injection into the researcher toolset
# --------------------------------------------------------------------------- #
def _tool_names(overrides):
    cfg = {"configurable": dict(overrides)}
    return {t.name for t in asyncio.run(get_all_tools(cfg))}


def test_get_all_tools_injects_career_tools():
    names = _tool_names(
        {
            "search_api": "none",
            "rag_enabled": True,
            "career_config": {"enable_github_search": True, "enable_gap_analysis": True},
        }
    )
    assert "search_github_projects" in names
    assert "analyze_job_fit" in names
    assert "retrieve_knowledge_base" in names


def test_get_all_tools_respects_career_flags():
    names = _tool_names(
        {
            "search_api": "none",
            "rag_enabled": False,
            "career_config": {
                "enable_github_search": False,
                "enable_gap_analysis": False,
            },
        }
    )
    assert "search_github_projects" not in names
    assert "analyze_job_fit" not in names
    assert "retrieve_knowledge_base" not in names


def test_get_all_tools_no_career_config_means_no_career_tools():
    names = _tool_names({"search_api": "none", "rag_enabled": False})
    assert "search_github_projects" not in names
    assert "analyze_job_fit" not in names


def test_analyze_job_fit_is_a_tool():
    assert gap_analysis.analyze_job_fit.name == "analyze_job_fit"


# --------------------------------------------------------------------------- #
# 5. Gap analysis (LLM mocked)
# --------------------------------------------------------------------------- #
def test_run_gap_analysis_no_profile_returns_warning():
    result = asyncio.run(
        gap_analysis.run_gap_analysis(
            "JD 内容",
            config={"configurable": {"career_config": {"resume_path": "/no/file.md"}}},
        )
    )
    assert "未找到用户背景资料" in result


def test_run_gap_analysis_with_mocked_llm():
    from langchain_core.messages import AIMessage

    sample = (
        "综合匹配度: 82/100\n"
        "匹配优势: 具备 Python 与 RAG 经验\n"
        "Skill Gap: 缺少大规模分布式训练经验\n"
        "提升路径: 补强分布式系统，参与相关项目\n"
    )
    fake_model = AsyncMock()
    fake_model.ainvoke.return_value = AIMessage(content=sample)

    with patch(
        "open_deep_research.gap_analysis.init_chat_model", return_value=fake_model
    ), patch(
        "open_deep_research.gap_analysis.get_api_key_for_model",
        return_value="test-key",
    ):
        result = asyncio.run(
            gap_analysis.run_gap_analysis(
                "招聘 AI 工程师，要求 Python、RAG 经验",
                config={
                    "configurable": {
                        "career_config": {"user_profile": "我精通 Python 与 RAG"}
                    }
                },
            )
        )

    assert "综合匹配度" in result
    assert "提升路径" in result
    # The mocked model must have been invoked (i.e. the happy path ran).
    assert fake_model.ainvoke.called


# --------------------------------------------------------------------------- #
# 6. RAG store backend selection / guard rails
# --------------------------------------------------------------------------- #
def _dummy_embedding():
    # Only needs to exist; backend-selection tests short-circuit before use.
    return object()


def test_rag_store_chroma_not_built_returns_message():
    with tempfile.TemporaryDirectory() as d:
        store = rag.RAGStore(
            embedding_model=_dummy_embedding(),
            index_path=os.path.join(d, "does_not_exist"),
            vector_store="chroma",
        )
        out = store.retrieve("任意查询")
    assert "尚未构建" in out


def test_rag_store_supabase_no_connection_returns_message():
    store = rag.RAGStore(
        embedding_model=_dummy_embedding(),
        vector_store="supabase",
        connection_string=None,
    )
    assert "SUPABASE_CONNECTION_STRING" in store.retrieve("任意查询")


def test_rag_store_build_routes_to_supabase_requires_connection():
    pytest.importorskip("langchain_postgres")
    from langchain_core.documents import Document

    store = rag.RAGStore(
        embedding_model=_dummy_embedding(),
        vector_store="supabase",
        connection_string=None,
    )
    with pytest.raises(RuntimeError):
        store.build([Document(page_content="x")])


# --------------------------------------------------------------------------- #
# 7. End-to-end Chroma retrieval (skipped if deps absent)
# --------------------------------------------------------------------------- #
def test_rag_store_build_and_retrieve_chroma(tmp_path):
    pytest.importorskip("langchain_chroma")
    from langchain_core.documents import Document

    class _HashEmbeddings:
        """Deterministic, dependency-free embedding for the test only."""

        dim = 32

        def _vec(self, text):
            vec = [0.0] * self.dim
            for ch in text:
                vec[hash(ch) % self.dim] += 1.0
            return vec

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    # 用 pytest 的 tmp_path，而不是 tempfile.TemporaryDirectory()：
    # Chroma 会一直握着索引文件（chroma.sqlite3 / data_level0.bin）的句柄，
    # 在 Windows 上 TemporaryDirectory 退出时 rmtree 会撞 PermissionError
    # (WinError 32) —— 断言其实已经通过了，失败只发生在清理阶段。
    # tmp_path 由 pytest 统一管理，清理失败只警告、不判定测试失败。
    store = rag.RAGStore(
        embedding_model=_HashEmbeddings(),
        index_path=str(tmp_path),
        collection="test_career_kb",
        top_k=4,
        vector_store="chroma",
    )
    store.build(
        [
            Document(
                page_content="字节跳动 AI 工程师 JD 要求 Python 与 RAG。",
                metadata={"source": "字节跳动_JD.md"},
            ),
            Document(
                page_content="我的简历：3 年 Python 后端，做过 RAG 项目。",
                metadata={"source": "简历.md"},
            ),
        ]
    )
    out = store.retrieve("字节跳动 AI 工程师 要求", k=4)

    assert "字节跳动_JD.md" in out
    assert "简历.md" in out
    assert "[来源" in out
