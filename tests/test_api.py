"""FastAPI 服务的冒烟测试（无需 API Key，LLM 调用已 mock）。

运行::

    pytest tests/test_api.py -q

若环境中未安装 fastapi / httpx，该测试会自动跳过。
"""
import asyncio
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402

from open_deep_research import api as api_module  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # 屏蔽 LLM 调用
    async def fake_llm(*args, **kwargs):
        return "【mock】LLM 返回内容"

    async def fake_gap(*args, **kwargs):
        return "【mock】匹配度分析结果"

    monkeypatch.setattr(
        "open_deep_research.career_workflow._llm_completion", fake_llm
    )
    # gap_analysis.run_gap_analysis 既被 API 懒加载引用，也被 career_workflow 顶层导入引用
    monkeypatch.setattr(
        "open_deep_research.gap_analysis.run_gap_analysis", fake_gap
    )
    monkeypatch.setattr(
        "open_deep_research.career_workflow.run_gap_analysis", fake_gap
    )
    return TestClient(api_module.app)


def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "endpoints" in body

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_profile_no_llm(client):
    r = client.get("/api/profile")
    assert r.status_code == 200
    assert "profile" in r.json()


def test_discover_jobs(client):
    r = client.post(
        "/api/career/discover-jobs",
        json={"target_role": "AI工程师", "target_city": "深圳"},
    )
    assert r.status_code == 200
    assert "discovery" in r.json()


def test_gap_analysis(client):
    r = client.post(
        "/api/career/gap-analysis",
        json={"jd_text": "招聘 AI 工程师，要求 Python、PyTorch。"},
    )
    assert r.status_code == 200
    assert "gap_analysis" in r.json()


def test_career_workflow(client):
    r = client.post(
        "/api/career/workflow",
        json={"target_role": "AI工程师", "target_city": "北京"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["report"]
    assert "cover_letter" in body["artifacts"]


def test_gap_analysis_cache_dedupes(monkeypatch):
    """相同输入只跑一次 run_gap_analysis（#5 省去重复 LLM 开销）。"""
    calls = {"n": 0}

    async def counting_gap(jd_text, config):
        calls["n"] += 1
        return "GAP_RESULT"

    monkeypatch.setattr(
        "open_deep_research.gap_analysis.run_gap_analysis", counting_gap
    )
    api_module._GAP_CACHE.clear()
    cfg = {"configurable": {"user_profile": "张三"}}
    r1 = asyncio.run(api_module.run_gap_analysis_cached("JD-A", cfg))
    r2 = asyncio.run(api_module.run_gap_analysis_cached("JD-A", cfg))
    # 不同输入应再次触发
    r3 = asyncio.run(api_module.run_gap_analysis_cached("JD-B", cfg))
    assert (r1, r2, r3) == ("GAP_RESULT", "GAP_RESULT", "GAP_RESULT")
    assert calls["n"] == 2
    api_module._GAP_CACHE.clear()


def test_rate_limit(monkeypatch):
    """API_RATE_LIMIT_PER_MINUTE>0 时，超出窗口请求返回 429（#3）。"""
    monkeypatch.setattr(api_module, "_RATE_LIMIT", 1)
    api_module._RATE_BUCKETS.clear()
    cli = TestClient(api_module.app)
    assert cli.get("/api/profile").status_code == 200
    assert cli.get("/api/profile").status_code == 429
    api_module._RATE_BUCKETS.clear()
    monkeypatch.setattr(api_module, "_RATE_LIMIT", 0)


def test_checkpointer_fallback(monkeypatch):
    """未配置连接串的 postgres/redis 安全回退到内存 checkpointer（#2）。"""
    from langgraph.checkpoint.memory import MemorySaver

    monkeypatch.delenv("API_CHECKPOINTER_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("API_CHECKPOINTER_REDIS_URI", raising=False)
    monkeypatch.delenv("REDIS_URI", raising=False)

    assert api_module._build_checkpointer("none") is None
    assert isinstance(api_module._build_checkpointer("memory"), MemorySaver)
    assert isinstance(api_module._build_checkpointer("postgres"), MemorySaver)
    assert isinstance(api_module._build_checkpointer("redis"), MemorySaver)
