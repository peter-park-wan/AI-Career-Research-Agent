"""FastAPI 服务的冒烟测试（无需 API Key，LLM 调用已 mock）。

运行::

    pytest tests/test_api.py -q

若环境中未安装 fastapi / httpx，该测试会自动跳过。
"""
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
