"""研究接口（/api/research、/api/research/stream）的单测。

通过 mock ``_get_research_graph`` 规避对完整 LangGraph 重依赖的导入，
验证：请求解析、最终答案抽取、thread_id 透传、SSE result 事件、消息上限。
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: E402

import open_deep_research.api as api_module  # noqa: E402


class FakeGraph:
    def __init__(self, captured=None):
        self._captured = captured if captured is not None else {}

    async def ainvoke(self, inputs, config=None):
        self._captured["config"] = config
        return {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="最终答案XYZ"),
            ]
        }

    async def astream(self, inputs, config=None, stream_mode="messages"):
        yield AIMessageChunk(content="最终"), {
            "langgraph_node": "final_report_generation"
        }
        yield AIMessageChunk(content="答案XYZ"), {
            "langgraph_node": "final_report_generation"
        }


@pytest.fixture
def client():
    with patch.object(
        api_module, "_get_research_graph", return_value=FakeGraph()
    ):
        yield TestClient(api_module.app)


def test_research_returns_final_answer(client):
    r = client.post(
        "/api/research", json={"messages": [{"role": "user", "content": "测试"}]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["final_answer"] == "最终答案XYZ"
    assert body["messages"][-1]["type"] == "AIMessage"


def test_research_thread_id_passed_to_config(client):
    captured: dict = {}
    with patch.object(
        api_module, "_get_research_graph", return_value=FakeGraph(captured)
    ):
        client.post(
            "/api/research",
            json={
                "messages": [{"role": "user", "content": "x"}],
                "thread_id": "conv-123",
            },
        )
    assert captured["config"]["configurable"]["thread_id"] == "conv-123"


def test_research_stream_emits_result_event(client):
    r = client.post(
        "/api/research/stream",
        json={"messages": [{"role": "user", "content": "测试"}]},
    )
    assert r.status_code == 200
    text = r.text
    # 流式结尾应补发完整最终答案，且以 [DONE] 结束
    assert "最终答案XYZ" in text
    assert '"type": "result"' in text
    assert "[DONE]" in text


def test_research_message_limit(client):
    msgs = [{"role": "user", "content": "x"}] * 51
    r = client.post("/api/research", json={"messages": msgs})
    assert r.status_code == 413
