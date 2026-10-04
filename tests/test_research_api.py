"""研究接口（/api/research、/api/research/stream）的单测。

通过 mock ``research_service.get_research_graph`` 规避对完整 LangGraph 重依赖的
导入，验证：请求解析、最终答案抽取、thread_id 透传、SSE result 事件、消息上限。

改造前 patch 的是 ``api_module._get_research_graph``（api 模块里的私有函数），
现在研究图归位到 ``app.services.research_service``，patch 目标随之改变 ——
这也是重构的可见收益：依赖点从"某个大模块的内部细节"变成"服务层的公开接口"。
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

from app.services import research_service  # noqa: E402

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


def _fake_graph(instance):
    """构造一个 async 的 get_research_graph 替身（新签名是 async def）。"""

    async def _getter(settings=None):
        return instance

    return _getter


@pytest.fixture
def client():
    with patch.object(
        research_service, "get_research_graph", new=_fake_graph(FakeGraph())
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
    # 每个响应都带 request_id，便于和日志对齐
    assert r.headers.get("X-Request-ID")


def test_research_thread_id_passed_to_config(client):
    captured: dict = {}
    with patch.object(
        research_service,
        "get_research_graph",
        new=_fake_graph(FakeGraph(captured)),
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
    # 关掉代理缓冲，否则流式会退化成"等很久然后一次性吐出"
    assert r.headers.get("X-Accel-Buffering") == "no"


def test_research_message_limit(client):
    msgs = [{"role": "user", "content": "x"}] * 51
    r = client.post("/api/research", json={"messages": msgs})
    assert r.status_code == 413
    body = r.json()
    # 统一错误体：可编程的 title + 可关联的 request_id
    assert body["title"] == "PAYLOAD_TOO_LARGE"
    assert body["request_id"]


def test_research_timeout_maps_to_504():
    """上游超时应映射为 504（可重试），而不是含异常原文的 500。"""
    import asyncio

    from app.core.errors import UpstreamTimeoutError, is_retryable

    class SlowGraph:
        async def ainvoke(self, inputs, config=None):
            await asyncio.sleep(0.5)
            return {"messages": []}

    with pytest.raises(UpstreamTimeoutError) as excinfo:
        asyncio.run(
            research_service.run_research(
                SlowGraph(), {"messages": []}, {}, timeout=0.01
            )
        )
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "UPSTREAM_TIMEOUT"
    assert is_retryable(excinfo.value) is True
