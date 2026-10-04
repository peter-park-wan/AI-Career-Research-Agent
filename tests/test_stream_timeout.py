"""SSE 超时的计时口径：只算研究耗时，不算客户端读取耗时。

【要防的事故】

改造前 ``stream_research_events`` 用 ``asyncio.timeout(timeout)`` 包住整个
``async for``，而这个循环体里有 ``yield``：

    async with asyncio.timeout(timeout):
        async for chunk, meta in graph.astream(...):
            yield sse_event(...)      # ← 这一段也被计进 timeout

``yield`` 的阻塞时长取决于**客户端多快把响应体读走**。弱网、反向代理缓冲、
移动端切后台、用户不看着页面 —— 都会让它阻塞很久。于是：

* 研究本身可能只用了 1 分钟，预算是 3 分钟；
* 但用户总共花了 3 分半把流读完；
* 结果用户收到"研究执行超时"，而研究早就跑完了。

危害不止是误报一次：这条错误会**误导排查方向**。看到"研究超时"，
正常反应是去调大模型、加并发、砍搜索轮次 —— 而真正的瓶颈在网络。

修复后的口径：``timeout`` 只累计"等待研究产出下一个 chunk"的时间，
``yield`` 给客户端的时间一律不计。

本文件用假图把两种耗时分离开，分别验证：
* 客户端慢（读取耗时长）→ 不超时；
* 研究慢（产出间隔长）→ 照常超时。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List

from langchain_core.messages import AIMessageChunk

from app.services import research_service

FINAL = {"langgraph_node": "final_report_generation"}


class FakeGraph:
    """按脚本产出 chunk 的假图。

    ``think`` 模拟"研究产出下一个 chunk 需要的真实耗时"，
    与消费者读取速度完全解耦 —— 正是这个解耦让两类耗时能被分别断言。
    """

    def __init__(self, chunks: List[Any], think: float = 0.0) -> None:
        self._chunks = chunks
        self._think = think
        self.closed = False

    def astream(self, inputs: Any, config: Any = None, stream_mode: Any = None) -> Any:
        return self._generate()

    async def _generate(self) -> Any:
        try:
            for chunk, meta in self._chunks:
                if self._think:
                    await asyncio.sleep(self._think)  # 研究在推进，耗费真实时间
                yield chunk, meta
        finally:
            self.closed = True


def _report_chunks(count: int = 3) -> List[Any]:
    return [(AIMessageChunk(content=f"片段{i}"), FINAL) for i in range(count)]


def _parse(events: List[str]) -> List[dict]:
    """把 SSE 文本块还原成事件体，[DONE] 也转成一个可断言的字典。"""
    parsed: List[dict] = []
    for raw in events:
        body = raw[len("data: ") :].strip()
        parsed.append({"type": "[DONE]"} if body == "[DONE]" else json.loads(body))
    return parsed


def _collect(graph: FakeGraph, timeout: int, read_delay: float = 0.0) -> List[dict]:
    """消费整个流；``read_delay`` 模拟慢客户端逐条读取的耗时。"""

    async def consume() -> List[str]:
        events: List[str] = []
        async for event in research_service.stream_research_events(
            graph, {"messages": []}, {}, timeout
        ):
            events.append(event)
            if read_delay:
                await asyncio.sleep(read_delay)  # 客户端读得慢
        return events

    return _parse(asyncio.run(consume()))


def test_slow_client_does_not_consume_research_budget():
    """核心用例：读取耗时不算研究耗时。

    研究瞬时完成，但客户端每条读 0.6s，三条共 1.8s —— 远超 1s 预算。
    改造前这会报"研究执行超时"，而研究其实一秒内就跑完了。
    """
    events = _collect(FakeGraph(_report_chunks(3)), timeout=1, read_delay=0.6)

    assert not [e for e in events if e["type"] == "error"]
    result = [e for e in events if e["type"] == "result"]
    assert result, "慢客户端不应导致研究被判超时"
    assert result[0]["content"] == "片段0片段1片段2"


def test_slow_research_still_times_out():
    """研究本身慢时，超时判定必须照常生效 —— 修复不能把真超时也放过去。"""
    # 每个 chunk 前研究要"思考"0.6s，预算 1s，累计到第二个 chunk 就该耗尽
    graph = FakeGraph(_report_chunks(5), think=0.6)
    events = _collect(graph, timeout=1)

    errors = [e for e in events if e["type"] == "error"]
    assert errors, "研究本身超时必须被报出来"
    assert "超时" in errors[0]["content"]
    # 超时意味着没跑完，不该再发 result
    assert not [e for e in events if e["type"] == "result"]


def test_timeout_closes_upstream_stream():
    """超时只取消了"等下一个 chunk"；必须主动关流，否则后台 LLM 还在烧 token。"""
    graph = FakeGraph(_report_chunks(5), think=0.6)
    _collect(graph, timeout=1)
    assert graph.closed, "超时后底层研究流仍开着，会在后台继续消耗算力"


def test_zero_timeout_means_unlimited():
    """``timeout=0`` 表示不限时（与改造前语义一致，防止改动引入回归）。"""
    events = _collect(FakeGraph(_report_chunks(3), think=0.3), timeout=0, read_delay=0.2)

    assert not [e for e in events if e["type"] == "error"]
    assert [e for e in events if e["type"] == "result"]


def test_stream_always_ends_with_done():
    """无论正常结束还是异常，都要以 [DONE] 收尾，否则前端 EventSource 会一直等。"""
    for graph, timeout in (
        (FakeGraph(_report_chunks(2)), 10),
        (FakeGraph(_report_chunks(5), think=0.6), 1),
    ):
        assert _collect(graph, timeout)[-1]["type"] == "[DONE]"


def test_client_disconnect_does_not_raise():
    """客户端中途断开时，``finally`` 里的 yield 会抛
    ``RuntimeError: async generator ignored GeneratorExit``。

    用户刷新页面、关标签页都会触发，服务端日志会被这类噪音刷满。
    """

    async def consume_then_drop() -> None:
        stream = research_service.stream_research_events(
            FakeGraph(_report_chunks(5)), {"messages": []}, {}, 0
        )
        await stream.__anext__()  # 读一条
        await stream.aclose()  # 客户端断开

    asyncio.run(consume_then_drop())
