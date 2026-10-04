"""深度研究接口：同步返回与 SSE 流式输出。

流式响应额外补了两个头部，都是踩过的坑：

* ``Cache-Control: no-cache`` —— 否则中间的反向代理可能缓冲整个响应，
  用户等了 3 分钟才发现"没有流式效果"。
* ``X-Accel-Buffering: no`` —— 明确告知 Nginx 关闭缓冲（Nginx 不认 Cache-Control
  对 ``text/event-stream`` 的处理）。
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.config import Settings
from app.core.deps import get_settings_dep, require_token
from app.schemas.requests import ResearchRequest
from app.services import research_service
from app.services.messages import (
    build_runnable_config,
    final_content,
    parse_messages,
)

router = APIRouter(prefix="/api", tags=["research"], dependencies=[Depends(require_token)])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/research")
async def research(
    body: ResearchRequest,
    settings: Settings = Depends(get_settings_dep),
) -> Dict[str, Any]:
    """运行 LangGraph 深度研究图，返回最终消息列表与答案。"""
    graph = await research_service.get_research_graph(settings)
    cfg = build_runnable_config(body.configurable, body.thread_id, body.recursion_limit)
    parsed = parse_messages(body.messages, settings.max_messages)

    result = await research_service.run_research(
        graph, {"messages": parsed}, cfg, settings.research_timeout
    )

    msgs = result.get("messages", [])
    return {
        "thread_id": body.thread_id,
        "messages": [{"type": type(m).__name__, "content": m.content} for m in msgs],
        "final_answer": final_content(result),
    }


@router.post("/research/stream")
async def research_stream(
    body: ResearchRequest,
    settings: Settings = Depends(get_settings_dep),
) -> StreamingResponse:
    """以 SSE 流式返回研究图执行过程（LLM token 与工具调用）。"""
    graph = await research_service.get_research_graph(settings)
    cfg = build_runnable_config(body.configurable, body.thread_id, body.recursion_limit)
    # 注意：入参校验（如消息条数超限）必须发生在返回 StreamingResponse 之前，
    # 否则错误只能以 SSE 事件的形式发出，HTTP 状态码仍是 200。
    parsed = parse_messages(body.messages, settings.max_messages)

    return StreamingResponse(
        research_service.stream_research_events(
            graph, {"messages": parsed}, cfg, settings.research_timeout
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
