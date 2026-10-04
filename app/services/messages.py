"""HTTP 消息 ↔ LangChain 消息的转换（无 FastAPI 依赖）。

这一层被"抽出来"的原因是它原本写在 ``api.py`` 里并直接 ``raise HTTPException``，
于是"一次转换"既包含协议转换、又包含 HTTP 状态码语义。现在它只抛类型化错误，
由 ``errors.register_exception_handlers`` 统一翻译成响应体。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from app.core.errors import PayloadTooLargeError

DEFAULT_MAX_MESSAGES = 50

_ROLE_MAP = {
    "user": "human",
    "human": "human",
    "assistant": "ai",
    "ai": "ai",
    "system": "system",
    "tool": "tool",
}

FINAL_REPORT_NODE = "final_report_generation"


def build_runnable_config(
    configurable: Optional[Dict[str, Any]],
    thread_id: Optional[str],
    recursion_limit: int = 50,
) -> RunnableConfig:
    """组装 LangGraph 的 ``RunnableConfig``。"""
    cfg = dict(configurable or {})
    if thread_id:
        cfg["thread_id"] = thread_id
    cfg.setdefault("recursion_limit", recursion_limit)
    return {"configurable": cfg}


def parse_messages(
    messages: List[Any], max_messages: int = DEFAULT_MAX_MESSAGES
) -> List[Any]:
    """把接口消息转换为 LangChain ``BaseMessage`` 列表。

    直接构造消息对象，避免 ``messages_from_dict`` 对 ``data`` 字段的依赖。
    超出上限抛 ``PayloadTooLargeError``，由全局处理器翻译为 413。
    """
    if max_messages > 0 and len(messages) > max_messages:
        raise PayloadTooLargeError(
            f"消息数量超出上限（最大 {max_messages} 条，实际 {len(messages)} 条）"
        )

    out: List[Any] = []
    for idx, item in enumerate(messages):
        role = _ROLE_MAP.get(getattr(item, "role", "user"), "human")
        content = getattr(item, "content", "")
        if role == "ai":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=f"api-{idx}"))
        else:
            out.append(HumanMessage(content=content))
    return out


def content_to_text(content: Any) -> str:
    """将消息 content（可能是多模态 list）规整为纯文本。"""
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def final_content(result: Dict[str, Any]) -> str:
    """从图结果里取"最终答案"。

    若最后一条是用户/系统消息（说明这一轮没有新增 AI 回复），回退到最近一条
    AI 消息；否则取最后一条消息的文本。
    """
    msgs = result.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    if isinstance(last, (HumanMessage, SystemMessage)):
        for message in reversed(msgs):
            if isinstance(message, AIMessage):
                return content_to_text(message.content)
        return ""
    return content_to_text(last.content)


def chunk_to_event(chunk: Any, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把流式 chunk 转成 SSE 事件体。"""
    return {
        "type": type(chunk).__name__,
        "content": chunk.content,
        "name": getattr(chunk, "name", None),
        "tool_call_chunks": getattr(chunk, "tool_call_chunks", None),
        "langgraph_node": (meta or {}).get("langgraph_node"),
    }


def is_final_report_chunk(chunk: Any, meta: Optional[Dict[str, Any]]) -> bool:
    """判断该 chunk 是否属于最终报告节点（用于无需额外 LLM 调用地拼出完整报告）。"""
    return (meta or {}).get("langgraph_node") == FINAL_REPORT_NODE and isinstance(
        chunk, AIMessageChunk
    )
