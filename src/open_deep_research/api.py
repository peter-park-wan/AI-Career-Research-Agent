"""FastAPI 服务：将 AI Career Research Agent 暴露为 HTTP 接口。

提供两类入口：
1. 通用深度研究图（LangGraph ``deep_researcher``）—— ``POST /api/research``，
   支持同步返回与 SSE 流式输出。
2. 求职场景单点能力 —— 岗位发现、匹配度分析、面试准备、简历优化、求职信、
   以及端到端工作流 ``POST /api/career/workflow``。

启动方式::

    uvicorn open_deep_research.api:app --host 0.0.0.0 --port 8000

或::

    python -m open_deep_research.api
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import nullcontext as _nullcontext
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

app = FastAPI(
    title="AI Career Research Agent API",
    version="1.0.0",
    description="基于 LangGraph 多 Agent 工作流的智能求职研究助手 HTTP 接口",
)

# --------------------------------------------------------------------------- #
# CORS & 可选鉴权
# --------------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("API_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 若设置了 API_BEARER_TOKEN，则所有 /api/* 接口需携带 Authorization: Bearer <token>
_API_TOKEN = os.getenv("API_BEARER_TOKEN")


async def require_token(request: Request) -> None:
    if not _API_TOKEN:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[len("Bearer "):] != _API_TOKEN:
        raise HTTPException(status_code=401, detail="无效或缺失的 Bearer Token")


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class MessageItem(BaseModel):
    role: str = Field(..., description="user / assistant / system / tool")
    content: Any = Field(..., description="文本内容或结构化内容")


class ResearchRequest(BaseModel):
    messages: List[MessageItem] = Field(..., description="对话消息列表")
    configurable: Optional[Dict[str, Any]] = Field(
        default=None, description="覆盖 Configuration 中的 configurable 参数"
    )
    thread_id: Optional[str] = Field(
        default=None, description="会话线程 ID，用于多轮记忆"
    )
    recursion_limit: int = Field(default=50, description="图递归上限")


class CareerWorkflowRequest(BaseModel):
    target_role: str = Field(default="AI工程师", description="目标岗位")
    target_city: str = Field(default="北京", description="目标城市")
    jd_text: Optional[str] = Field(default=None, description="具体 JD 文本（可跳过发现/调研）")
    configurable: Optional[Dict[str, Any]] = Field(default=None)
    thread_id: Optional[str] = Field(default=None)


class GapAnalysisRequest(BaseModel):
    jd_text: str = Field(..., description="待分析的职位描述文本")
    configurable: Optional[Dict[str, Any]] = Field(default=None)
    thread_id: Optional[str] = Field(default=None)


class CareerStageRequest(BaseModel):
    target_role: str = Field(default="AI工程师", description="目标岗位")
    target_city: str = Field(default="北京", description="目标城市")
    jd_text: Optional[str] = Field(default=None, description="可选 JD，用于更精准分析")
    configurable: Optional[Dict[str, Any]] = Field(default=None)
    thread_id: Optional[str] = Field(default=None)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def build_runnable_config(
    configurable: Optional[Dict[str, Any]],
    thread_id: Optional[str],
    recursion_limit: int = 50,
) -> RunnableConfig:
    cfg = dict(configurable or {})
    if thread_id:
        cfg["thread_id"] = thread_id
    cfg.setdefault("recursion_limit", recursion_limit)
    return {"configurable": cfg}


# 单次研究允许的最大消息条数，防止请求体滥用
_MAX_MESSAGES = int(os.getenv("API_MAX_MESSAGES", "50"))
# 研究接口整体超时（秒），0 表示不限制
_RESEARCH_TIMEOUT = int(os.getenv("API_RESEARCH_TIMEOUT", "0"))


def parse_messages(messages: List[MessageItem]) -> List[Any]:
    """将接口消息转换为 LangChain BaseMessage 列表。

    支持 role: user/human、assistant/ai、system、tool（content 可为字符串或
    结构化内容）。直接构造消息对象，避免 ``messages_from_dict`` 对 ``data``
    字段的依赖。
    """
    if len(messages) > _MAX_MESSAGES:
        raise HTTPException(
            status_code=413,
            detail=f"消息数量超出上限（最大 {_MAX_MESSAGES} 条）",
        )
    role_map = {
        "user": "human",
        "human": "human",
        "assistant": "ai",
        "ai": "ai",
        "system": "system",
        "tool": "tool",
    }
    out: List[Any] = []
    for idx, m in enumerate(messages):
        role = role_map.get(m.role, "human")
        content = m.content
        if role == "human":
            out.append(HumanMessage(content=content))
        elif role == "ai":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=f"api-{idx}"))
        else:
            out.append(HumanMessage(content=content))
    return out


def _final_content(result: Dict[str, Any]) -> str:
    msgs = result.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    if isinstance(last, (HumanMessage, SystemMessage)):
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                return m.content
        return ""
    return last.content


# --------------------------------------------------------------------------- #
# 通用信息 / 健康检查
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "AI Career Research Agent API",
        "version": "1.0.0",
        "endpoints": {
            "research": "POST /api/research | POST /api/research/stream",
            "career_workflow": "POST /api/career/workflow",
            "gap_analysis": "POST /api/career/gap-analysis",
            "discover_jobs": "POST /api/career/discover-jobs",
            "interview": "POST /api/career/interview",
            "resume": "POST /api/career/resume",
            "cover_letter": "POST /api/career/cover-letter",
            "profile": "GET /api/profile",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# 1. 深度研究图
# --------------------------------------------------------------------------- #
# API 使用「带 checkpointer 的独立编译图」，从而让 thread_id 真正具备跨请求记忆。
# 注意：langgraph.json 平台部署复用的是无 checkpointer 的 deep_researcher 对象，
# 这里额外编译一份不会破坏平台部署；生产环境可将 API_CHECKPOINTER 切换到
# Postgres/Redis 等共享 checkpointer 以支持多副本与重启持久化。
_RESEARCH_GRAPH = None


def _get_research_graph():
    global _RESEARCH_GRAPH
    if _RESEARCH_GRAPH is not None:
        return _RESEARCH_GRAPH

    mode = os.getenv("API_CHECKPOINTER", "memory").lower()
    if mode in ("none", "off", "false"):
        from open_deep_research.deep_researcher import deep_researcher

        _RESEARCH_GRAPH = deep_researcher
    else:
        # 默认：进程内内存 checkpointer，使 thread_id 多轮记忆可用（单容器/单副本）
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except Exception:  # noqa: BLE001
            from open_deep_research.deep_researcher import deep_researcher

            _RESEARCH_GRAPH = deep_researcher
        else:
            from open_deep_research.deep_researcher import deep_researcher_builder

            _RESEARCH_GRAPH = deep_researcher_builder.compile(
                checkpointer=MemorySaver()
            )
    return _RESEARCH_GRAPH


@app.post("/api/research", dependencies=[Depends(require_token)], tags=["research"])
async def research(body: ResearchRequest):
    """运行 LangGraph 深度研究图，返回最终消息列表与答案。"""
    graph = _get_research_graph()

    cfg = build_runnable_config(body.configurable, body.thread_id, body.recursion_limit)
    parsed = parse_messages(body.messages)
    try:
        if _RESEARCH_TIMEOUT > 0:
            result = await asyncio.wait_for(
                graph.ainvoke({"messages": parsed}, config=cfg),
                timeout=_RESEARCH_TIMEOUT,
            )
        else:
            result = await graph.ainvoke({"messages": parsed}, config=cfg)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="研究执行超时")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"研究执行失败：{e}")

    msgs = result.get("messages", [])
    return {
        "thread_id": body.thread_id,
        "messages": [
            {"type": type(m).__name__, "content": m.content}
            for m in msgs
        ],
        "final_answer": _final_content(result),
    }


@app.post(
    "/api/research/stream",
    dependencies=[Depends(require_token)],
    tags=["research"],
)
async def research_stream(body: ResearchRequest):
    """以 SSE 流式返回研究图执行过程（LLM token 与工具调用）。"""
    graph = _get_research_graph()

    cfg = build_runnable_config(body.configurable, body.thread_id, body.recursion_limit)
    parsed = parse_messages(body.messages)

    async def event_generator() -> AsyncIterator[str]:
        final_parts: List[str] = []
        try:
            if _RESEARCH_TIMEOUT > 0:
                stream_cm = asyncio.timeout(_RESEARCH_TIMEOUT)
            else:
                stream_cm = _nullcontext()

            async with stream_cm:
                async for chunk, meta in graph.astream(
                    {"messages": parsed}, config=cfg, stream_mode="messages"
                ):
                    node = (meta or {}).get("langgraph_node")
                    data = {
                        "type": type(chunk).__name__,
                        "content": chunk.content,
                        "name": getattr(chunk, "name", None),
                        "tool_call_chunks": getattr(chunk, "tool_call_chunks", None),
                        "langgraph_node": node,
                    }
                    # 同时累积最终报告节点的输出，用于结尾的 result 事件
                    if node == "final_report_generation" and isinstance(
                        chunk, AIMessageChunk
                    ):
                        if isinstance(chunk.content, str):
                            final_parts.append(chunk.content)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            # 发送完整最终答案（无需额外 LLM 调用）
            yield f"data: {json.dumps({'type': 'result', 'content': ''.join(final_parts)}, ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'content': '研究执行超时'}, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# 2. 求职场景单点能力
# --------------------------------------------------------------------------- #
@app.get("/api/profile", dependencies=[Depends(require_token)], tags=["career"])
async def get_profile(configurable: Optional[str] = None):
    """读取候选人背景画像（简历文件或 CareerConfig.user_profile）。"""
    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(
        json.loads(configurable) if configurable else None, None
    )
    return {"profile": load_user_profile(cfg)}


@app.post(
    "/api/career/gap-analysis",
    dependencies=[Depends(require_token)],
    tags=["career"],
)
async def api_gap_analysis(body: GapAnalysisRequest):
    """针对具体 JD 做人岗匹配度分析（Gap Analysis）。"""
    from open_deep_research.gap_analysis import run_gap_analysis

    cfg = build_runnable_config(body.configurable, body.thread_id)
    try:
        result = await run_gap_analysis(body.jd_text, cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"匹配度分析失败：{e}")
    return {"jd_text": body.jd_text, "gap_analysis": result}


@app.post(
    "/api/career/discover-jobs",
    dependencies=[Depends(require_token)],
    tags=["career"],
)
async def api_discover_jobs(body: CareerStageRequest):
    """岗位发现：推荐优先投递的岗位方向与公司类型。"""
    from open_deep_research.career_workflow import run_discover_jobs
    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(body.configurable, body.thread_id)
    try:
        result = await run_discover_jobs(
            load_user_profile(cfg), body.target_role, body.target_city, cfg
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"岗位发现失败：{e}")
    return {
        "target_role": body.target_role,
        "target_city": body.target_city,
        "discovery": result,
    }


@app.post(
    "/api/career/interview",
    dependencies=[Depends(require_token)],
    tags=["career"],
)
async def api_interview(body: CareerStageRequest):
    """面试准备：结合目标岗位与匹配度分析，生成备考清单。"""
    from open_deep_research.career_workflow import (
        run_gap_analysis,
        run_interview_prep,
    )
    from open_deep_research.gap_analysis import run_gap_analysis as _gap
    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(body.configurable, body.thread_id)
    profile = load_user_profile(cfg)
    jd = body.jd_text or f"目标岗位：{body.target_role}"
    try:
        gap = await _gap(jd, cfg)
        result = await run_interview_prep(body.target_role, profile, gap, cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"面试准备失败：{e}")
    return {"target_role": body.target_role, "interview_prep": result}


@app.post(
    "/api/career/resume",
    dependencies=[Depends(require_token)],
    tags=["career"],
)
async def api_resume(body: CareerStageRequest):
    """简历优化：针对目标岗位给出策略 + 改写后的核心经历 bullet。"""
    from open_deep_research.career_workflow import run_resume_optimization
    from open_deep_research.gap_analysis import run_gap_analysis as _gap
    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(body.configurable, body.thread_id)
    profile = load_user_profile(cfg)
    jd = body.jd_text or f"目标岗位：{body.target_role}"
    try:
        gap = await _gap(jd, cfg)
        result = await run_resume_optimization(body.target_role, profile, gap, cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"简历优化失败：{e}")
    return {"target_role": body.target_role, "resume_optimization": result}


@app.post(
    "/api/career/cover-letter",
    dependencies=[Depends(require_token)],
    tags=["career"],
)
async def api_cover_letter(body: CareerStageRequest):
    """求职信：为目标岗位写一封中文求职信。"""
    from open_deep_research.career_workflow import run_cover_letter
    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(body.configurable, body.thread_id)
    try:
        result = await run_cover_letter(
            body.target_role, load_user_profile(cfg), cfg
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"求职信生成失败：{e}")
    return {"target_role": body.target_role, "cover_letter": result}


@app.post(
    "/api/career/workflow",
    dependencies=[Depends(require_token)],
    tags=["career"],
)
async def api_career_workflow(body: CareerWorkflowRequest):
    """端到端求职工作流：岗位发现 → 调研 → 匹配度 → 面试准备 → 简历优化 → 求职信。"""
    from open_deep_research.career_workflow import run_career_workflow

    cfg = build_runnable_config(body.configurable, body.thread_id)
    try:
        result = await run_career_workflow(
            target_role=body.target_role,
            target_city=body.target_city,
            jd_text=body.jd_text or "",
            config=cfg,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"工作流执行失败：{e}")
    return {
        "target_role": result.target_role,
        "target_city": result.target_city,
        "report": result.report,
        "artifacts": {
            "discovery": result.discovery,
            "research": result.research,
            "gap_analysis": result.gap_analysis,
            "interview_prep": result.interview_prep,
            "resume_optimization": result.resume_optimization,
            "cover_letter": result.cover_letter,
        },
    }


# --------------------------------------------------------------------------- #
# 本地启动入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "open_deep_research.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
