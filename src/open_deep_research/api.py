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

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext as _nullcontext

logger = logging.getLogger(__name__)
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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

# --------------------------------------------------------------------------- #
# 简易限流（#3：基于客户端 IP 的滑动窗口；生产可换为 Redis 共享限流）
# --------------------------------------------------------------------------- #
_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "0"))  # 0 表示不限流
_RATE_WINDOW = 60.0
_RATE_BUCKETS: "dict[str, list[float]]" = {}
_RATE_LOCK = asyncio.Lock()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _RATE_LIMIT <= 0:
        return await call_next(request)
    # 健康检查与文档等不计入限流
    if request.url.path in ("/", "/health", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)

    forwarded = request.headers.get("x-forwarded-for", "")
    client_ip = (
        forwarded.split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    now = time.monotonic()
    async with _RATE_LOCK:
        hits = _RATE_BUCKETS.setdefault(client_ip, [])
        cutoff = now - _RATE_WINDOW
        hits[:] = [t for t in hits if t > cutoff]
        if len(hits) >= _RATE_LIMIT:
            retry = int(max(hits[0] + _RATE_WINDOW - now, 1))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "请求过于频繁，请稍后再试",
                    "retry_after_seconds": retry,
                },
                headers={"Retry-After": str(retry)},
            )
        hits.append(now)
    return await call_next(request)

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


class ProfileTextRequest(BaseModel):
    content: str = Field(..., description="简历 / 背景资料全文（Markdown 或纯文本）")
    name: Optional[str] = Field(
        default=None, description="档案名；为空则写入当前生效的简历文件"
    )


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

# --------------------------------------------------------------------------- #
# Gap Analysis 进程内缓存（#5：避免 interview/resume/gap-analysis 等重复跑 LLM）
# --------------------------------------------------------------------------- #
_GAP_CACHE_ENABLED = os.getenv("API_GAP_CACHE", "on").lower() not in (
    "off",
    "0",
    "false",
)
_GAP_CACHE_MAX = int(os.getenv("API_GAP_CACHE_MAX", "256"))
_GAP_CACHE: "OrderedDict[str, str]" = OrderedDict()
_GAP_CACHE_LOCK = asyncio.Lock()


def _gap_cache_key(jd_text: str, config: "RunnableConfig") -> str:
    configurable = (config or {}).get("configurable", {}) or {}
    payload = {"jd_text": jd_text, "configurable": configurable}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def run_gap_analysis_cached(jd_text: str, config: "RunnableConfig") -> str:
    """调用 ``run_gap_analysis``，对相同输入做进程内缓存以省去重复 LLM 开销。

    注意：缓存基于 ``jd_text`` 与 ``config.configurable``（含 profile / 模型等）。
    若简历文件等内容在进程内被外部修改，请设置 ``API_GAP_CACHE=off`` 或重启服务。
    """
    if not _GAP_CACHE_ENABLED:
        from open_deep_research.gap_analysis import run_gap_analysis

        return await run_gap_analysis(jd_text, config)

    key = _gap_cache_key(jd_text, config)
    async with _GAP_CACHE_LOCK:
        if key in _GAP_CACHE:
            return _GAP_CACHE[key]

    from open_deep_research.gap_analysis import run_gap_analysis

    result = await run_gap_analysis(jd_text, config)
    async with _GAP_CACHE_LOCK:
        _GAP_CACHE[key] = result
        _GAP_CACHE.move_to_end(key)
        while len(_GAP_CACHE) > _GAP_CACHE_MAX:
            _GAP_CACHE.popitem(last=False)
    return result


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


def _content_to_text(content: Any) -> str:
    """将消息 content（可能是多模态 list）规整为纯文本。"""
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _final_content(result: Dict[str, Any]) -> str:
    msgs = result.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    if isinstance(last, (HumanMessage, SystemMessage)):
        for m in reversed(msgs):
            if isinstance(m, AIMessage):
                return _content_to_text(m.content)
        return ""
    return _content_to_text(last.content)


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
# 这里额外编译一份不会破坏平台部署。
#
# 通过 API_CHECKPOINTER 选择 checkpointer：
#   - memory  （默认）进程内内存，单容器/单副本多轮记忆；重启即丢失
#   - none    复用平台无 checkpointer 版本（不保留多轮记忆）
#   - postgres 需设置 API_CHECKPOINTER_POSTGRES_DSN，支持多副本共享与持久化
#   - redis    需安装 langgraph-checkpoint-redis 并设置 API_CHECKPOINTER_REDIS_URI
# 生产环境推荐 postgres/redis，以便横向扩展（多 worker/多副本）并持久化会话。
_RESEARCH_GRAPH = None


def _build_checkpointer(mode: str):
    """根据模式构造 checkpointer，任意失败都安全回退到内存版。"""
    if mode in ("none", "off", "false"):
        return None
    if mode in ("", "memory"):
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if mode == "postgres":
        dsn = os.getenv("API_CHECKPOINTER_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
        if not dsn:
            logger.warning(
                "API_CHECKPOINTER=postgres 但未设置连接串，回退到 memory"
            )
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            return PostgresSaver.from_conn_string(dsn)
        except Exception as e:  # noqa: BLE001
            logger.warning("Postgres checkpointer 初始化失败，回退 memory: %s", e)
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
    if mode == "redis":
        uri = os.getenv("API_CHECKPOINTER_REDIS_URI") or os.getenv("REDIS_URI")
        if not uri:
            logger.warning(
                "API_CHECKPOINTER=redis 但未设置连接串，回退到 memory"
            )
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        try:
            from langgraph.checkpoint.redis import RedisSaver

            return RedisSaver.from_conn_string(uri)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Redis checkpointer 初始化失败（缺少 langgraph-checkpoint-redis？），回退 memory: %s",
                e,
            )
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
    logger.warning("未知 API_CHECKPOINTER=%s，回退到 memory", mode)
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def _build_store():
    """构造长期记忆 Store，供跨会话记忆（用户画像 / 研究知识）使用。

    通过 MEMORY_STORE 环境变量选择后端：
      - memory   （默认）进程内 InMemoryStore，开发/单副本；重启即丢失
      - postgres 需设置 MEMORY_STORE_POSTGRES_DSN，支持持久化与多副本共享
      - none     不挂载 Store，长期记忆退化为无操作（不影响单次任务运行）

    返回 ``None`` 表示不挂载；此时 memory 模块会自动使用进程内兜底 Store。
    """
    mode = os.getenv("MEMORY_STORE", "memory").lower()
    if mode in ("none", "off", "false"):
        return None
    if mode in ("", "memory"):
        try:
            from langgraph.store.memory import InMemoryStore

            return InMemoryStore(index=None)
        except Exception as e:  # noqa: BLE001
            logger.warning("InMemoryStore 初始化失败：%s", e)
            return None
    if mode == "postgres":
        dsn = os.getenv("MEMORY_STORE_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
        if not dsn:
            logger.warning(
                "MEMORY_STORE=postgres 但未设置连接串，回退到 memory"
            )
            from langgraph.store.memory import InMemoryStore

            return InMemoryStore(index=None)
        try:
            from langgraph.store.postgres import PostgresStore

            store = PostgresStore.from_conn_string(dsn)
            # Best-effort 表结构初始化
            setup = getattr(store, "setup", None)
            if callable(setup):
                try:
                    res = setup()
                    if hasattr(res, "__await__"):
                        logger.warning(
                            "PostgresStore.setup() 为协程，需在其事件循环中 await；"
                            "如首次写入报错请手动执行 setup()"
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "PostgresStore.setup() 失败（表可能已存在，可忽略）：%s", e
                    )
            return store
        except Exception as e:  # noqa: BLE001
            logger.warning("PostgresStore 初始化失败，回退 memory: %s", e)
            from langgraph.store.memory import InMemoryStore

            return InMemoryStore(index=None)
    logger.warning("未知 MEMORY_STORE=%s，回退到 memory", mode)
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore(index=None)


def _get_research_graph():
    global _RESEARCH_GRAPH
    if _RESEARCH_GRAPH is not None:
        return _RESEARCH_GRAPH

    mode = os.getenv("API_CHECKPOINTER", "memory").lower()
    cp = _build_checkpointer(mode)
    store = _build_store()
    try:
        if cp is None:
            from open_deep_research.deep_researcher import deep_researcher

            _RESEARCH_GRAPH = (
                deep_researcher.compile(store=store) if store is not None else deep_researcher
            )
        else:
            # 表结构初始化（best-effort；若为异步 setup 会给出提示）
            setup = getattr(cp, "setup", None)
            if callable(setup):
                try:
                    res = setup()
                    if hasattr(res, "__await__"):
                        logger.warning(
                            "checkpointer.setup() 为协程，需在其事件循环中 await；"
                            "如首次写入报错请手动执行 setup()"
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "checkpointer.setup() 失败（表可能已存在，可忽略）：%s", e
                    )

            from open_deep_research.deep_researcher import deep_researcher_builder

            compile_kwargs = {"checkpointer": cp}
            if store is not None:
                compile_kwargs["store"] = store
            _RESEARCH_GRAPH = deep_researcher_builder.compile(**compile_kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning("研究图编译失败，回退无 checkpointer 版本：%s", e)
        from open_deep_research.deep_researcher import deep_researcher

        _RESEARCH_GRAPH = deep_researcher
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
async def get_profile(configurable: Optional[str] = None, name: Optional[str] = None):
    """读取候选人背景画像。

    指定 ``name`` 时读取对应档案的内容；否则读取当前生效的简历
    （``resume_path`` 指向的文件，即最近一次激活的档案）。
    """
    if name:
        name = _safe_profile_name(name)
        path = _profile_file(name)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"档案不存在：{name}")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return {"profile": f.read(), "name": name}
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"读取档案失败：{e}")

    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(
        json.loads(configurable) if configurable else None, None
    )
    return {"profile": load_user_profile(cfg), "name": _read_active_name()}


# --------------------------------------------------------------------------- #
# 简历 / 背景资料写入（供 Web 界面上传与在线编辑使用）
#
# 支持多档案：档案存放在 data/profiles/<名字>.md，「切换到某档案」即把该档案内容
# 写入 resume_path（默认 data/简历.md）。后端所有求职功能都从 resume_path 读取，
# 因此切换后全部功能立即生效，调用方无需感知档案机制。
# --------------------------------------------------------------------------- #
_UPLOAD_MAX_BYTES = int(os.getenv("API_UPLOAD_MAX_MB", "5")) * 1024 * 1024
_UPLOAD_ALLOWED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_PROFILE_DIR = os.getenv("API_PROFILE_DIR", "data/profiles")
_ACTIVE_MARK = ".active"


def _resolve_resume_path() -> str:
    """解析简历保存路径：优先 CareerConfig.resume_path，回退 ./data/简历.md。"""
    try:
        from open_deep_research.configuration import CareerConfig, Configuration

        # 与 load_user_profile 保持一致：未启用 CareerConfig 时回退默认配置
        career = Configuration.from_runnable_config({}).career_config or CareerConfig()
        path = getattr(career, "resume_path", None)
        if path:
            return path
    except Exception as e:  # noqa: BLE001
        logger.warning("解析 resume_path 失败，使用默认路径：%s", e)
    return "./data/简历.md"


def _extract_text(filename: str, raw: bytes) -> str:
    """把上传文件解析为纯文本：md/txt 直接解码，PDF 用 PyMuPDF 提取文字。"""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix == ".pdf":
        try:
            import pymupdf  # PyMuPDF 新版本的导入名
        except ImportError:
            try:
                import fitz as pymupdf  # type: ignore  # 旧版本兼容
            except ImportError as e:
                raise HTTPException(
                    status_code=500, detail=f"PDF 解析需要 pymupdf 依赖：{e}"
                )
        try:
            with pymupdf.open(stream=raw, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc).strip()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"PDF 解析失败：{e}")

    for encoding in ("utf-8", "gbk", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _safe_profile_name(name: str) -> str:
    """校验档案名：禁止路径穿越与隐藏文件，返回规范化后的名字。"""
    cleaned = (name or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="档案名不能为空")
    if cleaned in (".", "..") or cleaned.startswith("."):
        raise HTTPException(status_code=400, detail=f"非法档案名：{name}")
    if "/" in cleaned or "\\" in cleaned or os.path.basename(cleaned) != cleaned:
        raise HTTPException(status_code=400, detail=f"档案名不能包含路径分隔符：{name}")
    return cleaned


def _profile_file(name: str) -> str:
    """返回指定档案的文件路径。"""
    return os.path.join(_PROFILE_DIR, f"{_safe_profile_name(name)}.md")


def _list_profile_names() -> List[str]:
    """列出所有档案名（按名称排序）。"""
    if not os.path.isdir(_PROFILE_DIR):
        return []
    return sorted(
        f[:-3]
        for f in os.listdir(_PROFILE_DIR)
        if f.endswith(".md") and os.path.isfile(os.path.join(_PROFILE_DIR, f))
    )


def _read_active_name() -> Optional[str]:
    """读取当前激活的档案名；未设置则返回 None。"""
    path = os.path.join(_PROFILE_DIR, _ACTIVE_MARK)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _set_active_name(name: Optional[str]) -> None:
    """记录当前激活的档案名。"""
    try:
        os.makedirs(_PROFILE_DIR, exist_ok=True)
        with open(
            os.path.join(_PROFILE_DIR, _ACTIVE_MARK), "w", encoding="utf-8"
        ) as f:
            f.write(name or "")
    except OSError as e:
        logger.warning("写入激活标记失败：%s", e)


def _write_text(path: str, text: str) -> None:
    """把文本写入指定路径（自动创建父目录）。"""
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"保存失败：{e}")


def _save_profile_text(text: str, name: Optional[str]) -> Dict[str, Any]:
    """保存简历文本。

    指定 ``name`` 时保存到对应档案（不存在则创建）；若该档案正是当前激活档案，
    则同步写入 resume_path，保证「编辑后对全部功能立即生效」。
    """
    if name:
        name = _safe_profile_name(name)
        path = _profile_file(name)
        _write_text(path, text)
        if name == _read_active_name():
            _write_text(_resolve_resume_path(), text)
        return {"status": "ok", "name": name, "path": path, "chars": len(text)}

    path = _resolve_resume_path()
    _write_text(path, text)
    return {"status": "ok", "path": path, "chars": len(text)}


@app.get(
    "/api/profile/names", dependencies=[Depends(require_token)], tags=["career"]
)
async def list_profiles():
    """列出所有简历档案名，以及当前激活（生效中）的档案。"""
    return {"names": _list_profile_names(), "active": _read_active_name()}


@app.post("/api/profile", dependencies=[Depends(require_token)], tags=["career"])
async def save_profile(body: ProfileTextRequest):
    """保存候选人背景资料文本；指定 name 时保存到对应档案（不存在则创建）。"""
    content = body.content or ""
    if not content.strip():
        raise HTTPException(status_code=400, detail="内容为空，未保存")
    if len(content.encode("utf-8")) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"内容过大，上限 {_UPLOAD_MAX_BYTES // 1024 // 1024} MB",
        )
    return _save_profile_text(content, body.name)


@app.post(
    "/api/profile/upload", dependencies=[Depends(require_token)], tags=["career"]
)
async def upload_profile(file: UploadFile = File(...), name: Optional[str] = None):
    """上传简历文件（.md/.txt/.pdf）；指定 name 时保存到对应档案。"""
    filename = file.filename or ""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in _UPLOAD_ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"不支持的文件类型（{suffix or '未知'}），"
                f"仅支持 {', '.join(sorted(_UPLOAD_ALLOWED_SUFFIXES))}"
            ),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(raw) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，上限 {_UPLOAD_MAX_BYTES // 1024 // 1024} MB",
        )

    text = _extract_text(filename, raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="未能从文件中提取到文本内容")

    result = _save_profile_text(text, name)
    result["filename"] = filename
    result["preview"] = text[:300]
    return result


class ProfileActivateRequest(BaseModel):
    name: str = Field(..., description="要切换到的档案名")


@app.post(
    "/api/profile/activate", dependencies=[Depends(require_token)], tags=["career"]
)
async def activate_profile(body: ProfileActivateRequest):
    """切换当前生效的简历档案：把档案内容写入 resume_path，全部求职功能立即生效。"""
    name = _safe_profile_name(body.name)
    path = _profile_file(name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"档案不存在：{name}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"读取档案失败：{e}")
    if not text.strip():
        raise HTTPException(status_code=400, detail=f"档案内容为空，无法切换：{name}")

    _write_text(_resolve_resume_path(), text)
    _set_active_name(name)
    return {"status": "ok", "active": name, "chars": len(text)}


@app.delete("/api/profile", dependencies=[Depends(require_token)], tags=["career"])
async def delete_profile(name: str):
    """删除指定简历档案；不允许删除当前激活档案。"""
    name = _safe_profile_name(name)
    if name == _read_active_name():
        raise HTTPException(
            status_code=400, detail="不能删除正在使用的档案，请先切换到其他档案"
        )
    path = _profile_file(name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"档案不存在：{name}")
    try:
        os.remove(path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败：{e}")
    return {"status": "ok", "deleted": name}


@app.post(
    "/api/career/gap-analysis",
    dependencies=[Depends(require_token)],
    tags=["career"],
)
async def api_gap_analysis(body: GapAnalysisRequest):
    """针对具体 JD 做人岗匹配度分析（Gap Analysis）。"""
    cfg = build_runnable_config(body.configurable, body.thread_id)
    try:
        result = await run_gap_analysis_cached(body.jd_text, cfg)
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
    from open_deep_research.career_workflow import run_interview_prep
    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(body.configurable, body.thread_id)
    profile = load_user_profile(cfg)
    jd = body.jd_text or f"目标岗位：{body.target_role}"
    try:
        gap = await run_gap_analysis_cached(jd, cfg)
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
    from open_deep_research.profile import load_user_profile

    cfg = build_runnable_config(body.configurable, body.thread_id)
    profile = load_user_profile(cfg)
    jd = body.jd_text or f"目标岗位：{body.target_role}"
    try:
        gap = await run_gap_analysis_cached(jd, cfg)
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
