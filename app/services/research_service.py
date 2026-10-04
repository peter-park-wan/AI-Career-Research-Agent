"""研究服务：checkpointer / store 装配与"编译一次、复用多次"的图单例。

【改造要点】

1. **消灭 import 期求值**。原实现在请求处理函数里 ``os.getenv`` 读配置，
   现在统一走 ``Settings``，配置来源只有一处。
2. **单例加锁**。原 ``_get_research_graph`` 是"先查后建"的裸写法：

       if _RESEARCH_GRAPH is not None: return _RESEARCH_GRAPH
       ... 构建（含 import + compile，耗时数百毫秒）...
       _RESEARCH_GRAPH = ...

   两个并发请求会在同一个事件循环里交错，各自编译一份图。除了白干活，
   MemorySaver 会出现两份，于是**同一个 thread_id 的记忆随机丢失**。
   现在用 ``asyncio.Lock`` 做双重检查，保证只编译一次。
3. **阻塞构建移出事件循环**。``compile()`` 会 import 大量模块、建立连接，
   放到 ``anyio.to_thread.run_sync`` 里执行，避免首个请求把整个进程卡住。
4. **降级不许静默**。声明了 postgres/redis 却装配失败时：

       生产环境  → 抛 ``ConfigError``，进程起不来（fail fast）
       其它环境  → 退回内存版，但记 error 日志并写入 ``runtime_status()``

   改造前这里是无条件回退：服务照常返回 200，而多轮记忆已经悄悄变成
   进程内内存版 —— 监控看到的"健康"是假的。详见 ``_degrade``。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import anyio

from app.core.config import Settings, get_settings
from app.core.errors import ConfigError, UpstreamError, UpstreamTimeoutError
from app.services import messages

logger = logging.getLogger(__name__)

_graph: Optional[Any] = None
_graph_lock = asyncio.Lock()

# 装配留痕。"声明 postgres、实际 memory" 这类偏差必须能被 /ready 直接看到，
# 而不是埋在日志里等人翻 —— 静默降级等同于对监控系统说谎。
_effective_backends: Dict[str, str] = {}
_degraded_reasons: List[str] = []


# --------------------------------------------------------------------------- #
# 降级策略
# --------------------------------------------------------------------------- #
def _strict_backend(settings: Settings) -> bool:
    """生产环境是否禁止"装配失败就悄悄退回内存版"。

    开发环境不做要求：本地没起 Postgres 也应该能把服务跑起来。
    生产环境默认严格，因为"声称持久化、实际丢数据"比"起不来"危害更大。
    """
    return settings.is_prod and not settings.allow_memory_fallback


# 形如 postgres://user:password@host:5432/db 的连接串
_CREDENTIAL_RE = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<user>[^:/@\s]+):(?P<pwd>[^@\s]*)@"
)


def _mask_url(text: str) -> str:
    """把文本里所有 ``scheme://user:password@host`` 的密码替换为 ``***``。"""
    return _CREDENTIAL_RE.sub(lambda m: f"{m['scheme']}{m['user']}:***@", text)


def _redact(text: str, settings: Optional[Settings] = None) -> str:
    """异常原文可能含带密码的连接串 / Bearer Token，进日志与报错前必须脱敏。

    ``PostgresSaver.from_conn_string`` 失败时，psycopg 通常把整条 DSN 回显在
    异常里；直接记日志等于把数据库密码写进日志系统，且日志往往比数据库
    本身的访问权限更宽松。
    """
    settings = settings or get_settings()
    for secret in (
        settings.resolved_checkpointer_dsn,
        settings.resolved_checkpointer_redis_uri,
        settings.resolved_store_dsn,
        settings.redis_uri,
        settings.bearer_token,
    ):
        if secret and secret in text:
            text = text.replace(secret, "***")
    return _mask_url(text)


def _mark_effective(component: str, label: str) -> None:
    _effective_backends[component] = label


def _degrade(
    component: str,
    declared: str,
    reason: object,
    settings: Settings,
    fallback: Callable[[], Any],
) -> Any:
    """降级的唯一出口。

    严格模式：抛 ``ConfigError`` 让进程起不来（fail fast）。
    宽松模式：退回内存版，但必须写 error 日志并记入 ``_degraded_reasons``，
    让 /ready 能报出 degraded —— 降级可以发生，但**不允许静默**。
    """
    safe_reason = _redact(str(reason), settings)
    if _strict_backend(settings):
        raise ConfigError(
            f"{component} 声明为 '{declared}' 但装配失败：{safe_reason}。"
            f"生产环境按 fail-fast 终止启动；若确需临时降级到内存版，"
            f"请显式设置 API_ALLOW_MEMORY_FALLBACK=true"
            f"（代价：会话记忆不再持久化，重启即丢失）"
        )
    value = fallback()
    effective = "memory" if value is not None else "none"
    _mark_effective(component, effective)
    _degraded_reasons.append(
        f"{component} 声明 '{declared}' 但未装配成功，实际使用 '{effective}'：{safe_reason}"
    )
    logger.error(
        "持久化后端降级：声明与实际不一致",
        extra={
            "component": component,
            "declared": declared,
            "effective": effective,
            "reason": safe_reason,
        },
    )
    return value


# --------------------------------------------------------------------------- #
# checkpointer（短期会话记忆）
# --------------------------------------------------------------------------- #
def build_checkpointer(mode: str, settings: Optional[Settings] = None) -> Any:
    """按模式构造 checkpointer。

    memory  （默认）进程内内存，单副本多轮记忆；重启即丢失
    none    复用无 checkpointer 版本（不保留多轮记忆）
    postgres 需配置 API_CHECKPOINTER_POSTGRES_DSN，支持多副本共享与持久化
    redis    需安装 langgraph-checkpoint-redis 并配置 API_CHECKPOINTER_REDIS_URI

    失败时**不一定**回退：生产环境直接抛 ``ConfigError``（fail fast），
    其它环境才回退到内存版，且两种情况都会留下痕迹。见 ``_degrade``。
    """
    settings = settings or get_settings()

    def _memory() -> Any:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if mode in ("none", "off", "false"):
        _mark_effective("checkpointer", "none")
        return None

    if mode in ("", "memory"):
        _mark_effective("checkpointer", "memory")
        return _memory()

    if mode == "postgres":
        dsn = settings.resolved_checkpointer_dsn
        if not dsn:
            return _degrade(
                "checkpointer",
                mode,
                "未配置连接串（API_CHECKPOINTER_POSTGRES_DSN / API_POSTGRES_DSN）",
                settings,
                _memory,
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            checkpointer = PostgresSaver.from_conn_string(dsn)
        except Exception as exc:  # noqa: BLE001
            return _degrade("checkpointer", mode, exc, settings, _memory)
        _mark_effective("checkpointer", "postgres")
        return checkpointer

    if mode == "redis":
        uri = settings.resolved_checkpointer_redis_uri
        if not uri:
            return _degrade(
                "checkpointer",
                mode,
                "未配置连接串（API_CHECKPOINTER_REDIS_URI / API_REDIS_URI）",
                settings,
                _memory,
            )
        try:
            from langgraph.checkpoint.redis import RedisSaver

            checkpointer = RedisSaver.from_conn_string(uri)
        except Exception as exc:  # noqa: BLE001
            return _degrade("checkpointer", mode, exc, settings, _memory)
        _mark_effective("checkpointer", "redis")
        return checkpointer

    return _degrade("checkpointer", mode, "未知的模式取值", settings, _memory)


# --------------------------------------------------------------------------- #
# store（跨会话长期记忆）
# --------------------------------------------------------------------------- #
def build_store(settings: Optional[Settings] = None) -> Any:
    """构造长期记忆 Store；返回 ``None`` 表示不挂载（memory 模块会自带兜底）。"""
    settings = settings or get_settings()
    mode = settings.memory_store

    def _memory() -> Any:
        try:
            from langgraph.store.memory import InMemoryStore

            return InMemoryStore(index=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("InMemoryStore 初始化失败：%s", exc)
            return None

    if mode in ("none", "off", "false"):
        _mark_effective("memory_store", "none")
        return None
    if mode in ("", "memory"):
        _mark_effective("memory_store", "memory")
        return _memory()

    if mode == "postgres":
        dsn = settings.resolved_store_dsn
        if not dsn:
            return _degrade(
                "memory_store",
                mode,
                "未配置连接串（API_MEMORY_STORE_POSTGRES_DSN / API_POSTGRES_DSN）",
                settings,
                _memory,
            )
        try:
            from langgraph.store.postgres import PostgresStore

            store = PostgresStore.from_conn_string(dsn)
            _best_effort_setup(store, "PostgresStore")
        except Exception as exc:  # noqa: BLE001
            return _degrade("memory_store", mode, exc, settings, _memory)
        _mark_effective("memory_store", "postgres")
        return store

    return _degrade("memory_store", mode, "未知的模式取值", settings, _memory)


def _best_effort_setup(target: Any, label: str) -> None:
    """尽力初始化后端表结构；异步 setup 只提示不阻塞。"""
    setup = getattr(target, "setup", None)
    if not callable(setup):
        return
    try:
        result = setup()
        if hasattr(result, "__await__"):
            logger.warning(
                "%s.setup() 是协程，需在其事件循环中 await；若首次写入报错请手动执行 setup()",
                label,
            )
    except Exception as exc:  # noqa: BLE001
        # setup 失败常伴随整条 DSN 回显，脱敏后再记
        logger.warning("%s.setup() 失败（表可能已存在，可忽略）：%s", label, _redact(str(exc)))


# --------------------------------------------------------------------------- #
# 图单例
# --------------------------------------------------------------------------- #
def _compile_graph(settings: Settings) -> Any:
    """同步构建图（在工作线程中执行）。"""
    checkpointer = build_checkpointer(settings.checkpointer, settings)
    store = build_store(settings)

    try:
        if checkpointer is None:
            from open_deep_research.deep_researcher import deep_researcher

            return (
                deep_researcher.compile(store=store) if store is not None else deep_researcher
            )

        _best_effort_setup(checkpointer, "checkpointer")
        from open_deep_research.deep_researcher import deep_researcher_builder

        compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer}
        if store is not None:
            compile_kwargs["store"] = store
        return deep_researcher_builder.compile(**compile_kwargs)
    except Exception as exc:  # noqa: BLE001
        # 改造前这里无条件回退：生产上 checkpointer 装配失败会静默变成
        # "无记忆版"，服务照常返回 200，但多轮记忆已经没了。
        # 现在与后端降级走同一套策略：生产 fail fast，其余环境降级但留痕。
        safe_reason = _redact(str(exc), settings)
        if _strict_backend(settings):
            raise ConfigError(f"研究图编译失败：{safe_reason}") from exc
        logger.error("研究图编译失败，回退无 checkpointer 版本：%s", safe_reason)
        _degraded_reasons.append(
            f"research_graph 编译失败，回退无 checkpointer 版本：{safe_reason}"
        )
        from open_deep_research.deep_researcher import deep_researcher

        return deep_researcher


async def get_research_graph(settings: Optional[Settings] = None) -> Any:
    """获取研究图单例。并发调用只编译一次。"""
    global _graph
    if _graph is not None:
        return _graph

    async with _graph_lock:
        if _graph is not None:  # 双重检查：等锁期间可能已被别的请求建好
            return _graph
        _graph = await anyio.to_thread.run_sync(
            _compile_graph, settings or get_settings()
        )
        return _graph


def reset_research_graph() -> None:
    """丢弃已编译的图与装配留痕（测试或运行时改配置后调用）。"""
    global _graph
    _graph = None
    _effective_backends.clear()
    _degraded_reasons.clear()


def runtime_status(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """返回持久化后端的"声明 vs 实际"，供 ``/ready`` 暴露给编排器与监控。

    这是本次修复的关键产出：降级本身不再被禁止（运维可能确实需要），
    但**静默**降级被禁止了 —— 任何偏差都必须在这里可见。
    """
    settings = settings or get_settings()
    return {
        "graph_compiled": _graph is not None,
        "declared": {"checkpointer": settings.checkpointer, "memory_store": settings.memory_store},
        "effective": dict(_effective_backends),
        "degraded": bool(_degraded_reasons),
        "degraded_reasons": list(_degraded_reasons),
    }


# --------------------------------------------------------------------------- #
# 执行（同步返回 / SSE 流式）
# --------------------------------------------------------------------------- #
async def run_research(
    graph: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    timeout: int = 0,
) -> dict[str, Any]:
    """执行研究图并返回最终状态。

    超时与上游失败都翻译成类型化错误：前者 504（网关超时，客户端可重试），
    后者 502（上游 LLM/搜索失败）。改造前两者都被压成 500 并回显异常原文。
    """
    try:
        if timeout > 0:
            return await asyncio.wait_for(
                graph.ainvoke(inputs, config=config), timeout=timeout
            )
        return await graph.ainvoke(inputs, config=config)
    except asyncio.TimeoutError as exc:
        raise UpstreamTimeoutError("研究执行", timeout) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("研究执行失败")
        raise UpstreamError("研究执行", str(exc)) from exc


# 流结束哨兵。不直接把 ``__anext__()`` 交给 ``wait_for``：后者会把它包成
# Task，而 StopAsyncIteration 穿过 Task 边界的语义在各 Python 版本上不一致。
# 用哨兵把"结束"变成普通返回值，这类边界问题就不存在了。
_STREAM_END = object()
_DONE_CHUNK = "data: [DONE]\n\n"
# 超时/断开后关闭底层流的最长等待；关闭失败只记日志，不影响已发出的响应
_CLOSE_GRACE = 5.0


async def _next_chunk(iterator: Any) -> Any:
    """取下一个 chunk；流结束时返回 ``_STREAM_END``。"""
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return _STREAM_END


async def _aclose_quietly(iterator: Any) -> None:
    """尽力关闭底层流。

    超时只取消了"等下一个 chunk"这一步，图里的 LLM 调用不会自动停。不主动
    关闭的话，用户已经收到超时提示了，后台还在继续烧 token。
    """
    aclose = getattr(iterator, "aclose", None)
    if aclose is None:
        return
    try:
        await asyncio.wait_for(aclose(), _CLOSE_GRACE)
    except Exception:  # noqa: BLE001
        logger.debug("研究流关闭失败（可忽略）")


async def stream_research_events(
    graph: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    timeout: int = 0,
) -> AsyncIterator[str]:
    """以 SSE 文本块的形式流式输出研究过程。

    结尾补发一个 ``result`` 事件，其内容是 ``final_report_generation`` 节点
    已流出的 token 拼接 —— 这样前端不必再发一次请求，也不用额外调用一次 LLM。

    **超时语义**：``timeout`` 只累计"等待研究产出下一个 chunk"的时间，
    **不含**把事件写给客户端的时间。

    改造前用 ``asyncio.timeout`` 包住整个 ``async for``，而循环体里有
    ``yield``。慢客户端（弱网、反向代理缓冲、移动端）会让 ``yield`` 阻塞
    很久，这段时间被算进了研究预算：研究早已跑完，用户却收到"研究执行超时"。
    更麻烦的是这条错误会误导排查方向 —— 让人以为该换更快的模型，
    而真正的瓶颈在网络。
    """
    final_parts: List[str] = []
    iterator = graph.astream(inputs, config=config, stream_mode="messages")
    # None 表示不限时；否则是剩余预算（秒）
    remaining: Optional[float] = float(timeout) if timeout > 0 else None
    # 生成器已被关闭/取消时 finally 不能再 yield，否则抛 RuntimeError
    aborted = False

    try:
        while True:
            if remaining is not None and remaining <= 0:
                yield sse_event(
                    {"type": "error", "content": f"研究执行超时（>{timeout}s）"}
                )
                break

            # 只有这一段计入预算：等研究推进
            started = time.monotonic()
            try:
                item = await asyncio.wait_for(_next_chunk(iterator), remaining)
            except asyncio.TimeoutError:
                yield sse_event(
                    {"type": "error", "content": f"研究执行超时（>{timeout}s）"}
                )
                break
            if remaining is not None:
                remaining -= time.monotonic() - started

            if item is _STREAM_END:
                yield sse_event({"type": "result", "content": "".join(final_parts)})
                break

            chunk, meta = item
            if messages.is_final_report_chunk(chunk, meta) and isinstance(
                chunk.content, str
            ):
                final_parts.append(chunk.content)
            # 这一行 yield 的阻塞时间不计入预算：那是客户端读得慢，不是研究慢
            yield sse_event(messages.chunk_to_event(chunk, meta))
    except GeneratorExit:
        aborted = True  # 客户端断开
        raise
    except asyncio.CancelledError:
        aborted = True
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("研究流式执行失败")
        yield sse_event({"type": "error", "content": str(exc)})
    finally:
        if not aborted:
            await _aclose_quietly(iterator)
            yield _DONE_CHUNK


def sse_event(payload: dict[str, Any]) -> str:
    """把事件体编码为一条 SSE 消息。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------- #
# 向后兼容别名（旧代码 / 测试引用的是下划线命名）
# --------------------------------------------------------------------------- #
_build_checkpointer = build_checkpointer
_build_store = build_store
_get_research_graph = get_research_graph
