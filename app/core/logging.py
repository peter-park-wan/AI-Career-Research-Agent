"""结构化日志 + 请求上下文（request_id）。

【要解决的问题】

改造前的日志是 ``logging.getLogger(__name__)`` 加零散 ``logger.warning``：

1. **串不起来**：用户报"我那次请求失败了"，日志里只有一条
   ``Postgres checkpointer 初始化失败``，无法知道它属于哪一次请求。
   本模块用 ``RequestIdMiddleware`` 给每个请求分配 id，写入
   ``request.state.request_id``（给 ``errors.py`` 用）与 ``ContextVar``（给日志用），
   响应头回写 ``X-Request-ID``，于是"日志 ↔ 响应 ↔ 客户端上报"三者可对齐。
2. **机器不可读**：多行文本日志难以被采集系统按字段检索。``JsonFormatter``
   输出单行 JSON，``level`` / ``request_id`` / ``path`` / ``duration_ms`` 都是字段。
3. **敏感信息落盘**：``Authorization`` 头一旦被写进日志就等于凭据泄漏。

依赖方向是单向的：``logging`` 不 import ``errors``。
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

from app.core.config import Settings

# 请求级上下文：同一请求内任意深度均可读取，无需层层传参
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# 额外需要落到日志里的结构化字段（来自 extra={...}）
_EXTRA_FIELDS = (
    "code",
    "status",
    "path",
    "method",
    "client_ip",
    "duration_ms",
    "fields",
    "service",
)

_SENSITIVE_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "cookie", "x-api-key", "x-supabase-access-token"}
)
_MASK = "***"


def redact_headers(headers: Any) -> dict[str, str]:
    """返回脱敏后的请求头副本，供调试日志使用。"""
    safe: dict[str, str] = {}
    for key, value in dict(headers).items():
        safe[key] = _MASK if key.lower() in _SENSITIVE_HEADERS else value
    return safe


class _RequestContextFilter(logging.Filter):
    """把 contextvar 里的 request_id 注入到每条 record，供文本格式使用。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """单行 JSON 日志。"""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(settings: Settings) -> None:
    """按配置初始化根 logger。幂等：重复调用只保留一个 handler。"""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestContextFilter())
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
            )
        )

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # uvicorn 已有自己的 access log，我们中间件里也记录了耗时，避免重复
    for noisy, noisy_level in (
        ("uvicorn.access", logging.WARNING),
        ("httpx", logging.WARNING),
        ("httpcore", logging.WARNING),
        ("urllib3", logging.WARNING),
    ):
        logging.getLogger(noisy).setLevel(noisy_level)


logger = logging.getLogger("app.http")

# 健康检查/文档这类高频且无信息量的路径不记访问日志
_QUIET_PATHS = frozenset({"/", "/health", "/ready", "/docs", "/openapi.json", "/redoc"})


class RequestIdMiddleware(BaseHTTPMiddleware):
    """注入 request_id、记录访问日志、回写 ``X-Request-ID``。"""

    def __init__(self, app: ASGIApp, *, quiet_paths: frozenset[str] = _QUIET_PATHS) -> None:
        super().__init__(app)
        self._quiet_paths = quiet_paths

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        # 允许上游网关传入以便跨服务串联，否则自行生成
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)

        started = time.perf_counter()
        client_ip = _client_ip(request)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "请求处理异常",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "client_ip": client_ip,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            if request.url.path not in self._quiet_paths:
                logger.info(
                    "请求完成",
                    extra={
                        "request_id": request_id,
                        "path": request.url.path,
                        "method": request.method,
                        "status": response.status_code,
                        "client_ip": client_ip,
                        "duration_ms": _elapsed_ms(started),
                    },
                )
            return response
        finally:
            # 必须在所有日志之后 reset：顺序反了会让访问日志里的 request_id 退化成 "-"，
            # 那样"日志可串联"这个目标就落空了。
            request_id_var.reset(token)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _client_ip(request: Request) -> str:
    """取真实客户端 IP：优先 X-Forwarded-For 的第一跳。"""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"
