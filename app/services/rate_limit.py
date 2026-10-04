"""限流：进程内滑动窗口 + 中间件。

【为什么自己写而不是引三方】

单副本部署下，20 行代码即可覆盖需求。生产多副本时应替换为 Redis 共享计数，
因此这里刻意留出 ``InMemoryRateLimiter`` 这个可替换的接口。

【修掉的两个毛病】

1. **桶不会回收**。原实现 ``_RATE_BUCKETS.setdefault(ip, [])`` 只增不减，
   每个访问过的 IP 都永久占一个 key —— 被扫描器打一轮就是内存泄漏。
   现在到期的桶会被删除。
2. **健康检查路径未豁免准确**。原实现豁免 ``/``，但 ``/ready`` 等新探针需要
   一并豁免，否则 K8s 探针会被限流判定为"服务不可用"而触发重启风暴。
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from app.core.errors import RateLimitedError

RATE_WINDOW_SECONDS = 60.0

# 不计入限流的路径：探针与文档
EXEMPT_PATHS = frozenset(
    {"/", "/health", "/ready", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}
)


class InMemoryRateLimiter:
    """按 key（客户端 IP）限流的滑动窗口计数器。"""

    def __init__(self, limit: int = 0, window: float = RATE_WINDOW_SECONDS) -> None:
        self.limit = max(int(limit), 0)
        self.window = window
        self._hits: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    @property
    def buckets(self) -> Dict[str, List[float]]:
        return self._hits

    async def check(self, key: str) -> Optional[int]:
        """记录一次访问。返回 ``None`` 表示放行，否则返回建议的重试秒数。"""
        if self.limit <= 0:
            return None

        now = time.monotonic()
        cutoff = now - self.window
        async with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                self._hits[key] = [now]
                return None

            hits[:] = [ts for ts in hits if ts > cutoff]
            if not hits:
                # 回收空桶，避免被扫描器打爆内存
                del self._hits[key]
                self._hits[key] = [now]
                return None
            if len(hits) >= self.limit:
                retry_after = int(max(hits[0] + self.window - now, 1))
                return retry_after
            hits.append(now)
        return None

    def reset(self) -> None:
        self._hits.clear()


# 进程级单例：中间件每次请求都从模块全局取，便于测试替换
_RATE_LIMITER = InMemoryRateLimiter(limit=0)


def get_rate_limiter() -> InMemoryRateLimiter:
    return _RATE_LIMITER


def configure_rate_limiter(limit: int, window: float = RATE_WINDOW_SECONDS) -> InMemoryRateLimiter:
    """按配置重建限流器（应用启动时调用一次）。"""
    global _RATE_LIMITER
    _RATE_LIMITER = InMemoryRateLimiter(limit=limit, window=window)
    return _RATE_LIMITER


class RateLimitMiddleware:
    """纯 ASGI 中间件（不用 BaseHTTPMiddleware，避免多包一层 task 开销）。"""

    def __init__(self, app, *, exempt_paths: frozenset[str] = EXEMPT_PATHS) -> None:
        self.app = app
        self.exempt_paths = exempt_paths

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limiter = get_rate_limiter()
        if limiter.limit <= 0 or scope.get("path") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        client_ip = _client_ip_from_scope(scope)
        retry_after = await limiter.check(client_ip)
        if retry_after is not None:
            error = RateLimitedError(retry_after)
            request_id = _request_id_from_scope(scope)
            body = _json_bytes(error.to_body(request_id))
            headers = [
                (b"content-type", b"application/json"),
                (b"retry-after", str(retry_after).encode()),
                (b"content-length", str(len(body)).encode()),
            ]
            if request_id:
                headers.append((b"x-request-id", request_id.encode()))
            await send({"type": "http.response.start", "status": 429, "headers": headers})
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


def _client_ip_from_scope(scope) -> str:
    headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = scope.get("client")
    return client[0] if client else "unknown"


def _request_id_from_scope(scope) -> str:
    # RequestIdMiddleware 是更外层中间件，正常路径下 state 已被写入
    state = scope.get("state") or {}
    return state.get("request_id", "") if isinstance(state, dict) else ""


def _json_bytes(payload: dict) -> bytes:
    import json

    return json.dumps(payload, ensure_ascii=False).encode("utf-8")
