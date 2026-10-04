"""应用组装层（composition root）。

整个进程只有这里知道"有哪些路由、哪些中间件、按什么顺序装配"。
``open_deep_research/api.py`` 退化为薄兼容层（``from app.main import app``），
老部署脚本 ``uvicorn open_deep_research.api:app`` 不受影响。

【中间件顺序】

Starlette 里"后添加的中间件更靠外"，因此装配顺序必须反过来读：

    添加顺序：CORS → RateLimit → RequestId
    实际执行：RequestId（最外）→ RateLimit → CORS → 路由

这样 request_id 在进入限流判断前就已生成，429 响应也带上 ``X-Request-ID``，
便于把"被限流的请求"和"上游网关日志"对上。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import RequestIdMiddleware, setup_logging
from app.routers import career, health, profile, research
from app.services import research_service
from app.services.rate_limit import RateLimitMiddleware, configure_rate_limiter

logger = logging.getLogger("app.main")

APP_VERSION = "1.1.0"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """按配置构造应用。抽出工厂函数后，测试可传入自定义 ``Settings``。"""
    settings = settings or get_settings()

    setup_logging(settings)
    configure_rate_limiter(settings.rate_limit_per_minute)

    # 浏览器规范不允许“任意来源 + 携带凭据”同时成立：
    # 若两者都为真，响应头会被浏览器直接拒绝。这里显式降级并告警，
    # 避免出现“本地能跑、线上跨域失败”的隐蔽问题。
    allow_credentials = settings.cors_allow_credentials
    if settings.allows_any_origin and allow_credentials:
        logger.warning(
            "CORS 配置为 '*' 且开启了 allow_credentials，浏览器会拒绝该组合，已自动关闭凭据"
        )
        allow_credentials = False

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "服务启动",
            extra={
                "service": "api",
                "env": settings.env,
                "auth_enabled": bool(settings.bearer_token),
                "checkpointer": settings.checkpointer,
                "memory_store": settings.memory_store,
                "rate_limit_per_minute": settings.rate_limit_per_minute,
            },
        )
        if settings.is_prod:
            # 预热研究图：把"后端装配失败"从"首个请求 500"提前到"进程起不来"。
            # 前者要等用户撞上才发现，后者会让编排器直接判定部署失败。
            await research_service.get_research_graph(settings)
            status = research_service.runtime_status(settings)
            logger.info(
                "生产环境配置校验通过，研究图预热完成",
                extra={
                    "effective_backends": status["effective"],
                    "degraded": status["degraded"],
                },
            )
        yield
        logger.info("服务停止", extra={"service": "api"})

    application = FastAPI(
        title="AI Career Research Agent API",
        version=APP_VERSION,
        description="基于 LangGraph 多 Agent 工作流的智能求职研究助手 HTTP 接口",
        lifespan=lifespan,
    )
    application.state.settings = settings

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(RequestIdMiddleware)

    register_exception_handlers(application)

    application.include_router(health.router)
    application.include_router(profile.router)
    application.include_router(career.router)
    application.include_router(research.router)

    return application


app = create_app()
