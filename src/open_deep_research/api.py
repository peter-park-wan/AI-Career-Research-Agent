"""FastAPI 服务入口（**薄兼容层**）。

改造前这里是 730 行的"上帝模块"：配置、中间件、鉴权、请求模型、消息转换、
checkpointer 装配、限流、业务编排、路由声明全挤在一个文件里。带来的实际成本是
**改任何一处都要读完整文件**，且无法对业务逻辑做不依赖 HTTP 的单元测试。

现在职责被拆到 ``app/`` 包：

    app/main.py            组装层（composition root）：路由 + 中间件 + 异常处理器
    app/core/              config（配置）/ logging（日志）/ errors（错误）/ deps（依赖）
    app/routers/           HTTP 层：参数绑定、鉴权声明、响应组装
    app/services/          业务层：编排领域逻辑，不 import fastapi
    app/repositories/      仓储层：唯一触碰文件系统/数据库的地方
    app/schemas/           对外契约（请求模型）

本文件只保留向后兼容的启动入口，既有部署命令无需改动：

    uvicorn open_deep_research.api:app --host 0.0.0.0 --port 8000
    python -m open_deep_research.api
"""

from __future__ import annotations

from app.main import app

__all__ = ["app", "main"]


def main() -> None:
    """本地启动入口：监听地址等参数统一从 ``Settings`` 读取。"""
    import uvicorn

    from app.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "open_deep_research.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
