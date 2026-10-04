"""类型化错误体系 + 全局异常处理器。

【本模块要解决的三个真实问题】

问题 1：错误不可编程
    改造前 ``api.py`` 里到处是这种写法（L530 / L872 / L892 …）：

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"研究执行失败：{e}")

    两个后果：
      a) 客户端拿到的是**内部异常原文**（可能是文件路径、SQL 片段、SDK 内部信息），
         既泄漏实现细节，也无法做稳定的错误分支；
      b) 不同接口的响应体结构不一致：有的 ``{"detail": "..."}``，
         pydantic 校验失败却是 ``{"detail": [{"loc": [...], "msg": "..."}]}``。
         前端只能靠 if-else 猜结构。

问题 2：运维/业务错误混在一起
    "用户上传了 6MB 文件"和"Postgres 连不上"都被压成 500。
    前者不该告警，后者必须告警。所以错误必须**分类**，并携带"是否可重试"语义。

问题 3：错误发生点没有上下文
    排障时需要 ``request_id``，才能在成千上万行日志里把一次请求串起来。
    本模块不自己生成 id，而是从 ``request.state`` 读取（中间件负责写入），
    这样 errors 模块不依赖日志模块，保持单向依赖。

【响应体约定】
统一为 RFC7807 风格的四字段：

    {"title": "NOT_FOUND", "status": 404, "detail": "档案不存在：张三", "request_id": "..."}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 错误基类与子类
# --------------------------------------------------------------------------- #
class AppError(Exception):
    """所有业务错误的基类。

    ``status_code`` / ``code`` 定义在类上而不是实例上，好处是：
    调用方只需 ``raise NotFoundError("档案", name)``，不用每次重复写 404。
    """

    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    # 是否属于"可预期"的业务错误。
    # True  → 业务错误，正常记 warning，不触发告警，不打印堆栈
    # False → 程序缺陷（bug），记 error + 堆栈，需要人工介入
    operational: bool = True

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_body(self, request_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "title": self.code,
            "status": self.status_code,
            "detail": self.message,
            "request_id": request_id,
        }
        if self.details is not None:
            body["details"] = self.details
        return body


class ConfigError(AppError):
    """进程级配置缺失或非法。启动阶段抛出 → 直接终止进程（fail fast）。"""

    status_code = 500
    code = "CONFIG_ERROR"
    operational = False  # 配置错误属于部署缺陷，需要告警


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"

    def __init__(self, resource: str, key: str = "") -> None:
        super().__init__(f"{resource}不存在" + (f"：{key}" if key else ""))
        self.resource = resource


class BadRequestError(AppError):
    status_code = 400
    code = "BAD_REQUEST"


class ValidationError(AppError):
    """入参未通过业务校验（与 pydantic 的 schema 校验区分开）。"""

    status_code = 422
    code = "VALIDATION_ERROR"


class UnauthorizedError(AppError):
    status_code = 401
    code = "UNAUTHORIZED"


class ForbiddenError(AppError):
    status_code = 403
    code = "FORBIDDEN"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "PAYLOAD_TOO_LARGE"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "UNSUPPORTED_MEDIA_TYPE"


class RateLimitedError(AppError):
    status_code = 429
    code = "RATE_LIMITED"

    def __init__(self, retry_after_seconds: int, message: str = "请求过于频繁，请稍后再试") -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class UpstreamError(AppError):
    """依赖的外部服务失败：LLM / 搜索 API / 数据库。

    单独成类是为了让重试策略能按类型判断：
    上游错误通常可重试，业务校验错误重试一万次也没用。
    """

    status_code = 502
    code = "UPSTREAM_ERROR"

    def __init__(self, service: str, message: str, *, details: Any = None) -> None:
        super().__init__(f"{service} 调用失败：{message}", details=details)
        self.service = service


class UpstreamTimeoutError(AppError):
    """上游在约定时限内未返回。

    与请求方超时（408）区分开：408 表示"你发得太慢"，
    504 表示"我们的依赖太慢"，后者的重试通常有意义。
    """

    status_code = 504
    code = "UPSTREAM_TIMEOUT"

    def __init__(self, service: str, timeout_seconds: int = 0) -> None:
        detail = f"在 {timeout_seconds}s 内未完成" if timeout_seconds else "执行超时"
        super().__init__(f"{service} 超时：{detail}")
        self.service = service
        self.timeout_seconds = timeout_seconds


RETRYABLE_CODES = frozenset({"UPSTREAM_ERROR", "UPSTREAM_TIMEOUT", "RATE_LIMITED"})


def is_retryable(exc: BaseException) -> bool:
    """给客户端 / 任务队列判断"这个错误值不值得重试"。"""
    if isinstance(exc, AppError):
        return exc.code in RETRYABLE_CODES
    return False


# --------------------------------------------------------------------------- #
# 全局异常处理器
# --------------------------------------------------------------------------- #
def _request_id(request: Request) -> str | None:
    """从请求上下文取 request_id（由日志中间件写入 ``request.state``）。"""
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器。

    为什么必须有它：任何一个未被捕获的异常，如果交给框架默认处理，
    响应体会是框架自带的格式，且**不会经过你的日志**。
    有了它，所有出口被收敛到一处：统一响应体 + 统一分级日志。
    """

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        rid = _request_id(request)
        # 业务错误（4xx）记 warning 就够；程序缺陷记 error 并带堆栈
        if exc.operational:
            logger.warning(
                "业务错误", extra={"code": exc.code, "status": exc.status_code, "path": request.url.path}
            )
        else:
            logger.error(
                "服务内部错误",
                exc_info=exc,
                extra={"code": exc.code, "status": exc.status_code, "path": request.url.path},
            )
        headers = {}
        if isinstance(exc, RateLimitedError):
            headers["Retry-After"] = str(exc.retry_after_seconds)
        return JSONResponse(status_code=exc.status_code, content=exc.to_body(rid), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _handle_schema_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """把 pydantic 的 schema 错误整形为统一的字段级错误列表。

        整形后的 ``fields`` 可以直接被前端用来把错误贴到对应输入框旁边，
        而不需要前端自己解析 ``loc`` 数组。
        """
        rid = _request_id(request)
        fields = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ()) if part != "body"),
                "message": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()
        ]
        logger.warning(
            "入参校验失败", extra={"path": request.url.path, "fields": len(fields)}
        )
        return JSONResponse(
            status_code=422,
            content={
                "title": "VALIDATION_ERROR",
                "status": 422,
                "detail": "请求参数不合法",
                "request_id": rid,
                "fields": fields,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """兜住框架自身抛出的 HTTPException（如 404 路由未命中）。"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "title": f"HTTP_{exc.status_code}",
                "status": exc.status_code,
                "detail": str(exc.detail),
                "request_id": _request_id(request),
            },
            headers=getattr(exc, "headers", None) or None,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """最后一道防线：任何未预期异常都在这里被记录并转换成 500。

        注意：**绝不把 ``str(exc)`` 返回给客户端**，只返回一个安全的话术；
        真正的异常内容进日志（配 ``exc_info`` 保留堆栈），靠 request_id 关联。
        """
        logger.error(
            "未捕获异常",
            exc_info=exc,
            extra={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(
            status_code=500,
            content={
                "title": "INTERNAL_ERROR",
                "status": 500,
                "detail": "服务内部错误，请稍后重试",
                "request_id": _request_id(request),
            },
        )
