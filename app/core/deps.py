"""FastAPI 依赖：配置注入与 Bearer 鉴权。

把这两件事放进 ``deps`` 而不是写在路由里，是为了：

* **配置可替换**：路由只声明 ``Depends(get_settings_dep)``，测试里可以直接
  ``app.dependency_overrides[get_settings_dep] = lambda: Settings(...)`` 造场景，
  不需要改环境变量再 reload 整个模块。
* **鉴权只有一处实现**：改造前是路由上挂 ``dependencies=[Depends(require_token)]``，
  但校验逻辑写在 ``api.py`` 顶部、直接比较字符串常量。这里换成
  ``secrets.compare_digest``，消除时序侧信道（逐字符 ``!=`` 会因为提前返回
  而泄漏"前 N 位猜对了"的信息）。
"""

from __future__ import annotations

import secrets

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.errors import UnauthorizedError


def get_settings_dep() -> Settings:
    """请求级配置依赖（内含进程级缓存，开销可忽略）。"""
    return get_settings()


async def require_token(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> None:
    """校验 ``Authorization: Bearer <token>``。

    未配置 ``API_BEARER_TOKEN`` 时视为不开启鉴权（dev/内网场景）。
    """
    expected = settings.bearer_token
    if not expected:
        return

    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    # 常量时间比较，避免通过响应耗时逐位爆破 token
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        token.encode("utf-8"), expected.encode("utf-8")
    ):
        raise UnauthorizedError("无效或缺失的 Bearer Token")
