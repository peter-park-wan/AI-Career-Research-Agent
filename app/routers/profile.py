"""画像 / 简历档案相关接口。

这一组接口是**补齐的缺口**：``streamlit_app.py`` 一直调用
``/api/profile/names``、``/api/profile/activate``、``POST /api/profile``、
``/api/profile/upload``、``DELETE /api/profile``，但后端只实现了 ``GET /api/profile``，
前端"档案切换 / 上传 / 在线编辑"整块功能实际处于不可用状态。

同时把 ``DELETE`` 的语义收紧：删除正在生效的档案会被拒绝（业务规则在 service 层），
否则"生效中的档案"会指向一个不存在的文件，后续所有分析静默退化成无画像。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.deps import get_settings_dep, require_token
from app.core.errors import ValidationError
from app.core.config import Settings
from app.schemas.requests import (
    ProfileActivateRequest,
    ProfileSaveRequest,
)
from app.services.career_service import load_profile
from app.services.messages import build_runnable_config
from app.services.profile_service import ProfileService, build_profile_service

router = APIRouter(prefix="/api", tags=["profile"])


def get_profile_service(
    settings: Settings = Depends(get_settings_dep),
) -> ProfileService:
    """服务工厂依赖：无状态，按请求构造（仅持有路径，开销可忽略）。"""
    return build_profile_service(settings)


def _parse_configurable(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """解析 query 里的 configurable（JSON 字符串）。"""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"configurable 不是合法 JSON：{exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValidationError("configurable 必须是 JSON 对象")
    return parsed


@router.get("/profile", dependencies=[Depends(require_token)])
async def get_profile(
    name: Optional[str] = None,
    configurable: Optional[str] = None,
    service: ProfileService = Depends(get_profile_service),
) -> Dict[str, Any]:
    """读取画像。

    * 带 ``name``：直接读该档案文件（前端"在线编辑"用）
    * 不带 ``name``：返回**当前生效**的画像，与 Agent 运行时同源
      （inline → 简历文件 → 长期记忆）。这样前端看到的就是模型实际看到的。
    """
    if name:
        return {"profile": await service.read(name)}

    cfg = build_runnable_config(_parse_configurable(configurable), None)
    # 这里保持同步调用：load_user_profile 会回退到长期记忆 Store，
    # 而 Store 的可用性依赖请求上下文，放到线程池反而会丢失上下文。
    return {"profile": load_profile(cfg)}


@router.get("/profile/names", dependencies=[Depends(require_token)])
async def list_profiles(
    service: ProfileService = Depends(get_profile_service),
) -> Dict[str, Any]:
    return await service.list_profiles()


@router.post("/profile", dependencies=[Depends(require_token)])
async def save_profile(
    body: ProfileSaveRequest,
    service: ProfileService = Depends(get_profile_service),
) -> Dict[str, Any]:
    """新建档案或保存内容（``name`` 为空时写入当前生效的简历文件）。"""
    return await service.save(body.content, body.name)


@router.post("/profile/activate", dependencies=[Depends(require_token)])
async def activate_profile(
    body: ProfileActivateRequest,
    service: ProfileService = Depends(get_profile_service),
) -> Dict[str, Any]:
    """切换生效档案：把档案内容写入简历文件并记录 active 标记。"""
    return await service.activate(body.name)


@router.post("/profile/upload", dependencies=[Depends(require_token)])
async def upload_profile(
    file: UploadFile = File(..., description="简历文件（.md / .markdown / .txt / .pdf）"),
    name: Optional[str] = None,
    service: ProfileService = Depends(get_profile_service),
) -> Dict[str, Any]:
    """上传简历并解析为文本，写入指定档案（``name`` 为空时写入生效文件）。"""
    data = await file.read()
    return await service.upload(file.filename or "", data, name)


@router.delete("/profile", dependencies=[Depends(require_token)])
async def delete_profile(
    name: str,
    service: ProfileService = Depends(get_profile_service),
) -> Dict[str, Any]:
    return await service.delete(name)
