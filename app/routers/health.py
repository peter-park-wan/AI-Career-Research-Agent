"""服务元信息、存活探针与就绪探针。

区分 ``/health``（liveness）与 ``/ready``（readiness）：

* liveness 只回答"进程还活着吗"，**不能**依赖任何外部资源，
  否则数据库抖动会让编排器重启一个本来健康的进程。
* readiness 回答"现在能接流量吗"，需要检查关键依赖的配置是否到位。

``/ready`` 刻意不做真实连接（连 Postgres 会引入启动延迟与抖动），
只校验"配置是否自洽 + 数据目录是否可写"，这是单副本部署下最实用的信号。
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.core.config import Settings
from app.core.deps import get_settings_dep
from app.services import research_service

router = APIRouter(tags=["meta"])

ENDPOINTS: Dict[str, str] = {
    "research": "POST /api/research | POST /api/research/stream",
    "career_workflow": "POST /api/career/workflow",
    "gap_analysis": "POST /api/career/gap-analysis",
    "discover_jobs": "POST /api/career/discover-jobs",
    "interview": "POST /api/career/interview",
    "resume": "POST /api/career/resume",
    "cover_letter": "POST /api/career/cover-letter",
    "profile": (
        "GET /api/profile | GET /api/profile/names | POST /api/profile | "
        "POST /api/profile/activate | POST /api/profile/upload | DELETE /api/profile"
    ),
    "health": "GET /health | GET /ready",
}


@router.get("/", tags=["meta"])
async def root() -> Dict[str, Any]:
    return {
        "service": "AI Career Research Agent API",
        "version": "1.1.0",
        "endpoints": ENDPOINTS,
    }


@router.get("/health", tags=["meta"])
async def health() -> Dict[str, str]:
    """存活探针：无外部依赖，永远轻量返回。"""
    return {"status": "ok"}


@router.get("/ready", tags=["meta"])
async def ready(settings: Settings = Depends(get_settings_dep)) -> Dict[str, Any]:
    """就绪探针：检查配置自洽性与数据目录可写性。"""
    checks: Dict[str, Any] = {
        "env": settings.env,
        "auth_enabled": bool(settings.bearer_token),
        "checkpointer": settings.checkpointer,
        "memory_store": settings.memory_store,
        "profile_dir_writable": _is_writable(settings.profile_dir),
        "resume_path_exists": os.path.isfile(settings.resume_path),
    }
    # 持久化后端的"声明 vs 实际"。配置写的是 postgres，实际跑的是 memory
    # 也必须在这里暴露出来，否则监控看到的是一个"健康"的假象。
    runtime = research_service.runtime_status(settings)
    checks["graph"] = runtime

    # 生产环境配置有硬伤时直接报 not ready，避免带着坏配置接流量
    problems = settings.production_problems() if settings.is_prod else []
    checks["config_problems"] = problems
    healthy = (
        not problems
        and checks["profile_dir_writable"]
        and not runtime["degraded"]
    )
    checks["status"] = "ready" if healthy else "degraded"
    return checks


def _is_writable(directory: str) -> bool:
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(directory, ".write_probe")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False
