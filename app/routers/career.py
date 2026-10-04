"""求职场景单点能力接口：岗位发现 / 匹配度 / 面试 / 简历 / 求职信 / 端到端工作流。

这些端点几乎只是"绑定参数 → 调 service → 组装响应"，因此路由里不出现任何
``try/except``：错误分类与状态码映射统一由 ``errors.register_exception_handlers``
完成。改造前每个端点各写一次 ``except Exception as e: raise HTTPException(500, f"...{e}")``，
既泄漏内部信息，也让"哪些错误可重试"无从判断。
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from app.core.deps import get_settings_dep, require_token
from app.core.config import Settings
from app.schemas.requests import (
    CareerStageRequest,
    CareerWorkflowRequest,
    GapAnalysisRequest,
)
from app.services import career_service
from app.services.messages import build_runnable_config

router = APIRouter(
    prefix="/api/career",
    tags=["career"],
    dependencies=[Depends(require_token)],
)


def _config(body: Any) -> Dict[str, Any]:
    return build_runnable_config(body.configurable, body.thread_id)


@router.post("/gap-analysis")
async def gap_analysis(
    body: GapAnalysisRequest,
    settings: Settings = Depends(get_settings_dep),
) -> Dict[str, Any]:
    """针对具体 JD 做人岗匹配度分析（结果进程内缓存，供面试/简历复用）。"""
    cfg = _config(body)
    result = await career_service.run_gap_analysis_cached(body.jd_text, cfg, settings)
    return {"jd_text": body.jd_text, "gap_analysis": result}


@router.post("/discover-jobs")
async def discover_jobs(body: CareerStageRequest) -> Dict[str, Any]:
    """岗位发现：推荐优先投递的岗位方向与公司类型。"""
    cfg = _config(body)
    result = await career_service.discover_jobs(body.target_role, body.target_city, cfg)
    return {
        "target_role": body.target_role,
        "target_city": body.target_city,
        "discovery": result,
    }


@router.post("/interview")
async def interview(body: CareerStageRequest) -> Dict[str, Any]:
    """面试准备：结合目标岗位与匹配度分析，生成备考清单。"""
    cfg = _config(body)
    result = await career_service.interview_prep(body.target_role, body.jd_text, cfg)
    return {"target_role": body.target_role, "interview_prep": result}


@router.post("/resume")
async def resume(body: CareerStageRequest) -> Dict[str, Any]:
    """简历优化：针对目标岗位给出策略 + 改写后的核心经历 bullet。"""
    cfg = _config(body)
    result = await career_service.resume_optimization(body.target_role, body.jd_text, cfg)
    return {"target_role": body.target_role, "resume_optimization": result}


@router.post("/cover-letter")
async def cover_letter(body: CareerStageRequest) -> Dict[str, Any]:
    """求职信：为目标岗位写一封中文求职信。"""
    cfg = _config(body)
    result = await career_service.cover_letter(body.target_role, cfg)
    return {"target_role": body.target_role, "cover_letter": result}


@router.post("/workflow")
async def workflow(body: CareerWorkflowRequest) -> Dict[str, Any]:
    """端到端求职工作流：岗位发现 → 调研 → 匹配度 → 面试准备 → 简历优化 → 求职信。"""
    cfg = _config(body)
    result = await career_service.career_workflow(
        target_role=body.target_role,
        target_city=body.target_city,
        jd_text=body.jd_text or "",
        config=cfg,
    )
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
