"""求职业务服务：Gap 分析缓存 + 各阶段 LLM 调用的统一入口。

【分层约束】本模块**不导入 fastapi**。它只抛 ``AppError`` 家族的类型化错误，
HTTP 状态码由 ``errors.register_exception_handlers`` 统一翻译。
好处是这层逻辑可以被 CLI / 定时任务 / 消息队列复用，也可以脱离 HTTP 单测。

【Gap 分析缓存】

``gap-analysis`` / ``interview`` / ``resume`` 三个接口都要先跑一次 Gap 分析。
同一份 JD + 同一份画像重复跑 LLM 是纯浪费（一次几秒到几十秒）。

缓存键是 ``(jd_text, configurable)`` 的 sha256 —— 必须包含 ``configurable``，
因为画像、模型、检索量都可能不同，只按 JD 缓存会串结果。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

from app.core.config import Settings, get_settings
from app.core.errors import UpstreamError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Gap 分析进程内 LRU 缓存
# --------------------------------------------------------------------------- #
_GAP_CACHE: "OrderedDict[str, str]" = OrderedDict()
_GAP_CACHE_LOCK = asyncio.Lock()


def gap_cache_key(jd_text: str, config: Optional[Dict[str, Any]]) -> str:
    configurable = (config or {}).get("configurable", {}) or {}
    payload = {"jd_text": jd_text, "configurable": configurable}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def run_gap_analysis_cached(
    jd_text: str,
    config: Dict[str, Any],
    settings: Optional[Settings] = None,
) -> str:
    """调用 ``run_gap_analysis``，对相同输入做进程内缓存。

    注意：缓存基于 ``jd_text`` 与 ``config.configurable``（含画像 / 模型等）。
    若简历文件在进程内被外部修改，请设置 ``API_GAP_CACHE=off`` 或重启服务。
    """
    settings = settings or get_settings()
    if not settings.gap_cache:
        return await _call_gap_analysis(jd_text, config)

    key = gap_cache_key(jd_text, config)
    async with _GAP_CACHE_LOCK:
        cached = _GAP_CACHE.get(key)
        if cached is not None:
            _GAP_CACHE.move_to_end(key)
            return cached

    # 刻意在锁外执行 LLM 调用：否则一次慢请求会阻塞所有缓存读取
    result = await _call_gap_analysis(jd_text, config)

    async with _GAP_CACHE_LOCK:
        _GAP_CACHE[key] = result
        _GAP_CACHE.move_to_end(key)
        while len(_GAP_CACHE) > settings.gap_cache_max > 0:
            _GAP_CACHE.popitem(last=False)
    return result


async def _call_gap_analysis(jd_text: str, config: Dict[str, Any]) -> str:
    # 懒加载：保持模块导入轻量，也让测试可以 patch 目标模块
    from open_deep_research.gap_analysis import run_gap_analysis

    return await run_gap_analysis(jd_text, config)


def clear_gap_cache() -> None:
    _GAP_CACHE.clear()


# --------------------------------------------------------------------------- #
# 画像 / 各阶段能力（薄封装 + 错误分类）
# --------------------------------------------------------------------------- #
def load_profile(config: Dict[str, Any]) -> str:
    """读取候选人画像（inline → 简历文件 → 长期记忆）。"""
    from open_deep_research.profile import load_user_profile

    return load_user_profile(config)


async def discover_jobs(
    target_role: str, target_city: str, config: Dict[str, Any]
) -> str:
    from open_deep_research.career_workflow import run_discover_jobs

    profile = load_profile(config)
    return await _guard(
        "岗位发现",
        run_discover_jobs(profile, target_role, target_city, config),
    )


async def interview_prep(
    target_role: str,
    jd_text: Optional[str],
    config: Dict[str, Any],
) -> str:
    from open_deep_research.career_workflow import run_interview_prep

    gap = await run_gap_analysis_cached(_jd_or_role(jd_text, target_role), config)
    return await _guard(
        "面试准备",
        run_interview_prep(target_role, load_profile(config), gap, config),
    )


async def resume_optimization(
    target_role: str,
    jd_text: Optional[str],
    config: Dict[str, Any],
) -> str:
    from open_deep_research.career_workflow import run_resume_optimization

    gap = await run_gap_analysis_cached(_jd_or_role(jd_text, target_role), config)
    return await _guard(
        "简历优化",
        run_resume_optimization(target_role, load_profile(config), gap, config),
    )


async def cover_letter(target_role: str, config: Dict[str, Any]) -> str:
    from open_deep_research.career_workflow import run_cover_letter

    return await _guard(
        "求职信生成",
        run_cover_letter(target_role, load_profile(config), config),
    )


async def career_workflow(
    target_role: str,
    target_city: str,
    jd_text: str,
    config: Dict[str, Any],
) -> Any:
    from open_deep_research.career_workflow import run_career_workflow

    return await _guard(
        "端到端工作流",
        run_career_workflow(
            target_role=target_role,
            target_city=target_city,
            jd_text=jd_text or "",
            config=config,
        ),
    )


def _jd_or_role(jd_text: Optional[str], target_role: str) -> str:
    """没有具体 JD 时，用目标岗位合成一个最小 JD，让 Gap 分析仍有输入。"""
    return jd_text or f"目标岗位：{target_role}"


async def _guard(stage: str, awaitable: Any) -> Any:
    """把未预期的上游异常统一翻译为 ``UpstreamError``（502 + 可重试语义）。

    只吞 ``Exception``，``asyncio.CancelledError``（BaseException）会自然向上传播，
    否则客户端断开连接时任务无法被取消。
    """
    try:
        return await awaitable
    except UpstreamError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s 阶段失败", stage)
        raise UpstreamError(stage, str(exc)) from exc
