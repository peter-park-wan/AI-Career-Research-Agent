"""持久化后端"声明 vs 实际"的一致性保证。

【要防的事故】

配置写 ``API_CHECKPOINTER=postgres``，但 DSN 缺失或数据库连不上。
改造前 ``build_checkpointer`` 会静默返回一个 ``MemorySaver``：

* 服务照常启动、接口照常返回 200；
* 多轮会话记忆全部落在单个进程的内存里，重启即丢、多副本各存各的；
* 没有任何告警，只有一条 warning 级别的日志。

也就是说，**监控系统看到的是一个健康的服务，但它已经不满足它被部署
出来要满足的那个前提（持久化）**。这比直接起不来危险得多。

本文件锁定的行为：

* 生产环境：装配失败 = fail fast，进程起不来；
* 其它环境 / 显式开 ``API_ALLOW_MEMORY_FALLBACK``：允许降级，但必须留痕，
  ``/ready`` 报 ``degraded``；
* 任何降级相关的日志与报错都不能泄漏连接串凭据。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import Settings
from app.core.deps import get_settings_dep
from app.core.errors import ConfigError
from app.main import app as main_app
from app.main import create_app
from app.services import research_service
from app.services.rate_limit import _RATE_LIMITER as _GLOBAL_LIMITER


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    """每个用例前后清空装配留痕，避免用例间互相污染。"""
    research_service.reset_research_graph()
    yield
    research_service.reset_research_graph()


def _prod_settings(**overrides) -> Settings:
    return Settings(
        env="prod",
        bearer_token="test-token",
        cors_origins=["https://example.com"],
        **overrides,
    )


def test_dev_still_falls_back_to_memory():
    """开发环境下没配 DSN 也要能把服务跑起来（不阻断本地开发）。"""
    settings = Settings(checkpointer="postgres")
    assert isinstance(
        research_service.build_checkpointer("postgres", settings), MemorySaver
    )


def test_dev_fallback_is_recorded_not_silent():
    """降级本身可以发生，但必须留痕：effective 要如实反映用的是 memory。"""
    settings = Settings(checkpointer="postgres")
    research_service.build_checkpointer("postgres", settings)

    status = research_service.runtime_status(settings)
    assert status["effective"]["checkpointer"] == "memory"
    assert status["degraded"] is True
    assert status["degraded_reasons"]


def test_prod_rejects_silent_fallback():
    """生产环境：声明 postgres 却装配不上，直接 fail fast，不许悄悄降级。"""
    settings = _prod_settings(checkpointer="postgres")
    with pytest.raises(ConfigError) as excinfo:
        research_service.build_checkpointer("postgres", settings)
    assert "checkpointer" in str(excinfo.value)


def test_prod_fallback_requires_explicit_opt_in():
    """显式开 ``API_ALLOW_MEMORY_FALLBACK`` 才能降级，且仍然被标记为 degraded。"""
    settings = _prod_settings(checkpointer="postgres", allow_memory_fallback=True)
    checkpointer = research_service.build_checkpointer("postgres", settings)

    assert isinstance(checkpointer, MemorySaver)
    assert research_service.runtime_status(settings)["degraded"] is True


def test_prod_store_rejects_silent_fallback():
    """同样的保证覆盖长期记忆 Store，而不只是 checkpointer。"""
    settings = _prod_settings(memory_store="postgres")
    with pytest.raises(ConfigError):
        research_service.build_store(settings)


def test_dsn_password_never_reaches_logs_or_errors():
    """装配失败的异常原文常含整条 DSN，脱敏后再进日志与报错。

    日志系统的访问权限通常比数据库本身宽松得多，把密码写进去等于
    把数据库凭据广播给了所有能看日志的人。
    """
    dsn = "postgresql://app:sup3rs3cr3t@db.internal:5432/langgraph"
    settings = Settings(
        env="prod", checkpointer="postgres", checkpointer_postgres_dsn=dsn
    )

    # 精确替换：settings 里已知的连接串
    assert "sup3rs3cr3t" not in research_service._redact(f"failed: {dsn}", settings)
    # 兜底替换：异常里冒出的、settings 不知道的连接串
    assert "pwd123" not in research_service._mask_url("redis://u:pwd123@h:6379/0")
    # 正常内容不应被误伤
    assert research_service._mask_url("连接超时") == "连接超时"


def test_memory_store_accepts_every_alias_form():
    """``MEMORY_STORE`` 的三种写法都必须生效，任何一种静默失效都会导致
    "以为配了 postgres、实际跑的是 memory"。"""
    assert Settings(memory_store="postgres").memory_store == "postgres"
    assert Settings(MEMORY_STORE="postgres").memory_store == "postgres"
    assert Settings(API_MEMORY_STORE="postgres").memory_store == "postgres"
    assert Settings().memory_store == "memory"


def test_ready_reports_degraded_backend():
    """/ready 必须把"声明 postgres、实际 memory"暴露出来，让编排器摘掉流量。"""
    settings = Settings(
        checkpointer="postgres",
        profile_dir="./data/profiles",
        resume_path="./data/简历.md",
    )
    research_service.build_checkpointer("postgres", settings)

    main_app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        body = TestClient(main_app).get("/ready").json()
    finally:
        main_app.dependency_overrides.clear()

    assert body["status"] == "degraded"
    assert body["graph"]["declared"]["checkpointer"] == "postgres"
    assert body["graph"]["effective"]["checkpointer"] == "memory"


def test_prod_startup_fails_fast_when_backend_unavailable():
    """最关键的保证：装配失败发生在启动阶段，而不是等第一个用户请求。"""
    settings = _prod_settings(
        checkpointer="postgres",
        profile_dir="./data/profiles",
        resume_path="./data/简历.md",
    )
    prod_app = create_app(settings)
    try:
        with pytest.raises(ConfigError):
            with TestClient(prod_app):
                pass  # lifespan 在此触发预热，装配失败应当终止启动
    finally:
        # create_app 会改全局限流器，还原以免影响其它用例
        from app.services import rate_limit

        rate_limit._RATE_LIMITER = _GLOBAL_LIMITER
