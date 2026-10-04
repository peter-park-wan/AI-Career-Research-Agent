"""FastAPI 服务的冒烟测试（无需 API Key，LLM 调用已 mock）。

运行::

    pytest tests/test_api.py -q

若环境中未安装 fastapi / httpx，该测试会自动跳过。

【分层重构后的测试改动】
被测对象从「api 模块里的私有全局变量」换成「服务层的公开接口」：

    api_module._GAP_CACHE            → career_service.clear_gap_cache()
    api_module.run_gap_analysis_cached → career_service.run_gap_analysis_cached
    api_module._RATE_LIMIT / _RATE_BUCKETS → rate_limit._RATE_LIMITER
    api_module._build_checkpointer    → research_service.build_checkpointer

这本身就是重构收益：状态归位到它真正所属的模块，测试不再依赖"某个模块恰好
把变量挂在自己身上"这种巧合。
"""
import asyncio
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient  # noqa: E402

from app.services import career_service  # noqa: E402
from app.services import rate_limit as rate_limit_module  # noqa: E402
from app.services import research_service  # noqa: E402

# 刻意从兼容层导入 app：既测业务，也保证 `uvicorn open_deep_research.api:app` 仍然可用
from open_deep_research.api import app as api_app  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # 屏蔽 LLM 调用
    async def fake_llm(*args, **kwargs):
        return "【mock】LLM 返回内容"

    async def fake_gap(*args, **kwargs):
        return "【mock】匹配度分析结果"

    monkeypatch.setattr(
        "open_deep_research.career_workflow._llm_completion", fake_llm
    )
    # gap_analysis.run_gap_analysis 既被 API 懒加载引用，也被 career_workflow 顶层导入引用
    monkeypatch.setattr(
        "open_deep_research.gap_analysis.run_gap_analysis", fake_gap
    )
    monkeypatch.setattr(
        "open_deep_research.career_workflow.run_gap_analysis", fake_gap
    )
    return TestClient(api_app)


def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "endpoints" in body

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_probe(client):
    """/ready 只做配置与目录检查，不应触发任何外部连接。"""
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ready", "degraded"}
    assert "checkpointer" in body


def test_profile_no_llm(client):
    r = client.get("/api/profile")
    assert r.status_code == 200
    assert "profile" in r.json()


def test_discover_jobs(client):
    r = client.post(
        "/api/career/discover-jobs",
        json={"target_role": "AI工程师", "target_city": "深圳"},
    )
    assert r.status_code == 200
    assert "discovery" in r.json()


def test_gap_analysis(client):
    r = client.post(
        "/api/career/gap-analysis",
        json={"jd_text": "招聘 AI 工程师，要求 Python、PyTorch。"},
    )
    assert r.status_code == 200
    assert "gap_analysis" in r.json()


def test_career_workflow(client):
    r = client.post(
        "/api/career/workflow",
        json={"target_role": "AI工程师", "target_city": "北京"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["report"]
    assert "cover_letter" in body["artifacts"]


def test_gap_analysis_cache_dedupes(monkeypatch):
    """相同输入只跑一次 run_gap_analysis（#5 省去重复 LLM 开销）。"""
    calls = {"n": 0}

    async def counting_gap(jd_text, config):
        calls["n"] += 1
        return "GAP_RESULT"

    monkeypatch.setattr(
        "open_deep_research.gap_analysis.run_gap_analysis", counting_gap
    )
    career_service.clear_gap_cache()
    cfg = {"configurable": {"user_profile": "张三"}}
    r1 = asyncio.run(career_service.run_gap_analysis_cached("JD-A", cfg))
    r2 = asyncio.run(career_service.run_gap_analysis_cached("JD-A", cfg))
    # 不同输入应再次触发
    r3 = asyncio.run(career_service.run_gap_analysis_cached("JD-B", cfg))
    assert (r1, r2, r3) == ("GAP_RESULT", "GAP_RESULT", "GAP_RESULT")
    assert calls["n"] == 2
    career_service.clear_gap_cache()


def test_gap_cache_can_be_disabled(monkeypatch):
    """API_GAP_CACHE=off 时不做缓存（改了简历文件后需要立即生效）。"""
    from app.core.config import Settings

    calls = {"n": 0}

    async def counting_gap(jd_text, config):
        calls["n"] += 1
        return "GAP_RESULT"

    monkeypatch.setattr(
        "open_deep_research.gap_analysis.run_gap_analysis", counting_gap
    )
    career_service.clear_gap_cache()
    cfg = {"configurable": {"user_profile": "张三"}}
    off = Settings(gap_cache=False)
    asyncio.run(career_service.run_gap_analysis_cached("JD-A", cfg, off))
    asyncio.run(career_service.run_gap_analysis_cached("JD-A", cfg, off))
    assert calls["n"] == 2


def test_rate_limit(monkeypatch):
    """API_RATE_LIMIT_PER_MINUTE>0 时，超出窗口请求返回 429（#3）。"""
    monkeypatch.setattr(
        rate_limit_module, "_RATE_LIMITER", rate_limit_module.InMemoryRateLimiter(limit=1)
    )
    cli = TestClient(api_app)
    assert cli.get("/api/profile").status_code == 200
    r = cli.get("/api/profile")
    assert r.status_code == 429
    # 429 也是统一错误体，且带 Retry-After 与 request_id
    assert r.json()["title"] == "RATE_LIMITED"
    assert r.headers.get("Retry-After")
    assert r.headers.get("X-Request-ID")


def test_rate_limit_exempts_health(monkeypatch):
    """探针路径不能被限流，否则编排器会把健康进程判定为故障并反复重启。"""
    monkeypatch.setattr(
        rate_limit_module, "_RATE_LIMITER", rate_limit_module.InMemoryRateLimiter(limit=1)
    )
    cli = TestClient(api_app)
    for _ in range(5):
        assert cli.get("/health").status_code == 200


def test_checkpointer_fallback(monkeypatch):
    """未配置连接串的 postgres/redis 安全回退到内存 checkpointer（#2）。"""
    from langgraph.checkpoint.memory import MemorySaver

    monkeypatch.delenv("API_CHECKPOINTER_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("API_CHECKPOINTER_REDIS_URI", raising=False)
    monkeypatch.delenv("REDIS_URI", raising=False)

    build = research_service.build_checkpointer
    assert build("none") is None
    assert isinstance(build("memory"), MemorySaver)
    assert isinstance(build("postgres"), MemorySaver)
    assert isinstance(build("redis"), MemorySaver)


def test_unauthorized_error_shape(monkeypatch):
    """鉴权失败走统一错误体（改造前 detail 结构在各接口间不一致）。"""
    from app.core.config import Settings
    from app.core.deps import get_settings_dep
    from app.main import app as main_app

    main_app.dependency_overrides[get_settings_dep] = lambda: Settings(bearer_token="s3cret")
    try:
        cli = TestClient(main_app)
        r = cli.get("/api/profile")
        assert r.status_code == 401
        assert r.json()["title"] == "UNAUTHORIZED"
        # 常量时间比较不应把期望值本身泄漏出去
        assert "s3cret" not in r.text

        r = cli.get("/api/profile", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200
    finally:
        main_app.dependency_overrides.clear()


def test_profile_crud_and_safety(tmp_path):
    """档案管理全链路 + 路径穿越防护（隔离到临时目录，不污染真实数据）。"""
    from app.core.config import Settings
    from app.core.deps import get_settings_dep
    from app.main import app as main_app

    settings = Settings(
        profile_dir=str(tmp_path / "profiles"),
        resume_path=str(tmp_path / "resume.md"),
    )
    main_app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        cli = TestClient(main_app)

        # 新建档案
        r = cli.post("/api/profile", json={"content": "# 张三是谁", "name": "张三"})
        assert r.status_code == 200, r.text
        assert r.json()["chars"] > 0
        assert cli.get("/api/profile/names").json()["names"] == ["张三"]

        # 读取档案内容
        assert cli.get("/api/profile", params={"name": "张三"}).json()["profile"] == "# 张三是谁"

        # 切换生效
        assert cli.post("/api/profile/activate", json={"name": "张三"}).status_code == 200
        assert (tmp_path / "resume.md").read_text(encoding="utf-8") == "# 张三是谁"
        assert cli.get("/api/profile/names").json()["active"] == "张三"

        # 生效中的档案不允许删除
        assert cli.delete("/api/profile", params={"name": "张三"}).status_code == 422

        # 路径穿越 / 非法字符必须在进入文件系统前被拒
        for bad in ("../evil", "a/b", "..", "x" * 65):
            assert cli.post(
                "/api/profile", json={"content": "x", "name": bad}
            ).status_code == 422, bad
        assert not (tmp_path / "evil.md").exists()

        # 空内容不允许保存
        assert cli.post("/api/profile", json={"content": "   "}).status_code == 422
    finally:
        main_app.dependency_overrides.clear()


def test_profile_upload_markdown(tmp_path):
    """上传 .md 落到指定档案，并返回预览。"""
    from app.core.config import Settings
    from app.core.deps import get_settings_dep
    from app.main import app as main_app

    settings = Settings(
        profile_dir=str(tmp_path / "profiles"),
        resume_path=str(tmp_path / "resume.md"),
        upload_max_mb=1,
    )
    main_app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        cli = TestClient(main_app)
        files = {"file": ("resume.md", "# 李四的简历\n\nPython / PyTorch", "text/markdown")}
        r = cli.post("/api/profile/upload", files=files, params={"name": "李四"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["chars"] > 0
        assert "李四" in body["preview"]
        assert cli.get("/api/profile/names").json()["names"] == ["李四"]

        # 不支持的格式
        bad = {"file": ("a.exe", b"binary", "application/octet-stream")}
        assert cli.post("/api/profile/upload", files=bad).status_code == 415

        # 超过体积上限
        big = {"file": ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")}
        assert cli.post("/api/profile/upload", files=big).status_code == 413
    finally:
        main_app.dependency_overrides.clear()
