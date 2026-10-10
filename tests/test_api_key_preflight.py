"""启动期 API Key 预检的回归测试。

【要防的是什么】

缺 API Key 的失败点非常靠后：研究跑到一半、研究员第一次调用搜索时才抛
``MissingAPIKeyError``，而且往往被 ReAct 的错误处理吞成一条 ToolMessage，
表现为"静默写了一份没有依据的报告"。所以必须把它提前到启动阶段。

本文件的核心断言有两条：

1. **推导跟随默认值**：required 名单来自 ``Configuration`` 的默认值，
   不是硬编码。默认值从 deepseek 改成别的，名单要跟着变。
2. **开发态也能看见**：``env != prod`` 时不拦启动，但必须告警；
   否则对默认 dev 部署来说这套检查等于没写。
"""

from __future__ import annotations

import pytest

from app.core import config as config_module
from app.core.config import Settings, _required_api_keys
from app.core.errors import ConfigError

# 与 configuration.py 当前默认值保持一致的一份"影子配置"：
# 用来在不依赖真实 import 的前提下验证推导逻辑。
_FAKE_DEFAULTS = {
    "search_api": "tavily",
    "summarization_model": "openai:gpt-4.1-mini",
    "research_model": "deepseek:deepseek-chat",
    "compression_model": "deepseek:deepseek-chat",
    "final_report_model": "deepseek:deepseek-chat",
}

_ALL_KEYS = (
    "TAVILY_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture
def no_env_keys(monkeypatch):
    """清空宿主机 / .env 带来的 API Key，保证断言不受运行环境污染。"""
    for name in _ALL_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("GET_API_KEYS_FROM_CONFIG", raising=False)


@pytest.fixture
def fake_defaults(monkeypatch):
    """把 Configuration 默认值替换成影子配置。"""
    monkeypatch.setattr(config_module, "_configuration_default", _FAKE_DEFAULTS.get)


def _settings(**overrides) -> Settings:
    # _env_file=None：不读项目 .env，让用例完全由 monkeypatch 控制
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- #
# 推导逻辑
# --------------------------------------------------------------------------- #
def test_required_keys_derived_from_search_and_models(fake_defaults):
    """搜索后端 + 各模型的 provider 前缀 → key 名单，且去重后顺序稳定。"""
    assert _required_api_keys() == [
        "TAVILY_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]


def test_required_keys_match_real_configuration_defaults():
    """防漂移：真实 Configuration 改默认值时，名单必须跟着变。

    这条用例不 patch，直接读项目里的真实默认值。若有人把默认搜索从 tavily
    换成别的、或把模型 provider 换掉，而 ``_PROVIDER_TO_ENV_KEY`` /
    ``_SEARCH_API_TO_ENV_KEY`` 没收录，这里会立刻失败。
    """
    keys = _required_api_keys()
    assert "TAVILY_API_KEY" in keys, "默认搜索后端是 tavily，应要求 TAVILY_API_KEY"
    assert "DEEPSEEK_API_KEY" in keys, "默认模型是 deepseek:deepseek-chat"


def test_search_none_requires_nothing(fake_defaults, monkeypatch):
    monkeypatch.setitem(_FAKE_DEFAULTS, "search_api", "none")
    assert "TAVILY_API_KEY" not in _required_api_keys()


def test_unknown_provider_is_not_reported(monkeypatch):
    """未收录的 provider 按"不需要 key"处理：宁可漏报，不拦正常部署。"""
    monkeypatch.setattr(
        config_module,
        "_configuration_default",
        lambda name: "totally-new-provider:some-model" if name == "research_model" else None,
    )
    assert _required_api_keys() == []


def test_local_provider_needs_no_key(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "_configuration_default",
        lambda name: "ollama:llama3" if name == "research_model" else None,
    )
    assert _required_api_keys() == []


def test_malformed_model_string_is_skipped(monkeypatch):
    """模型串没有 provider 前缀时不猜，交给实际调用处报错。"""
    monkeypatch.setattr(
        config_module,
        "_configuration_default",
        lambda name: "gpt-4-without-prefix" if name == "research_model" else None,
    )
    assert _required_api_keys() == []


# --------------------------------------------------------------------------- #
# 缺失判定
# --------------------------------------------------------------------------- #
def test_missing_keys_are_reported(no_env_keys, fake_defaults):
    missing = _settings(env="dev").missing_api_keys()
    assert "TAVILY_API_KEY" in missing
    assert "DEEPSEEK_API_KEY" in missing
    assert "OPENAI_API_KEY" in missing


def test_present_keys_are_not_reported(no_env_keys, fake_defaults, monkeypatch):
    for name in ("TAVILY_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "dummy-value")
    assert _settings(env="dev").missing_api_keys() == []


def test_blank_value_counts_as_missing(no_env_keys, fake_defaults, monkeypatch):
    """`KEY=` 与未配置等价——这是 .env 里最常见的填法。"""
    monkeypatch.setenv("TAVILY_API_KEY", "   ")
    assert "TAVILY_API_KEY" in _settings(env="dev").missing_api_keys()


def test_keys_from_config_skips_preflight(no_env_keys, fake_defaults, monkeypatch):
    """多租户：key 由请求体提供时，环境变量为空是正常的，不能算缺失。"""
    monkeypatch.setenv("GET_API_KEYS_FROM_CONFIG", "true")
    assert _settings(env="dev").missing_api_keys() == []


# --------------------------------------------------------------------------- #
# 分级：prod 致命 / dev 只告警
# --------------------------------------------------------------------------- #
_PROD_SAFE = dict(
    env="prod",
    cors_origins=["https://example.com"],
    bearer_token="a-token",
    checkpointer="postgres",
    memory_store="postgres",
)


def test_prod_reports_missing_key_as_problem(no_env_keys, fake_defaults):
    problems = _settings(**_PROD_SAFE).production_problems()
    assert any("TAVILY_API_KEY" in item for item in problems)


def test_prod_with_all_keys_has_no_key_problem(no_env_keys, fake_defaults, monkeypatch):
    for name in ("TAVILY_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "dummy-value")
    problems = _settings(**_PROD_SAFE).production_problems()
    assert not any("API_KEY" in item for item in problems)


def test_prod_fails_fast_on_missing_key(no_env_keys, fake_defaults):
    with pytest.raises(ConfigError):
        _settings(**_PROD_SAFE).assert_production_ready()


def test_dev_reports_keys_but_is_not_blocked(no_env_keys, fake_defaults):
    """dev 下检查同样生效（能报出缺 key），但"起不起来"由 get_settings 决定。

    注意 ``assert_production_ready()`` 本身是无条件的：它只负责"报告 + 抛"，
    判断该不该调用它的是 ``get_settings()`` 里的 ``is_prod`` 分支。
    所以 dev 不被拦住这一点由下一个用例（走 get_settings）来保证。
    """
    settings = _settings(env="dev")
    assert any("TAVILY_API_KEY" in item for item in settings.production_problems())
    assert settings.missing_api_keys()


def test_get_settings_non_prod_only_warns(no_env_keys, fake_defaults, monkeypatch):
    """get_settings() 在非 prod 下不抛，仅打日志。"""
    monkeypatch.setenv("API_ENV", "dev")
    config_module.clear_settings_cache()
    try:
        settings = config_module.get_settings()
    finally:
        config_module.clear_settings_cache()
    assert settings.env == "dev"
