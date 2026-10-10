"""embedding 构造方式 + 建库前自检。

【要防的是什么】

1. **官方端点不可达时无路可走**：OpenAI 官方地址在部分网络环境下直接超时，
   原本只能改代码。现在应支持用环境变量指向中转端点。
2. **失败发生在半个流程之后**：embedding 不可用时，原本要等文档全部读完切完
   才失败，且抛的是 httpx 内部异常，看不出该改哪个配置。
"""

from __future__ import annotations

import pytest

from open_deep_research import rag
from open_deep_research.ingest import _preflight_embedding


@pytest.fixture
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_BASE_URL", raising=False)


def test_model_name_is_parsed_from_config(openai_key):
    emb = rag._get_embeddings("openai:text-embedding-3-small")
    assert emb.model == "text-embedding-3-small"


def test_base_url_from_env_is_applied(openai_key, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://relay.test/v1")
    emb = rag._get_embeddings("openai:text-embedding-3-small")
    assert str(emb.openai_api_base) == "https://relay.test/v1"


def test_embedding_specific_base_url_takes_precedence(openai_key, monkeypatch):
    """更具体的变量应压过通用变量：embedding 常走与对话不同的端点。"""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://chat.test/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_BASE_URL", "https://embed.test/v1")
    emb = rag._get_embeddings("openai:text-embedding-3-small")
    assert str(emb.openai_api_base) == "https://embed.test/v1"


def test_official_endpoint_when_no_base_url(openai_key):
    """没配就走官方默认，行为与改动前一致。"""
    emb = rag._get_embeddings("openai:text-embedding-3-small")
    base = emb.openai_api_base
    assert base is None or "api.openai.com" in str(base)


def test_local_model_reports_actionable_install_guide():
    """未装 sentence-transformers 时，错误必须告诉用户该执行什么。"""
    import importlib.util

    # 只探测是否存在，不真的 import：一旦 import，torch 会被整条拉起来，
    # 光这一步就要几十秒，足以让整个测试套件超时。
    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers 已安装")

    with pytest.raises(RuntimeError) as exc:
        rag._get_embeddings("BAAI/bge-small-zh-v1.5")
    assert "pip install sentence-transformers" in str(exc.value)


class _BrokenEmbeddings:
    def embed_query(self, text):
        raise TimeoutError("connect timeout")


class _WorkingEmbeddings:
    def embed_query(self, text):
        return [0.1] * 8


def test_preflight_turns_raw_error_into_actionable_guide():
    with pytest.raises(RuntimeError) as exc:
        _preflight_embedding(_BrokenEmbeddings())
    message = str(exc.value)
    assert "embedding 不可用" in message
    # 两条出路都必须在错误里说清楚，否则用户不知道下一步做什么
    assert "OPENAI_BASE_URL" in message
    assert "sentence-transformers" in message


def test_preflight_passes_for_working_embeddings():
    _preflight_embedding(_WorkingEmbeddings())  # 不抛即通过


# --------------------------------------------------------------------------- #
# 部署级固定：默认值必须同源，且不能被复制成两份后各自漂移
# --------------------------------------------------------------------------- #

def test_ingest_default_matches_configuration_default():
    """建库 CLI 与运行时配置必须用同一个 embedding 模型。

    防的是"两处各写一遍、只改了一处"：那样索引的向量空间会对不上，检索结果
    全是噪声，而且**不报错**——是最难排查的一类问题。
    """
    from open_deep_research.configuration import Configuration
    from open_deep_research.ingest import _build_parser

    cli_default = _build_parser().get_default("embedding")
    config_default = Configuration.model_fields["rag_embedding_model"].default
    assert cli_default == config_default


def test_ui_default_matches_field_default():
    """同一个默认值在字段里写了两遍（``default`` 与 UI 的 ``default``）。

    UI 显示的和实际生效的若不一致，使用者会按错误的模型去建库。
    """
    from open_deep_research.configuration import Configuration

    field = Configuration.model_fields["rag_embedding_model"]
    ui_default = field.json_schema_extra["x_oap_ui_config"]["default"]
    assert ui_default == field.default


# --------------------------------------------------------------------------- #
# 默认开启后的兜底：装配失败要降级，不能拖垮整条研究链路
# --------------------------------------------------------------------------- #

def _reset_store_cache(monkeypatch):
    monkeypatch.setattr(rag, "_STORE", None)
    monkeypatch.setattr(rag, "_STORE_FAILED_REASON", None)


def test_rag_enabled_defaults_to_true():
    """默认开启：简历 / JD 这类私有资料是研究依据，关掉等于只靠公网搜索猜。"""
    from open_deep_research.configuration import Configuration

    assert Configuration.model_fields["rag_enabled"].default is True


def test_assembly_failure_degrades_instead_of_raising(monkeypatch):
    """装配失败必须返回 None 而不是抛异常。

    RAG 是补充手段，抛错会让工具调用失败、整条研究链路中断；为辅助功能付出
    这个代价不划算。
    """
    _reset_store_cache(monkeypatch)

    def boom(_model):
        raise RuntimeError("sentence-transformers 未安装")

    monkeypatch.setattr(rag, "_get_embeddings", boom)
    assert rag.get_rag_store({}) is None


def test_failed_assembly_is_not_retried(monkeypatch):
    """失败过就记住，不再重试——否则每个请求都要重新加载一次模型。"""
    _reset_store_cache(monkeypatch)
    calls = []

    def boom(_model):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(rag, "_get_embeddings", boom)
    rag.get_rag_store({})
    rag.get_rag_store({})
    assert len(calls) == 1
