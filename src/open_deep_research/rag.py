"""RAG toolkit for the AI Career Research Agent.

Provides a local vector index (Chroma) over private documents (resume, JDs,
interview experiences, learning resources, ...) and exposes a retrieval tool
that the research agents can call alongside web search.
"""

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from typing_extensions import Annotated

from open_deep_research.configuration import Configuration

logger = logging.getLogger(__name__)

_STORE: Optional["RAGStore"] = None
# 装配失败的原因。非空 = RAG 不可用且不再重试（否则每个请求都要重加载一次模型）。
_STORE_FAILED_REASON: Optional[str] = None


def _get_embeddings(model: str):
    """按配置字符串构造 embeddings 对象。

    * ``"openai:<model>"`` → OpenAI，或任何 OpenAI **兼容**端点
    * 其他值 → 本地 sentence-transformers 模型名

    为什么要支持 base_url：OpenAI 官方端点在部分网络环境下直接不可达
    （表现为 ``ConnectTimeout`` / ``APITimeoutError``）。此时不必改代码，
    把 ``OPENAI_BASE_URL``（或更具体的 ``OPENAI_EMBEDDING_BASE_URL``）指向
    一个可访问的中转端点即可。两者都没设时走官方地址，行为与之前一致。
    """
    if model and model.startswith("openai:"):
        from langchain_openai import OpenAIEmbeddings

        base_url = (
            os.getenv("OPENAI_EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        kwargs = {"base_url": base_url} if base_url else {}
        return OpenAIEmbeddings(
            model=model.split(":", 1)[1],
            api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs,
        )

    # Local / offline embeddings via sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "本地 embedding 需要 sentence-transformers，请先执行：\n"
            "    pip install sentence-transformers\n"
            "注意它会一并装上 torch（体积较大）。若模型下载慢，可设：\n"
            "    HF_ENDPOINT=https://hf-mirror.com"
        ) from exc

    class _LocalEmbeddings:
        def __init__(self, name: str):
            self._m = SentenceTransformer(name)

        def embed_documents(self, texts):
            return self._m.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self._m.encode([text], normalize_embeddings=True)[0].tolist()

    return _LocalEmbeddings(model or "BAAI/bge-small-zh-v1.5")


class RAGStore:
    """Vector store abstraction for RAG.

    Backends:
      - "chroma":   local file-based Chroma (default, no external service)
      - "supabase": remote pgvector on Supabase, via langchain-postgres
    """

    def __init__(
        self,
        embedding_model,
        index_path: str = "./data/rag_index",
        collection: str = "career_kb",
        top_k: int = 4,
        vector_store: str = "chroma",
        connection_string: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.collection = collection
        self.top_k = top_k
        self.vector_store = vector_store
        self.connection_string = connection_string
        # 底层 VectorStore 实例（懒加载 + 复用）
        self._vs: Optional[object] = None

    # ------------------------------------------------------------------ #
    # 底层客户端
    # ------------------------------------------------------------------ #
    def _client(self):
        """返回底层 VectorStore 实例并复用。

        原先每次检索都重新构造一次 Chroma/PGVector，等于每查一次就把持久化
        目录重新加载一遍；这里缓存实例，避免重复的磁盘 IO / 连接开销。
        """
        if self._vs is not None:
            return self._vs
        if self.vector_store == "supabase":
            from langchain_postgres import PGVector

            if not self.connection_string:
                raise RuntimeError(
                    "使用 supabase 向量库需要设置 SUPABASE_CONNECTION_STRING "
                    "(Postgres 连接串，例如 postgresql://user:pass@host:5432/postgres)。"
                )
            self._vs = PGVector(
                collection_name=self.collection,
                embedding_function=self.embedding_model,
                connection_string=self.connection_string,
            )
        else:
            from langchain_chroma import Chroma

            self._vs = Chroma(
                collection_name=self.collection,
                embedding_function=self.embedding_model,
                persist_directory=self.index_path,
            )
        return self._vs

    def add_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None
    ) -> int:
        """写入片段。``ids`` 相同时为覆盖语义（底层 add 即 upsert）。"""
        if not documents:
            return 0
        self._client().add_documents(documents, ids=ids)
        logger.info(
            "写入 %d 个片段（vector_store=%s, collection=%s）",
            len(documents),
            self.vector_store,
            self.collection,
        )
        return len(documents)

    def delete_by_ids(self, ids: List[str]) -> int:
        """按 id 删除片段——与 ``add_documents`` 配合实现"先删后插"的幂等重建。

        为什么需要它：直接重复 ``add`` 只会追加，同一份文档改版后新旧副本会
        同时留在库里，检索命中哪份是随机的（对简历/JD 这类时效性强的资料，
        等于可能给出过期答案）。
        """
        ids = [item for item in (ids or []) if item]
        if not ids:
            return 0
        self._client().delete(ids)
        logger.info(
            "删除 %d 个旧片段（vector_store=%s）", len(ids), self.vector_store
        )
        return len(ids)

    def build(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        """Embed and persist documents into the configured vector store."""
        self.add_documents(documents, ids=ids)

    def retrieve(self, query: str, k: Optional[int] = None) -> str:
        """Return formatted, source-tagged context for a query."""
        if self.vector_store == "chroma" and not os.path.isdir(self.index_path):
            return "知识库尚未构建，请先运行索引脚本（例如: python -m open_deep_research.ingest）。"
        if self.vector_store == "supabase" and not self.connection_string:
            return "未配置 SUPABASE_CONNECTION_STRING，无法连接远端向量库。"

        k = k or self.top_k
        if self.vector_store == "supabase":
            results = self._similarity_search_supabase(query, k)
        else:
            results = self._similarity_search_chroma(query, k)

        if not results:
            return "未在知识库中检索到相关内容。"
        blocks = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "未知来源")
            blocks.append(f"[来源 {i}] {src}\n{doc.page_content}")
        return "\n\n".join(blocks)

    def _similarity_search_chroma(self, query: str, k: int):
        return self._client().similarity_search(query, k=k)

    def _similarity_search_supabase(self, query: str, k: int):
        return self._client().similarity_search(query, k=k)


def get_rag_store(config: Optional[RunnableConfig]) -> Optional[RAGStore]:
    """Return (cached) RAGStore derived from the current configuration.

    装配失败时返回 ``None`` 而不是抛异常：RAG 是**补充**手段，研究员还能靠
    公网搜索完成研究。若在这里抛错，工具调用失败会拖垮整条研究链路——为一个
    辅助功能付出中断研究的代价不划算。失败原因记在 ``_STORE_FAILED_REASON``，
    由 ``retrieve_knowledge_base`` 提示出来，不让调用方对着空结果猜。
    """
    global _STORE, _STORE_FAILED_REASON
    if _STORE is not None:
        return _STORE
    if _STORE_FAILED_REASON is not None:
        return None
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_enabled:
        return None
    try:
        emb = _get_embeddings(configurable.rag_embedding_model)
    except Exception as exc:  # noqa: BLE001
        _STORE_FAILED_REASON = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "RAG 已启用但 embedding 装配失败，后续检索将跳过知识库：%s",
            _STORE_FAILED_REASON,
        )
        return None
    _STORE = RAGStore(
        embedding_model=emb,
        index_path=configurable.rag_index_path,
        collection=configurable.rag_collection,
        top_k=configurable.rag_top_k,
        vector_store=configurable.rag_vector_store,
        connection_string=os.getenv("SUPABASE_CONNECTION_STRING"),
    )
    return _STORE


@tool(
    description=(
        "从私有知识库检索相关内容（如个人简历、目标公司岗位JD、面经、学习资源库等），"
        "用于补充联网搜索无法覆盖的私有/内部资料。当问题涉及用户个人背景或内部资料时应优先调用。"
    )
)
def retrieve_knowledge_base(
    query: str,
    top_k: Annotated[Optional[int], InjectedToolArg] = None,
    config: RunnableConfig = None,
) -> str:
    """检索知识库。

    Args:
        query: 检索查询，应描述你想在知识库中查找的信息（如“我的简历中的项目经历”“字节跳动 AI 工程师 JD 要求”）。
        top_k: 返回的最大结果数（可选，默认使用配置值）。
        config: 运行时配置（自动注入）。
    """
    store = get_rag_store(config)
    if store is None:
        if _STORE_FAILED_REASON:
            return (
                f"知识库检索不可用（{_STORE_FAILED_REASON}）。"
                "本次研究将只使用公网搜索，请修好后重试。"
            )
        return "知识库检索未启用（rag_enabled=false），请在配置中开启 RAG。"
    return store.retrieve(query, top_k)
