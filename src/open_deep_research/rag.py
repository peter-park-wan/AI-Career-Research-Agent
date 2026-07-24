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


def _get_embeddings(model: str):
    """Build an embeddings object from a config string.

    Format: "openai:<model>" uses OpenAI embeddings (needs OPENAI_API_KEY);
    anything else is treated as a local sentence-transformers model name
    (needs `pip install sentence-transformers`).
    """
    if model and model.startswith("openai:"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model.split(":", 1)[1],
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Local / offline embeddings via sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        class _LocalEmbeddings:
            def __init__(self, name: str):
                self._m = SentenceTransformer(name)

            def embed_documents(self, texts):
                return self._m.encode(texts, normalize_embeddings=True).tolist()

            def embed_query(self, text):
                return self._m.encode([text], normalize_embeddings=True)[0].tolist()

        return _LocalEmbeddings(model or "BAAI/bge-small-zh-v1.5")
    except ImportError:
        raise RuntimeError(
            "本地 embedding 需要 sentence-transformers，请先执行: pip install sentence-transformers"
        )


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

    def build(self, documents: List[Document]) -> None:
        """Embed and persist documents into the configured vector store."""
        if self.vector_store == "supabase":
            self._build_supabase(documents)
        else:
            self._build_chroma(documents)

    def _build_chroma(self, documents: List[Document]) -> None:
        from langchain_chroma import Chroma

        Chroma.from_documents(
            documents,
            embedding=self.embedding_model,
            collection_name=self.collection,
            persist_directory=self.index_path,
        )
        logger.info("RAG index built at %s with %d docs", self.index_path, len(documents))

    def _build_supabase(self, documents: List[Document]) -> None:
        from langchain_postgres import PGVector

        if not self.connection_string:
            raise RuntimeError(
                "使用 supabase 向量库需要设置 SUPABASE_CONNECTION_STRING "
                "(Postgres 连接串，例如 postgresql://user:pass@host:5432/postgres)。"
            )
        PGVector.from_documents(
            documents,
            embedding=self.embedding_model,
            collection_name=self.collection,
            connection_string=self.connection_string,
        )
        logger.info("RAG index built on Supabase (pgvector), collection=%s", self.collection)

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
        from langchain_chroma import Chroma

        vs = Chroma(
            collection_name=self.collection,
            embedding_function=self.embedding_model,
            persist_directory=self.index_path,
        )
        return vs.similarity_search(query, k=k)

    def _similarity_search_supabase(self, query: str, k: int):
        from langchain_postgres import PGVector

        vs = PGVector(
            collection_name=self.collection,
            embedding_function=self.embedding_model,
            connection_string=self.connection_string,
        )
        return vs.similarity_search(query, k=k)


def get_rag_store(config: Optional[RunnableConfig]) -> Optional[RAGStore]:
    """Return (cached) RAGStore derived from the current configuration."""
    global _STORE
    if _STORE is not None:
        return _STORE
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_enabled:
        return None
    emb = _get_embeddings(configurable.rag_embedding_model)
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
        return "知识库检索未启用（rag_enabled=false），请在配置中开启 RAG。"
    return store.retrieve(query, top_k)
