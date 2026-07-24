"""
零依赖 RAG 可行性演示  (demo_rag.py)
================================

目的：在不安装任何第三方包、不需要 API Key、不连接任何数据库的前提下，
证明「文档 → 切片 → 向量化 → 相似度检索」这条 RAG 链路是完全可行的。

本演示使用「字符级词频向量 + 余弦相似度」作为最轻量的 embedding，
仅用于验证流程可跑通、检索能命中。

真实生产实现见  src/open_deep_research/rag.py
（使用 Chroma 向量库 + OpenAI / sentence-transformers 真实语义 embedding，
 通过 python -m open_deep_research.ingest 建库，并在配置中开启 rag_enabled）。
"""

import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# 1. 示例知识库文档（内置，无需外部数据库 / 文件）
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "source": "简历.md",
        "content": (
            "张伟，三年后端开发工程师。熟悉 Python、Go、Java 等编程语言，"
            "主导过后端高并发订单系统的设计与重构，使用 Redis 做缓存、"
            "Kafka 做异步消息队列。熟悉 MySQL 与 PostgreSQL 的索引优化，"
            "有 Kubernetes 容器化部署经验。曾负责日均千万级请求的服务稳定性治理。"
        ),
    },
    {
        "source": "字节跳动_AI工程师_JD.md",
        "content": (
            "AI 工程师岗位要求：扎实的机器学习与深度学习基础，熟悉 PyTorch、"
            "TensorFlow 框架，有自然语言处理（NLP）或计算机视觉（CV）项目经验。"
            "要求掌握 Transformer、大语言模型微调与 RAG 检索增强生成技术，"
            "熟悉向量数据库（如 Milvus、Chroma、pgvector）的使用。"
            "有分布式训练与模型部署上线经验者优先。"
        ),
    },
    {
        "source": "面经_某大厂.md",
        "content": (
            "某大厂算法岗面试记录：一面手撕 Transformer 的注意力机制公式，"
            "二面考察大语言模型的预训练与微调区别，三面问了 RAG 检索增强生成的"
            "工程实现与向量召回的召回率优化。建议重点复习 embedding 模型选型、"
            "文本切片策略以及相似度检索的调参经验。"
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. 文本切片（最小实现，等价于 RecursiveCharacterTextSplitter 的定长切分）
# ---------------------------------------------------------------------------
def split_text(text: str, chunk_size: int = 120, overlap: int = 30):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# 3. 向量化（字符级 TF 向量）+ 余弦相似度
# ---------------------------------------------------------------------------
_PUNCT = set(" \n\t，。、；：？！“”‘’（）《》【】…—.,;:?!\'\"()[]<>/")


def _tokenize(text: str):
    return [c for c in text.lower() if c not in _PUNCT]


def _vectorize(text: str):
    vec = Counter(_tokenize(text))
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return vec, norm


def _cosine(v1, n1, v2, n2) -> float:
    dot = sum(v1[t] * v2[t] for t in set(v1) & set(v2))
    return dot / (n1 * n2)


# ---------------------------------------------------------------------------
# 4. 内存向量库（迷你 Chroma）
# ---------------------------------------------------------------------------
class MiniRAG:
    def __init__(self):
        self.chunks = []  # {content, source, vec, norm}

    def add(self, source: str, content: str):
        for piece in split_text(content):
            vec, norm = _vectorize(piece)
            self.chunks.append({"content": piece, "source": source, "vec": vec, "norm": norm})

    def search(self, query: str, top_k: int = 2):
        qv, qn = _vectorize(query)
        scored = [( _cosine(qv, qn, c["vec"], c["norm"]), c) for c in self.chunks]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 5. 运行演示
# ---------------------------------------------------------------------------
def main():
    rag = MiniRAG()
    for d in DOCUMENTS:
        rag.add(d["source"], d["content"])
    print(f"✅ 已构建知识库：{len(rag.chunks)} 个片段（来自 {len(DOCUMENTS)} 份文档）\n")

    queries = [
        "候选人熟悉哪些编程语言？",
        "AI 工程师岗位要求掌握哪些技术？",
        "面试常问什么技术题目？",
    ]

    for q in queries:
        print(f"🔍 查询: {q}")
        results = rag.search(q, top_k=2)
        if not results:
            print("   （未检索到相关内容）")
        for score, c in results:
            snippet = c["content"].replace("\n", " ")
            print(f"   相似度={score:.3f}  来源={c['source']}")
            print(f"   片段: {snippet[:90]}...")
        print("-" * 70)

    print("\n🎉 RAG 链路验证成功：文档切片 → 向量化 → 相似度检索 全部跑通。")


if __name__ == "__main__":
    main()
