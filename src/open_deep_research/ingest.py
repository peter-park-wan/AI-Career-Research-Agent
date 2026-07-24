"""Ingest documents from a directory into a local Chroma RAG index."""

import argparse
import glob
import os

from langchain_core.documents import Document

from open_deep_research.rag import RAGStore, _get_embeddings


def _read_pdf(path: str) -> str:
    import fitz  # pymupdf

    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def _load_documents(src: str) -> list[Document]:
    """Load text / markdown / json / html / pdf files from `src` recursively."""
    docs: list[Document] = []
    patterns = {
        "**/*.txt": "text",
        "**/*.md": "text",
        "**/*.json": "text",
        "**/*.html": "text",
        "**/*.htm": "text",
        "**/*.pdf": "pdf",
    }
    for pat, kind in patterns.items():
        for path in sorted(glob.glob(os.path.join(src, pat), recursive=True)):
            try:
                if kind == "pdf":
                    text = _read_pdf(path)
                else:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
            except Exception as e:  # noqa: BLE001
                print(f"跳过 {path}: {e}")
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.relpath(path)},
                )
            )
    return docs


def _split(docs, chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 知识库索引")
    parser.add_argument("--src", default="./data", help="文档目录")
    parser.add_argument("--index-path", default="./data/rag_index", help="索引持久化目录")
    parser.add_argument("--collection", default="career_kb", help="Chroma 集合名")
    parser.add_argument(
        "--embedding",
        default="openai:text-embedding-3-small",
        help="embedding 模型，openai:<model> 或本地 sentence-transformers 模型名",
    )
    parser.add_argument("--top-k", type=int, default=4, help="默认检索返回数量")
    parser.add_argument(
        "--vector-store",
        default="chroma",
        choices=["chroma", "supabase"],
        help="向量库后端：chroma（本地）或 supabase（pgvector 远端）",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    args = parser.parse_args()

    print(f"加载文档: {args.src}")
    docs = _load_documents(args.src)
    print(f"原始文档数: {len(docs)}")
    splits = _split(docs, args.chunk_size, args.chunk_overlap)
    print(f"切分后片段数: {len(splits)}")

    emb = _get_embeddings(args.embedding)
    store = RAGStore(
        embedding_model=emb,
        index_path=args.index_path,
        collection=args.collection,
        top_k=args.top_k,
        vector_store=args.vector_store,
        connection_string=os.getenv("SUPABASE_CONNECTION_STRING"),
    )
    store.build(splits)
    print("索引构建完成 ✅")


if __name__ == "__main__":
    main()
