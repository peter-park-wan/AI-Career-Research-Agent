"""ingest 的幂等 / 增量 / 不静默失败 —— 回归测试。

【要防的是什么】

1. **重复导入叠加**：写入原本是纯追加，同一份文档改版后新旧副本并存，检索
   命中哪份是随机的，对简历/JD 这类时效性资料等于可能给出过期答案。
2. **源文件删了，索引里还在**：删除必须联动清理，否则"删掉的简历还能被搜到"。
3. **空内容静默入库**：扫描版 PDF 提取出空字符串，空字符串不是异常，于是照常
   入库，脚本照样打印成功，实际什么都没贡献。
4. **索引目录被当成文档目录**：默认 ``index_path`` 就在 ``src`` 里，清单文件
   会匹配 ``**/*.json`` 被扫进去，导致每次都"有待入库"，永远到不了"已是最新"。

用假 embedding 跑流程：这些行为与向量语义无关，不该依赖网络和 API 额度。
"""

from __future__ import annotations

import os
import sys

import pytest

import open_deep_research.ingest as ingest


class DummyEmb:
    """固定向量，只为跑通流程。"""

    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


@pytest.fixture(autouse=True)
def fake_embedding(monkeypatch):
    monkeypatch.setattr(ingest, "_get_embeddings", lambda model: DummyEmb())


def _run(src, index_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest",
            "--src",
            str(src),
            "--index-path",
            str(index_path),
            "--collection",
            "pytest_kb",
        ],
    )
    ingest.main()


def _count(index_path) -> int:
    if not os.path.isdir(index_path):
        return 0
    from langchain_chroma import Chroma

    vs = Chroma(
        collection_name="pytest_kb",
        embedding_function=DummyEmb(),
        persist_directory=str(index_path),
    )
    return vs._collection.count()


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_repeat_ingest_does_not_duplicate(tmp_path, monkeypatch):
    src = tmp_path / "docs"
    idx = tmp_path / "idx"
    _write(src / "a.txt", "张三是资深后端工程师，精通 Python。" * 20)

    _run(src, idx, monkeypatch)
    first = _count(idx)
    assert first > 0

    _run(src, idx, monkeypatch)
    assert _count(idx) == first, "无改动重复导入导致片段叠加"


def test_changed_file_replaces_instead_of_appending(tmp_path, monkeypatch):
    src = tmp_path / "docs"
    idx = tmp_path / "idx"
    resume = src / "resume.txt"
    _write(resume, "旧版简历：三年经验。" * 20)

    _run(src, idx, monkeypatch)
    before = _count(idx)

    # 简历改版后重建：总数应保持稳定（先删后插），而不是翻倍
    _write(resume, "新版简历：五年经验，新增大模型应用经历。" * 20)
    _run(src, idx, monkeypatch)
    assert _count(idx) == before, "改版后新旧副本并存，片段数翻倍"


def test_deleted_source_file_is_purged(tmp_path, monkeypatch):
    src = tmp_path / "docs"
    idx = tmp_path / "idx"
    _write(src / "keep.txt", "保留的文档内容。" * 20)
    _write(src / "drop.txt", "要被删除的文档内容。" * 20)

    _run(src, idx, monkeypatch)
    before = _count(idx)

    os.remove(src / "drop.txt")
    _run(src, idx, monkeypatch)
    assert _count(idx) < before, "源文件已删除，索引里仍留有片段"


def test_empty_document_is_not_ingested(tmp_path, monkeypatch, caplog):
    """空内容必须被跳过并报出，而不是静默产生一条空片段。"""
    src = tmp_path / "docs"
    idx = tmp_path / "idx"
    _write(src / "empty.txt", "   \n\t\n ")
    _write(src / "real.txt", "有效内容。" * 20)

    with caplog.at_level("ERROR"):
        _run(src, idx, monkeypatch)

    assert _count(idx) > 0
    assert any("empty.txt" in rec.message for rec in caplog.records), (
        "空文档被静默处理，没有任何错误日志"
    )


def test_index_dir_inside_src_is_not_scanned(tmp_path, monkeypatch):
    """索引目录位于 src 内部时，清单文件不能被当成待入库文档。

    这是真实踩过的坑：清单匹配 ``**/*.json`` 且每次运行都会变，导致永远
    "有待入库"，索引永远到不了最新状态。
    """
    src = tmp_path / "docs"
    idx = src / "rag_index"  # 故意放在 src 里面
    _write(src / "a.txt", "文档内容。" * 20)

    _run(src, idx, monkeypatch)
    first = _count(idx)

    _run(src, idx, monkeypatch)
    assert _count(idx) == first, "索引目录自身被扫描入库，导致每次都产生新片段"
