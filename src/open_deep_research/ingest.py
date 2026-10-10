"""把目录中的文档灌进向量索引（幂等 + 增量 + 不静默失败）。

【为什么要重写这个脚本】

原实现有三个"看起来成功、实际是错的"问题：

1. **重复导入会叠加**：写入是追加语义，同一份文档改版后新旧副本同时留在库里，
   检索命中哪份是随机的。对简历 / JD 这类时效性强的资料，等于可能给出过期
   答案，而且无从察觉。更隐蔽的是：源文件删掉后，索引里的副本仍然可被检索到。
2. **空内容静默入库**：扫描版 PDF 没有文本层，``get_text()`` 返回空字符串——
   空字符串不是异常，于是照常入库，脚本最后照样打印"索引构建完成 ✅"，
   但这个 PDF 什么都没贡献。
3. **不支持的类型毫无提示**：图片根本不在匹配列表里，放进 data/ 后连
   "发现不支持的文件"这类提示都没有。

现在的行为：

* 内容未变的文件直接跳过（增量，省掉重复的 embedding 开销）
* 变化的文件**先删旧片段再写入**（幂等，不叠加）
* 磁盘上已删除的文件，其片段一并清理（不留幽灵数据）
* 空文本 / 不支持类型 / 读取失败一律**明确报出**并给出可操作建议

【清单文件】

增量依赖 ``<index_path>/_ingest_manifest.json``，记录每个 source 的内容哈希
与已写入的片段 id。删掉它等于回到全量重建（也可用 ``--rebuild``）。
"""

import argparse
import glob
import hashlib
import json
import logging
import os

from langchain_core.documents import Document

from open_deep_research.configuration import Configuration
from open_deep_research.rag import RAGStore, _get_embeddings

logger = logging.getLogger(__name__)

# 支持的类型 → 解析方式
_SUPPORTED = {
    "**/*.txt": "text",
    "**/*.md": "text",
    "**/*.json": "text",
    "**/*.html": "text",
    "**/*.htm": "text",
    "**/*.pdf": "pdf",
    # 图片走 OCR（见 _read_image）。gif 不在内：动图逐帧 OCR 没意义还慢。
    "**/*.png": "image",
    "**/*.jpg": "image",
    "**/*.jpeg": "image",
    "**/*.webp": "image",
    "**/*.bmp": "image",
    "**/*.tiff": "image",
    "**/*.tif": "image",
}

# 看到了但当前不支持。单独识别它们，是为了给出**可操作的提示**，
# 而不是让用户对着"索引里查不到"一头雾水。
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_OFFICE_EXTS = {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}

# 文本解码顺序。写死 utf-8 + errors="ignore" 会让 GBK 中文文件**静默丢字节**，
# 内容残缺且不报错；这里按序尝试，全都不行才降级并标记。
_ENCODINGS = ("utf-8", "gb18030", "big5")


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def _read_text(path: str) -> tuple:
    """返回 ``(文本, 实际使用的编码)``。"""
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(), enc
        except UnicodeDecodeError:
            continue
    # 都不行：不静默丢弃，用 replace 保住可读部分，但明确标记出来
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(), "utf-8(replace: 部分字节无法解码)"


# --------------------------------------------------------------------------- #
# OCR（图片 / 扫描件）
#
# 选 RapidOCR（onnxruntime 版）而不是 PaddleOCR / tesseract / 多模态大模型：
#   * 中文准确率：用的就是 PaddleOCR 的模型，远高于 tesseract 的 chi_sim
#   * 离线：模型随 pip 包一起带来，运行时不再联网下载（本项目部署环境里
#     huggingface.co 不可达，任何"运行时拉模型"的方案都会卡死）
#   * 轻量：推理走 onnxruntime，不必装 paddlepaddle，也不依赖外网 LLM
# 代价是每张图几百毫秒，建库比纯文本慢——但这个代价只花在图片/扫描件上。
# --------------------------------------------------------------------------- #
_OCR_ENGINE = None
_OCR_MISSING = False


def _get_ocr():
    """惰性构造 OCR 引擎。

    惰性有两个原因：加载模型要几百毫秒，纯文本建库不该付这个代价；没装依赖时
    也不能让 **import** 就失败，否则不处理图片的人也被卡住。
    """
    global _OCR_ENGINE, _OCR_MISSING
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if _OCR_MISSING:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        _OCR_MISSING = True
        return None
    _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _ocr_lines(result) -> str:
    """RapidOCR 返回 ``[(box, text, score), ...]``，按行拼回文本。"""
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def _read_image(path: str) -> tuple:
    """图片 → 文本（OCR）。"""
    ocr = _get_ocr()
    if ocr is None:
        raise RuntimeError(
            "图片入库需要 OCR，请先执行: pip install rapidocr-onnxruntime"
        )
    result, _ = ocr(path)
    return _ocr_lines(result), {"ocr": "rapidocr", "lines": len(result or [])}


def _ocr_pdf_pages(doc) -> str:
    """把 PDF 每页渲染成位图再 OCR——扫描件唯一的救法。"""
    ocr = _get_ocr()
    if ocr is None:
        raise RuntimeError(
            "扫描件 PDF 需要 OCR，请先执行: pip install rapidocr-onnxruntime"
        )
    import cv2
    import numpy as np

    texts = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = cv2.imdecode(
            np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR
        )
        result, _ = ocr(img)
        line = _ocr_lines(result)
        if line:
            texts.append(line)
    return "\n".join(texts)


def _read_pdf(path: str) -> tuple:
    """返回 ``(文本, 诊断信息)``。

    PyMuPDF 的 ``get_text()`` 只取**文本层**：扫描件 / 截图转出来的 PDF 没有
    文本层，会得到空字符串。这种情况不再直接放弃，而是把页面渲染成位图走
    OCR——扫描件是真实存在的输入形式，直接跳过等于让用户自己去预处理。
    """
    import fitz  # pymupdf

    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    empty_pages = sum(1 for p in pages if not p.strip())
    image_only = (
        bool(pages)
        and all(not p.strip() for p in pages)
        and any(page.get_images() for page in doc)
    )
    info = {"pages": len(pages), "empty_pages": empty_pages, "image_only": image_only}
    if image_only:
        info["ocr"] = "rapidocr"
        return _ocr_pdf_pages(doc), info
    return "\n".join(pages), info


def _file_hash(path: str) -> str:
    """按**文件内容**算哈希，用于判断"这个文档到底改没改"。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 清单（增量与幂等的依据）
# --------------------------------------------------------------------------- #
def _manifest_path(index_path: str) -> str:
    return os.path.join(index_path, "_ingest_manifest.json")


def _load_manifest(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sources"), dict):
            return data["sources"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("清单文件无法解析，按全量重建处理：%s（%s）", path, exc)
    return {}


def _save_manifest(path: str, sources: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "sources": sources}, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #
def _scan(src: str, exclude_dir: str | None = None):
    """返回 ``(loaded, empty, failed)``，元素分别为可入库 / 空文本 / 读取失败。

    ``exclude_dir`` 必须排除索引目录。默认参数下 ``index_path`` **就在** ``src``
    里面（``./data/rag_index`` 位于 ``./data`` 之下），若不排除，清单文件和索引
    内部文件会被当成待入库文档扫进去：清单每次运行都会变，于是每次都"有待入库"，
    永远到不了"已是最新"，索引也被垃圾内容污染。
    """
    skip_root = os.path.abspath(exclude_dir) if exclude_dir else None
    loaded, empty, failed = [], [], []
    for pattern, kind in _SUPPORTED.items():
        for path in sorted(glob.glob(os.path.join(src, pattern), recursive=True)):
            abs_path = os.path.abspath(path)
            if skip_root and (
                abs_path == skip_root or abs_path.startswith(skip_root + os.sep)
            ):
                continue
            rel = os.path.relpath(path, src)
            try:
                if kind == "pdf":
                    text, info = _read_pdf(path)
                elif kind == "image":
                    text, info = _read_image(path)
                else:
                    text, enc = _read_text(path)
                    info = {"encoding": enc}
            except Exception as exc:  # noqa: BLE001
                failed.append((rel, str(exc)))
                continue
            if not text.strip():
                # 关键改动：空文本**不再**照常入库，而是明确报出来
                empty.append((rel, info))
                continue
            loaded.append((rel, text, info))
    return loaded, empty, failed


def _ocr_installed() -> bool:
    """只探测依赖是否装了，不加载模型（构造引擎要几百毫秒）。

    用途是决定"该不该把图片算作不支持"——这个判断在还没见到任何图片时就要做，
    不该为此先把模型加载起来。
    """
    import importlib.util

    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


def _find_unsupported(src: str, index_path: str):
    """扫出"看到了但不支持"的文件。"""
    images, office = [], []
    skip = {os.path.abspath(index_path)}
    for root, dirs, files in os.walk(src):
        if os.path.abspath(root) in skip:
            dirs[:] = []
            continue
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            rel = os.path.relpath(os.path.join(root, name), src)
            if ext in _IMAGE_EXTS:
                # 装了 OCR 就能入库，不算"不支持"；没装才提示，并给出安装命令
                if not _ocr_installed():
                    images.append(rel)
            elif ext in _OFFICE_EXTS:
                office.append(rel)
    return images, office


def _report(images, office, empty, failed) -> None:
    """把"没入库"的原因全部说清楚——静默失败是最难排查的一类问题。"""
    if images:
        logger.warning(
            "发现 %d 个图片文件，未装 OCR 依赖因此跳过：%s%s"
            "（执行 pip install rapidocr-onnxruntime 后即可入库）",
            len(images),
            "、".join(images[:5]),
            " …" if len(images) > 5 else "",
        )
    if office:
        logger.warning(
            "发现 %d 个 Office 文档，当前不支持：%s%s（请先另存为 .md / .txt / .pdf）",
            len(office),
            "、".join(office[:5]),
            " …" if len(office) > 5 else "",
        )
    for rel, info in empty:
        if info.get("image_only"):
            if info.get("ocr"):
                logger.error("跳过 %s：扫描件 PDF，且 OCR 也未识别出任何文字", rel)
            else:
                logger.error(
                    "跳过 %s：PDF 只有图像、没有文本层（疑似扫描件），"
                    "且未装 OCR 依赖，无法入库",
                    rel,
                )
        else:
            logger.error(
                "跳过 %s：提取到的文本为空（共 %s 页，其中 %s 页无文本）",
                rel,
                info.get("pages", "?"),
                info.get("empty_pages", "?"),
            )
    for rel, err in failed:
        logger.error("读取失败，跳过 %s：%s", rel, err)


def _split(docs, chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def _preflight_embedding(emb) -> None:
    """建库之前先真的调一次 embedding。

    不做这一步会怎样：embedding 不可用时（Key 错、网络不通、依赖没装），
    要等文档全部读完切完、真正写入时才失败。而网络类失败往往是**几十秒的
    超时**，且抛出的是 httpx 内部异常，从堆栈根本看不出该去改哪个配置。
    这里提前花一次调用把它变成一条能照着操作的错误。
    """
    try:
        emb.embed_query("连通性自检")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"embedding 不可用，无法构建索引（{type(exc).__name__}: {exc}）。\n"
            "请先修好下面任意一项再重试：\n"
            "  1) OpenAI 兼容端点：确认 OPENAI_API_KEY 已设置；若官方端点不可达，"
            "设置 OPENAI_BASE_URL 指向可访问的中转端点\n"
            "  2) 本地模型：pip install sentence-transformers，并把 "
            "rag_embedding_model 设为 BAAI/bge-small-zh-v1.5"
        ) from exc


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _build_parser():
    """单独抽成函数是为了可测试：见 tests 中"建库默认模型 == 检索默认模型"的断言。

    建库与检索若用了不同的 embedding 模型，索引的向量空间就对不上，检索出来
    全是噪声且不报错。靠人记住"两处要一起改"迟早漂移，所以 CLI 的默认值直接
    从 ``Configuration`` 取——让"只改一处"成为唯一正确的改法。
    """
    parser = argparse.ArgumentParser(description="构建 RAG 知识库索引（幂等 + 增量）")
    parser.add_argument("--src", default="./data", help="文档目录")
    parser.add_argument("--index-path", default="./data/rag_index", help="索引持久化目录")
    parser.add_argument("--collection", default="career_kb", help="向量库集合名")
    parser.add_argument(
        "--embedding",
        default=Configuration.model_fields["rag_embedding_model"].default,
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
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="忽略增量清单，删除已知片段后全量重建",
    )
    return parser


def main():
    args = _build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not os.path.isdir(args.src):
        # 打印绝对路径：只报相对路径 "./data" 会让人以为是项目坏了，
        # 实际上绝大多数情况是工作目录不对（在项目父目录里跑了命令）。
        logger.error(
            "文档目录不存在：%s\n  当前工作目录：%s",
            os.path.abspath(args.src),
            os.getcwd(),
        )
        return

    loaded, empty, failed = _scan(args.src, args.index_path)
    images, office = _find_unsupported(args.src, args.index_path)
    _report(images, office, empty, failed)

    manifest_path = _manifest_path(args.index_path)
    manifest = {} if args.rebuild else _load_manifest(manifest_path)

    # 逐个比对内容哈希，决定"要重建 / 可跳过 / 已删除"
    to_build, skipped, stale_ids = [], [], []
    current_rels = set()
    for rel, text, info in loaded:
        current_rels.add(rel)
        file_hash = _file_hash(os.path.join(args.src, rel))
        prev = manifest.get(rel)
        if prev and prev.get("hash") == file_hash and not args.rebuild:
            skipped.append(rel)
            continue
        if prev:
            stale_ids.extend(prev.get("ids", []))
        to_build.append((rel, text, file_hash, info))

    # 源文件已从磁盘删除 → 连同索引里的片段一起清掉，避免"删除了还能搜到"
    removed = [rel for rel in manifest if rel not in current_rels]
    for rel in removed:
        stale_ids.extend(manifest.get(rel, {}).get("ids", []))

    logger.info(
        "扫描完成：待入库 %d，未变化跳过 %d，已删除待清理 %d",
        len(to_build),
        len(skipped),
        len(removed),
    )

    if not to_build and not stale_ids:
        logger.info("索引已是最新，无需写入 ✅")
        return

    os.makedirs(args.index_path, exist_ok=True)
    emb = _get_embeddings(args.embedding)
    # 只有真正要写入时才需要 embedding 服务可用；单纯清理残留片段不需要
    if to_build:
        _preflight_embedding(emb)
    store = RAGStore(
        embedding_model=emb,
        index_path=args.index_path,
        collection=args.collection,
        top_k=args.top_k,
        vector_store=args.vector_store,
        connection_string=os.getenv("SUPABASE_CONNECTION_STRING"),
    )

    if stale_ids:
        store.delete_by_ids(stale_ids)

    new_docs, new_ids = [], []
    ids_by_source = {}
    for rel, text, file_hash, info in to_build:
        metadata = {"source": rel, "file_hash": file_hash}
        if "pages" in info:
            metadata["pages"] = info["pages"]
        if "encoding" in info:
            metadata["encoding"] = info["encoding"]
        splits = _split(
            [Document(page_content=text, metadata=dict(metadata))],
            args.chunk_size,
            args.chunk_overlap,
        )
        ids = []
        for index, doc in enumerate(splits):
            doc.metadata["source"] = rel
            doc.metadata["file_hash"] = file_hash
            new_docs.append(doc)
            chunk_id = f"{rel}#{index}"
            ids.append(chunk_id)
            new_ids.append(chunk_id)
        ids_by_source[rel] = ids

    written = store.add_documents(new_docs, ids=new_ids)

    for rel, _text, file_hash, _info in to_build:
        manifest[rel] = {"hash": file_hash, "ids": ids_by_source.get(rel, [])}
    for rel in removed:
        manifest.pop(rel, None)
    _save_manifest(manifest_path, manifest)

    logger.info("索引构建完成 ✅ 写入 %d 个片段，清单：%s", written, manifest_path)


if __name__ == "__main__":
    main()
