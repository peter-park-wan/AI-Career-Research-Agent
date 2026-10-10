"""图片 / 扫描件的 OCR 入库分支。

【要防的是什么】

1. **有文本层的 PDF 被送去 OCR**：白白慢几十倍，还可能引入识别错误。
2. **扫描件被当成"空文件"直接丢弃**：扫描件是真实存在的输入形式，
   静默跳过会让用户以为入库了、实际查不到。
3. **没装 OCR 依赖时崩溃或静默**：必须给出可执行的安装命令。
"""

from __future__ import annotations

import pytest

from open_deep_research import ingest


@pytest.fixture
def fitz():
    return pytest.importorskip("fitz")


def _make_text_pdf(fitz, path, content="ByteDance AI Engineer Requirements"):
    # 用英文而非中文：PyMuPDF 内置字体没有中文字形，写进去会变成一串点，
    # 那样测出来的失败跟"有没有文本层"无关，是字体问题。
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), content)
    doc.save(str(path))
    doc.close()
    return path


def test_pdf_with_text_layer_does_not_ocr(fitz, tmp_path, monkeypatch):
    """有文本层就直接用，不能走 OCR：慢几十倍且会引入识别误差。"""
    path = _make_text_pdf(fitz, tmp_path / "a.pdf")

    called = []
    monkeypatch.setattr(ingest, "_ocr_pdf_pages", lambda d: called.append(1) or "OCR")

    text, info = ingest._read_pdf(str(path))
    assert "ByteDance" in text
    assert called == []  # 没触发 OCR
    assert info["image_only"] is False


def test_image_only_pdf_falls_back_to_ocr(fitz, tmp_path, monkeypatch):
    """扫描件（无文本层）必须走 OCR 抢救，而不是当成空文件丢弃。"""
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 200, 200))
    page.insert_image(fitz.Rect(0, 0, 200, 200), pixmap=pix)
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    doc.close()

    monkeypatch.setattr(ingest, "_ocr_pdf_pages", lambda d: "OCR_RECOVERED")
    text, info = ingest._read_pdf(str(path))

    assert text == "OCR_RECOVERED"
    assert info["image_only"] is True
    assert info["ocr"] == "rapidocr"


def test_read_image_requires_ocr_dependency(monkeypatch):
    """没装依赖时，错误必须告诉用户该执行什么，而不是抛 ImportError。"""
    monkeypatch.setattr(ingest, "_get_ocr", lambda: None)
    with pytest.raises(RuntimeError) as exc:
        ingest._read_image("whatever.png")
    assert "pip install rapidocr-onnxruntime" in str(exc.value)


def test_image_extensions_are_supported():
    assert ingest._SUPPORTED["**/*.png"] == "image"
    assert ingest._SUPPORTED["**/*.jpg"] == "image"
    # 文本类不受影响
    assert ingest._SUPPORTED["**/*.md"] == "text"


def test_ocr_installed_matches_actual_dependency():
    """探测结果必须与真实可导入性一致，否则会误报"不支持"或误判可用。"""
    import importlib.util

    expected = importlib.util.find_spec("rapidocr_onnxruntime") is not None
    assert ingest._ocr_installed() is expected
