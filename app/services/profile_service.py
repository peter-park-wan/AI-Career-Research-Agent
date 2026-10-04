"""画像 / 简历档案业务层。

【职责】
* 档案名合法性校验（安全边界：**杜绝路径穿越**）
* "正在使用的档案不允许删除" 这类业务规则
* 上传内容解析（md / txt / pdf）与体积上限

【为什么校验放在这里而不是仓储层】
仓储层只负责"把内容写到某个路径"。如果校验写在仓储层，任何绕过仓储的直接
调用都会失去保护；放在业务层，意味着所有入口都必然经过它。

档案名用 ``^[\\w\\- ]{1,64}$`` 约束：``\\w`` 不含 ``/`` ``\\`` ``.``，
因此 ``../../etc/passwd`` 之类的输入会在进入文件系统之前就被拒掉。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import anyio

from app.core.config import Settings, get_settings
from app.core.errors import (
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)
from app.repositories.profile_repo import ProfileRepository

logger = logging.getLogger(__name__)

_PROFILE_NAME_RE = re.compile(r"^[\w\- ]{1,64}$")
_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
_SUPPORTED_SUFFIXES = _TEXT_SUFFIXES | {".pdf"}
_PREVIEW_CHARS = 800


def validate_profile_name(name: str) -> str:
    """校验并归一化档案名。"""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValidationError("档案名不能为空")
    if not _PROFILE_NAME_RE.fullmatch(cleaned):
        raise ValidationError(
            "档案名只能包含中英文、数字、下划线与连字符（1-64 个字符）"
        )
    return cleaned


def decode_resume_bytes(data: bytes, suffix: str) -> str:
    """把上传字节解码为文本（同步实现，在线程池中执行）。"""
    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - 取决于部署环境
            raise UnsupportedMediaTypeError(
                "服务端未安装 pymupdf，暂时无法解析 PDF，请改传 .md / .txt"
            ) from exc
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)

    # 中文环境常见编码兜底：UTF-8 → GB18030
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


class ProfileService:
    """画像档案业务逻辑。"""

    def __init__(
        self,
        repo: ProfileRepository,
        settings: Optional[Settings] = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    async def list_profiles(self) -> Dict[str, Any]:
        names = await self._repo.list_names()
        active = await self._repo.active_name()
        if active and active not in names:
            # 档案被手工删除后 .active 会变成悬空引用，这里做自愈
            active = None
        return {"names": names, "active": active}

    async def read(self, name: Optional[str] = None) -> str:
        if name:
            return await self._repo.read_archive(validate_profile_name(name))
        return await self._repo.read_resume()

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    async def save(self, content: str, name: Optional[str] = None) -> Dict[str, Any]:
        if not content or not content.strip():
            raise ValidationError("内容不能为空")

        if name:
            cleaned = validate_profile_name(name)
            path = await self._repo.write_archive(cleaned, content)
        else:
            cleaned = None
            path = await self._repo.write_resume(content)

        logger.info(
            "档案已保存", extra={"path": str(path), "service": "profile"}
        )
        return {
            "name": cleaned,
            "chars": len(content),
            "path": str(path),
        }

    async def delete(self, name: str) -> Dict[str, Any]:
        cleaned = validate_profile_name(name)
        if await self._repo.active_name() == cleaned:
            raise ValidationError("不能删除正在使用的档案，请先切换到其他档案")
        if not await self._repo.delete_archive(cleaned):
            raise NotFoundError("档案", cleaned)
        return {"deleted": cleaned}

    async def activate(self, name: str) -> Dict[str, Any]:
        """把档案内容写入"当前生效的简历文件"，供 Agent 运行时读取。"""
        cleaned = validate_profile_name(name)
        content = await self._repo.read_archive(cleaned)  # 不存在会抛 NotFoundError
        path = await self._repo.write_resume(content)
        await self._repo.set_active(cleaned)
        logger.info("档案已启用", extra={"service": "profile", "path": str(path)})
        return {"active": cleaned, "chars": len(content), "path": str(path)}

    async def upload(
        self,
        filename: str,
        data: bytes,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        max_bytes = self._settings.upload_max_mb * 1024 * 1024
        if max_bytes > 0 and len(data) > max_bytes:
            raise PayloadTooLargeError(
                f"文件大小 {len(data) / 1024 / 1024:.1f}MB 超出上限 "
                f"{self._settings.upload_max_mb}MB"
            )

        suffix = Path(filename or "").suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise UnsupportedMediaTypeError(
                f"不支持的文件格式：{suffix or '未知'}（支持 .md / .markdown / .txt / .pdf）"
            )

        # PDF 解析是 CPU 密集的阻塞操作，必须让出事件循环
        text = (await anyio.to_thread.run_sync(decode_resume_bytes, data, suffix)).strip()
        if not text:
            raise ValidationError("未能从文件中解析出文本内容，请确认文件不是扫描图片版 PDF")

        saved = await self.save(text, name)
        saved["preview"] = text[:_PREVIEW_CHARS]
        return saved


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def build_profile_service(settings: Optional[Settings] = None) -> ProfileService:
    settings = settings or get_settings()
    repo = ProfileRepository(
        profiles_dir=settings.profile_dir,
        resume_path=settings.resume_path,
    )
    return ProfileService(repo, settings)
