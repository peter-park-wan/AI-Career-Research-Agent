"""资料档案仓储层：**本项目唯一直接触碰文件系统的地方**。

【分层的意义】

改造前"读简历文件"这件事同时存在于 ``profile.py``（懒加载缓存）、
``api.py``（若即若离的 `_write_text`）和 streamlit 前端里，路径拼接、编码、
目录创建各写一遍。任何一处改了默认目录，另外两处就静默不一致。

现在文件 IO 收敛到本层，并遵守两条纪律：

1. **只做 IO，不做业务判断**。是否允许这个名字、内容是否合法，属于
   ``ProfileService``。仓储层只负责"把 bytes 写到某个路径"。
2. **不阻塞事件循环**。所有读写都通过 ``anyio.to_thread.run_sync`` 扔到线程池。
   文件 IO 是阻塞系统调用，直接在 async 路由里 `open()` 会卡死整个事件循环
   —— 一个 5MB 的简历上传就能让同进程的其他请求全部排队。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import anyio

from app.core.errors import NotFoundError

ARCHIVE_SUFFIX = ".md"
ACTIVE_MARKER = ".active"


class ProfileRepository:
    """基于本地目录的画像档案仓储。"""

    def __init__(self, profiles_dir: str | Path, resume_path: str | Path) -> None:
        self._profiles_dir = Path(profiles_dir)
        self._resume_path = Path(resume_path)

    # ------------------------------------------------------------------ #
    # 路径
    # ------------------------------------------------------------------ #
    @property
    def profiles_dir(self) -> Path:
        return self._profiles_dir

    @property
    def resume_path(self) -> Path:
        """当前生效的简历文件（Agent 运行时读取的就是它）。"""
        return self._resume_path

    def archive_path(self, name: str) -> Path:
        return self._profiles_dir / f"{name}{ARCHIVE_SUFFIX}"

    @property
    def _active_marker(self) -> Path:
        return self._profiles_dir / ACTIVE_MARKER

    # ------------------------------------------------------------------ #
    # 同步实现（在线程池中执行，禁止直接调用）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_text(path: Path, default: str = "") -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return default

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再原子替换：避免写一半被读到半截内容
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _list_dirs_files(directory: Path) -> List[Path]:
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.glob(f"*{ARCHIVE_SUFFIX}") if p.is_file())

    def _list_names(self) -> List[str]:
        return [p.stem for p in self._list_dirs_files(self._profiles_dir)]

    def _read_active_name(self) -> Optional[str]:
        raw = self._read_text(self._active_marker).strip()
        return raw or None

    def _delete(self, path: Path) -> bool:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------ #
    # 异步对外接口
    # ------------------------------------------------------------------ #
    async def list_names(self) -> List[str]:
        return await anyio.to_thread.run_sync(self._list_names)

    async def active_name(self) -> Optional[str]:
        return await anyio.to_thread.run_sync(self._read_active_name)

    async def exists(self, name: str) -> bool:
        return await anyio.to_thread.run_sync(self.archive_path(name).is_file)

    async def read_archive(self, name: str) -> str:
        path = self.archive_path(name)
        content = await anyio.to_thread.run_sync(self._read_text, path)
        if not content:
            raise NotFoundError("档案", name)
        return content

    async def write_archive(self, name: str, content: str) -> Path:
        path = self.archive_path(name)
        await anyio.to_thread.run_sync(self._write_text, path, content)
        return path

    async def delete_archive(self, name: str) -> bool:
        return await anyio.to_thread.run_sync(self._delete, self.archive_path(name))

    async def read_resume(self) -> str:
        """读取当前生效的简历文件；不存在返回空串。"""
        return await anyio.to_thread.run_sync(self._read_text, self._resume_path)

    async def write_resume(self, content: str) -> Path:
        path = self._resume_path
        await anyio.to_thread.run_sync(self._write_text, path, content)
        return path

    async def set_active(self, name: Optional[str]) -> None:
        if name:
            await anyio.to_thread.run_sync(self._write_text, self._active_marker, name)
        else:
            await anyio.to_thread.run_sync(self._delete, self._active_marker)

    async def write_bytes(self, name: str, data: bytes) -> Path:
        def _write() -> Path:
            path = self.archive_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path

        return await anyio.to_thread.run_sync(_write)
