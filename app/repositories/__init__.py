"""仓储层：外部资源（文件 / 数据库）读写的唯一入口。"""

from app.repositories.profile_repo import ProfileRepository

__all__ = ["ProfileRepository"]
