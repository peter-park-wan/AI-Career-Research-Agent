"""进程级配置（Settings）：端口 / CORS / 鉴权 / 上限 / 连接串。

【为什么要单独抽这一层】

改造前这些配置以 ``os.getenv(...)`` 的形式散落在 ``api.py`` 的十来个地方，
带来三个具体问题：

1. **读取时机不一致**：模块级 ``os.getenv`` 在 import 时求值一次，
   而函数内的 ``os.getenv`` 每次调用都重读。同一个 ``API_CHECKPOINTER``
   在两处可能得到不同结果，行为变得不可预测。
2. **没有类型与默认值的单一出处**：``int(os.getenv("API_MAX_MESSAGES", "50"))``
   这种写法一旦拼错变量名会静默退化成默认值，部署时"看起来配了其实没生效"。
3. **危险配置无人拦截**：生产环境用 ``CORS=*``、不设 Bearer Token、
   用进程内 MemorySaver 存会话 —— 这些都能正常启动，直到出事故才发现。

因此本模块把"换个部署就变"的配置集中成 ``Settings``：

    ``Settings``      —— 进程级：端口、CORS、鉴权、上限、连接串（本模块）
    ``Configuration`` —— 运行时：一次请求的模型、检索量、循环次数
                          （见 ``open_deep_research.configuration``）

判断标准很简单：**"换一台机器就要改的" 属于 Settings；"换一个用户/一次任务就要改的"
属于 Configuration**。两者混在一起的后果是 `Settings` 会被滥用成万能参数袋。

【fail fast】

``get_settings()`` 在 ``env=prod`` 时执行 ``assert_production_ready()``，
配置不安全就直接抛 ``ConfigError`` 让进程起不来。宁可起不来，也不要带着
``CORS=*`` + 内存 checkpointer 上线。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, List, Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.errors import ConfigError

# 语义化布尔值的可接受写法（pydantic 默认认不出 "on"/"off"）
_TRUTHY = {"on", "true", "1", "yes", "y"}
_FALSY = {"off", "false", "0", "no", "n", ""}

# 需要"空字符串视为未配置"的字段：.env 里写 `API_BEARER_TOKEN=` 很常见，
# 若不归一化，鉴权会认为"已开启"，但实际 token 是空串，导致所有请求 401。
_BLANKABLE = (
    "bearer_token",
    "checkpointer_postgres_dsn",
    "checkpointer_redis_uri",
    "memory_store_postgres_dsn",
    "postgres_dsn",
    "redis_uri",
)


class Settings(BaseSettings):
    """进程级配置。所有字段可用 ``API_`` 前缀的环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 运行环境 / 监听 ----------------
    env: Literal["dev", "staging", "prod"] = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # ---------------- CORS ----------------
    # 逗号分隔；空值 → ["*"]
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["*"]
    )
    cors_allow_credentials: bool = False

    # ---------------- 鉴权 ----------------
    # 为空表示不开启鉴权（仅适合 dev；prod 会被 assert_production_ready 拦下）
    bearer_token: Optional[str] = None

    # ---------------- 限流与上限 ----------------
    rate_limit_per_minute: int = 0  # 0 = 不限流
    max_messages: int = 50
    research_timeout: int = 0  # 秒；0 = 不限制
    upload_max_mb: int = 5

    # ---------------- 日志 ----------------
    log_json: bool = True
    log_level: str = "INFO"

    # ---------------- 档案 / 简历文件 ----------------
    profile_dir: str = "./data/profiles"
    resume_path: str = "./data/简历.md"

    # ---------------- LangGraph checkpointer（短期会话记忆） ----------------
    checkpointer: Literal["memory", "none", "postgres", "redis"] = "memory"
    checkpointer_postgres_dsn: Optional[str] = None
    checkpointer_redis_uri: Optional[str] = None

    # ---------------- LangGraph store（跨会话长期记忆） ----------------
    # 三个名字都接受：
    #   MEMORY_STORE      —— 既有部署脚本在用的名字，必须保持兼容
    #   API_MEMORY_STORE  —— 与其余字段保持一致的带前缀写法
    #   memory_store      —— 代码里 ``Settings(memory_store=...)`` 构造时要能用
    #
    # 只写 ``validation_alias="MEMORY_STORE"`` 会**替换**掉默认别名，后果有两处：
    #   1) 环境变量写 API_MEMORY_STORE 静默失效（alias 会覆盖 env_prefix）；
    #   2) ``Settings(memory_store="postgres")`` 也静默失效 —— 字段名不再被接受，
    #      而 extra="ignore" 让这个错误不报错，直接退回默认值。
    memory_store: str = Field(
        default="memory",
        validation_alias=AliasChoices("MEMORY_STORE", "API_MEMORY_STORE", "memory_store"),
    )
    memory_store_postgres_dsn: Optional[str] = None

    # 通用兜底连接串（回退用）
    postgres_dsn: Optional[str] = None
    redis_uri: Optional[str] = None

    # ---------------- 持久化后端降级策略 ----------------
    # 声明了 postgres/redis 但装配失败时，是否允许退回到内存版。
    # 默认 False：生产环境宁可起不来，也不能"声称会持久化、实际丢数据"。
    # 仅在明确的应急场景（如数据库故障但必须继续提供服务）显式设为 true。
    allow_memory_fallback: bool = False

    # ---------------- Gap 分析进程内缓存 ----------------
    gap_cache: bool = True
    gap_cache_max: int = 256

    # ------------------------------------------------------------------ #
    # 归一化
    # ------------------------------------------------------------------ #
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if value is None or value == "":
            return ["*"]
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",") if part.strip()]
            return items or ["*"]
        return value

    @field_validator(*_BLANKABLE, mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator(
        "gap_cache",
        "log_json",
        "cors_allow_credentials",
        "reload",
        "allow_memory_fallback",
        mode="before",
    )
    @classmethod
    def _coerce_bool(cls, value: object) -> object:
        """接受 on/off/yes/no 等人类写法，避免 `API_GAP_CACHE=on` 直接校验失败。"""
        if isinstance(value, str):
            token = value.strip().lower()
            if token in _TRUTHY:
                return True
            if token in _FALSY:
                return False
        return value

    @field_validator("checkpointer", "memory_store", mode="before")
    @classmethod
    def _normalize_backend(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, value: List[str]) -> List[str]:
        return [item.rstrip("/") for item in value]

    # ------------------------------------------------------------------ #
    # 派生属性
    # ------------------------------------------------------------------ #
    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def resolved_checkpointer_dsn(self) -> Optional[str]:
        return self.checkpointer_postgres_dsn or self.postgres_dsn

    @property
    def resolved_checkpointer_redis_uri(self) -> Optional[str]:
        return self.checkpointer_redis_uri or self.redis_uri

    @property
    def resolved_store_dsn(self) -> Optional[str]:
        return self.memory_store_postgres_dsn or self.postgres_dsn

    @property
    def allows_any_origin(self) -> bool:
        return "*" in self.cors_origins

    # ------------------------------------------------------------------ #
    # 启动校验
    # ------------------------------------------------------------------ #
    def production_problems(self) -> List[str]:
        """返回生产环境下的配置问题清单（空列表 = 通过）。"""
        problems: List[str] = []
        if self.allows_any_origin:
            problems.append(
                "API_CORS_ORIGINS 为 '*'，生产环境必须显式列出允许的来源"
            )
        if not self.bearer_token:
            problems.append("API_BEARER_TOKEN 未设置，/api/* 将对公网完全开放")
        if self.checkpointer == "memory":
            problems.append(
                "API_CHECKPOINTER=memory，会话记忆仅存在于单个进程内存中，"
                "重启即丢失且多副本无法共享，生产应使用 postgres/redis"
            )
        if self.checkpointer == "none":
            problems.append(
                "API_CHECKPOINTER=none，多轮对话记忆被完全关闭，"
                "生产应使用 postgres/redis"
            )
        if self.memory_store == "memory":
            problems.append(
                "MEMORY_STORE=memory，跨会话长期记忆无法持久化，生产应使用 postgres"
            )
        if self.log_json is False:
            problems.append("API_LOG_JSON=false，生产环境应输出结构化日志供采集")
        if self.reload:
            problems.append("API_RELOAD=true，热重载只应在开发环境开启")
        if self.host == "0.0.0.0" and not self.bearer_token:
            problems.append("监听 0.0.0.0 且无鉴权，等于把接口暴露给整个内网")
        return problems

    def assert_production_ready(self) -> None:
        """生产环境配置不合格时抛 ``ConfigError``，让进程启动即失败。"""
        problems = self.production_problems()
        if problems:
            raise ConfigError(
                "生产环境配置校验未通过：\n"
                + "\n".join(f"  - {item}" for item in problems)
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取进程内唯一配置实例。

    使用 ``lru_cache`` 而不是模块级全局变量，是为了在测试里可以通过
    ``get_settings.cache_clear()`` 干净地重建配置。
    """
    settings = Settings()
    if settings.is_prod:
        # fail fast：配置不安全就不要把进程拉起来
        settings.assert_production_ready()
    return settings


def clear_settings_cache() -> None:
    """清空配置缓存（测试 / 运行时改配置后调用）。"""
    get_settings.cache_clear()
