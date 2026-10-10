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

【API Key 预检】

"缺哪个 API Key"不是本模块能静态断定的：真正决定它的是 ``Configuration``
的**默认值**（用哪家搜索、哪个模型）。所以本模块**不二次声明**这份名单，
而是从 ``Configuration`` 的默认值动态推导——硬编码名单必然随默认值漂移，
``docker-compose.yml`` 的 environment 白名单就是前车之鉴：它抄自上游模板，
项目把默认模型改成 deepseek 后没跟上，结果白名单里的 OpenAI/Anthropic
用不上、真正需要的 DeepSeek 反而没进容器。

分级与上面 fail fast 一致：
    生产缺失  → 启动失败（宁可起不来）
    开发/预发 → 只告警不拦截（你可能只想跑某一个模块，不该被整体卡住）
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Annotated, Any, List, Literal, Optional

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

# ------------------------------------------------------------------ #
# API Key 预检：从 Configuration 的默认值推导"需要哪些 key"
# ------------------------------------------------------------------ #

# 模型字符串形如 "deepseek:deepseek-chat"，冒号前即 provider。
# 值为 None 表示该 provider 不需要环境变量（本地模型等）。
# 未收录的 provider 一律按"不需要"处理：预检宁可漏报，也不能因为
# 不认识某个新 provider 而把正常的部署拦在门外。
_PROVIDER_TO_ENV_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "ollama": None,  # 本地推理，无 key
}

# 搜索后端 → 所需环境变量（取值见 configuration.SearchAPI）
_SEARCH_API_TO_ENV_KEY = {
    "tavily": "TAVILY_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "none": None,  # 关闭搜索
}

# 参与预检的模型字段：一次研究必然走到的环节。
# 故意不含 embedding_model —— 它只在构建 RAG 索引时使用，把"研究主链路"
# 和"建库链路"混在一起，会误伤只想跑研究、不建库的场景。
_PREFLIGHT_MODEL_FIELDS = (
    "summarization_model",
    "research_model",
    "compression_model",
    "final_report_model",
)


def _configuration_default(field_name: str) -> Any:
    """读取 ``open_deep_research.Configuration`` 某字段的默认值。

    用**函数内延迟 import** 而不是模块顶部 import，有两个原因：

    1. 避免仅仅为了读一个默认值就把 langchain 整条依赖链拉起来，拖慢启动；
    2. 万一 open_deep_research 不可用（未安装 / 正在重构），预检应当
       降级成"不报"，而不是让**配置校验本身**变成新的启动失败原因。
    """
    try:
        from open_deep_research.configuration import Configuration
    except Exception:  # pragma: no cover - 依赖不可用时的降级路径
        return None
    field = Configuration.model_fields.get(field_name)
    if field is None:  # pragma: no cover - 字段被改名/删除
        return None
    default = field.default
    # SearchAPI 是 Enum 而非 str Enum，默认值是成员对象；其余字段已是 str。
    return getattr(default, "value", default)


def _required_api_keys() -> List[str]:
    """按 Configuration 的默认值推导"跑一次研究至少需要哪些 key"。

    返回去重且顺序稳定的列表，便于日志阅读与测试断言。
    """
    needed: List[str] = []

    search_api = _configuration_default("search_api")
    if isinstance(search_api, str):
        key = _SEARCH_API_TO_ENV_KEY.get(search_api.strip().lower())
        if key:
            needed.append(key)

    for field_name in _PREFLIGHT_MODEL_FIELDS:
        model = _configuration_default(field_name)
        if not isinstance(model, str) or ":" not in model:
            continue  # 写法异常时不猜，交给实际调用处报错
        provider = model.split(":", 1)[0].strip().lower()
        key = _PROVIDER_TO_ENV_KEY.get(provider)
        if key:
            needed.append(key)

    return list(dict.fromkeys(needed))


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

    def missing_api_keys(self) -> List[str]:
        """默认研究配置需要、但当前环境里为空的 API key 名称列表。

        只针对**默认值**做判断：运行时请求可以覆盖模型与搜索后端，
        那时所需的 key 由请求携带，不是启动期能预知的。

        为什么这里用 ``os.getenv`` 而不是直接读 .env 文件：
        真正取 key 的逻辑（``utils.get_tavily_api_key``）同样走 ``os.getenv``，
        预检必须与它共用同一数据源才有意义。若改读文件，就会出现
        ".env 里填了、但进程环境未加载"的情况——预检放行而实际调用仍失败，
        等于把问题又推回运行时。反过来，若某个启动方式没把 .env 注入环境，
        这里报缺失是**准确的**：那种方式下研究确实会失败。
        """
        # key 由请求体提供（多租户）时，环境变量为空是正常的，不算缺失
        if os.getenv("GET_API_KEYS_FROM_CONFIG", "false").strip().lower() == "true":
            return []
        # 空字符串等同于未配置，与 _BLANKABLE 的口径保持一致
        return [key for key in _required_api_keys() if not (os.getenv(key) or "").strip()]

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
        for key in self.missing_api_keys():
            problems.append(
                f"默认研究配置需要 {key}，但环境变量为空："
                "研究会在首次调用该服务时失败（而非启动时）"
            )
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
    else:
        # 非生产环境不拦启动（你可能只想跑某一个模块，不该被整体卡住），
        # 但缺 key 必须在启动时说出来——否则只能等研究跑到一半抛
        # MissingAPIKeyError 才发现，而那个阶段的表现往往是"静默写了一份
        # 没有依据的报告"，比直接报错更难发现。
        missing = settings.missing_api_keys()
        if missing:
            logging.getLogger(__name__).warning(
                "以下 API Key 未配置，相关功能将在调用时失败：%s",
                ", ".join(missing),
            )
    return settings


def clear_settings_cache() -> None:
    """清空配置缓存（测试 / 运行时改配置后调用）。"""
    get_settings.cache_clear()
