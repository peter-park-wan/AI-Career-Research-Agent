# Agent 全栈学习方案

> 练兵场：`AI_Career_Research_Agent`（你的真实项目，不另开玩具工程）
> 目标：把「Agent 能跑」升级为「Agent 能交付」

---

## 0. 先破一个误区：Agent 全栈 ≠ 前端 + 后端

传统 Web 全栈的核心矛盾是「**请求-响应**」；Agent 全栈的核心矛盾是「**长时、有状态、要看得见过程**」。
同一个「全栈」词，难度分布完全不同：

| 维度 | 传统 Web | Agent 应用 | 你的项目现状 |
|---|---|---|---|
| 单请求耗时 | 毫秒级 | 分钟级（LLM + 多次工具调用） | `/api/research` 同步 `await`，靠 `API_RESEARCH_TIMEOUT` 兜底 |
| 幂等性 | 天然幂等 | 非幂等（每跑一次花一次钱） | 无幂等键，重复提交 = 重复烧 token |
| 响应形态 | 一次性 JSON | 流式（token / 工具事件 / 状态迁移） | `/api/research/stream` 已有 SSE ✅ 但协议语义不完整 |
| 状态 | 业务数据在 DB | 会话状态 + 长期记忆 + 向量库 **三层** | 三层都已接入 ✅ 但默认全是进程内存 |
| 交互 | 表单提交 | 多轮澄清 + 人工确认（HITL）+ 过程可视化 | 后端有 `clarify_with_user` ✅，前端没有交互闭环 ❌ |
| 成本 | CPU / 内存 | **Token = 钱**，需缓存 / 配额 / 预算 | 有 `_GAP_CACHE` ✅，无配额、无计量 ❌ |
| 失败处理 | 重试即可 | 部分失败需断点续跑，工具失败要降级 | 无重试、无续跑、工具异常直接吞成 500 ❌ |
| 部署 | 无状态，随便扩 | 有状态 checkpointer/store，扩容要先做状态外置 | 进程内 `MemorySaver` / `InMemoryStore` ❌ |

**一句话结论**：你的项目已经越过了「Agent 能跑」的门槛（图编排、Reflection、三层记忆、RAG 都有），
真正的短板集中在 **服务层工程化**（分层 / 配置 / 错误 / 日志 / 异步 / 观测）和 **前端交付能力**。

---

## 1. 现状盘点：已具备 vs 待补齐

### 已经具备（说明你 Agent 侧不是零基础）

| 能力 | 位置 |
|---|---|
| LangGraph 多 Agent 树形编排（supervisor + 并行子研究员） | `src/open_deep_research/deep_researcher.py` |
| Reflexion 式 Critic→Revise 闭环 | `reflect_and_revise_report` |
| 三层记忆（会话 checkpointer / 长期 Store / 向量 RAG） | `memory.py`、`rag.py`、`deep_researcher.py` |
| 业务工作流编排 | `career_workflow.py` |
| 双入口：Studio（`langgraph dev`）+ HTTP（`api.py`） | `langgraph.json`、`api.py` |
| SSE 流式 | `POST /api/research/stream` |
| 简单前端 + 容器化 | `streamlit_app.py`、`Dockerfile`、`docker-compose.yml` |

### 待补齐（这就是全栈要学的清单）

1. **服务层无分层**：`api.py` 单文件 1008 行，路由 / 中间件 / 鉴权 / 请求模型 / 业务编排 / 文件 IO 全混在一起。
2. **配置散落**：`os.getenv` 在模块导入期求值约 20 处，缺值静默降级而不是启动失败。
3. **无类型化错误**：`HTTPException(500, f"...{e}")` 把内部异常原文返回给客户端。
4. **无结构化日志**：没有 `request_id`，一次请求跨 4 个服务 / 6 次 LLM 调用后无法串联。
5. **长任务同步阻塞**：一次研究跑 2 分钟，占住一个 HTTP 连接，无法查询进度 / 取消 / 续跑。
6. **前端是工具级**：Streamlit 适合内部演示，做不了流式 token、事件时间轴、HITL 卡片。
7. **状态全在进程内**：`MemorySaver` / `InMemoryStore` / 内存限流 / 内存缓存 → 无法多副本。
8. **无就绪探针 / 无指标 / 无成本计量 / 无优雅停机**：上线即黑盒。

---

## 2. 五阶段路线图

每个阶段都以**一个可运行的增量**收尾，绝不"学完再做"。

---

### 阶段 0 · 认知对齐与工程基线（0.5 天）

**目标**：建立 Agent 全栈的心智模型，能一眼看出一段代码属于哪一层。

**知识点**
- 7 条工程铁律（按 Feature 组织、控制器无业务逻辑、配置集中校验、错误类型化、边界校验、结构化日志、每层职责单一）
- 画三张图：**Agent 拓扑图**（节点 / 边 / 中断点）、**请求时序图**（HTTP → 图 → LLM → 工具 → SSE → 前端）、**数据流图**（会话态 / 长期记忆 / 向量库 / 业务库）
- 「算法包」与「服务包」的边界：`open_deep_research` 不该知道 HTTP 的存在

**动手任务**
1. 用 Mermaid 画出你项目当前的三张图
2. 给 `deep_researcher.py` 的每个节点标注：纯计算 / 外部 IO / LLM 调用 / 可中断点

**验收**：能口头回答「我要加一个'简历评分'功能，代码应该写在哪几个文件里？各自职责是什么？」

---

### 阶段 1 · 后端地基与分层（2–3 天）

**目标**：把 1008 行的 `api.py` 拆成可测试、可协作、可上线的服务层。

**知识点**
- 三层架构：`Router`（HTTP）→ `Service`（业务编排）→ `Repository`（数据访问）的职责红线
- 依赖注入：FastAPI `Depends` + `lru_cache` 单例；为什么不要在路由里 `new` 对象
- **两种配置必须分开**（你项目里最容易踩的坑）：
  - 进程级配置 → `pydantic-settings` 的 `Settings`（端口 / CORS / 上限 / 连接串）
  - Agent 运行时配置 → `RunnableConfig.configurable` 里的 `Configuration`（模型 / 搜索 API / 反思轮数）
- 类型化错误体系 + 全局异常处理器 + 统一错误响应体 `{title, status, detail, request_id}`
- 结构化 JSON 日志 + `request_id` 全链路贯穿 + 日志脱敏（不能记 token / 简历原文）
- 同步阻塞 IO 不能写在 `async def` 里（`anyio.to_thread.run_sync`），否则一个上传卡死整个事件循环
- 就绪探针 `/health`（存活）vs `/ready`（依赖可用）

**动手任务**
1. 新建 `app/` 包，与算法包 `src/open_deep_research/` 物理分离
2. `app/core/config.py` — `Settings`，启动即校验，生产禁 `*` CORS ✅ **（本课已示范）**
3. `app/core/errors.py` — `AppError` 体系 + 全局处理器 ✅ **（本课已示范）**
4. `app/core/logging.py` — JSON 日志 + `request_id` 中间件
5. `app/core/deps.py` — 鉴权依赖、`get_settings()` 注入
6. `app/repositories/profile_repo.py` + `app/services/profile_service.py` — 简历档案文件 IO 下沉
7. `app/routers/{health,profile,career,research}.py` — 只做「解析 → 校验 → 调 service → 成形」
8. `app/main.py` — 组装；`api.py` 退化为薄兼容层（`from app.main import app`）

**验收**
- 兼容层 `api.py` ≤ 20 行；`app/routers/` 里 `open(` 出现 0 次
- `ENV=prod` 且 `API_CORS_ORIGINS=*` → 进程启动即退出并打印明确原因
- 任一 4xx 响应体统一为 `{title, status, detail, request_id}`
- 一次 `/api/career/workflow` 的所有日志能用同一个 `request_id` 串起来

---

### 阶段 2 · Agent 即服务：异步任务与流式协议（3–4 天）

**目标**：分钟级、可能失败、需要过程可见的 Agent 任务，变成可靠的服务资源。

**知识点**
- 任务状态机：`queued → running → succeeded | failed | cancelled`
- 异步化演进：`BackgroundTasks`（够用）→ ARQ / Dramatiq / Celery（生产）；为什么长任务不能占住 HTTP 连接
- API 进程与 Worker 进程分离；`async` 不是"后台线程"
- SSE 协议设计：`event:` / `id:` / `data:` 语义、`Last-Event-ID` 断线续传、15s 心跳保活、多订阅者扇出
- 幂等键 `Idempotency-Key`：重复提交返回同一个 `task_id`
- 断点续跑：`checkpointer` + `thread_id` + LangGraph `interrupt()` 实现 Human-in-the-loop
- 重试策略：区分可重试（429 / 5xx / 网络抖动）与不可重试（4xx / 内容违规），指数退避 + 抖动
- 协作式取消：置标志 + 在节点边界检查（不要强杀协程）

**动手任务**
1. 任务资源：`POST /api/tasks`（返回 `task_id`）、`GET /api/tasks/{id}`（状态 + 结果）、`DELETE /api/tasks/{id}`（取消）
2. `GET /api/tasks/{id}/events` — 标准 SSE：带 `id` / `event`（`status` `token` `tool_call` `done` `error`）/ 心跳
3. 把 `research` 与 `career/workflow` 都改为「提交任务 + 订阅事件」模型
4. `clarify_with_user` 接上 HITL：Agent 提问 → 前端卡片 → `POST /api/tasks/{id}/resume` 续跑

**验收**
- 两个标签页订阅同一 `task_id`，事件序列一致（验证扇出）
- 断网 10 秒重连，带 `Last-Event-ID` 能补齐丢失事件
- 同一 `Idempotency-Key` 重复 POST 只产生一个任务，只烧一次 token
- 提交接口 P95 < 200ms，任务在后台跑完

---

### 阶段 3 · 前端工程化：从 Streamlit 到可交付产品（4–5 天）

**目标**：把 Agent 的「过程」变成用户看得懂、能干预的界面。

**知识点**
- 工程化：Vite + React + TS（轻量，适合学习）或 Next.js；`VITE_API_BASE_URL`；开发期 proxy 避免 CORS
- 状态分层：服务端状态（React Query）vs 客户端状态（Zustand / Context）——别把一切都塞进全局 store
- 类型安全：用后端 OpenAPI 生成 TS 类型（`openapi-typescript`），让"前后端类型漂移"在编译期暴露
- API 客户端：baseURL / 自动带 token / 401 静默刷新重放 / 错误映射成人话
- SSE 消费的坑：`EventSource` 只能 GET 且不能带自定义头 → 需要 POST 时用 `fetch` + `ReadableStream` 手写解析
- 流式渲染：增量 Markdown、`requestAnimationFrame` 批量 flush 防闪烁、自动滚动不打断用户
- 交互设计：澄清问题卡片、工具调用时间轴、可取消、乐观更新、skeleton、离线提示
- 错误映射：HTTP 状态 → 用户能懂的话；字段级校验错误贴到对应输入框

**动手任务**
1. `web/` 脚手架 + `lib/api-client.ts`（typed、带 token、401 自动刷新）
2. 生成 `generated/api.d.ts` 并接进请求层
3. 聊天页：输入 → 创建任务 → 订阅 SSE → 流式渲染报告 + 工具调用时间轴
4. 澄清卡片 + 任务列表（可取消 / 重试）
5. 简历档案管理页（迁移现有 4 个 profile 接口）

**验收**
- 后端改一个响应字段，前端 `tsc` 立刻报错
- token 边生成边渲染，不整段闪
- 401 → 静默刷新 → 重放，用户无感
- 4xx 不重试；5xx 重试 3 次指数退避

---

### 阶段 4 · 数据与状态：持久化、缓存、多副本（3 天）

**目标**：从"单机内存"升级为"可横向扩展"。

**知识点**
- Postgres + Alembic：迁移可回滚、禁止手改线上表、Review 迁移 SQL
- 连接池：pool size 估算、连接超时、事务边界、多语句写必须事务
- LangGraph 状态外置：`PostgresSaver`（会话）+ `PostgresStore`（长期记忆）；为什么内存版无法多副本
- Redis：分布式限流（令牌桶）、`_GAP_CACHE` 换成带 TTL 的 Redis 缓存、任务队列 broker
- 向量库选型：Chroma（本地单机）vs pgvector（与业务库同源、可事务一致）
- N+1 / 索引 / 慢查询；幂等写靠唯一约束兜底

**动手任务**
1. `docker-compose` 加 `postgres` + `redis`，引入 `alembic/`
2. 迁移建表：`tasks`、`profiles`、`idempotency_keys`
3. 缓存：`_GAP_CACHE` → Redis（TTL + 写时失效）
4. 限流：进程内滑动窗口 → Redis 令牌桶
5. `API_CHECKPOINTER=postgres` + `MEMORY_STORE=postgres`，验证重启不丢

**验收**
- `docker compose up --scale api=2`，两个副本共享同一 `thread_id` 的会话
- 重启全部容器，历史任务与用户画像仍在
- 缓存命中/未命中在日志里可区分

---

### 阶段 5 · 可观测、成本与上线（2–3 天）

**目标**：能上线、能排障、能控成本。

**知识点**
- 可观测三件套：结构化日志 / 指标（RED：Rate、Errors、Duration）/ 追踪（LangSmith + OpenTelemetry）
- 成本治理：token 计量与预算、按用户配额、超预算降级（换小模型 / 截断上下文）
- 安全：JWT + refresh + httpOnly cookie、RBAC、上传类型与大小校验、路径穿越防护、
  **RAG 文档是不可信输入（Prompt 注入防护）**、密钥管理
- 优雅停机：`SIGTERM` → 停止接新请求 → drain 在跑任务 → 关闭连接池；`liveness` / `readiness` 探针
- 部署：多阶段 Dockerfile、非 root 用户、镜像瘦身；CI（lint + 类型 + 测试 + 镜像扫描）；灰度与回滚

**动手任务**
1. `/metrics`（Prometheus 格式）：任务数、耗时、token 成本、LLM 失败率
2. `request_id` + `task_id` 贯穿日志与追踪；接 Sentry
3. LLM 预算中间件：超配额返回 429 并记录
4. 优雅停机 + 多阶段 Dockerfile + `HEALTHCHECK`
5. GitHub Actions：ruff + mypy + pytest + 镜像构建

**验收**
- 一次线上报错，靠一个 `request_id` 能串出完整链路（请求 → 节点 → LLM → 工具 → 响应）
- `docker compose stop -t 30` 时在跑任务能完成或被安全标记，不丢数据
- CI 全绿才允许合并

---

## 3. 里程碑地图

| 里程碑 | 完成于 | 你会得到什么 | 一句话 |
|---|---|---|---|
| M1 | 阶段 1 末 | 分层的服务层 + 配置/错误/日志地基 | **能跑得明白** |
| M2 | 阶段 2 末 | 异步任务 + 标准事件流 + HITL | **能扛得住** |
| M3 | 阶段 3 末 | 产品级前端 + 类型安全 | **能给人用** |
| M4 | 阶段 4 末 | Postgres/Redis + 多副本 | **能扩得开** |
| M5 | 阶段 5 末 | 指标 + 成本 + CI + 优雅停机 | **能管得起** |

预估总投入：**15–18 天**（每天 2–3 小时）。可随时在任一里程碑停下，都是完整可用的状态。

---

## 4. 我们怎么配合（每课固定 4 步）

1. **讲原理** — 对着你项目里的**真实代码和行号**讲，不讲空泛概念
2. **我示范一处** — 把最典型的那块写出来，逐行解释"为什么这么写"
3. **你写一处** — 同构的另一块交给你，卡住我再给参照实现，我 review 你的代码
4. **一起验收** — 跑真实命令（curl / pytest / docker）验证，不通过不算完

**产物**：每阶段一个可运行的增量 + 一份「我改了什么 / 为什么这么改」的说明。
**原则**：不另开玩具项目；不在你项目里留半成品（每课结束都能 `git diff` 看懂）。

---

## 5. 技术选型建议（可调整）

| 决策点 | 建议 | 理由 |
|---|---|---|
| 后端框架 | 保留 FastAPI | 已成熟，异步 + OpenAPI 原生 |
| 前端框架 | **Vite + React + TS** | 比 Next.js 少一层 SSR 心智负担，专注学"流式 + 类型安全" |
| 服务端状态 | TanStack Query | 请求缓存 / 重试 / 失效 开箱即用 |
| 类型同步 | `openapi-typescript` 生成 | 零运行时成本，编译期报错 |
| 任务队列 | ARQ（先）→ Celery（如需） | ARQ 基于 asyncio + Redis，与 FastAPI 同构 |
| 状态后端 | Postgres（checkpointer/store/业务库同源） | 少一套组件，事务一致 |
| 缓存/限流 | Redis | 唯一的分布式原语需求 |
| 流式 | SSE（不用 WebSocket） | Agent 场景是单向推送，SSE 更简单且能穿代理 |

---

## 6. 第 1 课已交付

见 `01-第1课-后端地基.md`，仓库中已落地：
- `app/core/config.py` — 进程级配置（启动即校验）
- `app/core/errors.py` — 类型化错误 + 全局异常处理器
