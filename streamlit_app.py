"""Streamlit 中文前端：AI Career Research Agent 可视化界面。

本模块只做「展示 + 调用」，不含任何业务逻辑：所有能力都通过 HTTP 调用
FastAPI 服务（``open_deep_research.api``）实现。

启动方式::

    # 先启动后端 API 服务（默认 8000 端口）
    uvicorn open_deep_research.api:app --host 0.0.0.0 --port 8000

    # 再启动前端（默认 8501 端口）
    streamlit run streamlit_app.py

前端需要的依赖已拆到 ``ui`` extras：``pip install -e ".[ui]"``。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterator, Optional

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# 基础配置
# --------------------------------------------------------------------------- #
DEFAULT_API_BASE = "http://localhost:8000"
SAMPLE_JD = """招聘 AI 工程师（社招，3-5 年经验）

岗位职责：
1. 负责大模型应用（RAG / Agent）的设计、开发与落地；
2. 参与推荐、搜索等场景的算法优化；
3. 与产品、工程团队协作，推动模型上线与迭代。

任职要求：
1. 本科及以上学历，计算机 / 数学 / 人工智能相关专业；
2. 熟悉 Python，扎实的数据结构与算法基础；
3. 熟练使用 PyTorch，有 LLM 微调 / 推理优化经验者优先；
4. 了解 LangChain / LangGraph、向量数据库（Milvus / Chroma）者优先；
5. 有大规模数据处理经验，熟悉 Docker、Kubernetes 加分。
"""

# 页面标题 -> 渲染函数，见文件末尾的 PAGES
PAGES: Dict[str, Any] = {}


def page(title: str):
    """注册页面函数的装饰器，用于构造侧边栏导航。"""

    def _wrap(func):
        PAGES[title] = func
        return func

    return _wrap


# --------------------------------------------------------------------------- #
# 会话状态
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """初始化 Streamlit 会话状态（连接配置 + 各阶段结果缓存）。"""
    defaults = {
        "api_base": DEFAULT_API_BASE,
        "token": "",
        "thread_id": f"streamlit-{uuid.uuid4().hex[:8]}",
        "target_role": "AI工程师",
        "target_city": "北京",
        "configurable_json": "",
        "results": {},  # key -> 各阶段结果，切换页面不丢失
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_configurable() -> Optional[Dict[str, Any]]:
    """解析侧边栏的高级配置 JSON，返回 configurable 字典或 None。"""
    raw = (st.session_state.configurable_json or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        st.sidebar.error(f"高级配置 JSON 解析失败：{e}")
        return None
    if not isinstance(parsed, dict):
        st.sidebar.error("高级配置必须是 JSON 对象（如 {\"search_api\": \"tavily\"}）")
        return None
    return parsed


# --------------------------------------------------------------------------- #
# HTTP 调用封装
# --------------------------------------------------------------------------- #
def _headers() -> Dict[str, str]:
    """构造请求头，按需带上 Bearer Token。"""
    headers = {"Content-Type": "application/json"}
    token = st.session_state.get("token", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _base_url() -> str:
    """返回去除了结尾斜杠的 API 根地址。"""
    return str(st.session_state.api_base).rstrip("/")


def api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20):
    """调用 GET 接口，成功返回 dict，失败显示错误并返回 None。"""
    try:
        resp = requests.get(
            f"{_base_url()}{path}", headers=_headers(), params=params, timeout=timeout
        )
    except requests.RequestException as e:
        st.error(f"无法连接后端服务（{_base_url()}{path}）：{e}")
        return None
    if resp.status_code != 200:
        st.error(f"请求失败（HTTP {resp.status_code}）：{_safe_detail(resp)}")
        return None
    return resp.json()


def api_post(path: str, payload: Dict[str, Any], timeout: int = 1800):
    """调用 POST 接口，成功返回 dict，失败显示错误并返回 None。

    timeout 默认给到 30 分钟：深度研究 / 端到端工作流可能耗时很久，
    而 HTTP 读超时不应该由前端单方面掐断。
    """
    try:
        resp = requests.post(
            f"{_base_url()}{path}",
            headers=_headers(),
            json=payload,
            timeout=(10, timeout),
        )
    except requests.RequestException as e:
        st.error(f"无法连接后端服务（{_base_url()}{path}）：{e}")
        return None
    if resp.status_code != 200:
        st.error(f"请求失败（HTTP {resp.status_code}）：{_safe_detail(resp)}")
        return None
    return resp.json()


def _safe_detail(resp: requests.Response) -> str:
    """尽量从错误响应中取出可读的 detail 字段。"""
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:500]
    if isinstance(body, dict):
        return str(body.get("detail", body))[:500]
    return str(body)[:500]


def stream_research(payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """以 SSE 方式调用流式研究接口，逐个 yield 已解析的事件字典。"""
    url = f"{_base_url()}/api/research/stream"
    try:
        resp = requests.post(
            url, headers=_headers(), json=payload, stream=True, timeout=(10, 1800)
        )
    except requests.RequestException as e:
        yield {"type": "error", "content": f"无法连接后端服务：{e}"}
        return

    if resp.status_code != 200:
        yield {
            "type": "error",
            "content": f"请求失败（HTTP {resp.status_code}）：{_safe_detail(resp)}",
        }
        return

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


# --------------------------------------------------------------------------- #
# 渲染辅助
# --------------------------------------------------------------------------- #
def _as_text(content: Any) -> str:
    """把接口返回的 content（可能是多模态 list）规整为纯文本。"""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def render_content(title: str, content: Any, download_name: Optional[str] = None) -> None:
    """渲染一段结果：字符串按 Markdown 展示，结构化数据用 JSON 展示。

    同时提供 Markdown 下载按钮，便于把分析结果带走使用。
    """
    if content is None or (isinstance(content, str) and not content.strip()):
        st.info(f"{title}：暂无内容（若结果为空，请检查后端配置或简历文件）")
        return

    st.subheader(title)
    if isinstance(content, str):
        st.markdown(content)
        download_body = content
    elif isinstance(content, (dict, list)):
        st.json(content)
        download_body = json.dumps(content, ensure_ascii=False, indent=2)
    else:
        st.write(content)
        download_body = str(content)

    if download_name:
        st.download_button(
            label=f"⬇️ 下载《{title}》（Markdown）",
            data=download_body,
            file_name=f"{download_name}-{time.strftime('%Y%m%d-%H%M%S')}.md",
            mime="text/markdown",
        )


def save_result(key: str, value: Any) -> None:
    """把结果写入会话缓存，便于切换页面后仍可查看。"""
    st.session_state.results[key] = value


def common_payload(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造求职类接口的公共请求体（岗位 / 城市 / 记忆 / 配置）。"""
    payload: Dict[str, Any] = {
        "target_role": st.session_state.target_role,
        "target_city": st.session_state.target_city,
        "thread_id": st.session_state.thread_id,
        "configurable": get_configurable(),
    }
    if extra:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    """渲染侧边栏连接配置，返回当前选中的页面标题。"""
    st.sidebar.title("⚙️ 连接配置")

    st.sidebar.text_input(
        "后端 API 地址",
        key="api_base",
        help="FastAPI 服务地址，例如 http://localhost:8000",
    )
    st.sidebar.text_input(
        "Bearer Token（可选）",
        key="token",
        type="password",
        help="仅当后端设置了 API_BEARER_TOKEN 时需要填写",
    )
    st.sidebar.text_input(
        "会话 thread_id",
        key="thread_id",
        help="相同 thread_id 可跨请求复用多轮记忆",
    )
    st.sidebar.text_input("目标岗位", key="target_role")
    st.sidebar.text_input("目标城市", key="target_city")
    st.sidebar.text_area(
        "高级配置 configurable（JSON，可选）",
        key="configurable_json",
        height=80,
        help="覆盖 Configuration 配置项，例如 {\"search_api\": \"tavily\"}",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🩺 检测后端连接"):
        health = api_get("/health", timeout=10)
        if health:
            st.sidebar.success(f"后端正常：{health}")

    st.sidebar.markdown("---")
    return st.sidebar.radio("功能导航", list(PAGES.keys()))


# --------------------------------------------------------------------------- #
# 页面：服务与画像
# --------------------------------------------------------------------------- #
@page("🏠 服务与画像")
def page_overview() -> None:
    """展示后端服务状态、可用端点与候选人背景画像。"""
    st.header("🏠 服务与候选人画像")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**后端地址**")
        st.code(_base_url())
    with col2:
        st.markdown("**会话 ID**")
        st.code(st.session_state.thread_id)

    health = api_get("/health", timeout=10)
    if health:
        st.success(f"服务状态：{health.get('status', 'ok')}")

    with st.expander("📡 可用接口一览", expanded=False):
        info = api_get("/", timeout=10)
        if info:
            st.json(info)

    st.markdown("---")
    if st.button("🔄 重新加载候选人画像"):
        st.session_state.results.pop("profile", None)

    if "profile" not in st.session_state.results:
        with st.spinner("正在加载候选人画像…"):
            resp = api_get("/api/profile", timeout=60)
            if resp is not None:
                save_result("profile", resp.get("profile"))

    if "profile" in st.session_state.results:
        render_content("候选人画像", st.session_state.results["profile"], "profile")


# --------------------------------------------------------------------------- #
# 页面：岗位发现
# --------------------------------------------------------------------------- #
@page("🔍 岗位发现")
def page_discover_jobs() -> None:
    """根据目标岗位与城市推荐优先投递方向。"""
    st.header("🔍 岗位发现")
    st.caption(f"目标岗位：{st.session_state.target_role}｜城市：{st.session_state.target_city}")

    if st.button("🚀 开始岗位发现", type="primary"):
        with st.spinner("正在分析岗位方向与公司类型…"):
            resp = api_post("/api/career/discover-jobs", common_payload())
            if resp is not None:
                save_result("discovery", resp.get("discovery"))

    if "discovery" in st.session_state.results:
        render_content("岗位发现结果", st.session_state.results["discovery"], "岗位发现")


# --------------------------------------------------------------------------- #
# 页面：匹配度分析
# --------------------------------------------------------------------------- #
@page("📊 匹配度分析")
def page_gap_analysis() -> None:
    """粘贴 JD 做岗位匹配度（Gap）分析。"""
    st.header("📊 岗位匹配度分析")

    if st.button("📋 填入示例 JD"):
        st.session_state["jd_text"] = SAMPLE_JD

    jd_text = st.text_area(
        "职位描述（JD）",
        key="jd_text",
        height=240,
        placeholder="粘贴目标岗位的 JD 文本，越详细分析越准确…",
    )

    if st.button("🚀 开始匹配度分析", type="primary"):
        if not jd_text.strip():
            st.warning("请先填写 JD 文本（可点击上方「填入示例 JD」）")
        else:
            with st.spinner("正在对比画像与 JD，生成差距分析…"):
                resp = api_post(
                    "/api/career/gap-analysis",
                    {
                        "jd_text": jd_text,
                        "configurable": get_configurable(),
                        "thread_id": st.session_state.thread_id,
                    },
                )
                if resp is not None:
                    save_result("gap_analysis", resp.get("gap_analysis"))

    if "gap_analysis" in st.session_state.results:
        render_content("匹配度分析", st.session_state.results["gap_analysis"], "匹配度分析")


# --------------------------------------------------------------------------- #
# 页面：面试准备 / 简历优化 / 求职信
# --------------------------------------------------------------------------- #
def _render_stage_page(
    title: str,
    icon: str,
    endpoint: str,
    result_key: str,
    result_label: str,
    spinner_text: str,
) -> None:
    """渲染「目标岗位 + 可选 JD」这一类通用阶段的页面。"""
    st.header(f"{icon} {title}")
    st.caption(f"目标岗位：{st.session_state.target_role}｜城市：{st.session_state.target_city}")

    jd_text = st.text_area(
        "职位描述（可选，填写后分析更精准）",
        key=f"jd_{result_key}",
        height=160,
    )

    if st.button(f"🚀 生成{title}", type="primary"):
        with st.spinner(spinner_text):
            resp = api_post(endpoint, common_payload({"jd_text": jd_text or None}))
            if resp is not None:
                save_result(result_key, resp.get(result_key))

    if result_key in st.session_state.results:
        render_content(result_label, st.session_state.results[result_key], result_label)


@page("🎤 面试准备")
def page_interview() -> None:
    """生成面试备考清单。"""
    _render_stage_page(
        "面试准备",
        "🎤",
        "/api/career/interview",
        "interview_prep",
        "面试准备清单",
        "正在生成面试备考清单…",
    )


@page("📄 简历优化")
def page_resume() -> None:
    """生成简历优化建议与改写 bullet。"""
    _render_stage_page(
        "简历优化",
        "📄",
        "/api/career/resume",
        "resume_optimization",
        "简历优化建议",
        "正在生成简历优化建议…",
    )


@page("✉️ 求职信")
def page_cover_letter() -> None:
    """生成中文求职信。"""
    _render_stage_page(
        "求职信",
        "✉️",
        "/api/career/cover-letter",
        "cover_letter",
        "求职信",
        "正在撰写求职信…",
    )


# --------------------------------------------------------------------------- #
# 页面：端到端工作流
# --------------------------------------------------------------------------- #
@page("🚀 端到端工作流")
def page_workflow() -> None:
    """一次跑完六个阶段，产出整合报告与各阶段产物。"""
    st.header("🚀 端到端求职工作流")
    st.caption("岗位发现 → 调研 → 匹配度 → 面试准备 → 简历优化 → 求职信")

    jd_text = st.text_area(
        "职位描述（可选）",
        key="jd_workflow",
        height=150,
        help="填写后可跳过岗位发现/调研阶段，直接进入匹配度分析",
    )

    if st.button("🚀 运行完整工作流", type="primary"):
        with st.spinner("工作流执行中（含多次 LLM 调用，可能需要数分钟）…"):
            resp = api_post(
                "/api/career/workflow",
                common_payload({"jd_text": jd_text or None}),
            )
            if resp is not None:
                save_result("workflow", resp)

    if "workflow" not in st.session_state.results:
        return

    wf = st.session_state.results["workflow"]
    render_content("整合报告", wf.get("report"), "求职工作流报告")

    artifacts = wf.get("artifacts") or {}
    labels = {
        "discovery": "岗位发现",
        "research": "岗位调研",
        "gap_analysis": "匹配度分析",
        "interview_prep": "面试准备",
        "resume_optimization": "简历优化",
        "cover_letter": "求职信",
    }
    for key, label in labels.items():
        if artifacts.get(key):
            with st.expander(f"📦 {label}", expanded=False):
                content = artifacts[key]
                if isinstance(content, str):
                    st.markdown(content)
                else:
                    st.json(content)


# --------------------------------------------------------------------------- #
# 页面：深度研究（流式）
# --------------------------------------------------------------------------- #
@page("🧠 深度研究（流式）")
def page_deep_research() -> None:
    """调用多智能体深度研究图，实时展示 SSE 流式输出。"""
    st.header("🧠 深度研究（流式输出）")
    st.caption("基于 LangGraph 多智能体研究图，实时展示执行过程与最终报告")

    question = st.text_area(
        "研究问题",
        key="research_question",
        height=120,
        placeholder="例如：分析深圳 AI 工程师的就业前景与薪资水平",
    )

    if st.button("🚀 开始研究", type="primary"):
        if not question.strip():
            st.warning("请先填写研究问题")
        else:
            payload = {
                "messages": [{"role": "user", "content": question}],
                "thread_id": st.session_state.thread_id,
                "configurable": get_configurable(),
            }

            node_hint = st.empty()
            stream_box = st.empty()
            accumulated = ""
            final_answer = ""

            for event in stream_research(payload):
                etype = event.get("type")
                if etype == "error":
                    st.error(event.get("content", "研究失败"))
                    break
                if etype == "result":
                    final_answer = event.get("content", "")
                    continue

                node = event.get("langgraph_node")
                if node:
                    node_hint.caption(f"⚙️ 当前节点：{node}")
                chunk = _as_text(event.get("content") or "")
                if chunk:
                    accumulated += chunk
                    stream_box.markdown(accumulated)

            stream_box.empty()
            node_hint.empty()
            result_text = final_answer or accumulated
            save_result("research", result_text)
            st.session_state.results["research_process"] = accumulated

    if "research" in st.session_state.results:
        render_content("最终研究报告", st.session_state.results["research"], "深度研究报告")
        with st.expander("🪵 查看流式执行过程", expanded=False):
            st.text(st.session_state.results.get("research_process", ""))


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> None:
    """应用入口：渲染页面配置、侧边栏与当前选中页面。"""
    st.set_page_config(
        page_title="AI 求职研究助手",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.title("🎯 AI 求职研究助手")
    st.caption("基于 LangGraph 多智能体工作流的智能求职研究平台")

    current = render_sidebar()

    # 轻量探测后端连通性（本地通常 <50ms），未连通时给出明确指引
    if api_get("/health", timeout=3) is None:
        st.warning(
            f"⚠️ 未能连接到后端 API 服务 `{_base_url()}`。请先在另一个终端启动后端：\n\n"
            "```bash\n"
            "pip install -e \".[api]\"\n"
            "uvicorn open_deep_research.api:app --host 0.0.0.0 --port 8000\n"
            "```"
        )

    PAGES[current]()


main()
