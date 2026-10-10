"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,

    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,

    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,

    )
    """Whether the MCP server requires authentication"""


class CareerConfig(BaseModel):
    """Configuration for career research features."""
    
    default_target_role: Optional[str] = Field(
        default="AI工程师",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "AI工程师",
                "description": "Default target job role for career research (e.g., AI工程师, 机器学习工程师)"
            }
        }
    )
    """Default target job role for career research"""
    
    default_target_city: Optional[str] = Field(
        default="北京",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "北京",
                "description": "Default target city for job search (e.g., 北京, 上海, 深圳)"
            }
        }
    )
    """Default target city for job search"""
    
    default_education: Optional[str] = Field(
        default="本科",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "本科",
                "description": "Default education level for career analysis"
            }
        }
    )
    """Default education level for career analysis"""
    
    default_experience_years: Optional[str] = Field(
        default="3年",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "3年",
                "description": "Default years of work experience for career analysis"
            }
        }
    )
    """Default years of work experience"""
    
    default_skills: Optional[str] = Field(
        default="Python, PyTorch, TensorFlow, 机器学习",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "Python, PyTorch, TensorFlow, 机器学习",
                "description": "Default skill list for career analysis (comma-separated)"
            }
        }
    )
    """Default skill list for career analysis"""
    
    enable_github_search: Optional[bool] = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to enable GitHub project search for career recommendations"
            }
        }
    )
    """Whether to enable GitHub project search"""
    
    max_github_results: Optional[int] = Field(
        default=10,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10,
                "min": 1,
                "max": 30,
                "description": "Maximum number of GitHub projects to return"
            }
        }
    )
    """Maximum number of GitHub projects to return"""
    resume_path: str = Field(
        default="./data/简历.md",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "./data/简历.md",
                "description": "Resume / background file path; injected into research prompts so the agent knows your profile."
            }
        }
    )
    """Resume / background file path (injected into prompts)"""
    user_profile: Optional[str] = Field(
        default=None,

        json_schema_extra={
            "x_oap_ui_config": {
                "type": "textarea",
                "description": "Paste your background summary here to override resume_path. The agent tailors research to it."
            }
        }
    )
    """Inline user profile text (overrides resume_path)"""
    enable_gap_analysis: Optional[bool] = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否启用岗位匹配度分析工具（analyze_job_fit / Gap Analysis）。"
            }
        }
    )
    """Whether to enable job-fit / gap analysis tool"""
    
    enable_career_workflow: Optional[bool] = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否启用端到端求职工作流工具（岗位发现/面试准备/简历优化/求职信）。"
            }
        }
    )
    """Whether to enable the end-to-end career workflow tools"""
    

    skill_match_threshold: Optional[float] = Field(
        default=0.6,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.6,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Threshold for skill match score to recommend a role"
            }
        }
    )
    """Threshold for skill match score"""
    
    learning_phase_duration_weeks: Optional[int] = Field(
        default=4,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4,
                "min": 2,
                "max": 12,
                "description": "Default duration (weeks) for each learning phase in the roadmap"
            }
        }
    )
    """Default duration for each learning phase"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_reflection_rounds: int = Field(
        default=1,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 1,
                "min": 0,
                "max": 3,
                "step": 1,
                "description": "报告生成后的自我批评-修订轮数。0 = 关闭反思（保持原行为），1 = 批评一轮并修订，2-3 = 多轮迭代（更高质量但更耗 token）。"
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        #default="openai:gpt-4.1",
        default="deepseek:deepseek-chat",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                #"default": "openai:gpt-4.1",
                "default": "deepseek:deepseek-chat",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        #default="openai:gpt-4.1",
        default="deepseek:deepseek-chat",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
               # "default": "openai:gpt-4.1",
                "default": "deepseek:deepseek-chat",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        #default="openai:gpt-4.1",
        default="deepseek:deepseek-chat",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                #"default": "openai:gpt-4.1",
                "default": "deepseek:deepseek-chat",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,

        json_schema_extra={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,

        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    # RAG (Retrieval-Augmented Generation) Configuration
    # 默认开启：简历 / 岗位 JD / 面经这类私有资料是研究的依据，关掉等于让
    # 研究员只靠公网搜索去猜。装配失败不会中断研究，而是降级为"不可用"
    # （见 rag.get_rag_store），所以默认开着不存在"服务起不来"的风险。
    rag_enabled: bool = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "启用 RAG 私有知识库检索（需先运行索引脚本 ingest）。"
            }
        }
    )
    # 部署级固定：整个部署只用这一个 embedding 模型，不对外提供选择。
    #
    # 为什么不在运行时让用户选：建库用什么模型，索引就是那个模型的向量空间，
    # 检索必须同模型才有效。一旦允许每次请求换模型，就得按模型隔离索引
    # （多份索引、维度冲突、换模型时幂等清单误判"已是最新"），复杂度远超收益。
    # 所以钉死一个值——要换就改这一处，然后重建索引。
    #
    # 为什么选本地 bge 而不是 OpenAI：官方端点在本项目部署环境下不可达
    # （APITimeoutError），而本地模型离线可用、无调用成本、中文效果更好。
    # 代价是要装 sentence-transformers（见 pyproject 的 local-embeddings extra）。
    rag_embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "BAAI/bge-small-zh-v1.5",
                "description": "Embedding 模型。openai:<model> 使用 OpenAI，其他值视为本地 sentence-transformers 模型名。"
            }
        }
    )
    rag_vector_store: str = Field(
        default="chroma",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "select",
                "default": "chroma",
                "options": [
                    {"label": "Chroma (本地)", "value": "chroma"},
                    {"label": "Supabase pgvector", "value": "supabase"}
                ],
                "description": "向量库类型（当前仅实现 chroma）。"
            }
        }
    )
    rag_index_path: str = Field(
        default="./data/rag_index",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "./data/rag_index",
                "description": "Chroma 索引持久化目录。"
            }
        }
    )
    rag_collection: str = Field(
        default="career_kb",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "string",
                "default": "career_kb",
                "description": "向量库集合名称。"
            }
        }
    )
    rag_top_k: int = Field(
        default=4,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "integer",
                "default": 4,
                "description": "每次检索返回的最大片段数。"
            }
        }
    )
    rag_force: bool = Field(
        default=False,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "强制优先检索私有知识库：在 RAG 启用时，要求 Agent 在联网检索前先检索简历/JD/面经等私有资料。"
            }
        }
    )

    # Career Research Configuration
    career_config: Optional[CareerConfig] = Field(
        default=None,

        json_schema_extra={
            "x_oap_ui_config": {
                "type": "object",
                "description": "Configuration for career research features"
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    model_config = ConfigDict(arbitrary_types_allowed=True)