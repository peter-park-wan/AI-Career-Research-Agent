"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


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
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""


class CareerConfig(BaseModel):
    """Configuration for career research features."""
    
    default_target_role: Optional[str] = Field(
        default="AI工程师",
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "description": "Paste your background summary here to override resume_path. The agent tailors research to it."
            }
        }
    )
    """Inline user profile text (overrides resume_path)"""
    enable_gap_analysis: Optional[bool] = Field(
        default=True,
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
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
        metadata={
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
        metadata={
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
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
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
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        metadata={
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
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )

    # RAG (Retrieval-Augmented Generation) Configuration
    rag_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "启用 RAG 私有知识库检索（需先运行索引脚本 ingest）。"
            }
        }
    )
    rag_embedding_model: str = Field(
        default="openai:text-embedding-3-small",
        metadata={
            "x_oap_ui_config": {
                "type": "string",
                "default": "openai:text-embedding-3-small",
                "description": "Embedding 模型。openai:<model> 使用 OpenAI，其他值视为本地 sentence-transformers 模型名。"
            }
        }
    )
    rag_vector_store: str = Field(
        default="chroma",
        metadata={
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
        metadata={
            "x_oap_ui_config": {
                "type": "string",
                "default": "./data/rag_index",
                "description": "Chroma 索引持久化目录。"
            }
        }
    )
    rag_collection: str = Field(
        default="career_kb",
        metadata={
            "x_oap_ui_config": {
                "type": "string",
                "default": "career_kb",
                "description": "向量库集合名称。"
            }
        }
    )
    rag_top_k: int = Field(
        default=4,
        metadata={
            "x_oap_ui_config": {
                "type": "integer",
                "default": 4,
                "description": "每次检索返回的最大片段数。"
            }
        }
    )
    
    # Career Research Configuration
    career_config: Optional[CareerConfig] = Field(
        default=None,
        optional=True,
        metadata={
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

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True