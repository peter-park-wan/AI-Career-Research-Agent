"""HTTP 入参模型（请求体）。

放在 ``schemas`` 而不是路由文件里，是为了让"对外契约"与"实现细节"分离：
契约变更是需要 review 的事，改路由实现不需要。响应体目前仍是普通 dict
（结构与前端约定一致，改动收益低于成本），此处只收敛入参。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    role: str = Field(..., description="user / assistant / system / tool")
    content: Any = Field(..., description="文本内容或结构化内容")


class ResearchRequest(BaseModel):
    messages: List[MessageItem] = Field(..., description="对话消息列表")
    configurable: Optional[Dict[str, Any]] = Field(
        default=None, description="覆盖 Configuration 中的 configurable 参数"
    )
    thread_id: Optional[str] = Field(default=None, description="会话线程 ID，用于多轮记忆")
    recursion_limit: int = Field(default=50, ge=1, description="图递归上限")


class CareerStageRequest(BaseModel):
    target_role: str = Field(default="AI工程师", description="目标岗位")
    target_city: str = Field(default="北京", description="目标城市")
    jd_text: Optional[str] = Field(default=None, description="可选 JD，用于更精准分析")
    configurable: Optional[Dict[str, Any]] = Field(default=None)
    thread_id: Optional[str] = Field(default=None)


class GapAnalysisRequest(BaseModel):
    jd_text: str = Field(..., min_length=1, description="待分析的职位描述文本")
    configurable: Optional[Dict[str, Any]] = Field(default=None)
    thread_id: Optional[str] = Field(default=None)


class CareerWorkflowRequest(BaseModel):
    target_role: str = Field(default="AI工程师", description="目标岗位")
    target_city: str = Field(default="北京", description="目标城市")
    jd_text: Optional[str] = Field(default=None, description="具体 JD 文本（可跳过发现/调研）")
    configurable: Optional[Dict[str, Any]] = Field(default=None)
    thread_id: Optional[str] = Field(default=None)


class ProfileSaveRequest(BaseModel):
    """保存画像：name 为空表示直接覆盖当前生效的简历文件。"""

    content: str = Field(..., description="画像 / 简历全文")
    name: Optional[str] = Field(default=None, description="档案名；给出则存为档案")


class ProfileActivateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="要启用的档案名")
