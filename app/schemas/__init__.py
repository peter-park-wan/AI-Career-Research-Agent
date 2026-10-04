"""对外 HTTP 契约（请求/响应模型）。"""

from app.schemas.requests import (
    CareerStageRequest,
    CareerWorkflowRequest,
    GapAnalysisRequest,
    MessageItem,
    ProfileActivateRequest,
    ProfileSaveRequest,
    ResearchRequest,
)

__all__ = [
    "CareerStageRequest",
    "CareerWorkflowRequest",
    "GapAnalysisRequest",
    "MessageItem",
    "ProfileActivateRequest",
    "ProfileSaveRequest",
    "ResearchRequest",
]
