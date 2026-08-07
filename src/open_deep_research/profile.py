"""Load the user's resume / background profile for career research.

This lets the agent "know who you are" by injecting your background text
(resume file or an inline profile) into the research prompts, so that
researchers can tailor findings to your skills, experience and goals.
"""
import os
from langchain_core.runnables import RunnableConfig

from open_deep_research.configuration import Configuration
from open_deep_research.memory import load_user_memory


def load_user_profile(config: RunnableConfig) -> str:
    """Return the user's background text to inject into prompts.

    Resolution priority:
        1. explicit ``user_profile`` text configured inline (highest)
        2. resume file located at ``resume_path``
        3. long-term memory recalled from the Store (cross-session)
        4. empty string (agent simply has no background context)

    The long-term memory (Tier-1) lets the agent "remember" who you are across
    separate research runs, e.g. target role, skills and applied companies.
    """
    configurable = Configuration.from_runnable_config(config)
    career = configurable.career_config
    if not career:
        return ""

    # 1. Explicit inline profile text takes highest priority
    explicit = getattr(career, "user_profile", None)
    if explicit and explicit.strip():
        return explicit.strip()

    # 2. Fall back to a resume file on disk
    path = getattr(career, "resume_path", None)
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    # 3. Fall back to cross-session long-term memory (may be empty on first run)
    recalled = load_user_memory(config)
    return recalled
