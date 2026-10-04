"""
Pytest configuration for AI_Career_Research_Agent tests.
"""

import sys
from pathlib import Path

import pytest

# ``src/legacy`` 是上游 open_deep_research 的遗留副本，包内部用的是
# ``from legacy.configuration import ...`` 这种**绝对导入**（上游假设
# ``legacy`` 是顶层包，靠 ``pip install -e .`` 后 package-dir 映射生效）。
#
# 但当前环境的 site-packages 里装了同名的第三方模块 ``legacy.py``，
# 于是 ``legacy`` 被解析成那个模块而不是本项目的包，报
# ``ModuleNotFoundError: 'legacy' is not a package`` —— 并且这个 ImportError
# 发生在收集阶段，会让**整个 pytest 中断**，一条测试都跑不了。
#
# 把 ``src/`` 放到 sys.path 最前面，让 ``legacy`` 稳定指向 src/legacy（真正的包）。
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption("--research-agent", action="store", help="Agent type: multi_agent or graph")
    parser.addoption("--search-api", action="store", help="Search API to use")
    parser.addoption("--eval-model", action="store", help="Model for evaluation")
    parser.addoption("--supervisor-model", action="store", help="Model for supervisor agent")
    parser.addoption("--researcher-model", action="store", help="Model for researcher agent")
    parser.addoption("--planner-provider", action="store", help="Provider for planner model")
    parser.addoption("--planner-model", action="store", help="Model for planning")
    parser.addoption("--writer-provider", action="store", help="Provider for writer model")
    parser.addoption("--writer-model", action="store", help="Model for writing")
    parser.addoption("--max-search-depth", action="store", help="Maximum search depth")