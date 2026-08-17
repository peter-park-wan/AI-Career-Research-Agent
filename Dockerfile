# 基于 Python 3.11 的 slim 镜像构建 langgraph dev 演示环境
FROM python:3.11-slim

LABEL org.opencontainers.image.title="AI Career Research Agent (langgraph dev)" \
      org.opencontainers.image.description="Docker 化打包 langgraph dev, 跨平台一键启动含 Studio 界面的职业研究智能体" \
      org.opencontainers.image.licenses="MIT"

# 设置环境变量（可在 docker run 时覆盖）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # langgraph dev 默认端口
    LANGGRAPH_PORT=2024 \
    # 开启内存版 checkpointer / store（inmem 已含依赖）
    LANGGRAPH_CONFIG=

# 系统依赖：仅需构建 wheel 的编译工具
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖声明，利用层缓存
COPY pyproject.toml README.md ./
COPY src ./src

# 安装项目（主依赖已含 langgraph-cli[inmem]，api extras 提供 uvicorn/fastapi）
RUN pip install --upgrade pip \
    && pip install -e ".[api]"

# 数据目录：简历示例与知识库挂载点（构建时会 COPY 仓库内 data 示例）
COPY data ./data

# 暴露 langgraph dev 端口（Studio 通过此端口访问）
EXPOSE 2024

# 启动 langgraph dev。--allow-blocking 允许阻塞式工具调用（深度研究用）。
# 用 python -m langgraph 以兼容不同安装名；--host 0.0.0.0 让容器外可访问。
CMD ["sh", "-c", "langgraph dev --host 0.0.0.0 --port ${LANGGRAPH_PORT:-2024} --allow-blocking"]
