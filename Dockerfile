# AI Career Research Agent — FastAPI 服务镜像
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# 先复制依赖清单，利用层缓存加速构建
COPY pyproject.toml uv.lock* ./
COPY src ./src
COPY README.md ./README.md
COPY LICENSE ./LICENSE
COPY data ./data

# 安装项目（可编辑安装，使 open_deep_research 包可被导入）
RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 8000

# 默认启动 FastAPI 服务
CMD ["uvicorn", "open_deep_research.api:app", "--host", "0.0.0.0", "--port", "8000"]
