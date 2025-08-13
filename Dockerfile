# 使用官方的 Python 3.12 slim 版本作为基础镜像
FROM python:3.12-slim

# 设置环境变量，防止 Python 生成 .pyc 文件和进行缓冲
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 从官方镜像中复制 uv 二进制文件，这是推荐的安装方式
# 锁定到特定版本以确保可复现的构建
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

# 设置容器内的工作目录
WORKDIR /app

# 复制构建和安装所需的所有文件
# 这必须包括项目源代码(src)，因为 uv sync 需要构建它
COPY pyproject.toml uv.lock* README.md* ./
COPY ./src ./src

# 使用 uv sync 安装依赖
# 现在它可以找到 src 目录并成功构建本地项目了
RUN uv sync --locked

# 使用 uv run 来执行 uvicorn 启动应用
# 这是在 uv 管理的环境中运行命令的标准方式
CMD ["uv", "run", "text2speech"]
