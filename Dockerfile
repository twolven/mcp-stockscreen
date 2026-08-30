# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 AS uv
FROM python:3.13-slim-bookworm@sha256:c45a22ea000adfd9cda29364bbe7edd23001ce5cc2ad15857cfbf7766943b9ca

COPY --from=uv /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    MCP_PATH=/mcp

WORKDIR /app
RUN groupadd --system --gid 10001 mcp \
    && useradd --system --uid 10001 --gid mcp --create-home --home-dir /home/mcp mcp

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY stockscreen.py ./
RUN uv sync --frozen --no-dev --no-editable \
    && mkdir -p /home/mcp/.stockscreen \
    && chown -R mcp:mcp /app /home/mcp

USER 10001:10001
EXPOSE 8000
VOLUME ["/home/mcp/.stockscreen"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"
ENTRYPOINT ["python", "stockscreen.py"]
