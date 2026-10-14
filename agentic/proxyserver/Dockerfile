# -------------------------------------------------------------------
# Dockerfile for the standalone LLM Proxy Server (relay mode)
#
# A lightweight, framework-agnostic relay service that bridges
# OpenAI-compatible HTTP requests to backend inference workers via
# WebSocket.  No GPU, no PyTorch, no Ray — just FastAPI + Uvicorn.
#
# Build (from project root):
#   docker build -f recipe/agentic/proxyserver/Dockerfile \
#       -t llm-proxy-server:latest .
#
# Run:
#   docker run -p 8080:8080 llm-proxy-server:latest
#
# The proxy exposes:
#   HTTP  :8080  — OpenAI-compatible /v1/chat/completions + session mgmt
#   WS    :8080  — /ws/worker for framework-side inference workers
#   HTTP  :8080  — /health for liveness/readiness probes
# -------------------------------------------------------------------

FROM python:3.12-slim

LABEL maintainer="llm-proxy"
LABEL description="Standalone LLM Proxy Server — framework-agnostic inference relay"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# -------------------------------------------------------------------
# Install only the minimal dependencies for relay mode
# -------------------------------------------------------------------
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.30.0 \
    pydantic>=2.0.0 \
    websockets>=12.0

# -------------------------------------------------------------------
# Copy only the generic proxy files (no framework-specific code)
# -------------------------------------------------------------------
COPY recipe/agentic/proxyserver/__init__.py  /app/recipe/agentic/proxyserver/
COPY recipe/agentic/proxyserver/server.py    /app/recipe/agentic/proxyserver/
COPY recipe/agentic/proxyserver/relay.py     /app/recipe/agentic/proxyserver/
COPY recipe/agentic/proxyserver/recorder.py  /app/recipe/agentic/proxyserver/
COPY recipe/agentic/proxyserver/models.py    /app/recipe/agentic/proxyserver/
COPY recipe/agentic/proxyserver/proxy_server.py /app/recipe/agentic/proxyserver/

# Ensure parent packages are importable
RUN touch /app/recipe/__init__.py \
          /app/recipe/agentic/__init__.py

# -------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------
ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

RUN groupadd -r proxyuser || true && useradd -r -g proxyuser proxyuser
USER proxyuser

ENTRYPOINT ["python", "-m", "recipe.agentic.proxyserver.proxy_server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
