FROM python:3.11-slim

LABEL org.opencontainers.image.title="RAGTune Governance" \
      org.opencontainers.image.description="Finite publication-safe RAG policy governance job" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# RAGTune runs as a finite governance job: read configs, write /outputs,
# emit promotion_decision.json, validate claim boundaries, then exit.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV RAGTUNE_CONTAINER=1
ENV RAGTUNE_OUTPUT_ROOT=/outputs

COPY pyproject.toml requirements.txt README.md LICENSE CITATION.cff Makefile .gitattributes Dockerfile .dockerignore docker-compose.yml ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs
COPY data ./data
COPY results ./results
COPY artifacts ./artifacts
COPY docker ./docker
COPY deploy ./deploy
COPY deployment_review ./deployment_review
COPY schemas ./schemas
COPY paper ./paper
COPY tests ./tests

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . \
    && groupadd --system --gid 10001 ragtune \
    && useradd --system --uid 10001 --gid ragtune --create-home --home-dir /home/ragtune --shell /usr/sbin/nologin ragtune \
    && mkdir -p /outputs /inputs /configs /data \
    && chown -R ragtune:ragtune /outputs /inputs /configs /data /app

USER ragtune

HEALTHCHECK --interval=5m --timeout=30s --start-period=30s --retries=1 CMD ["ragtune", "inspect-environment"]
STOPSIGNAL SIGTERM

ENTRYPOINT ["ragtune"]
CMD ["--help"]
