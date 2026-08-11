FROM python:3.11-slim

WORKDIR /app

# RAGTune runs as a finite governance job: read configs, write /outputs,
# emit promotion_decision.json, validate claim boundaries, then exit.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RAGTUNE_CONTAINER=1
ENV RAGTUNE_OUTPUT_ROOT=/outputs

COPY pyproject.toml requirements.txt README.md LICENSE CITATION.cff Makefile ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs
COPY data ./data
COPY results ./results
COPY artifacts ./artifacts
COPY schemas ./schemas
COPY paper ./paper
COPY tests ./tests

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . \
    && useradd -m ragtune \
    && mkdir -p /outputs /inputs /configs /data \
    && chown -R ragtune:ragtune /outputs /inputs /configs /data /app

USER ragtune

HEALTHCHECK --interval=5m --timeout=30s --start-period=30s --retries=1 CMD ["ragtune", "inspect-environment"]

ENTRYPOINT ["ragtune"]
CMD ["--help"]
