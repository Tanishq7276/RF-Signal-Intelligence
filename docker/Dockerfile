# ---------------------------------------------------------------------------------------------
# SIH26147 RF Signal Intelligence Platform - single production image.
#
# Stage 1 builds the React/TypeScript SPA (Vite).  Stage 2 is the Python runtime that installs
# the DSP/API dependencies, copies the built front-end and serves both from one origin, so the
# browser needs no CORS configuration and no second container.
#
#   docker build -f docker/Dockerfile -t sih26147-rf-platform .
#   docker run --rm -p 8000:8000 -v sih-data:/app/data sih26147-rf-platform
# ---------------------------------------------------------------------------------------------

# ------------------------------------------------ stage 1: front-end -------------------------
FROM node:20-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/tsconfig.json frontend/vite.config.ts frontend/index.html ./
COPY frontend/src ./src
RUN npm run build          # -> /build/dist

# ------------------------------------------------ stage 2: API + DSP + UI --------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIH_DATA_DIR=/app/data \
    SIH_DOCS_DIR=/app/docs \
    SIH_FRONTEND_DIST=/app/frontend/dist \
    SIH_JOB_WORKERS=2

WORKDIR /app

# system libs needed by scipy/sklearn wheels are already bundled; curl is used by the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# application code (backend, dsp engine, ml model, tests, sample data, docs)
COPY backend ./backend
COPY dsp ./dsp
COPY ml ./ml
COPY tests ./tests
COPY sample_data ./sample_data
COPY docs ./docs
COPY docker ./docker
COPY README.md ./

# built front-end from stage 1
COPY --from=frontend /build/dist ./frontend/dist

# non-root runtime user; the data volume is owned by it so uploads/cache/reports are writable
RUN useradd --create-home --uid 10001 sih \
    && mkdir -p /app/data/uploads /app/data/cache /app/data/reports /app/data/generated \
    && chown -R sih:sih /app
USER sih

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
