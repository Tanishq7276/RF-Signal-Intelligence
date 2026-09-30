# Docker

| File | Purpose |
| --- | --- |
| `Dockerfile` | two-stage build: Node 20 builds the SPA, Python 3.13-slim installs the DSP/API stack, then **one** image serves both from one origin |
| `entrypoint.sh` | creates the data directories, waits for PostgreSQL when `SIH_DATABASE_URL` points at one, refreshes `docs/index.json`, then execs the CMD |
| `../docker-compose.yml` | the runnable stack (API + UI on port 8000, named volume for `data/`, optional PostgreSQL profile) |

```bash
# from the repository root
docker compose up --build            # http://localhost:8000  (UI, /docs = Swagger, /redoc)
docker compose logs -f api           # job/stage progress is logged by the API process
docker compose down                  # keeps the data volume
docker compose down -v               # also removes stored analyses, uploads and reports
```

## Why a single container

The FastAPI process serves the REST API *and* the built front-end (`frontend/dist`). One origin means
no CORS configuration, no reverse proxy and no chance of the UI losing its styles after a restart -
the exact failure mode that motivated the single-image design. Scaling out is still possible: run
several containers behind a proxy with a shared `SIH_DATA_DIR` volume and PostgreSQL.

## PostgreSQL

```bash
cp .env.example .env       # uncomment and adjust the postgres lines
docker compose --profile postgres up --build
```

`psycopg[binary]` is part of `requirements.txt`, so the same image speaks to SQLite (default) and
PostgreSQL; only `SIH_DATABASE_URL` changes. The API waits for the database healthcheck before it
starts (see `entrypoint.sh`).

## Offline / air-gapped build

The image has no runtime downloads: the Python wheels are the only network access needed at build
time. A machine with a pip/npm cache can pre-pull both base images (`node:20-alpine`,
`python:3.13-slim`) and run the build offline afterwards.

## Resource notes

* DSP analysis is CPU-bound and single-process per job; `SIH_JOB_WORKERS` (default 2) sets the job
  pool, which is also the number of analyses that can run concurrently.
* Uploads are streamed to disk with a size cap (`SIH_MAX_UPLOAD_BYTES`, default 256 MiB in compose,
  512 MiB in code) and analyses are sample-capped (`SIH_MAX_SAMPLES_ANALYSE`), so the container
  degrades gracefully instead of being OOM-killed.
* `/api/health` reports the real subsystem state (database, storage directories, job runner, memory)
  and is used as the container healthcheck.
