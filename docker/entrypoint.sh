#!/bin/sh
# Container entrypoint: make sure the writable directories exist, optionally wait for a database
# and then hand over to the CMD (uvicorn by default).  Nothing here is required for the app to
# work - it only removes two classic container problems: a missing data volume and a database that
# is not ready yet.
set -e

DATA_DIR="${SIH_DATA_DIR:-/app/data}"
mkdir -p "$DATA_DIR/uploads" "$DATA_DIR/cache" "$DATA_DIR/reports" "$DATA_DIR/generated" 2>/dev/null || true

# --- wait for PostgreSQL when the database URL points at one ----------------------------------
case "${SIH_DATABASE_URL:-}" in
  postgres*|postgresql*)
    host="$(printf '%s' "$SIH_DATABASE_URL" | sed -n 's|.*@\([^:/]*\).*|\1|p')"
    port="$(printf '%s' "$SIH_DATABASE_URL" | sed -n 's|.*@[^:]*:\([0-9]*\).*|\1|p')"
    host="${host:-db}"; port="${port:-5432}"
    echo "[entrypoint] waiting for database ${host}:${port}"
    i=0
    while [ "$i" -lt 60 ]; do
      if python - "$host" "$port" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(1.5)
try:
    s.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
      then
        echo "[entrypoint] database reachable"
        break
      fi
      i=$((i + 1))
      sleep 2
    done
    ;;
esac

# --- generate the documentation index if it is missing ----------------------------------------
python - <<'PY' 2>/dev/null || true
from backend.app.routes.docs import sync_index
sync_index()
PY

echo "[entrypoint] starting: $*"
exec "$@"
