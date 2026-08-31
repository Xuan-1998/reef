#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/../../../.." && pwd)
UV_PROJECT_ENVIRONMENT="$ROOT/.venv-gepa"
export UV_PROJECT_ENVIRONMENT

exec uv run \
  --project "$HERE" \
  --extra reference \
  --with-editable "$ROOT" \
  python "$HERE/run.py" "$@"
