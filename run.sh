#!/usr/bin/env bash
# One-command setup for Git Bash (Windows), macOS and Linux.
#   ./run.sh            -> setup + download real data + train + live test + demo scan
#   ./run.sh serve      -> web UI on http://127.0.0.1:8000
#   ./run.sh live       -> re-test the model on today's phishing feed
#   ./run.sh test       -> run the test suite
set -euo pipefail
cd "$(dirname "$0")"

# On Windows, "python3" can be a Microsoft Store stub that does nothing, so test each candidate.
PY=""
for cand in python python3 py; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys; assert sys.version_info >= (3, 9)" >/dev/null 2>&1; then
    PY="$cand"; break
  fi
done
[ -z "$PY" ] && { echo "Python 3.9+ is required: https://www.python.org/downloads/ (tick 'Add to PATH')"; exit 1; }

if [ ! -d .venv ]; then
  echo ">> creating virtual environment"
  "$PY" -m venv .venv
fi
# Windows venvs use Scripts/, Unix uses bin/
if [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi
export PYTHONIOENCODING=utf-8

if [ ! -f .venv/.installed ]; then
  echo ">> installing dependencies"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements-dev.txt
  touch .venv/.installed
fi

need_model() { [ -f models/sentinel_mlp.npz ] || python -m sentinel train; }

case "${1:-demo}" in
  serve) need_model; python -m sentinel serve ;;
  test)  python -m pytest -q ;;
  train) shift; python -m sentinel train "$@" ;;
  live)  need_model; python -m sentinel live ;;
  scan)  shift; need_model; python -m sentinel scan "$@" || true ;;
  demo)
    need_model
    python -m sentinel live || echo "(live feed unavailable right now, skipping)"
    python -m sentinel batch examples/urls.txt
    echo ""
    echo "README.md now contains your real results. Commit it!"
    echo "next:  ./run.sh serve   (web UI)   |   ./run.sh test   (tests)"
    ;;
  *) echo "usage: ./run.sh [demo|serve|live|test|train|scan <url>...]"; exit 1 ;;
esac
