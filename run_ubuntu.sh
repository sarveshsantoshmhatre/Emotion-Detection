#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it with your distribution's package manager."
  echo "On Ubuntu, install: sudo apt install python3.11-venv python3-tk"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  python3.11 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-desktop.txt
exec python desktop_app.py
