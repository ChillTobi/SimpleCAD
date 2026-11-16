#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ ! -f ".venv/bin/python" ]; then
    echo "[error] .venv not found. Run: python build.py --venv --platform linux"
    exit 1
fi
source .venv/bin/activate
python SimpleCAD.py
