#!/bin/bash
cd "$(dirname "$0")" || exit 1

VENV_DIR="psdenv"
PY="$VENV_DIR/bin/python"

if [ ! -x "$PY" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || { echo "Setup failed"; exit 1; }
fi

# Install requirements when requirements.txt changed since the last install
if ! cmp -s requirements.txt "$VENV_DIR/requirements.installed"; then
    echo "Installing requirements..."
    "$PY" -m pip install -r requirements.txt || { echo "Setup failed"; exit 1; }
    cp requirements.txt "$VENV_DIR/requirements.installed"
fi

"$PY" main.py
