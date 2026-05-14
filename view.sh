#!/bin/bash
# Launch the simulation with the mjpython interpreter required for the
# MuJoCo 3D viewer on macOS (Apple Metal backend).
# Prefer the project's venv copy so no manual `source .venv/bin/activate` is
# needed before invoking this script.
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$DIR/.venv/bin/mjpython" ]; then
    exec "$DIR/.venv/bin/mjpython" "$DIR/main.py" "$@"
fi
exec mjpython "$DIR/main.py" "$@"
