#!/usr/bin/env bash
# Pack the code (no venv / data / runs / media) for upload to a Colab runtime as /content/RobotPluggingSim.zip
set -e
cd "$(dirname "$0")/../.."
zip -qr RobotPluggingSim.zip RobotPluggingSim -x "RobotPluggingSim/.venv/*" "RobotPluggingSim/data/*" "RobotPluggingSim/runs/*" \
    "RobotPluggingSim/media/*" "RobotPluggingSim/plugsim/_assets/*" "*/__pycache__/*" "*.ipynb_checkpoints*"
ls -lh RobotPluggingSim.zip
