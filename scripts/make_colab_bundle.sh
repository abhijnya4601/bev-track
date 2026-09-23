#!/usr/bin/env bash
# Zip the repo (code + pinned BEVFormer submodule, no data/checkpoints) for upload to
# Google Drive at MyDrive/bevtrack/bev-track.zip, which notebooks/colab_gpu.ipynb unpacks.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/bev-track.zip
cd ..
zip -qr "$OLDPWD/dist/bev-track.zip" "$(basename "$OLDPWD")" \
  -x "*/.git/*" "*/.git" "*/.venv/*" "*/__pycache__/*" "*/.pytest_cache/*" "*/dist/*" \
     "*/data/nuscenes/*" "*/data/can_bus/*" "*/data/processed/*" "*/data/synthetic/*" \
     "*/ckpts/*" "*/work_dirs/*" "*.pth" "*.pkl" "*/.DS_Store"
cd "$OLDPWD"
# The notebook expects the top-level folder to be called bev-track
if [ "$(basename "$PWD")" != "bev-track" ]; then
  tmp=$(mktemp -d); unzip -q dist/bev-track.zip -d "$tmp"; mv "$tmp/$(basename "$PWD")" "$tmp/bev-track"
  (cd "$tmp" && zip -qr bev-track.zip bev-track) && mv "$tmp/bev-track.zip" dist/bev-track.zip && rm -rf "$tmp"
fi
ls -lh dist/bev-track.zip
