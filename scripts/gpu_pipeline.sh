#!/usr/bin/env bash
# End-to-end BEV-Track run on a CUDA machine: inside the docker/Dockerfile image, or the Colab env from notebooks/colab_gpu.ipynb.
# Run all steps, or only some:  scripts/gpu_pipeline.sh 5 6 7
# Optional env: TRAIN_CFG_OPTIONS="data.workers_per_gpu=2"  (extra mmcv overrides for step 6)
#
# Prerequisites (manual, needs a nuscenes.org account):
#   data/nuscenes/   <- extracted v1.0-mini.tgz  (maps/, samples/, sweeps/, v1.0-mini/)
#   data/can_bus/    <- extracted can_bus.zip
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
BF=$REPO/third_party/BEVFormer
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# Preflight: detectron2 0.6 (imported by BEVFormer's plugin) needs PIL.Image.LINEAR, removed in Pillow 10.
# Anything that reinstalls Pillow (a fresh env, an older copy of the Colab notebook) silently breaks steps 5-7.
if ! python -c "from PIL import Image; Image.LINEAR" 2>/dev/null; then
  echo ">>> Pillow >= 10 found; installing pillow==9.5.0 (detectron2 0.6 needs Image.LINEAR)"
  python -m pip install -q "pillow==9.5.0"
fi

STEPS=" ${*:-0 1 2 3 4 5 6 7 8 9 10} "
step() { echo; echo "=== $* ==="; }
want() { [[ "$STEPS" == *" $1 "* ]]; }

if want 0; then
  step "0. GPU sanity check"
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
fi
if want 1; then
  step "1. BEVFormer sees the data via symlinks"
  mkdir -p "$BF/data"
  ln -sfn "$REPO/data/nuscenes" "$BF/data/nuscenes"
  ln -sfn "$REPO/data/can_bus" "$BF/data/can_bus"
fi
run_step2() {
  step "2. BEVFormer info files (temporal infos + CAN bus)"
  mkdir -p "$BF/data"
  ln -sfn "$REPO/data/nuscenes" "$BF/data/nuscenes"
  ln -sfn "$REPO/data/can_bus" "$BF/data/can_bus"
  [ -d "$REPO/data/nuscenes/v1.0-mini" ] || { echo "ERROR: data/nuscenes/v1.0-mini missing (run the data cell)"; exit 1; }
  [ -d "$REPO/data/can_bus" ] || { echo "ERROR: data/can_bus missing (run the data cell; check can_bus.zip)"; exit 1; }
  # BEVFormer's converters import `tools.data_converter...`. detectron2 0.6 installs its own top-level
  # `tools` package into site-packages, and a regular package beats BEVFormer's __init__-less `tools/`
  # (a namespace package) wherever it sits on sys.path. An empty __init__.py makes BEVFormer's win.
  touch "$BF/tools/__init__.py"
  (cd "$BF" && PYTHONPATH="$BF:$PYTHONPATH" python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes \
      --extra-tag nuscenes --version v1.0-mini --canbus ./data)
  ls -la data/nuscenes/*.pkl
}
run_step3() {
  step "3. Split report, GT files, 5-class info files"
  python data/prepare_nuscenes.py report
  python data/prepare_nuscenes.py gt
  python data/prepare_nuscenes.py remap-infos --infos-dir data/nuscenes
}
# Steps 5-7 need the 5-class info files; build them (and show any error) if they are missing.
require_infos() {
  if [ ! -f data/nuscenes/nuscenes_infos_temporal_clean_5cls.pkl ] || [ ! -f data/nuscenes/nuscenes_infos_temporal_seen_5cls.pkl ]; then
    echo ">>> 5-class info files missing; running steps 2 and 3 first"
    [ -f data/nuscenes/nuscenes_infos_temporal_train.pkl ] || run_step2
    run_step3
  fi
  mkdir -p "$BF/data"
  ln -sfn "$REPO/data/nuscenes" "$BF/data/nuscenes"
  ln -sfn "$REPO/data/can_bus" "$BF/data/can_bus"
}

if want 2; then run_step2; fi
if want 3; then run_step3; fi
if want 4; then
  step "4. Pretrained checkpoint"
  mkdir -p ckpts
  [ -f ckpts/bevformer_tiny_epoch_24.pth ] || wget -q -O ckpts/bevformer_tiny_epoch_24.pth \
      https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
fi
if want 5; then
  require_infos
  step "5. Baseline: unmodified 10-class checkpoint on the clean scenes, mapped to 5 classes"
  python -m src.model export --config "$BF/projects/configs/bevformer/bevformer_tiny.py" \
      --checkpoint ckpts/bevformer_tiny_epoch_24.pth \
      --ann-file data/nuscenes/nuscenes_infos_temporal_clean_5cls.pkl \
      --out results/preds_pretrained.json
  python -m src.eval.slice_eval --gt data/processed/gt_clean.json --pred results/preds_pretrained.json \
      --out results/pretrained
fi
if want 6; then
  require_infos
  step "6. Fine-tune the 5-class head on the 'seen' scenes"
  python -m src.train --gpus 1 ${TRAIN_CFG_OPTIONS:+--cfg-options $TRAIN_CFG_OPTIONS}
fi
if want 7; then
  require_infos
  step "7. Fine-tuned model on the same clean scenes"
  python -m src.model export --config configs/bevformer_tiny_nusc.py \
      --checkpoint work_dirs/bevformer_tiny_5cls/latest.pth --out results/preds_finetuned.json
  python -m src.eval.slice_eval --gt data/processed/gt_clean.json --pred results/preds_finetuned.json \
      --out results/finetuned
  python -m src.eval.failure_cases --gt data/processed/gt_clean.json --pred results/preds_finetuned.json \
      --out results/failure_examples --min-score 0.2
fi
if want 8; then
  step "8. Tracking"
  python -m src.track --gt data/processed/gt_clean.json --pred results/preds_finetuned.json \
      --out results/tracks.json --metrics-out results/tracking_metrics.csv
fi
if want 9; then
  require_infos
  step "9. LiDAR baseline: CenterPoint (pretrained, 10 classes mapped to 5) on the clean scenes"
  # CenterPoint ships with mmdet3d 0.17.1 (already installed); only its config files are needed.
  M3D="$REPO/.cache/mmdetection3d"
  [ -d "$M3D/configs" ] || git clone -q --depth 1 --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git "$M3D"
  CP=ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.pth
  mkdir -p ckpts
  [ -f "$CP" ] || wget -q -O "$CP" https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth
  python -m src.model export --config "$M3D/configs/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py" \
      --checkpoint "$CP" --ann-file data/nuscenes/nuscenes_infos_temporal_clean_5cls.pkl --out results/preds_lidar.json
  python -m src.eval.slice_eval --gt data/processed/gt_clean.json --pred results/preds_lidar.json --out results/lidar
fi
if want 10; then
  step "10. Late fusion, tracking, bootstrap intervals, demo data (CPU)"
  GT=data/processed/gt_clean.json
  python -m src.fusion --gt $GT --camera results/preds_pretrained.json --lidar results/preds_lidar.json --out results/preds_fused.json
  python -m src.eval.slice_eval --gt $GT --pred results/preds_fused.json --out results/fused
  for m in lidar fused; do
    echo "--- tracking on $m detections ---"
    python -m src.track --gt $GT --pred results/preds_$m.json --out results/tracks_$m.json \
        --metrics-out results/tracking_$m.csv | grep -E "MOTA|^ALL"
  done
  PREDS=""
  for m in pretrained headonly finetuned lidar fused; do
    [ -f results/preds_$m.json ] && PREDS="$PREDS --pred $m=results/preds_$m.json"
  done
  python -m src.eval.bootstrap --gt $GT $PREDS -n 1000 --out results/bootstrap.json
  python scripts/build_demo.py --gt $GT $PREDS --bootstrap results/bootstrap.json --out results/demo_data.json
fi
echo; echo "Done. Tables in results/pretrained, results/finetuned; failure plots in results/failure_examples."
