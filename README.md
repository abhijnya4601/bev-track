# BEV-Track

Multi-camera BEV 3D detection (BEVFormer-tiny) and tracking (AB3DMOT) on nuScenes-mini, with an
evaluation harness that reports performance **by class, distance and scene condition** instead of
one aggregate number.

> **Status:** the eval harness, tracker, data prep, config and tests are built and verified
> (53 tests, including cross-checks against the official nuScenes devkit and motmetrics).
> Model training and inference haven't been run yet. The results sections below are empty until
> they are. No numbers in this repo come from a real model.

## Problem

Camera-only 3D perception has to infer depth from 2D images. BEV methods lift features from all six
cameras into one top-down grid around the car, the frame that planning works in. Detection quality
is very uneven: small objects, far objects and night scenes are much harder, and a single mAP hides
all of that. This project measures where it breaks.

## Approach

```
6 camera images ──► BEVFormer-tiny ───────────► per-frame 3D boxes ──► AB3DMOT ──► tracks
 (+ CAN bus)        R50 backbone (frozen)          (ego frame)           Kalman + Hungarian,
                    50×50 BEV grid, 3-frame                              run in the global frame
                    temporal self-attention
                    5-class head (fine-tuned)
                                 │                                          │
                                 ▼                                          ▼
                     slicing eval harness                         CLEAR-MOT: MOTA, MOTP,
                     mAP/NDS by class × distance × condition      ID switches, fragmentation
                     failure-case extraction + BEV plots
```

**Classes:** car · pedestrian · cyclist (bicycle+motorcycle) · truck (truck+construction) · barrier.

**Leakage-free split.** The public BEVFormer-tiny checkpoint was trained on the official nuScenes
train split, which contains 6 of the 8 `mini_train` scenes and every night scene in mini. So:
- the 5-class head is fine-tuned on those 6 already-seen scenes (no new information leaks), and
- the pretrained and fine-tuned models are both evaluated on the 4 mini scenes from the official
  **val** split (`clean`), so they're compared on the same held-out data.

`python data/prepare_nuscenes.py report` prints each scene's official split and conditions
([results/split_report.csv](results/split_report.csv)):

| split | scenes | samples | GT boxes (5 classes) | conditions |
|---|---|---|---|---|
| `seen` (fine-tune) | 0061 0655 0757 1077 1094 1100 | 242 | 9,648 | 3 day, 3 night (1 rain) |
| `clean` (evaluate) | 0103 0553 0796 0916 | 162 | 6,893 | **all day, all dry**; 2 Boston, 2 Singapore |

**Consequence:** every night and rain scene in nuScenes-mini is one the pretrained model was trained on,
so a day-vs-night comparison can't be measured without leakage on mini. The condition slice
reported on `clean` is **location** (Boston vs Singapore). Lighting and weather slicing is built and
tested, and it works as soon as the eval set has night or rain scenes (e.g. the full val split).

**Checkpoint surgery.** The 10-class classifier rows are merged into 5 (averaging
bicycle+motorcycle → cyclist and so on) rather than re-initialised, and the backbone is frozen.
Layer-wise learning rates are set in `configs/bevformer_tiny_nusc.py`.

**Why a custom metric implementation.** The devkit's `DetectionEval` hard-codes its 10 classes and
can't slice. `src/eval/metrics.py` reimplements its matching, AP and TP errors. The tests check it
against the devkit's own functions to 1e-9. Slices share one global matching, so a prediction at
19.5 m matched to a GT at 20.5 m counts in the 20–40 m bucket, rather than showing up as a FP in one
bucket and a FN in the other.

## Results

_To be filled from `results/` after running `scripts/gpu_pipeline.sh`. Every table reports `n_gt`
next to each number: with 4 eval scenes, many cells rest on a few dozen objects._

| | mAP | NDS | car | ped | cyclist | truck | barrier |
|---|---|---|---|---|---|---|---|
| Pretrained (10→5 class mapping) | | | | | | | |
| Fine-tuned 5-class head | | | | | | | |

By distance (`metrics_by_distance.csv`), by condition (`metrics_by_condition.csv`). Note that devkit
class ranges exclude pedestrians and cyclists beyond 40 m and barriers beyond 30 m.

## BEV visualizations

`notebooks/visualize_bev.ipynb`: frame grids, AP-by-distance small multiples, and a track GIF.

## Failure analysis

`python -m src.eval.failure_cases` pulls the worst false negatives (most lidar points yet missed),
false positives (highest score) and localization errors, tags each one
(`duplicate`, `class_confusion`, `mislocalized`, `hallucination`, `missed`) and renders it in BEV.

## Tracking

AB3DMOT with a 10-state constant-velocity Kalman filter, per-class Hungarian association on BEV
centre distance (IoU is harsh at 2 Hz for small objects), initial velocity taken from BEVFormer's
velocity head. MOTA/MOTP/IDSW/FRAG come from `src/eval/track_metrics.py`, which matches motmetrics
on randomized sequences.

## Running it

**CPU (eval, tracking, tests). Works on a laptop:**
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-eval.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.synthetic --out-dir data/synthetic            # smoke-test data, NOT results
.venv/bin/python -m src.eval.slice_eval --gt data/synthetic/gt.json --pred data/synthetic/preds.json --out /tmp/r
```

**GPU (model), free:** Google Colab's free T4 via [notebooks/colab_gpu.ipynb](notebooks/colab_gpu.ipynb).
1. `scripts/make_colab_bundle.sh` → upload `dist/bev-track.zip` to Google Drive at `MyDrive/bevtrack/`.
2. Get the CAN bus expansion from nuscenes.org (free account): upload `can_bus.zip` to the same folder,
   or paste its download link into the notebook.
3. Open the notebook in Colab, select *Runtime → T4 GPU*, run all. nuScenes-mini and the checkpoint download automatically.

**GPU (model), own machine / docker.** Needs an NVIDIA T4/V100/A10/A100/RTX 30xx. CUDA 11.1 predates Ada and Hopper cards.
1. Download `v1.0-mini.tgz` → `data/nuscenes/` and `can_bus.zip` → `data/can_bus/` (free nuscenes.org account).
2. `git submodule update --init` (BEVFormer, pinned).
3. `docker build -t bevtrack-gpu -f docker/Dockerfile .`
4. `docker run --gpus all --shm-size=8g -it -v $PWD:/workspace/bev-track bevtrack-gpu scripts/gpu_pipeline.sh`

## Repo layout

```
configs/bevformer_tiny_nusc.py   5-class fine-tune config (inherits upstream bevformer_tiny.py)
data/prepare_nuscenes.py         split report, GT files, 5-class BEVFormer info files
src/model.py                     classifier remap, prediction export
src/train.py                     fine-tune launcher (BEVFormer's own training loop)
src/track.py                     AB3DMOT
src/eval/metrics.py              AP / TP errors / NDS (devkit-equivalent, slice-aware)
src/eval/slice_eval.py           class × distance × condition tables
src/eval/failure_cases.py        worst-case extraction, diagnosis, BEV renders
src/eval/track_metrics.py        CLEAR-MOT
tests/                           tests of the eval code itself
NOTES.md                         build log: what broke and how it was fixed
```

## With more time

Full nuScenes (and full-val numbers comparable to the paper), all 10 classes, AMOTA via the devkit's
tracking eval, LiDAR fusion (e.g. BEVFusion), robustness to camera-extrinsic perturbation, and TensorRT export.
