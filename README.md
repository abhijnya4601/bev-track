# BEV-Track

Multi-camera BEV 3D detection (BEVFormer-tiny) and tracking (AB3DMOT) on nuScenes-mini, with an
evaluation harness that reports performance **by class, distance and scene condition** instead of
one aggregate number.

> **[Interactive demo →](https://abhijnya4601.github.io/bev-track/)**: step through the held-out scenes in
> bird's-eye view, compare the three models, and adjust the score threshold.
>
> **Status:** end-to-end pipeline has run on a free Colab T4 (2026-09-24). Detection results below are
> real, on the 4 held-out scenes. With 162 samples and 4,667 GT objects, treat differences of a few
> mAP points as noise.

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

All on the 4 `clean` scenes (162 samples, 4,667 GT boxes after devkit filtering). NDS uses mAAE = 1
because no attributes are predicted, so it is lower than a devkit NDS would be.

| model | mAP | NDS | car | ped | cyclist | truck | barrier |
|---|---|---|---|---|---|---|---|
| **Pretrained, 10 classes relabeled to 5 (no training)** | **0.452** | **0.436** | 0.453 | 0.441 | 0.223 | 0.352 | 0.793 |
| Fine-tuned (neck/encoder 0.1×, transformer 0.5×, heads 1×; 12 ep) | 0.388 | 0.379 | 0.426 | 0.376 | 0.193 | 0.366 | 0.579 |
| Fine-tuned, head only (4 ep) | 0.432 | 0.413 | 0.447 | 0.440 | 0.228 | 0.311 | 0.737 |

**Fine-tuning hurt (−6.4 mAP).** The 5-class taxonomy is a pure merge of existing nuScenes classes,
so relabeling the pretrained model's outputs already solves the task at zero cost. Tuning the
transformer on 242 samples from 6 scenes can then only move the model toward those scenes, and
barrier, whose test instances come from a single Boston scene unlike the training barriers, lost
the most (0.79 → 0.58). **Head-only fine-tuning recovers 4.4 of the 6.4 points**, so most of the damage
came from updating the shared representation. Car, pedestrian and cyclist are unchanged vs the
baseline (±0.01), but the head alone still loses 4–6 points on truck and barrier, the two classes
whose training instances look least like the test ones. Conclusion: when a new taxonomy is a merge of
existing classes, relabel the outputs; don't fine-tune on mini-scale data.

**By distance (pretrained):** mAP 0.591 (0–20 m) → 0.367 (20–40 m) → 0.044 (40 m+). Cyclists fall
fastest: 0.41 → 0.07. The fine-tuned model shows the same shape, shifted down.

**By condition:** the clean scenes are all day and dry, so lighting and weather can't be measured
without leakage on mini (see split above). By location, compare on `mAP*` (classes present in both
cities): per class, Boston and Singapore are within noise of each other (car 0.47 vs 0.45,
ped 0.43 vs 0.47, truck 0.37 vs 0.36). The raw mAP gap (0.454 vs 0.385) is because Singapore has
no barrier GT.

**Failure cases (fine-tuned, 2 m matching):** of 3,503 FPs, 1,650 are hallucinations, 1,049
mislocalized (a same-class GT within 2–4 m), 670 duplicates and 134 class confusions. Of 1,344 FNs,
763 are mislocalized rather than missed outright. Most errors are localization, not detection:
2,066 of the worst-matched boxes are dominated by translation error. Renders are in
`results/failure_examples/`.

**Tracking (AB3DMOT, score ≥ 0.3, 2 m CLEAR-MOT):**

| detections from | MOTA | MOTP (m) | ID switches | fragmentations |
|---|---|---|---|---|
| Pretrained, relabeled | 0.195 | 0.83 | 536 | 293 |
| Fine-tuned, head only | 0.183 | 0.85 | 543 | 295 |
| Fine-tuned, most layers | 0.210 | 0.86 | 448 | 327 |

Tracking ranks the models **opposite** to detection. The best detector gets the lowest MOTA because
at a fixed 0.3 threshold it passes more boxes per frame to the tracker (35 vs 31), so more false
positives reach it. MOTA at a single threshold rewards score calibration as much as detection
quality, which is why nuScenes' official metric (AMOTA) averages over thresholds. Static barriers
track well (MOTA 0.80–0.91). Cars and pedestrians switch IDs mostly in the two dense parking-lot
scenes (0103, 0916).

## BEV visualizations

The [interactive demo](https://abhijnya4601.github.io/bev-track/) (`docs/`, built by
`scripts/build_demo.py`) shows every held-out frame. `notebooks/visualize_bev.ipynb` has static
frame grids and a track GIF.

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
