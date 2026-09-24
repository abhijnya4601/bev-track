# BEV-Track

3D object detection from cameras (BEVFormer-tiny), from LiDAR (CenterPoint) and from a late fusion of
both, plus tracking (AB3DMOT), on nuScenes-mini, with an evaluation harness that reports performance
**by class, distance and scene condition** instead of one aggregate number, with bootstrap
confidence intervals.

> **[Interactive demo →](https://abhijnya4601.github.io/bev-track/)**: step through the held-out scenes in
> bird's-eye view, compare all five models (three camera, LiDAR, late fusion), and adjust the score threshold.
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
6 camera images + CAN bus ──► BEVFormer-tiny (R50, 50×50 BEV grid, 3-frame temporal) ──┐
LiDAR sweep + 9 past sweeps ──► CenterPoint (voxel 0.1 m, circle NMS) ───────────────────┤
                                                                                         ▼
                        late fusion: pair boxes per class, keep LiDAR geometry, noisy-OR scores
                                                                                         ▼
                              per-frame 3D boxes (ego frame) ──► AB3DMOT (Kalman + Hungarian, global frame)
                                         │                                   │
                                         ▼                                   ▼
                slicing eval: mAP/NDS by class × distance × condition   CLEAR-MOT: MOTA, MOTP,
                + paired bootstrap CIs + failure-case BEV renders        ID switches, fragmentation
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

All on the 4 `clean` scenes (162 samples, 4,667 GT boxes after devkit filtering). Both pretrained
checkpoints were trained on nuScenes' official train split, which excludes these scenes. NDS uses
mAAE = 1 because no attributes are predicted. The 95% CIs come from a paired frame-level bootstrap
(1000 resamples). Frames within a scene are correlated, so **the intervals are optimistic**.

| model | mAP (95% CI) | NDS | mATE | car | ped | cyclist | truck | barrier |
|---|---|---|---|---|---|---|---|---|
| Camera: BEVFormer-tiny, 10→5 relabeled | 0.452 (0.435–0.470) | 0.436 | 0.71 m | 0.453 | 0.441 | 0.223 | 0.352 | 0.793 |
| Camera: head-only fine-tune (4 ep) | 0.432 (0.415–0.449) | 0.413 | 0.76 m | 0.447 | 0.440 | 0.228 | 0.311 | 0.737 |
| Camera: fine-tune most layers (12 ep) | 0.388 (0.373–0.403) | 0.379 | 0.81 m | 0.426 | 0.376 | 0.193 | 0.366 | 0.579 |
| **LiDAR: CenterPoint, 10→5 relabeled** | 0.720 (0.695–0.745) | 0.663 | 0.30 m | **0.847** | **0.913** | 0.599 | 0.587 | 0.654 |
| **Late fusion: camera + LiDAR** | **0.753** (0.731–0.774) | **0.681** | **0.27 m** | 0.832 | 0.850 | **0.630** | **0.648** | **0.804** |

**Camera vs LiDAR.** LiDAR is +0.268 mAP ahead (paired 95% CI +0.248 to +0.287), and its
translation error is less than half (0.30 m vs 0.71 m): cameras have to infer depth, LiDAR measures it.
The gap is largest where camera depth is worst. Beyond 40 m, LiDAR mAP is 0.257 vs the camera's 0.044.
That bucket covers only cars and trucks at 40–50 m (269 objects), because the devkit's class ranges cut
pedestrians, cyclists and barriers off at 40 m or less.
From 0–20 m to 20–40 m, camera mAP keeps 62% of its value and LiDAR keeps 70%.

**Late fusion adds +0.033 mAP over LiDAR alone (CI +0.022 to +0.044), and all of the gain is at range.**
0–20 m: +0.001 (CI −0.011 to +0.011, no effect). 20–40 m: +0.039 (CI +0.017 to +0.060). Per class,
fusion helps where the camera is relatively strong or LiDAR is sparse: barrier 0.654 → 0.804, truck
0.587 → 0.648, cyclist 0.599 → 0.630. It **hurts** car (0.847 → 0.832) and pedestrian (0.913 → 0.850).
An ablation (`python -m src.fusion --ablation`, diagnosis only) shows why: the noisy-OR score boost for
boxes both sensors agree on drives **both** the gains and the losses. For barriers the camera's agreement
is informative (camera alone: 0.79 AP vs LiDAR 0.65). For pedestrians the camera also "agrees" with
some LiDAR false positives, lifting them above real detections. Duplicate camera boxes and unconfirmed
camera boxes barely matter (suppressing duplicates changed AP by ≤ 0.002). The fusion parameters were
fixed a priori. The principled fix is to learn per class how much camera agreement is worth, on the
training scenes, not these.

**Camera fine-tuning hurt.** Head-only −0.020 (CI −0.026 to −0.014); most layers −0.064 (CI −0.072 to
−0.056). The 5-class taxonomy is a pure merge of existing nuScenes classes, so relabeling the pretrained
model's outputs already solves the task at zero cost. Tuning the transformer on 242 samples from 6
scenes can only move the model toward those scenes. Barrier, whose test instances come from a single
Boston scene unlike the training barriers, lost the most (0.79 → 0.58). Freezing everything but the
head recovers 4.4 of the 6.4 points, so most of the damage came from updating the shared
representation. Conclusion: when a new taxonomy is a merge of existing classes, relabel the outputs.
Don't fine-tune on mini-scale data.

**By distance (pretrained):** mAP 0.591 (0–20 m) → 0.367 (20–40 m) → 0.044 (40 m+, cars and trucks only). Cyclists fall
fastest: 0.41 → 0.07. The fine-tuned model shows the same shape, shifted down.

**By condition:** the clean scenes are all day and dry, so lighting and weather can't be measured
without leakage on mini (see split above). By location, compare on `mAP*` (classes present in both
cities): per class, Boston and Singapore are within noise of each other (car 0.47 vs 0.45,
ped 0.43 vs 0.47, truck 0.37 vs 0.36). The raw mAP gap (0.454 vs 0.385) is because Singapore has
no barrier GT after filtering (its 4 raw barriers are all 33–54 m away, beyond the devkit's
30 m barrier range).

**Failure cases:** `results/failure_examples/{pretrained,fused}/`. The worst misses, false positives and
localization errors of the camera baseline and the fused model, each tagged (duplicate, class confusion,
mislocalized, hallucination) and rendered in BEV.

**Tracking (AB3DMOT, score ≥ 0.3, 2 m CLEAR-MOT):**

| detections from | MOTA | ID switches |
|---|---|---|
| Camera, relabeled | 0.195 | 536 |
| Camera, head-only FT | 0.183 | 543 |
| Camera, full FT | 0.210 | 448 |
| **LiDAR** | **0.567** | **244** |
| Late fusion | 0.321 | 398 |

LiDAR tracks far better: MOTA 0.57 vs 0.20, with half the ID switches, because accurate positions keep
the Kalman filter's associations stable. **Late fusion has higher mAP than LiDAR but much lower MOTA.**
Noisy-OR raises scores, and down-weighted camera-only boxes still clear the tracker's 0.3 threshold,
so more false positives reach the tracker. mAP integrates over all thresholds and doesn't see this.
Tracking uses one. The camera models show the same effect (the best detector has the lowest MOTA).
The lesson is that a detector's scores need calibrating for the operating threshold its consumer
uses. That's why nuScenes' official tracking metric (AMOTA) averages over thresholds.

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
centre distance (IoU is harsh at 2 Hz for small objects). A new track's initial velocity comes from
the detector's own velocity estimate (BEVFormer's or CenterPoint's; fused boxes carry CenterPoint's). MOTA/MOTP/IDSW/FRAG come from `src/eval/track_metrics.py`, which matches motmetrics
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
configs/bevformer_tiny_nusc.py           5-class fine-tune config (inherits upstream bevformer_tiny.py)
configs/bevformer_tiny_nusc_headonly.py  head-only ablation (everything but cls/reg branches frozen)
data/prepare_nuscenes.py                 split report, GT files, 5-class BEVFormer info files
src/classes.py                           5-class taxonomy and nuScenes label mappings
src/boxes.py                             box type, frame transforms, BEV geometry
src/dataio.py                            GT/prediction file I/O, devkit-equivalent filtering
src/model.py                             classifier checkpoint remap, prediction export (BEVFormer + CenterPoint)
src/train.py                             fine-tune launcher (BEVFormer's own training loop)
src/fusion.py                            late fusion of camera + LiDAR boxes
src/track.py                             AB3DMOT
src/synthetic.py                         synthetic scenes/detector for tests and smoke runs
src/eval/metrics.py                      AP / TP errors / NDS (devkit-equivalent, slice-aware)
src/eval/slice_eval.py                   class × distance × condition tables
src/eval/bootstrap.py                    paired bootstrap confidence intervals
src/eval/failure_cases.py                worst-case extraction, diagnosis, BEV renders
src/eval/track_metrics.py                CLEAR-MOT
scripts/gpu_pipeline.sh                  GPU run, steps 0-10 (data prep → models → fusion → CIs → demo data)
scripts/build_demo.py                    builds docs/demo_data.json for the demo page
docs/                                    interactive demo (GitHub Pages)
notebooks/colab_gpu.ipynb                free-GPU (Colab T4) runner
tests/                                   tests of the eval code, tracker, fusion and bootstrap
NOTES.md                                 build log: what broke, what I decided, and why
```

## With more time

Full nuScenes (and full-val numbers comparable to the papers), all 10 classes, per-class fusion
weights tuned on held-out training scenes, score calibration before tracking, AMOTA via the devkit's
tracking eval, a learned fusion model (e.g. BEVFusion), robustness to camera-extrinsic perturbation,
and TensorRT export.
