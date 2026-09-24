# Build log: what broke, what I decided, and why

Newest entries at the bottom.

## Setup: two environments

My laptop has no NVIDIA GPU (Apple M3 Pro), and BEVFormer depends on mmcv-full 1.4.0's CUDA
kernels for deformable attention, so it can't run locally at all. I split the project:
- **CPU environment** (`requirements-eval.txt`): evaluation harness, tracker, tests, notebooks.
  Everything I want to iterate on quickly runs on the laptop.
- **GPU environment**: CUDA 11.1 / torch 1.9.1 / mmcv-full 1.4.0 / mmdet 2.14.0 / mmdet3d 0.17.1,
  pinned exactly as BEVFormer's `docs/install.md` specifies (`docker/Dockerfile`). CUDA 11.1 only goes
  up to sm_86, so it needs a T4/V100/A10/A100, not an L4/4090/H100.

## Data leakage: the pretrained model has seen most of nuScenes-mini

The public BEVFormer-tiny checkpoint was trained on nuScenes' official train split. Checking
`nuscenes.utils.splits`, 6 of the 8 mini_train scenes are in it, including all three night scenes
(1077, 1094, 1100). Only 0103, 0553, 0796 and 0916 come from the official val split. Evaluating on
mini_train would measure memorisation.

So I defined two splits in `data/prepare_nuscenes.py`: `seen` (fine-tune only here) and `clean`
(evaluate only here). Every model is evaluated on the same 4 clean scenes. Once the data was
downloaded, the split report showed all 4 are daytime and dry. That means I **can't** measure
lighting or weather without leakage on mini, so the condition slice I report is location
(Boston 0103/0553 vs Singapore 0796/0916).

## Why I wrote my own metric code, and how I made sure it's right

The devkit's `DetectionEval` hard-codes its 10 classes and can only score a whole split. I needed 5
merged classes and per-slice scores, so I reimplemented matching, AP and the TP errors in
`src/eval/metrics.py`. A reimplementation is only worth anything if it's verified, so:
- Tests compare my AP and TP errors against the devkit's own `accumulate`/`calc_ap`/`calc_tp` on
  random data (agreement to 1e-9), plus hand-calculated cases.
- The first cross-check failed (0.337 vs 0.322). The cause turned out to be score **ties**: the devkit
  breaks ties by position in its box list, and my test inserted samples in a different order. The
  metric was right and the test was wrong, but it showed me AP depends on tie-breaking when scores are
  quantized.
- On real data, my GT loading and filtering keep exactly the boxes the devkit keeps (mini_val: car
  1913, ped 1067, cyclist 250, truck 95).

**Distance slicing.** If you filter GT and predictions into distance buckets independently, a
prediction at 19.5 m matched to an object at 20.5 m becomes a false positive in one bucket and a miss
in the other. I match once over all the data and let a matched prediction follow its object's bucket.
There's a test for that exact case.

## Tracking metrics disagreed with motmetrics, twice

I implemented CLEAR-MOT myself so every counting rule is visible, and cross-checked it against
motmetrics on random sequences. ID switches disagreed (18 vs 14): I only carried over the previous
frame's matches, but motmetrics re-establishes each object's *last-ever* match, so after a one-frame
miss my version could hand the object to a different track and count a false switch. After fixing
that, fragmentations were off by one (29 vs 28), because the order in which two objects reclaim a
shared track matters, and motmetrics goes in frame order. Both are fixed and the cross-check passes.

## Config and framework gotchas

- mmcv config inheritance merges dicts but **replaces** lists, and variables like `class_names` are
  already baked into the parent's dicts. Overriding `class_names` alone changes nothing, so I restated
  every pipeline that mentions classes and checked the merged config with `mmcv.Config`.
- BEVFormer's dataset `evaluate()` looks up nuScenes default attributes by class name, so it can't
  handle my class names. I disabled in-training validation and evaluate with my own exporter + harness.
- Changing `bev_h/bev_w` or `point_cloud_range` would break the checkpoint: `bev_embedding` has
  bev_h·bev_w rows, and reference points are normalised by pc_range. I kept the upstream values.
- Library breakage: motmetrics 1.4 fails on string IDs under pandas 3, so I pinned `pandas<3`.

## Getting it to run on a free GPU (Colab T4)

nuScenes-mini downloads without a login, but the CAN bus expansion needs one. Colab ships Python 3.12
and CUDA 12, far newer than BEVFormer supports, so the notebook builds a Python 3.8 environment
with micromamba:
- torch 1.9.1+cu111 bundles its own CUDA runtime and runs on Colab's newer driver.
- mmdet3d's CUDA ops are compiled with nvcc 11.8. torch 1.9 only rejects a *major* CUDA version
  mismatch, and gcc-9 is used because gcc 11 is too new for torch 1.9's headers. The compiled wheel
  is cached in Drive, so later sessions skip the ~20-minute build.
- `yapf==0.40.1` pinned (newer yapf breaks mmcv 1.4's config printing), and `numpy==1.19.5` enforced
  with a constraints file on every install.

Three failures on the first run, each teaching something:
1. **Pillow 10 removed `Image.LINEAR`**, which detectron2 0.6 uses at import time. BEVFormer's plugin
   imports detectron2 (for DD3D, which BEVFormer-tiny never uses), so a dependency I don't need still
   has to import cleanly. I pinned `pillow==9.5.0`. It came back after a session restart, so the
   pipeline script now checks for it and repairs it itself. Fixing it in one place wasn't enough.
2. **`No module named 'tools.data_converter'`.** My first fix, putting BEVFormer's root on
   PYTHONPATH, changed nothing. The real cause: detectron2 installs its own top-level `tools`
   package, and BEVFormer's `tools/` has no `__init__.py`. Python picks a regular package over an
   `__init__`-less (namespace) one no matter the path order. An empty `__init__.py` fixed it. Lesson: I
   treated the symptom without first checking *which* `tools` was being imported.
3. After that, steps 5–7 build any missing inputs themselves, so an upstream failure shows its real
   error instead of a confusing "file not found" later.

## Result 1: fine-tuning the camera model made it worse

Relabeling the pretrained 10-class outputs to my 5 classes: mAP 0.452. Fine-tuning most layers:
0.388. My reasoning afterwards:
1. My 5 classes are merges of existing nuScenes classes, so relabeling at inference already solves
   the task completely. Training had nothing to add and plenty to break.
2. My first config updated far more than the classification head (encoder at 0.1×, the rest of the
   transformer at 0.5×) using only 242 training frames. That's a lot of capacity to pull toward 6 scenes.

To separate the two, I ran a head-only fine-tune (everything frozen except the class/box branches):
0.432. That recovers 4.4 of the 6.4 points, so most of the damage came from updating the shared
layers. It's still below plain relabeling, and the bootstrap interval (−0.026 to −0.014) says that
gap is real, not noise.

I also caught a comparison bug in my own tables: per-slice mAP averaged over whichever classes had
ground truth in that slice. Singapore's 4 barriers are all beyond the 30 m barrier range, so the
filter removes them, and Singapore looked 7 points worse than Boston for that reason alone. Per class,
the two cities are the same. I added `mAP*`, the mean over classes present in every slice of an axis.

Training cost: 12 epochs × 242 steps at ~2 s/step ≈ 1 h 40 min on a free T4, with ~0.7 s of each step
spent waiting on image decoding (2 CPU cores, 18 JPEGs per step).

## Result 2: LiDAR, late fusion, and confidence intervals

- **LiDAR baseline: CenterPoint** from the mmdet3d 0.17.1 model zoo (56.2 mAP / 64.4 NDS on full val).
  I chose it because it runs on the exact mmdet3d build I'd already compiled, so there was no new
  environment. It was trained on the same official train split, so my clean scenes stay held out, and
  the info files already had LiDAR paths and sweeps. It ran on the first try.
- **Late fusion** (`src/fusion.py`): pair LiDAR and camera boxes per class within a radius, keep the
  LiDAR geometry, combine scores by noisy-OR, down-weight camera boxes LiDAR didn't confirm. I fixed
  the parameters in advance, because tuning them on the 4 evaluation scenes would leak test information
  into the method. I call it *late* fusion because it combines two models' finished outputs. It is not
  a network that learns from both sensors.
- **Bootstrap confidence intervals** (`src/eval/bootstrap.py`): matching is independent per frame, so I
  run it once and each resample just reweights frames. 1000 resamples × 5 models takes ~10 s. Every
  model sees the same resampled frames, so differences are paired. Frames within a scene are
  correlated, so these intervals are optimistic, and I print that caveat with every result.

Results: LiDAR 0.720 mAP vs camera 0.452. Fusion 0.753, with the whole gain at 20–40 m. Fusion *hurts*
car and pedestrian, where LiDAR is already near its ceiling and unconfirmed camera boxes only add
false positives. Tracking: LiDAR MOTA 0.567, fusion only 0.321. It's the same calibration problem I'd
seen with the camera models, now much larger: mAP integrates over all score thresholds, while the
tracker runs at one.

## A test file that CI never had

`src/synthetic.py` (synthetic scenes for tests) was never committed, so CI failed on every push while
all my tests passed locally. The cause was a `*SYNTHETIC*` ignore pattern I'd added for one notebook
output file. macOS git matches ignore patterns case-insensitively, so it also matched
`src/synthetic.py`. I force-added the file and scoped the pattern to `results/`. Lesson: a green local
test run proves nothing about a fresh clone. Check CI after pushing, and use narrow ignore patterns.

## Review follow-ups: fusion, devkit check, wording

**Fusion's car/pedestrian loss is not duplicates.** A reviewer suggested dropping unpaired camera boxes
near LiDAR boxes. I estimated it from the demo data first (which holds all predictions with score ≥ 0.15
and reproduces the official LiDAR mAP to 0.001): AP changed by ≤ 0.002. So I isolated the parts. With
LiDAR's own score for paired boxes and no camera-only boxes, fusion reproduces LiDAR exactly, as it should.
Turning on the noisy-OR boost alone produces both the barrier/truck gains and the pedestrian loss. The
camera's agreement is informative for some classes and misleading for others. I didn't adopt a variant
chosen on the test scenes. I added `--ablation` so the table is reproducible, and noted the principled
fix (per-class weighting learned on `seen` scenes).

**Devkit end-to-end check.** My metric tests only proved my AP matches the devkit's *formula*. Nothing
checked my export or coordinate conventions against the official scorer. `scripts/devkit_check.py`
converts my export back to the official submission format and runs the untouched `DetectionEval`.
Its self-test (ground truth fed through the same conversion) first scored car 0.82, not 1.0. The cause
was in the self-test, not the conversion: the devkit drops zero-point GT, my fake "predictions" didn't,
and those extras became false positives tied at score 1.0. After mirroring the filter, every class
present in mini_val scores 1.000. Pipeline step 11 then compares my export of the pretrained BEVFormer
against BEVFormer's own `tools/test.py --eval bbox` on the same checkpoint and split.

**Wording.** The 40 m+ bucket only holds cars and trucks (the devkit caps pedestrians, cyclists and
barriers at ≤ 40 m), so "LiDAR 0.257 vs camera 0.044 beyond 40 m" now says that. I also moved failure
analysis from the weakest model to the camera baseline and the fused model.

## Next
- [ ] Score calibration or a threshold sweep before tracking (fusion loses 0.25 MOTA vs LiDAR at 0.3)
- [ ] Per-class fusion camera weight, tuned on the `seen` scenes only
- [ ] Run pipeline step 11 and record: my export vs BEVFormer's own eval on mini_val (should agree)
- [ ] Per-class fusion weighting learned on `seen` scenes (needs camera + LiDAR predictions there)
