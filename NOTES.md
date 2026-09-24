# Build log: what broke and how it was fixed

Running notes, newest at the bottom. Interview material lives here.

## 2026-09-22: scaffold, eval harness, tracker

**Dev machine has no NVIDIA GPU (Apple M3 Pro).** BEVFormer needs mmcv-full 1.4.0 CUDA ops
(deformable attention), so it cannot run here at all. Split the project into two environments:
- `requirements-eval.txt`: CPU. Eval harness, tracker, tests and notebook run on the laptop.
- `docker/Dockerfile`: CUDA 11.1 / torch 1.9.1 / mmcv-full 1.4.0 / mmdet 2.14.0 / mmdet3d 0.17.1,
  pinned verbatim from BEVFormer's `docs/install.md`. CUDA 11.1 tops out at sm_86, so use a
  T4/V100/A10/A100, not an L4/4090/H100.

**Pretrained checkpoints have already seen most of nuScenes-mini.** The public BEVFormer-tiny
checkpoint was trained on the official train split. Checking `nuscenes.utils.splits`:
6 of 8 mini_train scenes are official train, including all three night scenes (1077, 1094, 1100).
Only 0103, 0553, 0796 and 0916 are official val. Evaluating the pretrained model on mini_train
(or slicing by night on it) measures memorisation.
→ Fix: `data/prepare_nuscenes.py` defines `seen` (fine-tune here) and `clean` (evaluate here).
Both models are evaluated on the same 4 clean scenes. Open question until the data is downloaded:
is any clean scene at night? If not, the lighting slice can't be measured without leakage, and the
README must say so rather than report a contaminated number.

**The devkit's `DetectionEval` can't do this project's eval.** It hard-codes the 10 detection
classes and only evaluates a whole split.
→ Reimplemented matching, AP and TP errors (`src/eval/metrics.py`) and cross-checked them against the
devkit's own `accumulate`/`calc_ap`/`calc_tp` on random data (`tests/test_eval_pipeline.py`).
- First cross-check failed on car @ 2 m (0.337 vs 0.322). Cause: score **ties**. The devkit breaks
  ties by position in its flattened box list, and the test inserted samples in set order. The
  metric code was right and the test was wrong, but it shows that AP can depend on tie-breaking
  when scores are quantized.

**Distance slicing creates fake errors if done naively.** Filtering GT and predictions into
distance buckets independently turns a pred at 19.5 m matched to a GT at 20.5 m into a FP in one
bucket and a FN in the other. → Match once globally; a matched prediction follows its GT's bucket.
There's a test for exactly this case.

**CLEAR-MOT ID switches disagreed with motmetrics (18 vs 14).** My version only re-used the
*previous frame's* correspondences. motmetrics re-establishes each GT's *last-ever* correspondence,
so after a one-frame miss my version could let a different hypothesis grab the object and count a
switch. Then fragmentation was off by one (29 vs 28): when two GTs share a last hypothesis after a
swap, the order of re-establishment matters, and motmetrics goes in frame order.
→ Both fixed. Randomised cross-check test against motmetrics now passes.

**Library breakage to remember:** motmetrics 1.4 breaks on string ids under pandas 3 → pin `pandas<3`.
mmcv 1.7 (for parsing configs locally) needs `--no-build-isolation` and `setuptools<70`.

**Config gotcha:** mmcv config inheritance merges dicts but *replaces* lists, and variables like
`class_names` are already substituted into the base's dicts. Overriding `class_names` alone changes
nothing. Every pipeline that mentions classes had to be restated. Verified by loading the merged
config with `mmcv.Config`.

**BEVFormer's dataset `evaluate()` can't handle custom class names** (read from the code, not yet run: it looks up nuScenes default
attributes per class name). → Disabled in-training validation; export predictions with our own
code (`src/model.py export`) and evaluate with our harness.

**Changing `bev_h/bev_w` or `point_cloud_range` breaks the checkpoint.** `bev_embedding` is
(bev_h·bev_w × 256), and reference points are normalised by pc_range. Kept upstream values.

## 2026-09-22 (later): real data + free-GPU path

**nuScenes-mini downloads without a login** (`https://www.nuscenes.org/data/v1.0-mini.tgz`); the
CAN bus expansion does not (returns the login page).

**Split report answered the open question: all 4 clean scenes are day + dry.** All night/rain scenes
(1077, 1094, 1100) are in the official train split. On mini, a lighting/weather slice can only be
measured on scenes the pretrained model has seen. The honest condition slice here is location
(Boston 0103/0553 vs Singapore 0796/0916).

**GT pipeline cross-checked against the devkit on real data:** for mini_val, `build_gt` +
`filter_boxes` keep exactly the boxes the devkit's `load_gt` → `add_center_dist` →
`filter_eval_boxes` keeps (car 1913, ped 1067, cyclist 250, truck 95; no barriers in mini_val).

**Free GPU = Colab T4, but Colab is Python 3.12 / CUDA 12.** Plan in `notebooks/colab_gpu.ipynb`
(not yet run, since the first run needs your Google session; expect to debug here first):
- micromamba env with Python 3.8 + torch 1.9.1+cu111 (bundles its CUDA runtime, runs on newer drivers)
- mmdet3d 0.17.1 ops compiled with conda nvcc **11.8**. torch 1.9 only rejects a CUDA *major* mismatch.
  gcc-9 from apt, because gcc 11 is too new for torch 1.9 headers. The wheel is cached in Drive.
- `yapf==0.40.1` pinned (newer yapf breaks mmcv 1.4 `Config.pretty_text`), `numpy==1.19.5`
  enforced with a pip constraints file on every install.
- All wheel URLs were verified to exist for cp38 before writing the notebook.

## 2026-09-24: first Colab run

Environment build and steps 0-4 worked on the first try (mmdet3d wheel compiled and cached in Drive).
**Step 5 died importing BEVFormer's plugin:** `projects.mmdet3d_plugin` → dd3d → detectron2 0.6 →
`detectron2.data.transforms` uses `PIL.Image.LINEAR`, **removed in Pillow 10**. Pillow wasn't pinned,
so pip picked 10.x. → Pinned `pillow==9.5.0` (notebook constraints + Dockerfile). A dependency
that is imported but never used for BEVFormer-tiny (DD3D) still has to import cleanly.

**Step 2 then failed with `No module named 'tools.data_converter'`.** BEVFormer's `create_data.py`
imports `indoor_converter`, which imports `tools.data_converter.*` absolutely. That only resolves with
the BEVFormer root on PYTHONPATH (its `dist_*.sh` wrappers set it; calling the script directly
doesn't). → Set `PYTHONPATH=$BF` for create_data. **That was not enough; same error.** Real cause:
the detectron2 0.6 wheel installs a top-level package named `tools` (its repo's `tools/` has an
`__init__.py` and setup.py uses `find_packages()`). BEVFormer's `tools/` has no `__init__.py`, so it is a
*namespace* package, and Python's path finder returns a regular package over a namespace portion
regardless of sys.path order. So `tools` resolved to detectron2's. → `touch third_party/BEVFormer/tools/__init__.py`
at runtime (keeps the pinned submodule untouched in git). Lesson: my first fix treated the symptom
("not on the path") without checking *which* `tools` was being imported. Also made steps 5-7 build missing info files
themselves, so an upstream failure shows its real error instead of a downstream FileNotFoundError.

**Pillow came back at 10.x before training** (Colab session restart + an older notebook copy without
the pin), same `Image.LINEAR` crash in step 6. → `gpu_pipeline.sh` now checks for `Image.LINEAR` and
reinstalls pillow 9.5.0 itself. Fixing it in one place (the notebook) wasn't enough, because the env
can be rebuilt from a different place.

## 2026-09-24: first real results

Baseline (pretrained, relabeled 10→5) mAP 0.452 / NDS 0.436. Fine-tuned 0.388 / 0.379. **Fine-tuning
made it worse.** In hindsight: (1) the 5 classes are merges of existing ones, so relabeling at
inference is already a complete solution, and training has nothing to add but plenty to break; (2) my
config trained far more than "the head" (encoder 0.1×, rest of transformer 0.5×) on 242 samples,
which is exactly the setup the spec warned against. → Added `configs/bevformer_tiny_nusc_headonly.py`
(everything except cls/reg branches frozen, 4 epochs) as the fair version of the experiment.

Also fixed a comparison bug I'd flagged: per-slice mAP averages over the classes present *in that
slice*, so Singapore (no barriers) looked 7 points worse than Boston. Per class they are the same.
Added `mAP*` = mean over classes present in every slice of an axis.

Training time: 12 epochs × 242 iters at ~2 s/iter = 1 h 40 min on a free T4; ~0.7 s/iter was data
loading (2 CPU cores decoding 18 JPEGs per step).

## Next
- [x] Download v1.0-mini; run `data/prepare_nuscenes.py report` and `gt`
- [x] Get the CAN bus expansion (login) → Drive
- [x] Run notebooks/colab_gpu.ipynb on a free T4
- [ ] Head-only fine-tune ablation
- [ ] Tracker on baseline detections; score-threshold sweep
- [ ] GPU box: build docker image, run `scripts/gpu_pipeline.sh` step by step
- [ ] Sanity: `run_official_devkit_eval` (10-class, mini_val) on the pretrained model should land near
      BEVFormer's full-val numbers (NDS 35.4 / mAP 25.2), allowing for 2-scene noise
