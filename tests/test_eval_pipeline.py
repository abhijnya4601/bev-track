"""Tests for the evaluation code itself. A silently wrong metric is worse than a silently wrong model."""
import numpy as np
import pytest

from src.boxes import Box
from src.classes import CLASSES
from src.eval.failure_cases import extract
from src.eval.metrics import (DIST_THRESHOLDS, TP_METRICS, angle_diff, evaluate, match)
from src.eval.slice_eval import bucket_names, distance_bucket, run, slice_masks
from src.synthetic import make_predictions, make_samples


def car(tok, x, y=0.0, score=-1.0, yaw=0.0, size=(2.0, 4.0, 1.5), vel=(0.0, 0.0), name="car", iid=None):
    return Box(tok, [x, y, 0.0], size, yaw, name, velocity=vel, score=score, instance_id=iid, num_pts=10)


# ---------------------------------------------------------------------------
# Hand-calculated AP
# ---------------------------------------------------------------------------
# nuScenes AP: precision is sampled at 101 recall points (0, 0.01, ..., 1). Recall points <= 0.10
# are dropped, 0.1 is subtracted from each precision (clipped at 0), and the mean over the
# remaining 90 points is divided by 0.9.

def test_perfect_detection_gives_ap_1_and_nds_0_9():
    gt = [car("s", 10), car("s", 20, 5)]
    preds = [car("s", 10, score=0.9), car("s", 20, 5, score=0.8)]
    res = evaluate(gt, preds)
    assert res.per_class["car"].ap == pytest.approx(1.0)
    assert res.mean_ap == pytest.approx(1.0)  # other classes have no GT -> excluded, not zero
    assert all(res.per_class["car"].tp_errors[m] == pytest.approx(0.0) for m in TP_METRICS)
    # NDS = (5 * mAP + sum(1 - err) over 4 TP metrics + (1 - mAAE=1)) / 10 = (5 + 4 + 0) / 10
    assert res.nds == pytest.approx(0.9)


def test_one_hit_one_miss_one_false_positive_fp_ranked_first():
    """2 GT; predictions: FP (score 0.9), then TP (score 0.5); second GT missed.

    Cumulative: after FP -> P=0, R=0; after TP -> P=0.5, R=0.5. The devkit linearly interpolates
    precision over recall, so P(r) = r for r in [0, 0.5], and 0 beyond (recall never reaches it).
    AP = sum_{i=11..50} (i/100 - 0.1) / 90 / 0.9 = (sum_{k=1..40} k/100) / 81 = 8.2 / 81
    """
    gt = [car("s", 10), car("s", -30)]
    preds = [car("s", 40, 20, score=0.9), car("s", 10, score=0.5)]
    res = evaluate(gt, preds)
    for th in DIST_THRESHOLDS:
        assert res.per_class["car"].ap_by_threshold[th] == pytest.approx(8.2 / 81)
    assert res.mean_ap == pytest.approx(8.2 / 81)


def test_one_hit_one_miss_one_false_positive_tp_ranked_first():
    """Same boxes, but the TP outranks the FP.

    Cumulative: after TP -> P=1, R=0.5; after FP -> P=0.5, R=0.5. So P=1 for r < 0.5; at r == 0.5
    np.interp takes the last value, 0.5; 0 beyond.
    AP = (39 * 0.9 + 0.4) / 90 / 0.9 = 35.5 / 81
    """
    gt = [car("s", 10), car("s", -30)]
    preds = [car("s", 10, score=0.9), car("s", 40, 20, score=0.5)]
    res = evaluate(gt, preds)
    assert res.per_class["car"].ap == pytest.approx(35.5 / 81)


def test_distance_thresholds_are_strict():
    gt = [car("s", 10)]
    preds = [car("s", 11.0, score=0.9)]  # exactly 1.0 m off -> miss at 0.5 m and 1 m (strict <), hit at 2 m and 4 m
    aps = evaluate(gt, preds).per_class["car"].ap_by_threshold
    assert aps[0.5] == 0 and aps[1.0] == 0 and aps[2.0] == pytest.approx(1) and aps[4.0] == pytest.approx(1)


def test_class_mismatch_is_fp_and_fn():
    res = evaluate([car("s", 10)], [car("s", 10, score=0.9, name="truck")])
    assert res.per_class["car"].ap == 0
    assert np.isnan(res.per_class["truck"].ap)  # no truck GT -> undefined, not 0
    assert res.per_class["truck"].n_pred == 1  # still reported, so the FP is visible in the tables


def test_greedy_matching_by_score_not_distance():
    # Higher-scoring pred steals the GT even though the lower-scoring one is closer (devkit behaviour).
    gt = [car("s", 10)]
    preds = [car("s", 11.5, score=0.9), car("s", 10.1, score=0.5)]
    mt = match(gt, preds, thresholds=(2.0,))[("car", 2.0)]
    assert list(mt.gt_idx) == [0, -1]


def test_orientation_error_barrier_period():
    assert abs(angle_diff(0.0, np.pi, 2 * np.pi)) == pytest.approx(np.pi)
    assert abs(angle_diff(0.0, np.pi, np.pi)) == pytest.approx(0.0)
    gt = [car("s", 10, name="barrier")]
    preds = [car("s", 10, score=0.9, name="barrier", yaw=np.pi)]  # flipped barrier is not an error
    assert evaluate(gt, preds).per_class["barrier"].tp_errors["orient_err"] == pytest.approx(0.0)
    assert np.isnan(evaluate(gt, preds).per_class["barrier"].tp_errors["vel_err"])


# ---------------------------------------------------------------------------
# Distance bucketing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d,bucket", [(0.0, "0-20m"), (19.999, "0-20m"), (20.0, "20-40m"), (39.99, "20-40m"),
                                      (40.0, "40m+"), (120.0, "40m+")])
def test_distance_bucket_boundaries(d, bucket):
    assert distance_bucket(d) == bucket


def test_distance_bucket_rejects_nan():
    with pytest.raises(ValueError):
        distance_bucket(float("nan"))


def test_ego_distance_is_xy_only():
    b = Box("s", [12.0, 16.0, 30.0], [1, 1, 1], 0.0, "car")
    assert b.ego_dist == pytest.approx(20.0)
    assert distance_bucket(b.ego_dist) == "20-40m"


def test_boundary_match_is_not_split_across_buckets():
    """GT at 20.5 m, prediction at 19.5 m. They match (1 m apart). Filtering GT and predictions
    independently would make this a FN in 20-40m and a FP in 0-20m. Here the prediction follows its GT."""
    gt = [car("s", 20.5)]
    preds = [car("s", 19.5, score=0.9)]
    gm = np.array([distance_bucket(b.ego_dist) == "20-40m" for b in gt])
    pm = np.array([distance_bucket(b.ego_dist) == "20-40m" for b in preds])
    near_g, near_p = ~gm, ~pm
    far = evaluate(gt, preds, gt_mask=gm, pred_mask=pm)
    near = evaluate(gt, preds, gt_mask=near_g, pred_mask=near_p)
    assert far.per_class["car"].ap_by_threshold[2.0] == pytest.approx(1.0)
    assert far.per_class["car"].n_pred == 1
    assert near.per_class["car"].n_gt == 0 and near.per_class["car"].n_pred == 0


# ---------------------------------------------------------------------------
# Slicing consistency on synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth():
    samples = make_samples(n_scenes=4, n_frames=8, n_objects=25, seed=3)
    preds = make_predictions(samples, seed=4)
    results, gt, pr, matches = run(samples, preds)
    return samples, results, gt, pr, matches


def test_slices_partition_gt_and_predictions(synth):
    samples, results, gt, pr, matches = synth
    for axis_values in (bucket_names(), ["day", "night"]):
        axis = "distance" if "0-20m" in axis_values else "lighting"
        for cls in CLASSES:
            tot = results[("all", "all")].per_class[cls]
            parts = [results[(axis, v)].per_class[cls] for v in axis_values]
            assert sum(p.n_gt for p in parts) == tot.n_gt
            assert sum(p.n_pred for p in parts) == tot.n_pred


def test_unsliced_equals_plain_evaluate(synth):
    samples, results, gt, pr, matches = synth
    plain = evaluate(gt, pr)
    assert results[("all", "all")].mean_ap == pytest.approx(plain.mean_ap)
    assert results[("all", "all")].nds == pytest.approx(plain.nds)


def test_harness_detects_the_injected_degradation(synth):
    """The synthetic detector is built to be worse far away and at night; the harness must see that."""
    _, results, *_ = synth
    assert results[("distance", "0-20m")].mean_ap > results[("distance", "20-40m")].mean_ap
    assert results[("lighting", "day")].mean_ap > results[("lighting", "night")].mean_ap


def test_slice_masks_cover_all_axes(synth):
    samples, _, gt, pr, _ = synth
    keys = slice_masks(gt, pr, samples).keys()
    assert ("distance", "40m+") in keys and ("lighting", "night") in keys and ("weather", "rain") in keys


# ---------------------------------------------------------------------------
# Cross-check against the official devkit
# ---------------------------------------------------------------------------

def _random_case(seed=0, n_samples=6):
    rng = np.random.default_rng(seed)
    gt, preds = [], []
    for s in range(n_samples):
        tok = f"s{s}"
        for cls in CLASSES:
            for _ in range(rng.integers(0, 6)):
                xy = rng.uniform(-40, 40, 2)
                size = rng.uniform(0.5, 5, 3)
                yaw = rng.uniform(-np.pi, np.pi)
                vel = rng.normal(0, 3, 2) if rng.random() > 0.1 else np.array([np.nan, np.nan])
                g = Box(tok, [*xy, 1.0], size, yaw, cls, velocity=vel, num_pts=5)
                gt.append(g)
                if rng.random() < 0.8:
                    preds.append(Box(tok, [*(xy + rng.normal(0, 1.2, 2)), 1.0], size * rng.uniform(0.8, 1.2, 3),
                                     yaw + rng.normal(0, 0.4), cls, velocity=rng.normal(0, 3, 2),
                                     score=float(np.round(rng.uniform(0, 1), 1))))  # rounding creates score ties
            for _ in range(rng.integers(0, 3)):
                preds.append(Box(tok, [*rng.uniform(-40, 40, 2), 1.0], rng.uniform(0.5, 5, 3), 0.0, cls,
                                 velocity=[0, 0], score=float(np.round(rng.uniform(0, 1), 1))))
    return gt, preds


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_official_devkit(seed):
    pytest.importorskip("nuscenes")
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.data_classes import DetectionBox
    from pyquaternion import Quaternion

    to_devkit = {"car": "car", "pedestrian": "pedestrian", "cyclist": "bicycle", "truck": "truck", "barrier": "barrier"}
    gt, preds = _random_case(seed)

    def eval_boxes(boxes, is_pred):
        eb = EvalBoxes()
        by_s = {}
        for b in boxes:
            by_s.setdefault(b.sample_token, []).append(DetectionBox(
                sample_token=b.sample_token, translation=tuple(b.translation), size=tuple(b.size),
                rotation=tuple(Quaternion(axis=[0, 0, 1], radians=b.yaw).elements), velocity=tuple(b.velocity),
                detection_name=to_devkit[b.name], detection_score=b.score if is_pred else -1.0, attribute_name=""))
        # Insertion order matters: the devkit breaks score ties by position in its flattened list.
        for s in sorted({b.sample_token for b in gt}):
            eb.add_boxes(s, by_s.get(s, []))
        return eb

    g_eb, p_eb = eval_boxes(gt, False), eval_boxes(preds, True)
    ours = evaluate(gt, preds)
    for cls in CLASSES:
        for th in DIST_THRESHOLDS:
            md = accumulate(g_eb, p_eb, to_devkit[cls], center_distance, th)
            assert ours.per_class[cls].ap_by_threshold[th] == pytest.approx(calc_ap(md, 0.1, 0.1), abs=1e-9), (cls, th)
            if th == 2.0:
                for m in TP_METRICS:
                    if np.isnan(ours.per_class[cls].tp_errors[m]):
                        continue
                    assert ours.per_class[cls].tp_errors[m] == pytest.approx(calc_tp(md, 0.1, m), abs=1e-9), (cls, m)


# ---------------------------------------------------------------------------
# Failure case extraction
# ---------------------------------------------------------------------------

def test_failure_diagnoses():
    gt = [car("s", 10), car("s", -20, 5), car("s", 30, -10)]
    preds = [
        car("s", 10.2, score=0.95),  # TP on GT0
        car("s", 10.5, 0.3, score=0.7),  # duplicate of GT0
        car("s", -20, 5, score=0.8, name="pedestrian"),  # class confusion on GT1
        car("s", 33.0, -10, score=0.6),  # 3 m off GT2 -> mislocalized FP and FN
        car("s", -45, -45, score=0.9),  # hallucination
    ]
    f = extract(gt, preds, n=10)
    fp = {round(x.box.translation[0], 1): x.diagnosis for x in f["fp"]}
    assert fp == {10.5: "duplicate", -20.0: "class_confusion", 33.0: "mislocalized", -45.0: "hallucination"}
    fn = {round(x.box.translation[0], 1): x.diagnosis for x in f["fn"]}
    assert fn == {-20.0: "class_confusion", 30.0: "mislocalized"}
    assert [x.box.translation[0] for x in f["loc"]] == [10.0]
    assert f["fp"][0].box.score == 0.9  # ranked by score
