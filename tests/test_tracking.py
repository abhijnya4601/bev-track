import numpy as np
import pytest
from pyquaternion import Quaternion

from src.boxes import Box
from src.dataio import Sample
from src.eval.track_metrics import clear_mot, evaluate_tracking
from src.track import TrackerConfig, track_all


def frame(gt, hyp):
    """gt/hyp: dict id -> (x, y)."""
    return (list(gt), np.array(list(gt.values()), float).reshape(-1, 2),
            list(hyp), np.array(list(hyp.values()), float).reshape(-1, 2))


def test_perfect_tracking():
    frames = [frame({"a": (t, 0), "b": (0, 10 + t)}, {"1": (t, 0), "2": (0, 10 + t)}) for t in range(10)]
    c = clear_mot(frames)
    assert (c.mota, c.id_switches, c.fragmentations, c.fp, c.fn) == (1.0, 0, 0, 0, 0)
    assert c.motp == 0.0 and c.mostly_tracked == 2


def test_id_swap_counts_two_switches():
    # Two far-apart objects; at t=5 the tracker swaps their ids.
    frames = []
    for t in range(10):
        ids = ("1", "2") if t < 5 else ("2", "1")
        frames.append(frame({"a": (t, 0), "b": (t, 20)}, {ids[0]: (t, 0), ids[1]: (t, 20)}))
    c = clear_mot(frames)
    assert c.id_switches == 2
    assert c.mota == pytest.approx(1 - 2 / 20)


def test_fragmentation_without_switch():
    # Object tracked, lost for 2 frames, re-acquired by the same track id: 1 fragmentation, 0 switches.
    hyp_present = [1, 1, 1, 0, 0, 1, 1]
    frames = [frame({"a": (t, 0)}, {"1": (t, 0)} if h else {}) for t, h in enumerate(hyp_present)]
    c = clear_mot(frames)
    assert (c.fragmentations, c.id_switches, c.fn) == (1, 0, 2)


def test_new_id_after_gap_is_switch_and_fragment():
    hyps = ["1", "1", None, "7", "7"]
    frames = [frame({"a": (t, 0)}, {h: (t, 0)} if h else {}) for t, h in enumerate(hyps)]
    c = clear_mot(frames)
    assert (c.fragmentations, c.id_switches) == (1, 1)


def test_previous_match_is_kept_over_closer_newcomer():
    frames = [frame({"a": (0, 0)}, {"1": (0.5, 0)}),
              frame({"a": (1, 0)}, {"1": (1.8, 0), "2": (1.0, 0)})]  # "1" still within 2 m -> keep it
    c = clear_mot(frames)
    assert c.id_switches == 0 and c.fp == 1


def test_matches_motmetrics_on_random_sequences():
    mm = pytest.importorskip("motmetrics")
    rng = np.random.default_rng(0)
    for trial in range(5):
        frames = []
        acc = mm.MOTAccumulator(auto_id=True)
        hid_int = {}
        for t in range(30):
            gt = {f"g{k}": (k * 3.0 + 0.1 * t, rng.normal(0, 0.3)) for k in range(6) if rng.random() > 0.1}
            hyp = {}
            for k in range(6):
                if f"g{k}" in gt and rng.random() > 0.2:
                    hid = f"h{k}" if rng.random() > 0.1 else f"h{(k + 1) % 6}"
                    if hid not in hyp:
                        hyp[hid] = (gt[f"g{k}"][0] + rng.normal(0, 0.8), gt[f"g{k}"][1] + rng.normal(0, 0.8))
            if rng.random() > 0.5:
                hyp[f"fp{t}"] = tuple(rng.uniform(0, 20, 2))
            f = frame(gt, hyp)
            frames.append(f)
            D = np.linalg.norm(f[1][:, None] - f[3][None], axis=2) if len(gt) and len(hyp) else np.empty((len(gt), len(hyp)))
            D = np.where(D < 2.0, D, np.nan)
            # motmetrics 1.4 breaks on string ids with pandas>=3, so hand it integers.
            acc.update([int(g[1:]) for g in f[0]], [hid_int.setdefault(h, len(hid_int)) for h in f[2]], D)
        s = mm.metrics.create().compute(acc, metrics=["num_switches", "num_fragmentations", "num_misses",
                                                      "num_false_positives", "mota", "motp"]).iloc[0]
        c = clear_mot(frames)
        assert c.id_switches == s["num_switches"]
        assert c.fragmentations == s["num_fragmentations"]
        assert c.fn == s["num_misses"] and c.fp == s["num_false_positives"]
        assert c.mota == pytest.approx(s["mota"])
        assert c.motp == pytest.approx(s["motp"])


def _moving_scene(n_frames=12, ego_speed=5.0):
    """Two cars moving in the global frame while the ego car also drives and turns."""
    samples, preds = {}, {}
    for f in range(n_frames):
        t = 0.5 * f
        ego_yaw = 0.05 * f
        pose = {"translation": [ego_speed * t, 0.0, 0.0],
                "rotation": list(Quaternion(axis=[0, 0, 1], radians=ego_yaw).elements)}
        tok = f"f{f}"
        from src.boxes import global_to_ego

        gts = [global_to_ego(Box(tok, [10 + 8 * t, 3, 0], [2, 4.5, 1.6], 0.0, "car", velocity=[8, 0],
                                 instance_id="A", num_pts=10), pose),
               global_to_ego(Box(tok, [30 - 6 * t, -3, 0], [2, 4.5, 1.6], np.pi, "car", velocity=[-6, 0],
                                 instance_id="B", num_pts=10), pose)]
        samples[tok] = Sample(tok, "sc", "scene-x", 1_000_000 + int(t * 1e6), pose, {}, gts)
        preds[tok] = [Box(tok, g.translation, g.size, g.yaw, g.name, velocity=g.velocity, score=0.9) for g in gts]
    return samples, preds


def test_tracker_end_to_end_with_moving_ego():
    samples, preds = _moving_scene()
    tracks = track_all(samples, preds, TrackerConfig(min_hits=1))
    res = evaluate_tracking(samples, tracks)
    car = res.per_class["car"]
    # The two cars pass each other (closing speed 14 m/s); a tracker working in the ego frame, or
    # without a motion model, would swap their ids at the crossing.
    assert car.id_switches == 0 and car.fragmentations == 0
    assert car.mota == pytest.approx(1.0)
    ids = {b.instance_id for bs in tracks.values() for b in bs}
    assert len(ids) == 2


def test_tracker_survives_missed_frame():
    samples, preds = _moving_scene()
    preds["f5"] = []  # detector drops both cars for one frame
    tracks = track_all(samples, preds, TrackerConfig(min_hits=1, max_age=2))
    res = evaluate_tracking(samples, tracks)
    car = res.per_class["car"]
    assert car.id_switches == 0  # tracks coasted through the gap
    assert car.fragmentations == 2 and car.fn == 2  # the gap itself is still a miss
