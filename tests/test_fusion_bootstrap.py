import numpy as np
import pytest

from src.boxes import Box
from src.eval.bootstrap import run as bootstrap_run
from src.eval.slice_eval import run as slice_run
from src.fusion import FusionConfig, fuse_sample
from src.synthetic import make_predictions, make_samples


def b(x, y=0.0, name="car", score=0.5, size=(2, 4.5, 1.6)):
    return Box("s", [x, y, 0], size, 0.0, name, velocity=[1, 0], score=score)


def test_fusion_pairs_keep_lidar_geometry_and_noisy_or_score():
    cam = [b(10.8, score=0.6)]
    lid = [b(10.0, score=0.5, size=(2.1, 4.8, 1.7))]
    out = fuse_sample(cam, lid)
    assert len(out) == 1
    assert out[0].translation[0] == 10.0 and out[0].size[1] == 4.8  # LiDAR geometry
    assert out[0].score == pytest.approx(1 - 0.5 * 0.4)  # noisy-OR


def test_fusion_unpaired_and_class_separation():
    cam = [b(10.0, score=0.8, name="pedestrian"), b(30.0, score=0.6)]
    lid = [b(10.0, score=0.7), b(50.0, score=0.4)]
    out = {(o.name, round(o.translation[0])): o.score for o in fuse_sample(cam, lid)}
    # the camera pedestrian and LiDAR car at the same spot are NOT merged (different classes)
    assert out == {("car", 10): pytest.approx(0.7), ("car", 50): pytest.approx(0.4),
                   ("pedestrian", 10): pytest.approx(0.8 * 0.5), ("car", 30): pytest.approx(0.6 * 0.5)}


def test_fusion_radius_is_per_class():
    cam = [b(11.5, name="pedestrian", size=(0.7, 0.7, 1.8), score=0.5)]
    lid = [b(10.0, name="pedestrian", size=(0.7, 0.7, 1.8), score=0.5)]
    assert len(fuse_sample(cam, lid)) == 2  # 1.5 m apart > pedestrian radius 1.0
    assert len(fuse_sample(cam, lid, FusionConfig(radius={"pedestrian": 2.0}))) == 1


@pytest.fixture(scope="module")
def synth():
    samples = make_samples(n_scenes=3, n_frames=8, n_objects=20, seed=5)
    return samples, make_predictions(samples, seed=6), make_predictions(samples, seed=7, base_miss=0.2)


def test_bootstrap_point_estimate_equals_evaluator(synth):
    samples, pa, pb = synth
    res = bootstrap_run(samples, {"a": pa, "b": pb}, n=20)
    ref, *_ = slice_run(samples, pa)
    assert res["models"]["a"]["all"]["mAP"] == pytest.approx(ref[("all", "all")].mean_ap, abs=1e-9)
    assert res["models"]["a"]["20-40m"]["mAP"] == pytest.approx(ref[("distance", "20-40m")].mean_ap, abs=1e-9)


def test_bootstrap_paired_difference(synth):
    samples, pa, pb = synth
    same = bootstrap_run(samples, {"a": pa, "a2": pa}, n=30)
    d = same["paired"]["a2 - a"]["all"]
    assert d["diff"] == 0 and d["ci95"] == [0.0, 0.0]
    res = bootstrap_run(samples, {"a": pa, "worse": pb}, n=200)
    w = res["paired"]["worse - a"]["all"]
    assert w["diff"] < 0 and w["ci95"][1] < 0 and w["p_gt_0"] < 0.05  # the degraded detector is reliably worse
    lo, hi = res["models"]["a"]["all"]["ci95"]
    assert lo <= res["models"]["a"]["all"]["mAP"] <= hi
