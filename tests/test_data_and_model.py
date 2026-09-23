import json
import os
import sys

import numpy as np
import pytest
from pyquaternion import Quaternion

from src.boxes import Box, bev_iou, ego_to_global, global_to_ego, point_in_box
from src.classes import category_to_class, detection_name_to_class
from src.dataio import Sample, filter_boxes, load_gt, load_predictions, save_gt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from prepare_nuscenes import parse_condition, remap_info_names  # noqa: E402

POSE = {"translation": [100.0, -50.0, 1.0], "rotation": list(Quaternion(axis=[0.1, 0.05, 1], radians=0.7).elements)}


@pytest.mark.parametrize("cat,cls", [("vehicle.car", "car"), ("human.pedestrian.adult", "pedestrian"),
                                     ("human.pedestrian.police_officer", "pedestrian"), ("vehicle.bicycle", "cyclist"),
                                     ("vehicle.motorcycle", "cyclist"), ("vehicle.construction", "truck"),
                                     ("vehicle.truck", "truck"), ("movable_object.barrier", "barrier"),
                                     ("vehicle.bus.rigid", None), ("movable_object.trafficcone", None),
                                     ("vehicle.emergency.police", None)])
def test_category_mapping(cat, cls):
    assert category_to_class(cat) == cls


def test_detection_name_mapping():
    assert detection_name_to_class("bicycle") == "cyclist"
    assert detection_name_to_class("construction_vehicle") == "truck"
    assert detection_name_to_class("bus") is None and detection_name_to_class("traffic_cone") is None


def test_frame_round_trip():
    b = Box("s", [3.0, 4.0, 0.5], [2, 4, 1.5], 0.3, "car", velocity=[1.0, -2.0])
    back = global_to_ego(ego_to_global(b, POSE), POSE)
    np.testing.assert_allclose(back.translation, b.translation, atol=1e-9)
    # Velocity is stored as (vx, vy) only, like the devkit, so the part rotated into z by pitch/roll is
    # lost. POSE is tilted ~4 degrees on purpose (real ego tilt is ~1-2), so this is a worst case.
    np.testing.assert_allclose(back.velocity, b.velocity, atol=0.05)
    # Only yaw is kept, so a tilted round trip drops the pitch/roll coupling (~1e-3 rad here). The eval path
    # never round-trips: GT and predictions each go through exactly one transform, like the devkit.
    assert back.yaw == pytest.approx(b.yaw, abs=2e-3)


def test_yaw_only_transform_is_exact():
    pose = {"translation": [10, 20, 0], "rotation": list(Quaternion(axis=[0, 0, 1], radians=np.pi / 2).elements)}
    g = ego_to_global(Box("s", [1, 0, 0], [1, 1, 1], 0.0, "car", velocity=[1, 0]), pose)
    np.testing.assert_allclose(g.translation, [10, 21, 0], atol=1e-9)
    np.testing.assert_allclose(g.velocity, [0, 1], atol=1e-9)
    assert g.yaw == pytest.approx(np.pi / 2)


def test_bev_iou():
    a = Box("s", [0, 0, 0], [2, 4, 1], 0.0, "car")
    assert bev_iou(a, a) == pytest.approx(1.0)
    assert bev_iou(a, Box("s", [10, 0, 0], [2, 4, 1], 0.0, "car")) == 0.0
    assert bev_iou(a, Box("s", [2, 0, 0], [2, 4, 1], 0.0, "car")) == pytest.approx(1 / 3)  # half overlap along l
    rot = Box("s", [0, 0, 0], [2, 4, 1], np.pi / 2, "car")  # cross shape: overlap 2x2=4, union 8+8-4
    assert bev_iou(a, rot) == pytest.approx(4 / 12)


def test_filters():
    s = Sample("s", "sc", "scene", 0, POSE, bike_racks=[Box("s", [5, 5, 0], [3, 3, 2], 0.0, "bicycle_rack")])
    boxes = [Box("s", [49.9, 0, 0], [1, 1, 1], 0, "car", num_pts=3),
             Box("s", [50.0, 0, 0], [1, 1, 1], 0, "car", num_pts=3),  # at range limit -> dropped (strict <)
             Box("s", [30.0, 0, 0], [1, 1, 1], 0, "barrier", num_pts=3),  # barrier range is 30 m -> dropped
             Box("s", [10.0, 0, 0], [1, 1, 1], 0, "pedestrian", num_pts=0),  # zero points -> dropped for GT
             Box("s", [5.5, 5.5, 0], [1, 2, 1], 0, "cyclist", num_pts=4),  # inside bike rack -> dropped
             Box("s", [-5.5, 5.5, 0], [1, 2, 1], 0, "cyclist", num_pts=4)]
    kept = filter_boxes(boxes, s, is_gt=True)
    assert [(b.name, b.translation[0]) for b in kept] == [("car", 49.9), ("cyclist", -5.5)]
    assert len(filter_boxes(boxes, s, is_gt=False)) == 3  # predictions are not point-filtered
    assert point_in_box([5, 5, 0.9], s.bike_racks[0]) and not point_in_box([5, 5, 1.1], s.bike_racks[0])


def test_nuscenes_format_predictions_are_converted(tmp_path):
    s = Sample("tok", "sc", "scene", 0, POSE, gt=[Box("tok", [5, 0, 0], [2, 4, 1.5], 0.0, "car", num_pts=5)])
    gt_path = tmp_path / "gt.json"
    save_gt(str(gt_path), {"tok": s}, {"split": "t"})
    samples = load_gt(str(gt_path))
    g = ego_to_global(Box("tok", [5, 0, 0], [2, 4, 1.5], 0.25, "bicycle", velocity=[1, 2], score=0.7), POSE)
    rec = {"sample_token": "tok", "translation": g.translation.tolist(), "size": [2, 4, 1.5],
           "rotation": list(Quaternion(axis=[0, 0, 1], radians=g.yaw).elements), "velocity": g.velocity.tolist(),
           "detection_name": "bicycle", "detection_score": 0.7, "attribute_name": ""}
    bus = {**rec, "detection_name": "bus"}
    (tmp_path / "p.json").write_text(json.dumps({"meta": {}, "results": {"tok": [rec, bus]}}))
    preds = load_predictions(str(tmp_path / "p.json"), samples)["tok"]
    assert len(preds) == 1 and preds[0].name == "cyclist"  # bus dropped, bicycle -> cyclist
    np.testing.assert_allclose(preds[0].translation, [5, 0, 0], atol=1e-6)
    assert preds[0].yaw == pytest.approx(0.25, abs=1e-3)  # small pitch/roll in POSE -> not exact


def test_gt_round_trip_keeps_nan_velocity(tmp_path):
    s = Sample("tok", "sc", "scene", 0, POSE, {"lighting": "night"},
               gt=[Box("tok", [5, 0, 0], [2, 4, 1.5], 0.0, "car", velocity=[np.nan, np.nan], instance_id="i", num_pts=5)])
    save_gt(str(tmp_path / "gt.json"), {"tok": s}, {})
    back = load_gt(str(tmp_path / "gt.json"))["tok"]
    assert np.all(np.isnan(back.gt[0].velocity)) and back.condition == {"lighting": "night"}


def test_parse_condition():
    assert parse_condition("Night, after rain, many peds", "singapore-hollandvillage") == {
        "lighting": "night", "weather": "rain", "location": "singapore-hollandvillage"}
    assert parse_condition("Parked truck, construction", "boston-seaport")["lighting"] == "day"


def test_remap_info_names():
    infos = [{"gt_names": np.array(["car", "bicycle", "motorcycle", "bus", "construction_vehicle", "traffic_cone"])}]
    remap_info_names(infos)
    assert list(infos[0]["gt_names"]) == ["car", "cyclist", "cyclist", "bus", "truck", "traffic_cone"]


def test_checkpoint_classifier_remap():
    torch = pytest.importorskip("torch")
    from src.model import BEVFORMER_NUSC_CLASSES, remap_classifier_state_dict

    sd = {}
    for layer in range(6):
        sd[f"pts_bbox_head.cls_branches.{layer}.0.weight"] = torch.randn(256, 256)
        sd[f"pts_bbox_head.cls_branches.{layer}.1.weight"] = torch.randn(256)  # LayerNorm: untouched
        sd[f"pts_bbox_head.cls_branches.{layer}.6.weight"] = torch.randn(10, 256)
        sd[f"pts_bbox_head.cls_branches.{layer}.6.bias"] = torch.randn(10)
    sd["pts_bbox_head.reg_branches.0.4.weight"] = torch.randn(10, 256)  # 10 box params, NOT classes: untouched
    orig = {k: v.clone() for k, v in sd.items()}
    changed = remap_classifier_state_dict(sd)
    assert len(changed) == 12
    w, w0 = sd["pts_bbox_head.cls_branches.3.6.weight"], orig["pts_bbox_head.cls_branches.3.6.weight"]
    assert w.shape == (5, 256)
    idx = {c: i for i, c in enumerate(BEVFORMER_NUSC_CLASSES)}
    torch.testing.assert_close(w[0], w0[idx["car"]])
    torch.testing.assert_close(w[1], w0[idx["pedestrian"]])
    torch.testing.assert_close(w[2], (w0[idx["bicycle"]] + w0[idx["motorcycle"]]) / 2)
    torch.testing.assert_close(w[3], (w0[idx["truck"]] + w0[idx["construction_vehicle"]]) / 2)
    torch.testing.assert_close(w[4], w0[idx["barrier"]])
    assert sd["pts_bbox_head.reg_branches.0.4.weight"].shape == (10, 256)
    assert sd["pts_bbox_head.cls_branches.0.1.weight"].shape == (256,)
