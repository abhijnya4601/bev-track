"""End-to-end check of our export and coordinate conventions against the untouched nuScenes devkit.

    python scripts/devkit_check.py --gt data/processed/gt_mini_val.json \
        --pred results/preds_pretrained_raw10_minival.json --dataroot data/nuscenes

The input is BEV-Track's own export (``src.model export --raw-names``): ego frame, the model's original
10 nuScenes class names. This script converts it to the official submission format (global frame,
quaternion rotation, per-class default attributes exactly as mmdet3d assigns them) and runs the devkit's
``DetectionEval``. If our LiDAR→ego conversion, yaw convention, velocity frame or ego→global transform
were wrong, mAP/NDS would collapse or drift. Compare against BEVFormer's own ``tools/test.py --eval bbox``
on the same checkpoint and split (``gpu_pipeline.sh 11`` runs both). They should agree.

``--self-test`` instead feeds the devkit the ground truth itself (converted through the same code path).
Every class present in the split must score AP 1.000 (classes with no GT in the split score 0 by devkit
convention: on mini_val that is trailer, construction_vehicle and barrier). It needs only the dataset.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.boxes import Box, ego_to_global, global_to_ego, yaw_of  # noqa: E402
from src.dataio import load_gt  # noqa: E402

# mmdet3d 0.17 NuScenesDataset.DefaultAttribute + the velocity rule in its _format_bbox
DEFAULT_ATTRIBUTE = {"car": "vehicle.parked", "pedestrian": "pedestrian.moving", "trailer": "vehicle.parked",
                     "truck": "vehicle.parked", "bus": "vehicle.moving", "motorcycle": "cycle.without_rider",
                     "construction_vehicle": "vehicle.parked", "bicycle": "cycle.without_rider",
                     "barrier": "", "traffic_cone": ""}


def attribute(name: str, velocity) -> str:
    moving = np.hypot(velocity[0], velocity[1]) > 0.2
    if moving:
        if name in ("car", "construction_vehicle", "bus", "truck", "trailer"):
            return "vehicle.moving"
        if name in ("bicycle", "motorcycle"):
            return "cycle.with_rider"
    else:
        if name == "pedestrian":
            return "pedestrian.standing"
        if name == "bus":
            return "vehicle.stopped"
    return DEFAULT_ATTRIBUTE[name]


def to_submission(samples, boxes_by_sample) -> dict:
    results = {}
    for tok, s in samples.items():
        out = []
        for b in boxes_by_sample.get(tok, []):
            g = ego_to_global(b, s.ego_pose)
            vel = [float(v) for v in np.nan_to_num(g.velocity)]
            out.append({"sample_token": tok, "translation": g.translation.tolist(), "size": g.size.tolist(),
                        "rotation": list(Quaternion(axis=[0, 0, 1], radians=g.yaw).elements),
                        "velocity": vel, "detection_name": b.name, "detection_score": float(b.score),
                        "attribute_name": attribute(b.name, vel)})
        results[tok] = out
    return {"meta": {"use_camera": True, "use_lidar": True, "use_radar": False, "use_map": False,
                     "use_external": False}, "results": results}


def load_raw(path, samples):
    with open(path) as f:
        raw = json.load(f)["results"]
    return {tok: [Box.from_dict({**b, "sample_token": tok}) for b in raw.get(tok, [])] for tok in samples}


def gt_as_predictions(samples, nusc):
    """Ground truth in the ego frame with 10-class detection names, via the same transforms as the export."""
    from nuscenes.eval.detection.utils import category_to_detection_name

    out = {}
    for tok, s in samples.items():
        boxes = []
        for a in nusc.get("sample", tok)["anns"]:
            ann = nusc.get("sample_annotation", a)
            name = category_to_detection_name(ann["category_name"])
            # The devkit drops GT boxes with no lidar/radar points; keep the same set as "predictions",
            # otherwise those objects become false positives tied at score 1.0 with the true positives.
            if name is None or ann["num_lidar_pts"] + ann["num_radar_pts"] == 0:
                continue
            g = Box(tok, ann["translation"], ann["size"], yaw_of(Quaternion(ann["rotation"])), name,
                    velocity=nusc.box_velocity(a)[:2], score=1.0)
            boxes.append(global_to_ego(g, s.ego_pose))
        out[tok] = boxes
    return out


def evaluate(submission, dataroot, version, eval_set, out_dir):
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "submission.json")
    with open(path, "w") as f:
        json.dump(submission, f)
    ev = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=path,
                       eval_set=eval_set, output_dir=out_dir, verbose=False)
    metrics, _ = ev.evaluate()
    return metrics.serialize(), nusc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="data/processed/gt_mini_val.json", help="GT file (for sample list + ego poses)")
    ap.add_argument("--pred", help="BEV-Track export made with --raw-names")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--eval-set", default="mini_val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true", help="score the GT itself through our conversion (expect mAP ~ 1)")
    args = ap.parse_args(argv)

    samples = load_gt(args.gt)
    out_dir = args.out or tempfile.mkdtemp(prefix="devkit_check_")
    if args.self_test:
        from nuscenes import NuScenes

        boxes = gt_as_predictions(samples, NuScenes(version=args.version, dataroot=args.dataroot, verbose=False))
    else:
        boxes = load_raw(args.pred, samples)
    m, _ = evaluate(to_submission(samples, boxes), args.dataroot, args.version, args.eval_set, out_dir)
    print(f"devkit ({args.eval_set}, 10 classes): mAP {m['mean_ap']:.4f}  NDS {m['nd_score']:.4f}")
    print("  TP errors: " + "  ".join(f"{k} {v:.3f}" for k, v in m["tp_errors"].items()))
    print("  per-class AP: " + "  ".join(f"{k} {v:.3f}" for k, v in m["mean_dist_aps"].items()))
    with open(os.path.join(out_dir, "devkit_summary.json"), "w") as f:
        json.dump({"mAP": m["mean_ap"], "NDS": m["nd_score"], "per_class": m["mean_dist_aps"],
                   "tp_errors": m["tp_errors"]}, f, indent=2)


if __name__ == "__main__":
    main()
