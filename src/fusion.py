"""Late fusion of camera and LiDAR detections, box by box, in the ego frame.

    python -m src.fusion --gt data/processed/gt_clean.json \
        --camera results/preds_pretrained.json --lidar results/preds_lidar.json \
        --out results/preds_fused.json

This is **late** fusion: two independently trained detectors, their finished boxes combined. It is
not a fusion network (e.g. BEVFusion, which learns from both sensors jointly), and it shouldn't be
described as one.

Per sample and class:
  1. greedily pair each LiDAR box (highest score first) with the closest unpaired camera box of the
     same class within ``radius[class]`` metres (BEV centre distance)
  2. paired: keep the **LiDAR geometry** (centre, size, heading, velocity; LiDAR measures range
     directly, cameras infer it) and combine scores by noisy-OR, ``1 - (1 - s_lidar)(1 - s_camera)``,
     so agreement between sensors raises confidence
  3. unpaired boxes survive with their score scaled by ``lidar_only_weight`` / ``camera_only_weight``

Every parameter is fixed a priori, **not tuned on the evaluation scenes**. Tuning fusion weights on the 4
clean scenes and then reporting results on them would leak test information into the method.
"""
import argparse
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.boxes import Box
from src.dataio import MAX_BOXES_PER_SAMPLE, load_gt, load_predictions, save_predictions


@dataclass
class FusionConfig:
    radius: Dict[str, float] = field(default_factory=lambda: {
        "car": 2.0, "truck": 2.5, "pedestrian": 1.0, "cyclist": 1.0, "barrier": 1.0})
    lidar_only_weight: float = 1.0
    camera_only_weight: float = 0.5  # a camera box LiDAR did not confirm is more likely a depth error


def fuse_sample(camera: List[Box], lidar: List[Box], cfg: FusionConfig = FusionConfig()) -> List[Box]:
    out: List[Box] = []
    for cls in sorted({b.name for b in camera} | {b.name for b in lidar}):
        cam = [b for b in camera if b.name == cls]
        lid = sorted((b for b in lidar if b.name == cls), key=lambda b: -b.score)
        used = set()
        for lb in lid:
            best, best_d = -1, cfg.radius.get(cls, 1.0)
            for j, cb in enumerate(cam):
                if j in used:
                    continue
                d = float(np.linalg.norm(cb.translation[:2] - lb.translation[:2]))
                if d < best_d:
                    best, best_d = j, d
            fused = Box(**{**lb.__dict__})
            if best >= 0:
                used.add(best)
                fused.score = 1.0 - (1.0 - lb.score) * (1.0 - cam[best].score)
            else:
                fused.score = lb.score * cfg.lidar_only_weight
            out.append(fused)
        for j, cb in enumerate(cam):
            if j not in used:
                b = Box(**{**cb.__dict__})
                b.score = cb.score * cfg.camera_only_weight
                out.append(b)
    out.sort(key=lambda b: -b.score)
    return out[:MAX_BOXES_PER_SAMPLE]


def fuse_all(camera: Dict[str, List[Box]], lidar: Dict[str, List[Box]], cfg: FusionConfig = FusionConfig()):
    return {tok: fuse_sample(camera.get(tok, []), lidar.get(tok, []), cfg) for tok in set(camera) | set(lidar)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="GT file (only used for the sample list and ego poses)")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--lidar", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    samples = load_gt(args.gt)
    fused = fuse_all(load_predictions(args.camera, samples), load_predictions(args.lidar, samples))
    save_predictions(args.out, fused, {"method": "late fusion (LiDAR geometry, noisy-OR scores)",
                                       "config": FusionConfig().__dict__})
    print(f"Fused {len(fused)} samples -> {args.out}")


if __name__ == "__main__":
    main()
