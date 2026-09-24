"""Synthetic scenes and detector outputs for testing the harness without nuScenes.

    python -m src.synthetic --out-dir data/synthetic

The generated "detector" has known, controllable failure modes: miss rate grows with distance,
localization noise grows with distance, and night scenes are harder. So the slicing harness
has something real to find. These numbers are **not results**. They exist to smoke-test the
pipeline.
"""
import argparse
import os
from typing import Dict, List

import numpy as np
from pyquaternion import Quaternion

from src.boxes import Box, global_to_ego
from src.classes import CLASSES
from src.dataio import Sample, save_gt, save_predictions

_SIZES = {"car": (1.9, 4.6, 1.7), "pedestrian": (0.7, 0.7, 1.8), "cyclist": (0.6, 1.8, 1.5),
          "truck": (2.5, 7.0, 3.0), "barrier": (2.5, 0.5, 1.0)}
_SPEED = {"car": 6.0, "pedestrian": 1.3, "cyclist": 4.0, "truck": 5.0, "barrier": 0.0}


def make_samples(n_scenes: int = 4, n_frames: int = 20, n_objects: int = 30, seed: int = 0) -> Dict[str, Sample]:
    rng = np.random.default_rng(seed)
    samples: Dict[str, Sample] = {}
    for sc in range(n_scenes):
        scene = f"synth-{sc:04d}"
        lighting = "night" if sc % 2 else "day"
        weather = "rain" if sc % 4 == 3 else "dry"
        ego_speed, ego_yaw0 = rng.uniform(3, 8), rng.uniform(-np.pi, np.pi)
        # objects in global frame at t=0
        objs = []
        for k in range(n_objects):
            cls = CLASSES[rng.choice(len(CLASSES), p=[0.4, 0.25, 0.1, 0.1, 0.15])]
            r, th = rng.uniform(3, 55), rng.uniform(-np.pi, np.pi)
            yaw = rng.uniform(-np.pi, np.pi)
            v = _SPEED[cls] * rng.uniform(0, 1) * np.array([np.cos(yaw), np.sin(yaw)])
            size = np.array(_SIZES[cls]) * rng.uniform(0.9, 1.1, 3)
            objs.append(dict(cls=cls, p0=np.array([r * np.cos(th), r * np.sin(th), 0.8]), yaw=yaw, v=v, size=size,
                             id=f"{scene}-obj{k}", pts=int(rng.integers(0, 200))))
        for f in range(n_frames):
            t = 0.5 * f
            ts = 1_600_000_000_000_000 + sc * 100_000_000 + int(t * 1e6)
            ego_xy = ego_speed * t * np.array([np.cos(ego_yaw0), np.sin(ego_yaw0)])
            pose = {"translation": [ego_xy[0], ego_xy[1], 0.0],
                    "rotation": list(Quaternion(axis=[0, 0, 1], radians=ego_yaw0).elements)}
            tok = f"{scene}-f{f:03d}"
            gt = []
            for o in objs:
                p = o["p0"] + np.array([*(o["v"] * t), 0.0])
                g = Box(sample_token=tok, translation=p, size=o["size"], yaw=o["yaw"], name=o["cls"],
                        velocity=o["v"], instance_id=o["id"], num_pts=o["pts"])
                gt.append(global_to_ego(g, pose))
            samples[tok] = Sample(tok, f"scene-tok-{sc}", scene, ts, pose,
                                  {"lighting": lighting, "weather": weather, "location": "synthetic"}, gt)
    return samples


def make_predictions(samples: Dict[str, Sample], seed: int = 1, base_miss: float = 0.05, fp_per_frame: float = 3.0
                     ) -> Dict[str, List[Box]]:
    rng = np.random.default_rng(seed)
    classes = list(CLASSES)
    preds: Dict[str, List[Box]] = {}
    for tok, s in samples.items():
        night = s.condition.get("lighting") == "night"
        out = []
        for g in s.gt:
            d = g.ego_dist
            miss = base_miss + 0.012 * d + (0.15 if night else 0.0) + (0.1 if g.name in ("pedestrian", "cyclist") else 0)
            if rng.random() < miss:
                continue
            sigma = 0.1 + 0.02 * d + (0.2 if night else 0.0)
            name = g.name if rng.random() > 0.05 else classes[rng.integers(len(classes))]
            yaw = g.yaw + rng.normal(0, 0.1) + (np.pi if rng.random() < 0.05 else 0.0)
            score = float(np.clip(0.95 - 0.01 * d - (0.1 if night else 0) + rng.normal(0, 0.1), 0.05, 1.0))
            out.append(Box(tok, g.translation + np.r_[rng.normal(0, sigma, 2), 0.0], g.size * rng.normal(1, 0.05, 3),
                           yaw, name, g.velocity + rng.normal(0, 0.5, 2), score=score))
        for _ in range(rng.poisson(fp_per_frame * (1.5 if night else 1.0))):
            r, th = rng.uniform(3, 50), rng.uniform(-np.pi, np.pi)
            cls = classes[rng.integers(len(classes))]
            out.append(Box(tok, [r * np.cos(th), r * np.sin(th), 0.8], _SIZES[cls], rng.uniform(-np.pi, np.pi), cls,
                           score=float(rng.uniform(0.05, 0.6))))
        preds[tok] = out
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    samples = make_samples(seed=args.seed)
    preds = make_predictions(samples, seed=args.seed + 1)
    save_gt(os.path.join(args.out_dir, "gt.json"), samples, {"version": "synthetic", "split": "synthetic"})
    save_predictions(os.path.join(args.out_dir, "preds.json"), preds, {"model": "synthetic-noisy-oracle"})
    print(f"Wrote {len(samples)} samples to {args.out_dir}/gt.json and preds.json")


if __name__ == "__main__":
    main()
