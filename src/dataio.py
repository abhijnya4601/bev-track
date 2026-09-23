"""Reading and writing the two files the harness runs on: a GT file and a predictions file.

GT file (written by ``data/prepare_nuscenes.py``)::

    {"meta": {"version": "v1.0-mini", "split": "mini_val", "frame": "ego"},
     "samples": [{"token", "scene_token", "scene_name", "timestamp",
                  "condition": {"lighting": "day"|"night", "weather": "dry"|"rain", "location": ...},
                  "ego_pose": {"translation": [...], "rotation": [w, x, y, z]},
                  "boxes": [Box dicts, ego frame],
                  "bike_racks": [Box dicts, ego frame]}, ...]}

Predictions file, either of:
  * BEV-Track format: ``{"meta": {"frame": "ego"}, "results": {sample_token: [Box dicts]}}``
  * the official nuScenes submission format (global frame, quaternion ``rotation``,
    ``detection_name``/``detection_score``), which is what BEVFormer's own test script writes.
    It is converted to the ego frame and the 5-class taxonomy on load.
"""
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from pyquaternion import Quaternion

from src.boxes import Box, global_to_ego, point_in_box, yaw_of
from src.classes import CLASS_RANGE, CLASSES, detection_name_to_class

MAX_BOXES_PER_SAMPLE = 500  # devkit limit


@dataclass
class Sample:
    token: str
    scene_token: str
    scene_name: str
    timestamp: int
    ego_pose: dict
    condition: Dict[str, str] = field(default_factory=dict)
    gt: List[Box] = field(default_factory=list)
    bike_racks: List[Box] = field(default_factory=list)


def load_gt(path: str) -> Dict[str, Sample]:
    """Load a GT file. Returned dict is ordered by (scene, timestamp), which the tracker relies on."""
    with open(path) as f:
        raw = json.load(f)
    samples = []
    for s in raw["samples"]:
        samples.append(Sample(
            token=s["token"], scene_token=s["scene_token"], scene_name=s["scene_name"],
            timestamp=int(s["timestamp"]), ego_pose=s["ego_pose"], condition=s.get("condition", {}),
            gt=[Box.from_dict({**b, "sample_token": s["token"]}) for b in s["boxes"]],
            bike_racks=[Box.from_dict({**b, "sample_token": s["token"]}) for b in s.get("bike_racks", [])],
        ))
    samples.sort(key=lambda x: (x.scene_name, x.timestamp))
    return {s.token: s for s in samples}


def save_gt(path: str, samples: Dict[str, Sample], meta: dict) -> None:
    out = {"meta": {**meta, "frame": "ego"}, "samples": []}
    for s in samples.values():
        out["samples"].append({
            "token": s.token, "scene_token": s.scene_token, "scene_name": s.scene_name,
            "timestamp": s.timestamp, "ego_pose": s.ego_pose, "condition": s.condition,
            "boxes": [_strip(b) for b in s.gt], "bike_racks": [_strip(b) for b in s.bike_racks],
        })
    with open(path, "w") as f:
        json.dump(out, f)


def _strip(b: Box) -> dict:
    d = b.to_dict()
    d.pop("sample_token")
    return d


def save_predictions(path: str, preds: Dict[str, List[Box]], meta: Optional[dict] = None) -> None:
    out = {"meta": {**(meta or {}), "frame": "ego"},
           "results": {tok: [_strip(b) for b in boxes] for tok, boxes in preds.items()}}
    with open(path, "w") as f:
        json.dump(out, f)


def load_predictions(path: str, samples: Dict[str, Sample]) -> Dict[str, List[Box]]:
    """Load predictions in either supported format, returning ego-frame, 5-class boxes per sample.

    Predictions whose class has no 5-class equivalent (bus, trailer, cone) are dropped.
    """
    with open(path) as f:
        raw = json.load(f)
    results = raw["results"]
    is_nusc_format = any("detection_name" in b for boxes in results.values() for b in boxes)
    frame = "global" if is_nusc_format else raw.get("meta", {}).get("frame", "ego")

    missing = set(samples) - set(results)
    if missing:
        warnings.warn(f"{len(missing)} GT samples have no prediction entry; treating them as empty.")

    out: Dict[str, List[Box]] = {tok: [] for tok in samples}
    for tok, boxes in results.items():
        if tok not in samples:
            continue
        if len(boxes) > MAX_BOXES_PER_SAMPLE:
            raise ValueError(f"Sample {tok} has {len(boxes)} predictions; max is {MAX_BOXES_PER_SAMPLE}.")
        for b in boxes:
            box = _parse_nusc_box(tok, b) if is_nusc_format else Box.from_dict({**b, "sample_token": tok})
            box.name = detection_name_to_class(box.name)
            if box.name is None:
                continue
            if frame == "global":
                box = global_to_ego(box, samples[tok].ego_pose)
            out[tok].append(box)
    return out


def _parse_nusc_box(tok: str, b: dict) -> Box:
    return Box(sample_token=tok, translation=b["translation"], size=b["size"],
               yaw=yaw_of(Quaternion(b["rotation"])), name=b["detection_name"],
               velocity=b.get("velocity", [0.0, 0.0]), score=float(b["detection_score"]))


def filter_boxes(boxes: List[Box], sample: Sample, is_gt: bool, use_class_range: bool = True) -> List[Box]:
    """Devkit-equivalent filtering: class range, zero-point GT, and cyclists inside bike racks."""
    kept = []
    for b in boxes:
        if b.name not in CLASSES:
            continue
        if use_class_range and not b.ego_dist < CLASS_RANGE[b.name]:
            continue
        if is_gt and b.num_pts == 0:
            continue
        if b.name == "cyclist" and any(point_in_box(b.translation, r) for r in sample.bike_racks):
            continue
        kept.append(b)
    return kept


def prepare_eval_boxes(samples: Dict[str, Sample], preds: Dict[str, List[Box]], use_class_range: bool = True):
    """Apply :func:`filter_boxes` to every sample. Returns (gt, preds) as flat lists."""
    gt_all, pred_all = [], []
    for tok, s in samples.items():
        gt_all.extend(filter_boxes(s.gt, s, is_gt=True, use_class_range=use_class_range))
        pred_all.extend(filter_boxes(preds.get(tok, []), s, is_gt=False, use_class_range=use_class_range))
    return gt_all, pred_all


def as_array(boxes: List[Box], attr: str) -> np.ndarray:
    return np.array([getattr(b, attr) for b in boxes])
