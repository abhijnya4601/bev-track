"""CLEAR-MOT tracking metrics: MOTA, MOTP, ID switches, fragmentation, mostly tracked/lost.

These are implemented here instead of calling ``motmetrics`` directly so every counting rule
is visible and testable. ``tests/test_tracking.py`` cross-checks the numbers against
motmetrics when it is installed. The matching rules follow motmetrics:

  1. each GT's most recent correspondence (from any earlier frame, not just the previous one) is
     re-established first if both are present and within ``dist_th``. This keeps a track from being
     stolen by a slightly closer newcomer, including right after a missed frame
  2. the remaining GT and hypotheses are matched with the Hungarian algorithm on center distance
  3. an **ID switch** is counted when a GT is matched to a hypothesis other than the one it was
     last matched to (at any earlier frame)
  4. a **fragmentation** is counted each time a GT trajectory goes from tracked to untracked
     between its first and last tracked frame

Ground truth is filtered exactly like detection GT (class range, zero-point boxes).
"""
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.boxes import Box
from src.classes import CLASSES
from src.dataio import Sample, filter_boxes

Frame = Tuple[List[str], np.ndarray, List[str], np.ndarray]  # gt ids, gt xy (n,2), hyp ids, hyp xy (m,2)


@dataclass
class MOTCounts:
    num_frames: int = 0
    num_gt: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    id_switches: int = 0
    fragmentations: int = 0
    dist_sum: float = 0.0
    mostly_tracked: int = 0
    mostly_lost: int = 0
    num_objects: int = 0
    num_tracks: int = 0

    def __iadd__(self, o: "MOTCounts"):
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self

    @property
    def mota(self) -> float:
        return 1.0 - (self.fn + self.fp + self.id_switches) / self.num_gt if self.num_gt else np.nan

    @property
    def motp(self) -> float:
        return self.dist_sum / self.tp if self.tp else np.nan

    def row(self) -> dict:
        return {"MOTA": self.mota, "MOTP_m": self.motp, "IDSW": self.id_switches, "FRAG": self.fragmentations,
                "TP": self.tp, "FP": self.fp, "FN": self.fn, "GT": self.num_gt, "MT": self.mostly_tracked,
                "ML": self.mostly_lost, "gt_objects": self.num_objects, "tracks": self.num_tracks,
                "frames": self.num_frames}


def clear_mot(frames: Sequence[Frame], dist_th: float = 2.0) -> MOTCounts:
    """CLEAR-MOT counts for one sequence (one scene, one class)."""
    c = MOTCounts()
    last_match: Dict[str, str] = {}
    history: Dict[str, List[bool]] = {}
    hyp_seen = set()
    for gt_ids, gt_xy, hyp_ids, hyp_xy in frames:
        c.num_frames += 1
        c.num_gt += len(gt_ids)
        hyp_seen.update(hyp_ids)
        gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
        hyp_xy = np.asarray(hyp_xy, dtype=float).reshape(-1, 2)
        D = np.linalg.norm(gt_xy[:, None, :] - hyp_xy[None, :, :], axis=2) if len(gt_ids) and len(hyp_ids) \
            else np.zeros((len(gt_ids), len(hyp_ids)))
        valid = D < dist_th
        gi_of = {g: i for i, g in enumerate(gt_ids)}
        hi_of = {h: j for j, h in enumerate(hyp_ids)}
        matches: Dict[str, str] = {}

        # 1. re-establish each GT's most recent correspondence if still valid. Two GTs can share a
        #    last hypothesis (after an id switch), so iterate in frame order like motmetrics.
        for g in gt_ids:
            h = last_match.get(g)
            if h in hi_of and h not in matches.values() and valid[gi_of[g], hi_of[h]]:
                matches[g] = h
        # 2. Hungarian on the rest
        free_g = [i for i, g in enumerate(gt_ids) if g not in matches]
        used_h = set(matches.values())
        free_h = [j for j, h in enumerate(hyp_ids) if h not in used_h]
        if free_g and free_h:
            sub = D[np.ix_(free_g, free_h)]
            cost = np.where(valid[np.ix_(free_g, free_h)], sub, 1e6)
            for r, col in zip(*linear_sum_assignment(cost)):
                if cost[r, col] < 1e6:
                    g, h = gt_ids[free_g[r]], hyp_ids[free_h[col]]
                    # 3. ID switch
                    if g in last_match and last_match[g] != h:
                        c.id_switches += 1
                    matches[g] = h

        for g, h in matches.items():
            c.dist_sum += D[gi_of[g], hi_of[h]]
            last_match[g] = h
        c.tp += len(matches)
        c.fn += len(gt_ids) - len(matches)
        c.fp += len(hyp_ids) - len(matches)
        for g in gt_ids:
            history.setdefault(g, []).append(g in matches)

    # 4. fragmentation, MT/ML
    for g, h in history.items():
        tracked = np.array(h)
        c.num_objects += 1
        ratio = tracked.mean()
        c.mostly_tracked += int(ratio >= 0.8)
        c.mostly_lost += int(ratio < 0.2)
        idx = np.nonzero(tracked)[0]
        if len(idx):
            seg = tracked[idx[0]: idx[-1] + 1].astype(int)
            c.fragmentations += int(np.sum(np.diff(seg) == -1))
    c.num_tracks = len(hyp_seen)
    return c


@dataclass
class TrackingResult:
    per_class: Dict[str, MOTCounts] = field(default_factory=dict)
    per_scene: Dict[str, MOTCounts] = field(default_factory=dict)

    @property
    def overall(self) -> MOTCounts:
        tot = MOTCounts()
        for c in self.per_class.values():
            tot += c
        return tot


def build_frames(samples: Dict[str, Sample], tracks: Dict[str, List[Box]], cls: str, scene: str) -> List[Frame]:
    seq = sorted((s for s in samples.values() if s.scene_name == scene), key=lambda s: s.timestamp)
    frames = []
    for s in seq:
        gts = [b for b in filter_boxes(s.gt, s, is_gt=True) if b.name == cls]
        hyps = [b for b in filter_boxes(tracks.get(s.token, []), s, is_gt=False) if b.name == cls]
        frames.append(([b.instance_id for b in gts], np.array([b.translation[:2] for b in gts]).reshape(-1, 2),
                       [b.instance_id for b in hyps], np.array([b.translation[:2] for b in hyps]).reshape(-1, 2)))
    return frames


def evaluate_tracking(samples: Dict[str, Sample], tracks: Dict[str, List[Box]], dist_th: float = 2.0,
                      classes: Sequence[str] = CLASSES) -> TrackingResult:
    res = TrackingResult()
    scenes = sorted({s.scene_name for s in samples.values()})
    for cls in classes:
        res.per_class[cls] = MOTCounts()
        for scene in scenes:
            counts = clear_mot(build_frames(samples, tracks, cls, scene), dist_th)
            res.per_class[cls] += counts
            res.per_scene.setdefault(scene, MOTCounts())
            res.per_scene[scene] += counts
    return res


def print_tracking_table(res: TrackingResult) -> None:
    print(f"{'':<14}{'MOTA':>8}{'MOTP':>8}{'IDSW':>6}{'FRAG':>6}{'FP':>7}{'FN':>7}{'GT':>7}{'MT':>5}{'ML':>5}")
    rows = list(res.per_class.items()) + [("ALL", res.overall)] + [(f"scene {k}", v) for k, v in res.per_scene.items()]
    for name, c in rows:
        print(f"{name:<14}{c.mota:8.3f}{c.motp:8.3f}{c.id_switches:6d}{c.fragmentations:6d}"
              f"{c.fp:7d}{c.fn:7d}{c.num_gt:7d}{c.mostly_tracked:5d}{c.mostly_lost:5d}")


def write_tracking_csv(res: TrackingResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [{"slice": "class", "value": k, **v.row()} for k, v in res.per_class.items()]
    rows.append({"slice": "all", "value": "all", **res.overall.row()})
    rows += [{"slice": "scene", "value": k, **v.row()} for k, v in res.per_scene.items()]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
