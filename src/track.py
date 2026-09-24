"""AB3DMOT-style 3D multi-object tracking on top of per-frame detections.

    python -m src.track --gt data/processed/gt_mini_val.json --pred preds.json \
        --out results/tracks.json

Classical tracking-by-detection (Weng et al., "AB3DMOT", IROS 2020):
  1. drop detections below ``score_threshold``
  2. Kalman-predict every live track forward by the real time gap between samples
  3. per class, build a cost matrix between predicted tracks and detections
     (BEV centre distance by default, or 1 - BEV IoU) and solve it with the Hungarian algorithm
  4. matched tracks get a Kalman update; unmatched detections start new tracks; tracks unseen for
     more than ``max_age`` frames are deleted
  5. a track is reported once it has ``min_hits`` hits (or during the first ``min_hits`` frames)

Tracking runs in the **global** frame: a constant-velocity model only makes sense in a frame
that does not move with the ego car. Output boxes are converted back to the ego frame.

Two deliberate deviations from vanilla AB3DMOT:
  * Center-distance gating per class is the default. nuScenes keyframes are 2 Hz, so a
    pedestrian box can fail to overlap its own previous box, and IoU association is harsh
    for small objects.
  * New tracks can take their initial velocity from the detection, since both BEVFormer and
    CenterPoint predict velocity. Vanilla AB3DMOT starts at zero velocity with a huge variance.
"""
import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.boxes import Box, bev_iou, ego_to_global, global_to_ego, wrap_angle
from src.dataio import Sample, load_gt, load_predictions, save_predictions

# state: x, y, z, yaw, w, l, h, vx, vy, vz
_DIM_X, _DIM_Z = 10, 7


@dataclass
class TrackerConfig:
    score_threshold: float = 0.3
    max_age: int = 2  # frames; 2 frames = 1 s at nuScenes' 2 Hz keyframe rate
    min_hits: int = 2
    cost: str = "center"  # "center" | "iou_bev"
    center_gate: Dict[str, float] = field(default_factory=lambda: {
        "car": 4.0, "truck": 4.0, "pedestrian": 1.5, "cyclist": 3.0, "barrier": 1.0})
    iou_threshold: float = 0.01
    init_velocity_from_detection: bool = True


class KalmanTrack:
    _next_id = 0

    def __init__(self, det: Box, cfg: TrackerConfig):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.name = det.name
        self.score = det.score
        self.x = np.zeros(_DIM_X)
        self.x[:3] = det.translation
        self.x[3] = det.yaw
        self.x[4:7] = det.size
        self.P = np.eye(_DIM_X) * 10.0
        if cfg.init_velocity_from_detection and np.all(np.isfinite(det.velocity)):
            self.x[7:9] = det.velocity
            self.P[7:, 7:] *= 10.0
        else:
            self.P[7:, 7:] *= 1000.0
        self.Q = np.eye(_DIM_X)
        self.Q[7:, 7:] *= 0.01
        self.R = np.eye(_DIM_Z)
        self.H = np.eye(_DIM_Z, _DIM_X)
        self.hits = 1
        self.time_since_update = 0
        self.age = 0

    def predict(self, dt: float) -> None:
        F = np.eye(_DIM_X)
        F[0, 7] = F[1, 8] = F[2, 9] = dt
        self.x = F @ self.x
        self.x[3] = wrap_angle(self.x[3])
        self.P = F @ self.P @ F.T + self.Q
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Box) -> None:
        # If the detection points the opposite way, flip the track's heading instead of letting the
        # filter average two headings 180 degrees apart (AB3DMOT's orientation correction).
        if abs(wrap_angle(det.yaw - self.x[3])) > np.pi / 2:
            self.x[3] = wrap_angle(self.x[3] + np.pi)
        z = np.concatenate([det.translation, [det.yaw], det.size])
        y = z - self.H @ self.x
        y[3] = wrap_angle(y[3])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])
        self.P = (np.eye(_DIM_X) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
        self.score = det.score

    def as_box(self, sample_token: str) -> Box:
        return Box(sample_token=sample_token, translation=self.x[:3], size=self.x[4:7], yaw=self.x[3],
                   name=self.name, velocity=self.x[7:9], score=self.score, instance_id=str(self.id))


class AB3DMOT:
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.tracks: List[KalmanTrack] = []
        self.frame_count = 0
        self.last_timestamp = None

    def _cost(self, trk: Box, det: Box) -> float:
        if self.cfg.cost == "iou_bev":
            return 1.0 - bev_iou(trk, det)
        return float(np.linalg.norm(trk.translation[:2] - det.translation[:2]))

    def _gate(self, name: str) -> float:
        return 1.0 - self.cfg.iou_threshold if self.cfg.cost == "iou_bev" else self.cfg.center_gate.get(name, 2.0)

    def step(self, dets: List[Box], timestamp_us: int, sample_token: str) -> List[Box]:
        """Process one frame of (global-frame) detections; return the confirmed tracks for this frame."""
        dt = 0.5 if self.last_timestamp is None else (timestamp_us - self.last_timestamp) * 1e-6
        self.last_timestamp = timestamp_us
        self.frame_count += 1
        dets = [d for d in dets if d.score >= self.cfg.score_threshold]

        for t in self.tracks:
            t.predict(dt)

        for cls in sorted({d.name for d in dets} | {t.name for t in self.tracks}):
            c_trk = [t for t in self.tracks if t.name == cls]
            c_det = [d for d in dets if d.name == cls]
            matched_d = set()
            if c_trk and c_det:
                pred_boxes = [t.as_box(sample_token) for t in c_trk]
                cost = np.array([[self._cost(pb, d) for d in c_det] for pb in pred_boxes])
                rows, cols = linear_sum_assignment(cost)
                gate = self._gate(cls)
                for r, c in zip(rows, cols):
                    if cost[r, c] < gate:
                        c_trk[r].update(c_det[c])
                        matched_d.add(c)
            for j, d in enumerate(c_det):
                if j not in matched_d:
                    self.tracks.append(KalmanTrack(d, self.cfg))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.cfg.max_age]
        return [t.as_box(sample_token) for t in self.tracks
                if t.time_since_update == 0 and (t.hits >= self.cfg.min_hits or self.frame_count <= self.cfg.min_hits)]


def track_all(samples: Dict[str, Sample], preds: Dict[str, List[Box]], cfg: TrackerConfig = TrackerConfig()
              ) -> Dict[str, List[Box]]:
    """Track every scene independently. Input and output boxes are in each sample's ego frame.

    Track ids are made globally unique as ``<scene_name>/<id>``.
    """
    by_scene: Dict[str, List[Sample]] = {}
    for s in samples.values():
        by_scene.setdefault(s.scene_name, []).append(s)
    out: Dict[str, List[Box]] = {}
    for scene, seq in by_scene.items():
        seq.sort(key=lambda s: s.timestamp)
        KalmanTrack._next_id = 0
        tracker = AB3DMOT(cfg)
        for s in seq:
            dets_global = [ego_to_global(d, s.ego_pose) for d in preds.get(s.token, [])]
            tracks = tracker.step(dets_global, s.timestamp, s.token)
            out[s.token] = []
            for t in tracks:
                b = global_to_ego(t, s.ego_pose)
                b.instance_id = f"{scene}/{t.instance_id}"
                out[s.token].append(b)
    return out


def main(argv=None):
    from src.eval.track_metrics import evaluate_tracking, print_tracking_table, write_tracking_csv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True, help="Per-frame detections (either supported format).")
    ap.add_argument("--out", default="results/tracks.json")
    ap.add_argument("--metrics-out", default="results/tracking_metrics.csv")
    ap.add_argument("--score-threshold", type=float, default=TrackerConfig.score_threshold)
    ap.add_argument("--max-age", type=int, default=TrackerConfig.max_age)
    ap.add_argument("--min-hits", type=int, default=TrackerConfig.min_hits)
    ap.add_argument("--cost", choices=["center", "iou_bev"], default=TrackerConfig.cost)
    ap.add_argument("--no-det-velocity", action="store_true")
    args = ap.parse_args(argv)

    cfg = TrackerConfig(score_threshold=args.score_threshold, max_age=args.max_age, min_hits=args.min_hits,
                        cost=args.cost, init_velocity_from_detection=not args.no_det_velocity)
    samples = load_gt(args.gt)
    preds = load_predictions(args.pred, samples)
    tracks = track_all(samples, preds, cfg)
    save_predictions(args.out, tracks, meta={"tracker": "AB3DMOT", "config": cfg.__dict__})
    res = evaluate_tracking(samples, tracks)
    print_tracking_table(res)
    write_tracking_csv(res, args.metrics_out)
    print(f"\nTracks -> {args.out}\nMetrics -> {args.metrics_out}")
    with open(args.metrics_out.replace(".csv", "_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)


if __name__ == "__main__":
    main()
