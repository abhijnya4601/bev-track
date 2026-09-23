"""Box representation, frame transforms and BEV geometry.

Convention (same as the nuScenes devkit):
  * ``translation`` is the box centre (x, y, z) in metres.
  * ``size`` is (w, l, h): width across the heading, length along the heading, height.
  * ``yaw`` is the heading angle around +z, in radians, measured from +x.
  * ``velocity`` is (vx, vy) in m/s.

Every box inside the eval harness lives in the **ego frame of the sample's LIDAR_TOP
timestamp**. The tracker is the only component that works in the global frame, because a
constant-velocity motion model is meaningless in a frame that moves with the car.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
from pyquaternion import Quaternion


@dataclass
class Box:
    sample_token: str
    translation: np.ndarray  # (3,)
    size: np.ndarray  # (3,) w, l, h
    yaw: float
    name: str
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    score: float = -1.0  # predictions only
    instance_id: Optional[str] = None  # GT: nuScenes instance token. Tracks: track id.
    num_pts: int = -1  # GT only: lidar + radar points inside the box

    def __post_init__(self):
        self.translation = np.asarray(self.translation, dtype=float).reshape(3)
        self.size = np.asarray(self.size, dtype=float).reshape(3)
        # NaN is kept on purpose: nuScenes GT velocity is NaN when it can't be estimated, and the devkit
        # ignores those matches in mAVE rather than scoring them as zero velocity.
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(-1)[:2]
        self.yaw = float(self.yaw)

    @property
    def ego_dist(self) -> float:
        """xy distance from the ego origin. Only meaningful for ego-frame boxes."""
        return float(np.hypot(self.translation[0], self.translation[1]))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["translation"] = self.translation.tolist()
        d["size"] = self.size.tolist()
        d["velocity"] = self.velocity.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(**d)


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def yaw_of(q: Quaternion) -> float:
    """Heading of a quaternion around +z (same formula as nuscenes.eval.common.utils.quaternion_yaw)."""
    v = np.dot(q.rotation_matrix, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(v[1], v[0]))


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def transform_box(box: Box, rotation: Quaternion, translation: Sequence[float]) -> Box:
    """Apply a rigid transform ``p' = R p + t`` to a box (centre, heading and velocity)."""
    R = rotation.rotation_matrix
    new = Box(**{**box.__dict__})
    new.translation = R @ box.translation + np.asarray(translation, dtype=float)
    v3 = R @ np.array([box.velocity[0], box.velocity[1], 0.0])
    new.velocity = v3[:2]
    # Compose full rotations rather than adding yaws: ego poses have small pitch/roll.
    new.yaw = yaw_of(rotation * Quaternion(axis=[0.0, 0.0, 1.0], radians=box.yaw))
    return new


def inverse_transform_box(box: Box, rotation: Quaternion, translation: Sequence[float]) -> Box:
    """Inverse of :func:`transform_box`: ``p = R^T (p' - t)``."""
    inv = rotation.inverse
    t = -(inv.rotation_matrix @ np.asarray(translation, dtype=float))
    return transform_box(box, inv, t)


def ego_to_global(box: Box, ego_pose: dict) -> Box:
    return transform_box(box, Quaternion(ego_pose["rotation"]), ego_pose["translation"])


def global_to_ego(box: Box, ego_pose: dict) -> Box:
    return inverse_transform_box(box, Quaternion(ego_pose["rotation"]), ego_pose["translation"])


# ---------------------------------------------------------------------------
# BEV geometry
# ---------------------------------------------------------------------------

def bev_corners(center_xy, size_wl, yaw) -> np.ndarray:
    """Four BEV corners (counter-clockwise) of a box. ``size_wl`` is (w, l); l lies along the heading."""
    w, l = float(size_wl[0]), float(size_wl[1])
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[l / 2, w / 2], [-l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.asarray(center_xy, dtype=float)[:2]


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a polygon against a convex CCW polygon."""
    out = list(subject)
    n = len(clipper)
    for i in range(n):
        if not out:
            break
        a, b = clipper[i], clipper[(i + 1) % n]
        inp, out = out, []
        edge = b - a

        def inside(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12

        def intersect(p, q):
            d = q - p
            denom = edge[0] * d[1] - edge[1] * d[0]
            if abs(denom) < 1e-15:
                return p
            t = (edge[1] * (p[0] - a[0]) - edge[0] * (p[1] - a[1])) / denom
            return p + t * d

        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out.append(intersect(p, q))
                out.append(q)
            elif inside(p):
                out.append(intersect(p, q))
    return np.array(out) if out else np.zeros((0, 2))


def bev_iou(a: Box, b: Box) -> float:
    """Rotated-rectangle IoU in the ground plane."""
    pa = bev_corners(a.translation, a.size[:2], a.yaw)
    pb = bev_corners(b.translation, b.size[:2], b.yaw)
    inter = _polygon_area(_clip(pa, pb))
    union = a.size[0] * a.size[1] + b.size[0] * b.size[1] - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_box(point, box: Box) -> bool:
    """Whether a 3D point lies inside a (yaw-only rotated) box."""
    d = np.asarray(point, dtype=float)[:3] - box.translation
    c, s = np.cos(-box.yaw), np.sin(-box.yaw)
    x = c * d[0] - s * d[1]
    y = s * d[0] + c * d[1]
    w, l, h = box.size
    return abs(x) <= l / 2 and abs(y) <= w / 2 and abs(d[2]) <= h / 2
