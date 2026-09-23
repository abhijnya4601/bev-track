"""Top-down (BEV) plots of GT vs predicted boxes in the ego frame."""
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from src.boxes import Box, bev_corners  # noqa: E402

GT_COLOR = "#2a78d6"  # blue, solid (categorical slot 1)
PRED_COLOR = "#eb6834"  # orange, dashed (categorical slot 2)
HIGHLIGHT_COLOR = "#d0021b"  # red, thick


def draw_box(ax, box: Box, color: str, linestyle: str = "-", lw: float = 1.2, label: Optional[str] = None):
    corners = bev_corners(box.translation, box.size[:2], box.yaw)
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color, linestyle=linestyle, linewidth=lw))
    front = corners[[0, 3]].mean(axis=0)  # midpoint of the front edge: shows heading
    ax.plot([box.translation[0], front[0]], [box.translation[1], front[1]], color=color, lw=lw)
    if label:
        ax.text(corners[:, 0].max() + 0.3, corners[:, 1].max(), label, color=color, fontsize=6, clip_on=True)


def plot_bev(gt: Iterable[Box], preds: Iterable[Box], highlight: Optional[Box] = None, extent: float = 55.0,
             score_threshold: float = 0.3, title: str = "", ax=None, center=None, show_scores: bool = True):
    """Draw GT (blue, solid) and predictions (orange, dashed) around the ego vehicle.

    ``center`` zooms onto a point; ``extent`` is the half-width of the view in metres.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    cx, cy = (0.0, 0.0) if center is None else (float(center[0]), float(center[1]))
    for r in (20, 40):
        ax.add_patch(Circle((0, 0), r, fill=False, color="0.8", lw=0.8, ls=":"))
    ax.plot(0, 0, marker=(3, 0, -90), color="k", ms=10)  # ego, pointing +x

    for b in gt:
        draw_box(ax, b, GT_COLOR, "-", 1.2, b.name[:3] if center is not None else None)
    for p in preds:
        if p.score >= score_threshold:
            draw_box(ax, p, PRED_COLOR, "--", 1.0, f"{p.name[:3]} {p.score:.2f}" if show_scores and center is not None else None)
    if highlight is not None:
        draw_box(ax, highlight, HIGHLIGHT_COLOR, "-", 2.5)

    ax.set_xlim(cx - extent, cx + extent)
    ax.set_ylim(cy - extent, cy + extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.plot([], [], color=GT_COLOR, ls="-", label="ground truth")
    ax.plot([], [], color=PRED_COLOR, ls="--", label=f"prediction (score >= {score_threshold})")
    if highlight is not None:
        ax.plot([], [], color=HIGHLIGHT_COLOR, lw=2.5, label="failure case")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title, fontsize=9)
    return ax
