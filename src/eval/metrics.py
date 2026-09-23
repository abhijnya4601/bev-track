"""nuScenes-style detection metrics (AP by center distance, TP errors, NDS) that support slicing.

Why not just call ``nuscenes.eval.detection.evaluate.DetectionEval``?
  * It hard-codes the 10 nuScenes detection classes (we use 5 merged ones).
  * It can only evaluate a whole split. Slicing by distance or condition means filtering GT and
    predictions separately, and that creates spurious false positives and false negatives at
    slice boundaries (a prediction at 19.8 m matched to a GT at 20.3 m becomes a FP in one bucket
    and a FN in the other).

So matching is done **once**, globally, exactly as the devkit does it. A slice is then a
subset of GT plus a subset of predictions, with this rule: *a matched prediction belongs to
the slice iff its matched GT does; an unmatched prediction belongs to the slice by its own
attributes.* Slices that partition the GT therefore also partition the predictions, and the
unsliced numbers equal the devkit's. ``tests/test_eval_pipeline.py`` checks this equality
against the devkit's own ``accumulate``/``calc_ap``/``calc_tp``.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.boxes import Box
from src.classes import CLASSES, ORIENTATION_PERIOD, UNDEFINED_TP_METRICS

DIST_THRESHOLDS: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
TP_DIST_THRESHOLD = 2.0
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
N_ELEM = 101
TP_METRICS = ("trans_err", "scale_err", "orient_err", "vel_err")
MEAN_AP_WEIGHT = 5
TP_METRIC_SHORT = {"trans_err": "mATE", "scale_err": "mASE", "orient_err": "mAOE", "vel_err": "mAVE"}


# ---------------------------------------------------------------------------
# Per-match error functions (devkit definitions)
# ---------------------------------------------------------------------------

def center_distance(a: Box, b: Box) -> float:
    return float(np.linalg.norm(a.translation[:2] - b.translation[:2]))


def scale_iou(a: Box, b: Box) -> float:
    """IoU of two boxes after aligning their centres and headings (so it only measures size)."""
    inter = np.prod(np.minimum(a.size, b.size))
    union = np.prod(a.size) + np.prod(b.size) - inter
    return float(inter / union)


def angle_diff(x: float, y: float, period: float) -> float:
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - 2 * np.pi
    return diff


def yaw_error(gt: Box, pred: Box, period: float) -> float:
    return abs(angle_diff(gt.yaw, pred.yaw, period))


def tp_errors(gt: Box, pred: Box) -> Dict[str, float]:
    return {
        "trans_err": center_distance(gt, pred),
        "scale_err": 1.0 - scale_iou(gt, pred),
        "orient_err": yaw_error(gt, pred, ORIENTATION_PERIOD[gt.name]),
        "vel_err": float(np.linalg.norm(pred.velocity - gt.velocity)),
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class Matches:
    """Greedy devkit matching of one class's predictions at one distance threshold.

    Arrays are aligned and in processing order (descending score).
    """
    pred_idx: np.ndarray  # index into the flat prediction list
    gt_idx: np.ndarray  # matched index into the flat GT list, -1 for a false positive
    scores: np.ndarray
    errors: Dict[str, np.ndarray]  # NaN for false positives

    @property
    def is_tp(self) -> np.ndarray:
        return self.gt_idx >= 0


def _descending_order(scores: Sequence[float]) -> List[int]:
    # Identical tie-breaking to the devkit: sort by (score, index) ascending, then reverse.
    return [i for (_, i) in sorted((v, i) for i, v in enumerate(scores))][::-1]


def match(gt: List[Box], preds: List[Box], thresholds: Sequence[float] = DIST_THRESHOLDS,
          classes: Sequence[str] = CLASSES) -> Dict[Tuple[str, float], Matches]:
    """Run greedy center-distance matching for every (class, threshold)."""
    out = {}
    for cls in classes:
        gt_by_sample: Dict[str, List[int]] = {}
        for i, g in enumerate(gt):
            if g.name == cls:
                gt_by_sample.setdefault(g.sample_token, []).append(i)
        p_idx = [i for i, p in enumerate(preds) if p.name == cls]
        order = [p_idx[k] for k in _descending_order([preds[i].score for i in p_idx])]

        for th in thresholds:
            taken = set()
            matched = np.full(len(order), -1, dtype=int)
            errs = {m: np.full(len(order), np.nan) for m in TP_METRICS}
            for k, pi in enumerate(order):
                pred = preds[pi]
                best, best_d = -1, np.inf
                for gi in gt_by_sample.get(pred.sample_token, []):
                    if gi in taken:
                        continue
                    d = center_distance(gt[gi], pred)
                    if d < best_d:
                        best, best_d = gi, d
                if best_d < th:
                    taken.add(best)
                    matched[k] = best
                    for m, v in tp_errors(gt[best], pred).items():
                        errs[m][k] = v
            out[(cls, th)] = Matches(np.array(order, dtype=int), matched,
                                     np.array([preds[i].score for i in order], dtype=float), errs)
    return out


# ---------------------------------------------------------------------------
# PR curve, AP, TP errors
# ---------------------------------------------------------------------------

@dataclass
class MetricData:
    precision: np.ndarray
    recall: np.ndarray
    confidence: np.ndarray
    errors: Dict[str, np.ndarray]

    @property
    def max_recall_ind(self) -> int:
        nz = np.nonzero(self.confidence)[0]
        return int(nz[-1]) if len(nz) else 0

    @classmethod
    def no_predictions(cls) -> "MetricData":
        return cls(precision=np.zeros(N_ELEM), recall=np.linspace(0, 1, N_ELEM),
                   confidence=np.zeros(N_ELEM), errors={m: np.ones(N_ELEM) for m in TP_METRICS})


def _cummean(x: np.ndarray) -> np.ndarray:
    if np.all(np.isnan(x)):
        return np.ones(len(x))
    s = np.nancumsum(x.astype(float))
    c = np.cumsum(~np.isnan(x))
    return np.divide(s, c, out=np.zeros_like(s), where=c != 0)


def accumulate(scores: np.ndarray, is_tp: np.ndarray, errors: Dict[str, np.ndarray], n_gt: int) -> MetricData:
    """PR curve + recall-resampled TP errors from matched predictions (in descending-score order)."""
    if n_gt == 0 or not np.any(is_tp):
        return MetricData.no_predictions()
    tp = np.cumsum(is_tp).astype(float)
    fp = np.cumsum(~is_tp).astype(float)
    prec = tp / (tp + fp)
    rec = tp / float(n_gt)
    rec_interp = np.linspace(0, 1, N_ELEM)
    prec_i = np.interp(rec_interp, rec, prec, right=0)
    conf_i = np.interp(rec_interp, rec, scores, right=0)
    tp_conf = scores[is_tp]
    errs = {}
    for m in TP_METRICS:
        tmp = _cummean(errors[m][is_tp])
        errs[m] = np.interp(conf_i[::-1], tp_conf[::-1], tmp[::-1])[::-1]
    return MetricData(prec_i, rec_interp, conf_i, errs)


def calc_ap(md: MetricData, min_recall: float = MIN_RECALL, min_precision: float = MIN_PRECISION) -> float:
    prec = np.copy(md.precision)[round(100 * min_recall) + 1:]
    prec -= min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: MetricData, metric: str, min_recall: float = MIN_RECALL) -> float:
    first, last = round(100 * min_recall) + 1, md.max_recall_ind
    if last < first:
        return 1.0
    return float(np.mean(md.errors[metric][first:last + 1]))


# ---------------------------------------------------------------------------
# Full evaluation of one slice
# ---------------------------------------------------------------------------

@dataclass
class ClassResult:
    n_gt: int
    n_pred: int
    ap_by_threshold: Dict[float, float]
    ap: float
    tp_errors: Dict[str, float]


@dataclass
class DetectionResult:
    per_class: Dict[str, ClassResult]
    mean_ap: float
    mean_tp_errors: Dict[str, float]
    nds: float
    extra: dict = field(default_factory=dict)

    def summary_row(self) -> dict:
        row = {"mAP": self.mean_ap, "NDS": self.nds}
        row.update({TP_METRIC_SHORT[m]: v for m, v in self.mean_tp_errors.items()})
        return row


def evaluate(gt: List[Box], preds: List[Box], matches: Optional[Dict] = None,
             gt_mask: Optional[np.ndarray] = None, pred_mask: Optional[np.ndarray] = None,
             classes: Sequence[str] = CLASSES) -> DetectionResult:
    """Evaluate one slice.

    ``gt_mask``/``pred_mask`` select the slice (default: everything). ``matches`` can be passed
    in so many slices share one global matching. Classes with no GT in the slice get NaN AP
    and are excluded from mAP (the devkit would score them 0, which makes sparse slices
    meaningless).

    NDS follows the devkit formula with mAAE fixed at 1, since we predict no attributes.
    That is also what the devkit itself gives when attributes are missing.
    """
    if matches is None:
        matches = match(gt, preds, classes=classes)
    gt_mask = np.ones(len(gt), bool) if gt_mask is None else np.asarray(gt_mask, bool)
    pred_mask = np.ones(len(preds), bool) if pred_mask is None else np.asarray(pred_mask, bool)
    gt_names = np.array([g.name for g in gt])

    per_class = {}
    for cls in classes:
        n_gt = int(np.sum(gt_mask & (gt_names == cls))) if len(gt) else 0
        aps, n_pred, md_tp = {}, 0, None
        for (c, th), mt in matches.items():
            if c != cls:
                continue
            keep = np.where(mt.is_tp, gt_mask[np.maximum(mt.gt_idx, 0)] if len(gt) else False,
                            pred_mask[mt.pred_idx] if len(preds) else False)
            md = accumulate(mt.scores[keep], mt.is_tp[keep], {m: e[keep] for m, e in mt.errors.items()}, n_gt)
            aps[th] = calc_ap(md) if n_gt > 0 else np.nan
            if th == TP_DIST_THRESHOLD:
                md_tp, n_pred = md, int(keep.sum())
        errs = {}
        for m in TP_METRICS:
            if n_gt == 0 or m in UNDEFINED_TP_METRICS.get(cls, ()):
                errs[m] = np.nan
            else:
                errs[m] = calc_tp(md_tp, m)
        ap = float(np.mean(list(aps.values()))) if n_gt > 0 else np.nan
        per_class[cls] = ClassResult(n_gt, n_pred, aps, ap, errs)

    valid = [r for r in per_class.values() if r.n_gt > 0]
    mean_ap = float(np.mean([r.ap for r in valid])) if valid else np.nan
    mean_errs = {}
    for m in TP_METRICS:
        vals = [r.tp_errors[m] for r in valid if not np.isnan(r.tp_errors[m])]
        mean_errs[m] = float(np.mean(vals)) if vals else np.nan
    nds = compute_nds(mean_ap, mean_errs)
    return DetectionResult(per_class, mean_ap, mean_errs, nds)


def compute_nds(mean_ap: float, mean_errs: Dict[str, float], attr_err: float = 1.0) -> float:
    if np.isnan(mean_ap):
        return np.nan
    errs = [1.0 if np.isnan(e) else e for e in mean_errs.values()] + [attr_err]
    tp_scores = [max(1.0 - e, 0.0) for e in errs]
    return float((MEAN_AP_WEIGHT * mean_ap + np.sum(tp_scores)) / (MEAN_AP_WEIGHT + len(tp_scores)))


# ---------------------------------------------------------------------------
# Official devkit wrapper (10-class sanity check for the unmodified pretrained model)
# ---------------------------------------------------------------------------

def run_official_devkit_eval(result_path: str, dataroot: str, version: str = "v1.0-mini",
                             eval_set: str = "mini_val", output_dir: str = "results/devkit") -> dict:
    """Run the untouched nuScenes DetectionEval on a nuScenes-format result file.

    Use this for step 1 (pretrained checkpoint, its original 10 classes) to make sure the
    inference pipeline reproduces sensible official numbers before trusting anything custom.
    """
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    ev = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=result_path,
                       eval_set=eval_set, output_dir=output_dir, verbose=True)
    return ev.main(plot_examples=0, render_curves=False)
