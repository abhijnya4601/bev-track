"""Break detection metrics down by class, distance bucket and scene condition.

    python -m src.eval.slice_eval --gt data/processed/gt_mini_val.json \
        --pred work_dirs/pretrained/results_nusc.json --out results/

Writes ``metrics_by_class.csv``, ``metrics_by_distance.csv``, ``metrics_by_condition.csv`` and
``summary.json``. Every row carries ``n_gt``/``n_pred`` so a reader can tell when a number rests on
a handful of objects. On nuScenes-mini that is most slices, and it is the first thing an
interviewer should ask about.
"""
import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.boxes import Box
from src.classes import CLASSES
from src.dataio import Sample, load_gt, load_predictions, prepare_eval_boxes
from src.eval.metrics import DIST_THRESHOLDS, TP_METRIC_SHORT, TP_METRICS, DetectionResult, evaluate, match

DISTANCE_EDGES: Tuple[float, ...] = (20.0, 40.0)


def distance_bucket(dist: float, edges: Sequence[float] = DISTANCE_EDGES) -> str:
    """Half-open buckets ``[lo, hi)``: 0-20m, 20-40m, 40m+. An object at exactly 20.0 m is in 20-40m."""
    if dist < 0 or np.isnan(dist):
        raise ValueError(f"invalid distance {dist}")
    lo = 0.0
    for hi in edges:
        if dist < hi:
            return f"{lo:g}-{hi:g}m"
        lo = hi
    return f"{lo:g}m+"


def bucket_names(edges: Sequence[float] = DISTANCE_EDGES) -> List[str]:
    names, lo = [], 0.0
    for hi in edges:
        names.append(f"{lo:g}-{hi:g}m")
        lo = hi
    return names + [f"{lo:g}m+"]


def slice_masks(gt: List[Box], preds: List[Box], samples: Dict[str, Sample]) -> Dict[Tuple[str, str], Tuple]:
    """All slices as ``(axis, value) -> (gt_mask, pred_mask)``."""
    out = {("all", "all"): (np.ones(len(gt), bool), np.ones(len(preds), bool))}

    g_bucket = np.array([distance_bucket(b.ego_dist) for b in gt])
    p_bucket = np.array([distance_bucket(b.ego_dist) for b in preds])
    for name in bucket_names():
        out[("distance", name)] = (g_bucket == name, p_bucket == name)

    axes = sorted({k for s in samples.values() for k in s.condition})
    for axis in axes:
        g_val = np.array([samples[b.sample_token].condition.get(axis, "unknown") for b in gt])
        p_val = np.array([samples[b.sample_token].condition.get(axis, "unknown") for b in preds])
        for v in sorted(set(g_val) | set(p_val)):
            out[(axis, v)] = (g_val == v, p_val == v)
    return out


def run(samples: Dict[str, Sample], preds_by_sample: Dict[str, List[Box]], use_class_range: bool = True):
    gt, preds = prepare_eval_boxes(samples, preds_by_sample, use_class_range=use_class_range)
    matches = match(gt, preds)
    results = {}
    for key, (gm, pm) in slice_masks(gt, preds, samples).items():
        res = evaluate(gt, preds, matches=matches, gt_mask=gm, pred_mask=pm)
        tokens = {b.sample_token for b, keep in zip(gt, gm) if keep} | {b.sample_token for b, keep in zip(preds, pm) if keep}
        res.extra["n_samples"] = len(tokens) if key[0] != "distance" else len(samples)
        results[key] = res
    return results, gt, preds, matches


def _class_rows(axis: str, value: str, res: DetectionResult) -> List[dict]:
    rows = []
    for cls in CLASSES:
        r = res.per_class[cls]
        row = {"slice_axis": axis, "slice": value, "class": cls, "n_gt": r.n_gt, "n_pred": r.n_pred, "AP": r.ap}
        row.update({f"AP@{th:g}m": r.ap_by_threshold.get(th, np.nan) for th in DIST_THRESHOLDS})
        row.update({TP_METRIC_SHORT[m][1:]: r.tp_errors[m] for m in TP_METRICS})
        rows.append(row)
    total = {"slice_axis": axis, "slice": value, "class": "ALL",
             "n_gt": sum(r.n_gt for r in res.per_class.values()),
             "n_pred": sum(r.n_pred for r in res.per_class.values()),
             "AP": res.mean_ap, "NDS": res.nds, "n_samples": res.extra.get("n_samples")}
    total.update({TP_METRIC_SHORT[m][1:]: res.mean_tp_errors[m] for m in TP_METRICS})
    rows.append(total)
    return rows


def _write_csv(path: str, rows: List[dict]) -> None:
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def write_reports(results, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    by_axis: Dict[str, List[dict]] = {}
    for (axis, value), res in results.items():
        by_axis.setdefault(axis, []).extend(_class_rows(axis, value, res))
    _write_csv(os.path.join(out_dir, "metrics_by_class.csv"), by_axis.pop("all"))
    _write_csv(os.path.join(out_dir, "metrics_by_distance.csv"), by_axis.pop("distance"))
    cond_rows = [r for rows in by_axis.values() for r in rows]
    if cond_rows:
        _write_csv(os.path.join(out_dir, "metrics_by_condition.csv"), cond_rows)
    summary = {f"{a}={v}": {**res.summary_row(), "n_samples": res.extra.get("n_samples")}
               for (a, v), res in results.items()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)


def print_table(results) -> None:
    print(f"{'slice':<28}{'n_gt':>7}{'mAP':>8}{'NDS':>8}  " + "  ".join(f"{c[:5]:>6}" for c in CLASSES))
    for (axis, value), res in results.items():
        n_gt = sum(r.n_gt for r in res.per_class.values())
        aps = "  ".join(f"{res.per_class[c].ap:6.3f}" for c in CLASSES)
        print(f"{axis + '=' + value:<28}{n_gt:>7}{res.mean_ap:8.3f}{res.nds:8.3f}  {aps}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-class-range", action="store_true",
                    help="Disable the devkit per-class range filter (e.g. to see 40m+ for pedestrians).")
    args = ap.parse_args(argv)

    samples = load_gt(args.gt)
    preds = load_predictions(args.pred, samples)
    results, *_ = run(samples, preds, use_class_range=not args.no_class_range)
    print_table(results)
    write_reports(results, args.out)
    print(f"\nWrote CSVs to {args.out}/")


if __name__ == "__main__":
    main()
