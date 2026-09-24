"""Bootstrap confidence intervals for mAP, and paired comparisons between models.

    python -m src.eval.bootstrap --gt data/processed/gt_clean.json \
        --pred camera=results/preds_pretrained.json --pred lidar=results/preds_lidar.json \
        -n 1000 --out results/bootstrap.json

Resamples **frames** (samples) with replacement, recomputes mAP overall and per distance bucket, and
reports percentile 95% intervals. Every model is evaluated on the *same* resampled frames, so
``model B - model A`` (for every pair) is a paired difference: its interval says whether B beats A on these scenes,
and ``P(diff > 0)`` is the fraction of resamples where it does.

How it stays fast: devkit matching is independent per sample, so it is done once. A resample then just
gives each frame a weight (how many times it was drawn), and AP is computed from weighted
cumulative TP/FP counts. That is the same as literally duplicating frames, except for the ordering
among exact score ties between copies.

**Caveat, printed with every result:** frames within a scene are strongly correlated (consecutive
2 Hz snapshots of the same objects), and there are only 4 scenes. Frame-level intervals are therefore
optimistic, i.e. too narrow. Read a difference as solid only if its interval clearly excludes 0.
"""
import argparse
import json
from typing import Dict, List

import numpy as np

from src.classes import CLASSES
from src.dataio import load_gt, load_predictions, prepare_eval_boxes
from src.eval.metrics import DIST_THRESHOLDS, MIN_PRECISION, MIN_RECALL, N_ELEM, match
from src.eval.slice_eval import bucket_names, distance_bucket

CAVEAT = ("Frame-level bootstrap: frames within a scene are correlated and there are only 4 scenes, "
          "so these intervals are optimistic (too narrow).")


class ModelTable:
    """Per-(class, threshold) predictions sorted by score, with their sample index, TP flag and bucket."""

    def __init__(self, samples, preds_by_s, tokens: List[str]):
        gt, preds = prepare_eval_boxes(samples, preds_by_s)
        tok_idx = {t: i for i, t in enumerate(tokens)}
        buckets = bucket_names()
        b_idx = {b: i for i, b in enumerate(buckets)}
        g_bucket = np.array([b_idx[distance_bucket(g.ego_dist)] for g in gt], dtype=int)
        p_bucket = np.array([b_idx[distance_bucket(p.ego_dist)] for p in preds], dtype=int)
        # n_gt[class][bucket][sample]
        self.n_gt = {c: np.zeros((len(buckets), len(tokens))) for c in CLASSES}
        for g, b in zip(gt, g_bucket):
            self.n_gt[g.name][b, tok_idx[g.sample_token]] += 1
        self.tables = {}
        for (cls, th), mt in match(gt, preds).items():
            order = np.argsort(-mt.scores, kind="stable")  # already descending; keep devkit order
            pi, gi = mt.pred_idx[order], mt.gt_idx[order]
            sample = np.array([tok_idx[preds[i].sample_token] for i in pi], dtype=int)
            # a matched prediction belongs to its GT's bucket; an unmatched one to its own
            bucket = np.where(gi >= 0, g_bucket[np.maximum(gi, 0)] if len(gt) else 0, p_bucket[pi] if len(preds) else 0)
            self.tables[(cls, th)] = (sample, gi >= 0, bucket)

    def ap(self, cls: str, th: float, w: np.ndarray, bucket=None) -> float:
        n_gt_arr = self.n_gt[cls] if bucket is None else self.n_gt[cls][bucket:bucket + 1]
        n_gt = float((n_gt_arr * w).sum())
        if n_gt == 0:
            return np.nan
        sample, tp, bk = self.tables[(cls, th)]
        pw = w[sample]
        keep = pw > 0 if bucket is None else (pw > 0) & (bk == bucket)
        if not np.any(tp[keep]):
            return 0.0
        pw, t = pw[keep], tp[keep]
        tpc = np.cumsum(pw * t)
        fpc = np.cumsum(pw * ~t)
        prec = tpc / (tpc + fpc)
        rec = tpc / n_gt
        prec_i = np.interp(np.linspace(0, 1, N_ELEM), rec, prec, right=0)
        p = prec_i[round(100 * MIN_RECALL) + 1:] - MIN_PRECISION
        p[p < 0] = 0
        return float(np.mean(p)) / (1.0 - MIN_PRECISION)

    def mean_ap(self, w: np.ndarray, bucket=None) -> float:
        aps = []
        for c in CLASSES:
            per_th = [self.ap(c, th, w, bucket) for th in DIST_THRESHOLDS]
            if not np.isnan(per_th[0]):
                aps.append(np.mean(per_th))
        return float(np.mean(aps)) if aps else np.nan


def run(samples, preds: Dict[str, Dict], n: int = 1000, seed: int = 0) -> dict:
    tokens = list(samples)
    tables = {name: ModelTable(samples, p, tokens) for name, p in preds.items()}
    names, buckets = list(preds), bucket_names()
    slices = [("all", None)] + [(b, i) for i, b in enumerate(buckets)]
    ones = np.ones(len(tokens))
    point = {m: {s: tables[m].mean_ap(ones, bi) for s, bi in slices} for m in names}
    rng = np.random.default_rng(seed)
    draws = {m: {s: [] for s, _ in slices} for m in names}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(tokens), len(tokens)), minlength=len(tokens)).astype(float)
        for m in names:
            for s, bi in slices:
                draws[m][s].append(tables[m].mean_ap(w, bi))
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    out = {"n_resamples": n, "n_frames": len(tokens), "caveat": CAVEAT, "models": {}, "paired": {}}
    for m in names:
        out["models"][m] = {s: {"mAP": point[m][s], "ci95": ci(np.array(draws[m][s]))} for s, _ in slices}
    for i, base in enumerate(names):
        for m in names[i + 1:]:
            key = f"{m} - {base}"
            out["paired"][key] = {}
            for s, _ in slices:
                d = np.array(draws[m][s]) - np.array(draws[base][s])
                out["paired"][key][s] = {"diff": point[m][s] - point[base][s], "ci95": ci(d),
                                         "p_gt_0": float(np.nanmean(d > 0))}
    return out


def print_report(res: dict) -> None:
    slices = list(next(iter(res["models"].values())))
    print(f"mAP with 95% bootstrap intervals ({res['n_resamples']} resamples of {res['n_frames']} frames)")
    print(f"{'model':<14}" + "".join(f"{s:>24}" for s in slices))
    for m, v in res["models"].items():
        print(f"{m:<14}" + "".join(f"{v[s]['mAP']:>8.3f} [{v[s]['ci95'][0]:.3f},{v[s]['ci95'][1]:.3f}]" for s in slices))
    print("\npaired differences (same resampled frames); P = share of resamples where diff > 0")
    for pair, v in res["paired"].items():
        print(f"{pair:<24}" + "".join(f"  {s}: {v[s]['diff']:+.3f} [{v[s]['ci95'][0]:+.3f},{v[s]['ci95'][1]:+.3f}] P={v[s]['p_gt_0']:.2f}"
                                      for s in slices))
    print("\n" + res["caveat"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bootstrap.json")
    args = ap.parse_args(argv)
    samples = load_gt(args.gt)
    preds = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        preds[name] = load_predictions(path, samples)
    res = run(samples, preds, n=args.n, seed=args.seed)
    print_report(res)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
