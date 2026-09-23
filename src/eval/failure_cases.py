"""Pull out the worst predictions of an evaluation run, diagnose them, and render them in BEV.

    python -m src.eval.failure_cases --gt data/processed/gt_mini_val.json \
        --pred preds.json --out results/failure_examples -n 10

All three lists come from the matching at the 2 m TP threshold:
  * false negatives: unmatched GT, ranked by lidar+radar point count. These are the most
    observable objects the model still missed. GT has no confidence, so "worst" means "easiest".
  * false positives: unmatched predictions, ranked by score.
  * localization errors: matched pairs, ranked by a composite
    ``trans_err + scale_err + orient_err / pi`` (each term is roughly 0 to 1 in the typical range).

Each failure also gets a diagnosis tag, because "FP" alone does not tell you what to fix:
  * FP ``duplicate``: a same-class prediction already matched that GT (an NMS/dedup problem)
  * FP ``class_confusion``: a GT of a different class is within 2 m
  * FP ``mislocalized``: a same-class GT is within 4 m but not 2 m
  * FP ``hallucination``: nothing nearby
  * FN ``class_confusion``: a prediction of another class is within 2 m
  * FN ``mislocalized``: a same-class prediction is within 4 m but not 2 m
  * FN ``missed``: nothing nearby
"""
import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.boxes import Box
from src.dataio import Sample, load_gt, load_predictions, prepare_eval_boxes
from src.eval.metrics import TP_DIST_THRESHOLD, center_distance, match, tp_errors


@dataclass
class Failure:
    kind: str  # fn | fp | loc
    rank_value: float
    sample_token: str
    box: Box  # the GT box for fn/loc, the prediction for fp
    partner: Box = None  # matched prediction for loc
    diagnosis: str = ""
    detail: str = ""


def _near(box: Box, others: List[Box], radius: float, same_class: bool) -> List[Box]:
    return [o for o in others if (o.name == box.name) == same_class and center_distance(o, box) < radius]


def extract(gt: List[Box], preds: List[Box], matches=None, n: int = 10) -> Dict[str, List[Failure]]:
    if matches is None:
        matches = match(gt, preds, thresholds=(TP_DIST_THRESHOLD,))
    gt_matched, pred_to_gt = set(), {}
    for (cls, th), mt in matches.items():
        if th != TP_DIST_THRESHOLD:
            continue
        for pi, gi in zip(mt.pred_idx, mt.gt_idx):
            if gi >= 0:
                gt_matched.add(int(gi))
                pred_to_gt[int(pi)] = int(gi)

    gt_by_s: Dict[str, List[Box]] = {}
    pred_by_s: Dict[str, List[Box]] = {}
    for g in gt:
        gt_by_s.setdefault(g.sample_token, []).append(g)
    for p in preds:
        pred_by_s.setdefault(p.sample_token, []).append(p)
    matched_gt_ids = {id(gt[i]) for i in gt_matched}

    fns = []
    for i, g in enumerate(gt):
        if i in gt_matched:
            continue
        others = pred_by_s.get(g.sample_token, [])
        conf = _near(g, others, TP_DIST_THRESHOLD, same_class=False)
        loc = _near(g, others, 2 * TP_DIST_THRESHOLD, same_class=True)
        if conf:
            diag, det = "class_confusion", "predicted as " + max(conf, key=lambda o: o.score).name
        elif loc:
            diag, det = "mislocalized", f"nearest same-class pred {min(center_distance(o, g) for o in loc):.2f} m away"
        else:
            diag, det = "missed", ""
        fns.append(Failure("fn", g.num_pts, g.sample_token, g, diagnosis=diag,
                           detail=f"{det}; dist={g.ego_dist:.1f} m, pts={g.num_pts}".lstrip("; ")))
    fns.sort(key=lambda f: (-f.rank_value, f.box.ego_dist))

    fps = []
    for i, p in enumerate(preds):
        if i in pred_to_gt:
            continue
        others = gt_by_s.get(p.sample_token, [])
        same_near = _near(p, others, TP_DIST_THRESHOLD, same_class=True)
        conf = _near(p, others, TP_DIST_THRESHOLD, same_class=False)
        loc = _near(p, others, 2 * TP_DIST_THRESHOLD, same_class=True)
        if same_near and all(id(o) in matched_gt_ids for o in same_near):
            diag, det = "duplicate", "same-class GT already matched by a higher-scoring prediction"
        elif conf:
            diag, det = "class_confusion", "GT is " + min(conf, key=lambda o: center_distance(o, p)).name
        elif loc:
            diag, det = "mislocalized", f"nearest same-class GT {min(center_distance(o, p) for o in loc):.2f} m away"
        else:
            diag, det = "hallucination", ""
        fps.append(Failure("fp", p.score, p.sample_token, p, diagnosis=diag,
                           detail=f"{det}; dist={p.ego_dist:.1f} m".lstrip("; ")))
    fps.sort(key=lambda f: -f.rank_value)

    locs = []
    for pi, gi in pred_to_gt.items():
        e = tp_errors(gt[gi], preds[pi])
        composite = e["trans_err"] + e["scale_err"] + e["orient_err"] / np.pi
        worst = max(("trans_err", e["trans_err"] / TP_DIST_THRESHOLD), ("scale_err", e["scale_err"]),
                    ("orient_err", e["orient_err"] / np.pi), key=lambda kv: kv[1])[0]
        locs.append(Failure("loc", composite, gt[gi].sample_token, gt[gi], partner=preds[pi], diagnosis=worst,
                            detail=f"ATE={e['trans_err']:.2f} m, ASE={e['scale_err']:.2f}, "
                                   f"AOE={np.degrees(e['orient_err']):.0f} deg, dist={gt[gi].ego_dist:.1f} m"))
    locs.sort(key=lambda f: -f.rank_value)

    return {"fn": fns[:n], "fp": fps[:n], "loc": locs[:n], "_counts": {
        "fn": _count(fns), "fp": _count(fps), "loc": _count(locs)}}


def _count(fs: List[Failure]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for f in fs:
        out[f.diagnosis] = out.get(f.diagnosis, 0) + 1
    return out


def render(failures: Dict[str, List[Failure]], samples: Dict[str, Sample], gt: List[Box], preds: List[Box],
           out_dir: str) -> List[str]:
    import matplotlib.pyplot as plt

    from src.viz import plot_bev

    os.makedirs(out_dir, exist_ok=True)
    gt_by_s: Dict[str, List[Box]] = {}
    pred_by_s: Dict[str, List[Box]] = {}
    for g in gt:
        gt_by_s.setdefault(g.sample_token, []).append(g)
    for p in preds:
        pred_by_s.setdefault(p.sample_token, []).append(p)

    paths, rows = [], []
    titles = {"fn": "False negative", "fp": "False positive", "loc": "Localization error"}
    for kind in ("fn", "fp", "loc"):
        for rank, f in enumerate(failures[kind], 1):
            s = samples[f.sample_token]
            fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), layout="constrained")
            head = f"{titles[kind]} #{rank}: {f.box.name} [{f.diagnosis}]\n{s.scene_name} ({s.condition.get('lighting', '?')}, " \
                   f"{s.condition.get('weather', '?')}) · {f.detail}"
            plot_bev(gt_by_s.get(f.sample_token, []), pred_by_s.get(f.sample_token, []), highlight=f.box,
                     ax=axes[0], title="full scene", score_threshold=0.3)
            plot_bev(gt_by_s.get(f.sample_token, []), pred_by_s.get(f.sample_token, []), highlight=f.box,
                     ax=axes[1], center=f.box.translation, extent=10.0, score_threshold=0.1,
                     title="zoom (preds score >= 0.1)")
            fig.suptitle(head, fontsize=10)
            path = os.path.join(out_dir, f"{kind}_{rank:02d}.png")
            fig.savefig(path, dpi=110)
            plt.close(fig)
            paths.append(path)
            rows.append({"kind": kind, "rank": rank, "class": f.box.name, "diagnosis": f.diagnosis,
                         "rank_value": f"{f.rank_value:.4f}", "scene": s.scene_name, "sample_token": f.sample_token,
                         "ego_dist_m": f"{f.box.ego_dist:.2f}", "detail": f.detail, "image": os.path.basename(path)})
    with open(os.path.join(out_dir, "failures.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["kind"])
        w.writeheader()
        w.writerows(rows)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="results/failure_examples")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Ignore predictions below this score (otherwise low-score noise dominates the FP list).")
    args = ap.parse_args(argv)

    samples = load_gt(args.gt)
    preds_by_s = load_predictions(args.pred, samples)
    preds_by_s = {t: [p for p in ps if p.score >= args.min_score] for t, ps in preds_by_s.items()}
    gt, preds = prepare_eval_boxes(samples, preds_by_s)
    failures = extract(gt, preds, n=args.n)
    for kind, counts in failures["_counts"].items():
        print(f"{kind}: {sum(counts.values())} total, by diagnosis: {counts}")
    paths = render(failures, samples, gt, preds, args.out)
    print(f"Rendered {len(paths)} images to {args.out}/")


if __name__ == "__main__":
    main()
