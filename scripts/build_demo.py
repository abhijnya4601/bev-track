"""Build docs/demo_data.json for the interactive demo page (docs/index.html).

    python scripts/build_demo.py --gt data/processed/gt_clean.json \
        --pred pretrained=results/preds_pretrained.json \
        --pred headonly=results/preds_headonly.json \
        --pred finetuned=results/preds_finetuned.json \
        --out docs/demo_data.json

For every model: detection metrics (per class, per distance bucket), CLEAR-MOT tracking metrics, and
per-frame predictions and tracks. The page re-does the devkit's greedy 2 m matching per frame at the
slider's score threshold (matching is independent per sample, so this is exact), which is why no
match flags are stored here.
Boxes are compact arrays (see the *_FIELDS lists), in the ego frame: x forward, y left.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.boxes import Box  # noqa: E402
from src.classes import CLASSES  # noqa: E402
from src.dataio import filter_boxes, load_gt, load_predictions  # noqa: E402
from src.eval.metrics import TP_METRICS  # noqa: E402
from src.eval.slice_eval import bucket_names, run  # noqa: E402
from src.eval.track_metrics import evaluate_tracking  # noqa: E402
from src.track import TrackerConfig, track_all  # noqa: E402

GT_FIELDS = ["x", "y", "w", "l", "yaw", "cls"]
PRED_FIELDS = ["x", "y", "w", "l", "yaw", "cls", "score"]
TRACK_FIELDS = ["x", "y", "w", "l", "yaw", "cls", "track"]
LABELS = {"pretrained": "Pretrained, relabeled (no training)",
          "headonly": "Fine-tuned: head only",
          "finetuned": "Fine-tuned: most layers"}


def r(v, nd=2):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def box_row(b: Box, extra: list) -> list:
    return [r(b.translation[0]), r(b.translation[1]), r(b.size[0]), r(b.size[1]), r(b.yaw, 3),
            CLASSES.index(b.name)] + extra


def build_model(samples, preds_by_s: Dict[str, List[Box]], min_score: float) -> dict:
    results, gt, preds, _ = run(samples, preds_by_s)
    tracks = track_all(samples, preds_by_s, TrackerConfig())
    trk = evaluate_tracking(samples, tracks)
    track_ids: Dict[str, int] = {}

    frames = {tok: {"preds": [], "tracks": [box_row(b, [track_ids.setdefault(b.instance_id, len(track_ids))])
                                            for b in filter_boxes(tracks.get(tok, []), s, is_gt=False)]}
              for tok, s in samples.items()}
    for p in preds:  # already devkit-filtered (class range) by run()
        if p.score >= min_score:
            frames[p.sample_token]["preds"].append(box_row(p, [r(p.score)]))

    allr = results[("all", "all")]
    per_dist = {b: {c: r(results[("distance", b)].per_class[c].ap, 3) for c in CLASSES} for b in bucket_names()}
    return {
        "summary": {"mAP": r(allr.mean_ap, 3), "NDS": r(allr.nds, 3),
                    **{m: r(allr.mean_tp_errors[m], 3) for m in TP_METRICS}},
        "per_class": {c: {"AP": r(allr.per_class[c].ap, 3), "n_gt": allr.per_class[c].n_gt} for c in CLASSES},
        "per_distance": per_dist,
        "distance_mAP": {b: r(results[("distance", b)].mean_ap, 3) for b in bucket_names()},
        "tracking": {name: {"MOTA": r(c.mota, 3), "MOTP": r(c.motp, 3), "IDSW": c.id_switches,
                            "FRAG": c.fragmentations, "GT": c.num_gt}
                     for name, c in [*trk.per_class.items(), ("ALL", trk.overall)]},
        "frames": frames,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=path, in display order")
    ap.add_argument("--out", default="docs/demo_data.json")
    ap.add_argument("--min-score", type=float, default=0.15, help="Drop boxes below this score to keep the file small.")
    ap.add_argument("--note", default="", help="Shown on the page (e.g. to mark stand-in data).")
    args = ap.parse_args(argv)

    samples = load_gt(args.gt)
    scenes: Dict[str, dict] = {}
    frame_meta = []
    for tok, s in samples.items():
        sc = scenes.setdefault(s.scene_name, {"name": s.scene_name, "condition": s.condition, "frames": []})
        sc["frames"].append(len(frame_meta))
        gt = filter_boxes(s.gt, s, is_gt=True)
        frame_meta.append({"token": tok, "scene": s.scene_name, "t": s.timestamp,
                           "gt": [box_row(b, []) for b in gt]})

    models = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        print(f"building {name} from {path}")
        m = build_model(samples, load_predictions(path, samples), args.min_score)
        per_frame = m.pop("frames")
        m["label"] = LABELS.get(name, name)
        m["frames"] = [per_frame[f["token"]] for f in frame_meta]
        models[name] = m

    for f in frame_meta:
        f.pop("token")
    out = {"classes": list(CLASSES), "gt_fields": GT_FIELDS, "pred_fields": PRED_FIELDS,
           "track_fields": TRACK_FIELDS, "distance_buckets": bucket_names(), "note": args.note,
           "scenes": list(scenes.values()), "frames": frame_meta, "models": models}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
