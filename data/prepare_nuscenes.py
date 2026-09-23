"""Prepare nuScenes-mini for BEV-Track.

Download first (needs a free account at nuscenes.org, so it can't be scripted):
  * v1.0-mini.tgz           -> extract into data/nuscenes/
  * can_bus.zip (CAN bus expansion; BEVFormer needs it) -> extract into data/  (gives data/can_bus/)

Then:

    python data/prepare_nuscenes.py report   # scene table: conditions + which ones a pretrained ckpt has seen
    python data/prepare_nuscenes.py gt       # GT files for the eval harness -> data/processed/gt_<split>.json
    python data/prepare_nuscenes.py remap-infos --infos-dir data/nuscenes   # 5-class BEVFormer info pkls

Splits
------
nuScenes-mini's own split is mini_train (8 scenes) / mini_val (2 scenes). But the public
BEVFormer checkpoints were trained on the **official train** split, and 6 of the 8 mini_train
scenes are in it, including every night scene. Evaluating the pretrained model on those scenes
measures memorisation, not generalisation. So this script also defines:

  * ``seen``  = mini scenes in the official train split (the checkpoint has trained on them)
  * ``clean`` = mini scenes in the official val split (the checkpoint has never seen them)

The recommended protocol is to fine-tune the 5-class head on ``seen`` and evaluate both
the pretrained and fine-tuned models on ``clean``. Nothing new leaks, and both models are
compared on the same held-out scenes. The ``report`` subcommand prints the actual membership
and conditions from your copy of the data.
"""
import argparse
import csv
import os
import pickle
import sys
from typing import Dict, Iterable, List, Set

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classes import CLASSES, category_to_class, detection_name_to_class  # noqa: E402

DEFAULT_DATAROOT = "data/nuscenes"


def parse_condition(description: str, location: str) -> Dict[str, str]:
    """Rough condition tags from the free-text scene description.

    nuScenes has no structured weather/lighting field; descriptions look like
    "Night, after rain, many peds, PMD, ...". "after rain" counts as rain (wet road, reflections).
    """
    d = description.lower()
    return {
        "lighting": "night" if "night" in d else "day",
        "weather": "rain" if "rain" in d else "dry",
        "location": location,
    }


def official_split_of(scene_name: str) -> str:
    from nuscenes.utils.splits import create_splits_scenes

    s = create_splits_scenes()
    if scene_name in s["train"]:
        return "train"
    if scene_name in s["val"]:
        return "val"
    return "test"


def scene_splits(nusc) -> Dict[str, Set[str]]:
    from nuscenes.utils.splits import create_splits_scenes

    s = create_splits_scenes()
    names = {sc["name"] for sc in nusc.scene}
    return {
        "mini_train": set(s["mini_train"]) & names,
        "mini_val": set(s["mini_val"]) & names,
        "seen": {n for n in names if official_split_of(n) == "train"},
        "clean": {n for n in names if official_split_of(n) == "val"},
        "all": names,
    }


def _nusc(dataroot: str, version: str):
    from nuscenes import NuScenes

    return NuScenes(version=version, dataroot=dataroot, verbose=False)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(nusc, out_csv: str) -> List[dict]:
    splits = scene_splits(nusc)
    rows = []
    for sc in sorted(nusc.scene, key=lambda x: x["name"]):
        log = nusc.get("log", sc["log_token"])
        cond = parse_condition(sc["description"], log["location"])
        counts = {c: 0 for c in CLASSES}
        tok = sc["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            for a in sample["anns"]:
                c = category_to_class(nusc.get("sample_annotation", a)["category_name"])
                if c:
                    counts[c] += 1
            tok = sample["next"]
        rows.append({
            "scene": sc["name"],
            "mini_split": "mini_train" if sc["name"] in splits["mini_train"] else "mini_val",
            "official_split": official_split_of(sc["name"]),
            "seen_by_pretrained": sc["name"] in splits["seen"],
            **cond, "n_samples": sc["nbr_samples"], **{f"n_{c}": n for c, n in counts.items()},
            "description": sc["description"],
        })
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'scene':<12}{'mini':<12}{'official':<10}{'seen':<6}{'light':<7}{'weather':<8}{'loc':<22}"
          + "".join(f"{c[:5]:>7}" for c in CLASSES))
    for r in rows:
        print(f"{r['scene']:<12}{r['mini_split']:<12}{r['official_split']:<10}{str(r['seen_by_pretrained'])[0]:<6}"
              f"{r['lighting']:<7}{r['weather']:<8}{r['location']:<22}" + "".join(f"{r['n_' + c]:>7}" for c in CLASSES))
    return rows


# ---------------------------------------------------------------------------
# gt
# ---------------------------------------------------------------------------

def build_gt(nusc, scene_names: Iterable[str]):
    from pyquaternion import Quaternion

    from src.boxes import Box, global_to_ego, yaw_of
    from src.dataio import Sample

    wanted = set(scene_names)
    samples: Dict[str, Sample] = {}
    for sc in nusc.scene:
        if sc["name"] not in wanted:
            continue
        log = nusc.get("log", sc["log_token"])
        cond = parse_condition(sc["description"], log["location"])
        tok = sc["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            ego_pose = {"translation": pose["translation"], "rotation": pose["rotation"]}
            boxes, racks = [], []
            for a in sample["anns"]:
                ann = nusc.get("sample_annotation", a)
                if ann["category_name"] == "static_object.bicycle_rack":
                    racks.append(global_to_ego(Box(tok, ann["translation"], ann["size"],
                                                   yaw_of(Quaternion(ann["rotation"])), "bicycle_rack"), ego_pose))
                    continue
                cls = category_to_class(ann["category_name"])
                if cls is None:
                    continue
                g = Box(tok, ann["translation"], ann["size"], yaw_of(Quaternion(ann["rotation"])), cls,
                        velocity=nusc.box_velocity(a)[:2], instance_id=ann["instance_token"],
                        num_pts=ann["num_lidar_pts"] + ann["num_radar_pts"])
                boxes.append(global_to_ego(g, ego_pose))
            samples[tok] = Sample(tok, sc["token"], sc["name"], sample["timestamp"], ego_pose, cond, boxes, racks)
            tok = sample["next"]
    return samples


# ---------------------------------------------------------------------------
# remap-infos
# ---------------------------------------------------------------------------

def remap_info_names(infos: List[dict]) -> List[dict]:
    """Rewrite ``gt_names`` from the 10 detection names to the 5 working classes, in place.

    Names with no 5-class equivalent (bus, trailer, traffic_cone) are left as-is. The BEVFormer
    dataset turns any name not in ``classes`` into label -1, and ``ObjectNameFilter`` drops those.
    """
    for info in infos:
        info["gt_names"] = np.array([detection_name_to_class(n) or n for n in info["gt_names"]])
    return infos


def remap_infos(nusc, infos_dir: str, tag: str = "nuscenes_infos_temporal") -> None:
    infos, metadata = [], None
    for split in ("train", "val"):
        path = os.path.join(infos_dir, f"{tag}_{split}.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        infos += d["infos"]
        metadata = metadata or d["metadata"]
    remap_info_names(infos)
    scene_name = {sc["token"]: sc["name"] for sc in nusc.scene}
    for split, scenes in scene_splits(nusc).items():
        sel = sorted((i for i in infos if scene_name[i["scene_token"]] in scenes), key=lambda i: i["timestamp"])
        out = os.path.join(infos_dir, f"{tag}_{split}_5cls.pkl")
        with open(out, "wb") as f:
            pickle.dump({"infos": sel, "metadata": metadata}, f)
        print(f"{out}: {len(sel)} samples from {len(scenes)} scenes")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["report", "gt", "remap-infos"])
    ap.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--infos-dir", default=DEFAULT_DATAROOT,
                    help="Where BEVFormer's tools/create_data.py wrote nuscenes_infos_temporal_{train,val}.pkl")
    args = ap.parse_args(argv)

    nusc = _nusc(args.dataroot, args.version)
    if args.cmd == "report":
        report(nusc, os.path.join("results", "split_report.csv"))
    elif args.cmd == "gt":
        from src.dataio import save_gt

        os.makedirs(args.out, exist_ok=True)
        for split, scenes in scene_splits(nusc).items():
            samples = build_gt(nusc, scenes)
            path = os.path.join(args.out, f"gt_{split}.json")
            save_gt(path, samples, {"version": args.version, "split": split, "scenes": sorted(scenes)})
            n_boxes = sum(len(s.gt) for s in samples.values())
            print(f"{path}: {len(scenes)} scenes, {len(samples)} samples, {n_boxes} boxes")
    else:
        remap_infos(nusc, args.infos_dir)


if __name__ == "__main__":
    main()
