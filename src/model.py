"""Load and adapt the pretrained BEVFormer-tiny checkpoint, and export its predictions.

Two subcommands:

    # 10-class checkpoint -> 5-class checkpoint (classifier rows merged, everything else untouched).
    # CPU-only; torch is the only dependency.
    python -m src.model remap --src ckpts/bevformer_tiny_epoch_24.pth --dst ckpts/bevformer_tiny_5cls_init.pth

    # Run a (5- or 10-class) model over the eval split and write BEV-Track-format predictions.
    # Needs the full BEVFormer environment and a CUDA GPU; run from the repo root.
    python -m src.model export --config configs/bevformer_tiny_nusc.py \
        --checkpoint work_dirs/bevformer_tiny_5cls/latest.pth --out results/preds_finetuned.json

Why remap instead of letting the new head initialise randomly: every BEVFormer decoder layer has
a classification branch ending in ``Linear(256, 10)``. Loading a 10-class checkpoint into a 5-class
model would skip those layers on a size mismatch and train them from scratch. Averaging the
source rows that feed each target class (bicycle+motorcycle -> cyclist,
truck+construction_vehicle -> truck) gives the new head a starting point that already detects
the right things. With a sigmoid focal loss each logit is an independent one-vs-rest score, so
averaging rows is a reasonable initialisation rather than an exact equivalence.
"""
import argparse
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

from src.classes import CLASSES, detection_name_to_class

# Class order of every public BEVFormer nuScenes checkpoint (projects/configs/bevformer/*.py).
BEVFORMER_NUSC_CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                          'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')


def class_row_groups(src_classes: Sequence[str] = BEVFORMER_NUSC_CLASSES,
                     dst_classes: Sequence[str] = CLASSES) -> List[List[int]]:
    """For each target class, the indices of the source classes that map onto it."""
    groups = []
    for dst in dst_classes:
        rows = [i for i, s in enumerate(src_classes) if detection_name_to_class(s) == dst]
        if not rows:
            raise ValueError(f"No source class maps to '{dst}'")
        groups.append(rows)
    return groups


def remap_classifier_state_dict(state_dict: Dict, src_classes: Sequence[str] = BEVFORMER_NUSC_CLASSES,
                                dst_classes: Sequence[str] = CLASSES, prefix: str = 'pts_bbox_head.cls_branches'
                                ) -> List[str]:
    """In place: shrink every classifier output layer from len(src) to len(dst) rows. Returns changed keys."""
    import torch

    groups = class_row_groups(src_classes, dst_classes)
    changed = []
    for key in list(state_dict):
        t = state_dict[key]
        if not key.startswith(prefix) or t.dim() == 0 or t.shape[0] != len(src_classes):
            continue
        # LayerNorm params inside the branch are 256-dim, so the shape test picks out only the final Linear.
        state_dict[key] = torch.stack([t[rows].mean(dim=0) for rows in groups]).contiguous()
        changed.append(key)
    if not changed:
        raise RuntimeError(f"No '{prefix}*' tensors with {len(src_classes)} rows found; wrong checkpoint?")
    return changed


def remap_checkpoint(src: str, dst: str, src_classes=BEVFORMER_NUSC_CLASSES, dst_classes=CLASSES) -> List[str]:
    import torch

    ckpt = torch.load(src, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    changed = remap_classifier_state_dict(sd, src_classes, dst_classes)
    out = {'state_dict': sd, 'meta': {**ckpt.get('meta', {}), 'CLASSES': list(dst_classes),
                                      'bevtrack_remapped_from': os.path.basename(src)}}
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    torch.save(out, dst)  # optimizer state dropped on purpose: it belongs to the old head
    return changed


# ---------------------------------------------------------------------------
# Prediction export (mmdet3d v0.17 outputs -> BEV-Track ego-frame boxes)
# ---------------------------------------------------------------------------

def lidar_boxes_to_ego(boxes_3d, scores, labels, class_names: Sequence[str], info: dict, sample_token: str,
                       keep_names: bool = False):
    """Convert one sample's BEVFormer output (LiDARInstance3DBoxes, mmdet3d<1.0 conventions) to ego-frame Boxes.

    Mirrors ``NuScenesDataset.output_to_nusc_box`` + ``lidar_nusc_box_to_global`` (ego half only):
    gravity centre, dims are (w, l, h), yaw_nusc = -yaw - pi/2, velocity in tensor[:, 7:9].
    """
    from pyquaternion import Quaternion

    from src.boxes import Box, transform_box

    centers = boxes_3d.gravity_center.numpy()
    tensor = boxes_3d.tensor.numpy()
    out = []
    for i in range(len(tensor)):
        raw = class_names[int(labels[i])]
        name = raw if keep_names else detection_name_to_class(raw)
        if name is None:
            continue
        b = Box(sample_token, centers[i], tensor[i, 3:6], -tensor[i, 6] - np.pi / 2, name,
                velocity=tensor[i, 7:9], score=float(scores[i]))
        out.append(transform_box(b, Quaternion(info['lidar2ego_rotation']), info['lidar2ego_translation']))
    return out


def export(config: str, checkpoint: str, out: str, bevformer_root: str = 'third_party/BEVFormer',
           ann_file: str = None, keep_names: bool = False) -> None:
    """Run inference with BEVFormer's own test loop and write BEV-Track predictions (ego frame).

    Works for any mmdet3d-0.17 nuScenes detector whose outputs are ``pts_bbox`` LiDARInstance3DBoxes:
    the unmodified 10-class BEVFormer checkpoint (upstream config + ``ann_file``), the 5-class fine-tunes,
    and the CenterPoint LiDAR baseline. 10-class labels are mapped to the 5 classes here, so every model
    goes through the same eval path.
    """
    root = os.path.abspath(bevformer_root)
    sys.path.insert(0, root)
    for k, v in (('RANK', '0'), ('WORLD_SIZE', '1'), ('MASTER_ADDR', '127.0.0.1'), ('MASTER_PORT', '29511')):
        os.environ.setdefault(k, v)

    import importlib

    import torch
    from mmcv import Config
    from mmcv.parallel import MMDistributedDataParallel
    from mmcv.runner import init_dist, load_checkpoint, wrap_fp16_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    from src.dataio import save_predictions

    cfg = Config.fromfile(os.path.abspath(config))
    cwd = os.getcwd()
    os.chdir(root)  # data_root and plugin_dir in the configs are relative to the BEVFormer root
    try:
        importlib.import_module('projects.mmdet3d_plugin')
        from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader

        init_dist('pytorch', **cfg.get('dist_params', {}))
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        if ann_file:
            cfg.data.test.ann_file = ann_file
        dataset = build_dataset(cfg.data.test)
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                  dist=True, shuffle=False,
                                  nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', dict(type='DistributedSampler')))
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        if cfg.get('fp16', None) is not None:
            wrap_fp16_model(model)
        ckpt = load_checkpoint(model, os.path.join(cwd, checkpoint), map_location='cpu')
        class_names = ckpt.get('meta', {}).get('CLASSES') or dataset.CLASSES
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, loader, tmpdir=None, gpu_collect=False)
    finally:
        os.chdir(cwd)

    if isinstance(outputs, dict):
        outputs = outputs['bbox_results']
    preds = {}
    for info, res in zip(dataset.data_infos, outputs):
        pb = res['pts_bbox']
        preds[info['token']] = lidar_boxes_to_ego(pb['boxes_3d'], pb['scores_3d'].numpy(), pb['labels_3d'].numpy(),
                                                  class_names, info, info['token'], keep_names)
    save_predictions(out, preds, {'model': 'BEVFormer-tiny', 'config': config, 'checkpoint': checkpoint,
                                  'class_names': list(class_names)})
    print(f'Wrote predictions for {len(preds)} samples to {out}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('remap')
    r.add_argument('--src', required=True)
    r.add_argument('--dst', required=True)
    e = sub.add_parser('export')
    e.add_argument('--config', required=True)
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--bevformer-root', default='third_party/BEVFormer')
    e.add_argument('--ann-file', default=None,
                   help='Override data.test.ann_file (relative to the BEVFormer root), '
                        'e.g. data/nuscenes/nuscenes_infos_temporal_clean_5cls.pkl')
    e.add_argument('--raw-names', action='store_true',
                   help="Keep the model's own class names (e.g. the 10 nuScenes ones) instead of mapping to 5; "
                        'used by scripts/devkit_check.py')
    args = ap.parse_args(argv)
    if args.cmd == 'remap':
        for k in remap_checkpoint(args.src, args.dst):
            print('remapped', k)
    else:
        export(args.config, args.checkpoint, args.out, args.bevformer_root, args.ann_file, args.raw_names)


if __name__ == '__main__':
    main()
