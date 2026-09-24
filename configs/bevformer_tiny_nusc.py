# BEVFormer-tiny, fine-tuned to the BEV-Track 5-class taxonomy on nuScenes-mini.
#
# Inherits everything from the upstream config (third_party/BEVFormer, pinned by the git submodule)
# and overrides only what the 5-class head fine-tune needs. mmcv merges nested dicts but REPLACES
# lists, and variables like `class_names` are already baked into the base's dicts, so every place
# a class list appears is restated below.
#
# Things that deliberately stay at their upstream values, and why:
#   * bev_h_/bev_w_ = 50: the pretrained `bev_embedding` (bev_h*bev_w x 256) and the learned
#     positional encoding (row/col_num_embed) have this shape baked in. Changing the grid means
#     re-learning those from scratch, which the 242-sample `seen` training split cannot support.
#   * point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]: the encoder's 3D reference points and
#     the box regression normalisation are defined relative to it, so the checkpoint only makes
#     sense with this exact range.
#   * queue_length = 3: already short (current frame + 2 history frames for temporal self-attention).
#     Drop to 2 if memory is tight.
#
# Launch through `python -m src.train`, which remaps the checkpoint's classifier and sets load_from.

_base_ = ['../third_party/BEVFormer/projects/configs/bevformer/bevformer_tiny.py']

class_names = ['car', 'pedestrian', 'cyclist', 'truck', 'barrier']  # must match src/classes.py
num_classes = len(class_names)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_h_ = 50
bev_w_ = 50
queue_length = 3
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    # Frozen: all 4 ResNet stages (+ stem) and BN statistics. The backbone saw far more images
    # during pretraining than mini will ever show it.
    img_backbone=dict(frozen_stages=4),
    pts_bbox_head=dict(
        num_classes=num_classes,
        bbox_coder=dict(num_classes=num_classes)))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data_root = 'data/nuscenes/'
# *_5cls.pkl are written by `python data/prepare_nuscenes.py remap-infos`.
# "seen" = mini scenes in the official train split (the checkpoint already trained on them),
# "clean" = mini scenes in the official val split. See data/prepare_nuscenes.py for why.
train_split = 'seen'
eval_split = 'clean'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        ann_file=data_root + f'nuscenes_infos_temporal_{train_split}_5cls.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        queue_length=queue_length),
    val=dict(
        ann_file=data_root + f'nuscenes_infos_temporal_{eval_split}_5cls.pkl',
        pipeline=test_pipeline,
        classes=class_names),
    test=dict(
        ann_file=data_root + f'nuscenes_infos_temporal_{eval_split}_5cls.pkl',
        pipeline=test_pipeline,
        classes=class_names))

# Layer-wise learning rates: frozen backbone, gentle on the neck/BEV encoder, full on the head.
# mmcv applies the longest matching key, so the head's branches override the generic head entry.
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.0, decay_mult=0.0),
        'img_neck': dict(lr_mult=0.1),
        'pts_bbox_head.transformer.encoder': dict(lr_mult=0.1),
        'pts_bbox_head.bev_embedding': dict(lr_mult=0.1),
        'pts_bbox_head.transformer': dict(lr_mult=0.5),
        'pts_bbox_head.cls_branches': dict(lr_mult=1.0),
        'pts_bbox_head.reg_branches': dict(lr_mult=1.0),
    }))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

# The dataset's built-in evaluate() looks up nuScenes default attributes by class name, so it cannot
# handle our class names; in-training validation is disabled (src.train passes --no-validate). Evaluate with
# `python -m src.model export` + `python -m src.eval.slice_eval` instead.
evaluation = dict(interval=10 ** 9, pipeline=test_pipeline)

# Set by src/train.py to the classifier-remapped checkpoint.
load_from = None
