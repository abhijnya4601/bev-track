# Head-only ablation of bevformer_tiny_nusc.py: only the classification/regression branches train.
#
# The first fine-tune (neck/encoder at 0.1x, transformer at 0.5x, 12 epochs) scored BELOW the unmodified
# pretrained model with its classes remapped (mAP 0.388 vs 0.452 on the clean scenes). This config tests
# the narrower hypothesis from the project spec: adapt only the final layers, keep every learned
# representation (backbone, neck, BEV encoder, decoder, queries) exactly as pretrained.
_base_ = ['./bevformer_tiny_nusc.py']

frozen = dict(lr_mult=0.0, decay_mult=0.0)  # decay_mult too: AdamW's decoupled weight decay would still shrink them
optimizer = dict(
    lr=5e-5,
    paramwise_cfg=dict(
        _delete_=True,  # replace the parent's custom_keys instead of merging into them
        custom_keys={
            'img_backbone': frozen,
            'img_neck': frozen,
            'pts_bbox_head.transformer': frozen,
            'pts_bbox_head.bev_embedding': frozen,
            'pts_bbox_head.positional_encoding': frozen,
            'pts_bbox_head.query_embedding': frozen,
            'pts_bbox_head.cls_branches': dict(lr_mult=1.0),
            'pts_bbox_head.reg_branches': dict(lr_mult=1.0),
        }))

lr_config = dict(warmup_iters=50)
total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
