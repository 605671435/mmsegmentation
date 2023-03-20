_base_ = ['./fcn_r50-d8_1xb8-40k_synapse-512x512.py']

# model settings
model = dict(
    backbone=dict(_delete_=True,
                  type='MobileNetV2',
                  widen_factor=1.,
                  strides=(1, 2, 2, 1, 1, 1, 1),
                  dilations=(1, 1, 1, 2, 2, 4, 4),
                  out_indices=(1, 2, 4, 6),
                  norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(in_channels=320),
    auxiliary_head=dict(in_channels=96))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_dataloader = dict(batch_size=2, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_table=True, interval=50),
    checkpoint=dict(type='CheckpointHook',
                      by_epoch=False,
                      interval=4000,
                      max_keep_ckpts=3,
                      save_best=['mIoU'], rule='greater'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-mobilenetv2-40k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

randomness = dict(seed=50000000,
                  deterministic=False,
                  diff_rank_seed=False)