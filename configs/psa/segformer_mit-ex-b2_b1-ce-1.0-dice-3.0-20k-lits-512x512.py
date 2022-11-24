_base_ = [
    '../_base_/models/segformer_mit-ex-b2.py',
    '../_base_/datasets/lits_32x512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (32, 256, 256)

data_preprocessor = dict(
    _scope_='mmseg',
    type='SegDataPreProcessor',
    mean=None,#-105,
    std=None,#124,
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)


model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(in_channels=32),
    decode_head=dict(num_classes=2, channels=256, out_channel3d=32),
    # test_cfg=dict(mode='slide', crop_size=(32, 256, 256), stride=(32, 196, 196)),
    test_cfg=dict(mode='slide_3d', crop_size=(32, 256, 256), stride=(32, 170, 170))
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            # 'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_ct=True, interval=10),
    checkpoint=dict(type='CheckpointHook',
                      by_epoch=False,
                      interval=2000,
                      max_keep_ckpts=3,
                      # save_best=['mDice'], rule='greater'
))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='LITS', name='segformer-ex-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')