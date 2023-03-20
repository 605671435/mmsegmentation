_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/synapse.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(num_classes=14),
    auxiliary_head=dict(num_classes=14),
    test_cfg=dict(mode='whole')
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
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
            project='synapse', name='fcn-r50-20k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')