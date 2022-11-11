_base_ = [
    '../_base_/models/segformer_mit-ex-b2.py',
    '../_base_/datasets/hrf_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2,
                     loss_decode=[
                         dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.01, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
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
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mmsegmentation', name='ex-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')