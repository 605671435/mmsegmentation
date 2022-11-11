_base_ = [
    '../_base_/models/segformer_mit-ex-b0.py',
    '../_base_/datasets/hrf_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PSAMixVisionTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512],
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
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
