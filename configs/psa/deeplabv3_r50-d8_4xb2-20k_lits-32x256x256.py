_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/lits_32x512x512.py',
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
    pretrained=None,
    backbone=dict(in_channels=32),
    decode_head=dict(num_classes=2,
                     out_channels=2,
                     out_channel3d=32,
                     loss_decode=[
                         dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    auxiliary_head=dict(out_channel3d=32,
                        num_classes=2),
    test_cfg=dict(mode='slide_3d', crop_size=(32, 256, 256), stride=(32, 170, 170))
)

train_dataloader = dict(batch_size=2, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_ct=True, interval=10),
    checkpoint=dict(type='CheckpointHook',
                      by_epoch=False,
                      interval=2000,
                      max_keep_ckpts=3,
                      save_best=['mDice'], rule='greater'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='LITS', name='resnet-20k'),
        define_metric_cfg=dict(Dice_FG='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')