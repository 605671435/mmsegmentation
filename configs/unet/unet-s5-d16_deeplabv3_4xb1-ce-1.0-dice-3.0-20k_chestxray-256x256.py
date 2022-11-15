_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/chestxray_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))

train_dataloader = dict(batch_size=2, num_workers=8)

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_table=True, interval=10),
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
            project='chest X-rays', name='unet-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
