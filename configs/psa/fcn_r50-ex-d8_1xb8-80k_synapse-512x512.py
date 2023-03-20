_base_ = ['./fcn_r50-ex-d8_1xb8-40k_synapse-512x512.py']

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-r50fftt-ex-80k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                      by_epoch=False,
                      interval=8000,
                      max_keep_ckpts=3,
                      save_best=['mIoU'], rule='greater'))