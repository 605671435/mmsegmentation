_base_ = ['./fcn_mobilenev2-d8_1xb8-40k_synapse-512x512.py']

# model settings
model = dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type='EX_Module', attn_types=('sp', 'ch'), fusion_types=('pr', 'sq')),
                      stages=(True, True, True, True, True, True, True),
                      position=1),
                 ]
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-mobilenetv2-ex-40k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
