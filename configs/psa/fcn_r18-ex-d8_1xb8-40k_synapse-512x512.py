_base_ = ['./fcn_r18-d8_1xb8-40k_synapse-512x512.py']

model = dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type='EX_Module', attn_types=('sp', 'ch'), fusion_type='dsa'),
                stages=(True, True, True, True),
                position='after_conv1'),
           ]
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='fcn-r18-ex-40k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# custom_hooks = [
#     dict(type='TrainingScheduleHook', use_fcn=False, interval=2000)
# ]
