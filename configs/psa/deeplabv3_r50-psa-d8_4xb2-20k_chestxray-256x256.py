_base_ = ['./deeplabv3_r50-d8_4xb2-20k_chestxray-256x256.py']

model = dict(
    backbone=dict(with_attn='psa', with_attn_stage=[2, 2, 2, 2]),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='chest X-rays', name='resnet-psa-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')