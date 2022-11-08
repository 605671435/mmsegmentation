_base_ = [
    '../_base_/models/segformer_caformer_s18.py',
    '../_base_/datasets/hrf.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002 / 8, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

train_dataloader = dict(batch_size=1, num_workers=1)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
