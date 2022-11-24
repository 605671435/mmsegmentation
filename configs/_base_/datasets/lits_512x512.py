# dataset settings
dataset_type = 'LITSDataset'
data_root = 'data/LITS'
img_scale = (75, 512, 512)
crop_size = (3, 256, 256)
train_pipeline = [
    dict(type='LoadBiomedicalImageFromFile'),
    dict(type='LoadBiomedicalAnnotation'),
    dict(
        type='BioPatchCrop',
        crop_size=crop_size,
        crop_mode='nnUNet'),
    dict(type='PackMedicalInputs')
]
test_pipeline = [
    dict(type='LoadBiomedicalImageFromFile'),
    dict(type='LoadBiomedicalAnnotation'),
    dict(
        type='BioPatchCrop',
        crop_size=crop_size,
        crop_mode='nnUNet'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator
