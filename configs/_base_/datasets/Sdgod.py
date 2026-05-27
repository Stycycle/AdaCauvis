# dataset settings
dataset_type = 'SdgodDataset'
data_root = '/home/stycycle/workspace/Cauvis/wufan___S-DGOD/'

backend_args = None

img_scales = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scales, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scales, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='daytime_clear/VOC2007/ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root='daytime_clear/VOC2007/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        ignore_keys=['dataset_type'],
        datasets=[
            dict(type=dataset_type, data_root=data_root,  # Daytime_Sunny
                 ann_file='daytime_clear/VOC2007/ImageSets/Main/test.txt',
                 data_prefix=dict(sub_data_root='daytime_clear/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Daytime-Foggy
                 ann_file='daytime_foggy/VOC2007/ImageSets/Main/test.txt',
                 data_prefix=dict(sub_data_root='daytime_foggy/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Dusk-rainy
                 ann_file='dusk_rainy/VOC2007/ImageSets/Main/test.txt',
                 data_prefix=dict(sub_data_root='dusk_rainy/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Night_rainy
                 ann_file='night_rainy/VOC2007/ImageSets/Main/test.txt',
                 data_prefix=dict(sub_data_root='night_rainy/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
            dict(type=dataset_type, data_root=data_root,  # Night-Sunny
                 ann_file='Night-Sunny/VOC2007/ImageSets/Main/test.txt',
                 data_prefix=dict(sub_data_root='Night-Sunny/VOC2007/'),
                 filter_cfg=dict(
                     filter_empty_gt=True, min_size=32, bbox_min_size=32),
                 pipeline=test_pipeline),
        ])
)
test_dataloader = val_dataloader

val_evaluator = dict(type='SDGODMetric',
                     dataset_keys=['daytime_clear', 'daytime_foggy', 'dusk_rainy', 'night_rainy', 'Night-Sunny'],
                     eval_mode='11points', metric=['mAP'], )
test_evaluator = val_evaluator
