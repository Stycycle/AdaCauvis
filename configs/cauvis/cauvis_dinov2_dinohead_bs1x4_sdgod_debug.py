_base_ = './cauvis_dinov2_dinohead_bs1x4_sdgod.py'

dataset_type = 'SdgodDataset'
data_root = '/home/stycycle/workspace/Cauvis/wufan___S-DGOD/'
backend_args = None
img_scales = (640, 640)
debug_samples_per_domain = 100

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scales, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['dataset_type'],
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='daytime_clear/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='daytime_clear/VOC2007/'),
                filter_cfg=dict(
                    filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=test_pipeline,
                indices=debug_samples_per_domain),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='daytime_foggy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='daytime_foggy/VOC2007/'),
                filter_cfg=dict(
                    filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=test_pipeline,
                indices=debug_samples_per_domain),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='dusk_rainy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='dusk_rainy/VOC2007/'),
                filter_cfg=dict(
                    filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=test_pipeline,
                indices=debug_samples_per_domain),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='night_rainy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='night_rainy/VOC2007/'),
                filter_cfg=dict(
                    filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=test_pipeline,
                indices=debug_samples_per_domain),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='Night-Sunny/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='Night-Sunny/VOC2007/'),
                filter_cfg=dict(
                    filter_empty_gt=True, min_size=32, bbox_min_size=32),
                pipeline=test_pipeline,
                indices=debug_samples_per_domain),
        ]))

test_dataloader = val_dataloader
