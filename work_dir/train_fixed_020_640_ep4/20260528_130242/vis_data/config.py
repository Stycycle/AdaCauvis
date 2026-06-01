auto_scale_lr = dict(base_batch_size=16)
backend_args = None
classes = 7
data_root = '/home/autodl-tmp/Cauvis/wufan___S-DGOD/'
dataset_type = 'SdgodDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = False
img_scales = (
    640,
    640,
)
img_size = 640
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 4
model = dict(
    as_two_stage=True,
    backbone=dict(
        block_chunks=0,
        cauvis_config=dict(
            embed_dims=1024,
            img_size=640,
            link_token_to_query=False,
            max_low_freq_ratio=0.2,
            min_low_freq_ratio=0.2,
            num_layers=24,
            patch_size=16,
            token_length=1600,
            type='Cauvis'),
        depth=24,
        embed_dim=1024,
        ffn_bias=True,
        ffn_layer='mlp',
        img_size=640,
        init_cfg=dict(
            checkpoint='checkpoints/dinov2_converted_640.pth',
            type='Pretrained'),
        init_values=1e-05,
        mlp_ratio=4,
        num_heads=16,
        patch_size=16,
        proj_bias=True,
        qkv_bias=True,
        type='CauvisDINOv2'),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=7,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=4,
        gamma=0.1,
        milestones=[
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='daytime_clear/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='daytime_clear/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='daytime_foggy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='daytime_foggy/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='dusk_rainy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='dusk_rainy/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='night_rainy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='night_rainy/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='Night-Sunny/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='Night-Sunny/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
        ],
        ignore_keys=[
            'dataset_type',
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    dataset_keys=[
        'daytime_clear',
        'daytime_foggy',
        'dusk_rainy',
        'night_rainy',
        'Night-Sunny',
    ],
    eval_mode='11points',
    metric=[
        'mAP',
    ],
    type='SDGODMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=4, type='EpochBasedTrainLoop', val_interval=4)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='daytime_clear/VOC2007/ImageSets/Main/train.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='daytime_clear/VOC2007/'),
        data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                640,
                640,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='SdgodDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_per_gpu_batch_size = 2
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        640,
        640,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='daytime_clear/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='daytime_clear/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='daytime_foggy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='daytime_foggy/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='dusk_rainy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='dusk_rainy/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='night_rainy/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='night_rainy/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
            dict(
                ann_file='Night-Sunny/VOC2007/ImageSets/Main/test.txt',
                data_prefix=dict(sub_data_root='Night-Sunny/VOC2007/'),
                data_root='/home/autodl-tmp/Cauvis/wufan___S-DGOD/',
                filter_cfg=dict(
                    bbox_min_size=32, filter_empty_gt=True, min_size=32),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        640,
                        640,
                    ), type='Resize'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                        ),
                        type='PackDetInputs'),
                ],
                type='SdgodDataset'),
        ],
        ignore_keys=[
            'dataset_type',
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_keys=[
        'daytime_clear',
        'daytime_foggy',
        'dusk_rainy',
        'night_rainy',
        'Night-Sunny',
    ],
    eval_mode='11points',
    metric=[
        'mAP',
    ],
    type='SDGODMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dir/train_fixed_020_640_ep4'
