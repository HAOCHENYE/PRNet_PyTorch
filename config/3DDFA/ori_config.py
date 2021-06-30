# -*- coding: utf-8 -*-

"""
    @date: 2019.11.13
    @function: hyparameter for training & inference.
"""
base_lr = 1e-4
model = dict(type="DDFAModel",
             backbone=dict(type="MobileNet", num_classes=62),
             loss=dict(type="VDCLoss",
                       used_item=dict(param_mean=None,
                                      param_std=None,
                                      u=None,
                                      w_shp=None,
                                      w_exp=None,
                                      keypoints=None,
                                      )
                       )
             )

train_pipeline = [
            dict(type='LoadImage', to_float32=True),
            dict(type='PhotoMetricDistortion', brightness_delta=32),
            # dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='DefaultFormatBundle', label_padding_dim=None),
            dict(type='Collect')
        ]

test_pipeline = [
            dict(type='LoadImage', to_float32=True),
            dict(type='Resize', keep_ratio=False, img_scale=(120, 120), multiscale_mode="value"),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect'),
        ]

data_root = "/usr/videodate/yehc/AFLW/"
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type='DDFADataset',
        root_dir=data_root + "train_aug_120x120",
        filelists=data_root + 'train_aug_120x120.list.train',
        param_fp=data_root + "param_all_norm.pkl",
        pipeline=train_pipeline),
    test=dict(
        pipeline=test_pipeline
    ))


total_epochs = 300
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(interval=1)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
# lr_config = dict(policy='Cyclic',
#                  cyclic_times=5,
#                  warmup="linear",
#                  warmup_iters=1000,
#                  warmup_ratio=0.1,)
log_level = 'INFO'
dist_params = dict(backend='nccl')
optimizer = dict(type='AdamW', lr=1e-4)
optimizer_config = dict(grad_clip=None)
gpu_ids = range(0, 2)
work_dir = 'work_dirs/3DDFA'
resume_from = None
load_from = None