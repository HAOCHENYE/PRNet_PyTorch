# -*- coding: utf-8 -*-

"""
    @date: 2019.11.13
    @function: hyparameter for training & inference.
"""
base_lr = 1e-4
model = dict(type="PRNetModel",
             backbone=dict(type="ResFCN256"))

train_pipeline = [
            dict(type='LoadImage', to_float32=True),
            dict(type='LoadLabel'),
            dict(type="CoordNormalize", mean=0, std=256),
            dict(type='PhotoMetricDistortion', brightness_delta=32),
            dict(type="CutOut", n_holes=(1, 3), prob=0.3, cutout_ratio=(0.5, 0.5)),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            # dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect')
        ]

test_pipeline = [
            dict(type='LoadImage', to_float32=True),
            dict(type='Resize', keep_ratio=False, img_scale=(256, 256), multiscale_mode="value"),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect'),
        ]

data_root = "/usr/videodate/yehc/300LP/"
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='WLP300Dataset',
        root_dir=data_root,
        label_file=data_root + 'label.json',
        pipeline=train_pipeline,
        cache=False),
    test=dict(
        pipeline=test_pipeline
    ))


gauss_kernel = "original"
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
work_dir = 'work_dirs/PRNetCutOut'
resume_from = None
load_from = "work_dirs/PRNetOri/epoch_300.pth"