# -*- coding: utf-8 -*-

"""
    @date: 2019.07.19
    @author: samuel ko
    @func: same function as api.py in original PRNet Repo.
"""
from model.builder import build_face_model
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from datasets.pipelines import Compose

def init_face_model(config, checkpoint=None, device="cuda:0"):
    config.model.pretrained = None
    model = build_face_model(config.model)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        load_checkpoint(model, checkpoint, map_location=map_loc)

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def inference(model, img, test_pipeline):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # add information into dict
    data = dict(img_meta=dict(img_path=img), label_meta=dict(label_path=""))

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    result = model.inference(data)
    return result




