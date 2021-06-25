import argparse
import os.path as osp
from functools import partial
import tensorwatch as tw
import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv.runner import load_checkpoint
from mmcv.cnn import fuse_conv_bn
from model.builder import build_face_model

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=v1.0.4')

def load(module, prefix=''):
    for name, child in module._modules.items():
        if not hasattr(child, 'fuse_conv'):
            load(child, prefix + name + '.')
        else:
            child.fuse_conv()

def pytorch2onnx(model,
                 input_shape,
                 opset_version=9,
                 output_file='tmp.onnx'):
    # model=fuse_conv_bn(model)
    model.cpu().eval()
    # read image
    dummpy_input = torch.zeros(input_shape)
    load(model)
    print(tw.model_stats(model, input_shape))
    # output_names = ["P3_logits", "P4_logits", "P5_logits", "P6_logits","P7_logits",
    #                 "P3_bbox_reg", "P4_bbox_reg", "P5_bbox_reg", "P6_bbox_reg","P7_bbox_reg"]
    # output_names = ["hm", "wh"]
    output_names = ["vertices"]
    torch.onnx.export(
        model, dummpy_input,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=False,
        opset_version=11,
        input_names=['data'],
        output_names=output_names,
        do_constant_folding=True,
    )
    #
    # torch.onnx.export(
    #     model, [(one_img)],
    #     output_file,
    #     export_params=True,
    #     keep_initializers_as_inputs=True,
    #     verbose=show,
    #     opset_version=opset_version)
    #




def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    input_shape = [1, 3, 256, 256]
    # build the model
    model = build_face_model(cfg.model)
    try:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    except:
        Warning("Path of checkpoint is not correct")
    # conver model to onnx file
    pytorch2onnx(
        model,
        input_shape,
        opset_version=args.opset_version,
        output_file=args.output_file,)
