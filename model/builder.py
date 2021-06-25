from mmcv.utils import Registry, build_from_cfg

FACEMODEL=Registry("face_model")
BACKBONE=Registry("face_model")

def build_face_model(cfg):
    return build_from_cfg(cfg, FACEMODEL)

def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONE)

