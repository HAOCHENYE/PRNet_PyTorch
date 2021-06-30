from mmcv.utils import Registry, build_from_cfg

FACELOSS=Registry("face_loss")

def build_face_loss(cfg):
    return build_from_cfg(cfg, FACELOSS)


