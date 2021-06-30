import torch.nn as nn
from .loss.prnet_loss import WeightMaskLoss, SSIM
from .builder import FACEMODEL
from .builder import build_backbone
import numpy as np
import torch

@FACEMODEL.register_module()
class PRNetModel(nn.Module):
    def __init__(self,
                 mask_path="../utils/uv_data/uv_weight_mask_gdh.png",
                 backbone=dict(type="ResFCN256"),
                 face_index_path="../utils/uv_data/face_ind.txt",
                 uv_kpt_path="../utils/uv_data/uv_kpt_ind.txt",
                 **kwargs):
        super(PRNetModel, self).__init__()

        self.model = build_backbone(backbone)
        self.loss = WeightMaskLoss(mask_path=mask_path)
        self.stat_loss = SSIM(mask_path=mask_path, gauss="original")

        self.face_index_path = face_index_path
        self.uv_kpt_path = uv_kpt_path
        self.uv_kpt_ind = None
        self.face_ind = None


        self.resolution_op = 256

    def load_ind(self):
        self.uv_kpt_ind = np.loadtxt(self.uv_kpt_path).astype(np.int32)  # 2 x 68 get kpt
        self.face_ind = np.loadtxt(self.face_index_path).astype(np.int32)  # get valid vertices in the pos map

    def train_step(self, data, optimizer):
        uv_map, origin = data["label_meta"]['label'], data["img_meta"]['img']
        uv_map_result = self.model(origin)
        logit = self.loss(uv_map_result, uv_map)
        stat_logit = self.stat_loss(uv_map_result, uv_map)

        return {"loss": logit, "log_vars": {"logit":logit.item(), "stat_logit":stat_logit.item()}, "num_samples": len(uv_map)}

    def inference(self, data):
        if not isinstance(self.uv_kpt_ind, np.ndarray):
            self.load_ind()
        scale_factor = torch.Tensor(data["img_meta"]["scale_factor"])[0]
        origin = data["img_meta"]['img'][0]
        vertices = self.model(origin)
        vertices[0, 0, ...] = vertices[0, 0, ...] / scale_factor[0]
        vertices[0, 1, ...] = vertices[0, 1, ...] / scale_factor[1]
        result = self.get_result(vertices)
        return result

    def get_result(self, vertices):
        vertices = vertices * 255
        vertices = (vertices.squeeze()).permute(1, 2, 0)
        result = torch.reshape(vertices, [self.resolution_op ** 2, -1])
        valid_vertices = result[self.face_ind, :]
        kpt = vertices[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
        return dict(all_vertices=vertices.detach().cpu().numpy(),
                    dense_point=valid_vertices.detach().cpu().numpy(),
                    sparse_point=kpt.detach().cpu().numpy(),
                    face_idx=self.face_ind)

    def export_onnx(self, dummy_input):
        return self.model(dummy_input)

    def forward(self, data):
        #TODO train test and export onnx
        return self.export_onnx(data)


