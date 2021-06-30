import torch.nn as nn
from .builder import FACEMODEL
from .builder import build_backbone
from .loss.builder import build_face_loss
import numpy as np
from .temp_func import _load
import torch

@FACEMODEL.register_module()
class DDFAModel(nn.Module):
    def __init__(self,
                 backbone,
                 loss,
                 meta_path=dict(keypoints="../config/3DDFA/keypoints_sim.npy",
                                w_shp="../config/3DDFA/w_shp_sim.npy",
                                w_exp="../config/3DDFA/w_exp_sim.npy",
                                u_shp="../config/3DDFA/u_shp.npy",
                                u_exp="../config/3DDFA/u_exp.npy",
                                meta='../config/3DDFA/param_whitening.pkl'
                                ),
                 std_size=120,

                 **kwargs):

        super(DDFAModel, self).__init__()

        self.model = build_backbone(backbone)
        self.loss_type = loss["type"] #for print loss

        #TODO ugly code
        meta = _load(meta_path["meta"])
        #loss param
        self.keypoints = _load(meta_path["keypoints"])
        self.w_shp = _load(meta_path["w_shp"])
        self.w_exp = _load(meta_path["w_exp"])
        self.u_shp = _load(meta_path["u_shp"])
        self.u_exp = _load(meta_path["u_exp"])
        self.param_mean = meta.get('param_mean')
        self.param_std = meta.get('param_std')
        self.u = self.u_shp + self.u_exp

        #inference param
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.std_size = std_size

        used_item = loss["used_item"]
        for key in used_item.keys():
            used_item[key] = torch.from_numpy(getattr(self, key))

        self.loss = build_face_loss(loss)

    def train_step(self, data, optimizer):
        label, img = data["label_meta"]['label'], data["img_meta"]['img']
        out = self.model(img)
        loss = self.loss(out, label)

        return {"loss": loss, "log_vars": {self.loss_type: loss.item()}, "num_samples": len(img)}

    def inference(self, data):
        image = data["img_meta"]['img'][0]
        param = self.model(image)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        scale_factor = torch.Tensor(data["img_meta"]["scale_factor"])[0]
        sparse = self.reconstruct_vertex(param, dense=False).transpose((1, 0))
        sparse[:, 0] = sparse[:, 0] / scale_factor[0]
        sparse[:, 1] = sparse[:, 1] / scale_factor[1]
        sparse[:, 2] = sparse[:, 2] / (scale_factor[0] + scale_factor[1]) * 2

        dense = self.reconstruct_vertex(param, dense=True).transpose((1, 0))
        dense[:, 0] = dense[:, 0] / scale_factor[0]
        dense[:, 1] = dense[:, 1] / scale_factor[1]
        dense[:, 2] = dense[:, 2] / (scale_factor[0] + scale_factor[1]) * 2

        return dict(sparse_point=sparse, dense_point=dense)



    def reconstruct_vertex(self, param, whitening=True, dense=False, transform=True):
        """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
        dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
        image coordinate space, but without alignment caused by face cropping.
        transform: whether transform to image space
        """
        if len(param) == 12:
            param = np.concatenate((param, [0] * 50))
        if whitening:
            if len(param) == 62:
                param = param * self.param_std + self.param_mean
            else:
                param = np.concatenate((param[:11], [0], param[11:]))
                param = param * self.param_std + self.param_mean

        p, offset, alpha_shp, alpha_exp = self._parse_param(param)

        if dense:
            vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).reshape(3, -1, order='F') + offset

            if transform:
                # transform to image coordinate space
                vertex[1, :] = self.std_size + 1 - vertex[1, :]
        else:
            """For 68 pts"""
            vertex = p @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset

            if transform:
                # transform to image coordinate space
                vertex[1, :] = self.std_size + 1 - vertex[1, :]

        return vertex

    def _parse_param(self, param):
        """Work for both numpy and tensor"""
        p_ = param[:12].reshape(3, -1)
        p = p_[:, :3]
        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = param[12:52].reshape(-1, 1)
        alpha_exp = param[52:].reshape(-1, 1)
        return p, offset, alpha_shp, alpha_exp
    # def inference(self, data):
    #     if not isinstance(self.uv_kpt_ind, np.ndarray):
    #         self.load_ind()
    #     scale_factor = torch.Tensor(data["img_meta"]["scale_factor"])[0]
    #     origin = data["img_meta"]['img'][0]
    #     vertices = self.model(origin)
    #     vertices[0, 0, ...] = vertices[0, 0, ...] / scale_factor[0]
    #     vertices[0, 1, ...] = vertices[0, 1, ...] / scale_factor[1]
    #     result = self.get_result(vertices)
    #     return result
    #
    # def get_result(self, vertices):
    #     vertices = vertices * 255
    #     vertices = (vertices.squeeze()).permute(1, 2, 0)
    #     result = torch.reshape(vertices, [self.resolution_op ** 2, -1])
    #     valid_vertices = result[self.face_ind, :]
    #     kpt = vertices[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
    #     return dict(all_vertices=vertices.detach().cpu().numpy(),
    #                 valid_vertices=valid_vertices.detach().cpu().numpy(),
    #                 keypoint=kpt.detach().cpu().numpy(),
    #                 face_idx=self.face_ind)
    #
    # def export_onnx(self, dummy_input):
    #     return self.model(dummy_input)
    #
    # def forward(self, data):
    #     #TODO train test and export onnx
    #     return self.export_onnx(data)


