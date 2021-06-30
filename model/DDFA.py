import torch.nn as nn
from .builder import FACEMODEL
from .builder import build_backbone
from .loss.builder import build_face_loss

@FACEMODEL.register_module()
class DDFAModel(nn.Module):
    def __init__(self,
                 backbone,
                 loss,
                 **kwargs):
        super(DDFAModel, self).__init__()

        self.model = build_backbone(backbone)

        self.loss_type = loss["type"]
        self.loss = build_face_loss(loss)

    def train_step(self, data, optimizer):
        input, target = data["label_meta"]['label'], data["img_meta"]['img']
        out = self.model(input)
        loss = self.loss(out, target)

        return {"loss": loss, "log_vars": {self.loss_type: loss.item()}, "num_samples": len(input)}

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


