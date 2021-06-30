from argparse import ArgumentParser
import mmcv
from tools.api import inference, init_face_model
from datasets.pipelines import Compose
import cv2
import numpy as np
from face3d.mesh import mesh_core_cython
from face3d.morphable_model.load import load_BFM
import face3d

def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return uv_coords

def main():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out_img', default='result.jpg', help='Test image')
    args = parser.parse_args()

    if isinstance(args.config, str):
        config = mmcv.Config.fromfile(args.config)
    elif not isinstance(args.config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    triangles = np.loadtxt("../utils/uv_data/triangles.txt")

    image_mesh = np.zeros((256, 256, 3), dtype=np.float32)


    model = init_face_model(config, args.checkpoint, device=args.device)
    test_pipeline = Compose(config.data.test.pipeline)
    image = mmcv.imread(args.img_path)

    # cat_img = cv2.imread("../sample/dick_face/cat.jpg")
    result = inference(model, args.img_path, test_pipeline)
    valid_vertices = np.zeros_like(image).astype(np.float32).reshape(256*256, -1)
    face_idx = result["face_idx"]

    new_colors = cv2.imread("../sample/dick_face/result.jpg")
    new_colors = new_colors.reshape((256*256, -1))[face_idx]

    new_colors = new_colors.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    new_colors = new_colors.astype(np.float32).copy()




    vertices, valid_vertices = result["all_vertices"], result["valid_vertices"]
    depth_buffer = np.zeros([256, 256], dtype=np.float32, order='C') - 999999.
    mesh_core_cython.render_colors_core(image_mesh, valid_vertices, triangles, new_colors, depth_buffer,
                                        valid_vertices.shape[0], triangles.shape[0],
                                        256, 256, 3)

    face_mask = np.zeros((256, 256, 3), dtype=np.float32)
    vis_colors = np.ones((vertices.shape[0], 1))
    color_ones = np.ones_like(new_colors)
    depth_buffer = np.zeros([256, 256], dtype=np.float32, order='C') - 999999.
    mesh_core_cython.render_colors_core(face_mask, valid_vertices, triangles, color_ones, depth_buffer,
                                        valid_vertices.shape[0], triangles.shape[0],
                                        256, 256, 3)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_image = image * (1 - face_mask) + image_mesh * face_mask


    # valid_vertices[face_idx, :] = vertices.reshape(256*256, -1)[face_idx, :]
    # valid_vertices = valid_vertices.reshape(256, 256, 3)
    # image[0, 0, :] = 0
    # uv_result = cv2.remap(image_mesh, valid_vertices[..., :2], None, interpolation=cv2.INTER_NEAREST)


    cv2.imwrite("../sample/dick_face/result_cat.png", new_image)

    # cv2.imwrite(args.out_img, uv_result)




if __name__ == '__main__':
    main()
