from argparse import ArgumentParser
import mmcv
from tools.api import inference, init_face_model
from datasets.pipelines import Compose
import cv2
import numpy as np

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

    model = init_face_model(config, args.checkpoint, device=args.device)
    test_pipeline = Compose(config.data.test.pipeline)
    image = mmcv.imread(args.img_path)
    image = cv2.resize(image, (256, 256))
    # cat_img = cv2.imread("../sample/dick_face/cat.jpg")
    result = inference(model, args.img_path, test_pipeline)
    valid_vertices = np.zeros_like(image).astype(np.float32).reshape(256*256, -1)
    face_idx = result["face_idx"]

    uv_map = cv2.imread("../sample/dick_face/cat.jpg")


    vertices = result["all_vertices"]



    valid_vertices[face_idx, :] = vertices.reshape(256*256, -1)[face_idx, :]
    valid_vertices = valid_vertices.reshape(256, 256, 3)
    image[0, 0, :] = 0
    uv_result = cv2.remap(image, valid_vertices[..., :2], None, interpolation=cv2.INTER_NEAREST)

    # total_idx = np.arange(256*256)
    # no_face_idx = np.array(list(set(total_idx) - set(face_idx)))
    # mask = np.zeros((256*256))
    # mask[no_face_idx] = 255 / 65535
    # mask = mask.reshape((256, 256, 1))
    # uv_result = np.concatenate([uv_result, mask], axis=2)
    cv2.imwrite("../sample/dick_face/result_cat.png", uv_result)

    cv2.imwrite(args.out_img, uv_result)




if __name__ == '__main__':
    main()
