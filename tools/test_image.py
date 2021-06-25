from argparse import ArgumentParser
import mmcv
from tools.api import inference, init_face_model
import os
from tools.vis_tool import VisualAPI
from datasets.pipelines import Compose
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out_dir', default='result.jpg', help='Test image')
    args = parser.parse_args()

    if isinstance(args.config, str):
        config = mmcv.Config.fromfile(args.config)
    elif not isinstance(args.config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    model = init_face_model(config, args.checkpoint, device=args.device)
    test_pipeline = Compose(config.data.test.pipeline)
    image_list = os.listdir(args.img_dir)
    for image_name in tqdm(image_list, "Inference...."):
        assert image_name[-3:] in ["jpg", "png", "bmp"], "Wrong img format"
        name = image_name[:-4]

        image_path = os.path.join(args.img_dir, image_name)
        image = mmcv.imread(image_path)

        result = inference(model, image_path, test_pipeline)

        image_kpt = VisualAPI.plot_kpt(image, result["keypoint"])
        imgae_vertices = VisualAPI.plot_vertices(image, result["valid_vertices"])

        mmcv.imwrite(image_kpt, os.path.join(args.out_dir, f"{name}_kpt.jpg"))
        mmcv.imwrite(imgae_vertices, os.path.join(args.out_dir, f"{name}_vertices.jpg"))
        # show the results



if __name__ == '__main__':
    main()
