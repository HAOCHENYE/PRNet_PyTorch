import json
import os
from tqdm import tqdm

import argparse

def generate_json(root_dir):
    json_context = dict(image_path=[], npy_path=[])
    for data_dir in os.listdir(root_dir):
        data_name = data_dir
        data_dir = os.path.join(root_dir, data_dir)  # IBUG HELEN....
        for file_dir in tqdm(os.listdir(data_dir), "Dataset: {}".format(data_name)):
            file_name = file_dir
            file_dir = os.path.join(data_dir, file_dir)
            for data in os.listdir(file_dir):
                if data == "original.jpg":
                    json_context["image_path"].append(os.path.join(data_name, file_name, data))
                elif data.endswith("npy"):
                    json_context["npy_path"].append(os.path.join(data_name, file_name, data))

    with open(os.path.join(root_dir, "label.json"), 'w', encoding='utf8') as outfile:
        json.dump(json_context, outfile, indent=4, sort_keys=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="specify output uv_map directory.")
    args = parser.parse_args()
    generate_json(args.root_dir)