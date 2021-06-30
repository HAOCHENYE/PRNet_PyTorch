# -*- coding: utf-8 -*-
"""
    @author: samuel ko
    @date: 2019.07.18
    @readme: The implementation of PRNet Network DataLoader.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from .pipelines.compose import Compose
from .builder import DATASETS


@DATASETS.register_module()
class WLP300Dataset(object):
    def __init__(self,
                 root_dir,
                 label_file,
                 pipeline=None,
                 cache=False,
                 read_thread=8
                 ):
        """
        :param root_dir:
        :param label_file:
        :param pipelines:
        :param cache:
        :param read_thread:
        """
        self.img_meta = dict(img_path=[], img=[])
        self.label = dict(label_path=[], npy=[])

        self.cache = cache
        self.num_thread = read_thread

        # torch.distributed.barrier()
        self.get_image_npy(root_dir, label_file)
        # torch.distributed.barrier()
        self.root_dir = root_dir
        self._set_group_flag()

        assert isinstance(pipeline, list), "piplines must be list!"
        pipline_cache = []
        for transform in pipeline:
            if transform["type"] == "LoadImageAnno" and self.cache:
                continue
            pipline_cache.append(pipeline)
        self.pipeline = Compose(pipeline)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            # img_info = self.images[i].shape
            # if img_info['width'] / img_info['height'] > 1:
            self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def get_image_npy(self, root_dir, label_file):
        file_path = json.load(open(os.path.join(root_dir,label_file), 'r'))
        if not self.cache:
            self.img_meta["img_path"] = [os.path.join(root_dir, image)
                                         for image in tqdm(file_path["image_path"], "Loading image path")]
            self.label["label_path"] = [os.path.join(root_dir, npy)
                                        for npy in tqdm(file_path["npy_path"], "Loading npy path")]
        else:
            self.img_meta["img_path"] = [os.path.join(root_dir, image)
                                         for image in tqdm(file_path["image_path"], "Loading image path")]
            self.label["label_path"] = [os.path.join(root_dir, npy)
                                       for npy in tqdm(file_path["npy_path"], "Loading npy path")]
            self.img_meta["img"] = [cv2.imread(os.path.join(root_dir, image))
                                    for image in tqdm(file_path["image_path"], "reading image")]
            self.label["label"] = [np.load(os.path.join(root_dir, npy))
                                   for npy in tqdm(file_path["npy_path"], "Loading npy")]
        assert len(self.label) == len(self.img_meta), 'number of npy should be equal with number of imgae'

    def __len__(self):
        return len(self.label["label_path"])

    def __getitem__(self, idx):
        if self.cache:
            img = self.img_meta["img"][idx]
            label = self.label["label"][idx]
            label_path = self.img_meta["img_path"][idx]
            img_path = self.label["label_path"][idx]
        else:
            img_path = self.img_meta["img_path"][idx]
            label_path = self.label["label_path"][idx]
            img = cv2.imread(img_path)
            label = np.load(label_path)

        sample = {'img_meta': {"img": img, "img_path": img_path, "ori_shape": img.shape, "img_shape": img.shape},
                  'label_meta': {"label": label, "label_path": label_path}}

        sample = self.pipeline(sample)
        return sample

