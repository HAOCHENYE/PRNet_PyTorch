from pathlib import Path
import pickle
import numpy as np
import torch
import os.path as osp
import cv2
from .pipelines.compose import Compose
from .builder import DATASETS

@DATASETS.register_module()
class DDFADataset(object):
    #Current dataset only support pipline without LoadLabel
    def __init__(self, root_dir, filelists, param_fp, pipeline, **kargs):
        self.root_dir = root_dir
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.param_fp = param_fp
        self.params = torch.from_numpy(self._load(param_fp))
        self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def _target_loader(self, index):
        target = self.params[index]

        return target

    def __getitem__(self, index):
        path = osp.join(self.root_dir, self.lines[index])
        img = cv2.imread(path)

        label = self._target_loader(index)

        sample = {'img_meta': {"img": img, "img_path": path, "ori_shape": img.shape, "img_shape": img.shape},
                  'label_meta': {"label": label, "label_path": f"labels are all saved in {self.param_fp}"}}

        sample = self.pipeline(sample)
        return sample

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

    def __len__(self):
        return len(self.lines)

    def _get_suffix(self, filename):
        """a.jpg -> jpg"""
        pos = filename.rfind('.')
        if pos == -1:
            return ''
        return filename[pos + 1:]

    def _load(self, fp):
        suffix = self._get_suffix(fp)
        if suffix == 'npy':
            return np.load(fp)
        elif suffix == 'pkl':
            return pickle.load(open(fp, 'rb'))

