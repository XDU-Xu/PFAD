import json
import glob
import torch
import torch.nn
import numpy as np
from skimage.transform import resize
import os
import os.path
import SimpleITK as sitk
import nibabel
import random
import torchvision.utils as vutils


class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, npy_dir, transform):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.transform = transform
        self.image_size = image_size
        self.npy_path = sorted(glob.glob(f'{npy_dir}/*.npy'))

    def __getitem__(self, item):
        name = self.npy_path[item]
        data = np.load(name).astype(np.float32)
        if self.transform:
            norm = (data - data.min()) / (data.max() - data.min())
            norm = norm * 2 - 1
            image = norm
        else:
            image = data
        # image = resize(image, (256, 256), order=0)
        assert image.shape == (self.image_size, self.image_size)

        return np.expand_dims(image, axis=0), {}

    def __len__(self):
        if self.npy_path is None:
            return len(self.data_path)

        return len(self.npy_path)
