import os
import cv2
import csv
import glob
import torch
import kornia
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from osgeo import gdal
from PIL import Image
from torch.utils import data as data
from torchvision.transforms import Normalize, Compose, Lambda, Resize, InterpolationMode, ToTensor
from torchvision.transforms import functional as trans_fn
from basicsr.utils.registry import DATASET_REGISTRY

totensor = torchvision.transforms.ToTensor()


@DATASET_REGISTRY.register()
class PROBAVDataset(data.Dataset):
    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
    """

    def __init__(self, opt):
        super(PROBAVDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.data_root = opt['data_root']

        self.n_lr_images = opt['n_lr_images'] 
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        hr_fps = glob.glob(self.data_root + 'train/NIR/*/HR.png')
        
        # Filter filepaths based on if the split is train or validation.
        if self.split == 'train':
            hr_fps = glob.glob(self.data_root + 'train/NIR/*/HR.png')
        else:
            hr_fps = glob.glob(self.data_root + 'train/NIR/val/*/HR.png')

        self.datapoints = []
        lr_fps = []
        for hr_fp in hr_fps:
            lrs = []
            for i in range(self.n_lr_images):
                if i < 10:
                    lr = hr_fp.replace('HR', 'LR00' + str(i))
                else:
                    lr = hr_fp.replace('HR', 'LR0' + str(i))
                lrs.append(lr)
            self.datapoints.append([hr_fp, lrs])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs for split ", self.split)

    def __getitem__(self, index):
        hr_path, lr_paths  = self.datapoints[index]

        hr_im = cv2.imread(hr_path)
        hr_im = cv2.resize(hr_im, (120,120))  # resize from (384,384) to (128,128) to speed up training
        hr_tensor = totensor(hr_im)

        lr_ims = []
        for lr_path in lr_paths:
            lr_im = cv2.imread(lr_path)
            lr_im = cv2.resize(lr_im, (40, 40))  # resize from (128, 128) to (42, 42), which isn't perfect /3, to speed up training
            lr_tensor = totensor(lr_im)
            lr_ims.append(lr_tensor)

        if self.use_3d:
            img_LR = torch.stack(lr_ims)
        else:
            img_LR = torch.cat(lr_ims)

        img_HR = hr_tensor

        return {'gt': img_HR, 'lq': img_LR, 'Index': index}

    def __len__(self):
        return self.data_len
