
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import random

from beo_svf import SpatialTransformer, VecInt
from beo_nets import Unet
from beo_metrics import NCC


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        # Pick two different random indices
        idx1, idx2 = random.sample(range(len(self.dataset)), 2)

        return self.dataset[idx1],self.dataset[idx2]

    def __len__(self):
        return len(self.dataset)