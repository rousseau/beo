#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import nibabel

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl


# Customized data loader for only one pair of images
from torch.utils.data import Dataset
class TwoDataSet(Dataset):
    def __init__(self, target, source):
        self.target = target
        self.source = source 
        self.len = len(self.target)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.target, self.source    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', help='Target / Reference Image', type=str, required = True)
    parser.add_argument('-s', help='Source / Moving Image', type=str, required = True)

    args = parser.parse_args()


