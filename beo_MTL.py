#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import glob
import multiprocessing

from beo_torchio_datasets import get_dhcp



subjects = get_dhcp(max_subjects=100)

# DATA AUGMENTATION
normalization = tio.ZNormalization()
spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)
flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
transforms = [flip, spatial, normalization]
training_transform = tio.Compose(transforms)
validation_transform = tio.Compose([normalization])  

# SPLIT DATA
seed = 42  # for reproducibility
training_split_ratio = 0.9
num_subjects = len(subjects)
num_training_subjects = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects, generator=torch.Generator().manual_seed(seed))

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')