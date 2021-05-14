#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:43:26 2021

@author: rousseau
"""
from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import glob
import multiprocessing

max_subjects = 100
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5
num_workers = 0#multiprocessing.cpu_count()

training_batch_size = 1
validation_batch_size = 1 

patch_size = 128
samples_per_volume = 32
max_queue_length = 256

prefix = 'gromov'
prefix += '_epochs_'+str(num_epochs)
prefix += '_subj_'+str(max_subjects)
prefix += '_patches_'+str(patch_size)
prefix += '_sampling_'+str(samples_per_volume)

data_path = home+'/Sync/Data/DHCP/'
output_path = home+'/Sync/Experiments/'

all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)
all_t1s = glob.glob(data_path+'**/*desc-restore_space-T2w_T1w.nii.gz', recursive=True)

all_t1s = all_t1s[:max_subjects]
subjects = []

for t1_file in all_t1s:
    id_subject = t1_file.split('/')[6].split('_')[0:2]
    id_subject = id_subject[0]+'_'+id_subject[1]

    t2_file = [s for s in all_t2s if id_subject in s][0]
    seg_file = [s for s in all_seg if id_subject in s][0]
    
    subject = tio.Subject(
        t1=tio.ScalarImage(t1_file),
        t2=tio.ScalarImage(t2_file),
        seg=tio.LabelMap(seg_file),
    )
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')

#%%
normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
onehot = tio.OneHot()

spatial = tio.OneOf({
    tio.RandomAffine(scales=0.1,degrees=30): 0.8,
    #tio.RandomElasticDeformation(): 0.2,
  },
  p=0.75,
)

bias = tio.RandomBiasField(p=0.3)
flip = tio.RandomFlip()
noise = tio.RandomNoise(p=0.5)

transforms = [bias, normalization, flip, spatial, noise, onehot]

training_transform = tio.Compose(transforms)
validation_transform = tio.Compose([normalization, onehot])

#%%
